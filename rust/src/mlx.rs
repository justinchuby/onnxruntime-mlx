//! Safe RAII wrappers over the raw `sys::mlx` bindgen bindings.
//!
//! This is where the memory-safety win of the Rust rewrite lives: every MLX handle is owned by a
//! wrapper whose `Drop` calls the matching `mlx_*_free`, so op handlers never free by hand and a
//! leaked / double-freed `mlx_array` (a class of bug the C++ EP hit repeatedly) is impossible by
//! construction. Raw `unsafe`/FFI stays confined to `sys::mlx`; the engine and ops use these types.

use crate::sys::mlx;

/// Owning wrapper over an `mlx_stream` (freed once on drop).
pub struct Stream {
    raw: mlx::mlx_stream,
}

impl Stream {
    /// The default GPU stream (what every op in a plan runs on).
    pub fn new_default_gpu() -> Self {
        Stream {
            raw: unsafe { mlx::mlx_default_gpu_stream_new() },
        }
    }

    #[inline]
    pub fn as_raw(&self) -> mlx::mlx_stream {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { mlx::mlx_stream_free(self.raw) };
    }
}

/// Owning wrapper over an `mlx_array`. Holds exactly one reference; `Drop` releases it.
///
/// MLX ops do NOT consume their operands — they take their own internal references — so a handler
/// resolves an input to a borrowed raw handle (`as_raw`) and only the wrapper owns the reference.
/// Freshly produced arrays are wrapped with `from_raw` and kept alive (in the run arena or the plan
/// cache) until they are no longer needed.
pub struct Array {
    raw: mlx::mlx_array,
}

impl Array {
    /// Take ownership of a raw handle returned by an `mlx_*` call (e.g. the `res` out-param).
    #[inline]
    pub fn from_raw(raw: mlx::mlx_array) -> Self {
        Array { raw }
    }

    /// A fresh, empty array handle (the `mlx_array_new()` out-param sink for op results).
    #[inline]
    pub fn new() -> Self {
        Array {
            raw: unsafe { mlx::mlx_array_new() },
        }
    }

    /// Wrap host bytes into a new MLX array of `dtype` and the given shape (row-major). MLX copies
    /// the data (managed lifetime), so the source buffer need not outlive the array.
    pub fn from_data(data: *const std::os::raw::c_void, shape: &[i32], dtype: mlx::mlx_dtype) -> Self {
        let arr = Array {
            raw: unsafe {
                mlx::mlx_array_new_data(data, shape.as_ptr(), shape.len() as i32, dtype)
            },
        };
        // Memory view: a COPY-wrap (MLX copies the bytes into managed memory). Gated so a
        // traced-off run pays a single atomic load.
        let tr = crate::trace::tracer();
        if tr.active() {
            tr.record_copy_wrap((arr.size() * arr.itemsize()) as u64);
        }
        arr
    }

    /// Wrap an externally-owned buffer WITHOUT copying (zero-copy). MLX takes the raw pointer and,
    /// on Apple unified memory, hands it straight to Metal via `newBufferWithBytesNoCopy` — no host
    /// memcpy. If the pointer is not page-aligned (so Metal refuses the no-copy buffer) MLX silently
    /// falls back to allocate+copy, so correctness is preserved unconditionally; only the perf win is
    /// conditional on alignment.
    ///
    /// SAFETY / LIFETIME: `data` is owned by the caller (here: ORT owns the Compute input tensors for
    /// the whole `Compute` call). The registered deallocator is a NO-OP, so MLX never frees `data`.
    /// The buffer MUST stay valid and unmodified until every `mlx_eval` that reads this array has
    /// completed — guaranteed because the EP evaluates the whole boundary graph synchronously and
    /// drops the wrapping `Array` (and thus any MLX reference to `data`) before `Compute` returns.
    pub fn from_data_managed(
        data: *const std::os::raw::c_void,
        shape: &[i32],
        dtype: mlx::mlx_dtype,
    ) -> Self {
        // ORT owns the buffer; MLX must never free it. A no-op dtor makes the wrap purely borrowing.
        unsafe extern "C" fn noop_dtor(_: *mut std::os::raw::c_void) {}
        let arr = Array {
            raw: unsafe {
                mlx::mlx_array_new_data_managed(
                    data as *mut std::os::raw::c_void,
                    shape.as_ptr(),
                    shape.len() as i32,
                    dtype,
                    Some(noop_dtor),
                )
            },
        };
        // Memory view: the boundary zero-copy managed-wrap. A 16 KB page-aligned buffer takes MLX's
        // true `newBufferWithBytesNoCopy` no-copy path; an unaligned one silently falls back to an
        // internal allocate+copy — record which, plus the bytes borrowed. Gated (one atomic load off).
        let tr = crate::trace::tracer();
        if tr.active() {
            let aligned = (data as usize).is_multiple_of(16384);
            tr.record_managed_wrap((arr.size() * arr.itemsize()) as u64, aligned);
        }
        arr
    }

    /// The raw handle, for passing to `mlx_*` calls. Ownership is NOT transferred.
    #[inline]
    pub fn as_raw(&self) -> mlx::mlx_array {
        self.raw
    }

    pub fn ndim(&self) -> usize {
        unsafe { mlx::mlx_array_ndim(self.raw) }
    }

    pub fn shape(&self) -> Vec<i64> {
        let nd = self.ndim();
        let sh = unsafe { mlx::mlx_array_shape(self.raw) };
        (0..nd).map(|i| unsafe { *sh.add(i) } as i64).collect()
    }

    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        unsafe { mlx::mlx_array_size(self.raw) }
    }

    pub fn itemsize(&self) -> usize {
        unsafe { mlx::mlx_array_itemsize(self.raw) }
    }

    #[allow(dead_code)]
    pub fn dtype(&self) -> mlx::mlx_dtype {
        unsafe { mlx::mlx_array_dtype(self.raw) }
    }

    /// Force evaluation of this (single) array.
    #[allow(dead_code)]
    pub fn eval(&self) {
        unsafe { mlx::mlx_array_eval(self.raw) };
    }

    /// Raw byte pointer to the (evaluated) contiguous buffer, for the unified-memory copy-out.
    pub fn data_bytes(&self) -> *const u8 {
        unsafe { mlx::mlx_array_data_uint8(self.raw) }
    }
}

impl Default for Array {
    fn default() -> Self {
        Array::new()
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { mlx::mlx_array_free(self.raw) };
    }
}

/// Owning wrapper over an `mlx_vector_array` (the input list passed to a single `mlx_eval`).
pub struct VectorArray {
    raw: mlx::mlx_vector_array,
}

impl VectorArray {
    pub fn new() -> Self {
        VectorArray {
            raw: unsafe { mlx::mlx_vector_array_new() },
        }
    }

    /// Take ownership of a raw `mlx_vector_array` handle (e.g. a `mlx_split` out-param).
    #[inline]
    pub fn from_raw(raw: mlx::mlx_vector_array) -> Self {
        VectorArray { raw }
    }

    /// Append a borrowed array handle (the vector takes its own reference).
    pub fn append(&mut self, a: mlx::mlx_array) {
        unsafe { mlx::mlx_vector_array_append_value(self.raw, a) };
    }

    /// Number of arrays held.
    pub fn size(&self) -> usize {
        unsafe { mlx::mlx_vector_array_size(self.raw) }
    }

    /// A fresh owning reference to element `i` (the vector keeps its own; the returned `Array` owns
    /// the new reference and frees it on drop).
    pub fn get(&self, i: usize) -> Array {
        let mut a = unsafe { mlx::mlx_array_new() };
        unsafe { mlx::mlx_vector_array_get(&mut a, self.raw, i) };
        Array::from_raw(a)
    }

    #[inline]
    pub fn as_raw(&self) -> mlx::mlx_vector_array {
        self.raw
    }

    /// Consume the wrapper WITHOUT freeing, returning the raw handle (ownership transferred to the
    /// caller — e.g. handing a trace result to mlx via the closure's `out` param).
    #[inline]
    pub fn into_raw(self) -> mlx::mlx_vector_array {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut mlx::mlx_vector_array {
        &mut self.raw
    }
}

impl Default for VectorArray {
    fn default() -> Self {
        VectorArray::new()
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        unsafe { mlx::mlx_vector_array_free(self.raw) };
    }
}

/// Evaluate the whole boundary graph in one shot (mirrors the C++ single-`mlx_eval` boundary).
pub fn eval(outputs: &VectorArray) -> Result<(), String> {
    let rc = unsafe { mlx::mlx_eval(outputs.as_raw()) };
    if rc != 0 {
        return Err("mlx_eval failed".to_string());
    }
    Ok(())
}

/// Owning wrapper over an `mlx_closure` (a captured/compiled callable), freed once on drop.
///
/// Two flavours are used by the compiled-decode fast path:
///   * [`Closure::new_func_payload`] wraps a Rust `extern "C"` trace thunk plus an opaque payload
///     pointer — the *base* (un-compiled) closure whose body traces the whole decode subgraph.
///   * [`Closure::compile`] runs `mlx_compile` (shapeless) on a base closure and returns the
///     *compiled* closure that fuses the traced graph into far fewer kernel launches.
///     [`Closure::apply`] runs the closure over an input vector, returning the output arrays.
pub struct Closure {
    raw: mlx::mlx_closure,
}

impl Closure {
    /// Wrap a trace thunk + opaque payload as a base closure. The payload pointer must stay valid
    /// (and point at a stable allocation) for as long as this closure — and any closure compiled
    /// from it — may be applied. No destructor is registered (`dtor = None`); the payload is owned
    /// elsewhere (the plan).
    pub fn new_func_payload(
        fun: unsafe extern "C" fn(
            *mut mlx::mlx_vector_array,
            mlx::mlx_vector_array,
            *mut std::os::raw::c_void,
        ) -> std::os::raw::c_int,
        payload: *mut std::os::raw::c_void,
    ) -> Self {
        let raw = unsafe { mlx::mlx_closure_new_func_payload(Some(fun), payload, None) };
        Closure { raw }
    }

    /// Compile `base` shapeless (so a growing KV length never triggers a recompile) into a fused
    /// closure. Returns `Err` if `mlx_compile` fails (caller falls back to the eager path).
    pub fn compile(base: &Closure, shapeless: bool) -> Result<Closure, String> {
        let mut res = unsafe { mlx::mlx_closure_new() };
        let rc = unsafe { mlx::mlx_compile(&mut res, base.raw, shapeless) };
        if rc != 0 {
            unsafe { mlx::mlx_closure_free(res) };
            return Err("mlx_compile failed".to_string());
        }
        Ok(Closure { raw: res })
    }

    /// Apply the closure to `input`, returning the produced output arrays (owning). `Err` on any
    /// MLX failure inside the (traced or replayed) body.
    pub fn apply(&self, input: &VectorArray) -> Result<VectorArray, String> {
        let mut res = unsafe { mlx::mlx_vector_array_new() };
        let rc = unsafe { mlx::mlx_closure_apply(&mut res, self.raw, input.as_raw()) };
        if rc != 0 {
            unsafe { mlx::mlx_vector_array_free(res) };
            return Err("mlx_closure_apply failed".to_string());
        }
        Ok(VectorArray::from_raw(res))
    }
}

impl Drop for Closure {
    fn drop(&mut self) {
        unsafe { mlx::mlx_closure_free(self.raw) };
    }
}
