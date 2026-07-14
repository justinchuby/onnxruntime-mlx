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
        Array {
            raw: unsafe {
                mlx::mlx_array_new_data(data, shape.as_ptr(), shape.len() as i32, dtype)
            },
        }
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
        unsafe { mlx::mlx_array_data_uint8(self.raw) as *const u8 }
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
