//! `MlxEp` — our `OrtEp` C-ABI vtable, plus the per-fused-node compute info that
//! runs an `Add` node through mlx-c. Spike scope: claims `Add` (fp32) only.

use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

use crate::factory::ORT_API_VERSION;
use crate::sys::{mlx, ort};

#[repr(C)]
pub struct MlxEp {
    base: ort::OrtEp,
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    name: CString,
    stream: mlx::mlx_stream,
}

impl MlxEp {
    pub fn new(
        ort_api: *const ort::OrtApi,
        ep_api: *const ort::OrtEpApi,
        name: &CStr,
        _logger: *const ort::OrtLogger,
    ) -> Box<MlxEp> {
        let mut base: ort::OrtEp = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetCapability = Some(get_capability);
        base.Compile = Some(compile);
        base.ReleaseNodeComputeInfos = Some(release_node_compute_infos);
        base.GetDefaultMemoryDevice = Some(get_default_memory_device);
        Box::new(MlxEp {
            base,
            ort_api,
            ep_api,
            name: name.to_owned(),
            stream: unsafe { mlx::mlx_default_gpu_stream_new() },
        })
    }

    pub fn as_ptr(self: Box<Self>) -> *mut ort::OrtEp {
        Box::into_raw(self) as *mut ort::OrtEp
    }
}

impl Drop for MlxEp {
    fn drop(&mut self) {
        // RAII: the per-EP mlx stream is released exactly once when ORT calls
        // ReleaseEp (which drops our Box<MlxEp>). No manual free at call sites.
        unsafe { mlx::mlx_stream_free(self.stream) };
    }
}

#[inline]
unsafe fn this(p: *const ort::OrtEp) -> *const MlxEp {
    p as *const MlxEp
}

unsafe extern "C" fn get_name(p: *const ort::OrtEp) -> *const c_char {
    (*this(p)).name.as_ptr()
}

unsafe extern "C" fn get_default_memory_device(
    _p: *const ort::OrtEp,
    device: *mut *const ort::OrtMemoryDevice,
) -> *mut ort::OrtStatus {
    // Spike does not advertise device memory; I/O stays on the CPU allocator.
    *device = ptr::null();
    ptr::null_mut()
}

unsafe extern "C" fn get_capability(
    p: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    let ep = &*this(p);
    let api = &*ep.ort_api;
    let ep_api = &*ep.ep_api;

    let mut num: usize = 0;
    let st = (api.Graph_GetNumNodes.unwrap())(graph, &mut num);
    if !st.is_null() {
        return st;
    }
    if num == 0 {
        return ptr::null_mut();
    }
    let mut nodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num];
    let st = (api.Graph_GetNodes.unwrap())(graph, nodes.as_mut_ptr(), num);
    if !st.is_null() {
        return st;
    }

    let get_optype = api.Node_GetOperatorType.unwrap();
    let add_fuse = ep_api.EpGraphSupportInfo_AddNodesToFuse.unwrap();
    let mut claimed = 0usize;
    for &node in &nodes {
        let mut op: *const c_char = ptr::null();
        let st = get_optype(node, &mut op);
        if !st.is_null() {
            return st;
        }
        if op.is_null() {
            continue;
        }
        if CStr::from_ptr(op).to_bytes() == b"Add" {
            let mut opts: ort::OrtNodeFusionOptions = std::mem::zeroed();
            opts.ort_version_supported = ORT_API_VERSION;
            let group = [node];
            let st = add_fuse(support, group.as_ptr(), 1, &opts);
            if !st.is_null() {
                return st;
            }
            claimed += 1;
        }
    }
    eprintln!("[rust-mlx-ep] GetCapability: claimed {claimed} Add node(s) of {num}");
    ptr::null_mut()
}

unsafe extern "C" fn compile(
    p: *mut ort::OrtEp,
    _graphs: *mut *const ort::OrtGraph,
    _fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    _ep_context_nodes: *mut *mut ort::OrtNode,
) -> *mut ort::OrtStatus {
    let ep = &*this(p);
    for i in 0..count {
        let info = AddComputeInfo::new(ep.ort_api, ep.stream);
        *node_compute_infos.add(i) = Box::into_raw(info) as *mut ort::OrtNodeComputeInfo;
    }
    ptr::null_mut()
}

unsafe extern "C" fn release_node_compute_infos(
    _p: *mut ort::OrtEp,
    infos: *mut *mut ort::OrtNodeComputeInfo,
    num: usize,
) {
    for i in 0..num {
        let ptr = *infos.add(i);
        if !ptr.is_null() {
            drop(Box::from_raw(ptr as *mut AddComputeInfo));
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fused-node compute info: runs Add via mlx-c.
// ---------------------------------------------------------------------------

#[repr(C)]
struct AddComputeInfo {
    base: ort::OrtNodeComputeInfo,
    ort_api: *const ort::OrtApi,
    stream: mlx::mlx_stream,
}

impl AddComputeInfo {
    fn new(ort_api: *const ort::OrtApi, stream: mlx::mlx_stream) -> Box<AddComputeInfo> {
        let mut base: ort::OrtNodeComputeInfo = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.CreateState = Some(create_state);
        base.Compute = Some(compute_add);
        base.ReleaseState = Some(release_state);
        Box::new(AddComputeInfo {
            base,
            ort_api,
            stream,
        })
    }
}

unsafe extern "C" fn create_state(
    this_ptr: *mut ort::OrtNodeComputeInfo,
    _compute_context: *mut ort::OrtNodeComputeContext,
    compute_state: *mut *mut c_void,
) -> *mut ort::OrtStatus {
    *compute_state = this_ptr as *mut c_void;
    ptr::null_mut()
}

unsafe extern "C" fn release_state(_this: *mut ort::OrtNodeComputeInfo, _state: *mut c_void) {}

/// Read an ORT fp32 tensor and copy it into a fresh mlx array.
unsafe fn ort_to_mlx_f32(api: &ort::OrtApi, val: *const ort::OrtValue) -> mlx::mlx_array {
    let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
    (api.GetTensorTypeAndShape.unwrap())(val, &mut info);
    let mut nd: usize = 0;
    (api.GetDimensionsCount.unwrap())(info, &mut nd);
    let mut dims = vec![0i64; nd];
    (api.GetDimensions.unwrap())(info, dims.as_mut_ptr(), nd);
    (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);

    let mut data: *const c_void = ptr::null();
    (api.GetTensorData.unwrap())(val, &mut data);

    let shape_i32: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
    mlx::mlx_array_new_data(
        data,
        shape_i32.as_ptr(),
        nd as i32,
        mlx::mlx_dtype__MLX_FLOAT32,
    )
}

unsafe extern "C" fn compute_add(
    _this: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
    kctx: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    let info = &*(state as *const AddComputeInfo);
    let api = &*info.ort_api;

    let get_in = api.KernelContext_GetInput.unwrap();
    let mut a: *const ort::OrtValue = ptr::null();
    let mut b: *const ort::OrtValue = ptr::null();
    get_in(kctx, 0, &mut a);
    get_in(kctx, 1, &mut b);

    let a_arr = ort_to_mlx_f32(api, a);
    let b_arr = ort_to_mlx_f32(api, b);

    let mut c = mlx::mlx_array_new();
    mlx::mlx_add(&mut c, a_arr, b_arr, info.stream);
    mlx::mlx_array_eval(c);

    let cnd = mlx::mlx_array_ndim(c);
    let cshape = mlx::mlx_array_shape(c);
    let odims: Vec<i64> = (0..cnd).map(|i| *cshape.add(i) as i64).collect();

    let get_out = api.KernelContext_GetOutput.unwrap();
    let mut out: *mut ort::OrtValue = ptr::null_mut();
    get_out(kctx, 0, odims.as_ptr(), cnd, &mut out);
    let mut odata: *mut c_void = ptr::null_mut();
    (api.GetTensorMutableData.unwrap())(out, &mut odata);

    let n = mlx::mlx_array_size(c);
    let src = mlx::mlx_array_data_float32(c);
    ptr::copy_nonoverlapping(src as *const u8, odata as *mut u8, n * 4);

    mlx::mlx_array_free(a_arr);
    mlx::mlx_array_free(b_arr);
    mlx::mlx_array_free(c);

    eprintln!("[rust-mlx-ep] Add computed via mlx-c ({n} elems)");
    ptr::null_mut()
}
