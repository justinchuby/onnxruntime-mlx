//! Wave-1 op handler modules. Each op = translate handler + claim predicate + registry entry.

pub mod attention;
pub mod conv;
pub mod elementwise;
pub mod image;
pub mod math;
pub mod matmul;
pub mod norm;
pub mod onnx_ml_linear;
pub mod onnx_ml_preprocess;
pub mod quant;
pub mod reduction;
pub mod shape;
// signal/random/recurrent/ssm/misc/controlflow
pub mod controlflow;
pub mod misc;
pub mod random;
pub mod recurrent;
pub mod signal;
pub mod ssm;
pub mod stragglers;
pub mod vision;
