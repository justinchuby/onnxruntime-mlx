//! Wave-1 op handler modules. Each op = translate handler + claim predicate + registry entry.

pub mod attention;
pub mod conv;
pub mod elementwise;
pub mod math;
pub mod matmul;
pub mod norm;
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
pub mod vision;
