//! GGUF quantized inference on GPU.
//!
//! Provides GGUF file reading, Q4_0 GPU tensor storage, and a fused
//! dequant+matmul compute shader launched through Burn's custom kernel API.
//! Named after the container format rather than a specific quant type so
//! future formats (Q4_K_M, Q5_K_M, Q8_0, …) live alongside Q4_0.

pub mod loader;
pub mod reader;
pub mod tensor;

#[cfg(test)]
mod tests;

pub use loader::Q4ModelLoader;
pub use reader::{GgmlDtype, GgufReader, GgufTensorInfo, ShardedCursor};
