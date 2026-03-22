//! Q4_0 quantized weight tensor stored on GPU.
//!
//! [`Q4Tensor`] uploads raw Q4_0 blocks to a GPU storage buffer and provides
//! a [`dequantize`](Q4Tensor::dequantize) method for diagnostics/testing.
//! The primary inference path is [`q4_matmul`](super::op::q4_matmul), which
//! dequantizes on-the-fly inside a fused compute shader.

use anyhow::{ensure, Result};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use cubecl::quant::scheme::{
    QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};

/// A Q4_0 quantized weight tensor living on GPU.
///
/// The buffer contains raw Q4_0 blocks (18 bytes per block of 32 elements),
/// laid out exactly as in GGUF. The WGSL shader interprets the buffer as
/// `array<f16>` with 9 f16 slots per block.
pub struct Q4Tensor<B: Backend> {
    pub tensor: Tensor<B, 2>,
}

impl<B: Backend> Q4Tensor<B> {
    /// Upload raw Q4_0 bytes to a GPU storage buffer.
    ///
    /// Shape is `[N, K]` = `[out_features, in_features]`, matching PyTorch/GGUF
    /// convention. `raw_bytes` must contain exactly `(N * K / 32) * 18` bytes.
    /// The element count `N * K` must be divisible by 32.
    pub fn from_q4_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &B::Device) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        ensure!(
            num_elements % 32 == 0,
            "Q4_0 requires element count divisible by 32, got {num_elements}"
        );
        let num_blocks = num_elements / 32;
        let expected_bytes = num_blocks * 18;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q4_0 byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        let scheme = QuantScheme::default()
            .with_level(QuantLevel::block([32]))
            .with_mode(QuantMode::Symmetric)
            .with_value(QuantValue::Q4S)
            .with_store(QuantStore::PackedU32(0))
            .with_param(QuantParam::F16);

        // Convert between GGUF Q4_0 blocks and the format expected by burn.
        //
        // GGUF has the scale for each 32 value block in front of each block, burn
        // has first all values and then all scales. That's the first difference
        // that needs reorganization.
        let mut bytes = vec![0u8; (16 + 2) * num_blocks];
        let (values, scales) = bytes.split_at_mut(16 * num_blocks);
        for ((values, scales), chunk) in Iterator::zip(
            Iterator::zip(values.chunks_exact_mut(16), scales.chunks_exact_mut(2)),
            raw_bytes.chunks_exact(18),
        ) {
            scales.copy_from_slice(&chunk[..2]);
            let chunk = &chunk[2..];
            for i in 0..8 {
                // GGUF stores values 0..8 in the lower nibbles and values
                // 8..16 in the upper nibbles, burn wants them b0 in the low
                // nibble of byte 0, b1 in the high nibble of the same byte,
                // etc.
                let b0 = chunk[i * 2] & 0x0F;
                let b16 = (chunk[i * 2] >> 4) & 0x0F;
                let b1 = chunk[i * 2 + 1] & 0x0F;
                let b17 = (chunk[i * 2 + 1] >> 4) & 0x0F;

                // GGUF uses unsigned integers, burn uses signed integers.
                let b0 = (b0 as i16 - 8) as u8;
                let b0 = b0 & 0x0F;
                let b1 = (b1 as i16 - 8) as u8;
                let b1 = b1 & 0x0F;
                let b16 = (b16 as i16 - 8) as u8;
                let b16 = b16 & 0x0F;
                let b17 = (b17 as i16 - 8) as u8;

                values[i + 0] = (b1 << 4) | b0;
                values[i + 8] = (b17 << 4) | b16;
            }
        }

        let data = TensorData::from_bytes_vec(bytes, [k * n], burn::tensor::DType::QFloat(scheme));
        let tensor = Tensor::<B, 1>::from_data(data, device);
        // FIXME: Dequantization needed because of https://github.com/tracel-ai/burn/issues/4659
        let tensor = tensor.dequantize().reshape(shape).transpose();
        Ok(Self { tensor })
    }

    /// Logical weight dimensions `[N, K]` = `[out_features, in_features]`.
    pub fn shape(&self) -> [usize; 2] {
        [self.tensor.shape().dims[0], self.tensor.shape().dims[1]]
    }

    pub fn dequantize(&self) -> Tensor<B, 2> {
        self.tensor.clone().dequantize()
    }
}
