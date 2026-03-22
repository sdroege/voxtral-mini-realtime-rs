//! GGUF model loader for Q4-quantized Voxtral.
//!
//! Reads a GGUF file containing Q4_0 quantized weights and builds a
//! [`Q4VoxtralModel`]. Handles both native file I/O and in-memory bytes
//! for WASM deployment.

use anyhow::{bail, Context, Result};
use burn::module::{Param, ParamId};
use burn::nn::conv::Conv1d;
use burn::nn::{Embedding, Linear};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::Path;
use tracing::info;

use crate::models::adapter::AudioLanguageAdapter;
use crate::models::config;
use crate::models::decoder::LanguageModel;
use crate::models::encoder::AudioEncoder;
use crate::models::layers::{
    AdaRmsNorm, Attention, ConvDownsampler, DecoderLayer, EncoderLayer, RmsNorm, RoPEConfig,
};
use crate::models::voxtral::VoxtralModel;
use crate::models::weights::{
    adapter_weight_names, conv_weight_names, decoder_layer_weight_names,
    encoder_layer_weight_names, prefixes,
};

use super::reader::{GgmlDtype, GgufReader, ShardedCursor};
use super::tensor::Q4Tensor;

/// Loads a Q4-quantized Voxtral model from a GGUF file.
pub struct Q4ModelLoader<R: Read + Seek> {
    reader: GgufReader<R>,
}

impl Q4ModelLoader<BufReader<File>> {
    /// Open a GGUF file from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
        let reader = GgufReader::open(BufReader::new(file))?;
        Ok(Self { reader })
    }
}

impl<'a> Q4ModelLoader<Cursor<&'a [u8]>> {
    /// Open a GGUF file from in-memory bytes (for WASM).
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let reader = GgufReader::from_bytes(bytes)?;
        Ok(Self { reader })
    }
}

impl Q4ModelLoader<ShardedCursor> {
    /// Open a GGUF from multiple shards (for WASM where a single >2 GB
    /// allocation is impossible due to the 32-bit address space).
    pub fn from_shards(shards: Vec<Vec<u8>>) -> Result<Self> {
        let reader = GgufReader::open(ShardedCursor::new(shards))?;
        Ok(Self { reader })
    }
}

impl<R: Read + Seek> Q4ModelLoader<R> {
    /// Load the complete Q4 Voxtral model.
    pub fn load<B: Backend>(&mut self, device: &B::Device) -> Result<VoxtralModel<B>> {
        info!(
            version = self.reader.version(),
            tensors = self.reader.tensor_count(),
            "Loading Q4 Voxtral model from GGUF"
        );

        info!(layers = 32, "Loading audio encoder");
        let encoder = self.load_encoder(device)?;

        info!("Loading audio-language adapter");
        let adapter = self.load_adapter(device)?;

        info!(layers = 26, "Loading language model");
        let decoder = self.load_decoder(device)?;

        info!("Q4 model loaded");

        Ok(VoxtralModel::new(encoder, decoder, adapter, 4))
    }

    /// Load the audio encoder.
    fn load_encoder<B: Backend>(&mut self, device: &B::Device) -> Result<AudioEncoder<B>> {
        let enc_config = config::AudioEncoderConfig::default();

        let conv = self.load_conv_downsampler(device)?;

        let rope = RoPEConfig::new(enc_config.head_dim, 4096)
            .with_theta(enc_config.rope_theta)
            .init(device);

        let mut layers = Vec::with_capacity(enc_config.n_layers);
        for i in 0..enc_config.n_layers {
            let layer = self
                .load_encoder_layer(i, &enc_config, device)
                .with_context(|| format!("Failed to load encoder layer {i}"))?;
            layers.push(layer);
        }

        let norm_name = format!("{}.transformer.norm.weight", prefixes::ENCODER);
        let norm = self.load_rms_norm(&norm_name, enc_config.norm_eps, device)?;

        Ok(AudioEncoder::new(conv, rope, layers, norm))
    }

    /// Load a single encoder layer.
    fn load_encoder_layer<B: Backend>(
        &mut self,
        layer_idx: usize,
        enc_config: &config::AudioEncoderConfig,
        device: &B::Device,
    ) -> Result<EncoderLayer<B>> {
        let names = encoder_layer_weight_names(layer_idx);

        let attention_norm =
            self.load_rms_norm(&names.attention_norm, enc_config.norm_eps, device)?;

        let wq =
            self.load_q4_linear_with_optional_bias(&names.wq_weight, Some(&names.wq_bias), device)?;
        let wk = self.load_q4_linear(&names.wk_weight, device)?;
        let wv =
            self.load_q4_linear_with_optional_bias(&names.wv_weight, Some(&names.wv_bias), device)?;
        let wo =
            self.load_q4_linear_with_optional_bias(&names.wo_weight, Some(&names.wo_bias), device)?;

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            enc_config.n_heads,
            enc_config.n_heads, // MHA
            enc_config.head_dim,
            Some(enc_config.sliding_window),
        );

        let ffn_norm = self.load_rms_norm(&names.ffn_norm, enc_config.norm_eps, device)?;

        let w1 = self.load_q4_linear(&names.w1_weight, device)?;
        let w2 =
            self.load_q4_linear_with_optional_bias(&names.w2_weight, Some(&names.w2_bias), device)?;
        let w3 = self.load_q4_linear(&names.w3_weight, device)?;

        Ok(EncoderLayer::new(
            attention_norm,
            attention,
            ffn_norm,
            w1,
            w2,
            w3,
        ))
    }

    /// Load the conv downsampler (stays f32).
    fn load_conv_downsampler<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Result<ConvDownsampler<B>> {
        let names = conv_weight_names();

        let conv1_weight: Tensor<B, 3> = self.load_f32_tensor(&names.conv1_weight, device)?;
        let conv1_bias: Tensor<B, 1> = self.load_f32_tensor(&names.conv1_bias, device)?;
        let conv2_weight: Tensor<B, 3> = self.load_f32_tensor(&names.conv2_weight, device)?;
        let conv2_bias: Tensor<B, 1> = self.load_f32_tensor(&names.conv2_bias, device)?;

        let conv1 = conv1d_from_weights(conv1_weight, Some(conv1_bias));
        let conv2 = conv1d_from_weights(conv2_weight, Some(conv2_bias));

        Ok(ConvDownsampler::new(conv1, conv2))
    }

    /// Load the language model decoder.
    fn load_decoder<B: Backend>(&mut self, device: &B::Device) -> Result<LanguageModel<B>> {
        let dec_config = config::LanguageModelConfig::default();

        // Token embeddings — Q4_0 in the GGUF, dequantize to f32 for tied lm_head
        let tok_embeddings_weight = self.load_tok_embeddings(device)?;
        let d_model = tok_embeddings_weight.dims()[1];
        let tok_embeddings = Embedding {
            weight: Param::initialized(ParamId::new(), tok_embeddings_weight),
        };

        let rope = RoPEConfig::new(dec_config.head_dim, 16384)
            .with_theta(dec_config.rope_theta)
            .init(device);

        let mut layers = Vec::with_capacity(dec_config.n_layers);
        for i in 0..dec_config.n_layers {
            let layer = self
                .load_decoder_layer(i, &dec_config, device)
                .with_context(|| format!("Failed to load decoder layer {i}"))?;
            layers.push(layer);
        }

        let norm = self.load_rms_norm(prefixes::FINAL_NORM, dec_config.norm_eps, device)?;

        Ok(LanguageModel::new(
            tok_embeddings,
            rope,
            layers,
            norm,
            d_model,
        ))
    }

    /// Load token embeddings (Q4_0 → dequantized f32).
    ///
    /// Dequantizes on CPU to avoid a synchronous GPU readback, which panics
    /// on WASM where `block_on()` is unavailable.
    fn load_tok_embeddings<B: Backend>(&mut self, device: &B::Device) -> Result<Tensor<B, 2>> {
        let name = prefixes::TOK_EMBEDDINGS;
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        let shape = reverse_gguf_dims(info.shape());

        match info.dtype() {
            GgmlDtype::Q4_0 => {
                let bytes = self.reader.tensor_data(name)?;
                let f32_data = dequantize_q4_0_cpu(&bytes, shape[0] * shape[1]);
                let tensor_data = TensorData::new(f32_data, [shape[0], shape[1]]);
                Ok(Tensor::from_data(tensor_data, device))
            }
            GgmlDtype::F32 | GgmlDtype::F16 => self.load_f32_tensor(name, device),
            #[allow(unreachable_patterns)]
            other => bail!("Unsupported dtype {other:?} for tok_embeddings"),
        }
    }

    /// Load a single decoder layer.
    fn load_decoder_layer<B: Backend>(
        &mut self,
        layer_idx: usize,
        dec_config: &config::LanguageModelConfig,
        device: &B::Device,
    ) -> Result<DecoderLayer<B>> {
        let names = decoder_layer_weight_names(layer_idx);

        // ADA RMSNorm conditioning — Q4_0 in GGUF
        let ada_w0 = self.load_q4_linear(&names.ada_norm_down, device)?;
        let ada_w2 = self.load_q4_linear(&names.ada_norm_up, device)?;
        let ada_rms_norm = AdaRmsNorm::new(ada_w0, ada_w2, dec_config.norm_eps);

        let attention_norm =
            self.load_rms_norm(&names.attention_norm, dec_config.norm_eps, device)?;

        let wq = self.load_q4_linear(&names.wq_weight, device)?;
        let wk = self.load_q4_linear(&names.wk_weight, device)?;
        let wv = self.load_q4_linear(&names.wv_weight, device)?;
        let wo = self.load_q4_linear(&names.wo_weight, device)?;

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            dec_config.n_heads,
            dec_config.n_kv_heads,
            dec_config.head_dim,
            Some(dec_config.sliding_window),
        );

        let ffn_norm = self.load_rms_norm(&names.ffn_norm, dec_config.norm_eps, device)?;

        let w1 = self.load_q4_linear(&names.w1_weight, device)?;
        let w2 = self.load_q4_linear(&names.w2_weight, device)?;
        let w3 = self.load_q4_linear(&names.w3_weight, device)?;

        Ok(DecoderLayer::new(
            ada_rms_norm,
            attention_norm,
            attention,
            ffn_norm,
            w1,
            w2,
            w3,
        ))
    }

    /// Load the audio-language adapter.
    fn load_adapter<B: Backend>(&mut self, device: &B::Device) -> Result<AudioLanguageAdapter<B>> {
        let names = adapter_weight_names();
        let linear1 = self.load_q4_linear(&names.linear1_weight, device)?;
        let linear2 = self.load_q4_linear(&names.linear2_weight, device)?;
        Ok(AudioLanguageAdapter::new(linear1, linear2))
    }

    // -----------------------------------------------------------------------
    // Primitive loading helpers
    // -----------------------------------------------------------------------

    /// Load a Q4_0 tensor as a [`Q4Linear`] (no bias).
    fn load_q4_linear<B: Backend>(&mut self, name: &str, device: &B::Device) -> Result<Linear<B>> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        if info.dtype() != GgmlDtype::Q4_0 {
            bail!("Expected Q4_0 for '{name}', got {:?}", info.dtype());
        }

        let shape = reverse_gguf_dims(info.shape());
        let bytes = self.reader.tensor_data(name)?;
        let q4 = Q4Tensor::from_q4_bytes(&bytes, [shape[0], shape[1]], device)?;
        Ok(Linear {
            weight: Param::initialized(ParamId::new(), q4.tensor),
            bias: None,
        })
    }

    /// Load a Q4_0 tensor with an optional F32 bias as a [`Q4Linear`].
    fn load_q4_linear_with_optional_bias<B: Backend>(
        &mut self,
        weight_name: &str,
        bias_name: Option<&str>,
        device: &B::Device,
    ) -> Result<Linear<B>> {
        let info = self
            .reader
            .tensor_info(weight_name)
            .with_context(|| format!("Tensor '{weight_name}' not found"))?
            .clone();

        if info.dtype() != GgmlDtype::Q4_0 {
            bail!("Expected Q4_0 for '{weight_name}', got {:?}", info.dtype());
        }

        let shape = reverse_gguf_dims(info.shape());
        let bytes = self.reader.tensor_data(weight_name)?;
        let q4 = Q4Tensor::from_q4_bytes(&bytes, [shape[0], shape[1]], device)?;

        let bias = if let Some(bias_name) = bias_name {
            if self.reader.tensor_info(bias_name).is_some() {
                let bias_tensor: Tensor<B, 1> = self.load_f32_tensor(bias_name, device)?;
                Some(bias_tensor)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Linear {
            weight: Param::initialized(ParamId::new(), q4.tensor),
            bias: bias.map(|b| Param::initialized(ParamId::new(), b)),
        })
    }

    /// Load an F32/F16 tensor from GGUF.
    fn load_f32_tensor<const D: usize, B: Backend>(
        &mut self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        let shape: Vec<usize> = reverse_gguf_dims(info.shape());
        let bytes = self.reader.tensor_data(name)?;

        let data: Vec<f32> = match info.dtype() {
            GgmlDtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            GgmlDtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
            GgmlDtype::Q4_0 => bail!("Cannot load Q4_0 tensor '{name}' as f32; use load_q4_linear"),
        };

        let tensor_data = TensorData::new(data, shape);
        Ok(Tensor::from_data(tensor_data, device))
    }

    /// Load an RmsNorm layer from GGUF.
    fn load_rms_norm<B: Backend>(
        &mut self,
        name: &str,
        eps: f64,
        device: &B::Device,
    ) -> Result<RmsNorm<B>> {
        let weight: Tensor<B, 1> = self.load_f32_tensor(name, device)?;
        Ok(RmsNorm {
            weight: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), weight),
                epsilon: eps,
            },
        })
    }
}

/// Reverse GGUF dimension order to get PyTorch convention.
///
/// GGUF stores dimensions in reversed order (row-major innermost first),
/// while PyTorch uses `[out_features, in_features]` convention.
fn reverse_gguf_dims(gguf_dims: &[u64]) -> Vec<usize> {
    gguf_dims.iter().rev().map(|&d| d as usize).collect()
}

/// Dequantize Q4_0 blocks on CPU, returning `num_elements` f32 values.
///
/// Same logic as [`Q4Tensor::dequantize`] but operates on raw bytes without
/// a GPU round-trip, making it safe on WASM.
fn dequantize_q4_0_cpu(raw: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let mut output = vec![0.0f32; num_elements];
    for block_idx in 0..num_blocks {
        let offset = block_idx * 18;
        let d = half::f16::from_bits(u16::from_le_bytes([raw[offset], raw[offset + 1]])).to_f32();
        let base = block_idx * 32;
        for i in 0..16 {
            let byte = raw[offset + 2 + i];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
            output[base + i] = lo * d;
            output[base + i + 16] = hi * d;
        }
    }
    output
}

/// Create a `Conv1d` from weight tensors (matches existing `loader.rs` helper).
fn conv1d_from_weights<B: Backend>(weight: Tensor<B, 3>, bias: Option<Tensor<B, 1>>) -> Conv1d<B> {
    use burn::module::Ignored;

    Conv1d {
        weight: Param::initialized(ParamId::new(), weight),
        bias: bias.map(|b| Param::initialized(ParamId::new(), b)),
        stride: 2,
        kernel_size: 3,
        dilation: 1,
        groups: 1,
        padding: Ignored(burn::nn::PaddingConfig1d::Explicit(1)),
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;
    use cubecl::wgpu::WgpuDevice;

    use super::*;
    use std::path::PathBuf;

    fn gguf_path() -> PathBuf {
        PathBuf::from("models/voxtral-q4.gguf")
    }

    #[test]
    fn test_load_q4_model() {
        let path = gguf_path();
        if !path.exists() {
            println!("Skipping: GGUF model not found at {}", path.display());
            return;
        }

        let device = WgpuDevice::default();
        let mut loader = Q4ModelLoader::from_file(&path).unwrap();
        let model = loader.load::<Wgpu>(&device).unwrap();

        // Verify layer counts
        assert_eq!(model.encoder().n_layers(), 32);
        assert_eq!(model.decoder().n_layers(), 26);
        assert_eq!(model.decoder().d_model(), 3072);

        println!("Q4 model loaded successfully from GGUF!");
    }

    #[test]
    fn test_q4_forward_shape() {
        let path = gguf_path();
        if !path.exists() {
            println!("Skipping: GGUF model not found at {}", path.display());
            return;
        }

        let device = WgpuDevice::default();
        let mut loader = Q4ModelLoader::from_file(&path).unwrap();
        let model = loader.load(&device).unwrap();

        // Small mel input: [1, 128, 320]
        let mel = Tensor::<Wgpu, 3>::zeros([1, 128, 320], &device);
        let t_embed = Tensor::<Wgpu, 3>::zeros([1, 1, 3072], &device);

        let logits = model.forward(mel, t_embed);

        // 320 mel frames → 80 after conv → 20 after reshape(4)
        assert_eq!(logits.dims()[0], 1);
        assert_eq!(logits.dims()[1], 20);
        assert_eq!(logits.dims()[2], 131072);

        println!("Q4 forward pass shape: {:?}", logits.dims());
    }
}
