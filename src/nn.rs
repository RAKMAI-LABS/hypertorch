//! Inference-first NN layers. No autograd here — Problem 1 is serving,
//! so the forward pass is the product. HyperGrad's autograd can be
//! bolted on later if training is ever needed.

use crate::tensor::Tensor;

pub struct Linear {
    pub weight: Tensor, // [in_features, out_features]
    pub bias: Tensor,   // [out_features]
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        // He-style scaling so deep ReLU stacks don't explode
        let scale = (2.0 / in_features as f32).sqrt();
        let mut w = Tensor::randn(&[in_features, out_features], seed);
        for v in &mut w.data {
            *v *= scale;
        }
        Self {
            weight: w,
            bias: Tensor::zeros(&[out_features]),
        }
    }

    /// Load pretrained weights (e.g. exported from PyTorch).
    /// NOTE: PyTorch nn.Linear stores weight as [out, in] — transpose on export.
    pub fn from_weights(weight: Tensor, bias: Tensor) -> Self {
        assert_eq!(weight.ndim(), 2);
        assert_eq!(bias.shape[0], weight.shape[1], "bias must match out_features");
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add(&self.bias)
    }
}

/// Minimal MLP: Linear -> ReLU -> ... -> Linear (logits).
pub struct Mlp {
    pub layers: Vec<Linear>,
}

impl Mlp {
    pub fn new(dims: &[usize], seed: u64) -> Self {
        assert!(dims.len() >= 2, "need at least in/out dims");
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, w)| Linear::new(w[0], w[1], seed + i as u64 * 7919))
            .collect();
        Self { layers }
    }

    /// Forward to logits. Softmax is the caller's choice (serving often
    /// wants raw logits; argmax doesn't need softmax at all).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let last = self.layers.len() - 1;
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i != last {
                h = h.relu();
            }
        }
        h
    }

    pub fn predict(&self, x: &Tensor) -> Vec<usize> {
        let logits = self.forward(x);
        let cols = *logits.shape.last().unwrap();
        let rows = logits.numel() / cols;
        (0..rows)
            .map(|r| {
                let row = &logits.data[r * cols..(r + 1) * cols];
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_shapes_flow() {
        // batch of 4, MNIST-shaped: 784 -> 128 -> 10
        let mlp = Mlp::new(&[784, 128, 10], 42);
        let x = Tensor::randn(&[4, 784], 7);
        let logits = mlp.forward(&x);
        assert_eq!(logits.shape, vec![4, 10]);
        assert!(logits.data.iter().all(|v| v.is_finite()));
        let preds = mlp.predict(&x);
        assert_eq!(preds.len(), 4);
        assert!(preds.iter().all(|&p| p < 10));
    }

    #[test]
    fn linear_matches_manual_computation() {
        // y = xW + b, tiny known case
        let w = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]);
        let b = Tensor::from_vec(vec![0.5, -0.5], &[2]);
        let layer = Linear::from_weights(w, b);
        let x = Tensor::from_vec(vec![1., 1.], &[1, 2]);
        let y = layer.forward(&x);
        // [1,1] x [[1,2],[3,4]] = [4,6]; +bias = [4.5, 5.5]
        assert_eq!(y.data, vec![4.5, 5.5]);
    }
}
