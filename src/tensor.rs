//! HyperTorch tensor core — zero-dependency strided N-d tensor.
//!
//! Design decisions (deliberate, revisit later):
//! - f32 only (inference standard; f64 wastes bandwidth)
//! - Row-major, contiguous storage with explicit strides
//!   (strides make reshape/transpose free and prepare for views)
//! - Naive matmul first; correctness before speed.
//! - gather / scatter_add included from day one: these are the
//!   edge-indexed primitives BP-SIMP message passing lives on.

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    // ---------- constructors ----------

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0; n],
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "data length {} != shape product {}", data.len(), n);
        Self {
            data,
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
        }
    }

    /// Deterministic pseudo-random init (xorshift) — no rand crate needed.
    pub fn randn(shape: &[usize], seed: u64) -> Self {
        let n: usize = shape.iter().product();
        let mut state = seed.max(1);
        let mut data = Vec::with_capacity(n);
        for _ in 0..(n + 1) / 2 {
            // Box-Muller from two uniforms
            let u1 = xorshift_uniform(&mut state);
            let u2 = xorshift_uniform(&mut state);
            let r = (-2.0 * u1.max(1e-12).ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            data.push(r * theta.cos());
            data.push(r * theta.sin());
        }
        data.truncate(n);
        Self::from_vec(data, shape)
    }

    // ---------- shape utilities ----------

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Free reshape (contiguous only).
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        assert_eq!(self.numel(), shape.iter().product::<usize>(), "reshape numel mismatch");
        Tensor::from_vec(self.data.clone(), shape)
    }

    #[inline]
    pub fn at2(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.strides[0] + j * self.strides[1]]
    }

    // ---------- elementwise ----------

    pub fn add(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
            return Tensor::from_vec(data, &self.shape);
        }
        // Broadcast [rows, cols] + [cols]  (the bias-add case)
        if self.ndim() == 2 && other.ndim() == 1 && self.shape[1] == other.shape[0] {
            let (rows, cols) = (self.shape[0], self.shape[1]);
            let mut data = self.data.clone();
            for r in 0..rows {
                for c in 0..cols {
                    data[r * cols + c] += other.data[c];
                }
            }
            return Tensor::from_vec(data, &self.shape);
        }
        panic!("add: incompatible shapes {:?} vs {:?}", self.shape, other.shape);
    }

    pub fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::from_vec(data, &self.shape)
    }

    /// Numerically stable softmax over the last dimension.
    pub fn softmax_lastdim(&self) -> Tensor {
        let cols = *self.shape.last().expect("softmax on 0-d tensor");
        let rows = self.numel() / cols;
        let mut data = vec![0.0f32; self.numel()];
        for r in 0..rows {
            let row = &self.data[r * cols..(r + 1) * cols];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for c in 0..cols {
                data[r * cols + c] = exps[c] / sum;
            }
        }
        Tensor::from_vec(data, &self.shape)
    }

    // ---------- matmul ----------

    /// [m, k] x [k, n] -> [m, n]. Naive i-k-j loop order (cache-friendlier
    /// than i-j-k because the inner loop walks both B and C row-wise).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul: lhs must be 2-d");
        assert_eq!(other.ndim(), 2, "matmul: rhs must be 2-d");
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "matmul: inner dims {} vs {}", k, k2);

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                if a == 0.0 {
                    continue;
                }
                let brow = &other.data[p * n..(p + 1) * n];
                let orow = &mut out[i * n..(i + 1) * n];
                for j in 0..n {
                    orow[j] += a * brow[j];
                }
            }
        }
        Tensor::from_vec(out, &[m, n])
    }

    // ---------- graph primitives (BP-SIMP hooks) ----------

    /// gather rows: out[e] = self[index[e]]  — "read node features per edge".
    /// self: [num_nodes, feat], index: edge sources, out: [num_edges, feat]
    pub fn gather_rows(&self, index: &[usize]) -> Tensor {
        assert_eq!(self.ndim(), 2, "gather_rows: need [nodes, feat]");
        let feat = self.shape[1];
        let mut data = Vec::with_capacity(index.len() * feat);
        for &i in index {
            assert!(i < self.shape[0], "gather index {} out of bounds", i);
            data.extend_from_slice(&self.data[i * feat..(i + 1) * feat]);
        }
        Tensor::from_vec(data, &[index.len(), feat])
    }

    /// scatter-add rows: out[index[e]] += self[e] — "aggregate messages at
    /// destination nodes". self: [num_edges, feat] -> out: [num_nodes, feat]
    pub fn scatter_add_rows(&self, index: &[usize], num_nodes: usize) -> Tensor {
        assert_eq!(self.ndim(), 2, "scatter_add_rows: need [edges, feat]");
        assert_eq!(index.len(), self.shape[0], "one index per row");
        let feat = self.shape[1];
        let mut out = vec![0.0f32; num_nodes * feat];
        for (e, &dst) in index.iter().enumerate() {
            assert!(dst < num_nodes, "scatter index {} out of bounds", dst);
            let src = &self.data[e * feat..(e + 1) * feat];
            let dst_row = &mut out[dst * feat..(dst + 1) * feat];
            for f in 0..feat {
                dst_row[f] += src[f];
            }
        }
        Tensor::from_vec(out, &[num_nodes, feat])
    }

    /// L2 norm of (self - other): the convergence-halting signal from BP-SIMP.
    pub fn l2_diff(&self, other: &Tensor) -> f32 {
        assert_eq!(self.shape, other.shape, "l2_diff shape mismatch");
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn xorshift_uniform(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f32) / ((1u64 << 53) as f32) * 2.0_f32.powi(32)
        % 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_known_values() {
        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]);
        let b = Tensor::from_vec(vec![5., 6., 7., 8.], &[2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![19., 22., 43., 50.]);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let t = Tensor::from_vec(vec![1., 2., 3., 1000., 1001., 1002.], &[2, 3]);
        let s = t.softmax_lastdim();
        for r in 0..2 {
            let sum: f32 = s.data[r * 3..(r + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {} sums to {}", r, sum);
        }
        // large-value row must not produce NaN (stability check)
        assert!(s.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn bias_broadcast_add() {
        let x = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]);
        let b = Tensor::from_vec(vec![10., 20.], &[2]);
        let y = x.add(&b);
        assert_eq!(y.data, vec![11., 22., 13., 24.]);
    }

    #[test]
    fn gather_then_scatter_roundtrip() {
        // 3 nodes, feat=2; edges: 0->1, 1->2, 2->0, 0->2
        let nodes = Tensor::from_vec(vec![1., 1., 2., 2., 3., 3.], &[3, 2]);
        let src = vec![0usize, 1, 2, 0];
        let dst = vec![1usize, 2, 0, 2];
        let msgs = nodes.gather_rows(&src); // messages = source features
        let agg = msgs.scatter_add_rows(&dst, 3);
        // node0 receives from node2: [3,3]
        // node1 receives from node0: [1,1]
        // node2 receives from node1 and node0: [2+1, 2+1] = [3,3]
        assert_eq!(agg.data, vec![3., 3., 1., 1., 3., 3.]);
    }

    #[test]
    fn l2_diff_convergence_signal() {
        let a = Tensor::from_vec(vec![1., 2., 3.], &[3]);
        let b = Tensor::from_vec(vec![1., 2., 3.], &[3]);
        assert_eq!(a.l2_diff(&b), 0.0);
        let c = Tensor::from_vec(vec![1., 2., 4.], &[3]);
        assert!((a.l2_diff(&c) - 1.0).abs() < 1e-6);
    }
}
