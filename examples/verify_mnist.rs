//! Correctness milestone: load PyTorch-trained weights, run HyperTorch's
//! forward pass, and compare logits element-by-element against PyTorch's.
//!
//! Run: cargo run --release --example verify_mnist -- model.rtw verify.rtw
//!
//! Pass criterion: max |Δlogit| < 1e-4 (f32 matmul reassociation across
//! different backends makes exact equality impossible; 1e-4 on logits is
//! bit-level agreement for practical purposes, and argmax must match 64/64).

use hypertorch::nn::{Linear, Mlp};
use hypertorch::weights::load_rtw;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or("model.rtw");
    let verify_path = args.get(2).map(String::as_str).unwrap_or("verify.rtw");

    let mut w = load_rtw(model_path).expect("failed to load model.rtw");
    let mut v = load_rtw(verify_path).expect("failed to load verify.rtw");

    let mlp = Mlp {
        layers: vec![
            Linear::from_weights(w.remove("fc1.weight").unwrap(), w.remove("fc1.bias").unwrap()),
            Linear::from_weights(w.remove("fc2.weight").unwrap(), w.remove("fc2.bias").unwrap()),
        ],
    };

    let inputs = v.remove("inputs").expect("verify.rtw missing inputs");
    let torch_logits = v.remove("logits").expect("verify.rtw missing logits");
    let labels = v.remove("labels").expect("verify.rtw missing labels");

    let t0 = std::time::Instant::now();
    let rust_logits = mlp.forward(&inputs);
    let dt = t0.elapsed();

    assert_eq!(rust_logits.shape, torch_logits.shape, "logit shape mismatch");

    // element-wise comparison
    let mut max_diff = 0.0f32;
    for (a, b) in rust_logits.data.iter().zip(&torch_logits.data) {
        max_diff = max_diff.max((a - b).abs());
    }

    // argmax agreement + accuracy
    let preds = mlp.predict(&inputs);
    let cols = *torch_logits.shape.last().unwrap();
    let torch_preds: Vec<usize> = (0..torch_logits.shape[0])
        .map(|r| {
            let row = &torch_logits.data[r * cols..(r + 1) * cols];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect();
    let argmax_agree = preds.iter().zip(&torch_preds).filter(|(a, b)| a == b).count();
    let correct = preds
        .iter()
        .zip(&labels.data)
        .filter(|(p, l)| **p == **l as usize)
        .count();

    let n = inputs.shape[0];
    println!("batch size:            {}", n);
    println!("forward pass:          {:?} ({:.1} µs/sample)", dt, dt.as_micros() as f64 / n as f64);
    println!("max |Δlogit| vs torch: {:.2e}", max_diff);
    println!("argmax agreement:      {}/{}", argmax_agree, n);
    println!("accuracy vs labels:    {}/{}", correct, n);

    if max_diff < 1e-4 && argmax_agree == n {
        println!("\n✅ CORRECTNESS MILESTONE PASSED — HyperTorch forward pass matches PyTorch");
    } else {
        println!("\n❌ mismatch — check weight transpose ([out,in] vs [in,out]) first");
        std::process::exit(1);
    }
}
