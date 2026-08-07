//! Demonstrates the BP-SIMP-shaped inference loop on HyperTorch primitives:
//!   gather (read source nodes) -> transform -> scatter_add (aggregate at
//!   destinations) -> update -> check L2 convergence -> halt.
//!
//! This is inference-side only: variable-depth execution with adaptive
//! halting — the exact pattern vLLM-style transformer servers cannot run.
//!
//! Run: cargo run --example message_passing

use hypertorch::Tensor;

fn main() {
    // Ring graph: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
    let num_nodes = 6;
    let feat = 4;
    let src: Vec<usize> = (0..num_nodes).collect();
    let dst: Vec<usize> = (0..num_nodes).map(|i| (i + 1) % num_nodes).collect();

    // One "hot" node injects signal; watch it propagate around the ring.
    let mut state = Tensor::zeros(&[num_nodes, feat]);
    for f in 0..feat {
        state.data[f] = 1.0; // node 0
    }

    let max_iters = 50;
    let eps = 1e-4;
    let damping = 0.6; // new = damping*aggregated + (1-damping)*old

    for t in 1..=max_iters {
        let prev = state.clone();

        // message = source node state (identity message fn for the demo;
        // in BP-SIMP this is a learned belief-channel transform)
        let messages = state.gather_rows(&src);
        let aggregated = messages.scatter_add_rows(&dst, num_nodes);

        // damped update toward aggregated messages
        let mut next = Tensor::zeros(&[num_nodes, feat]);
        for i in 0..state.data.len() {
            next.data[i] = damping * aggregated.data[i] + (1.0 - damping) * state.data[i];
        }
        state = next;

        // convergence-based halting: the BP-SIMP signal
        let delta = state.l2_diff(&prev);
        println!("iter {:2}  ||Δstate||₂ = {:.6}", t, delta);
        if delta < eps {
            println!("\nconverged at iteration {} (adaptive halt, budget was {})", t, max_iters);
            break;
        }
    }

    println!("\nfinal node states (feature 0):");
    for n in 0..num_nodes {
        println!("  node {}: {:.4}", n, state.data[n * feat]);
    }
}
