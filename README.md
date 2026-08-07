# HyperTorch
[![CI](https://github.com/RAKMAI-LABS/hypertorch/actions/workflows/ci.yml/badge.svg)](https://github.com/RAKMAI-LABS/hypertorch/actions)
**Train in Python. Ship in Rust.**

A zero-dependency ML inference engine and model server in pure Rust.
Load weights trained in PyTorch, serve them from a single sub-megabyte
binary --> no Python runtime, no dependency tree, no gigabytes of RAM.

## Why

Inference is where AI budgets go: models are trained once and served
forever, and the serving stack around the model — startup time, memory,
deployment weight --> is pure overhead. HyperTorch attacks that overhead:

| metric | HyperTorch | FastAPI + PyTorch | advantage |
|---|---|---|---|
| cold start (launch → serving) | **50–60 ms** | 2.5–8 s | **~50–130×** |
| memory (RSS) | **5–6 MB** | ~235 MB | **~40×** |
| latency p50 (keep-alive) | **37–53 µs** | 670–720 µs | **~13–19×** |
| throughput (8 clients) | 1,800–3,700 req/s | 550–1,000 req/s | **~2–6×** |
| server binary | **~220 KB** | multi-GB environment | — |

*Measured on Windows 11 (x86-64) serving an MNIST MLP (784→128→10),
identical raw-socket benchmark client against both servers; ranges span
multiple runs. Reproduce with `python python/bench.py`. Model compute is
identical on both sides (same weights, same math) --> the gap is the stack
around the model. One honest footnote: connection-per-request latency on
Windows loopback shows a ~16 ms server-close teardown artifact (platform
TCP quirk, absent on Linux, absent under keep-alive --> which is what
production clients use).*

**Correctness:** HyperTorch's forward pass matches PyTorch's logits to
< 1e-5 (f32 noise floor), verified element-wise on exported weights —
`cargo run --release --example verify_mnist`.

## Quickstart

```bash
# 1. Train + export in PyTorch (or bring your own weights)
python python/export_mnist.py            # writes model.rtw + verify.rtw

# 2. Verify bit-level agreement with PyTorch
cargo run --release --example verify_mnist -- model.rtw verify.rtw

# 3. Serve
cargo run --release --example serve -- model.rtw 7878
# READY port=7878 startup_micros=1613

# 4. Predict (body = N×784 little-endian f32)
curl -X POST --data-binary @image.bin http://127.0.0.1:7878/predict
# {"predictions":[7],"micros":9}
```

## What's inside

- **`src/tensor.rs`**  strided N-d f32 tensor: matmul, broadcasting,
  ReLU, numerically stable softmax, plus graph primitives
  (`gather_rows`, `scatter_add_rows`, `l2_diff`) for message-passing
  models with convergence-based adaptive halting --> a model class that
  transformer-serving engines (vLLM et al.) structurally cannot run.
- **`src/nn.rs`**  inference-first `Linear`/`Mlp` with PyTorch weight
  loading (transpose handled at export, once).
- **`src/weights.rs`**  RTW, a deliberately simple binary weight format
  (~80 lines to parse); safetensors support planned.
- **`examples/serve.rs`**  hand-rolled HTTP/1.1 server on `std::net`:
  keep-alive, `Connection: close` compliance, `TCP_NODELAY`,
  thread-per-connection. ~130 lines, zero crates.
- **`python/`**  PyTorch export script, FastAPI baseline server, and the
  raw-socket benchmark harness.

Zero dependencies means zero: the library *and* every example build from
`std` alone. `cargo tree` prints one line.

## Design choices

- **Inference-first.** Training stays in PyTorch, where the ecosystem
  lives. HyperTorch owns the other 90% of a model's life.
- **f32, row-major, naive-but-cache-aware matmul.** Correctness before
  speed; SIMD/blocking lands when a benchmark demands it, not before.
- **Determinism.** Same weights, same inputs, same bits --> across Linux,
  Windows, and macOS.
- **Adaptive-compute ready.** The graph primitives + convergence halting
  exist because fixed compute per input is the industry's expensive
  default; variable-depth models need a runtime designed for them.

## How the benchmark client got fired (a debugging story)

Early runs showed 15 ms p50 latency against a server we'd measured at
81 µs with raw sockets. Hypothesis 1: Nagle's algorithm --> rewrote the
server with `TCP_NODELAY`; no change (but we got zero-dependency HTTP out
of it). Hypothesis 2: our server ignored `Connection: close`, an actual
spec violation --> fixed it; correct, but latency unchanged. The
diagnostic that settled it: phase-isolating timings showed connect at
101 µs, request at 81 µs, meaning the 15 ms lived in the *client* -->
Python's urllib on Windows, which penalized our server ~10× more than
uvicorn. An instrument that biased is not an instrument. The harness now
uses a raw-socket HTTP client, identical for both servers. Moral:
benchmark your benchmark.

## Roadmap

- [ ] safetensors loading
- [ ] blocked/SIMD matmul; optional BLAS feature flag
- [ ] INT8 quantized weights
- [ ] WASM build — inference in the browser
- [ ] conv/attention ops as demand dictates

## License

Dual-licensed under MIT or Apache-2.0, at your option --> the Rust
ecosystem standard. (Add both license texts as `LICENSE-MIT` and
`LICENSE-APACHE` before publishing.)
