//! HyperTorch model server — TRUE zero-dependency edition.
//!
//! Hand-rolled HTTP/1.1 on std::net. We dropped tiny_http after benchmarks
//! on Windows loopback showed ~15ms p50: Nagle's algorithm + delayed ACK
//! stalling small request/response exchanges. tiny_http doesn't expose
//! TCP_NODELAY; std::net does. ~130 lines buys us the socket flag, and the
//! whole project (library AND examples) is now dependency-free.
//!
//! Endpoints:
//!   GET  /health   -> "ok"
//!   POST /predict  -> body: N*784 little-endian f32, response JSON
//!
//! Run: cargo run --release --example serve -- model.rtw 7878

use hypertorch::nn::{Linear, Mlp};
use hypertorch::weights::load_rtw;
use hypertorch::Tensor;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Instant;

const IN_FEATURES: usize = 784;

fn main() {
    let t_start = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or("model.rtw");
    let port: u16 = args.get(2).and_then(|p| p.parse().ok()).unwrap_or(7878);

    let mut w = load_rtw(model_path).expect("failed to load model");
    let mlp = Arc::new(Mlp {
        layers: vec![
            Linear::from_weights(w.remove("fc1.weight").unwrap(), w.remove("fc1.bias").unwrap()),
            Linear::from_weights(w.remove("fc2.weight").unwrap(), w.remove("fc2.bias").unwrap()),
        ],
    });

    let listener = TcpListener::bind(("0.0.0.0", port)).expect("bind failed");
    println!("READY port={} startup_micros={}", port, t_start.elapsed().as_micros());

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let mlp = Arc::clone(&mlp);
        std::thread::spawn(move || {
            let _ = handle(stream, &mlp);
        });
    }
}

fn handle(mut stream: TcpStream, mlp: &Mlp) -> std::io::Result<()> {
    // THE FIX: disable Nagle so small responses flush immediately.
    stream.set_nodelay(true)?;

    // A client may send several requests on one connection (keep-alive).
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    loop {
        // read until we have full headers
        let header_end = loop {
            if let Some(pos) = find_crlfcrlf(&buf) {
                break pos;
            }
            let n = stream.read(&mut chunk)?;
            if n == 0 {
                return Ok(()); // client closed
            }
            buf.extend_from_slice(&chunk[..n]);
        };

        let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let mut lines = head.lines();
        let request_line = lines.next().unwrap_or("");
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();

        let mut content_length = 0usize;
        let mut want_close = false;
        for line in lines {
            if let Some((k, v)) = line.split_once(':') {
                if k.trim().eq_ignore_ascii_case("content-length") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
                if k.trim().eq_ignore_ascii_case("connection")
                    && v.trim().eq_ignore_ascii_case("close")
                {
                    want_close = true;
                }
            }
        }

        // read body
        let body_start = header_end + 4;
        while buf.len() < body_start + content_length {
            let n = stream.read(&mut chunk)?;
            if n == 0 {
                return Ok(());
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        let body = buf[body_start..body_start + content_length].to_vec();
        buf.drain(..body_start + content_length); // leave any pipelined bytes

        match (method.as_str(), path.as_str()) {
            ("GET", "/health") => respond(&mut stream, 200, "text/plain", b"ok", !want_close)?,
            ("POST", "/predict") => {
                if body.is_empty() || body.len() % (IN_FEATURES * 4) != 0 {
                    let msg = format!("body must be N*{}*4 bytes of f32, got {}", IN_FEATURES, body.len());
                    respond(&mut stream, 400, "text/plain", msg.as_bytes(), !want_close)?;
                    continue;
                }
                let n = body.len() / (IN_FEATURES * 4);
                let data: Vec<f32> = body
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let x = Tensor::from_vec(data, &[n, IN_FEATURES]);

                let t0 = Instant::now();
                let preds = mlp.predict(&x);
                let micros = t0.elapsed().as_micros();

                let preds_json: Vec<String> = preds.iter().map(|p| p.to_string()).collect();
                let json = format!("{{\"predictions\":[{}],\"micros\":{}}}", preds_json.join(","), micros);
                respond(&mut stream, 200, "application/json", json.as_bytes(), !want_close)?;
            }
            _ => respond(&mut stream, 404, "text/plain", b"not found", !want_close)?,
        }
        if want_close {
            return Ok(()); // HTTP spec: client asked us to close — do it
        }
    }
}

fn respond(stream: &mut TcpStream, status: u16, ctype: &str, body: &[u8], keep_alive: bool) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Not Found",
    };
    // single write: headers + body in one buffer, one TCP segment where possible
    let conn = if keep_alive { "keep-alive" } else { "close" };
    let mut out = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
        status, reason, ctype, body.len(), conn
    )
    .into_bytes();
    out.extend_from_slice(body);
    stream.write_all(&out)
}

fn find_crlfcrlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}
