//! RTW ("HyperTorch Weights") — a deliberately simple binary format for
//! moving tensors from PyTorch to HyperTorch. safetensors compatibility can
//! come later; this is ~80 lines and has zero dependencies.
//!
//! Layout (little-endian):
//!   magic:  6 bytes  b"RTWv1\0"
//!   count:  u32      number of tensors
//!   per tensor:
//!     name_len: u32
//!     name:     name_len bytes (utf-8)
//!     ndim:     u32
//!     dims:     ndim x u64
//!     data:     product(dims) x f32

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const MAGIC: &[u8; 6] = b"RTWv1\0";

pub fn load_rtw<P: AsRef<Path>>(path: P) -> io::Result<HashMap<String, Tensor>> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    parse_rtw(&buf)
}

pub fn parse_rtw(buf: &[u8]) -> io::Result<HashMap<String, Tensor>> {
    let mut pos = 0usize;
    let take = |pos: &mut usize, n: usize| -> io::Result<&[u8]> {
        if *pos + n > buf.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated RTW file"));
        }
        let s = &buf[*pos..*pos + n];
        *pos += n;
        Ok(s)
    };

    if take(&mut pos, 6)? != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad RTW magic"));
    }
    let count = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;

    let mut out = HashMap::with_capacity(count);
    for _ in 0..count {
        let name_len = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
        let name = String::from_utf8(take(&mut pos, name_len)?.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-utf8 tensor name"))?;
        let ndim = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap()) as usize);
        }
        let numel: usize = shape.iter().product();
        let raw = take(&mut pos, numel * 4)?;
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        out.insert(name, Tensor::from_vec(data, &shape));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(tensors: &[(&str, &Tensor)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for (name, t) in tensors {
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for &d in &t.shape {
                buf.extend_from_slice(&(d as u64).to_le_bytes());
            }
            for &v in &t.data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn roundtrip() {
        let w = Tensor::from_vec(vec![1.5, -2.0, 3.25, 0.0], &[2, 2]);
        let b = Tensor::from_vec(vec![0.1, 0.2], &[2]);
        let buf = encode(&[("layer0.weight", &w), ("layer0.bias", &b)]);
        let m = parse_rtw(&buf).unwrap();
        assert_eq!(m["layer0.weight"].data, w.data);
        assert_eq!(m["layer0.weight"].shape, vec![2, 2]);
        assert_eq!(m["layer0.bias"].data, b.data);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(parse_rtw(b"NOTRTW\0\0\0\0").is_err());
    }
}
