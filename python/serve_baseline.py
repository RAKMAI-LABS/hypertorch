"""Baseline for the benchmark: the same MLP served the standard Python way
(FastAPI + PyTorch). Same wire protocol as the Rust server so the benchmark
treats them identically.

Run:  pip install fastapi uvicorn torch numpy
      python serve_baseline.py model.rtw 7879
"""

import struct
import sys
import time

T_START = time.perf_counter()

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, Request, Response

MAGIC = b"RTWv1\x00"
IN_FEATURES = 784


def read_rtw(path):
    tensors = {}
    with open(path, "rb") as f:
        assert f.read(6) == MAGIC, "bad magic"
        (count,) = struct.unpack("<I", f.read(4))
        for _ in range(count):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode()
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
            n = int(np.prod(shape))
            data = np.frombuffer(f.read(4 * n), dtype=np.float32).reshape(shape)
            tensors[name] = torch.from_numpy(data.copy())
    return tensors


class Mlp(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.fc1 = nn.Linear(784, w["fc1.bias"].shape[0])
        self.fc2 = nn.Linear(w["fc1.bias"].shape[0], w["fc2.bias"].shape[0])
        with torch.no_grad():
            # RTW stores [in, out]; nn.Linear wants [out, in]
            self.fc1.weight.copy_(w["fc1.weight"].T)
            self.fc1.bias.copy_(w["fc1.bias"])
            self.fc2.weight.copy_(w["fc2.weight"].T)
            self.fc2.bias.copy_(w["fc2.bias"])

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


model_path = sys.argv[1] if len(sys.argv) > 1 else "model.rtw"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 7879

model = Mlp(read_rtw(model_path)).eval()
app = FastAPI()


@app.get("/health")
def health():
    return Response("ok", media_type="text/plain")


@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    n = len(body) // (IN_FEATURES * 4)
    x = torch.from_numpy(
        np.frombuffer(body, dtype=np.float32).reshape(n, IN_FEATURES).copy()
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(x).argmax(dim=1).tolist()
    micros = int((time.perf_counter() - t0) * 1e6)
    return {"predictions": preds, "micros": micros}


if __name__ == "__main__":
    print(f"READY port={port} startup_micros={int((time.perf_counter() - T_START) * 1e6)}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
