"""Train a tiny MNIST MLP in PyTorch, export weights + a verification
batch in RTW format for HyperTorch.

Run locally:  pip install torch torchvision
              python export_mnist.py

Outputs:
  model.rtw   — network weights (Linear weights pre-transposed to [in, out])
  verify.rtw  — 64 test images + PyTorch's logits for exact comparison
"""

import struct
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

MAGIC = b"RTWv1\x00"


def write_rtw(path, tensors: dict):
    """tensors: name -> float32 numpy array"""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(tensors)))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            name_b = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<Q", d))
            f.write(arr.tobytes())
    print(f"wrote {path}: {list(tensors.keys())}")


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    torch.manual_seed(0)
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000)

    model = Mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x.view(-1, 784)), y)
            loss.backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.view(-1, 784)).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        print(f"epoch {epoch + 1}: test acc {100 * correct / total:.2f}%")

    # --- export weights ---
    # CRITICAL: PyTorch nn.Linear.weight is [out, in]; HyperTorch expects
    # [in, out] so forward is x @ W. Transpose here, once, at export.
    sd = model.state_dict()
    write_rtw("model.rtw", {
        "fc1.weight": sd["fc1.weight"].numpy().T,
        "fc1.bias":   sd["fc1.bias"].numpy(),
        "fc2.weight": sd["fc2.weight"].numpy().T,
        "fc2.bias":   sd["fc2.bias"].numpy(),
    })

    # --- export verification batch: inputs + PyTorch's own logits ---
    x, y = next(iter(test_loader))
    x = x[:64].view(-1, 784)
    with torch.no_grad():
        logits = model(x)
    write_rtw("verify.rtw", {
        "inputs":  x.numpy(),
        "logits":  logits.numpy(),
        "labels":  y[:64].numpy().astype(np.float32),
    })
    print("done — copy model.rtw and verify.rtw next to the hypertorch crate")


if __name__ == "__main__":
    main()
