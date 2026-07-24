"""Export streaming AvatarFace step to ONNX with GRU hidden I/O (spec §35)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from avatarface.models.avatarface_model import AvatarFaceModel
from avatarface.morphs.channel_groups import N_CURRENT, N_HISTORY, N_UPPER


class StreamingStepONNX(nn.Module):
    """Single-frame step with explicit hidden state in/out."""

    def __init__(self, model: AvatarFaceModel):
        super().__init__()
        self.model = model
        self.num_layers = model.core.num_layers
        self.hidden_dim = model.core.hidden_dim

    def forward(self, current, history, u_prev, hidden):
        # current/history/u_prev: [B, D]; hidden: [L, B, H]
        out = self.model.step(current, history, u_prev, hidden=hidden)
        return (
            out["u_hat"],
            out["u_abs"],
            out["u_delta"],
            out["blink_onset_logits"],
            out["blink_duration"],
            out["blink_amplitude"],
            out["hidden"],
        )


def export_onnx(ckpt: Path, out: Path) -> Path:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = AvatarFaceModel()
    model.load_state_dict(blob["model"])
    model.eval()

    wrap = StreamingStepONNX(model)
    B = 1
    dummy_c = torch.zeros(B, N_CURRENT)
    dummy_h = torch.zeros(B, N_HISTORY)
    dummy_u = torch.zeros(B, N_UPPER)
    dummy_hid = torch.zeros(wrap.num_layers, B, wrap.hidden_dim)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap,
        (dummy_c, dummy_h, dummy_u, dummy_hid),
        out,
        input_names=["current", "history", "u_prev", "hidden_in"],
        output_names=[
            "u_hat",
            "u_abs",
            "u_delta",
            "blink_onset_logits",
            "blink_duration",
            "blink_amplitude",
            "hidden_out",
        ],
        opset_version=17,
        dynamic_axes={
            "current": {0: "batch"},
            "history": {0: "batch"},
            "u_prev": {0: "batch"},
            "hidden_in": {1: "batch"},
            "u_hat": {0: "batch"},
            "u_abs": {0: "batch"},
            "u_delta": {0: "batch"},
            "blink_onset_logits": {0: "batch"},
            "blink_duration": {0: "batch"},
            "blink_amplitude": {0: "batch"},
            "hidden_out": {1: "batch"},
        },
    )
    return out


def latency_torch(ckpt: Path, device: str = "cpu", steps: int = 300) -> dict:
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = AvatarFaceModel().to(device).eval()
    model.load_state_dict(blob["model"])
    cur = torch.zeros(1, N_CURRENT, device=device)
    hist = torch.zeros(1, N_HISTORY, device=device)
    u = torch.zeros(1, N_UPPER, device=device)
    hidden = None
    for _ in range(20):
        out = model.step(cur, hist, u, hidden=hidden)
        hidden = out["hidden"]
        u = out["u_hat"]
    t0 = time.perf_counter()
    hidden = None
    u = torch.zeros(1, N_UPPER, device=device)
    for _ in range(steps):
        out = model.step(cur, hist, u, hidden=hidden)
        hidden = out["hidden"]
        u = out["u_hat"]
    ms = 1000.0 * (time.perf_counter() - t0) / steps
    return {"step_ms": ms, "fps": 1000.0 / max(ms, 1e-6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/model/model.onnx"))
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    path = export_onnx(args.ckpt, args.out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    if args.bench:
        print("torch latency", latency_torch(args.ckpt))


if __name__ == "__main__":
    main()
