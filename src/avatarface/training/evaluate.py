"""Offline evaluation metrics (spec §31) — no extra video data required."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from avatarface.data.dataset import MorphCompletionDataset
from avatarface.models.avatarface_model import AvatarFaceModel
from avatarface.morphs.channel_groups import UPPER_FACE_CHANNELS
from avatarface.training.losses import CompletionLoss


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _onset_f1(pred: np.ndarray, tgt: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    p = pred >= thr
    t = tgt >= 0.5
    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1}


@torch.no_grad()
def evaluate_loader(
    model: AvatarFaceModel,
    loader: DataLoader,
    device: torch.device,
    *,
    autoregressive: bool = False,
) -> Dict[str, Any]:
    model.eval()
    criterion = CompletionLoss()
    upper_names = UPPER_FACE_CHANNELS
    n_u = len(upper_names)

    mae_sum = np.zeros(n_u, np.float64)
    mae_n = 0
    pred_flat: List[np.ndarray] = []
    tgt_flat: List[np.ndarray] = []
    onset_pred: List[np.ndarray] = []
    onset_tgt: List[np.ndarray] = []
    vel_err = 0.0
    vel_n = 0
    loss_sum = 0.0
    loss_n = 0
    freeze_frames = 0
    total_frames = 0

    # Baselines
    hold_mae_sum = 0.0
    neut_mae_sum = 0.0

    t0 = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        bsz, T, _ = batch["current"].shape
        if autoregressive:
            hidden = None
            u_prev = batch["upper_prev"][:, 0]
            u_hats = []
            u_abs_l, u_delta_l, log_var_l = [], [], []
            onset_l, dur_l, amp_l = [], [], []
            for t in range(T):
                out = model.step(
                    batch["current"][:, t],
                    batch["history"][:, t],
                    u_prev,
                    hidden=hidden,
                )
                hidden = out["hidden"]
                u_hats.append(out["u_hat"])
                u_abs_l.append(out["u_abs"])
                u_delta_l.append(out["u_delta"])
                log_var_l.append(out["log_var"])
                onset_l.append(out["blink_onset_logits"])
                dur_l.append(out["blink_duration"])
                amp_l.append(out["blink_amplitude"])
                u_prev = out["u_hat"]
            out = {
                "u_hat": torch.stack(u_hats, dim=1),
                "u_abs": torch.stack(u_abs_l, dim=1),
                "u_delta": torch.stack(u_delta_l, dim=1),
                "log_var": torch.stack(log_var_l, dim=1),
                "blink_onset_logits": torch.stack(onset_l, dim=1),
                "blink_duration": torch.stack(dur_l, dim=1),
                "blink_amplitude": torch.stack(amp_l, dim=1),
            }
        else:
            out = model(batch["current"], batch["history"], batch["upper_prev"])

        loss, _ = criterion(out, batch)
        loss_sum += float(loss) * bsz
        loss_n += bsz

        mask = batch["supervise"].cpu().numpy()  # [B,T]
        u_hat = out["u_hat"].cpu().numpy()
        u = batch["upper"].cpu().numpy()
        u_prev = batch["upper_prev"].cpu().numpy()

        for bi in range(bsz):
            m = mask[bi] > 0.5
            if not m.any():
                continue
            err = np.abs(u_hat[bi, m] - u[bi, m])
            mae_sum += err.sum(axis=0).astype(np.float64)
            mae_n += int(m.sum())
            pred_flat.append(u_hat[bi, m].astype(np.float64))
            tgt_flat.append(u[bi, m].astype(np.float64))

            # hold last GT / neutral baselines
            hold = np.broadcast_to(u_prev[bi, m][:1], u[bi, m].shape)
            hold_mae_sum += float(np.abs(hold - u[bi, m]).mean())
            neut_mae_sum += float(np.abs(u[bi, m]).mean())

            if m.sum() >= 2:
                dv = np.abs(np.diff(u_hat[bi, m], axis=0) - np.diff(u[bi, m], axis=0)).mean()
                vel_err += float(dv)
                vel_n += 1
                frozen = (np.abs(np.diff(u_hat[bi, m], axis=0)).mean(axis=1) < 1e-4).sum()
                freeze_frames += int(frozen)
                total_frames += int(m.sum() - 1)

        onset_logits = out["blink_onset_logits"].cpu().numpy()
        blink_tgt = batch["blink_targets"].cpu().numpy()
        onset_prob = 1.0 / (1.0 + np.exp(-onset_logits[..., 2]))  # bilateral
        for bi in range(bsz):
            m = mask[bi] > 0.5
            if not m.any():
                continue
            onset_pred.append(onset_prob[bi, m])
            onset_tgt.append(blink_tgt[bi, m, 2])

    elapsed = time.perf_counter() - t0
    frames = max(mae_n, 1)
    mae = mae_sum / max(mae_n, 1)
    per_morph = {UPPER_FACE_CHANNELS[i]: float(mae[i]) for i in range(n_u)}

    corr = {}
    if pred_flat:
        P = np.concatenate(pred_flat, axis=0)
        G = np.concatenate(tgt_flat, axis=0)
        for i, name in enumerate(UPPER_FACE_CHANNELS):
            corr[name] = _pearson(P[:, i], G[:, i])

    blink = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if onset_pred:
        blink = _onset_f1(np.concatenate(onset_pred), np.concatenate(onset_tgt))

    groups = {
        "blink": float(np.mean([mae[0], mae[1]])),
        "squint": float(np.mean([mae[2], mae[3]])),
        "wide": float(np.mean([mae[4], mae[5]])),
        "brow_down": float(np.mean([mae[6], mae[7]])),
        "brow_up": float(np.mean([mae[8], mae[9], mae[10]])),
    }

    return {
        "loss": float(loss_sum / max(loss_n, 1)),
        "mae_mean": float(mae.mean()),
        "mae_per_morph": per_morph,
        "mae_groups": {k: float(v) for k, v in groups.items()},
        "pearson_per_morph": {k: float(v) for k, v in corr.items()},
        "pearson_mean": float(np.mean(list(corr.values()))) if corr else 0.0,
        "velocity_mae": float(vel_err / max(vel_n, 1)),
        "freeze_rate": float(freeze_frames / max(total_frames, 1)),
        "blink_onset": {k: float(v) for k, v in blink.items()},
        "baseline_hold_mae": float(hold_mae_sum / max(loss_n, 1)),
        "baseline_neutral_mae": float(neut_mae_sum / max(loss_n, 1)),
        "frames_supervised": int(mae_n),
        "eval_seconds": float(elapsed),
        "approx_fps": float(frames / max(elapsed, 1e-6)),
        "mode": "autoregressive" if autoregressive else "teacher_forced",
    }


def load_model(ckpt: Path, device: torch.device) -> AvatarFaceModel:
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = AvatarFaceModel().to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def evaluate_checkpoint(
    ckpt: Path,
    seq_root: Path,
    splits: Path,
    *,
    split: str = "test",
    batch: int = 8,
    device: Optional[str] = None,
    autoregressive: bool = True,
) -> Dict[str, Any]:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    rows = json.loads(splits.read_text())[split]
    if not rows:
        rows = json.loads(splits.read_text())["val"] or json.loads(splits.read_text())["train"]
    ds = MorphCompletionDataset(seq_root, rows, sequence_length=128, burn_in=32, augment=False)
    loader = DataLoader(ds, batch_size=min(batch, max(1, len(ds))), shuffle=False)
    model = load_model(ckpt, dev)
    metrics = evaluate_loader(model, loader, dev, autoregressive=autoregressive)
    metrics["checkpoint"] = str(ckpt)
    metrics["split"] = split
    metrics["windows"] = len(ds)
    metrics["device"] = device
    return metrics


def latency_benchmark(model: AvatarFaceModel, device: torch.device, warmup: int = 20, steps: int = 200) -> Dict[str, float]:
    model.eval()
    from avatarface.morphs.channel_groups import N_CURRENT, N_HISTORY, N_UPPER

    cur = torch.zeros(1, N_CURRENT, device=device)
    hist = torch.zeros(1, N_HISTORY, device=device)
    u = torch.zeros(1, N_UPPER, device=device)
    hidden = None
    for _ in range(warmup):
        out = model.step(cur, hist, u, hidden=hidden)
        hidden = out["hidden"]
        u = out["u_hat"]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    hidden = None
    u = torch.zeros(1, N_UPPER, device=device)
    for _ in range(steps):
        out = model.step(cur, hist, u, hidden=hidden)
        hidden = out["hidden"]
        u = out["u_hat"]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = 1000.0 * elapsed / steps
    return {"step_ms": ms, "fps": 1000.0 / max(ms, 1e-6), "steps": steps}
