"""GRU trainer — Stage 1 teacher-forced + Stage 2 scheduled sampling (+ short AR rollout)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from avatarface.data.dataset import MorphCompletionDataset
from avatarface.data.splits import build_splits
from avatarface.models.avatarface_model import AvatarFaceModel
from avatarface.morphs.arkit_schema import ARKIT_INDEX, N_FULL
from avatarface.morphs.mediapipe_mapping import lower_from_full
from avatarface.training.losses import CompletionLoss
from avatarface.training.scheduled_sampling import teacher_forcing_prob


def load_split(splits_path: Path, name: str) -> list[dict]:
    return json.loads(splits_path.read_text())[name]


def write_synthetic_sequence(path: Path, person_id: str, emotion: str, angry_brows: bool = False):
    t = 256
    full = np.zeros((t, N_FULL), np.float32)
    for i in range(t):
        tt = i / 30.0
        smile = 0.3 + 0.4 * max(0.0, math.sin(tt * 1.1))
        full[i, ARKIT_INDEX["jawOpen"]] = 0.15 + 0.25 * abs(math.sin(tt * 8))
        full[i, ARKIT_INDEX["mouthSmileLeft"]] = smile
        full[i, ARKIT_INDEX["mouthSmileRight"]] = smile
        full[i, ARKIT_INDEX["browInnerUp"]] = 0.15 * smile
        blink = 1.0 if i % 40 in (10, 11) else 0.0
        full[i, ARKIT_INDEX["eyeBlinkLeft"]] = blink
        full[i, ARKIT_INDEX["eyeBlinkRight"]] = blink
        full[i, ARKIT_INDEX["eyeSquintLeft"]] = smile * 0.3
        full[i, ARKIT_INDEX["eyeSquintRight"]] = smile * 0.3
        if angry_brows:
            full[i, ARKIT_INDEX["browDownLeft"]] = 0.45
            full[i, ARKIT_INDEX["browDownRight"]] = 0.45

    meta = {
        "video_id": f"{person_id}/front/{emotion}/level_1/001",
        "track_id": "0",
        "person_id": person_id,
        "dataset": "mead",
        "camera": "front",
        "emotion": emotion,
        "intensity": "level_1",
        "fps": 30.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        full_morphs=full,
        lower_morphs=lower_from_full(full),
        head_features=np.zeros((t, 12), np.float32),
        tracking_confidence=np.ones((t, 3), np.float32),
        valid_mask=np.ones((t, N_FULL), bool),
        observed_mask=np.ones((t, N_FULL), bool),
        timestamps=np.arange(t, dtype=np.float64) / 30.0,
        meta_json=np.array(json.dumps(meta)),
    )


def ensure_synthetic(seq_root: Path, splits_path: Path):
    write_synthetic_sequence(seq_root / "mead" / "SYNTH__front__happy__level_1__001.npz", "SYNTH", "happy")
    write_synthetic_sequence(
        seq_root / "mead" / "SYNTH2__front__angry__level_1__001.npz", "SYNTH2", "angry", angry_brows=True
    )
    splits = build_splits(seq_root, seed=0)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, indent=2))


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def forward_teacher(model, batch):
    return model(batch["current"], batch["history"], batch["upper_prev"])


def forward_scheduled(model, batch, p_tf: float):
    """Per-frame AR with scheduled sampling on u_prev (spec §20.2)."""
    current = batch["current"]
    history = batch["history"]
    upper = batch["upper"]
    bsz, T, _ = current.shape
    device = current.device

    hidden = None
    u_prev = batch["upper_prev"][:, 0]
    u_hats, u_abs_l, u_delta_l, log_var_l = [], [], [], []
    onset_l, dur_l, amp_l = [], [], []

    for t in range(T):
        out = model.step(current[:, t], history[:, t], u_prev, hidden=hidden)
        hidden = out["hidden"]
        u_hats.append(out["u_hat"])
        u_abs_l.append(out["u_abs"])
        u_delta_l.append(out["u_delta"])
        log_var_l.append(out["log_var"])
        onset_l.append(out["blink_onset_logits"])
        dur_l.append(out["blink_duration"])
        amp_l.append(out["blink_amplitude"])

        use_tf = (torch.rand(bsz, device=device) < p_tf).unsqueeze(-1)
        u_prev = torch.where(use_tf, upper[:, t], out["u_hat"].detach())

    return {
        "u_hat": torch.stack(u_hats, dim=1),
        "u_abs": torch.stack(u_abs_l, dim=1),
        "u_delta": torch.stack(u_delta_l, dim=1),
        "log_var": torch.stack(log_var_l, dim=1),
        "blink_onset_logits": torch.stack(onset_l, dim=1),
        "blink_duration": torch.stack(dur_l, dim=1),
        "blink_amplitude": torch.stack(amp_l, dim=1),
    }


def forward_rollout(model, batch, horizon: int):
    """Pure AR over first `horizon` frames (spec §20.3 lite)."""
    current = batch["current"]
    history = batch["history"]
    bsz, T, _ = current.shape
    H = min(horizon, T)
    hidden = None
    u_prev = batch["upper_prev"][:, 0]
    u_hats, u_abs_l, u_delta_l, log_var_l = [], [], [], []
    onset_l, dur_l, amp_l = [], [], []
    for t in range(H):
        out = model.step(current[:, t], history[:, t], u_prev, hidden=hidden)
        hidden = out["hidden"]
        u_hats.append(out["u_hat"])
        u_abs_l.append(out["u_abs"])
        u_delta_l.append(out["u_delta"])
        log_var_l.append(out["log_var"])
        onset_l.append(out["blink_onset_logits"])
        dur_l.append(out["blink_duration"])
        amp_l.append(out["blink_amplitude"])
        u_prev = out["u_hat"]
    return {
        "u_hat": torch.stack(u_hats, dim=1),
        "u_abs": torch.stack(u_abs_l, dim=1),
        "u_delta": torch.stack(u_delta_l, dim=1),
        "log_var": torch.stack(log_var_l, dim=1),
        "blink_onset_logits": torch.stack(onset_l, dim=1),
        "blink_duration": torch.stack(dur_l, dim=1),
        "blink_amplitude": torch.stack(amp_l, dim=1),
    }, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-root", type=Path, default=Path("data/processed/sequences"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits/splits.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/model"))
    ap.add_argument("--init", type=Path, default=None, help="Warm-start checkpoint (e.g. stage1 best.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--burn-in", type=int, default=32)
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--synthetic-smoke", action="store_true")
    ap.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2],
        help="1=teacher-forced, 2=scheduled sampling (+ optional rollout)",
    )
    ap.add_argument("--rollout-horizon", type=int, default=60, help="Stage-2 AR rollout length (0=off)")
    ap.add_argument("--rollout-weight", type=float, default=0.5)
    args = ap.parse_args()

    if args.synthetic_smoke:
        ensure_synthetic(args.seq_root, args.splits)

    if not args.splits.exists():
        raise SystemExit(f"Missing {args.splits}. Run: python -m avatarface.data.splits")

    train_rows = load_split(args.splits, "train")
    val_rows = load_split(args.splits, "val")
    train_ds = MorphCompletionDataset(args.seq_root, train_rows, args.seq_len, args.burn_in, augment=True)
    val_ds = MorphCompletionDataset(
        args.seq_root, val_rows if val_rows else train_rows, args.seq_len, args.burn_in, augment=False
    )
    if len(train_ds) == 0:
        raise SystemExit("No training windows — extract MEAD first or pass --synthetic-smoke")

    train_loader = DataLoader(train_ds, batch_size=min(args.batch, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(args.batch, max(1, len(val_ds))), shuffle=False)

    device = torch.device(args.device)
    model = AvatarFaceModel().to(device)
    if args.init and args.init.exists():
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"warm-start from {args.init}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    criterion = CompletionLoss()
    args.out.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"params={nparams:,}  train_windows={len(train_ds)}  val_windows={len(val_ds)}  "
        f"stage={args.stage}  device={device}"
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        p_tf = 1.0 if args.stage == 1 else teacher_forcing_prob(epoch)
        model.train()
        tr = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train tf={p_tf:.2f}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            if args.stage == 1:
                out = forward_teacher(model, batch)
                loss, _ = criterion(out, batch)
            else:
                out = forward_scheduled(model, batch, p_tf)
                loss, _ = criterion(out, batch)
                if args.rollout_horizon > 0 and args.rollout_weight > 0:
                    roll_out, H = forward_rollout(model, batch, args.rollout_horizon)
                    roll_batch = {
                        "upper": batch["upper"][:, :H],
                        "upper_prev": batch["upper_prev"][:, :H],
                        "blink_targets": batch["blink_targets"][:, :H],
                        "supervise": batch["supervise"][:, :H],
                    }
                    roll_loss, _ = criterion(roll_out, roll_batch)
                    loss = loss + args.rollout_weight * roll_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss.detach()) * batch["current"].shape[0]
            n += batch["current"].shape[0]
        tr /= max(n, 1)

        model.eval()
        va = 0.0
        nv = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # Val always AR when stage>=2 (matches deployment)
                if args.stage == 1:
                    out = forward_teacher(model, batch)
                else:
                    out, _ = forward_rollout(model, batch, batch["current"].shape[1])
                loss, _ = criterion(out, batch)
                va += float(loss) * batch["current"].shape[0]
                nv += batch["current"].shape[0]
        va /= max(nv, 1)
        history.append({"epoch": epoch, "train": tr, "val": va, "p_tf": p_tf})
        print(f"epoch {epoch:03d}  train {tr:.4f}  val {va:.4f}  p_tf {p_tf:.2f}")

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val": va,
            "nparams": nparams,
            "stage": args.stage,
            "p_tf": p_tf,
        }
        torch.save(ckpt, args.out / "last.pt")
        if va < best:
            best = va
            torch.save(ckpt, args.out / "best.pt")
            print(f"  saved best.pt ({best:.4f})")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done → {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
