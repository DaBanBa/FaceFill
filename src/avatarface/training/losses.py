"""Training losses (spec §21–22)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_l1(pred, target, mask=None):
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if mask is not None:
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0) * loss.shape[-1]
        return loss.sum() / denom
    return loss.mean()


def blink_onset_loss(logits, targets, mask=None):
    # weighted BCE for sparse onsets
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # upweight positives
    weights = 1.0 + 9.0 * targets
    loss = loss * weights
    if mask is not None:
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0) / loss.shape[-1]
    return loss.mean()


def uncertainty_nll(log_var, mu, target, mask=None):
    # treat u_abs as mu
    loss = 0.5 * ((target - mu) ** 2) * torch.exp(-log_var) + 0.5 * log_var
    if mask is not None:
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0) / loss.shape[-1]
    return loss.mean()


def range_loss(u_hat, mask=None):
    loss = F.relu(-u_hat) + F.relu(u_hat - 1.0)
    if mask is not None:
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0) / max(loss.shape[-1], 1)
    return loss.mean()


class CompletionLoss:
    def __init__(
        self,
        w_abs=1.0,
        w_res=1.0,
        w_vel=0.5,
        w_acc=0.1,
        w_blink=1.0,
        w_range=0.05,
        w_unc=0.25,
    ):
        self.w = dict(
            abs=w_abs, res=w_res, vel=w_vel, acc=w_acc, blink=w_blink, range=w_range, unc=w_unc
        )

    def __call__(self, outputs, batch):
        mask = batch["supervise"]  # [B, T]
        u = batch["upper"]
        u_prev = batch["upper_prev"]
        u_abs = outputs["u_abs"]
        u_delta = outputs["u_delta"]
        u_hat = outputs["u_hat"]

        L_abs = smooth_l1(u_abs, u, mask)
        L_res = smooth_l1(u_delta, u - u_prev, mask)
        L_vel = smooth_l1(u_hat[:, 1:] - u_hat[:, :-1], u[:, 1:] - u[:, :-1], mask[:, 1:])
        # acceleration
        if u_hat.shape[1] >= 3:
            pred_a = (u_hat[:, 2:] - u_hat[:, 1:-1]) - (u_hat[:, 1:-1] - u_hat[:, :-2])
            tgt_a = (u[:, 2:] - u[:, 1:-1]) - (u[:, 1:-1] - u[:, :-2])
            L_acc = smooth_l1(pred_a, tgt_a, mask[:, 2:])
        else:
            L_acc = u_hat.new_zeros(())

        onset_logits = outputs["blink_onset_logits"]
        blink_tgt = batch["blink_targets"]
        L_onset = blink_onset_loss(onset_logits, blink_tgt[..., 0:3], mask)
        L_dur = smooth_l1(outputs["blink_duration"], blink_tgt[..., 3], mask)
        L_amp = smooth_l1(outputs["blink_amplitude"], blink_tgt[..., 4], mask)
        L_blink = L_onset + L_dur + L_amp

        L_range = range_loss(u_hat, mask)
        L_unc = uncertainty_nll(outputs["log_var"], u_abs, u, mask)

        total = (
            self.w["abs"] * L_abs
            + self.w["res"] * L_res
            + self.w["vel"] * L_vel
            + self.w["acc"] * L_acc
            + self.w["blink"] * L_blink
            + self.w["range"] * L_range
            + self.w["unc"] * L_unc
        )
        return total, {
            "abs": float(L_abs.detach()),
            "res": float(L_res.detach()),
            "vel": float(L_vel.detach()),
            "acc": float(L_acc.detach()),
            "blink": float(L_blink.detach()),
            "range": float(L_range.detach()),
            "unc": float(L_unc.detach()),
            "total": float(total.detach()),
        }
