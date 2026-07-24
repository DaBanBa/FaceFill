"""Person-disjoint train/val/test splits (80/10/10). MEAD primary, DISFA supplement."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def collect_sequences(seq_root: Path) -> list[dict]:
    rows = []
    for npz in sorted(seq_root.rglob("*.npz")):
        z = __import__("numpy").load(npz, allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        rows.append(
            {
                "path": str(npz.relative_to(seq_root)),
                "person_id": meta.get("person_id"),
                "dataset": meta.get("dataset"),
                "video_id": meta.get("video_id"),
            }
        )
    return rows


def split_people(people: list[str], seed: int = 0) -> dict[str, list[str]]:
    people = sorted(people)
    rng = random.Random(seed)
    rng.shuffle(people)
    n = len(people)
    n_train = max(1, int(0.8 * n)) if n >= 5 else max(1, n - 2)
    n_val = max(1, int(0.1 * n)) if n >= 5 else (1 if n >= 2 else 0)
    train = people[:n_train]
    val = people[n_train : n_train + n_val]
    test = people[n_train + n_val :]
    if not test and val:
        test = val[-1:]
        val = val[:-1] or val
    return {"train": train, "val": val, "test": test}


def build_splits(seq_root: Path, seed: int = 0) -> dict:
    rows = collect_sequences(seq_root)
    by_ds: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r["person_id"])

    people_split = {}
    for ds, ppl in by_ds.items():
        people_split[ds] = split_people(sorted(set(ppl)), seed=seed)

    out = {"train": [], "val": [], "test": [], "people": people_split}
    for r in rows:
        ds = r["dataset"]
        pid = r["person_id"]
        for split_name, plist in people_split[ds].items():
            if pid in plist:
                out[split_name].append(r)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-root", type=Path, default=Path("data/processed/sequences"))
    ap.add_argument("--out", type=Path, default=Path("data/splits/splits.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    splits = build_splits(args.seq_root, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2))
    print(
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} → {args.out}"
    )


if __name__ == "__main__":
    main()
