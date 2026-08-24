"""Load released weights and reproduce the 100-repeat few-shot evaluation.

Requires GIGA session1 files only (finetune/test session).
Pretraining sessions are not needed because the encoder is already in the .pth.

Example:
  python run_infer.py --condition without_robotic_arm --data_root /path/to/GIGA/RawData
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce few-shot results from released weights")
    p.add_argument("--condition", required=True,
                   choices=["with_robotic_arm", "without_robotic_arm"])
    p.add_argument("--data_root", required=True,
                   help="Folder that contains session1_subX_reaching_MI.vhdr etc.")
    p.add_argument("--subjects", default="",
                   help="Comma-separated list, e.g. sub1,sub2. Default: all sub1-sub25")
    p.add_argument("--tmin", type=float, default=0.0)
    p.add_argument("--tmax", type=float, default=4.0)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--shots", default="1,5,25")
    p.add_argument("--atol", type=float, default=0.05,
                   help="Allowed absolute difference (percentage points) vs sweep_result.json")
    return p.parse_args()


def list_subjects(condition, override):
    if override.strip():
        return [s.strip() for s in override.split(",") if s.strip()]
    root = os.path.join(REPO, "weights", condition)
    subs = [n for n in os.listdir(root) if n.startswith("sub") and os.path.isdir(os.path.join(root, n))]
    subs.sort(key=lambda s: int(s[3:]))
    return subs


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        raise SystemExit(f"DATA_ROOT not found: {data_root}")

    os.environ["DATA_ROOT"] = data_root
    os.environ["TMIN"] = str(args.tmin)
    os.environ["TMAX"] = str(args.tmax)
    os.environ["EEG_SUBJECT"] = "session1_sub1"
    os.environ["CACHE_DIR"] = os.path.join(REPO, "cache")

    sys.path.insert(0, HERE)
    import numpy as np
    import torch
    from torch.utils.data import Subset
    from band_prototype_main import (
        BAND_NAMES,
        BandEEGDataset,
        BandPrototypeNet,
        NUM_CLASSES,
        SEED,
        device,
        evaluate_fewshot_repeated,
        make_balanced_split,
    )

    shots = [int(s) for s in args.shots.split(",")]
    subjects = list_subjects(args.condition, args.subjects)
    print(f"[INFO] condition={args.condition}  device={device}  subjects={subjects}")
    print(f"[INFO] window=[{args.tmin}, {args.tmax}]s  repeats={args.repeats}  seed={SEED}")

    n_ok, n_fail, n_skip = 0, 0, 0
    for sub in subjects:
        weight_dir = os.path.join(REPO, "weights", args.condition, sub)
        model_path = os.path.join(weight_dir, "band_prototype_model.pth")
        expected_path = os.path.join(weight_dir, "sweep_result.json")
        vhdr = os.path.join(data_root, f"session1_{sub}_reaching_MI.vhdr")
        if not os.path.exists(model_path):
            print(f"[SKIP] {sub}: missing {model_path}")
            n_skip += 1
            continue
        if not os.path.exists(vhdr):
            print(f"[SKIP] {sub}: missing {vhdr}")
            n_skip += 1
            continue

        dataset = BandEEGDataset(subject_name=f"session1_{sub}", tmin=args.tmin, tmax=args.tmax)
        labels = np.asarray(dataset.label_list)
        train_indices, _, _, query_indices, _, _ = make_balanced_split(
            labels, seed=SEED, test_ratio=0.2, n_shot=25)
        train_ds = Subset(dataset, train_indices)
        query_ds = Subset(dataset, query_indices)

        model = BandPrototypeNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        w = model.band_weights().detach().cpu().numpy()
        wtxt = ", ".join(f"{n}={v:.3f}" for n, v in zip(BAND_NAMES, w))
        print(f"\n===== {sub} =====")
        print(f"[INFO] Loaded {model_path}")
        print(f"[INFO] band weights [{wtxt}]  temp={model.logit_scale.item():.2f}")

        res = evaluate_fewshot_repeated(model, train_ds, query_ds, shots, args.repeats, seed=SEED)
        print(f"{'shot':>5} | {'balanced acc (mean±std)':>24} | reaching | multigrasp | twist")
        print("-" * 78)
        for s in shots:
            m, sd, pc = res[s]
            print(f"{s:>5} | {m:>16.2f} ± {sd:5.2f}% | {pc[0]:7.1f}% | {pc[1]:9.1f}% | {pc[2]:6.1f}%")

        if not os.path.exists(expected_path):
            print(f"[WARN] {sub}: no sweep_result.json to compare")
            continue

        with open(expected_path, encoding="utf-8") as f:
            expected = json.load(f)
        mismatches = []
        for s in shots:
            got = res[s][0]
            exp = expected["shots"][str(s)]["mean"]
            if abs(got - exp) > args.atol:
                mismatches.append(f"{s}-shot got {got:.4f} expected {exp:.4f}")
        if mismatches:
            print(f"[MISMATCH] {sub}: " + "; ".join(mismatches))
            n_fail += 1
        else:
            print(f"[MATCH] {sub}: within ±{args.atol:.2f}%p of sweep_result.json")
            n_ok += 1

    print(f"\n[SUMMARY] match={n_ok}  mismatch={n_fail}  skip={n_skip}  / {len(subjects)}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
