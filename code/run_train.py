"""Train the early-fusion band-prototype model for one released condition.

with_robotic_arm     : pretrain session2+session3, finetune+test session1
without_robotic_arm  : pretrain session2,           finetune+test session1

Epoch window is 0-4 s. Seed is 42 (global_config).

Example:
  python run_train.py --condition without_robotic_arm --data_root /path/to/GIGA/RawData --subjects sub1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PYTHON = sys.executable
CLASSES = ["reaching", "multigrasp", "twist"]
SHOTS = [1, 5, 25]


def parse_args():
    p = argparse.ArgumentParser(description="Train early-fusion band-prototype models")
    p.add_argument("--condition", required=True,
                   choices=["with_robotic_arm", "without_robotic_arm"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--subjects", default="",
                   help="Comma-separated list. Default: sub1..sub25")
    p.add_argument("--retrain", action="store_true",
                   help="Retrain even if band_prototype_model.pth already exists")
    p.add_argument("--pretrain_epochs", type=int, default=40)
    p.add_argument("--train_episodes", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--tmin", type=float, default=0.0)
    p.add_argument("--tmax", type=float, default=4.0)
    return p.parse_args()


def list_subjects(override):
    if override.strip():
        return [s.strip() for s in override.split(",") if s.strip()]
    return [f"sub{i}" for i in range(1, 26)]


def pretrain_sessions(condition, sub):
    if condition == "with_robotic_arm":
        return f"session2_{sub},session3_{sub}"
    return f"session2_{sub}"


def run_subject(args, sub):
    weight_dir = os.path.join(REPO, "weights", args.condition, sub)
    os.makedirs(weight_dir, exist_ok=True)
    eeg_subject = f"session1_{sub}"
    result_json = os.path.join(weight_dir, "sweep_result.json")
    model_path = os.path.join(weight_dir, "band_prototype_model.pth")
    log_path = os.path.join(weight_dir, "run_log.txt")
    vhdr = os.path.join(args.data_root, f"{eeg_subject}_reaching_MI.vhdr")
    if not os.path.exists(vhdr):
        print(f"[SKIP] {sub}: missing {vhdr}")
        return None

    force = "1" if args.retrain else ("0" if os.path.exists(model_path) else "1")
    env = dict(os.environ)
    env.update({
        "DATA_ROOT": os.path.abspath(args.data_root),
        "CACHE_DIR": os.path.join(REPO, "cache"),
        "EEG_SUBJECT": eeg_subject,
        "SUBJECT_DIR": weight_dir,
        "N_SHOT": "25",
        "FORCE_TRAIN": force,
        "PRETRAIN_EPOCHS": str(args.pretrain_epochs),
        "TRAIN_EPISODES": str(args.train_episodes),
        "BALANCED_SPLIT": "1",
        "PRETRAIN_DATA": "other_session",
        "PRETRAIN_SESSIONS": pretrain_sessions(args.condition, sub),
        "EVAL_SWEEP": "1",
        "SHOT_LIST": "1,5,25",
        "EVAL_REPEATS": str(args.repeats),
        "RESULT_JSON": result_json,
        "MODEL_FILE": "band_prototype_model.pth",
        "TMIN": str(args.tmin),
        "TMAX": str(args.tmax),
        "PYTHONUNBUFFERED": "1",
    })
    print(f"[RUN ] {sub}: target={eeg_subject}  pretrain={env['PRETRAIN_SESSIONS']}  "
          f"train={'yes' if force == '1' else 'no(load)'}", flush=True)

    import subprocess
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            [PYTHON, "-u", os.path.join(HERE, "band_prototype_main.py")],
            env=env, cwd=HERE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            if (line.startswith("[pretrain]") or line.startswith("[finetune]") or
                    line.startswith("[INFO]") or line.startswith("shot") or "±" in line):
                sys.stdout.write(f"  {sub}| {line}")
                sys.stdout.flush()
            logf.write(line)
            logf.flush()
        proc.wait()
    if proc.returncode != 0 or not os.path.exists(result_json):
        print(f"[FAIL] {sub}: returncode={proc.returncode}, see {log_path}")
        return None
    with open(result_json, encoding="utf-8") as f:
        return json.load(f)


def write_excel(path, results):
    acc_rows, std_rows, weight_rows, perclass_rows = [], [], [], []
    for sub, r in results.items():
        acc = {"subject": sub}
        std = {"subject": sub}
        for s in SHOTS:
            sd = r["shots"][str(s)]
            acc[f"{s}-shot"] = round(sd["mean"], 2)
            std[f"{s}-shot"] = round(sd["std"], 2)
        acc_rows.append(acc)
        std_rows.append(std)
        weight_rows.append({
            "subject": sub,
            "mu": round(r["band_weights"]["mu"], 3),
            "hg": round(r["band_weights"]["hg"], 3),
            "temp": round(r["temp"], 2),
        })
        for s in SHOTS:
            pc = r["shots"][str(s)]["per_class"]
            perclass_rows.append({"subject": sub, "shot": s,
                                  **{c: round(pc[c], 1) for c in CLASSES}})
    acc_df = pd.DataFrame(acc_rows)
    summary_rows = []
    for s in SHOTS:
        col = f"{s}-shot"
        vals = acc_df[col].values.astype(float)
        summary_rows.append({
            "shot": f"{s}-shot", "subjects": len(vals),
            "mean_acc": round(float(np.mean(vals)), 2),
            "std_across_subjects": round(float(np.std(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2),
        })
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name="Summary", index=False)
        acc_df.to_excel(xw, sheet_name="Accuracy_by_shot", index=False)
        pd.DataFrame(std_rows).to_excel(xw, sheet_name="Std_by_shot", index=False)
        pd.DataFrame(perclass_rows).to_excel(xw, sheet_name="Per_class", index=False)
        pd.DataFrame(weight_rows).to_excel(xw, sheet_name="Band_weights", index=False)
    print(f"[XLSX] wrote {path}")


def main():
    args = parse_args()
    args.data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(args.data_root):
        raise SystemExit(f"DATA_ROOT not found: {args.data_root}")

    subjects = list_subjects(args.subjects)
    print(f"[INFO] condition={args.condition}")
    print(f"[INFO] pretrain={pretrain_sessions(args.condition, 'subX')}  "
          f"finetune+test=session1  window=[{args.tmin},{args.tmax}]s")
    results = {}
    for i, sub in enumerate(subjects, 1):
        print(f"\n===== [{i}/{len(subjects)}] {sub} =====")
        r = run_subject(args, sub)
        if r is None:
            continue
        results[sub] = r
        s25 = r["shots"]["25"]
        print(f"[DONE] {sub}: 25-shot={s25['mean']:.2f}±{s25['std']:.2f}%  "
              f"weights[mu={r['band_weights']['mu']:.3f}, hg={r['band_weights']['hg']:.3f}]")
        write_excel(os.path.join(REPO, "results", f"{args.condition}.xlsx"), results)
    print(f"\n[ALL DONE] {len(results)}/{len(subjects)} -> results/{args.condition}.xlsx")


if __name__ == "__main__":
    main()
