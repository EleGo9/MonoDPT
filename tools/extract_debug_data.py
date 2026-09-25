#!/usr/bin/env python

import argparse
import json
from pathlib import Path


DEBUG_KEYS = [
    "_step",
    "epoch",
    "global_step",
    "debug/optimizer_steps",
    "debug/param_delta_l2",
    "debug/grad_norm",
    "debug/amp_scale",
    "debug/optimizer_step_skipped",
    "debug/eval_model_checksum_before",
    "debug/eval_model_checksum_after",
    "debug/prediction_checksum",
    "2D_mAP",
    "BEV_mAP",
    "3D_mAP",
    "depth_MAE_mean_all",
    "depth_MAE_median_all",
    "depth_MAE_percent_mean_all",
    "depth_MAE_percent_median_all",
    "rotation_error_radians",
    "rotation_error_degrees",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the W&B run directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output JSONL file",
    )
    args = parser.parse_args()

    run_dir = args.run_dir

    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    print(f"Run directory: {run_dir}")

    # Look for W&B history files
    candidates = list(run_dir.rglob("*.jsonl"))

    if not candidates:
        print("No .jsonl files found.")
        print("Files in run directory:")
        for path in run_dir.rglob("*"):
            if path.is_file():
                print(f"  {path}")
        return

    print("\nJSONL files found:")
    for path in candidates:
        print(f"  {path}")

    # Try all JSONL files and extract records containing relevant keys.
    records = []

    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    selected = {
                        key: record[key]
                        for key in DEBUG_KEYS
                        if key in record
                    }

                    if selected:
                        records.append(selected)

        except Exception as e:
            print(f"Could not read {path}: {e}")

    if not records:
        print("\nNo matching debug records found.")
        return

    print(f"\nFound {len(records)} matching records.")

    # Print compact output
    for record in records:
        print(json.dumps(record, sort_keys=False))

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()