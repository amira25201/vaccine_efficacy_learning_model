"""Train the vaccine models and write the metrics out.

    python train.py                 # train from data/vaccine_data_with_all_doses.xlsx
    python train.py --quick         # small search, for a fast check
    python train.py --demo          # generate a synthetic spreadsheet and train on that
    python train.py --data FILE     # train from another spreadsheet

Training is separate from the interface on purpose: the original script
trained at import time, so the window only appeared minutes later and a
missing file produced a traceback instead of a message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config
from make_demo_data import write_demo_spreadsheet
from vaccine_data import check_groups, load_dataset
from vaccine_model import train_target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="spreadsheet to train from")
    parser.add_argument("--quick", action="store_true", help="small hyperparameter search")
    parser.add_argument("--demo", action="store_true", help="generate synthetic data first and train on that")
    args = parser.parse_args()

    if args.demo:
        path = write_demo_spreadsheet(config.ROOT / "data" / "demo_vaccine_data.xlsx")
        print(f"Wrote synthetic demo data to {path}")
        print("These are made-up numbers for checking that the code runs. They are not results.\n")
    else:
        path = args.data or config.DATA_PATH
        if not Path(path).exists():
            print(f"No spreadsheet at {path}.", file=sys.stderr)
            print("See data/README.md, or run 'python train.py --demo' to try the code on synthetic data.", file=sys.stderr)
            return 1

    df = load_dataset(path)
    note = check_groups(df, config.GROUP_COLUMN)
    print(note)
    if note.startswith("WARNING"):
        print()

    summary = {}
    for label, target in config.TARGETS.items():
        if target not in df.columns:
            print(f"[{label}] column {target!r} not in the spreadsheet; skipped.")
            continue
        print(f"[{label}] training on {target} ...")
        bundle = train_target(df, target, quick=args.quick, group_note=note)
        bundle.save(config.model_path(target))
        m = bundle.metrics
        verdict = "better than the baseline" if m["beats_baseline"] else "NO BETTER than predicting the mean"
        print(
            f"[{label}] test RMSE {m['test_rmse']:.3f} vs baseline {m['baseline_rmse']:.3f} ({verdict}); "
            f"R2 {m['test_r2']:.2f}; trained on {m['n_train']} rows / {m['n_groups_train']} groups, "
            f"tested on {m['n_test']} rows / {m['n_groups_test']} groups"
        )
        summary[target] = m

    if not summary:
        print("Nothing was trained.", file=sys.stderr)
        return 1

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "metrics.json"
    out.write_text(json.dumps({"grouping": note, "targets": summary}, indent=2) + "\n")
    print(f"\nMetrics written to {out}")
    print(f"Models written to {config.MODELS_DIR}/ (git-ignored). Run 'python app.py' for the interface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
