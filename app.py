"""Tkinter interface for the vaccine recommender.

Enter a challenge profile and get the vaccines ranked by predicted survival.
The models are loaded from models/; train them first with train.py. Keeping
training out of this file is what stops the window hanging on startup.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import config
from vaccine_model import ModelBundle, rank_vaccines


def load_bundles() -> dict[str, ModelBundle]:
    bundles = {}
    for label, target in config.TARGETS.items():
        path = config.model_path(target)
        if path.exists():
            bundles[label] = ModelBundle.load(path)
    return bundles


def options_from(bundles: dict[str, ModelBundle], column: str) -> list[str]:
    """Offer the category values the models were actually trained on."""
    values: set[str] = set()
    for bundle in bundles.values():
        encoder = bundle.pipeline.named_steps["prep"].named_transformers_.get("cat")
        if encoder is None or column not in bundle.categorical:
            continue
        index = bundle.categorical.index(column)
        values.update(str(v) for v in encoder.named_steps["onehot"].categories_[index])
    return sorted(values)


def build_window(bundles: dict[str, ModelBundle]) -> tk.Tk:
    root = tk.Tk()
    root.title("Vaccine recommender")
    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    variables = {}

    def add_row(row: int, label: str, widget_values=None, default: str = ""):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=default)
        if widget_values:
            ttk.Combobox(frame, textvariable=var, values=widget_values, width=32).grid(row=row, column=1, sticky="we")
        else:
            ttk.Entry(frame, textvariable=var, width=34).grid(row=row, column=1, sticky="we")
        return var

    dose_labels = list(bundles.keys())
    variables["dose"] = add_row(0, "Dose:", dose_labels, dose_labels[0])
    for i, (column, label) in enumerate(
        [("mice_breed", "Mouse strain (breed):"), ("strain", "Y. pestis strain:"), ("plague_type", "Plague type:")], start=1
    ):
        choices = options_from(bundles, column)
        variables[column] = add_row(i, label, choices, choices[0] if choices else "")
    variables["cfu"] = add_row(4, "Challenge dose (CFU), e.g. 1e6:", None, "1e6")
    variables["dosage_micrograms"] = add_row(5, "Vaccine dose (µg):", None, "3.0")

    output = tk.Text(frame, height=14, width=64, wrap="none")
    output.grid(row=7, column=0, columnspan=2, pady=(10, 0), sticky="nsew")
    frame.rowconfigure(7, weight=1)

    def on_recommend():
        try:
            bundle = bundles[variables["dose"].get()]
            profile = {k: v.get() for k, v in variables.items() if k != "dose"}
            ranked = rank_vaccines(bundle, profile)
        except Exception as error:  # surfaced in a dialog rather than a traceback
            messagebox.showerror("Could not produce a recommendation", str(error))
            return

        best = ranked.iloc[0]
        output.delete("1.0", tk.END)
        output.insert(tk.END, f"Best predicted: {best.vaccine}  ({best.predicted_survival:.1%} survival)\n\n")
        output.insert(tk.END, f"{'vaccine':<16}{'predicted survival':>20}\n{'-' * 36}\n")
        for row in ranked.itertuples():
            output.insert(tk.END, f"{row.vaccine:<16}{row.predicted_survival:>19.1%}\n")
        metrics = bundle.metrics
        output.insert(
            tk.END,
            f"\nModel: test RMSE {metrics.get('test_rmse', float('nan')):.3f} "
            f"vs baseline {metrics.get('baseline_rmse', float('nan')):.3f}. "
            f"Predictions are estimates from animal studies, not clinical guidance.\n",
        )

    ttk.Button(frame, text="Recommend", command=on_recommend).grid(row=6, column=0, columnspan=2, pady=8)
    return root


def main() -> int:
    bundles = load_bundles()
    if not bundles:
        print(f"No trained models in {config.MODELS_DIR}/.")
        print("Run 'python train.py' first, or 'python train.py --demo' to try it on synthetic data.")
        return 1
    build_window(bundles).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
