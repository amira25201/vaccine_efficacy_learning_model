"""Generate a synthetic spreadsheet with the expected columns.

The real data is transcribed from published animal studies and is not in this
repository, so this lets anyone run the code end to end. The numbers are
invented. Cell formats deliberately vary (1e6, 1x10^6, 10⁶, "3 µg") so the
parser is exercised the way the real sheet exercises it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VACCINES = ["rfv1", "ev76", "rad5-lcrv", "rad5-yfv", "f1-v"]
STRAINS = ["co92", "kim53", "c12 dyscn", "co92-lux"]
BREEDS = ["balb/c", "cd1", "swiss webster"]
PLAGUE = ["bubonic", "pneumonic"]


def _format_cfu(value: float, style: int) -> str:
    exponent = int(np.log10(value))
    coefficient = value / (10**exponent)
    if style == 0:
        return f"{coefficient:.1f}e{exponent}"
    if style == 1:
        return f"{coefficient:.1f}x10^{exponent}"
    if style == 2:
        superscript = str(exponent).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{coefficient:.1f}×10{superscript}"
    return f"10^{exponent}"  # the bare-power form the old parser read as 106


def write_demo_spreadsheet(path: Path, n_studies: int = 40, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    # Each study contributes several rows, which is what makes grouping matter.
    rows = []
    for study in range(1, n_studies + 1):
        for _ in range(rng.randint(3, 7)):
            vaccine = rng.choice(VACCINES)
            dose = float(np.round(rng.uniform(0.5, 20.0), 1))
            cfu = float(10 ** rng.uniform(4, 8))
            effect = {"rfv1": 0.35, "f1-v": 0.30, "ev76": 0.20, "rad5-lcrv": 0.10, "rad5-yfv": 0.05}[vaccine]
            base = 0.45 + effect - 0.04 * np.log10(cfu / 1e4) + 0.01 * dose + rng.normal(0, 0.08)
            one = float(np.clip(base, 0, 1))
            rows.append(
                {
                    "id": f"study_{study:03d}",
                    "reference": f"Author et al. {2000 + study % 25}",
                    "vaccine": vaccine,
                    "strain": rng.choice(STRAINS),
                    "mice_breed": rng.choice(BREEDS),
                    "plague_ type": rng.choice(PLAGUE),  # stray space, as in the real sheet
                    "CFU": _format_cfu(cfu, rng.randint(0, 4)),
                    "dosage_micrograms": f"{dose} µg" if rng.rand() < 0.5 else dose,
                    "postchallenge_1dose": round(one * 100, 1),
                    "postchallenge_2doses": round(float(np.clip(one + rng.uniform(0.02, 0.15), 0, 1)) * 100, 1),
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path


if __name__ == "__main__":
    import config

    print(write_demo_spreadsheet(config.ROOT / "data" / "demo_vaccine_data.xlsx"))
