"""Builds a SYNTHETIC features_dataset.csv for smoke-testing train_models.py (not real data).
Group means/stds copy the numbers from the real run; extra columns mimic the real layout."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
G = {0: (7.609, .065, 37.935, .294), 1: (7.634, .044, 39.708, .781), 2: (7.543, .091, 40.017, .895), 3: (7.405, .197, 39.365, 1.366)}
cases = [("Undamaged_1st", 0), ("Undamaged_2nd", 0)] + [(f"LM{d}{l}", d) for d in (1, 2, 3) for l in range(1, 8)]
rows = []
for cid, d in cases:
    m1, s1, m2, s2 = G[d]
    c1, c2 = rng.normal(0, s1 * .7), rng.normal(0, s2 * .7)       # case-level offset
    for r in range(20):
        f1 = m1 + c1 + rng.normal(0, s1 * .7)
        f2 = m2 + c2 + rng.normal(0, s2 * .7)
        f3 = 134 - 2 * d + rng.normal(0, 2.5)
        row = dict(recording_id=f"{cid}_{r}", case_id=cid, source_file=f"{cid}_hit{r}.xlsx", specimen=cid.split("_")[-1],
                   test_type=rng.choice(["Displacement", "Single", "Multi", "Random"]),
                   sampling_rate_hz=333.0 if d == 0 else 303.0,
                   condition="Undamaged" if d == 0 else "Damaged",
                   damage_level_mm=np.nan if d == 0 else float(d),
                   f1_hz=f1, f2_hz=f2, f3_hz=f3, f2_over_f1=f2 / f1, f3_over_f1=f3 / f1)
        for s in range(1, 6):   # amplitude features: healthy session gets a different gain/offset (leak on purpose)
            base = 1.5 if d == 0 else 1.0
            for stat in ("rms", "variance", "skewness", "kurtosis", "peak", "crest_factor", "p2p"):
                row[f"Sensor_{s}_{stat}"] = rng.normal(base * (1 + .1 * s), .2)
        rows.append(row)
df = pd.DataFrame(rows)
df.loc[3, "f3_hz"] = np.nan
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/features_dataset.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("wrote", out, df.shape)