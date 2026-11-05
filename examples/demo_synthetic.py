"""Minimal end-to-end example on synthetic data (runs in about a minute on CPU).

Shows the fit / predict interface, the reliability readouts and how a noisy
modality is down-weighted.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rrgef import RRGEF, concordance_index, select_fit_apply  # noqa: E402
from rrgef.corruption import corrupt_noise                    # noqa: E402

rng = np.random.default_rng(0)
torch.manual_seed(0)
n, mods = 300, ["mrna", "methyl", "cna"]
latent = rng.normal(size=n)
X = {m: (np.outer(latent, rng.normal(size=400)) + rng.normal(size=(n, 400))).astype(np.float32)
     for m in mods}
time = np.exp(-0.8 * latent + rng.normal(scale=0.5, size=n)) * 1000
event = (rng.random(n) < 0.6).astype(int)

tr, te = np.arange(240), np.arange(240, n)
Xtr, Xte = select_fit_apply({m: X[m][tr] for m in mods}, {m: X[m][te] for m in mods},
                            time[tr], event[tr], k=64, variance_k=200)

model = RRGEF()
model.fit(Xtr, time[tr], event[tr])
print("clean C-index:", round(concordance_index(time[te], model.predict_risk(Xte), event[te]), 3))

noisy = corrupt_noise(Xte, "methyl", 8.0, np.random.default_rng(1))
out = model.predict_all(noisy)
print("C-index with noisy methylation:",
      round(concordance_index(time[te], out["risk"], event[te]), 3))
print("mean relative reliability per modality:")
for m, r in zip(model.mods_, out["r_m"].mean(axis=1)):
    print(f"  {m:<7} {r:.2f}")
