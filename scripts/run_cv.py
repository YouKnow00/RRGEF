"""Reproduce the RRGEF evaluation protocol on one or more TCGA cohorts.

Protocol (as in the paper): five outer folds x three seeds (0, 1, 2); fold-wise
variance cap (1024) and univariate Cox selection (256) per modality; test-time
scenarios clean, Gaussian noise (sigma = 1, 2, 4, 8), wipeout and shuffle, each
applied to one modality at a time and averaged over modalities.

Usage:
    python scripts/download_data.py
    python scripts/run_cv.py                       # all cohorts
    python scripts/run_cv.py TCGA-LUAD --seeds=0   # quick run
Results are written to results/<cohort>.json.
"""
import json
import os
import sys
import time as _t

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rrgef import RRGEF                                         # noqa: E402
from rrgef.corruption import (SCENES, SIGMAS, corrupt_noise,     # noqa: E402
                              corrupt_shuffle, corrupt_wipeout, stable_seed)
from rrgef.data import load_cohort                              # noqa: E402
from rrgef.metrics import concordance_index                     # noqa: E402
from rrgef.preprocessing import SELECT_K, VARIANCE_K, select_fit_apply  # noqa: E402
from rrgef.splits import outer_splits                           # noqa: E402

COHORTS = ["TCGA-BRCA", "TCGA-KIRC", "TCGA-LIHC", "TCGA-LUAD"]


def run_cohort(cohort, seeds=(0, 1, 2)):
    t0 = _t.time()
    X_all, time_all, event_all, mods = load_cohort(cohort)
    fold_c = {s: [] for s in SCENES}
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        for fi, (tr, te) in enumerate(outer_splits(time_all, event_all, seed, 5)):
            ttr, tte = time_all[tr], time_all[te]
            etr, ete = event_all[tr], event_all[te]
            Xtr, Xte = select_fit_apply(
                {m: X_all[m][tr] for m in mods}, {m: X_all[m][te] for m in mods},
                ttr, etr, SELECT_K, preselect="raw_fold", variance_k=VARIANCE_K)
            # Seeding identical to the paper runs.
            key = "ours_rel" if len(tr) < 250 else "ours_rel_cross"
            method_seed = stable_seed("fit", seed, fi, key)
            np.random.seed(method_seed)
            torch.manual_seed(method_seed)
            model = RRGEF()
            model.fit(Xtr, ttr, etr)

            def cidx(Xv):
                return concordance_index(tte, model.predict_risk(Xv), ete)

            fold_c["clean"].append(cidx(Xte))
            for sg in SIGMAS:
                vals = []
                for which in mods:
                    rng = np.random.default_rng(
                        stable_seed("corrupt", seed, fi, f"noise{sg:g}", which))
                    vals.append(cidx(corrupt_noise(Xte, which, sg, rng)))
                fold_c[f"noise{sg:g}"].append(float(np.nanmean(vals)))
            fold_c["wipeout"].append(float(np.nanmean(
                [cidx(corrupt_wipeout(Xte, w)) for w in mods])))
            vals = []
            for which in mods:
                rng = np.random.default_rng(
                    stable_seed("corrupt", seed, fi, "shuffle", which))
                vals.append(cidx(corrupt_shuffle(Xte, which, rng)))
            fold_c["shuffle"].append(float(np.nanmean(vals)))
            print(f"  {cohort} seed {seed} fold {fi}: clean C-index {fold_c['clean'][-1]:.4f}",
                  flush=True)
    summary = {s: round(float(np.nanmean(v)), 4) for s, v in fold_c.items()}
    print(f"=== {cohort} ({_t.time() - t0:.0f}s) ===")
    for s in SCENES:
        print(f"  {s:<8} {summary[s]:.4f}")
    return {"cohort": cohort, "modalities": mods, "seeds": list(seeds),
            "mean_cindex": summary,
            "fold_cindex": {s: [round(float(x), 6) for x in v] for s, v in fold_c.items()}}


def main():
    args = sys.argv[1:]
    cohorts = [a for a in args if not a.startswith("--")] or COHORTS
    seeds = (0, 1, 2)
    for a in args:
        if a.startswith("--seeds="):
            seeds = tuple(int(x) for x in a.split("=", 1)[1].split(",") if x)
    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    for cohort in cohorts:
        res = run_cohort(cohort, seeds)
        with open(os.path.join(REPO, "results", f"{cohort}.json"), "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
