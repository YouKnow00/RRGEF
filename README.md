# RRGEF

**Relative reliability-guided evidential fusion for robust multi-omics survival prediction**

RRGEF is an evidential survival model that separates *input reliability* from *prognostic evidence*. For each patient, every omics modality produces low- and high-risk evidence through its own branch. A reconstruction decoder measures how far that patient's input departs from the structure learned on training data. These reconstruction deviations are calibrated on the training fold and compared across the patient's available modalities to give relative reliabilities. The reliabilities scale each modality's evidence before fusion, and the fused evidence yields a continuous risk score trained with the Cox partial likelihood.

The same signals also act as diagnostics. They identify patients whose inputs are corrupted, localize the affected modality, and (through readout-only pairing checks) flag cross-modal mismatches.

> Manuscript: *RRGEF: relative reliability-guided evidential fusion for robust multi-omics survival prediction* (submitted to *Bioinformatics*).

## Installation

```bash
git clone https://github.com/YouKnow00/RRGEF.git
cd RRGEF
pip install -r requirements.txt
```

Requirements are Python ≥ 3.9, PyTorch ≥ 2.0, NumPy, pandas and scikit-learn. All experiments run on CPU.

## Quick start

```python
from rrgef import RRGEF, select_fit_apply, concordance_index

# X_train / X_test: {modality name: (patients x features) numpy array}
Xtr, Xte = select_fit_apply(X_train, X_test, time_train, event_train, k=256)

model = RRGEF()                       # paper configuration
model.fit(Xtr, time_train, event_train)

risk = model.predict_risk(Xte)        # larger = higher risk
print(concordance_index(time_test, risk, event_test))

out = model.predict_all(Xte)
out["r_m"]            # (modalities x patients) relative reliability; order = model.mods_
out["u_total"]        # evidence-insufficiency score
out["energy_excess"]  # feature-energy diagnostic
out["pair_gap_m"]     # cross-modal pairing diagnostic per modality
```

A self-contained example on synthetic data:

```bash
python examples/demo_synthetic.py
```

## Reproducing the TCGA experiments

1. **Download the data.** The script fetches the public cBioPortal datahub tables for the studies `brca_tcga`, `kirc_tcga`, `lihc_tcga` and `luad_tcga`: mRNA (RSEM), HM450 methylation, linear CNA, and mutations for LIHC and LUAD.

   ```bash
   python scripts/download_data.py
   ```

   Patient order and overall-survival labels used in the paper are provided in `data/labels/`.

2. **Run five-fold cross-validation × three seeds** with all corruption scenarios:

   ```bash
   python scripts/run_cv.py                       # all four cohorts
   python scripts/run_cv.py TCGA-LUAD --seeds=0   # quick check
   ```

   Mean C-indices for the clean, noise (σ = 1, 2, 4, 8), wipeout and shuffle scenarios are printed and saved to `results/`.

**Protocol.** The following are fitted on the training fold only:
- the variance cap (1024 features) and univariate Cox selection (256 features per modality);
- standardization;
- reliability calibration and diagnostic calibration.

Test-fold labels are never used for feature selection, calibration or hyperparameter setting. Corruptions are applied to one modality at a time, and results are averaged over modalities.

**Note.** The cBioPortal datahub is updated over time. If upstream files have changed since the analysis, numbers may differ slightly from those reported in the paper.

## Repository layout

| Path | Content |
|---|---|
| `rrgef/model.py` | Network: modality branches, reliability gating, evidential fusion, losses |
| `rrgef/training.py` | Training loop and ensemble inference |
| `rrgef/rrgef.py` | `RRGEF` estimator (`fit` / `predict_risk` / `predict_all`) with the paper configuration |
| `rrgef/preprocessing.py` | Fold-safe variance cap, univariate Cox selection and standardization |
| `rrgef/diagnostics.py` | Readout-only cross-modal pairing diagnostics |
| `rrgef/corruption.py` | Noise, wipeout and shuffle scenarios |
| `rrgef/data.py` | Loader for the raw cBioPortal tables |
| `rrgef/metrics.py`, `rrgef/splits.py` | C-index; event-stratified outer folds |
| `scripts/` | Data download and cross-validation runner |
| `examples/`, `tests/` | Synthetic example and smoke test |

`rrgef/model.py` and `rrgef/training.py` are the frozen implementation used for the paper. Some development options remain as constructor flags but are switched off in the paper configuration, and a few inline comments are in Chinese.

## Citation

If you use RRGEF, please cite the accompanying article (citation details will be added on publication).

## License

MIT License. See `LICENSE`.
