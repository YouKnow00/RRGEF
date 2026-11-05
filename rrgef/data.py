"""Load TCGA cohorts as patient-level matrices without any global preprocessing.

Raw cBioPortal tables (see ``scripts/download_data.py``) are aggregated to
patients; duplicate aliquots are averaged and remaining missing values are set
to zero.  No feature selection or scaling happens here: every statistic is fitted
on the training fold by ``rrgef.preprocessing.select_fit_apply``.

Patient order and overall-survival labels for the four cohorts used in the
paper are provided in ``data/labels/<cohort>_labels.csv``.
"""
import csv
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR = os.path.join(REPO, "data", "labels")
RAW_DIR = os.path.join(REPO, "data", "raw")

RAW_FILES = {
    "mrna": "data_mrna_seq_v2_rsem.txt",
    "methyl": "data_methylation_hm450.txt",
    "cna": "data_linear_cna.txt",
    "mutation": "data_mutations.txt",
}
MOD_ORDER = ["mrna", "methyl", "cna", "mutation"]
# Modalities used in the paper for each cohort.
COHORT_MODALITIES = {
    "TCGA-BRCA": ["mrna", "methyl", "cna"],
    "TCGA-KIRC": ["mrna", "methyl", "cna"],
    "TCGA-LIHC": ["mrna", "methyl", "cna", "mutation"],
    "TCGA-LUAD": ["mrna", "methyl", "cna", "mutation"],
}


def load_labels(cohort, label_dir=LABEL_DIR):
    """Return (patients, time, event) in the fixed patient order."""
    patients, time, event = [], [], []
    with open(os.path.join(label_dir, f"{cohort}_labels.csv"), newline="") as f:
        for row in csv.DictReader(f):
            patients.append(row["patient_id"])
            time.append(float(row["os_time"]))
            event.append(int(row["os_event"]))
    return (np.asarray(patients, dtype=object).astype(str),
            np.asarray(time, dtype=float), np.asarray(event, dtype=int))


def pid_of(sample_id):
    """Map a TCGA sample barcode to its patient barcode."""
    return str(sample_id).rsplit("-", 1)[0]


def _sample_columns(columns):
    """Return raw matrix columns that look like TCGA sample barcodes."""
    return [c for c in columns if str(c).startswith("TCGA-")]


def _patient_mean(values, sample_pids, patients):
    """Aggregate a feature-by-sample array to patient-by-feature.

    Duplicate aliquots are averaged per patient.  Missing/non-numeric entries
    remain missing until the final conversion in ``load_cohort``.
    """
    values = np.asarray(values, dtype=np.float32)
    sample_pids = np.asarray(sample_pids, dtype=object)
    n_patients, n_features = len(patients), values.shape[0]
    out = np.full((n_patients, n_features), np.nan, dtype=np.float32)
    for pi, patient in enumerate(patients):
        idx = np.flatnonzero(sample_pids == patient)
        if idx.size == 0:
            continue
        block = values[:, idx]
        valid = np.isfinite(block)
        count = valid.sum(axis=1)
        total = np.nansum(block, axis=1, dtype=np.float32)
        good = count > 0
        if np.any(good):
            out[pi, good] = total[good] / count[good]
    return out


def _load_numeric_matrix(path, patients):
    """Read an expression/CNA/methylation table and aggregate to patients."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if df.shape[1] < 2:
        raise ValueError(f"raw matrix has no sample columns: {path}")
    sample_cols = _sample_columns(df.columns)
    if not sample_cols:
        # Keep the explicit fallback for unusually named cBioPortal exports.
        drop_cols = {df.columns[0], "Entrez_Gene_Id"}
        sample_cols = [c for c in df.columns if c not in drop_cols]
    sample_pids = np.asarray([pid_of(c) for c in sample_cols], dtype=object)
    wanted = np.isin(sample_pids, np.asarray(patients, dtype=object))
    if not np.any(wanted):
        raise ValueError(f"no cohort patients found in raw matrix: {path}")
    sample_cols = [c for c, keep in zip(sample_cols, wanted) if keep]
    sample_pids = sample_pids[wanted]

    # ``to_numeric`` handles the cBioPortal NA sentinel without relying on a
    # global fill or a global statistic.  Feature names are intentionally not
    # used for selection here; their row order is stable within this run.
    vals = df[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    patient_x = _patient_mean(vals, sample_pids, patients)
    return patient_x


def _load_mutation_matrix(path, patients):
    """Build a patient-by-gene binary mutation matrix without global top-k."""
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in {
        "Hugo_Symbol", "Tumor_Sample_Barcode"
    }, low_memory=False)
    if "Hugo_Symbol" not in df or "Tumor_Sample_Barcode" not in df:
        raise ValueError(f"mutation table missing required columns: {path}")
    df = df.dropna(subset=["Hugo_Symbol", "Tumor_Sample_Barcode"]).copy()
    df["_pid"] = df["Tumor_Sample_Barcode"].astype(str).map(pid_of)
    df = df[df["_pid"].isin(set(patients))]
    if df.empty:
        raise ValueError(f"no cohort patients found in mutation table: {path}")
    # Crosstab preserves a deterministic sorted gene order.  The training-fold
    # variance cap in select_fit_apply will decide which genes are retained.
    table = pd.crosstab(df["_pid"], df["Hugo_Symbol"])
    table = table.reindex(index=list(patients), fill_value=0)
    return (table.to_numpy(dtype=np.float32) > 0).astype(np.float32)


def load_cohort(cohort, raw_dir=RAW_DIR, label_dir=LABEL_DIR):
    """Return ``(X, time, event, modalities)`` for one cohort.

    ``X`` maps modality name to a (patients x features) float32 matrix.
    """
    patients, time_all, event_all = load_labels(cohort, label_dir)
    cdir = os.path.join(raw_dir, cohort)
    X = {}
    for mod in COHORT_MODALITIES.get(cohort, MOD_ORDER):
        path = os.path.join(cdir, RAW_FILES[mod])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; run scripts/download_data.py first")
        if mod == "mutation":
            arr = _load_mutation_matrix(path, patients)
        else:
            arr = _load_numeric_matrix(path, patients)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if arr.shape[0] != len(patients):
            raise ValueError(f"patient alignment failed for {cohort}/{mod}: {arr.shape}")
        X[mod] = np.ascontiguousarray(arr)
    mods = [m for m in MOD_ORDER if m in X]
    return X, time_all, event_all, mods
