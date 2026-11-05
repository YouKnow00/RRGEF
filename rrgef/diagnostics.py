"""Readout-only cross-modal pairing diagnostics.

For each training fold, every modality is reduced to a few principal
components and ridge maps predict each modality from every other on matched
training patients.  Test-time prediction errors are standardized by the
training-fold median and 90th percentile, truncated at zero and averaged per
patient and per modality.  A covariance-adjusted variant uses a Mahalanobis
distance with a Ledoit-Wolf residual covariance.  These scores never enter
the risk prediction.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

PCA_DIM = 8


def _fit_pair_state(Xtr, mods, pca_dim=PCA_DIM, ridge_alpha=10.0):
    state = {"mods": list(mods), "pca": {}, "maps": {}, "stats": {},
             "maha_precision": {}, "maha_stats": {}}
    ztr = {}
    for m in mods:
        k = min(int(pca_dim), Xtr[m].shape[0] - 1, Xtr[m].shape[1])
        pca = PCA(n_components=max(1, k), svd_solver="full", random_state=0)
        ztr[m] = pca.fit_transform(Xtr[m]).astype(np.float32)
        state["pca"][m] = pca
    for i, src in enumerate(mods):
        for j, tgt in enumerate(mods):
            if i == j:
                continue
            model = Ridge(alpha=float(ridge_alpha)).fit(ztr[src], ztr[tgt])
            err = ((model.predict(ztr[src]) - ztr[tgt]) ** 2).mean(axis=1)
            med = float(np.median(err))
            q90 = float(np.quantile(err, 0.90))
            residual = model.predict(ztr[src]) - ztr[tgt]
            precision = LedoitWolf().fit(residual).precision_
            maha_err = np.einsum("ni,ij,nj->n", residual, precision, residual)
            maha_med = float(np.median(maha_err))
            maha_q90 = float(np.quantile(maha_err, 0.90))
            state["maps"][(src, tgt)] = model
            state["stats"][(src, tgt)] = (med, max(q90 - med, 1e-8))
            state["maha_precision"][(src, tgt)] = precision
            state["maha_stats"][(src, tgt)] = (maha_med, max(maha_q90 - maha_med, 1e-8))
    return state


def _score(state, X, mods):
    z = {m: state["pca"][m].transform(X[m]).astype(np.float32) for m in mods}
    pair = {}
    for (src, tgt), model in state["maps"].items():
        err = ((model.predict(z[src]) - z[tgt]) ** 2).mean(axis=1)
        med, scale = state["stats"][(src, tgt)]
        pair[(src, tgt)] = np.maximum(0.0, (err - med) / scale)
    patient = np.mean(np.stack(list(pair.values()), axis=0), axis=0)
    modality = np.zeros((len(mods), len(patient)), dtype=float)
    for mi, m in enumerate(mods):
        edges = [v for (src, tgt), v in pair.items() if src == m or tgt == m]
        modality[mi] = np.mean(np.stack(edges), axis=0) if edges else 0.0
    return patient, modality


# Public aliases used by the optional diagnostic baseline wrapper.  Keeping
# the implementation here also lets the development probe and the wrapper
# share exactly the same fold calibration code.
def fit_pair_state(Xtr, mods, pca_dim=PCA_DIM, ridge_alpha=10.0):
    return _fit_pair_state(Xtr, mods, pca_dim=pca_dim, ridge_alpha=ridge_alpha)


def score_pair_state(state, X, mods):
    return _score(state, X, mods)


def score_maha_pair_state(state, X, mods):
    """Residual-covariance corrected pair-gap readout."""
    z = {m: state["pca"][m].transform(X[m]).astype(np.float32) for m in mods}
    pair = {}
    for (src, tgt), model in state["maps"].items():
        residual = model.predict(z[src]) - z[tgt]
        raw = np.einsum("ni,ij,nj->n", residual,
                        state["maha_precision"][(src, tgt)], residual)
        med, scale = state["maha_stats"][(src, tgt)]
        pair[(src, tgt)] = np.maximum(0.0, (raw - med) / scale)
    patient = np.mean(np.stack(list(pair.values()), axis=0), axis=0)
    modality = np.zeros((len(mods), len(patient)), dtype=float)
    for mi, m in enumerate(mods):
        edges = [v for (src, tgt), v in pair.items() if src == m or tgt == m]
        modality[mi] = np.mean(np.stack(edges), axis=0) if edges else 0.0
    return patient, modality
