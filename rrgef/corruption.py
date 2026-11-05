"""Test-time corruption scenarios used in the paper (one modality at a time)."""
import hashlib

import numpy as np

SIGMAS = [1.0, 2.0, 4.0, 8.0]
SCENES = ["clean"] + [f"noise{s:g}" for s in SIGMAS] + ["wipeout", "shuffle"]


def stable_seed(*parts):
    """Process-independent seed for reproducible permutation tests."""
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")

def corrupt_noise(Xte, which, sigma, rng):
    Y = {m: v.copy() for m, v in Xte.items()}
    Y[which] = (Y[which] + rng.normal(0, sigma, Y[which].shape)).astype(np.float32)
    return Y


def corrupt_wipeout(Xte, which):
    """Wipeout: set the whole modality to zero (failed or missing assay)."""
    Y = {m: v.copy() for m, v in Xte.items()}
    Y[which] = np.zeros_like(Y[which])
    return Y


def corrupt_shuffle(Xte, which, rng):
    """Shuffle: permute the modality across patients (sample mix-up)."""
    Y = {m: v.copy() for m, v in Xte.items()}
    perm = rng.permutation(len(Y[which]))
    Y[which] = Y[which][perm]
    return Y
