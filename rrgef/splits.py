"""Shared outer-CV split policy for all final experiments."""

import numpy as np
from sklearn.model_selection import StratifiedKFold


def outer_splits(time, event, seed, n_splits=5):
    """Return reproducible event-stratified folds.

    Survival time is kept only for the API/documentation here; event status is
    the stratification label.  Every runner must use this helper so that the
    clean, corruption and reliability experiments evaluate identical folds.
    """
    event = np.asarray(event).astype(int)
    dummy = np.zeros(len(event), dtype=np.int8)
    return StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed)
    ).split(dummy, event)

