"""Smoke test: the model trains and returns readouts of the expected shape."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rrgef import RRGEF  # noqa: E402


def test_fit_predict_shapes():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    n = 60
    X = {m: rng.normal(size=(n, 16)).astype(np.float32) for m in ["a", "b", "c"]}
    time = rng.uniform(10, 1000, size=n)
    event = (rng.random(n) < 0.5).astype(int)
    model = RRGEF()
    model.fit(X, time, event)
    out = model.predict_all(X)
    assert out["risk"].shape == (n,)
    assert out["r_m"].shape == (3, n)
    assert np.all(out["r_m"] <= 1.0 + 1e-6)
    assert out["pair_gap_m"].shape == (3, n)


if __name__ == "__main__":
    test_fit_predict_shapes()
    print("ok")
