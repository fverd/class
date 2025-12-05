"""Quick smoke test for the emulator using synthetic data."""
from __future__ import annotations
import numpy as np
from .emulator import PowerSpectrumEmulator
from .generate_synthetic import synthetic_pk


def test_predict_shapes():
    k = np.logspace(-3, 1, 50)
    em = PowerSpectrumEmulator(k)
    # train small
    params = np.array([[0.12, 67.0, 3.05, 0.965, 0.06], [0.13, 70.0, 3.1, 0.98, 0.065]])
    pk = synthetic_pk(params, k)
    em.fit(params, pk, epochs=5)
    p = em.predict(params[0])
    assert p.shape == (k.size,)
    P = em.predict(params)
    assert P.shape == (2, k.size)


if __name__ == "__main__":
    test_predict_shapes()
    print("smoke test passed")
