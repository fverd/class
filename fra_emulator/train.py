"""Train the emulator on synthetic data and save the model."""
from __future__ import annotations
import numpy as np
import os
from .emulator import PowerSpectrumEmulator
from .generate_synthetic import synthetic_pk


def make_training_set(n_samples=2000, k_grid=None, seed=0):
    rng = np.random.RandomState(seed)
    if k_grid is None:
        k_grid = np.logspace(-3, 1, 200)
    # sample cosmological parameters in plausible ranges
    omega_cdm = rng.uniform(0.08, 0.16, size=(n_samples, 1))
    H0 = rng.uniform(60.0, 75.0, size=(n_samples, 1))
    ln10A = rng.uniform(2.8, 3.4, size=(n_samples, 1))
    n_s = rng.uniform(0.9, 1.02, size=(n_samples, 1))
    tau = rng.uniform(0.04, 0.08, size=(n_samples, 1))

    params = np.hstack([omega_cdm, H0, ln10A, n_s, tau])
    pk = synthetic_pk(params, k_grid)
    return params, pk, k_grid


def main(outdir="models/demo", epochs=30):
    params, pk, k_grid = make_training_set(n_samples=1000)
    em = PowerSpectrumEmulator(k_grid)
    # small train for demo
    em.fit(params, pk, epochs=epochs, batch_size=64, verbose=1)
    em.save(outdir)
    print(f"Saved emulator to {outdir}")


if __name__ == "__main__":
    main()
