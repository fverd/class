"""Generate synthetic matter power spectra for simple emulator training.

This creates P(k) at z=0 on a fixed k-grid as a toy function of cosmological
parameters. It is NOT a replacement for CLASS — it's for quick local testing.
"""
from __future__ import annotations
import numpy as np


def synthetic_pk(params: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """Compute a toy matter power spectrum.

    params: shape (N,5) or (5,) in order [omega_cdm, H0, ln10A, n_s, tau]
    returns: shape (N, Nk)
    """
    params = np.asarray(params)
    k = np.asarray(k_grid)
    single = False
    if params.ndim == 1:
        params = params.reshape(1, -1)
        single = True

    # extract parameter vectors of shape (N,)
    omega_cdm = params[:, 0]
    H0 = params[:, 1]
    ln10A = params[:, 2]
    n_s = params[:, 3]
    tau = params[:, 4]

    # Convert ln10^{10}A_s to A_s
    A_s = np.exp(ln10A) / 1e10

    # A toy shape: power-law with transfer-like suppression
    # P(k) = A_s * k^{n_s} * T(k; omega_cdm, H0) * f(tau)
    # where T(k) ~ (1 + (k/k_eq)^alpha)^-beta ; k_eq set by omega_cdm and H0
    k_eq = 0.01 * (omega_cdm / 0.12) * (H0 / 67.0)
    alpha = 2.0
    beta = 2.5
    T = 1.0 / (1.0 + (k[None, :] / (k_eq[:, None] + 1e-12)) ** alpha) ** beta

    pk = A_s[:, None] * (k[None, :] ** n_s[:, None]) * T
    # small modulation with tau
    pk *= (1.0 + 0.1 * (tau[:, None] - 0.06))

    if single:
        return pk[0]
    return pk


if __name__ == "__main__":
    # quick demo
    k = np.logspace(-3, 1, 200)
    p = synthetic_pk([0.12, 67.0, 3.05, 0.965, 0.06], k)
    print(p.shape)
