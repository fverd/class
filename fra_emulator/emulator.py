"""Power spectrum emulator for CLASS-like outputs.

Provides a TensorFlow-based emulator (if TF is installed) with fallbacks
so the repository tests do not strictly require TF.

Model contract (simple):
- inputs: array-like shape (N,5) in order [omega_cdm, H0, ln10^{10}A_s, n_s, tau_reio]
- outputs: array-like shape (N,Nk) containing P(k) at z=0 on a fixed k-grid

The code will try to import TensorFlow; if unavailable it will use a
sklearn MLPRegressor if available; otherwise it will fall back to a
very small numpy-based ridge/regression.
"""
from __future__ import annotations
import os
import json
from typing import Optional
import numpy as np

HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except Exception:
    HAS_TF = False

HAS_SKLEARN = False
if not HAS_TF:
    try:
        from sklearn.neural_network import MLPRegressor
        HAS_SKLEARN = True
    except Exception:
        HAS_SKLEARN = False


class PowerSpectrumEmulator:
    def __init__(self, k_grid: np.ndarray, model_path: Optional[str] = None):
        self.k_grid = np.asarray(k_grid)
        self.Nk = len(self.k_grid)
        self.model = None
        self.model_path = model_path

    def build_tf_model(self, hidden=(128, 128)):
        if not HAS_TF:
            raise RuntimeError("TensorFlow not available")
        inp = keras.Input(shape=(5,), name="cosmo_params")
        x = inp
        for h in hidden:
            x = keras.layers.Dense(h, activation="relu")(x)
        out = keras.layers.Dense(self.Nk, activation="linear", name="pk_out")(x)
        model = keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        self.model = model
        return model

    def fit(self, params: np.ndarray, pk: np.ndarray, epochs: int = 50, batch_size: int = 32, verbose: int = 1):
        params = np.asarray(params)
        pk = np.asarray(pk)
        assert params.ndim == 2 and params.shape[1] == 5
        assert pk.ndim == 2 and pk.shape[1] == self.Nk

        if HAS_TF:
            if self.model is None:
                self.build_tf_model()
            history = self.model.fit(params, pk, epochs=epochs, batch_size=batch_size, verbose=verbose)
            return history

        if HAS_SKLEARN:
            # sklearn's MLPRegressor flattens multioutput automatically
            self.model = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=epochs)
            self.model.fit(params, pk)
            return None

        # very small numpy fallback: solve linear ridge for each k
        # pk = X @ W + b; we solve for W via ridge regression
        X = np.hstack([params, np.ones((params.shape[0], 1))])
        alpha = 1e-3
        XtX = X.T.dot(X)
        W = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), X.T.dot(pk))
        self.model = {"W": W}
        return None

    def predict(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params)
        single = False
        if params.ndim == 1:
            params = params.reshape(1, -1)
            single = True
        assert params.shape[1] == 5

        if HAS_TF and self.model is not None and hasattr(self.model, "predict"):
            pred = self.model.predict(params)
        elif HAS_SKLEARN and self.model is not None:
            pred = self.model.predict(params)
        else:
            # numpy fallback
            W = self.model["W"]
            X = np.hstack([params, np.ones((params.shape[0], 1))])
            pred = X.dot(W)

        if single:
            return pred[0]
        return pred

    def save(self, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        # save k grid
        np.save(os.path.join(outdir, "k_grid.npy"), self.k_grid)
        meta = {"Nk": self.Nk}
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f)
        if HAS_TF and self.model is not None and hasattr(self.model, "save"):
            self.model.save(os.path.join(outdir, "tf_model"))
        elif HAS_SKLEARN and self.model is not None:
            # sklearn model
            import joblib
            joblib.dump(self.model, os.path.join(outdir, "sk_model.joblib"))
        else:
            # numpy fallback
            np.save(os.path.join(outdir, "numpy_model_W.npy"), self.model["W"])

    @classmethod
    def load(cls, outdir: str) -> "PowerSpectrumEmulator":
        k_grid = np.load(os.path.join(outdir, "k_grid.npy"))
        em = cls(k_grid=k_grid, model_path=outdir)
        if HAS_TF and os.path.exists(os.path.join(outdir, "tf_model")):
            em.model = keras.models.load_model(os.path.join(outdir, "tf_model"))
        elif os.path.exists(os.path.join(outdir, "sk_model.joblib")):
            import joblib
            em.model = joblib.load(os.path.join(outdir, "sk_model.joblib"))
        elif os.path.exists(os.path.join(outdir, "numpy_model_W.npy")):
            W = np.load(os.path.join(outdir, "numpy_model_W.npy"))
            em.model = {"W": W}
        else:
            raise FileNotFoundError("No saved model found in " + outdir)
        return em


if __name__ == "__main__":
    print("This module provides PowerSpectrumEmulator. Import from Python to use.")
