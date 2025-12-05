"""Train emulator from CLASS output files saved as .npz by generate_from_class.py"""
from __future__ import annotations
import os
import glob
import json
import numpy as np
from .emulator import PowerSpectrumEmulator


def load_dataset(data_dir: str):
    # Prefer a single consolidated file
    allfile = os.path.join(data_dir, 'all_points.npz')
    if os.path.exists(allfile):
        d = np.load(allfile, allow_pickle=True)
        params = np.asarray(d['params'])
        pk = np.asarray(d['pk'])
        k = np.asarray(d['k'])
        return params, pk, k

    files = sorted(glob.glob(os.path.join(data_dir, 'point_*.npz')))
    if len(files) == 0:
        raise RuntimeError('No CLASS-generated .npz files found in ' + data_dir)
    params_list = []
    pk_list = []
    k_grid = None
    for fn in files:
        d = np.load(fn, allow_pickle=True)
        if k_grid is None:
            k_grid = d['k']
        params = d['params'].item() if isinstance(d['params'].item(), dict) else d['params']
        # convert to ordered parameter vector [omega_cdm, H0, ln10^{10}A_s, n_s, tau_reio]
        pv = [params['omega_cdm'], params['H0'], params['ln10^{10}A_s'], params['n_s'], params['tau_reio']]
        params_list.append(pv)
        pk_list.append(d['pk'])
    return np.asarray(params_list), np.asarray(pk_list), np.asarray(k_grid)


def normalize_data(X, Y):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0)
    Xn = (X - X_mean) / (X_std + 1e-12)
    Yn = (Y - Y_mean) / (Y_std + 1e-12)
    stats = {'X_mean': X_mean.tolist(), 'X_std': X_std.tolist(), 'Y_mean': Y_mean.tolist(), 'Y_std': Y_std.tolist()}
    return Xn, Yn, stats


def main(data_dir='fra_emulator/class_data', outdir='fra_emulator/models/class_emulator', epochs=100):
    X, Y, k = load_dataset(data_dir)
    Xn, Yn, stats = normalize_data(X, Y)
    em = PowerSpectrumEmulator(k)
    # build tf model if available
    try:
        em.build_tf_model()
    except Exception:
        pass
    em.fit(Xn, Yn, epochs=epochs, batch_size=32, verbose=1)
    os.makedirs(outdir, exist_ok=True)
    em.save(outdir)
    with open(os.path.join(outdir, 'scaler.json'), 'w') as f:
        json.dump(stats, f)
    print('trained and saved emulator to', outdir)


if __name__ == '__main__':
    main()
