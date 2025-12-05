"""Generate P(k) training data by calling CLASS.

This script attempts to import the Python wrapper `classy.Class`. If not
available, it will look for a `class` executable in the repository root.

It writes one npz file per parameter point with arrays `k` and `pk`.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Dict, Tuple


def run_class_point(params: Dict[str, float], k_grid: np.ndarray, outpath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run CLASS for a single parameter dict and return (k, pk).

    params: dictionary of parameters understood by CLASS (e.g., 'omega_cdm', 'H0', ...)
    k_grid: desired output k grid (in 1/Mpc)
    outpath: location to save the raw CLASS output if using binary mode
    """
    try:
        # Try python wrapper
        from classy import Class
    except Exception:
        # Try to load local python wrapper in repository `python/` folder
        import sys
        repo_root = os.path.dirname(os.path.dirname(__file__))
        local_python = os.path.join(repo_root, 'python')
        if local_python not in sys.path:
            sys.path.insert(0, local_python)
        try:
            from classy import Class
        except Exception:
            Class = None

    if Class is not None:
        cosmo = Class()
        # Expect params dict keys exactly:
        # 'omega_cdm','H0','ln10^{10}A_s','n_s','tau_reio'
        cosmo_params = {
            'omega_cdm': params['omega_cdm'],
            'h': params['H0'] / 100.0,
            'ln10^{10}A_s': params['ln10^{10}A_s'],
            'n_s': params['n_s'],
            'tau_reio': params['tau_reio'],
            'output': 'mPk',
            'P_k_max_1/Mpc': float(k_grid.max()),
        }
        cosmo.set(cosmo_params)
        cosmo.compute()
        # Use get_pk_and_k_and_z which returns pk[k_index, z_index], k vector and z vector
        try:
            pk_grid, k_vals, z_vals = cosmo.get_pk_and_k_and_z(nonlinear=False, only_clustering_species=False, h_units=False)
            # pk_grid has shape (len(k_vals), len(z_vals)); z_vals last entry should be 0
            # pick the column corresponding to z==0 (should be last)
            if z_vals[-1] == 0.0:
                pk = pk_grid[:, -1]
            else:
                # find index closest to z=0
                iz = int(np.argmin(np.abs(z_vals - 0.0)))
                pk = pk_grid[:, iz]
            k = k_vals
        except Exception:
            # fallback: query get_pk for each k using get_pk_all
            k = k_grid
            try:
                pk = cosmo.get_pk_all(k, [0.0], nonlinear=False)
                # get_pk_all returns array shape (len(z), len(k)) or similar; ensure shape (len(k),)
                # When asked with z array, it returns out_pk with shape (len(z), len(k))
                if isinstance(pk, np.ndarray) and pk.ndim == 2:
                    # pk[z_index, k_index]
                    pk = pk[0, :]
                else:
                    pk = np.asarray(pk)
            except Exception as e:
                raise RuntimeError('Could not extract P(k) from CLASS: ' + str(e))

        cosmo.empty()
        cosmo.struct_cleanup()
        return k, pk

    # Fallback: call class binary if present
    class_bin = os.path.join(os.getcwd(), 'class')
    if not os.path.exists(class_bin):
        raise RuntimeError('Neither python classy wrapper nor class binary found')

    # create an ini file with requested params and run class
    ini = os.path.join(outpath, 'tmp_params.ini')
    with open(ini, 'w') as f:
        f.write('output = mPk\n')
        f.write(f'P_k_max_1/Mpc = {float(k_grid.max())}\n')
        # minimal mapping
        f.write(f'omega_cdm = {params["omega_cdm"]}\n')
        f.write(f'H0 = {params["H0"]}\n')
        f.write(f'ln10^{10}A_s = {params["ln10A"]}\n')
        f.write(f'n_s = {params["n_s"]}\n')
        f.write(f'tau_reio = {params["tau_reio"]}\n')

    import subprocess
    subprocess.check_call([class_bin, ini], cwd=os.getcwd())
    # CLASS will write output files in ./output/ by default — search for latest
    # This simple fallback will fail for complex repos; prefer the python wrapper.
    raise RuntimeError('Binary invocation path not fully implemented in fallback')


def sample_grid(n_samples: int = 50):
    rng = np.random.RandomState(0)
    omega_cdm = rng.uniform(0.08, 0.16, size=n_samples)
    H0 = rng.uniform(60.0, 75.0, size=n_samples)
    ln10A = rng.uniform(2.8, 3.4, size=n_samples)
    n_s = rng.uniform(0.9, 1.02, size=n_samples)
    tau = rng.uniform(0.04, 0.08, size=n_samples)
    for i in range(n_samples):
        yield {
            'omega_cdm': float(omega_cdm[i]),
            'H0': float(H0[i]),
            'ln10^{10}A_s': float(ln10A[i]),
            'n_s': float(n_s[i]),
            'tau_reio': float(tau[i]),
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='fra_emulator/class_data')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    k_grid = np.logspace(-3, 1, 200)
    ks = []
    pks = []
    params_list = []
    param_keys = ['omega_cdm', 'H0', 'ln10^{10}A_s', 'n_s', 'tau_reio']
    success = 0
    for i, p in enumerate(sample_grid(args.n)):
        try:
            k, pk = run_class_point(p, k_grid, args.outdir)
        except Exception as e:
            print('CLASS run failed for sample', i, 'error:', e)
            continue
        # record
        ks.append(k)
        pvector = [p[key] for key in param_keys]
        params_list.append(pvector)
        pks.append(pk)
        success += 1
        print('saved sample', i)

    if success == 0:
        print('No successful CLASS runs; nothing to save')
    else:
        # Ensure consistent k grid (we requested same k_grid)
        # store the requested k_grid; pk array shape (N, Nk)
        params_array = np.asarray(params_list)
        pk_array = np.asarray(pks)
        outfn = os.path.join(args.outdir, 'all_points.npz')
        np.savez(outfn, k=k_grid, params=params_array, pk=pk_array, param_keys=np.array(param_keys))
        print(f'Saved {success} samples to {outfn}')
