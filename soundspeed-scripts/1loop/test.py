import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root
from scipy.special import sici
from classy import Class
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})
h = 0.67810
_H0_ = 3.336e-04 * h
_ev_to_HO_ = 1.56e29 / _H0_
k_out = [0.01,0.1, 0.5]
common_settings = {
'omega_b':0.0223828,
'h':h,
'z_reio':7.6711,
'YHe':0.25,
'output_verbose':90,
'output':'mPk',
'P_k_max_h/Mpc':0.009,
'z_max_pk':0.01,

}

kvals = np.logspace(-3,0.35,5).tolist()

Mnu=1
nuCDM = Class()
nuCDM.set(common_settings)
nuCDM.set({
    # 'N_ncdm':3,
    # 'm_ncdm':f'{Mnu/3:.1g},{Mnu/3:.1g},{Mnu/3:.1g}',
    'N_ncdm':1,
    'm_ncdm':Mnu,
    'ncdm_fluid_approximation':3,
})
nuCDM.set({'k_output_values':str(kvals).strip('[]')})
nuCDM.compute()

print(nuCDM.get_perturbations()['scalar'][-1]['delta_cdm'][-1])

# shear_nu_k = np.array([nuCDM.get_perturbations()['scalar'][i]['shear_ncdm[0]'][-1] for i in range(len(kvals))])
# cs2_nu_k = np.array([nuCDM.get_perturbations()['scalar'][i]['cs2_ncdm[0]'][-1] for i in range(len(kvals))])
# d_nu_k = np.array([nuCDM.get_perturbations()['scalar'][i]['delta_ncdm[0]'][-1] for i in range(len(kvals))])
# d_cdm_k = np.array([nuCDM.get_perturbations()['scalar'][i]['delta_cdm'][-1] for i in range(len(kvals))])

# print(shear_nu_k)