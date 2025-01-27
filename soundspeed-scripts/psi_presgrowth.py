#!/usr/bin/env python3
# coding: utf-8
# import classy module
from classy import Class
import numpy as np
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
from scipy.special import hyp2f1


common_settings = {
'omega_b':0.0223828,
'h':0.67810,
'z_reio':7.6711,
'YHe':0.25,
'N_ur': 3.05,
'perturbations_verbose':1,
'background_verbose':3,
'output':'mTk, vTk',
'gauge':'newtonian',
'P_k_max_1/Mpc':10,
'z_max_pk':1000,
'format':'class'
}

# sys.exit()
## CLASS part
chiCDM = Class()
# pass input parameters
chiCDM.set(common_settings)

chiCDM.set({
'omega_cdm':0.42,
'f_chi':0.1,
'acs_chi':0.8e-3,
'cs2_peak_chi':1./3.
})

standardCDM = Class()
standardCDM.set(common_settings)
standardCDM.set({'omega_cdm':0.11})

# run class
chiCDM.compute()

# standardCDM.compute()
h = chiCDM.h()
background = chiCDM.get_background() # load background table
bk_a = 1/(background['z']+1) # read redshift
Ha=bk_a*background['H [1/Mpc]']
background_kJ_chi=background['(.)kJ_chi']


a_eval=np.logspace(-2,-1,30)
chi_tk_z=[]
for a in a_eval:
    chi_tk_z.append(chiCDM.get_transfer(z=1/a-1))
chi_tk_z=np.array(chi_tk_z)

# stand_tk=standardCDM.get_transfer(z=z_eval)

print(chi_tk_z[1].keys())
kidx=80
print('Evaluating at k=',chi_tk_z[1]['k (h/Mpc)'][kidx])
# print(tk['d_chi'].shape)

d_chi_z=-np.array([d['d_chi'][kidx] for d in chi_tk_z])
d_cdm_z=-np.array([d['d_cdm'][kidx] for d in chi_tk_z])

CLASS_ratio=d_cdm_z/d_cdm_z[0]

plt.figure(figsize=(4,3), dpi=150)

plt.plot(a_eval,CLASS_ratio/a_eval,'g-o',label=r'$\delta_c/a$ ($k=0.35$)', markersize=1.)
plt.axhline(y=1/a_eval[0], color='k')
plt.plot(a_eval, pow(a_eval,(-3/5*0.1))/(pow(a_eval[0],(1-3/5*0.1))), color='k')

kidx=60
print('Evaluating at k=',chi_tk_z[1]['k (h/Mpc)'][kidx])
# print(tk['d_chi'].shape)

d_chi_z=-np.array([d['d_chi'][kidx] for d in chi_tk_z])
d_cdm_z=-np.array([d['d_cdm'][kidx] for d in chi_tk_z])

CLASS_ratio=d_cdm_z/d_cdm_z[0]
plt.plot(a_eval,CLASS_ratio/a_eval,'r-o',label=r'$\delta_c/a$ ($k=0.18$)', markersize=1.)

kidx=40
print('Evaluating at k=',chi_tk_z[1]['k (h/Mpc)'][kidx])
# print(tk['d_chi'].shape)

d_chi_z=-np.array([d['d_chi'][kidx] for d in chi_tk_z])
d_cdm_z=-np.array([d['d_cdm'][kidx] for d in chi_tk_z])

CLASS_ratio=d_cdm_z/d_cdm_z[0]
plt.plot(a_eval,CLASS_ratio/a_eval,'b-o',label=r'$\delta_c/a$ ($k=0.09$)', markersize=1.)


# plt.legend(loc='best')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plt.xlim([1.e-2,1])
plt.xscale('log')
plt.yscale('log')

plt.xlabel('a')
# plt.grid()
# axs[1].legend(loc='best')
# axs[1].set_xlabel('a')
# axs[1].set_yscale('log')

plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/growth.pdf',bbox_inches='tight')




chiCDM.empty()
standardCDM.empty()
