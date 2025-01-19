#!/usr/bin/env python3
# coding: utf-8
# import classy module
from classy import Class
import numpy as np

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
'z_max_pk':3000,
'format':'class'
}

chiCDM = Class()
# pass input parameters
chiCDM.set(common_settings)


chiCDM.set({
'omega_cdm':0.1,
'omega_chi':0.01,
'acs_chi':1.e-3,
'cs2_peak_chi':1./3.
})

standardCDM = Class()
standardCDM.set(common_settings)
standardCDM.set({'omega_cdm':0.11})

# run class
chiCDM.compute()

# standardCDM.compute()


import matplotlib.pyplot as plt

a_eval=np.logspace(-2.5,0,30)

chi_tk_z=[]
for a in a_eval:
    chi_tk_z.append(chiCDM.get_transfer(z=1/a-1))

chi_tk_z=np.array(chi_tk_z)

# stand_tk=standardCDM.get_transfer(z=z_eval)

print(chi_tk_z[1].keys())
kidx=60
print('Evaluating at k=',chi_tk_z[1]['k (h/Mpc)'][kidx])
# print(tk['d_chi'].shape)

d_chi_z=-np.array([d['d_chi'][kidx] for d in chi_tk_z])
d_cdm_z=-np.array([d['d_cdm'][kidx] for d in chi_tk_z])
t_cdm_z=np.array([d['t_cdm'][kidx] for d in chi_tk_z])
# t_chi_z=-np.array([d['t_chi'][kidx] for d in chi_tk_z])



plt.figure(2)
plt.xscale('log');
# plt.xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
plt.xlabel(r'$a$')

# tk_k=chiCDM.get_transfer(z=0)
# k=tk_k['k (h/Mpc)']
# # print(tk_k['d_cdm'][2])
# plt.loglog(k,(tk_k['d_cdm']*tk_k['d_cdm'])/k/k/k,'k-',label=r'd_cdm  z=100')



# plt.plot(a_eval,d_chi_z,'r-',label=r'd_chi for chiCDM')
# plt.plot(a_eval,d_cdm_z,'k-',label=r'd_cdm for chiCDM')

plt.plot(a_eval,d_chi_z/d_cdm_z,label=r'd_chi/d_cdm')
# plt.plot(a_eval,t_chi_z/d_cdm_z,label=r't_chi/d_cdm')
# plt.plot(a_eval,t_cdm_z,label=r't_cdm/d_cdm')

plt.legend(loc='best')
plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/soundspeed_tk.pdf',bbox_inches='tight')



chiCDM.empty()
standardCDM.empty()
