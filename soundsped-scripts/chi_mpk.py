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
'perturbations_verbose':0,
'background_verbose':0,
'fourier_verbose':3,
'output':'mPk',
'gauge':'newtonian',
'P_k_max_1/Mpc':10,
'z_max_pk':3000,
# 'k_min_tau0':0.0001,

# 'k_min_tau0':100.,
# 'evolver': 'rk'
}

chiCDM = Class()
# pass input parameters
chiCDM.set(common_settings)


chiCDM.set({
'omega_cdm':0.12,
'omega_chi':0.01,
'acs_chi':1.e-3,
'cs2_peak_chi':1/3,
# 'N_ncdm':1,
# 'm_ncdm':0.4
})

standardCDM = Class()
standardCDM.set(common_settings)
standardCDM.set({'omega_cdm':0.12})

# run class
chiCDM.compute()

standardCDM.compute()





import matplotlib.pyplot as plt
from math import pi

kk = np.logspace(-4,np.log10(1),500) # k in h/Mpc
Pkcann = [] # P(k) in (Mpc/h)**3
Pkstand = [] # P(k) in (Mpc/h)**3
h = chiCDM.h() # get reduced Hubble for conversions to 1/Mpc

lowz=10.

for k in kk:
    Pkcann.append(chiCDM.pk_cb_lin(k*h,lowz)*h**3) # function .pk(k,z)
    Pkstand.append(standardCDM.pk_lin(k*h,lowz)*h**3 ) # function .pk(k,z)
Pkcann=np.array(Pkcann)
# Pkcann *= Pkstand[0]/Pkcann[0] #normalize to LCDM large scale

plt.figure(figsize=(4,3))
plt.xscale('log');plt.yscale('log');plt.xlim(kk[0],kk[-1])
plt.xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
plt.ylabel(r'$P(k) \,\,\,\, [\mathrm{Mpc}/h]^3$')

plt.plot(kk,Pkstand,'k-',label=r'Standard CDM')
plt.plot(kk,Pkcann,'r-',label=r'$f_\chi$')

plt.title(f'z={str(lowz)}')

plt.legend(loc='best')
plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/soundspeed_mpk.pdf',bbox_inches='tight')



chiCDM.empty()
standardCDM.empty()
