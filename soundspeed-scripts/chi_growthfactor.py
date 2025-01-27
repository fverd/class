#!/usr/bin/env python3
# coding: utf-8
# import classy module
from classy import Class
import numpy as np
import N
common_settings = {
'omega_b':0.0223828,
'h':0.67810,
'z_reio':7.6711,
'YHe':0.25,
'N_ur': 3.05,
'perturbations_verbose':1,
'background_verbose':3,
'output':'mPk',
'gauge':'newtonian',
'P_k_max_1/Mpc':10,
'z_max_pk':3000,
}

chiCDM = Class()
# pass input parameters
chiCDM.set(common_settings)

np.l
chiCDM.set({
'omega_cdm':0.1,
'omega_chi':0.01
})
np.lo
standardCDM = Class()
standardCDM.set(common_settings)
# standardCDM.set({'Omega_cdm':1-0.0486773}) #EdS
standardCDM.set({'omega_cdm':0.11}) #LCDM

# run class
chiCDM.compute()
lis
standardCDM.compute()





import matplotlib.pyplot as plt
from math import pi

h = chiCDM.h() # get reduced Hubble for conversions to 1/Mpc


plt.figure(2)
plt.xscale('log');plt.yscale('log');


zini=500
avalues=np.logspace(-np.log10(1+zini),0,100)
D_values=[]
D_num_eds_a = [];D_num_chi_a = []

k_eval=0.1
for a in avalues:
    D_values.append(chiCDM.scale_independent_growth_factor(1/a-1)/chiCDM.scale_independent_growth_factor(zini))
    D_num_eds_a.append(np.sqrt(standardCDM.pk_lin(k_eval*h,1/a-1)/standardCDM.pk_lin(k_eval*h,zini)))
    D_num_chi_a.append(np.sqrt(chiCDM.pk_lin(k_eval*h,1/a-1)/chiCDM.pk_lin(k_eval*h,zini)))
plt.plot(avalues,D_values, label='EdS pred')
# plt.plot(avalues,avalues*(1+zini), label='a')
plt.plot(avalues,D_num_eds_a, label='EdS num')
plt.plot(avalues,D_num_chi_a, label='Chi num')


higz=50.
lowz=40.
kk = np.logspace(-2,np.log10(1),500) # k in h/Mpc
D_num_cs = [] 
D_num_eds = [] # P(k) in (Mpc/h)**3
for k in kk:
    D_num_cs.append(np.sqrt(chiCDM.pk_lin(k*h,lowz)/chiCDM.pk_lin(k*h,higz)))
    D_num_eds.append(np.sqrt(standardCDM.pk_lin(k*h,lowz)/standardCDM.pk_lin(k*h,higz)))

# plt.plot(kk,D_num_eds, label='EdS growth')
# plt.plot(kk,D_num_cs, label='Chi growth')
# plt.axhline(y=((1+higz)/(1+lowz)), color='purple', label='prediction')

plt.xlabel(r'$a$')
plt.ylabel(r'$D(a)$')
plt.legend(loc='best')
plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/growthfactor.pdf',bbox_inches='tight')



chiCDM.empty()
standardCDM.empty()
