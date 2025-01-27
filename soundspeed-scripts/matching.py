#!/usr/bin/env python3
# coding: utf-8
# import classy module
from classy import Class
import numpy as np
from scipy.special import hyp2f1

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
'z_max_pk':50,
}

chiCDM = Class()
# pass input parameters
chiCDM.set(common_settings)


chiCDM.set({
'omega_cdm':0.10,
'f_chi':0.01,
'acs_chi':1.,
'cs2_peak_chi':1.
})

standardCDM = Class()
standardCDM.set(common_settings)
# standardCDM.set({'Omega_cdm':1-0.0486773}) #EdS
standardCDM.set({'omega_cdm':0.11}) #LCDM

# run class
chiCDM.compute()

standardCDM.compute()


zmatch=25.




import matplotlib.pyplot as plt
from math import pi



plt.figure(2)
plt.xscale('log');plt.yscale('log');

h = chiCDM.h() # get reduced Hubble for conversions to 1/Mpc

avalues=np.logspace(-np.log10(1+zmatch),0,100)
D_values=[]
D_num_eds_a = [];D_num_chi_a = []

k_eval=0.01
for a in avalues:
    D_values.append(chiCDM.scale_independent_growth_factor(1/a-1)/chiCDM.scale_independent_growth_factor(zmatch))
    D_num_eds_a.append(np.sqrt(standardCDM.pk_lin(k_eval*h,1/a-1)/standardCDM.pk_lin(k_eval*h,zmatch)))
    D_num_chi_a.append(np.sqrt(chiCDM.pk_lin(k_eval*h,1/a-1)/chiCDM.pk_lin(k_eval*h,zmatch)))



amatch=1/26
background = chiCDM.get_background() # load background table
bk_a = 1/(background['z']+1) # read redshift
Ha=background['H [1/Mpc]']/(1.+background['z'])
cs2=1/3
k_jeans=Ha/(cs2**0.5)
beta=(k_eval/k_jeans)**2
alpha=3./2.
grexp=-3/5*0.01

# Om=0.287902
# Dz_lc = avalues * hyp2f1(1./3, 1., 11./6, (1. - 1./Om) * avalues**3)/hyp2f1(1./3, 1., 11./6, (1. - 1./Om) * amatch**3)/amatch


# plt.plot(avalues,D_values, label='LCDM pred')
# plt.plot(avalues,D_num_eds_a, label='LCDM num')
# plt.plot(avalues,Dz_lc, label='LCDM hyper')
# plt.plot(avalues,D_num_chi_a, label='Chi num')
# plt.plot(avalues,np.array(D_values)**(1+grexp), label='Chi anl')



plt.plot(avalues,np.abs(np.array(D_values)**(1+grexp)/D_num_chi_a-1), label='EdS analytic ratio')


# plt.xlim([0.1,0.15])
# plt.ylim([14,16])

plt.xlabel(r'$a$')
plt.ylabel(r'$D(a)$')
plt.legend(loc='best')
plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/matchgrowth.pdf',bbox_inches='tight')

chiCDM.empty()
standardCDM.empty()
