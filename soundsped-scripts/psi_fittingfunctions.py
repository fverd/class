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
'omega_chi':0.01,
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


a_eval=np.logspace(-2.5,0,30)
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

CLASS_ratio=d_chi_z/d_cdm_z

#Use kJean evaluated well in MD
a_star=1.e-2
k=chi_tk_z[1]['k (h/Mpc)'][kidx]
kJeans_func = interp1d(bk_a,background_kJ_chi)
kJ0=kJeans_func(a_star)
print(f'kJ0={kJ0}')

# Define the system of first order differential equations
def system(w, t):
    g,h = w
    dgdt = h - g
    dhdt = 1.5*(1-h) - 1.5*np.exp(-t)*g
    return [dgdt, dhdt]
# Initial conditions
w0 = [0.,0.]
t = np.linspace(-6, 20, 1000)

# Solve the differential equation
sol = odeint(system, w0, t)


# Plot the solution
eta_star=np.log(a_star)
print(f'eta star = {eta_star}')
print(f'logratio = {2*np.log(k/kJ0)}')
# if (np.log(k/kJ0)<5):
#     print('WARNING, start not ')
#t is tilde eta, ci ho perso una giornata eh
eta=t+eta_star+2*np.log(k/kJ0)
a=np.exp(eta)

g_int = interp1d(a,sol[:, 0])

# plt.plot(a_eval,CLASS_ratio,'r-o',label=r'CLASS $\delta_\chi / \delta_c$', markersize=3.)
# plt.plot(a, sol[:, 0], 'b', label='$g(a)$')

# Create the figure and the subplots
# fig, axs = plt.subplots(2, 1, sharex=True,gridspec_kw={'height_ratios': [4,1]}, figsize=(5,4), dpi=150)

# # Plot something in the upper panel
# axs[0].plot(a_eval,CLASS_ratio,'r-o',label=r'CLASS $\delta_\chi / \delta_c$', markersize=3.)
# axs[0].plot(a_eval, g_int(a_eval), 'b', label='$g(a)$')
# # Plot CLASS_ratio/sol[:, 0] in the lower panel
# axs[1].plot(a_eval,g_int(a_eval)/CLASS_ratio-1, 'maroon', label='Relative difference')

# axs[0].legend(loc='best')
# axs[0].set_xlim([1.e-2,1])
# axs[0].set_xscale('log')
# axs[0].set_xlabel('a')
# axs[0].grid()
# axs[1].legend(loc='best')
# axs[1].set_xlabel('a')
# axs[1].set_yscale('log')

# plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/g-ode_ratios.pdf',bbox_inches='tight')


plt.figure(figsize=(4,3), dpi=150)

plt.plot(a_eval,CLASS_ratio,'g-o',label=r'$\delta_\chi / \delta_c$ ($k=0.35$)', markersize=3.)
plt.axhline(y=1, color='k')
plt.axhline(y=0, color='k')
kidx=60
print('Evaluating at k=',chi_tk_z[1]['k (h/Mpc)'][kidx])
# print(tk['d_chi'].shape)

d_chi_z=-np.array([d['d_chi'][kidx] for d in chi_tk_z])
d_cdm_z=-np.array([d['d_cdm'][kidx] for d in chi_tk_z])

CLASS_ratio=d_chi_z/d_cdm_z
plt.plot(a_eval,CLASS_ratio,'r-o',label=r'$\delta_\chi / \delta_c$ ($k=0.18$)', markersize=3.)

kidx=40
print('Evaluating at k=',chi_tk_z[1]['k (h/Mpc)'][kidx])
# print(tk['d_chi'].shape)

d_chi_z=-np.array([d['d_chi'][kidx] for d in chi_tk_z])
d_cdm_z=-np.array([d['d_cdm'][kidx] for d in chi_tk_z])

CLASS_ratio=d_chi_z/d_cdm_z
plt.plot(a_eval,CLASS_ratio,'b-o',label=r'$\delta_\chi / \delta_c$ ($k=0.09$)', markersize=3.)


# plt.legend(loc='best')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlim([1.e-2,1])
plt.xscale('log')
plt.xlabel('a')
# plt.grid()
# axs[1].legend(loc='best')
# axs[1].set_xlabel('a')
# axs[1].set_yscale('log')

plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/transferratio.pdf',bbox_inches='tight')




chiCDM.empty()
standardCDM.empty()
