#!/usr/bin/env python3
# coding: utf-8
# import classy module
from classy import Class
from scipy.interpolate import interp1d
import numpy as np
_M_p_in_invMpc=3.7986e56
_eV_in_invMpc=1.5637e29

# Aggiungo una nuova specie con eq di stato p=w \rho
#la specie la chiamo as (added specie)
chiCDM = Class()
# pass input parameters
chiCDM.set({
'omega_b':0.0223828,
'h':0.67810,
'z_reio':7.6711,
'YHe':0.25,
'a_ini_over_a_today_default':5.e-9,
'N_ur': 3.05,
'acs_chi':1.e-3,
'cs2_peak_chi':1./3.

})

chiCDM.set({
'omega_chi':0.01,
'background_verbose':3,
'omega_cdm':0.1,
})
# run class
chiCDM.compute()
h = chiCDM.h() # get reduced Hubble for conversions to 1/Mpc

amatch=1/26.
background = chiCDM.get_background() # load background table
bk_a = 1/(background['z']+1) # read redshift
background_rho_b=background['(.)rho_b']
background_rho_g=background['(.)rho_g']
background_rho_cdm=background['(.)rho_cdm']
background_rho_chi=background['(.)rho_chi']
background_kJ_chi=background['(.)kJ_chi']
print(background_kJ_chi)
acs=8.e-4
cs2=np.ones_like(bk_a)*1/3
select=len(cs2[bk_a > acs])
cs2[-select:] *= ((acs/bk_a)[bk_a > acs])**2


Ha=background['H [1/Mpc]']/(1.+background['z'])

Om=0.287902
k_jeans=np.sqrt(3/2)*Ha/h/(np.sqrt(cs2))

aH_at_a = interp1d(bk_a,bk_a*background['H [1/Mpc]'])

myexp_k_jeans=0.020*pow(bk_a/1.e-2,1/2)

import matplotlib.pyplot as plt
plt.figure(figsize=(3.5,3), dpi=150)

plt.loglog(bk_a,k_jeans,'darkorange',label=r'$k_{Jeans}$')
# plt.loglog(bk_a,background_kJ_chi,'red',label=r'my $k_{Jeans}$',linewidth='0.6')

plt.loglog(bk_a,aH_at_a(bk_a)/h,'blue',label=r'$\mathcal{H}$')
plt.axhline(y=0.157,color='k', linestyle='--',linewidth='1.5', label=r'$k$')

# plt.axvline(x=acs,color='gray', linestyle='-',linewidth='0.6', label=r'$a_*$')

plt.xlim([1.e-2,1])
plt.ylim([None,1.])
plt.legend(loc='lower left')
plt.xlabel(r'$a$')
plt.ylabel(r'$k [h/\mathrm{Mpc}]$')
plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/kjeans.pdf',bbox_inches='tight')

k_jeans=np.sqrt(3/2)*aH_at_a(1.e-2)/h/((1/3)**0.5)
print(f'k_jeans at 1.e-2={k_jeans}')

chiCDM.empty()
