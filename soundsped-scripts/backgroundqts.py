#!/usr/bin/env python3
# coding: utf-8
# import classy module
from classy import Class
from scipy.interpolate import interp1d

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
'a_ini_over_a_today_default':1.e-16,
'N_ur': 3.05,

})

chiCDM.set({
'f_chi':0.00,
'background_verbose':3,
'omega_cdm':0.10,
})
# run class
chiCDM.compute()


background = chiCDM.get_background() # load background table
background_a = 1/(background['z']+1) # read redshift
background_rho_b=background['(.)rho_b']
background_rho_g=background['(.)rho_g']
background_rho_cdm=background['(.)rho_cdm']
background_rho_chi=background['(.)rho_chi']
background_cs2_chi=background['(.)cs2_chi']

#background_k_jeans=background['H [1/Mpc]']/(1.+background['z'])/(background_cs2_chi**0.5)

aH_at_a = interp1d(background_a,background_a*background['H [1/Mpc]'])

print(background_rho_chi.shape)
print(background_rho_cdm[0])

import matplotlib.pyplot as plt
plt.xlabel(r'$a $')
plt.loglog(background_a,background_rho_cdm,label=r'CDM')
plt.loglog(background_a,background_rho_chi,label=r'$\chi$')
plt.loglog(background_a,background_rho_b,label=r'baryons')
plt.loglog(background_a,background_rho_g,label=r'$\gamma$')

# plt.loglog(background_a,background_rho_chi/background_rho_cdm,label=r'$\chi$-CDM ratio')
plt.legend(loc='best')
plt.xlabel(r'$\log(a)$')
plt.ylabel(r'$[\mathrm{Mpc}]$')
plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/soundspeed-bkg.pdf',bbox_inches='tight')

chiCDM.empty()
