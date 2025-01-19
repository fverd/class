#!/usr/bin/env python3
# coding: utf-8
# import classy module
from classy import Class
import numpy as np
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import sys


## START WITH g AND h 
def gh_system(w, t):
    g,h = w
    dgdt = h - g
    dhdt = 3/2 *(1-h)-3/2*np.exp(-t)*g
    return [dgdt, dhdt]
# Initial conditions
w0 = [0.,0.]
t = np.linspace(-10,30, 100)

# Solve the differential equation
sol = odeint(gh_system, w0, t)
g_int = interp1d(t, sol[:, 0])
h_int = interp1d(t, sol[:, 1])


# NOW TURN TO F2 chi part

t = np.linspace(-8,15, 100)

def alpha(k1,k2,cT):
    return 1+0.5*cT*(k2/k1+k1/k2)

def beta(k1,k2,cT):
    t1=k1*k1+k2*k2+2*k1*k2*cT
    t2=k1*k2*cT
    t3=2.*k1*k1*k2*k2
    return t1*t2/t3

k1=0.1
k2=0.2
cT=0.5
F2a_0=5./7.*alpha(k1,k2,cT)+2./7.*beta(k1,k2,cT)
F2b_0=3./7.*alpha(k1,k2,cT)+4./7.*beta(k1,k2,cT)

k=np.sqrt(k1*k1+k2*k2+k1*k2*cT)

def Fchi_system(w, t):
    F_chi,G_chi = w
    dF_chidt = -2*F_chi + G_chi + g_int(t-2*np.log(k1/k))* g_int(t-2*np.log(k2/k))*alpha(k1,k2,cT)
    dG_chidt = -2*G_chi - 0.5* G_chi + 1.5*F2a_0-1.5*np.exp(-t)* F_chi + h_int(t-2*np.log(k1/k))* g_int(t-2*np.log(k2/k))*beta(k1,k2,cT)
    return [dF_chidt, dG_chidt]


w0 = [0,0]
t = np.linspace(-7,12, 100)
sol = odeint(Fchi_system, w0, t)
plt.figure(figsize=(5,3), dpi=130)
plt.plot(t, sol[:, 0], linestyle='--', label=r'$F_\chi^{(2)}$')
plt.plot(t, sol[:, 1], linestyle='--',label=r'$G_\chi^{(2)}$')

plt.axhline(y=F2a_0,color='blue', linewidth=1., label=r'$F_c^{(2)}$ (EdS)')
plt.axhline(y=F2b_0,color='red', linewidth=1., label=r'$G_c^{(2)}$(EdS)')

Fchi_int = interp1d(t, sol[:, 0])


# # NOW TURN TO F2 cdm part

def s_system(s, t):
    dsdt = 2.5*(g_int(t)-s-1)
    return dsdt
t = np.linspace(-10,20, 100)
sol = odeint(s_system, -1., t)

s_int = interp1d(t, sol[:, 0])
# plt.plot(t, sol[:,0], label='s1')
t = np.linspace(-7,10, 100)

def Fcb1_system(w, t):
    F_c1,G_c1 = w
    s1_symm=0.5*(s_int(t-2*np.log(k1/k))+s_int(t-2*np.log(k2/k)))
    dF_c1dt = -2*F_c1 + G_c1 - 9./35.*s1_symm*alpha(k1,k2,cT)- 12./35.*s1_symm*beta(k1,k2,cT)

    dG_c1dt = -2.5*G_c1 + 1.5*(F_c1+Fchi_int(t)) -(15/14*alpha(k1,k2,cT)+3/7*beta(k1,k2,cT))
    dG_c1dt -= 18/35*s1_symm*(alpha(k1,k2,cT)-beta(k1,k2,cT))
    return [dF_c1dt, dG_c1dt]


w0 = [6/245*(alpha(k1,k2,cT)-beta(k1,k2,cT)),
      -3/245*(17*alpha(k1,k2,cT)+32*beta(k1,k2,cT))]

sol = odeint(Fcb1_system, w0, t)
plt.plot(t, sol[:, 0], label=r'$\Delta F_c^{(2)}$')
plt.plot(t, sol[:, 1], label=r'$\Delta G_c^{(2)}$')


# plt.plot(t, sol[:, 0], label=r'$F^{(2)}_{a,1}$')
# plt.plot(t, sol[:, 1], label=r'$F^{(2)}_{b,1}$')

# ## Sum the two kernel contributions
# fx=0.2
# F_a=F2a_0+fx*sol[:, 0]
# F_b=F2b_0+fx*sol[:, 1]

# # plt.plot(t, F_a, label='F2a')
# # plt.plot(t, F_b, label='F2b')
plt.axhline(y=0,color='k', linewidth=0.4)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel(r'$\tilde \eta$')
plt.xlim([-7,10])
# plt.grid()
plt.savefig('/home/fverdian/class/soundspeed-scripts/figure/F2-odes.pdf',bbox_inches='tight')

