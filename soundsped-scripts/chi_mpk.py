#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
from scipy.optimize import root
import importlib
import classy
from scipy.special import sici
from classy import Class

common_settings = {
'omega_b':0.0223828,
'h':0.67810,
'z_reio':7.6711,
'YHe':0.25,
'perturbations_verbose':1,
'background_verbose':3,
'output':'mTk, vTk, mPk',
'gauge':'newtonian',
'P_k_max_1/Mpc':10,
'z_max_pk':1000,
'format':'class',
}

# ChatGPT says that 1 eV = 4.827e19 invMpc
chiCDM = Class()
chiCDM.set(common_settings)
aNR= 0.01
chiCDM.set({
'N_ur': 3.046,
'omega_cdm':0.10 ,
'omega_chi':0.02 ,
'm_ax':1.e-28*1.56e29,
})
print(f'm_a of 1.56e29 corresponds to {1.e-26*1.56e29}')

chiCDM.compute()
