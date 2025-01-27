import numpy as np
import time, os
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import functools
from scipy.special import sici
from classy import Class
from scipy.optimize import brentq
import argparse
import vegas
parser = argparse.ArgumentParser(description="Compute P1-loop with fx component")

#===============================
# Setup parameters
#===============================

# physical
fx=0.1
kref=0.1
zeval=1.

# technical
fullt=np.linspace(-6,1, 200)
idx_eta=np.abs(fullt - (0.)).argmin()
rtol=0.1
supprshift=5
fact=(2*np.pi)**3

parser.add_argument('-rtol', dest='rtol', type=float, default=rtol, help='Relative tolerance for ODE integration')
parser.add_argument('-fx', dest='fx', type=float, default=fx , help='Chi fraction')
parser.add_argument('-kref', dest='kref', type=float, default=kref , help='k Jeans at the time of evaluation')
parser.add_argument('-N', dest='neval', type=int, default=1000, help='vegas N evaluations')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-eds', '--EdS', action='store_true')
args = parser.parse_args()
fx=args.fx;rtol=args.rtol;kref=args.kref
print(f'Using fx={fx:.1e}, kref={kref}, rtol={rtol:.1e} and neval={args.neval}')

#===============================
# LINEAR PART
#===============================

common_settings = {
'h':0.67810,
'z_reio':7.6711,
'YHe':0.25,
'perturbations_verbose':0,
'background_verbose':0,
'output':'mTk, vTk, mPk',
'gauge':'newtonian',
'P_k_max_1/Mpc':200,
'z_max_pk':100,
'format':'class',
'N_ur': 3.046,
}

# FIND aNR by shooting
shootCDM = Class()
shootCDM.set({'h':0.67810,'z_reio':7.6711,'YHe':0.25,'omega_b':0.022,'omega_cdm':(1-fx)*0.142-0.022,'f_chi':fx,'cs2_peak_chi':1./3.,'background_verbose':0,'output':''})

def find_aNR_byshoot(log10aNR):
    shootCDM.set({
    'acs_chi':10**log10aNR,
    })
    shootCDM.compute()
    shootCDMbck = shootCDM.get_background() # load background table
    kJ_int_z=interp1d(shootCDMbck['z'],shootCDMbck['(.)kJ_chi'])
    return kJ_int_z(1.) - kref
aNR = 10**(float(brentq(find_aNR_byshoot, -5, 0)))
print(f'The value of aNR that gives kref={kref} is {aNR:.2e}')

chiCDM = Class()
chiCDM.set(common_settings)

chiCDM.set({
'omega_b':0.022,
# set total matter to omega_m=0.142
'omega_cdm':(1-fx)*0.142-0.022,
'f_chi':fx,
'acs_chi':aNR,
'cs2_peak_chi':1./3.,
# 'T_cmb':1.8,
})
chiCDM.compute()
#get the growth factor of a LCDM equivalent
pureCDM = Class()
pureCDM.set(common_settings)
pureCDM.set({
    'omega_cdm': 0.12,
    'omega_b':0.022,
})
pureCDM.compute()

kk = np.logspace(-5.,2.,500) # k in h/Mpc
Pk_chi = [];Pk_cdm = [];h = chiCDM.h()
for k in kk:
    Pk_chi.append(chiCDM.pk_cb_lin(k*h,zeval)*h**3)
    Pk_cdm.append(pureCDM.pk_lin(k*h,zeval)*h**3)
Pk_chi=np.array(Pk_chi);Pk_cdm=np.array(Pk_cdm)
Pcdmonly_int=interp1d(kk,Pk_cdm,fill_value='extrapolate')
PLc_int=interp1d(kk,Pk_chi,fill_value='extrapolate')
print('\nFinished with the linear part\n\n')
#===============================
# Kernels support definitions
#===============================

def alpha(k1,k2,cT):
    return 1+cT*k2/k1

def alphas(k1,k2,cT):
    return 1+0.5*cT*(k2/k1+k1/k2)

def beta(k1,k2,cT):
    t1=k1*k1+k2*k2+2*k1*k2*cT
    t2=k1*k2*cT
    t3=2.*k1*k1*k2*k2
    return t1*t2/t3

def F2(k,q,mu):
    k2=k*k;q2=q*q;mu2=mu*mu
    kMq2 = k2 + q2 - 2*k*q*mu 
    return (k2*(7*k*q*mu+3*q2)-10*k2*q2*mu2)/(14*q2*kMq2)
def F3_0(k,q,mu):
    x=k/q
    return (x**2 * (28 * mu**4 - 59 * mu**2 - 21 * mu**2 * x**4 + 2 * (38 * mu**4 - 22 * mu**2 + 5) * x**2 + 10)) / (126 * (x**4 + (2 - 4 * mu**2) * x**2 + 1))
def G3_0(k,q,mu):
    x=k/q
    return (-x**2 * (-4 * mu**4 + 9 * mu**2 + 7 * mu**2 * x**4 + (-20 * mu**4 + 4 * mu**2 + 2) * x**2 + 2)) / (42 * (x**4 + (2 - 4 * mu**2) * x**2 + 1))

#===============================
# Part with fx
#===============================

# this is solved fot k=1 and then I shift it y using the smmetry
def lin_system(w,t):
    dc,Tc,dx,Tx = w
    ddcdt = Tc
    dTcdt = -0.5*Tc+1.5*(1-fx)*dc+1.5*fx*dx
    ddxdt = Tx
    dTxdt = -0.5*Tx+1.5*(1-fx)*dc+1.5*(fx-(1)*np.exp(-t))*dx-np.exp(-t-supprshift)*ddxdt
    return [ddcdt, dTcdt,ddxdt, dTxdt]
t = fullt
w0 = [1,1-3/5*fx,0.,0.]
linsol = odeint(lin_system, w0, t)
g_c_int=interp1d(fullt,linsol[:,1]/linsol[:, 0],bounds_error=False,  fill_value=(5/4*np.sqrt(1-24/25*fx)-1/4,1))

def g_an(t):
    return  1 + 6 * np.exp(-t) * np.cos(np.sqrt(6) * np.exp(-t / 2)) * sici(np.sqrt(6) * np.exp(-t / 2))[1] - 3 * np.exp(-t) * np.pi * np.sin(np.sqrt(6) * np.exp(-t / 2)) + 6 * np.exp(-t) * np.sin(np.sqrt(6) * np.exp(-t / 2)) * sici(np.sqrt(6) * np.exp(-t / 2))[0]
def h_an(t):
    return 1 + 3 * np.sqrt(3 / 2) * np.exp(-3 * t / 2) * np.pi * np.cos(np.sqrt(6) * np.exp(-t / 2)) - 3 * np.exp(-t) * np.cos(np.sqrt(6) * np.exp(-t / 2))**2 + 3 * np.sqrt(6) * np.exp(-3 * t / 2) * sici(np.sqrt(6) * np.exp(-t / 2))[1] * np.sin(np.sqrt(6) * np.exp(-t / 2)) - 3 * np.sqrt(6) * np.exp(-3 * t / 2) * np.sin(np.sqrt(6) * np.exp(-t / 2)) * np.sinc(np.sqrt(6) * np.exp(-t / 2) / np.pi) - 3 * np.sqrt(6) * np.exp(-3 * t / 2) * np.cos(np.sqrt(6) * np.exp(-t / 2)) * sici(np.sqrt(6) * np.exp(-t / 2))[0]
#--------
# 2nd order
#--------
def solve_second_order(triplet, return_timedep=False):
    k1 , k2, cT = triplet

    Fc2_0=(5./7.+6/245*fx)*alphas(k1,k2,cT)+(2./7.-6/245*fx)*beta(k1,k2,cT)
    Gc2_0=(3./7-51/245*fx)*alphas(k1,k2,cT)+(4./7-96/245*fx)*beta(k1,k2,cT)

    def F2_system(w, t):
        Fc2,Gc2,Fx2,Gx2 = w
        g_c_k1=g_c_int(t-2*np.log(k1/kref)); g_c_k2=g_c_int(t-2*np.log(k2/kref))
        g_k1=g_an(t-2*np.log(k1/kref)); g_k2=g_an(t-2*np.log(k2/kref))
        h_k1=h_an(t-2*np.log(k1/kref)); h_k2=h_an(t-2*np.log(k2/kref))
                                                           
        SF2=0.5*g_c_k1*alpha(k1,k2,cT)+0.5*g_c_k2*alpha(k2,k1,cT)
        dFc2dt = -(g_c_k1+g_c_k2)*Fc2 + Gc2 + SF2
        dGc2dt = -(0.5+g_c_k1+g_c_k2)*Gc2 + 1.5 *((1-fx)*Fc2+fx*Fx2) +g_c_k1*g_c_k2*beta(k1,k2,cT)

        SF2=0.5*g_k2*h_k1*alpha(k1,k2,cT)+0.5*g_k1*h_k2*alpha(k2,k1,cT)
        dFx2dt = -(g_c_k1+g_c_k2)*Fx2 + Gx2 + SF2
        dGx2dt = -(0.5+g_c_k1+g_c_k2)*Gx2 + 1.5 *((1-fx)*Fc2+(fx-(k1**2+k2**2+2*k1*k2*cT)/(kref**2)*np.exp(-t))*Fx2) + h_k1*h_k2*beta(k1,k2,cT) -(k1**2+k2**2+2*k1*k2*cT)/(kref**2)*np.exp(-t-supprshift)*dFx2dt
        return [dFc2dt, dGc2dt,dFx2dt, dGx2dt]

    sol = odeint(F2_system, [Fc2_0,Gc2_0,0.,0.], fullt, rtol=rtol)
    if return_timedep:
        return sol
    return sol[idx_eta]

#--------
# 3rd order
#--------

def solve_F3_for_k(triplet):
    k , q, mu = triplet
    
    ker2_k_mq_full=solve_second_order([k,q,-mu], return_timedep=True)
    ker2_k_q_full=solve_second_order([k,q,mu], return_timedep=True)

    # Relevant expressions for alpha and beta couplings
    x=k/q;x2=x*x
    a_q_kMq=mu*x;a_kMq_q=x*(x-mu)/(x2-2*x*mu+1)
    b_q_kMq=x2*(mu*x-1)/2./(x2-2.*mu*x+1.)
    a_q_kPq=-mu*x;a_kPq_q=x*(x+mu)/(x2+2*x*mu+1)
    b_q_kPq=-x2*(mu*x+1)/2./(x2+2.*mu*x+1.)

    def F3_system(w, t):

        Fc3,Gc3,Fx3,Gx3 = w
        g_c_k=g_c_int(t-2*np.log(k/kref));g_c_q=g_c_int(t-2*np.log(q/kref))
        g_q=g_an(t-2*np.log(q/kref)); h_q=h_an(t-2*np.log(q/kref))
        fact=(g_c_k+2*g_c_q)

        idx_t_F3=np.abs(fullt - t).argmin()
        ker2_k_mq=ker2_k_mq_full[idx_t_F3]
        ker2_k_q=ker2_k_q_full[idx_t_F3]

        #F3_c and G3_c
        SF3=g_c_q*a_q_kMq*ker2_k_mq[0]+a_kMq_q*ker2_k_mq[1]
        SG3=g_c_q*b_q_kMq*2*ker2_k_mq[1]
        SF3+=g_c_q*a_q_kPq*ker2_k_q[0]+a_kPq_q*ker2_k_q[1]
        SG3+=g_c_q*b_q_kPq*2*ker2_k_q[1]

        dFc3dt = -fact*Fc3 + Gc3 + SF3/3
        dGc3dt = -(0.5+fact)*Gc3 + 1.5*((1-fx)*Fc3+fx*Fx3) + SG3/3

        #F3_x and G3_x
        SF3x=h_q*a_q_kMq*ker2_k_mq[2]+g_q*a_kMq_q*ker2_k_mq[3]
        SG3x=h_q*b_q_kMq*2*ker2_k_mq[3]
        SF3x+=h_q*a_q_kPq*ker2_k_q[2]+g_q*a_kPq_q*ker2_k_q[3]
        SG3x+=h_q*b_q_kPq*2*ker2_k_q[3]

        dFx3dt = -fact*Fx3 + Gx3 + SF3x/3
        dGx3dt = -(0.5+fact)*Gx3 + 1.5*((1-fx)*Fc3+(fx-(k**2)/(kref**2)*np.exp(-t))*Fx3) + SG3x/3 -(k**2)/(kref**2)*np.exp(-t-supprshift)*dFx3dt

        return [dFc3dt, dGc3dt,dFx3dt, dGx3dt]
    
    sol = odeint(F3_system, [F3_0(k,q,mu),G3_0(k,q,mu),0,0], fullt, rtol=rtol)
    return [sol[idx_eta,0],sol[idx_eta,2]]

#--------
# Loop integral
#--------
def training_int(vars,kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]
    P13=6*q*q*F3_0(kev,q,mu)*PLc_int(q)*PLc_int(kev)
    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22=0
    if kMq>q:
        P22+=2*q*q*F2(kev,q,mu)*F2(kev,q,mu)*PLc_int(q)*PLc_int(kMq)
    if kPq>q:
        P22+=2*q*q*F2(kev,q,-mu)*F2(kev,q,-mu)*PLc_int(q)*PLc_int(kPq)
    return (P13+P22)*4*q*np.pi/fact

def diff1l_int(vars, kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]
    g_k=g_an(-2*np.log(kev/kref));g_q=g_an(-2*np.log(q/kref))

    F3_cval,F3_xval=solve_F3_for_k([kev,q,mu])
    dif13cc=6*q*q*(F3_cval-F3_0(kev,q,mu))*PLc_int(q)*PLc_int(kev)
    dif13cx=3*q*q*(F3_xval-F3_0(kev,q,mu)*g_k*g_q*g_q)*PLc_int(q)*PLc_int(kev)

    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    dif22cc=0;dif22cx=0
    if kMq>q:
        cT12 = (kev*mu-q)/kMq
        F2_cval,_,F2_xval,_=solve_second_order([q,kMq,cT12])
        dif22cc+=2*q*q*(F2_cval**2-F2(kev,q,mu)**2)*PLc_int(q)*PLc_int(kMq)
        dif22cx+=2*q*q*(F2_xval*F2(kev,q,mu)-g_q*g_an(-2*np.log(kMq/kref))*F2(kev,q,mu)**2)*PLc_int(q)*PLc_int(kMq)
    if kPq>q:
        cT12 = -(kev*mu+q)/kPq
        F2_cval,_,F2_xval,_=solve_second_order([q,kPq,cT12])
        dif22cc+=2*q*q*(F2_cval**2-F2(kev,q,-mu)**2)*PLc_int(q)*PLc_int(kPq)
        dif22cx+=2*q*q*(F2_xval*F2(kev,q,-mu)-g_q*g_an(-2*np.log(kPq/kref))*F2(kev,q,-mu)**2)*PLc_int(q)*PLc_int(kPq)
    print(f'{q:.1e} {mu:.1f} -> {(dif22cx)*4*np.pi/fact:.2f} {(dif13cx)*4*np.pi/fact:.2f}')
    # The additional q below is due to logarithmic integration
    return [(dif13cc+dif22cc)*4*q**np.pi/fact,(dif13cx+dif22cx)*4*q*np.pi/fact]

#--------
# Integrate
#--------
print('Starting integral!!', flush=True)
kevList=np.logspace(np.log10(0.05),np.log10(1),50)
kevList = [0.45]
outfile='/home/fverdian/class/soundspeed-scripts/kernels-fullnum/numerical-integrals/allchi-oneloop-results/diff-kJ'+str(kref).replace('.','p')+'-fx'+str(fx).replace('.','p')+'.txt'
if os.path.exists(outfile):
    os.remove(outfile)
open(outfile, 'a').write(f'# k, EdS result, diffcc, diffcx , PL_cc, PL_cchi\n')


for kEval in kevList:
    integ = vegas.Integrator([[np.log(1.e-3), np.log(0.2)], [0., 1.]],mpi=True, nproc=48)
    train = integ(functools.partial(training_int, kev=kEval),neval=10000)
    if args.EdS:
        print(f'EdS result is {train}', flush=True)
        continue
    start_time = time.time()
    result = integ(functools.partial(diff1l_int, kev=kEval), neval=args.neval)
    rescc, rescx = result
    print(f'At k={kEval:.3f}  diffcc is {rescc} while diffcx is {rescx} (took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s). Training EdS result was {train}, which means a {rescc.mean/train.mean:.2f} difference', flush=True)
    gkk=1/(1+2*(kEval/kref)**1.5)
    open(outfile, 'a').write(f'{kEval} {train.mean} {rescc.mean} {rescx.mean} {PLc_int(kEval)} {PLc_int(kEval)*gkk}\n')
