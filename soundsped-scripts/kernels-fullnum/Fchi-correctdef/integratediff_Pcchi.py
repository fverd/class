import numpy as np
import time
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
fullt=np.linspace(-3,1, 200)
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

def F3(k,q,mu):
    k2=k*k;q2=q*q;mu2=mu*mu
    kMq2 = k2 + q2 - 2*k*q*mu 
    kPq2 = k2 + q2 + 2*k*q*mu 

    res1 = 1/kMq2*(5/126*k2-11/108*k*q*mu+7/108*q2*mu2-1/54*k2*k2*mu2/q2+4/189*k2*k*mu2*mu/q-23/756*k2*k*mu/q+25/252*k2*mu2-2/27*k*q*mu2*mu)
    res2 = 1/kPq2*(5/126*k2+11/108*k*q*mu-7/108*q2*mu2-4/27*k2*k2*mu2/q2-53/189*k2*k*mu2*mu/q+23/756*k2*k*mu/q-121/756*k2*mu2-5/27*k*q*mu2*mu)
    return res1+res2

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


def training_int(vars,kev):
    q=vars[0]
    mu=vars[1]
    P13=6*q*q*F3(kev,q,mu)*PLc_int(q)*PLc_int(kev)
    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22=0
    if kMq>q:
        P22+=2*q*q*F2(kev,q,mu)*F2(kev,q,mu)*PLc_int(q)*PLc_int(kMq)
    if kPq>q:
        P22+=2*q*q*F2(kev,q,-mu)*F2(kev,q,-mu)*PLc_int(q)*PLc_int(kPq)
    return (P13+P22)*4*np.pi/fact

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

def hOg_an(te):
    return 1+1/(1+np.exp(2*te/3))

PLcchi_int=interp1d(kk,Pk_chi*g_an(-2*np.log(kk/kref)),fill_value='extrapolate')
PLchichi_int=interp1d(kk,Pk_chi*g_an(-2*np.log(kk/kref)**2),fill_value='extrapolate')


#--------
# 2nd order
#--------
def solve_second_order(triplet, return_timedep=False):
    k1 , k2, cT = triplet

    Fc2_0=(5./7.+6/245*fx)*alphas(k1,k2,cT)+(2./7.-6/245*fx)*beta(k1,k2,cT)
    Gc2_0=(3./7-51/245*fx)*alphas(k1,k2,cT)+(4./7-96/245*fx)*beta(k1,k2,cT)

    def F2_system(w, t):
        Fc2,Gc2,Fx2,Gx2 = w
        Fc2,Gc2,Fx2,Gx2 = w
        g_c_k1=g_c_int(t-2*np.log(k1/kref)); g_c_k2=g_c_int(t-2*np.log(k2/kref))
        g_k1=g_an(t-2*np.log(k1/kref)); g_k2=g_an(t-2*np.log(k2/kref))
        hOg_k1=hOg_an(t-2*np.log(k1/kref)); hOg_k2=hOg_an(t-2*np.log(k2/kref))
                                                           
        SF2=0.5*g_c_int(t-2*np.log(k1/kref))*alpha(k1,k2,cT)+0.5*g_c_int(t-2*np.log(k2/kref))*alpha(k2,k1,cT)
        dGc2dt = -(0.5+g_c_k1+g_c_k2)*Gc2 + 1.5 *((1-fx)*Fc2+fx*Fx2*g_k1*g_k2) +g_c_int(t-2*np.log(k1/kref))*g_c_int(t-2*np.log(k2/kref))*beta(k1,k2,cT)
        dFc2dt = -(g_c_k1+g_c_k2)*Fc2 + Gc2 + SF2

        SF2=0.5*hOg_k1*alpha(k1,k2,cT)+0.5*hOg_k2*alpha(k2,k1,cT)
        dFx2dt = -(hOg_k1+hOg_k2)*Fx2 + Gx2 + SF2
        dGx2dt = -(0.5+hOg_k1+hOg_k2)*Gx2 + 1.5 *((1-fx)*Fc2/g_k1/g_k2+(fx-(k1**2+k2**2+2*k1*k2*cT)/(kref**2)*np.exp(-t))*Fx2) +hOg_k1*hOg_k2*beta(k1,k2,cT) -(k1**2+k2**2+2*k1*k2*cT)/(kref**2)*np.exp(-t-supprshift)*dFx2dt

        return [dFc2dt, dGc2dt,dFx2dt, dGx2dt]

    sol = odeint(F2_system, [Fc2_0,Gc2_0,0.,0.], fullt, rtol=rtol)
    if return_timedep:
        return sol
    return sol[idx_eta]

#--------
# 3rd order
#--------
def DeltaF3_ennio(k,q,mu):
    x2=(k/q)**2;mu2=mu*mu
    num=4*(mu2-1)*x2*(-49*mu2+(5*mu2+22)*x2 +22)
    den=945*(x2*x2+(2-4*mu2)*x2+1)
    return num/den

def G2F2combo(k1,k2,cT):
    return [(5./7+6/245*fx)*alphas(k1,k2,cT)+(2./7-6/245*fx)*beta(k1,k2,cT),(3./7-3*17/245*fx)*alphas(k1,k2,cT)+(4./7-3*32/245*fx)*beta(k1,k2,cT),0,0]
    
def DeltaF2for3_midlim(k,q,mu):
    # This is F2(k1=k,k2=q) eh
    x2=(k/q)**2;mu2=mu*mu;x=k/q
    num=-3*(35*mu+(12*mu2+23)*x)
    den=490*x
    numG=-3*(70*mu+49*mu*x2+(52*mu2+67)*x)
    return [num/den, numG/den,0,0]

def solve_F3_for_k(triplet, return_timedep=False):
    k , q, mu = triplet

    ker2_k_mq_full=solve_second_order([k,q,-mu], return_timedep=True)
    ker2_k_q_full=solve_second_order([k,q,mu], return_timedep=True)

    def F3_system(w, t):

        Fc3,Gc3,Fx3,Gx3 = w
        g_c_k=g_c_int(t-2*np.log(k/kref));g_c_q=g_c_int(t-2*np.log(q/kref))
        g_k=g_an(t-2*np.log(k/kref));g_q=g_an(t-2*np.log(q/kref))
        hOg_k=hOg_an(t-2*np.log(k/kref));hOg_q=hOg_an(t-2*np.log(q/kref))

        idx_t_F3=np.abs(fullt - t).argmin()
        ker2_k_mq=ker2_k_mq_full[idx_t_F3]
        ker2_k_q=ker2_k_q_full[idx_t_F3]

        #F3_c and G3_c
        kMq = np.sqrt(k*k + q*q - 2*k*q*mu)
        cT=(k*mu-q)/kMq
        SF3=g_c_int(t-2*np.log(q/kref))*alpha(q,kMq,cT)*ker2_k_mq[0]+alpha(kMq,q,cT)*ker2_k_mq[1]
        SG3=g_c_int(t-2*np.log(q/kref))*beta(kMq,q,cT)*2*ker2_k_mq[1]
        kPq = np.sqrt(k*k + q*q + 2*k*q*mu)
        cT=-(k*mu+q)/kPq
        SF3+=g_c_int(t-2*np.log(q/kref))*alpha(q,kPq,cT)*ker2_k_q[0]+alpha(kPq,q,cT)*ker2_k_q[1]
        SG3+=g_c_int(t-2*np.log(q/kref))*beta(kPq,q,cT)*2*ker2_k_q[1]

        dFc3dt = -(g_c_k+2*g_c_q)*Fc3 + Gc3 + SF3/3
        dGc3dt = -(0.5+g_c_k+2*g_c_q)*Gc3 + 1.5*((1-fx)*Fc3+fx*Fx3*g_k*g_q*g_q) + SG3/3

        #F3_c and G3_c
        kMq = np.sqrt(k*k + q*q - 2*k*q*mu)
        cT=(k*mu-q)/kMq
        SF3x=hOg_q*alpha(q,kMq,cT)*ker2_k_mq[2]+alpha(kMq,q,cT)*ker2_k_mq[3]
        SG3x=hOg_q*beta(kMq,q,cT)*2*ker2_k_mq[3]
        kPq = np.sqrt(k*k + q*q + 2*k*q*mu)
        cT=-(k*mu+q)/kPq
        SF3x+=hOg_q*alpha(q,kPq,cT)*ker2_k_q[2]+alpha(kPq,q,cT)*ker2_k_q[3]
        SG3x+=hOg_q*beta(kPq,q,cT)*2*ker2_k_q[3]

        dFx3dt = -(hOg_k+2*hOg_q)*Fx3 + Gx3 + SF3x/3
        dGx3dt = -(0.5+hOg_k+2*hOg_q)*Gx3 + 1.5*((1-fx)*Fc3/g_k/g_q/g_q+(fx-(k**2)/(kref**2)*np.exp(-t))*Fx3) + SG3x/3 -(k**2)/(kref**2)*np.exp(-t-supprshift)*dFx3dt
        return [dFc3dt, dGc3dt,dFx3dt, dGx3dt]
    
    sol = odeint(F3_system, [F3(k,q,mu),F3(k,q,mu),F3(k,q,mu),F3(k,q,mu)], fullt, rtol=rtol)
    if return_timedep:
        return sol[:,0]
    return sol[idx_eta,2]


#--------
# Loop integral
#--------
def P1loop_int(vars, kev, pbar=None):
    q=vars[0]
    mu=vars[1]
    F3_val=solve_F3_for_k([kev,q,mu])
    if args.verbose:
        print(f'Sampling at k={kev:.2e} q={q:.2e} mu={mu:.2f}',end='\r',flush=True)
    P13=3*q*q*F3_val*PLcchi_int(q)*PLcchi_int(kev)-3*F3(kev,q,mu)*PLc_int(q)*PLc_int(kev)
    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22=0
    if kMq>q:
        cT12 = (kev*mu-q)/kMq
        F2_val=solve_second_order([q,kMq,cT12])[2]
        DF2_2=F2_val*F2(kev,q,mu)*PLcchi_int(q)*PLcchi_int(kMq)-PLc_int(q)*PLc_int(kMq)*F2(kev,q,mu)**2
        P22+=2*q*q*DF2_2
    if kPq>q:
        cT12 = -(kev*mu+q)/kPq
        F2_val=solve_second_order([q,kPq,cT12])[2]
        DF2_2=F2_val*F2(kev,q,-mu)*PLcchi_int(q)*PLcchi_int(kPq)-PLc_int(q)*PLc_int(kPq)*F2(kev,q,-mu)**2
        P22+=2*q*q*DF2_2
    return (P13+P22)*4*np.pi/fact

#--------
# Integrate
#--------
print('Starting integral!!', flush=True)
reslist=[]
kevList=np.logspace(np.log10(0.05),np.log10(1),50)

# outfile='/home/fverdian/class/soundspeed-scripts/kernels-fullnum/proper-chi-oneloop-results/Pcchi-kJ'+str(kref).replace('.','p')+'-fx'+str(fx).replace('.','p')+'.txt'
# if os.path.exists(outfile):
#     os.remove(outfile)
# open(outfile, 'a').write(f'# k, EdS result, numerical-EdS difference, PL_cc, PL_cchi\n')


for kEval in kevList:
    integ = vegas.Integrator([[1.e-4, 3], [0., 1.]],mpi=True, nproc=48)
    train = integ(functools.partial(training_int, kev=kEval),neval=10000)
    if args.EdS:
        print(f'EdS result is {train}', flush=True)
        continue
    start_time = time.time()
    result = integ(functools.partial(P1loop_int, kev=kEval), neval=args.neval)
    reslist.append(result.mean)
    print(f'DeltaP1loop at k={kEval:.3f} is {result} (took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s). Training EdS result was {train}, which means a {result.mean/train.mean:.2f} difference', flush=True)
    gkk=1/(1+2*(kEval/kref)**1.5)
    # open(outfile, 'a').write(f'{kEval} {train.mean} {result.mean} {PLc_int(kEval)} {PLc_int(kEval)*gkk}\n')
