import numpy as np
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import functools
import os
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
kEval=0.3
zeval=1.

# technical
fullt=np.linspace(-8,1, 100)
idx_eta=np.abs(fullt - (0.)).argmin()
rtol=0.1
supprshift=5
fact=(2*np.pi)**3

parser.add_argument('-rtol', dest='rtol', type=float, default=rtol, help='Relative tolerance for ODE integration')
parser.add_argument('-fx', dest='fx', type=float, default=fx , help='Chi fraction')
parser.add_argument('-kEval', dest='kEval', type=float, default=kEval , help='k of evaluation')
parser.add_argument('-N', dest='neval', type=int, default=1000, help='vegas N evaluations')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-eds', '--EdS', action='store_true')
args = parser.parse_args()
fx=args.fx;rtol=args.rtol;kEval=args.kEval
print(f'Using fx={fx:.1e}, kEval={kEval}, rtol={rtol:.1e} and neval={args.neval}')

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

outfile='/home/fverdian/class/soundspeed-scripts/kernels-fullnum/proper-chi-oneloop-results/varykJ_atk'+str(kEval).replace('.','p')+'-fx'+str(fx).replace('.','p')+'.txt'
if os.path.exists(outfile):
    os.remove(outfile)
open(outfile, 'a').write(f'# kref, EdS result, numerical-EdS difference, PLpurecdm, PL\n')

kJlist=np.logspace(-3,1,70)

for kref in kJlist:

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
    phi_x_int=interp1d(fullt,linsol[:, 2]/linsol[:, 0],bounds_error=False,fill_value=(0.,1.))
    g_x_int=interp1d(fullt,linsol[:, 3]/linsol[:, 0],bounds_error=False,fill_value=(0.,1.))


    #--------
    # 2nd order
    #--------
    def solve_second_order(triplet, return_timedep=False):
        k1 , k2, cT = triplet

        Fc2_0=(5./7.+6/245*fx)*alphas(k1,k2,cT)+(2./7.-6/245*fx)*beta(k1,k2,cT)
        Gc2_0=(3./7-51/245*fx)*alphas(k1,k2,cT)+(4./7-96/245*fx)*beta(k1,k2,cT)

        def F2_system(w, t):
            Fc2,Gc2,Fx2,Gx2 = w
            fact=(g_c_int(t-2*np.log(k1/kref))+g_c_int(t-2*np.log(k2/kref)))
            SF2=0.5*g_c_int(t-2*np.log(k1/kref))*alpha(k1,k2,cT)+0.5*g_c_int(t-2*np.log(k2/kref))*alpha(k2,k1,cT)
            dGc2dt = -(0.5+fact)*Gc2 + 1.5 *((1-fx)*Fc2+fx*Fx2) +g_c_int(t-2*np.log(k1/kref))*g_c_int(t-2*np.log(k2/kref))*beta(k1,k2,cT)
            dFc2dt = -fact*Fc2 + Gc2 + SF2

            SF2=0.5*phi_x_int(t-2*np.log(k2/kref))*g_x_int(t-2*np.log(k1/kref))*alpha(k1,k2,cT)+0.5*phi_x_int(t-2*np.log(k1/kref))*g_x_int(t-2*np.log(k2/kref))*alpha(k2,k1,cT)
            dFx2dt = -fact*Fx2 + Gx2 + SF2
            dGx2dt = -(0.5+fact)*Gx2 + 1.5 *((1-fx)*Fc2+(fx-(k1**2+k2**2+2*k1*k2*cT)/(kref**2)*np.exp(-t))*Fx2) +g_x_int(t-2*np.log(k1/kref))*g_x_int(t-2*np.log(k2/kref))*beta(k1,k2,cT) -(k1**2+k2**2+2*k1*k2*cT)/(kref**2)*np.exp(-t-supprshift)*dFx2dt


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
            fact=(g_c_int(t-2*np.log(k/kref))+2*g_c_int(t-2*np.log(q/kref)))

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

            dFc3dt = -fact*Fc3 + Gc3 + SF3/3
            dGc3dt = -(0.5+fact)*Gc3 + 1.5*((1-fx)*Fc3+fx*Fx3) + SG3/3

            #F3_c and G3_c
            kMq = np.sqrt(k*k + q*q - 2*k*q*mu)
            cT=(k*mu-q)/kMq
            SF3x=g_x_int(t-2*np.log(q/kref))*alpha(q,kMq,cT)*ker2_k_mq[2]+phi_x_int(t-2*np.log(q/kref))*alpha(kMq,q,cT)*ker2_k_mq[3]
            SG3x=g_x_int(t-2*np.log(q/kref))*beta(kMq,q,cT)*2*ker2_k_mq[3]
            kPq = np.sqrt(k*k + q*q + 2*k*q*mu)
            cT=-(k*mu+q)/kPq
            SF3x+=g_x_int(t-2*np.log(q/kref))*alpha(q,kPq,cT)*ker2_k_q[2]+phi_x_int(t-2*np.log(q/kref))*alpha(kPq,q,cT)*ker2_k_q[3]
            SG3x+=g_x_int(t-2*np.log(q/kref))*beta(kPq,q,cT)*2*ker2_k_q[3]

            dFx3dt = -fact*Fx3 + Gx3 + SF3x/3
            dGx3dt = -(0.5+fact)*Gx3 + 1.5*((1-fx)*Fc3+(fx-(k**2)/(kref**2)*np.exp(-t))*Fx3) + SG3x/3 -(k**2)/(kref**2)*np.exp(-t-supprshift)*dFx3dt
            return [dFc3dt, dGc3dt,dFx3dt, dGx3dt]
        
        sol = odeint(F3_system, [F3(k,q,mu)-1/7*fx*DeltaF3_ennio(k,q,mu),F3(k,q,mu),0.,0.], fullt, rtol=rtol)
        if return_timedep:
            return sol[:,0]
        return sol[idx_eta,0]


    #--------
    # Loop integral
    #--------
    def P1loop_int(vars, kev, pbar=None):
        q=vars[0]
        mu=vars[1]
        F3_val=solve_F3_for_k([kev,q,mu])
        if args.verbose:
            print(f'Sampling at k={kev:.2e} q={q:.2e} mu={mu:.2f}',end='\r',flush=True)
        DF3_val=F3_val-F3(kev,q,mu)
        P13=6*q*q*DF3_val*PLc_int(q)*PLc_int(kev)
        kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
        kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
        P22=0
        if kMq>q:
            cT12 = (kev*mu-q)/kMq
            F2_val=solve_second_order([q,kMq,cT12])[0]
            DF2_2=F2_val**2-F2(kev,q,mu)**2
            P22+=2*q*q*DF2_2*PLc_int(q)*PLc_int(kMq)
        if kPq>q:
            cT12 = -(kev*mu+q)/kPq
            F2_val=solve_second_order([q,kPq,cT12])[0]
            DF2_2=F2_val**2-F2(kev,q,-mu)**2
            P22+=2*q*q*DF2_2*PLc_int(q)*PLc_int(kPq)
        return (P13+P22)*4*np.pi/fact

    #--------
    # Integrate
    #--------
    print('Starting integral!!', flush=True)

    integ = vegas.Integrator([[1.e-4, 3], [0., 1.]],mpi=True, nproc=48)
    train = integ(functools.partial(training_int, kev=kEval),neval=10000)
    start_time = time.time()
    result = integ(functools.partial(P1loop_int, kev=kEval), neval=args.neval)
    print(f'DeltaP1loop at k={kEval:.3f} is {result} (took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s). Training EdS result was {train}, which means a {result.mean/train.mean:.2f} difference', flush=True)


    open(outfile, 'a').write(f'{kref} {train.mean} {result.mean} {Pcdmonly_int(kEval)} {PLc_int(kEval)}\n')


