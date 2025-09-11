import numpy as np
import time, os, sys
import functools
import argparse
import vegas
from scipy.integrate import odeint


sys.path.append('/home/fverdian/class/soundspeed-scripts/numerical-integrals')
from num_kernels import NumKernels, PLaxfromClass

parser = argparse.ArgumentParser(description="Compute P1-loop with fx component")

#===============================
# Compute P_{c,chi} one-loop terms
# It seems that I am computing also the full P_{c,c}, but in reality the EdS integral here
# is the P_{c,chi} one with EdS kernels, to compare the full result to
#===============================

# physical
fx=0.1
ma=1.e-27
zeval=0.5

p=2

# kevList=np.logspace(np.log10(0.05),np.log10(0.5),30)
kevList=np.logspace(np.log10(0.05),np.log10(0.6),60)
kevList=[0.02,0.03,0.04]

# kevList = [0.44011]

# technical
fullt=np.linspace(-6,1, 200)
idx_eta=np.abs(fullt - (0.)).argmin()
rtol=0.1
supprshift=5
fact=(2*np.pi)**3

parser.add_argument('-rtol', dest='rtol', type=float, default=rtol, help='Relative tolerance for ODE integration')
parser.add_argument('-fx', dest='fx', type=float, default=fx , help='Chi fraction')
parser.add_argument('-ma', dest='ma', type=float, default=ma , help='Axion mass')
parser.add_argument('-p', '--p', type=int, default=2)
parser.add_argument('-N', dest='neval', type=int, default=1000, help='vegas N evaluations')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--nosave', action='store_true')
args = parser.parse_args()
fx=args.fx;rtol=args.rtol;ma=args.ma;p=args.p
print(f'Using fx={fx:.1e}, ma={ma}, rtol={rtol:.1e} and neval={args.neval}')

h=0.67810
kref =8.4e12 * pow(ma/h,1/2) * 0.666**0.25

if not args.nosave:
    outfile='/home/fverdian/class/soundspeed-scripts/numerical-integrals/quantumpressure/P13-wqp-NOQPtest_3points_ma'+str(-np.log10(ma)).replace('.','p')+'-fx'+str(fx).replace('.','p')+'-N'+str(args.neval)+'-p'+str(p)+'.txt'
    if os.path.exists(outfile):
        os.remove(outfile)
    open(outfile, 'a').write(f'# k, dcdc13, dcdc13_EdS, dcTc13, dcTc13_EdS, dcdx13, dcdx13_EdS, dcTx13, dcTx13_EdS, dxdx13, dxdx13_EdS, PL_cc, TL_x, dcdx13_IR \n')

#===============================
# LINEAR PART
#===============================

plclass = PLaxfromClass(fx=fx, m_a_ev=ma, zeval=zeval)
PLc_int = plclass.PLc_int
nk = NumKernels(fx=fx, kref=kref, fullt=fullt, rtol=rtol, p=p, g_class_k=plclass.g_class)

#===============================
# LOOP INTEGRAL
#===============================
def F3_IR(k,q,mu):
    return -1/6*mu**2 * (k/q)**2

def training_int(vars,kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]
    g_k=nk.g_num(-(2+p)*np.log(kev/kref));h_k=nk.h_num(-(2+p)*np.log(kev/kref))
    # g_k=plclass.g_class(kev);h_k=plclass.g_class(kev)/nk.g_an(-(2+p)*np.log(kev/kref))*nk.h_an(-(2+p)*np.log(kev/kref))

    P13dcdc=6*q*q*(nk.F3_0(kev,q,mu))*PLc_int(q)*PLc_int(kev)
    P13dcTc=3*q*q*(nk.F3_0(kev,q,mu)+ nk.G3_0(kev,q,mu))*nk.g_c_int(-(2+p)*np.log(kev/kref))*PLc_int(q)*PLc_int(kev)
    # P13dcTc=6*q*q*nk.g_c_int(-4*np.log(kev/kref))*nk.F3_0(kev,q,mu)*PLc_int(q)*PLc_int(kev)
    P13dcdx=6*q*q*g_k*(nk.F3_0(kev,q,mu))*PLc_int(q)*PLc_int(kev)
    P13dcTx=3*q*q*h_k*(nk.F3_0(kev,q,mu) + nk.G3_0(kev,q,mu))*PLc_int(q)*PLc_int(kev)
    # P13dcTx=6*q*q*h_k*nk.F3_0(kev,q,mu)*PLc_int(q)*PLc_int(kev)
    P13dxdx=6*q*q*g_k*g_k*(nk.F3_0(kev,q,mu))*PLc_int(q)*PLc_int(kev)

    P13dcdx_IR=6*q*q*g_k*(F3_IR(kev,q,mu))*PLc_int(q)*PLc_int(kev)

    intf=4*q*np.pi/fact
    return [intf*P13dcdc,intf*P13dcTc,intf*P13dcdx,intf*P13dcTx,intf*P13dxdx,intf*P13dcdx_IR]

def solve_F2_wqp( triplet, return_timedep=False):
    k1 , k2, cT = triplet
    kref = nk.kref
    fx=nk.fx
    p=nk.p
    Fc2_0=(5./7.+6/245*fx)*nk.alphas(k1,k2,cT)+(2./7.-6/245*fx)*nk.beta(k1,k2,cT)
    Gc2_0=(3./7-51/245*fx)*nk.alphas(k1,k2,cT)+(4./7-96/245*fx)*nk.beta(k1,k2,cT)

    def F2_system(w, t):
        Fc2,Gc2,Fx2,Gx2 = w
        g_c_k1=nk.g_c_int(t-(2+p)*np.log(k1/kref)); g_c_k2=nk.g_c_int(t-(2+p)*np.log(k2/kref))
        g_k1=nk.g_num(t-(2+p)*np.log(k1/kref)); g_k2=nk.g_num(t-(2+p)*np.log(k2/kref))
        h_k1=nk.h_num(t-(2+p)*np.log(k1/kref)); h_k2=nk.h_num(t-(2+p)*np.log(k2/kref))
                                                        
        SF2=0.5*g_c_k1*nk.alpha(k1,k2,cT)+0.5*g_c_k2*nk.alpha(k2,k1,cT)
        dFc2dt = -(g_c_k1+g_c_k2)*Fc2 + Gc2 + SF2
        dGc2dt = -(0.5+g_c_k1+g_c_k2)*Gc2 + 1.5 *((1-fx)*Fc2+fx*Fx2) +g_c_k1*g_c_k2*nk.beta(k1,k2,cT)

        SF2=0.5*g_k2*h_k1*nk.alpha(k1,k2,cT)+0.5*g_k1*h_k2*nk.alpha(k2,k1,cT)
        dFx2dt = -(g_c_k1+g_c_k2)*Fx2 + Gx2 + SF2
        dGx2dt = -(0.5+g_c_k1+g_c_k2)*Gx2 + 1.5 *((1-fx)*Fc2+(fx-(k1**2+k2**2+2*k1*k2*cT)**(2)/(kref**(4))*np.exp(-t))*Fx2) + h_k1*h_k2*nk.beta(k1,k2,cT)  -(k1**2+k2**2+2*k1*k2*cT)**(0.5*p+1)/(kref**(2+p))*np.exp(-t-nk.supprshift)*dFx2dt

        # dGx2dt += 1.5*(k1**2+k2**2+2*k1*k2*cT)**(2)/(kref**(4))*np.exp(-t) * 1/4*(1+(k1**2 +k2**2)/(k1**2+k2**2+2*k1*k2*cT))*g_k1*g_k2

        return [dFc2dt, dGc2dt,dFx2dt, dGx2dt]

    sol = odeint(F2_system, [Fc2_0,Gc2_0,0.,0.], nk.fullt, rtol=nk.rtol)
    if return_timedep:
        return sol
    return sol[nk.idx_eta]

def solve_F3_wqp(triplet, return_timedep=False):
    k , q, mu = triplet
    kref = nk.kref
    fx=nk.fx     
    p=nk.p

    ker2_k_mq_full=solve_F2_wqp([k,q,-mu], return_timedep=True)
    ker2_k_q_full=solve_F2_wqp([k,q,mu], return_timedep=True)

    # Relevant expressions for alpha and beta couplings
    x=k/q;x2=x*x
    a_q_kMq=mu*x;a_kMq_q=x*(x-mu)/(x2-2*x*mu+1)
    b_q_kMq=x2*(mu*x-1)/2./(x2-2.*mu*x+1.)
    a_q_kPq=-mu*x;a_kPq_q=x*(x+mu)/(x2+2*x*mu+1)
    b_q_kPq=-x2*(mu*x+1)/2./(x2+2.*mu*x+1.)
    
    def F3_system(w, t):

        Fc3,Gc3,Fx3,Gx3 = w
        g_c_k=nk.g_c_int(t-(2+p)*np.log(k/kref));g_c_q=nk.g_c_int(t-(2+p)*np.log(q/kref))
        g_q=nk.g_num(t-(2+p)*np.log(q/kref)); h_q=nk.h_num(t-(2+p)*np.log(q/kref))
        fact=(g_c_k+2*g_c_q)

        idx_t_F3=np.abs(nk.fullt - t).argmin()
        ker2_k_mq=ker2_k_mq_full[idx_t_F3]
        ker2_k_q=ker2_k_q_full[idx_t_F3]
        # ker2_k_mq=np.array([ker2_k_mq_int(t) for ker2_k_mq_int in ker2_k_mq_int])
        # ker2_k_q=np.array([ker2_k_q_int(t) for ker2_k_q_int in ker2_k_q_int])

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
        dGx3dt = -(0.5+fact)*Gx3 + 1.5*((1-fx)*Fc3+(fx-(k**(2+p))/(kref**(2+p))*np.exp(-t))*Fx3) + SG3x/3 -(k**2)/(kref**2)*np.exp(-t-nk.supprshift)*dFx3dt

        # dGx3dt += -1.5*(k/kref)**(4)*np.exp(-t) * 1/8*(8/3*(1/x2+1))*g_q*g_q*nk.g_num(t-(2+p)*np.log(k/kref))

        return [dFc3dt, dGc3dt,dFx3dt, dGx3dt]
    tsol = nk.fullt[:nk.idx_eta+2]
    sol = odeint(F3_system, [nk.F3_0(k,q,mu),nk.G3_0(k,q,mu),0,0], tsol, rtol=nk.rtol)
    if return_timedep:
        return tsol, sol
    # print(sol[nk.idx_eta], triplet[1:])
    return sol[nk.idx_eta]


def Pnum_int(vars, kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]
    g_k=nk.g_num(-(2+p)*np.log(kev/kref));h_k=nk.h_num(-(2+p)*np.log(kev/kref))
    # g_k=plclass.g_class(kev);g_q=plclass.g_class(q)

    F3_cval, G3_cval, F3_xval, G3_xval=solve_F3_wqp([kev,q,mu])
    P13dcdc=6*q*q*(F3_cval)*PLc_int(q)*PLc_int(kev)
    P13dcTc=3*q*q*(F3_cval*nk.g_c_int(-(2+p)*np.log(kev/kref))+G3_cval)*PLc_int(q)*PLc_int(kev)
    P13dcdx=3*q*q*(nk.F3_0(kev,q,mu)*g_k+F3_xval)*PLc_int(q)*PLc_int(kev)
    P13dcTx=3*q*q*(nk.F3_0(kev,q,mu)*h_k+G3_xval)*PLc_int(q)*PLc_int(kev)
    P13dxdx=6*q*q*g_k*(F3_xval)*PLc_int(q)*PLc_int(kev)

    # The additional q below is due to logarithmic integration
    intf=4*q*np.pi/fact
    return [intf*P13dcdc,intf*P13dcTc,intf*P13dcdx,intf*P13dcTx,intf*P13dxdx]

#--------
# Integrate
#--------
print('Starting integral')
for kEval in kevList:
    integ = vegas.Integrator([[np.log(1.e-3), np.log(5.)], [0., 1.]],mpi=True, nproc=70)
    traindcdc13,traindcTc13,traindcdx13,traindcTx13,traindxdx13,traindcdx13_IR = integ(functools.partial(training_int, kev=kEval),neval=40000)
    start_time = time.time()
    result = integ(functools.partial(Pnum_int, kev=kEval), neval=args.neval)
    resdcdc13, resdcTc13, resdcdx13, resdcTx13, resdxdx13= result

    print(f'At k={kEval:.3f} done (took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s). Val dcdx: num {resdcdx13} EdS {traindcdx13}', flush=True)
    print(f'num {resdcdx13.mean:.5g}, num {traindcdx13.mean:.5g}')
    if not args.nosave: open(outfile, 'a').write(f'{kEval:.5g} {resdcdc13.mean:.5g} {traindcdc13.mean:.5g} {resdcTc13.mean:.5g} {traindcTc13.mean:.5g} {resdcdx13.mean:.5g} {traindcdx13.mean:.5g} {resdcTx13.mean:.5g} {traindcTx13.mean:.5g} {resdxdx13.mean:.5g} {traindxdx13.mean:.5g} {PLc_int(kEval):.5g} {nk.g_num(-(2+p)*np.log(kEval/kref)):.5g} {traindcdx13_IR.mean:.5g}\n')

