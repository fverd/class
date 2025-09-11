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

# kevList=np.logspace(np.log10(0.05),np.log10(0.5),30)
kevList=np.logspace(np.log10(0.05),np.log10(0.6),60)
kevList=[0.02,0.03,0.04]

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
    outfile='/home/fverdian/class/soundspeed-scripts/numerical-integrals/quantumpressure/P22-wqp-NOQPtest_3points_ma'+str(-np.log10(ma)).replace('.','p')+'-fx'+str(fx).replace('.','p')+'-N'+str(args.neval)+'-p'+str(p)+'.txt'
    if os.path.exists(outfile):
        os.remove(outfile)
    open(outfile, 'a').write(f'# k, dcdc22, dcdc22_EdS, dcTc22, dcTc22_EdS, dcdx22, dcdx22_EdS, dcTx22, dcTx22_EdS, dxdx22, dxdx22_EdS, PL_cc, TL_x \n')

#===============================
# LINEAR PART
#===============================

plclass = PLaxfromClass(fx=fx, m_a_ev=ma, zeval=zeval)
PLc_int = plclass.PLc_int
nk = NumKernels(fx=fx, kref=kref, fullt=fullt, rtol=rtol, p=p, g_class_k=plclass.g_class)

#===============================
# LOOP INTEGRAL
#===============================

def training_int(vars,kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]
    g_k=nk.g_num(-(2+p)*np.log(kev/kref));h_k=nk.h_num(-(2+p)*np.log(kev/kref))

    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22dcdc=0; P22dcTc=0; P22dcdx=0; P22dcTx=0; P22dxdx=0
    if kMq>q:
        P22dcdc+=2*q*q*(nk.F2_0(kev,q,mu)**2)*PLc_int(q)*PLc_int(kMq)
        P22dcTc+=2*q*q*(nk.F2_0(kev,q,mu)*nk.G2_0(kev,q,mu))*nk.g_c_int(-(2+p)*np.log(kev/kref))*PLc_int(q)*PLc_int(kMq)
        P22dcdx+=2*q*q*g_k*(nk.F2_0(kev,q,mu)**2)*PLc_int(q)*PLc_int(kMq)
        P22dcTx+=2*q*q*h_k*(nk.F2_0(kev,q,mu)*nk.G2_0(kev,q,mu))*PLc_int(q)*PLc_int(kMq)
        P22dxdx+=2*q*q*g_k*g_k*(nk.F2_0(kev,q,mu)**2)*PLc_int(q)*PLc_int(kMq)
    if kPq>q:
        P22dcdc+=2*q*q*(nk.F2_0(kev,q,-mu)**2)*PLc_int(q)*PLc_int(kPq)
        P22dcTc+=2*q*q*(nk.F2_0(kev,q,-mu)*nk.G2_0(kev,q,-mu))*nk.g_c_int(-(2+p)*np.log(kev/kref))*PLc_int(q)*PLc_int(kPq)
        P22dcdx+=2*q*q*g_k*(nk.F2_0(kev,q,-mu)**2)*PLc_int(q)*PLc_int(kPq)
        P22dcTx+=2*q*q*h_k*(nk.F2_0(kev,q,-mu)*nk.G2_0(kev,q,-mu))*PLc_int(q)*PLc_int(kPq)
        P22dxdx+=2*q*q*g_k*g_k*(nk.F2_0(kev,q,-mu)**2)*PLc_int(q)*PLc_int(kPq)
    intf=4*q*np.pi/fact
    return [intf*P22dcdc,intf*P22dcTc,intf*P22dcdx,intf*P22dcTx,intf*P22dxdx]


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

def Pnum_int(vars, kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]

    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22dcdc=0; P22dcTc=0; P22dcdx=0; P22dcTx=0; P22dxdx=0
    if kMq>q:
        cT12 = (kev*mu-q)/kMq
        F2_cval,G_cval,F2_xval,G2_xval=solve_F2_wqp([q,kMq,cT12])
        P22dcdc+=2*q*q*(F2_cval**2)*PLc_int(q)*PLc_int(kMq)
        P22dcTc+=2*q*q*(F2_cval*G_cval)*PLc_int(q)*PLc_int(kMq)
        P22dcdx+=2*q*q*(F2_xval*nk.F2_0(kev,q,mu))*PLc_int(q)*PLc_int(kMq)
        P22dcTx+=2*q*q*(G2_xval*nk.F2_0(kev,q,mu))*PLc_int(q)*PLc_int(kMq)
        P22dxdx+=2*q*q*(F2_xval**2)*PLc_int(q)*PLc_int(kMq)

    if kPq>q:
        cT12 = -(kev*mu+q)/kPq
        F2_cval,G_cval,F2_xval,G2_xval=solve_F2_wqp([q,kPq,cT12])
        P22dcdc+=2*q*q*(F2_cval**2)*PLc_int(q)*PLc_int(kPq)
        P22dcTc+=2*q*q*(F2_cval*G_cval)*PLc_int(q)*PLc_int(kPq)
        P22dcdx+=2*q*q*(F2_xval*nk.F2_0(kev,q,-mu))*PLc_int(q)*PLc_int(kPq)
        P22dcTx+=2*q*q*(G2_xval*nk.F2_0(kev,q,-mu))*PLc_int(q)*PLc_int(kPq)
        P22dxdx+=2*q*q*(F2_xval**2)*PLc_int(q)*PLc_int(kPq)

    # The additional q below is due to logarithmic integration
    intf=4*q*np.pi/fact
    return [intf*P22dcdc,intf*P22dcTc,intf*P22dcdx,intf*P22dcTx,intf*P22dxdx]

#--------
# Integrate
#--------
print('Starting integral')
for kEval in kevList:
    integ = vegas.Integrator([[np.log(1.e-3), np.log(5.)], [0., 1.]],mpi=True, nproc=70)
    traindcdc22,traindcTc22,traindcdx22,traindcTx22,traindxdx22 = integ(functools.partial(training_int, kev=kEval),neval=40000)
    start_time = time.time()
    result = integ(functools.partial(Pnum_int, kev=kEval), neval=args.neval)
    resdcdc22, resdcTc22, resdcdx22, resdcTx22, resdxdx22 = result

    print(f'At k={kEval:.3f} done (took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s). Val dcdx: num {resdcdx22} EdS {traindcdx22}', flush=True)

    if not args.nosave: open(outfile, 'a').write(f'{kEval:.5g} {resdcdc22.mean:.5g} {traindcdc22.mean:.5g} {resdcTc22.mean:.5g} {traindcTc22.mean:.5g} {resdcdx22.mean:.5g} {traindcdx22.mean:.5g} {resdcTx22.mean:.5g} {traindcTx22.mean:.5g} {resdxdx22.mean:.5g} {traindxdx22.mean:.5g} {PLc_int(kEval):.5g} {nk.g_num(-(2+p)*np.log(kEval/kref)):.5g}\n')

