import numpy as np
import time, os, sys
import functools
import argparse
import vegas
sys.path.append('/home/fverdian/class/soundspeed-scripts/kernels-fullnum/numerical-integrals')
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
kevList=np.logspace(np.log10(0.03),np.log10(0.6),30)
# kevList = [0.15]

# technical
fullt=np.linspace(-6,1, 200)
idx_eta=np.abs(fullt - (0.)).argmin()
rtol=0.1
supprshift=5
fact=(2*np.pi)**3

parser.add_argument('-rtol', dest='rtol', type=float, default=rtol, help='Relative tolerance for ODE integration')
parser.add_argument('-fx', dest='fx', type=float, default=fx , help='Chi fraction')
parser.add_argument('-ma', dest='ma', type=float, default=ma , help='Axion mass')
parser.add_argument('-N', dest='neval', type=int, default=1000, help='vegas N evaluations')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--nosave', action='store_true')
args = parser.parse_args()
fx=args.fx;rtol=args.rtol;ma=args.ma
print(f'Using fx={fx:.1e}, ma={ma}, rtol={rtol:.1e} and neval={args.neval}')

h=0.67810
kref =8.4e12 * pow(ma/h,1/2) * 0.666**0.25

if not args.nosave:
    outfile='/home/fverdian/class/soundspeed-scripts/numerical-integrals/results/P22-ma'+str(-np.log10(ma)).replace('.','p')+'-fx'+str(fx).replace('.','p')+'-N'+str(args.neval)+'.txt'
    if os.path.exists(outfile):
        os.remove(outfile)
    open(outfile, 'a').write(f'# k, dcdc22, dcdc22_EdS, dcTc22, dcTc22_EdS, dcdx22, dcdx22_EdS, dcTx22, dcTx22_EdS, dxdx22, dxdx22_EdS, PL_cc, TL_x \n')

#===============================
# LINEAR PART
#===============================

nk = NumKernels(fx=fx, kref=kref, fullt=fullt, rtol=rtol)
plclass = PLaxfromClass(fx=fx, m_a_ev=ma, zeval=zeval)
PLc_int = plclass.PLc_int

#===============================
# LOOP INTEGRAL
#===============================

def training_int(vars,kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]
    g_k=nk.g_an(-2*np.log(kev/kref));h_k=nk.h_an(-2*np.log(kev/kref))

    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22dcdc=0; P22dcTc=0; P22dcdx=0; P22dcTx=0; P22dxdx=0
    if kMq>q:
        P22dcdc+=2*q*q*(nk.F2_0(kev,q,mu)**2)*PLc_int(q)*PLc_int(kMq)
        # P22dcTc+=2*q*q*(nk.F2_0(kev,q,mu)*nk.G2_0(kev,q,mu))*nk.g_c_int(-2*np.log(kev/kref))*PLc_int(q)*PLc_int(kMq)
        P22dcTc+=2*q*q*(nk.F2_0(kev,q,mu)*nk.F2_0(kev,q,mu))*nk.g_c_int(-2*np.log(kev/kref))*PLc_int(q)*PLc_int(kMq)
        P22dcdx+=2*q*q*g_k*(nk.F2_0(kev,q,mu)**2)*PLc_int(q)*PLc_int(kMq)
        # P22dcTx+=2*q*q*h_k*(nk.F2_0(kev,q,mu)*nk.G2_0(kev,q,mu))*PLc_int(q)*PLc_int(kMq)
        P22dcTx+=2*q*q*h_k*(nk.F2_0(kev,q,mu)*nk.F2_0(kev,q,mu))*PLc_int(q)*PLc_int(kMq)
        P22dxdx+=2*q*q*g_k*g_k*(nk.F2_0(kev,q,mu)**2)*PLc_int(q)*PLc_int(kMq)
    if kPq>q:
        P22dcdc+=2*q*q*(nk.F2_0(kev,q,-mu)**2)*PLc_int(q)*PLc_int(kPq)
        # P22dcTc+=2*q*q*(nk.F2_0(kev,q,-mu)*nk.G2_0(kev,q,-mu))*nk.g_c_int(-2*np.log(kev/kref))*PLc_int(q)*PLc_int(kPq)
        P22dcTc+=2*q*q*(nk.F2_0(kev,q,-mu)*nk.F2_0(kev,q,-mu))*nk.g_c_int(-2*np.log(kev/kref))*PLc_int(q)*PLc_int(kPq)
        P22dcdx+=2*q*q*g_k*(nk.F2_0(kev,q,-mu)**2)*PLc_int(q)*PLc_int(kPq)
        # P22dcTx+=2*q*q*h_k*(nk.F2_0(kev,q,-mu)*nk.G2_0(kev,q,-mu))*PLc_int(q)*PLc_int(kPq)
        P22dcTx+=2*q*q*h_k*(nk.F2_0(kev,q,-mu)*nk.F2_0(kev,q,-mu))*PLc_int(q)*PLc_int(kPq)
        P22dxdx+=2*q*q*g_k*g_k*(nk.F2_0(kev,q,-mu)**2)*PLc_int(q)*PLc_int(kPq)
    intf=4*q*np.pi/fact
    return [intf*P22dcdc,intf*P22dcTc,intf*P22dcdx,intf*P22dcTx,intf*P22dxdx]

def Pnum_int(vars, kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]

    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22dcdc=0; P22dcTc=0; P22dcdx=0; P22dcTx=0; P22dxdx=0
    if kMq>q:
        cT12 = (kev*mu-q)/kMq
        F2_cval,G_cval,F2_xval,G2_xval=nk.solve_F2([q,kMq,cT12])
        P22dcdc+=2*q*q*(F2_cval**2)*PLc_int(q)*PLc_int(kMq)
        P22dcTc+=2*q*q*(F2_cval*G_cval)*PLc_int(q)*PLc_int(kMq)
        P22dcdx+=2*q*q*(F2_xval*nk.F2_0(kev,q,mu))*PLc_int(q)*PLc_int(kMq)
        P22dcTx+=2*q*q*(G2_xval*nk.F2_0(kev,q,mu))*PLc_int(q)*PLc_int(kMq)
        P22dxdx+=2*q*q*(F2_xval**2)*PLc_int(q)*PLc_int(kMq)

    if kPq>q:
        cT12 = -(kev*mu+q)/kPq
        F2_cval,G_cval,F2_xval,G2_xval=nk.solve_F2([q,kPq,cT12])
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
    integ = vegas.Integrator([[np.log(1.e-3), np.log(1.)], [0., 1.]],mpi=True, nproc=70)
    traindcdc22,traindcTc22,traindcdx22,traindcTx22,traindxdx22 = integ(functools.partial(training_int, kev=kEval),neval=10000)
    start_time = time.time()
    result = integ(functools.partial(Pnum_int, kev=kEval), neval=args.neval)
    resdcdc22, resdcTc22, resdcdx22, resdcTx22, resdxdx22 = result

    print(f'At k={kEval:.3f} done (took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s)', flush=True)

    if not args.nosave: open(outfile, 'a').write(f'{kEval:.5g} {resdcdc22.mean:.5g} {traindcdc22.mean:.5g} {resdcTc22.mean:.5g} {traindcTc22.mean:.5g} {resdcdx22.mean:.5g} {traindcdx22.mean:.5g} {resdcTx22.mean:.5g} {traindcTx22.mean:.5g} {resdxdx22.mean:.5g} {traindxdx22.mean:.5g} {PLc_int(kEval):.5g} {nk.g_an(-2*np.log(kEval/kref)):.5g}\n')

