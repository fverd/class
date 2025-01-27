import numpy as np
import time, os
import functools
import argparse
import vegas
from num_kernels import NumKernels, PLfromClass

parser = argparse.ArgumentParser(description="Compute P1-loop with fx component")

#===============================
# Setup parameters
#===============================

# physical
fx=0.1
kref=0.1
# zeval=1. is somewhat hardcoded

kevList=np.logspace(np.log10(0.05),np.log10(0.5),30)
# kevList = [0.15]

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
parser.add_argument('--nosave', action='store_true')
args = parser.parse_args()
fx=args.fx;rtol=args.rtol;kref=args.kref
print(f'Using fx={fx:.1e}, kref={kref}, rtol={rtol:.1e} and neval={args.neval}')

if not args.nosave:
    outfile='/home/fverdian/class/soundspeed-scripts/kernels-fullnum/numerical-integrals/numint-results/ccbreak-kJ'+str(kref).replace('.','p')+'-fx'+str(fx).replace('.','p')+'-N'+str(args.neval)+'.txt'
    if os.path.exists(outfile):
        os.remove(outfile)
    open(outfile, 'a').write(f'# k, dd22, dd22_EdS, dd13, dd13_EdS, dT22, dT22_EdS, dT13,dT13_EdS, PL \n')
#===============================
# LINEAR PART
#===============================

nk = NumKernels(fx=fx, kref=kref, fullt=fullt, rtol=rtol)
plclass = PLfromClass(fx=fx, kref=kref, zeval=1)
PLc_int = plclass.PLc_int


#--------
# Loop integral
#--------
def training_int(vars,kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]

    P13=6*q*q*nk.F3_0(kev,q,mu)*PLc_int(q)*PLc_int(kev)
    P13dT=3*q*q*(nk.F3_0(kev,q,mu) + nk.G3_0(kev,q,mu))*PLc_int(q)*PLc_int(kev)

    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22=0;P22dT=0
    if kMq>q:
        P22+=2*q*q*nk.F2_0(kev,q,mu)*nk.F2_0(kev,q,mu)*PLc_int(q)*PLc_int(kMq)
        P22dT+=2*q*q*(nk.F2_0(kev,q,mu)*nk.G2_0(kev,q,mu))*PLc_int(q)*PLc_int(kMq)
    if kPq>q:
        P22+=2*q*q*nk.F2_0(kev,q,-mu)*nk.F2_0(kev,q,-mu)*PLc_int(q)*PLc_int(kPq)
        P22dT+=2*q*q*(nk.F2_0(kev,q,-mu)*nk.G2_0(kev,q,-mu))*PLc_int(q)*PLc_int(kPq)
    intf=4*q*np.pi/fact
    return [intf*P22,intf*P13,intf*P22dT,intf*P13dT]

def Pnum_int(vars, kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]

    F3_cval,G3_cval,_,_=nk.solve_F3([kev,q,mu])
    P13dd=6*q*q*(F3_cval)*PLc_int(q)*PLc_int(kev)
    P13dT=3*q*q*(F3_cval*nk.g_c_int(-2*np.log(kev/kref))+G3_cval)*PLc_int(q)*PLc_int(kev)

    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22dd=0; P22dT=0
    if kMq>q:
        cT12 = (kev*mu-q)/kMq
        F2_cval,G_cval,_,_=nk.solve_F2([q,kMq,cT12])
        P22dd+=2*q*q*(F2_cval**2)*PLc_int(q)*PLc_int(kMq)
        P22dT+=2*q*q*(F2_cval*G_cval)*PLc_int(q)*PLc_int(kMq)

    if kPq>q:
        cT12 = -(kev*mu+q)/kPq
        F2_cval,G_cval,_,_=nk.solve_F2([q,kPq,cT12])
        P22dd+=2*q*q*(F2_cval**2)*PLc_int(q)*PLc_int(kPq)
        P22dT+=2*q*q*(F2_cval*G_cval)*PLc_int(q)*PLc_int(kPq)

    # The additional q below is due to logarithmic integration
    intf=4*q*np.pi/fact
    return [intf*P22dd,intf*P13dd,intf*P22dT,intf*P13dT]

#--------
# Integrate
#--------
print('Starting integral!!', flush=True)

for kEval in kevList:
    integ = vegas.Integrator([[np.log(1.e-4), np.log(1)], [0., 1.]],mpi=True, nproc=72)
    traindd22, traindd13, traindT22, traindT13 = integ(functools.partial(training_int, kev=kEval),neval=10000)
    start_time = time.time()
    result = integ(functools.partial(Pnum_int, kev=kEval), neval=args.neval)
    resdd22, resdd13, resdT22, resdT13 = result
    print(f'At k={kEval:.3f} done: took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s', flush=True)

    if not args.nosave: open(outfile, 'a').write(f'{kEval:.5g} {resdd22.mean:.5g} {traindd22.mean:.5g} {resdd13.mean:.5g} {traindd13.mean:.5g} {resdT22.mean:.5g} {traindT22.mean:.5g} {resdT13.mean:.5g} {traindT13.mean:.5g} {PLc_int(kEval):.5g}\n')
