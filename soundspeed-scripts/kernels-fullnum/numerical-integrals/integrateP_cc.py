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
    outfile='/home/fverdian/class/soundspeed-scripts/kernels-fullnum/numerical-integrals/numint-results/Pcc-kJ'+str(kref).replace('.','p')+'-fx'+str(fx).replace('.','p')+'-N'+str(args.neval)+'.txt'
    if os.path.exists(outfile):
        os.remove(outfile)
    open(outfile, 'a').write(f'# k, 1ldd, 1ldd_EdS, 1ldT, 1ldT_EdS, PL \n')
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
    return [(P13+P22)*4*q*np.pi/fact,(P13dT+P22dT)*4*q*np.pi/fact]

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
    return [(P13dd+P22dd)*4*q*np.pi/fact,(P13dT+P22dT)*4*q*np.pi/fact]

#--------
# Integrate
#--------
print('Starting integral!!', flush=True)

for kEval in kevList:
    integ = vegas.Integrator([[np.log(1.e-4), np.log(15)], [0., 1.]],mpi=True, nproc=48)
    traindd, traindT = integ(functools.partial(training_int, kev=kEval),neval=10000)
    start_time = time.time()
    result = integ(functools.partial(Pnum_int, kev=kEval), neval=args.neval)
    resdd, resdT = result
    print(f'At k={kEval:.3f}  Pdd is {resdd}, PdT is {resdT}  (took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s). Training EdS result was {traindd} and {traindT}', flush=True)

    if not args.nosave: open(outfile, 'a').write(f'{kEval} {resdd.mean} {traindd.mean} {resdT.mean} {traindT.mean} {PLc_int(kEval)}\n')
