import numpy as np
import time, os, sys
import functools
import argparse
import vegas
sys.path.append('/home/fverdian/class/soundspeed-scripts/kernels-fullnum/numerical-integrals')
from num_kernels import NumKernels, PLfromClass

parser = argparse.ArgumentParser(description="Compute P1-loop with fx component")

#===============================
# Compute P_{c,chi} one-loop terms
# It seems that I am computing also the full P_{c,c}, but in reality the EdS integral here
# is the P_{c,chi} one with EdS kernels, to compare the full result to
#===============================

# physical
fx=0.1
kref=0.1
zeval=1.

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
    outfile='/home/fverdian/class/soundspeed-scripts/kernels-fullnum/numerical-integrals/checks-Theta/dT13EdSh-kJ'+str(kref).replace('.','p')+'-fx'+str(fx).replace('.','p')+'-N'+str(args.neval)+'.txt'
    if os.path.exists(outfile):
        os.remove(outfile)
    open(outfile, 'a').write(f'# k, dcTc13, dcTc13_EdS, dcTx13, dcTx13_EdS, PL_cc, TL_x \n')

#===============================
# LINEAR PART
#===============================

nk = NumKernels(fx=fx, kref=kref, fullt=fullt, rtol=rtol)
plclass = PLfromClass(fx=fx, kref=kref, zeval=zeval)
PLc_int = plclass.PLc_int

#===============================
# LOOP INTEGRAL
#===============================

def training_int(vars,kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]
    g_k=nk.g_an(-2*np.log(kev/kref));g_q=nk.g_an(-2*np.log(q/kref));h_k=nk.h_an(-2*np.log(kev/kref))

    P13dcTc=3*q*q*(nk.F3_0(kev,q,mu)+ nk.G3_0(kev,q,mu))*nk.g_c_int(-2*np.log(kev/kref))*PLc_int(q)*PLc_int(kev)
    P13dcTx=3*q*q*(nk.F3_0(kev,q,mu) + nk.G3_0(kev,q,mu)*g_q*g_q)*h_k*PLc_int(q)*PLc_int(kev)

    intf=4*q*np.pi/fact
    return [intf*P13dcTc,intf*P13dcTx]

def Pnum_int(vars, kev):
    logq=vars[0]
    q = np.exp(logq)
    mu=vars[1]
    g_k=nk.g_an(-2*np.log(kev/kref));g_q=nk.g_an(-2*np.log(q/kref))

    F3_cval, G3_cval, _, G3_xval=nk.solve_F3([kev,q,mu])
    P13dcTc=3*q*q*(F3_cval*nk.g_c_int(-2*np.log(kev/kref))+G3_cval)*PLc_int(q)*PLc_int(kev)
    P13dcTx=3*q*q*(nk.F3_0(kev,q,mu)*nk.h_an(-2*np.log(kev/kref))+G3_xval)*PLc_int(q)*PLc_int(kev)

    # The additional q below is due to logarithmic integration
    intf=4*q*np.pi/fact
    return [intf*P13dcTc,intf*P13dcTx]

#--------
# Integrate
#--------
print('Starting integral')
for kEval in kevList:
    integ = vegas.Integrator([[np.log(1.e-3), np.log(1.)], [0., 1.]],mpi=True, nproc=70)
    traindcTc13, traindcTx13 = integ(functools.partial(training_int, kev=kEval),neval=10000)
    start_time = time.time()
    result = integ(functools.partial(Pnum_int, kev=kEval), neval=args.neval)
    resdcTc13, resdcTx13 = result

    print(f'At k={kEval:.3f} done: took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s', flush=True)

    if not args.nosave: open(outfile, 'a').write(f'{kEval:.5g} {resdcTc13.mean:.5g} {traindcTc13.mean:.5g} {resdcTx13.mean:.5g} {traindcTx13.mean:.5g} {PLc_int(kEval):.5g} {nk.g_an(-2*np.log(kEval/kref))}\n')

