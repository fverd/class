import numpy as np
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import functools
import os
import matplotlib.pyplot as plt
import pickle, argparse
import vegas
parser = argparse.ArgumentParser(description="Compute P1-loop with fx component")

#===============================
# Setup parameters
#===============================

fx=0.1
fullt=np.linspace(-5,2, 100)
idx_eta=np.abs(fullt - (0.)).argmin()
# I will make the case of kJ=0.1 at a=0.1
kref=0.1
rtol=0.1

parser.add_argument('-fx', dest='fx', type=float, default=fx , help='Chi fraction')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-eds', '--EdS', action='store_true')
args = parser.parse_args()
fx=args.fx
print(f'Using fx={fx:.1e}')

#===============================
# Load Plin and kernels definitions
#===============================

with open('/home/fverdian/class/soundspeed-scripts/1loop/Pk-lin-int.pkl', 'rb') as f:
    Plin_int = pickle.load(f)
fact=(2*np.pi)**3

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

# deep hot kernels

def DeltaF3_ennio(k,q,mu):
    x2=(k/q)**2;mu2=mu*mu
    num=4*(mu2-1)*x2*(-49*mu2+(5*mu2+22)*x2 +22)
    den=945*(x2*x2+(2-4*mu2)*x2+1)
    return num/den

def G2F2combo(k1,k2,cT):
    return [(5./7+6/245*fx)*alphas(k1,k2,cT)+(2./7-6/245*fx)*beta(k1,k2,cT),(3./7-3*17/245*fx)*alphas(k1,k2,cT)+(4./7-3*32/245*fx)*beta(k1,k2,cT),0,0]



def training_int(vars,kev):
    q=vars[0]
    mu=vars[1]
    P13=6*q*q*F3(kev,q,mu)*Plin_int(q)*Plin_int(kev)
    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22=0
    if kMq>q:
        P22+=2*q*q*F2(kev,q,mu)*F2(kev,q,mu)*Plin_int(q)*Plin_int(kMq)
    if kPq>q:
        P22+=2*q*q*F2(kev,q,-mu)*F2(kev,q,-mu)*Plin_int(q)*Plin_int(kPq)
    return (P13+P22)*4*np.pi/fact

# kevList=[0.15]
kevList=np.logspace(np.log10(0.05),np.log10(1),50)

#--------
# Loop integral
#--------
def P1loop_int(vars, kev):
    q=vars[0]
    mu=vars[1]
    F3_val=-1/7*fx*DeltaF3_ennio(kev,q,mu)
    P13=6*q*q*F3_val*Plin_int(q)*Plin_int(kev)
    kMq = np.sqrt(kev*kev + q*q - 2*kev*q*mu)
    kPq = np.sqrt(kev*kev + q*q + 2*kev*q*mu)
    P22=0
    if kMq>q:
        cT12 = (kev*mu-q)/kMq
        DF2_2=G2F2combo(q,kMq,cT12)[0]**2-F2(kev,q,mu)**2
        P22+=2*q*q*DF2_2*Plin_int(q)*Plin_int(kMq)
    if kPq>q:
        cT12 = -(kev*mu+q)/kPq
        DF2_2=G2F2combo(q,kPq,cT12)[0]**2-F2(kev,q,-mu)**2
        P22+=2*q*q*DF2_2*Plin_int(q)*Plin_int(kPq)
    return (P13+P22)*4*np.pi/fact

#--------
# Integrate
#--------
print('Starting integral!!', flush=True)
reslist=[]
integ = vegas.Integrator([[1.e-5, 10], [0., 1.]],mpi=True, nproc=48)

outfile='/home/fverdian/class/soundspeed-scripts/kernels-fullnum/4em/1loop-fullhot-fx'+str(fx).replace('.','p')+'.txt'
if os.path.exists(outfile):
    os.remove(outfile)
open(outfile, 'a').write(f'# k, fullhot-EdS, EdS result\n')

for kEval in kevList:
    train = integ(functools.partial(training_int, kev=kEval),neval=500)
    start_time = time.time()
    result = integ(functools.partial(P1loop_int, kev=kEval), neval=500)
    reslist.append([result.mean, train.mean])
    print(f'Result at k={kEval:.3f} is {result} (took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s). Training EdS result was {train}', flush=True)
    open(outfile, 'a').write(f'{kEval} {result.mean} {train.mean}\n')
