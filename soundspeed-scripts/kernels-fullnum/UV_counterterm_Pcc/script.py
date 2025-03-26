
import numpy as np
import sys
sys.path.append('/home/fverdian/class/soundspeed-scripts/kernels-fullnum/numerical-integrals')
from num_kernels import NumKernels, PLfromClass
import vegas
import functools, time, os


fx=0.1
t_ini = -6
fullt=np.linspace(t_ini,1, 200)
teval=0
kref=1.
supprshift=7
nk = NumKernels(fx=fx, kref=kref, fullt=fullt, rtol=0.001)

outfile='/home/fverdian/class/soundspeed-scripts/kernels-fullnum/UV_counterterm_Pcc/int.txt'
if os.path.exists(outfile): os.remove(outfile)

def F3_UV_mu(mup, kp, qp):
    F3_c, G3_c, F3_x, G3_x =nk.solve_F3([kp,qp,mup])
    return F3_c*(qp/kp)**2


def F3_EdS_mu(mup, kp, qp):
    return nk.F3_0(kp,qp,mup)*(qp/kp)**2


Nk = 1

kplist = np.logspace(np.log10(0.1),np.log10(20), Nk)
qplist = np.logspace(np.log10(10),np.log10(200), Nk)

kplist = [5]

qplist = [100]
for i in range(Nk):
    kp = kplist[i]; qp = qplist[i]
    integ = vegas.Integrator([0., 1.],mpi=True, nproc=72)
    result_EdS = integ(functools.partial(F3_EdS_mu, kp=kp, qp=qp),neval=10000)

    start_time = time.time()
    result = integ(functools.partial(F3_UV_mu, kp=kp, qp=qp),neval=200)

    print(f'At k={kp:.3f}  EdS is {result_EdS}, full is {result}  (took {int((time.time()-start_time)//60)}m {(time.time()-start_time)%60:.0f}s).', flush=True)

    open(outfile, 'a').write(f'{kp} {result_EdS.mean} {result.mean}\n')
