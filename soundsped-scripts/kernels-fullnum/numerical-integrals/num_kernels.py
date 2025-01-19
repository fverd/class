import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.special import sici
from classy import Class
from scipy.optimize import brentq

class NumKernels:
    def __init__(self, fx, kref, fullt, supprshift=5, idx_eta=0, rtol=1.e-2):
        """
        Initialize the NumKernels class.

        Parameters:
        fx (any): A parameter representing some function or value.
        kerf (any): A parameter representing some kernel function or value.
        """
        self.fx = fx
        self.kref = kref
        self.fullt = fullt
        self.supprshift = supprshift
        self.rtol = rtol 
        self.idx_eta = np.abs(self.fullt - (0.)).argmin()

        # this is solved fot k=1 and then I shift it y using the smmetry
        def lin_system(w,t):
            dc,Tc,dx,Tx = w
            ddcdt = Tc
            dTcdt = -0.5*Tc+1.5*(1-fx)*dc+1.5*fx*dx
            ddxdt = Tx
            dTxdt = -0.5*Tx+1.5*(1-fx)*dc+1.5*(fx-(1)*np.exp(-t))*dx-np.exp(-t-supprshift)*ddxdt
            return [ddcdt, dTcdt,ddxdt, dTxdt]
        tlin =np.linspace(-10,10, 210)
        w0 = [1,1-3/5*fx,0.,0.]
        linsol = odeint(lin_system, w0, tlin)
        self.g_c_int=interp1d(tlin,linsol[:,1]/linsol[:, 0],bounds_error=False, fill_value=(5/4*np.sqrt(1-24/25*fx)-1/4,1))
    # Instead solving the linear system numerically I am using analytical formulas
    # def g_c_int(self,t):
    #     # This is just a phenomenological interpolation
    #     y = np.exp(-0.5*t)
    #     return 1-0.0614945/(1+0.386853/(y**1.69657))
   
    def g_an(self,t):
        return  1 + 6 * np.exp(-t) * np.cos(np.sqrt(6) * np.exp(-t / 2)) * sici(np.sqrt(6) * np.exp(-t / 2))[1] - 3 * np.exp(-t) * np.pi * np.sin(np.sqrt(6) * np.exp(-t / 2)) + 6 * np.exp(-t) * np.sin(np.sqrt(6) * np.exp(-t / 2)) * sici(np.sqrt(6) * np.exp(-t / 2))[0]
    def h_an(self,t):
        return 1 + 3 * np.sqrt(3 / 2) * np.exp(-3 * t / 2) * np.pi * np.cos(np.sqrt(6) * np.exp(-t / 2)) - 3 * np.exp(-t) * np.cos(np.sqrt(6) * np.exp(-t / 2))**2 + 3 * np.sqrt(6) * np.exp(-3 * t / 2) * sici(np.sqrt(6) * np.exp(-t / 2))[1] * np.sin(np.sqrt(6) * np.exp(-t / 2)) - 3 * np.sqrt(6) * np.exp(-3 * t / 2) * np.sin(np.sqrt(6) * np.exp(-t / 2)) * np.sinc(np.sqrt(6) * np.exp(-t / 2) / np.pi) - 3 * np.sqrt(6) * np.exp(-3 * t / 2) * np.cos(np.sqrt(6) * np.exp(-t / 2)) * sici(np.sqrt(6) * np.exp(-t / 2))[0]

    def alpha(self,k1,k2,cT):
        return 1+cT*k2/k1

    def alphas(self,k1,k2,cT):
        return 1+0.5*cT*(k2/k1+k1/k2)

    def beta(self,k1,k2,cT):
        t1=k1*k1+k2*k2+2*k1*k2*cT
        t2=k1*k2*cT
        t3=2.*k1*k1*k2*k2
        return t1*t2/t3

    def F2_0(self,k,q,mu):
        k2=k*k;q2=q*q;mu2=mu*mu
        kMq2 = k2 + q2 - 2*k*q*mu 
        return (k2*(7*k*q*mu+3*q2)-10*k2*q2*mu2)/(14*q2*kMq2)
    def G2_0(self,k,q,mu):
        x=k/q
        return (x**2 * (-6 * mu**2 + 7 * mu * x - 1)) / (14 * (x**2 - 2 * mu * x + 1))
    def F3_0(self,k,q,mu):
        x=k/q
        return (x**2 * (28 * mu**4 - 59 * mu**2 - 21 * mu**2 * x**4 + 2 * (38 * mu**4 - 22 * mu**2 + 5) * x**2 + 10)) / (126 * (x**4 + (2 - 4 * mu**2) * x**2 + 1))
    def G3_0(self,k,q,mu):
        x=k/q
        return (-x**2 * (-4 * mu**4 + 9 * mu**2 + 7 * mu**2 * x**4 + (-20 * mu**4 + 4 * mu**2 + 2) * x**2 + 2)) / (42 * (x**4 + (2 - 4 * mu**2) * x**2 + 1))

    #--------
    # 2nd order
    #--------
    def solve_F2(self, triplet, return_timedep=False):
        k1 , k2, cT = triplet
        kref = self.kref
        fx=self.fx
        Fc2_0=(5./7.+6/245*fx)*self.alphas(k1,k2,cT)+(2./7.-6/245*fx)*self.beta(k1,k2,cT)
        Gc2_0=(3./7-51/245*fx)*self.alphas(k1,k2,cT)+(4./7-96/245*fx)*self.beta(k1,k2,cT)

        def F2_system(w, t):
            Fc2,Gc2,Fx2,Gx2 = w
            g_c_k1=self.g_c_int(t-2*np.log(k1/kref)); g_c_k2=self.g_c_int(t-2*np.log(k2/kref))
            g_k1=self.g_an(t-2*np.log(k1/kref)); g_k2=self.g_an(t-2*np.log(k2/kref))
            h_k1=self.h_an(t-2*np.log(k1/kref)); h_k2=self.h_an(t-2*np.log(k2/kref))
                                                            
            SF2=0.5*g_c_k1*self.alpha(k1,k2,cT)+0.5*g_c_k2*self.alpha(k2,k1,cT)
            dFc2dt = -(g_c_k1+g_c_k2)*Fc2 + Gc2 + SF2
            dGc2dt = -(0.5+g_c_k1+g_c_k2)*Gc2 + 1.5 *((1-fx)*Fc2+fx*Fx2) +g_c_k1*g_c_k2*self.beta(k1,k2,cT)

            SF2=0.5*g_k2*h_k1*self.alpha(k1,k2,cT)+0.5*g_k1*h_k2*self.alpha(k2,k1,cT)
            dFx2dt = -(g_c_k1+g_c_k2)*Fx2 + Gx2 + SF2
            dGx2dt = -(0.5+g_c_k1+g_c_k2)*Gx2 + 1.5 *((1-fx)*Fc2+(fx-(k1**2+k2**2+2*k1*k2*cT)/(kref**2)*np.exp(-t))*Fx2) + h_k1*h_k2*self.beta(k1,k2,cT) -(k1**2+k2**2+2*k1*k2*cT)/(kref**2)*np.exp(-t-self.supprshift)*dFx2dt
            return [dFc2dt, dGc2dt,dFx2dt, dGx2dt]

        sol = odeint(F2_system, [Fc2_0,Gc2_0,0.,0.], self.fullt, rtol=self.rtol)
        if return_timedep:
            return sol
        return sol[self.idx_eta]

    #--------
    # 3rd order
    #--------

    def solve_F3(self, triplet, return_timedep=False):
        k , q, mu = triplet
        kref = self.kref
        fx=self.fx     

        ker2_k_mq_full=self.solve_F2([k,q,-mu], return_timedep=True)
        ker2_k_q_full=self.solve_F2([k,q,mu], return_timedep=True)
        # ker2_k_mq_int=[interp1d(self.fullt, ker2_k_mq_full[:, i], kind='linear') for i in range(4)]
        # ker2_k_q_int=[interp1d(self.fullt, ker2_k_q_full[:, i], kind='linear') for i in range(4)]


        # Relevant expressions for alpha and beta couplings
        x=k/q;x2=x*x
        a_q_kMq=mu*x;a_kMq_q=x*(x-mu)/(x2-2*x*mu+1)
        b_q_kMq=x2*(mu*x-1)/2./(x2-2.*mu*x+1.)
        a_q_kPq=-mu*x;a_kPq_q=x*(x+mu)/(x2+2*x*mu+1)
        b_q_kPq=-x2*(mu*x+1)/2./(x2+2.*mu*x+1.)

        def F3_system(w, t):

            Fc3,Gc3,Fx3,Gx3 = w
            g_c_k=self.g_c_int(t-2*np.log(k/kref));g_c_q=self.g_c_int(t-2*np.log(q/kref))
            g_q=self.g_an(t-2*np.log(q/kref)); h_q=self.h_an(t-2*np.log(q/kref))
            fact=(g_c_k+2*g_c_q)

            idx_t_F3=np.abs(self.fullt - t).argmin()
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
            dGx3dt = -(0.5+fact)*Gx3 + 1.5*((1-fx)*Fc3+(fx-(k**2)/(kref**2)*np.exp(-t))*Fx3) + SG3x/3 -(k**2)/(kref**2)*np.exp(-t-self.supprshift)*dFx3dt

            return [dFc3dt, dGc3dt,dFx3dt, dGx3dt]
        tsol = self.fullt[:self.idx_eta+2]
        sol = odeint(F3_system, [self.F3_0(k,q,mu),self.G3_0(k,q,mu),0,0], tsol, rtol=self.rtol)
        if return_timedep:
            return tsol, sol
        # print(sol[self.idx_eta], triplet[1:])
        return sol[self.idx_eta]
    
class PLfromClass:
    def __init__(self, fx, kref, zeval):

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
            return kJ_int_z(0.5) - kref
        if find_aNR_byshoot(0)<0:
            aNR = 10**(float(brentq(find_aNR_byshoot, -10, 0)))
            print(f'The value of aNR that gives kref={kref} is {aNR:.2e}, found by shooting')
        else:
            print(f'The provided value kref={kref} is impossible, as the NR transition would still have to happen. Setting aNR=1')
            aNR = 1.
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
        self.Pcdmonly_int=interp1d(kk,Pk_cdm,fill_value='extrapolate')
        self.PLc_int=interp1d(kk,Pk_chi,fill_value='extrapolate')
