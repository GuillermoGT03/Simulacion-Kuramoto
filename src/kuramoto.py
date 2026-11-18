
import numpy as np
from scipy.integrate import solve_ivp


class Kuramoto:
    
    def __init__(
        self,
        K: float,
        theta,
        omega,
        dt: float,
        T: float
    ):
        
        self.K = K
        self.dt = dt
        self.T = T
        self.theta = theta
        self.omega = omega
        
    def kuramoto_rhs(
        self,
        t,
        theta
    ):
        
        Z = np.mean(np.exp(1j * theta))
        dtheta_dt = self.omega + self.K * np.imag(Z * np.exp(-1j * theta))
        
        return dtheta_dt
    
    
    def integrate(
        self
    ):
        
        t_0 = 0.0
        t_eval = np.linspace(t_0, self.T, int(round(self.T / self.dt)) + 1)
        sol = solve_ivp(self.kuramoto_rhs, (t_0, self.T), self.theta, method = "RK45", t_eval = t_eval, rtol = 1e-6)
        
        return sol
    


