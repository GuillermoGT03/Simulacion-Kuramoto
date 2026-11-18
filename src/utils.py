import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

def kuramoto_teorico(K_values, sigma_w):
    
    Kc = sigma_w*np.sqrt(8/np.pi)
    
    def I(x):
        integral, error = quad(
            lambda phi: (1/(np.sqrt(2*np.pi)*sigma_w)* np.exp(-(x*np.sin(phi))**2/(2*sigma_w**2)))*np.cos(phi)**2,
            -np.pi/2,
            np.pi/2,
            epsabs=1e-11,
            epsrel=1e-11
        )
        
        return integral
    
    r_values = []
    
    for K in K_values:
        
        if K <= Kc:
            r_values.append(0.0)
            
        else:
            x = brentq(
                lambda x: K*I(x) - 1,
                0,
                K
            )
            
            r = x/K
            
            r_values.append(r)
    
    return np.asarray(r_values)

def calculate_kc(t, r_all, K_values):
    nsim = r_all.shape[0]
    mask = t >= 0.5 * t[-1]
    r_window = r_all[:, :, mask]
    r_stationary = r_window.mean(axis=2)
    
    r_mean = r_stationary.mean(axis=0)
    r_std = r_stationary.std(axis=0, ddof=1)
    
    Kc_values = []
    for i in range(nsim):
    
        dr_dK = np.gradient(r_stationary[i, :], K_values)
        idx_critical = np.argmax(dr_dK)
        Kc = K_values[idx_critical]
        Kc_values.append(Kc)
    
    Kc_values = np.asarray(Kc_values)
    Kc = Kc_values.mean()
    Kc_std = Kc_values.std(ddof=1)
    
    rng = np.random.default_rng(0)
    Kc_boot = []
    for _ in range(2000):
        sel = rng.integers(0, nsim, size=nsim)
        
        sample = Kc_values[sel]
        
        Kc_boot.append(sample.mean())
    
    Kc_err = np.std(Kc_boot, ddof=1)
    
    return Kc, Kc_err, Kc_std, r_mean, r_std