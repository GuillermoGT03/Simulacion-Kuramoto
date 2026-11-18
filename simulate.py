import numpy as np
import argparse
from pathlib import Path
import logging
import time
from src.kuramoto import Kuramoto

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Simulación modelo Kuramoto")
parser.add_argument(
    "--nsim", 
    type=int, 
    default=20, 
    help="Número de simulaciones del modelo"
)
parser.add_argument(
    "--sim-index",
    type=int,
    default=None,
    help="Ejecuta solo esta simulación (si no se da, se ejecutan las nsim)"
)
parser.add_argument(
    "--N",
    type=int,
    default=100,
    help="Número de osciladores"
)
parser.add_argument(
    "--T",
    type=float,
    default=1000.0,
    help="Tiempo total de simulación"
)
parser.add_argument(
    "--sigma_w",
    type=float,
    default=0.85,
    help="Dispersion de la frecuencia natural"
)
args = parser.parse_args()

N = args.N   # Numero de osciladores acoplados
sigma_w = args.sigma_w   # Desviacion estandar de la frecuencia natural
dt = 0.5
T = args.T

K_first = np.linspace(0.0, 1.0, 10, endpoint=False)
K_pre   = np.linspace(1.0, 1.27, 10)
K_fine  = np.linspace(1.30, 1.50, 41)
K_post  = np.linspace(1.53, 1.80, 10)
K_rest  = np.linspace(1.8, 4.0, 22)[1:]  
K_values = np.concatenate((K_first, K_pre, K_fine, K_post, K_rest))


output_dir = Path("results") / f"N_{N}"
output_dir.mkdir(parents=True, exist_ok=True)

sims = [args.sim_index] if args.sim_index is not None else range(args.nsim)

for sim in sims:
    
    SEED = 42 + sim
    rng = np.random.default_rng(SEED)

    theta_i = rng.uniform(low = 0.0, high = 2*np.pi, size = N)
    omega = rng.normal(loc = 0.0, scale = sigma_w, size = N)

    theta_all = [] if SEED == 42 and N == 100 else None
    r_all = []
    t = None
    
    log.info(
        "Inicio sim=%d seed=%d N=%d sigma_w=%g T=%g dt=%g (%d valores de K)",
        sim, SEED, N, sigma_w, T, dt, len(K_values)
    )
    t_sim = time.perf_counter()

    for k in K_values:
        
        kuramoto = Kuramoto(
            K = k,
            theta = theta_i,
            omega = omega,
            dt = dt,
            T = T
        )
        
        sol = kuramoto.integrate()
        
        theta = sol.y
        t = sol.t
        
        Z = np.mean(np.exp(1j * theta), axis=0)
        r = np.abs(Z)
        
        r_all.append(r)
        if SEED == 42 and N == 100:
            theta_all.append(theta.astype(np.float32))
        
    filename = output_dir / f"sim_{sim:03d}_seed_{SEED}.npz"

    r_all = np.stack(r_all, axis=0)
        
    np.savez_compressed(
        filename,
        t=t,
        K_values=K_values,
        r=r_all,
        N=N,
        sigma_w=sigma_w,
        dt=dt,
        T=T,
        seed=SEED,
        sim=sim
    )

    if SEED == 42 and N == 100:
        theta_all = np.stack(theta_all, axis=0)

        trajectory_dir = output_dir / "trayectorias"
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            trajectory_dir / f"seed_{SEED}.npz",
            theta=theta_all,
            t=t,
            K_values=K_values,
            N=N,
            sigma_w=sigma_w,
            dt=dt,
            T=T,
            seed=SEED
        )
        
        log.info("Trayectorias guardadas en %s", trajectory_dir)

    log.info(
        "Fin sim=%d en %.1f s, guardada en %s",
        sim, time.perf_counter() - t_sim, filename
    )