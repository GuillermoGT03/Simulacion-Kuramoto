import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
from scipy.stats import shapiro, norm
from scipy.optimize import curve_fit
import argparse
from pathlib import Path
import logging
from src.utils import kuramoto_teorico, calculate_kc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Simulación modelo Kuramoto")
parser.add_argument(
    "--estudio",
    choices=["inicial", "posterior"],
    default="inicial",
    help="Tipo de estudio: N fijo (inicial) o variación de N (posterior)"
)
parser.add_argument(
    "--N",
    type=int,
    default=100,
    help="Número de osciladores"
)
args = parser.parse_args()

estudio = args.estudio  # Estudio a realizar
N = args.N   # Numero de osciladores acoplados

plots_dir = Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)

if estudio == "inicial":
    
    results_dir = Path("results") / f"N_{N}"
    
    files = sorted(results_dir.glob("*.npz"))
    
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos .npz en {results_dir}"
        )
    
    r_all = []
    t = None
    
    for file in files:
        
        with np.load(file) as data:
            r_all.append(data["r"])
            
            if t is None:
                t = data["t"]
                K_values = data["K_values"]
                SEED = int(data["seed"])
                sigma_w = data["sigma_w"]
    
    r_all = np.stack(r_all, axis=0)
    
    #-------------------------------------------------
    # Creación gráfica distribucion de frecuencias naturales w
    #-------------------------------------------------
    
    rng = np.random.default_rng(SEED)
    theta_i = rng.uniform(low = 0.0, high = 2*np.pi, size = N)
    omega = rng.normal(loc = 0.0, scale = sigma_w, size = N)
    
    stat, p_val = shapiro(omega)
    
    if p_val > 0.05:
        log.info(f"p_valor = {p_val:.2f} > 0.05 -> Los datos parecen seguir una distribución normal")
    else:
        log.info(f"p_valor = {p_val:.2f} < 0.05 -> Los datos fallan el test de normalidad")
    
    x = np.linspace(min(omega), max(omega), 100)
    y = norm.pdf(x, 0, sigma_w)
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(omega, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    
    ax.plot(x, y, "--", color = "red", label = fr"Curva Normal Teórica ($\mu = 0.0$, $\sigma = {sigma_w:.2f}$)")
    
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('Densidad')
    ax.set_xlim(-2.5, 2.5)
    ax.legend(loc="upper right")
    fig.savefig(plots_dir / "omega_dist.png")
    plt.close(fig)
    
    
    #-------------------------------------------------
    # Creación gráfica r(t) para una única simulación (SEED = 42)
    #-------------------------------------------------
    
    norm = mpl.colors.Normalize(
        vmin=K_values.min(),
        vmax=K_values.max()
    )
    
    cmap = plt.colormaps["viridis"]
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    indices = np.linspace(0, len(K_values) - 1, 30, dtype=int)
    for i in indices:
        K = K_values[i]
        ax.plot(t, r_all[0, i, :], color=cmap(norm(K)))
        
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Fuerza de acoplamiento K")
    
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Parámetro de orden r")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(plots_dir / "r_t_graph.png")
    plt.close(fig)
    
    
    #-------------------------------------------------
    # Creación gráfica r(K) usando todas las simulaciones
    #-------------------------------------------------
    
    Kc, Kc_err, Kc_std, r_mean, r_std = calculate_kc(t, r_all, K_values)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(K_values, r_mean)
    ax.fill_between(
        K_values,
        r_mean - r_std,
        r_mean + r_std,
        alpha=0.2,
    )
    ax.axvline(Kc, color="red", linestyle="--",
            label=fr"$\langle K_c\rangle={Kc:.2f}\pm{Kc_err:.2f}$")
    ax.set_xlabel("Constante de acoplamiento $K$")
    ax.set_ylabel("Parámetro de orden $r$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(Path("plots") / "r(K)_graph.png")
    plt.close(fig)
    
    
    #-------------------------------------------------
    # Creación animación sincronización para K < Kc, K similar a Kc y K > Kc
    #-------------------------------------------------
    
    theta_dir = results_dir / "trayectorias" / "seed_42.npz"
    
    with np.load(theta_dir) as data:
        theta_all = data["theta"]
        
    idx_subcritico = np.argmin(np.abs(K_values - (Kc * 0.5)))
    idx_critico = np.argmin(np.abs(K_values - Kc))
    idx_supercritico = np.argmin(np.abs(K_values - (Kc * 1.5)))
    
    indices_K = [idx_subcritico, idx_critico, idx_supercritico]
    etiquetas_K = [
        rf"$K < K_c$ ($K={K_values[idx_subcritico]:.2f}$)",
        rf"$K \approx K_c$ ($K={K_values[idx_critico]:.2f}$)",
        rf"$K > K_c$ ($K={K_values[idx_supercritico]:.2f}$)",
    ]
    
    # Configuración figura y gs para customizar celdas de los gráficos
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Fila superior: 3 subplots polares para los osciladores
    ax_circ1 = fig.add_subplot(gs[0, 0], polar=True)
    ax_circ2 = fig.add_subplot(gs[0, 1], polar=True)
    ax_circ3 = fig.add_subplot(gs[0, 2], polar=True)
    axes_circ = [ax_circ1, ax_circ2, ax_circ3]
    
    # Fila inferior: 1 subplot que abarque toda la anchura para r(t)
    ax_r = fig.add_subplot(gs[1, :])
    
    # Dibujamos las curvas completas de r(t) para las 3 simulaciones seleccionadas
    colores = ["blue", "orange", "green"]
    for idx_k, col in zip(indices_K, colores):
        ax_r.plot(
            t,
            r_all[0, idx_k, :],
            color=col,
            alpha=0.4,
            linestyle="--",
        )
        
    # Curvas principales resaltadas
    ax_r.plot(t, r_all[0, idx_subcritico, :], label=r"$K < K_c$", color="blue")
    ax_r.plot(t, r_all[0, idx_critico, :], label=r"$K ≈ K_c$", color="orange")
    ax_r.plot(t, r_all[0, idx_supercritico, :], label=r"$K > K_c$", color="green")
    
    ax_r.set_xlabel("Tiempo (s)")
    ax_r.set_ylabel("Parámetro de orden $r$")
    ax_r.set_xlim(t[0], t[-1])
    ax_r.set_ylim(-0.05, 1.05)
    ax_r.legend(loc="lower right")
    ax_r.grid(True, linestyle=":", alpha=0.6)
    
    # Línea vertical móvil que indica el tiempo actual en la animación
    time_line = ax_r.axvline(t[0], color="red", linestyle="-", lw=2, label="Actual")
    
    # Inicialización de las gráficas polares superiores
    scatters = []
    r_vectors = []
    for ax, label in zip(axes_circ, etiquetas_K):
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(label, pad=30)
        # Scatter para los osciladores individuales en el círculo (r=1, theta)
        sc = ax.scatter([], [], c="purple", alpha=0.6, s=30, edgecolors="k")
        scatters.append(sc)
        # Flecha o línea roja que representa el vector de Kuramoto (r, psi)
        (line_rv,) = ax.plot([], [], color="red", lw=3)
        r_vectors.append(line_rv)
        
        
    # Animación
    def init():
        for sc, rv in zip(scatters, r_vectors):
            sc.set_offsets(np.empty((0, 2)))
            rv.set_data([], [])
        time_line.set_xdata([t[0], t[0]])
        return scatters + r_vectors + [time_line]
    
    
    def update(frame):
        current_t = t[frame]
        time_line.set_xdata([current_t, current_t])
        
        for k_idx, ax, sc, rv in zip(indices_K, axes_circ, scatters, r_vectors):
            thetas = theta_all[k_idx, :, frame]  # Fases de todos los osciladores
            rs = np.ones_like(thetas)  # Radio unitario para visualización en círculo
            
            # Actualizar posiciones de los osciladores
            sc.set_offsets(np.column_stack((thetas, rs)))
            
            # Calcular el parámetro de orden complejo Z = r * e^(i*psi) para este frame
            z = np.mean(np.exp(1j * thetas))
            r_val = np.abs(z)
            psi_val = np.angle(z)
            
            # Actualizar el vector promedio en el diagrama polar
            rv.set_data([psi_val, psi_val], [0, r_val])
            
        return scatters + r_vectors + [time_line]
    
    
    # Crear la animación
    ani = animation.FuncAnimation(
        fig, update, frames=int(round(len(t)/10)), init_func=init, blit=False, interval=120
    )
    
    fig.tight_layout()
    
    # Guardar la animación
    output_path = plots_dir / "kuramoto_sincronizacion_animacion.gif"
    ani.save(output_path, writer="pillow")
    plt.close(fig)
    log.info(f"Animación guardada con éxito en: {output_path}")
    

else:
    
    results_dir = Path("results")
    
    # Creamos listas para guardar los K criticos y su error para distintos N
    kc_list = []
    kerr_list = []
    kcstd_list = []
    r_mean_N = []
    
    # Creamos la figura para la primera gráfica
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Recorremos las carpetas de distintos N, key se usa para que recorra en orden numerico ascendente
    folders = sorted(
        results_dir.glob("N_*"),
        key=lambda folder: int(folder.name.split("_")[1])
    )
    
    N_values = []
    
    for folder in folders:
        
        # Seleccionamos el N de cada carpeta
        N = int(folder.name.split("_")[1])
        N_values.append(N)
        
        # Obtenemos los archivos .npz de cada carpeta
        files = sorted(folder.glob("*.npz"))
        
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos .npz en {results_dir}"
            )
        
        r_all = []
        t = None
        
        # Recorremos los archivos para obtener los datos
        for file in files:
            
            with np.load(file) as data:
                r_all.append(data["r"])
                
                if t is None:
                    t = data["t"]
                    K_values = data["K_values"]
                    sigma_w = data["sigma_w"]
        
        r_all = np.stack(r_all, axis=0)
        
        Kc, Kc_err, Kc_std, r_mean, r_std = calculate_kc(t, r_all, K_values)
        
        kc_list.append(Kc)
        kerr_list.append(Kc_err)
        kcstd_list.append(Kc_std)
        r_mean_N.append(r_mean)
        
        ax.plot(K_values, r_mean, label = f"N = {N}")
        
    Kteo_values = kuramoto_teorico(K_values, sigma_w)
    ax.plot(K_values, Kteo_values, "--", color = "black", label = fr"$N \to \infty$")
    
    ax.set_xlabel("Constante de acoplamiento $K$")
    ax.set_ylabel("Parámetro de orden $r$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(Path("plots") / "N_r(K)_graph.png")
    plt.close(fig)
    
    #-------------------------------------------------
    # Creación gráfica K critico para diferentes N
    #-------------------------------------------------
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Kc en funcion de N
    kc_teo = sigma_w*np.sqrt(8/np.pi)
    
    ax[0].errorbar(
        N_values,
        kc_list,
        yerr=kerr_list,
        fmt='o',
        capsize=4
    )
    ax[0].axhline(kc_teo, color="red", linestyle="--",
        label=fr"$K_c (teórico)={kc_teo:.3f}$")
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'Tamaño del sistema $N$')
    ax[0].set_ylabel(r'Acoplamiento crítico $K_c$')
    ax[0].legend(loc="lower right")
    
    # Errores en funcion de N
    ax[1].plot(N_values, kerr_list)
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'Tamaño del sistema $N$')
    ax[1].set_ylabel(r'Error Acoplamiento crítico $\delta K_c$')
    
    fig.tight_layout()
    fig.savefig(plots_dir / "Kc_N_graph.png")
    plt.close(fig)
    
    
    #-------------------------------------------------
    # Creación grafica logstd vs log N
    #-------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    logN = np.log(np.asarray(N_values))
    logstd = np.log(np.asarray(kcstd_list))
    
    xx = np.linspace(0, 1.05 * logN.max(), 100)
    
    f = lambda logN, logb, alpha: logb + logN * alpha
    popt, pcov = curve_fit(f, logN, logstd)
    
    ax.plot(logN, logstd, "o", color = "blue")
    ax.plot(xx, f(xx, *popt), "k--", label = fr"$alpha = {popt[1]:.3f} \pm {np.sqrt(np.diag(pcov))[1]:.3f}$")
    
    ax.set_xlabel(r"$log(N)$")
    ax.set_ylabel(r"$log(\sigma K_c)$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plots_dir / "alpha_graph.png")
    plt.close(fig)
    
    
    
    #-------------------------------------------------
    # Creación grafica N^-0.4
    #-------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.asarray(N_values)**(-0.4)
    f = lambda x, Kinf, a: Kinf + a * x
    popt, pcov = curve_fit(f, x, kc_list, sigma=kerr_list, absolute_sigma=True)
    
    log.info(f"El valor experimental de Kc cuando N tiende a infinito es: Kc = {popt[0]:.3f} ± {np.sqrt(pcov[0, 0]):.3f}")
    
    xx = np.linspace(0, 1.05 * x.max(), 100)
    
    ax.errorbar(x, 
        kc_list, 
        yerr = kerr_list, 
        fmt = "o", 
        capsize=4)
    ax.plot(xx, f(xx, *popt), "k--", label = fr"Ajuste: $K_c^\infty = {popt[0]:.3f} \pm {np.sqrt(pcov[0,0]):.3f}$")
    ax.axhline(kc_teo, color = "red", linestyle = "--", label = fr"Teoría $K_c = {kc_teo:.3f}$")
    
    ax.set_xlim(0, None)
    ax.set_xlabel(r"$N^{-0.4}$")
    ax.set_ylabel(r"Acoplamiento crítico $K_c$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plots_dir / "N_0.4_graph.png")
    plt.close(fig)

