# run_sim.py
# Figuras:
#  - G1_concentracion_vs_t.png : c(t) + asíntota superior (línea horizontal punteada)
#  - G2_velocidad_vs_t.png     : U_mezcla(t), U_fan(t) + asíntotas superiores + ref 0.2
#  - G3_densidad_y_potencia.png: ρ(t) y P_fan(t) con sus asíntotas superiores

import numpy as np
import matplotlib.pyplot as plt
from sim_logic import simulate

# --------- Estimación de la asíntota superior -------------------------------
def estimate_upper_asymptote(y, tail_frac=0.15, quantile=0.90):
    """
    Estima la 'asíntota superior' como el cuantil dado del tramo final de la señal.
    tail_frac: fracción final de datos usada (p.ej. 0.10–0.25)
    quantile : cuantil dentro de ese tramo (p.ej. 0.85–0.95)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return None
    i0 = max(0, int((1.0 - tail_frac) * n))
    tail = y[i0:]
    if len(tail) == 0:
        tail = y
    return float(np.quantile(tail, quantile))


def draw_h_asymptote(ax, y_inf, label="asíntota (≈)"):
    if y_inf is None or not np.isfinite(y_inf):
        return
    # línea horizontal punteada alineada con la asíntota superior
    ax.axhline(y_inf, linestyle=":", linewidth=1.8)
    # pequeña etiqueta al borde derecho
    xmin, xmax = ax.get_xlim()
    ax.text(xmax, y_inf, f"  {label} {y_inf:.3g}", va="center", ha="left", fontsize=9, alpha=0.9)


# ------------------------ Estética global ------------------------------------
plt.rcParams.update({
    "figure.figsize": (8.5, 5.2),
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
})

def main():
    res = simulate("Entrega 3/params.json")
    t   = res["t"]
    c   = res["c_mean"]
    u_b = res["u_bulk"]
    u_f = res["u_fan"]
    rho = res["rho"]
    Pf  = res["P_fan"]

    # ---------- (1) Concentración promedio vs tiempo -------------------------
    fig1, ax1 = plt.subplots()
    ax1.plot(t, c, linewidth=2, label="⟨c⟩(t)")
    ax1.set_xlabel("Tiempo, t [s]")
    ax1.set_ylabel("Concentración promedio, ⟨c⟩ [Kg_p/Kg_m]")
    ax1.set_title("Concentración de pintura en el ambiente vs tiempo")

    c_inf = estimate_upper_asymptote(c, tail_frac=0.15, quantile=0.90)
    draw_h_asymptote(ax1, c_inf)

    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig("G1_concentracion_vs_t.png")

    # ---------- (2) Velocidad: U_mezcla(t) y U_fan(t) ------------------------
    fig2, ax2 = plt.subplots()
    ax2.plot(t, u_b, linewidth=2, label="U_mezcla(t)")

    ax2.set_xlabel("Tiempo, t [s]")
    ax2.set_ylabel("Velocidad [m/s]")
    ax2.set_title("Velocidad del fluido vs tiempo")
    ub_inf = estimate_upper_asymptote(u_b, tail_frac=0.15, quantile=0.90)
    draw_h_asymptote(ax2, ub_inf, label="Equilibrio de U_mezcla ≈")
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig("G2_velocidad_vs_t.png")

    # ---------- (3) Densidad y Potencia del ventilador -----------------------
    fig3, ax3 = plt.subplots()
    l1, = ax3.plot(t, rho, linewidth=2, label="ρ(t)")
    ax3.set_xlabel("Tiempo, t [s]")
    ax3.set_ylabel("Densidad de mezcla, ρ(t) [kg/m³]")

    ax4 = ax3.twinx()
    l2, = ax4.plot(t, Pf, linewidth=2, label="P_fan(t)")
    ax4.set_ylabel("Potencia del ventilador, P_fan(t) [W]")

    # asíntotas superiores independientes para cada eje
    rho_inf = estimate_upper_asymptote(rho, tail_frac=0.15, quantile=0.90)
    Pf_inf  = estimate_upper_asymptote(Pf,  tail_frac=0.15, quantile=0.90)
    draw_h_asymptote(ax3, rho_inf, label="asíntota ρ ≈")
    draw_h_asymptote(ax4, Pf_inf,  label="asíntota P ≈")

    # leyenda combinada
    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax3.legend(lines, labels, loc="upper right")

    fig3.suptitle("Densidad de la mezcla y potencia del ventilador vs tiempo")
    fig3.tight_layout()
    fig3.savefig("G3_densidad_y_potencia.png")

    print("Figuras generadas:")
    print(" - G1_concentracion_vs_t.png")
    print(" - G2_velocidad_vs_t.png")
    print(" - G3_densidad_y_potencia.png")

if __name__ == "__main__":
    main()
