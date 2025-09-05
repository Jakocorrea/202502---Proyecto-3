# run_sim.py
# Figuras:
#  - G1_concentracion_vs_t.png : c(t) promedio (fracción másica de pintura)
#  - G2_velocidad_vs_t.png     : U_bulk(t) (arranca en 0.2) + referencia U_fan(t)
#  - G3_densidad_y_potencia.png: ρ(t) y Potencia del ventilador P_fan(t)

import matplotlib.pyplot as plt
from sim_logic import simulate

def main():
    res = simulate("Entrega 3/params.json")
    t   = res["t"]
    c   = res["c_mean"]
    u_b = res["u_bulk"]
    u_f = res["u_fan"]
    rho = res["rho"]
    Pf  = res["P_fan"]

    # -------- (1) Concentración promedio vs tiempo --------
    plt.figure()
    plt.plot(t, c, linewidth=2)
    plt.xlabel("Tiempo, t [s]")
    plt.ylabel("Concentración promedio, ⟨c⟩ [-]")
    plt.title("Concentración de pintura en el ambiente vs tiempo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("G1_concentracion_vs_t.png", dpi=180)

    # -------- (2) Velocidad: U_bulk(t) y U_fan(t) ----------
    plt.figure()
    plt.plot(t, u_b, linewidth=2, label="U_mezcla(t) ")
    # línea base a 0.2 para claridad visual
    plt.axhline(y=0.2, linestyle=":", linewidth=1)
    plt.xlabel("Tiempo, t [s]")
    plt.ylabel("Velocidad [m/s]")
    plt.title("Velocidad del fluido vs tiempo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("G2_velocidad_vs_t.png", dpi=180)

    # -------- (3) Densidad y Potencia del fan vs tiempo ---
    fig, ax1 = plt.subplots()
    ax1.plot(t, rho, linewidth=2, label="ρ(t)")
    ax1.set_xlabel("Tiempo, t [s]")
    ax1.set_ylabel("Densidad de mezcla, ρ(t) [kg/m³]")
    ax1.grid(True)
    fig.suptitle("Densidad de la mezcla vs tiempo")
    fig.tight_layout()
    fig.savefig("G3_densidad.png", dpi=180)

    print("Figuras generadas:")
    print(" - G1_concentracion_vs_t.png")
    print(" - G2_velocidad_vs_t.png")
    print(" - G3_densidad_y_potencia.png")

if __name__ == "__main__":
    main()