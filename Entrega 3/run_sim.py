# run_sim.py
# Resultados: (1) concentración promedio vs tiempo, (2) velocidad del fluido vs tiempo
import matplotlib.pyplot as plt
from sim_logic import simulate

def main():
    res = simulate("Entrega 3/params.json")
    t = res["t"]
    c = res["c_mean"]
    u = res["u_mean"]
    P = res["params"]

    # (1) Concentración promedio vs tiempo
    plt.figure()
    plt.plot(t, c, linewidth=2)
    sp = P.get("spray", None)
    if sp is not None:
        t_on = float(sp.get("t_on", 0.0))
        t_off = sp.get("t_off", None)
        if t_off is None:
            plt.axvline(t_on, linestyle="--", alpha=0.3)
        else:
            plt.axvspan(t_on, float(t_off), alpha=0.15)
    plt.xlabel("Tiempo, t [s]")
    plt.ylabel("Concentración promedio, ⟨c⟩ [-]")
    plt.title("Concentración en el ambiente vs tiempo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("G1_concentracion_vs_t.png", dpi=180)

    # (2) Velocidad del fluido vs tiempo
    plt.figure()
    plt.plot(t, u, linewidth=2)
    plt.xlabel("Tiempo, t [s]")
    plt.ylabel("Velocidad media, U(t) [m/s]")
    plt.title("Velocidad del fluido vs tiempo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("G2_velocidad_vs_t.png", dpi=180)

    print("Listo:")
    print(" - G1_concentracion_vs_t.png")
    print(" - G2_velocidad_vs_t.png")

if __name__ == "__main__":
    main()
