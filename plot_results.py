import argparse
import json
import sys
import matplotlib.pyplot as plt

from sim_logic import config_from_json, Simulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grafica resultados de la simulación de cabina de pintura."
    )
    p.add_argument(
        "-c", "--config", default="example_config.json",
        help="Ruta al archivo de configuración JSON (por defecto: example_config.json)"
    )
    p.add_argument(
        "--dt", type=float, default=None,
        help="Paso de tiempo [s] para sobreescribir el del JSON."
    )
    p.add_argument(
        "--t_end", type=float, default=None,
        help="Tiempo total de simulación [s] para sobreescribir el del JSON."
    )
    p.add_argument(
        "--save", default=None,
        help="Ruta base para guardar PNGs (sin extensión). Si no se especifica, solo se muestra en pantalla."
    )
    p.add_argument(
        "--dpi", type=int, default=120,
        help="Resolución de las imágenes al guardar (por defecto: 120 dpi)."
    )
    return p.parse_args()


def load_sim_params(path: str, override_dt, override_t_end):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    simsec = raw.get("sim", {}) if isinstance(raw, dict) else {}
    dt = float(simsec.get("dt_s", 1.0)) if override_dt is None else float(override_dt)
    t_end = float(simsec.get("t_end_s", 3600.0)) if override_t_end is None else float(override_t_end)
    return dt, t_end


def series(results, key, default=0.0):
    return [r.get(key, default) for r in results]


def main():
    args = parse_args()

    try:
        # 1) Parámetros y configuración
        dt_s, t_end_s = load_sim_params(args.config, args.dt, args.t_end)
        cfg = config_from_json(args.config)

        # 2) Simulación
        sim = Simulator(cfg)
        results, summary = sim.run(t_end_s=t_end_s, dt_s=dt_s)
        if not results:
            print("No hay resultados para graficar (lista vacía).", file=sys.stderr)
            sys.exit(2)

        # 3) Extraer series
        t = series(results, "time_s")
        Q = series(results, "Q_m3_s")
        V_face = series(results, "V_face_m_s")
        v_duct = series(results, "v_duct_m_s")

        DP_sys = series(results, "DP_system_Pa")
        DP_filt = series(results, "DP_filter_Pa")
        DP_net = series(results, "DP_network_Pa")

        C_v = series(results, "C_v_mg_m3")
        C_p = series(results, "C_p_mg_m3")
        C_tot = series(results, "C_tot_mg_m3")

        P = series(results, "P_e_W")
        # Energía acumulada [kWh]
        energy_kWh = []
        acc_Wh = 0.0
        for p in P:
            acc_Wh += p * dt_s / 3600.0
            energy_kWh.append(acc_Wh / 1000.0)

        # 4) Figuras
        # --- Figura 1: Caudal y velocidades
        plt.figure(figsize=(9, 5))
        plt.plot(t, Q, label="Q [m³/s]")
        plt.plot(t, V_face, label="V_face [m/s]")
        plt.plot(t, v_duct, label="v_duct [m/s]")
        plt.xlabel("Tiempo [s]")
        plt.title("Caudal y velocidades")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{args.save}_flows.png", dpi=args.dpi)

        # --- Figura 2: Pérdidas de presión
        plt.figure(figsize=(9, 5))
        plt.plot(t, DP_sys, label="ΔP sistema [Pa]")
        plt.plot(t, DP_net, label="ΔP red ductos [Pa]")
        plt.plot(t, DP_filt, label="ΔP filtro [Pa]")
        plt.xlabel("Tiempo [s]")
        plt.title("Pérdidas de presión")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{args.save}_pressure.png", dpi=args.dpi)

        # --- Figura 3: Concentraciones
        plt.figure(figsize=(9, 5))
        plt.plot(t, C_v, label="Vapor [mg/m³]")
        plt.plot(t, C_p, label="Aerosol [mg/m³]")
        plt.plot(t, C_tot, label="Total [mg/m³]")
        # Límites (si existen en config)
        if cfg.criteria and cfg.criteria.C_limit_mg_m3 is not None:
            import numpy as np
            plt.plot(t, [cfg.criteria.C_limit_mg_m3] * len(t), "--", label="Límite [mg/m³]")
        plt.xlabel("Tiempo [s]")
        plt.title("Concentraciones en cabina")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{args.save}_concentrations.png", dpi=args.dpi)

        # --- Figura 4: Potencia y energía acumulada
        plt.figure(figsize=(9, 5))
        plt.plot(t, P, label="Potencia eléctrica [W]")
        plt.plot(t, energy_kWh, label="Energía acumulada [kWh]")
        plt.xlabel("Tiempo [s]")
        plt.title("Potencia y energía")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{args.save}_power_energy.png", dpi=args.dpi)

        # 5) Mostrar en pantalla si no se pidió guardar
        if not args.save:
            plt.show()

        # 6) Mensaje de resumen útil en consola
        print("\n=== Resumen (para referencia) ===")
        print(f"Energía total [kWh]: {summary.get('energy_kWh', 0.0):.4f}")
        print(f"ΔP sistema máx [Pa]: {summary.get('DP_system_Pa_max', 0.0):.2f}")
        print(f"ΔP filtro final [Pa]: {summary.get('DP_filter_Pa_final', 0.0):.2f}")
        print(f"C_total máx [mg/m³]: {summary.get('C_total_mg_m3_max', 0.0):.3f}")
        print(f"C_total prom [mg/m³]: {summary.get('C_total_mg_m3_avg', 0.0):.3f}")

    except FileNotFoundError:
        print(f"Error: no se encontró el archivo de configuración '{args.config}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("Error al graficar resultados:", str(e), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()