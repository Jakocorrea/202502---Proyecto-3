import argparse
import json
import sys
from typing import Tuple, Dict, Any

from sim_logic import config_from_json, Simulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ejecuta la simulación de la cabina de pintura usando un archivo JSON de configuración."
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
        "--print_rows", type=int, default=5,
        help="Número de filas iniciales de resultado a imprimir (por defecto: 5)"
    )
    return p.parse_args()


def load_sim_params(path: str, override_dt: float | None, override_t_end: float | None) -> Tuple[float, float, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    simsec = raw.get("sim", {}) if isinstance(raw, dict) else {}
    dt = float(simsec.get("dt_s", 1.0)) if override_dt is None else float(override_dt)
    t_end = float(simsec.get("t_end_s", 3600.0)) if override_t_end is None else float(override_t_end)
    return dt, t_end, raw


def evaluate_criteria(cfg, results: list[dict]) -> dict:
    """Evalúa criterios básicos (si están definidos en el JSON)."""
    out = {}
    if not results:
        return out

    # Métricas agregadas
    min_v_face = min(r["V_face_m_s"] for r in results)
    min_v_duct = min(r["v_duct_m_s"] for r in results)
    max_C_tot = max(r["C_tot_mg_m3"] for r in results)

    crit = cfg.criteria
    if crit.V_face_obj is not None:
        out["V_face_obj"] = {
            "target_m_s": crit.V_face_obj,
            "min_observed_m_s": min_v_face,
            "ok": min_v_face >= crit.V_face_obj
        }
    if crit.v_duct_obj is not None:
        out["v_duct_obj"] = {
            "target_m_s": crit.v_duct_obj,
            "min_observed_m_s": min_v_duct,
            "ok": min_v_duct >= crit.v_duct_obj
        }
    if crit.C_limit_mg_m3 is not None:
        out["C_limit_mg_m3"] = {
            "limit_mg_m3": crit.C_limit_mg_m3,
            "max_observed_mg_m3": max_C_tot,
            "ok": max_C_tot <= crit.C_limit_mg_m3
        }
    return out


def main():
    args = parse_args()

    try:
        # 1) Cargar parámetros de simulación (dt, t_end) desde JSON con posibles overrides
        dt_s, t_end_s, raw_json = load_sim_params(args.config, args.dt, args.t_end)

        # 2) Construir configuración del sistema (sim_logic ya gestiona compatibilidad fan/fan_eff)
        cfg = config_from_json(args.config)

        # 3) Ejecutar simulación
        sim = Simulator(cfg)
        results, summary = sim.run(t_end_s=t_end_s, dt_s=dt_s)

        # 4) Imprimir resumen principal
        print("\n=== RESUMEN DE SIMULACIÓN ===")
        print(f"Archivo de configuración: {args.config}")
        print(f"t_end_s = {t_end_s:.3f}  |  dt_s = {dt_s:.3f}")
        print(f"Volumen cabina [m³]: {cfg.geometry.get_volume():.3f}")
        print(f"Ductos: {len(cfg.ducts)}  |  Área de cara [m²]: {cfg.geometry.area_face:.3f}")
        print("\n— Energía —")
        print(f"Energía consumida [kWh]: {summary.get('energy_kWh', 0.0):.4f}")
        print("\n— Presiones —")
        print(f"ΔP sistema máx [Pa]: {summary.get('DP_system_Pa_max', 0.0):.2f}")
        print(f"ΔP filtro final [Pa]: {summary.get('DP_filter_Pa_final', 0.0):.2f}")
        print("\n— Concentraciones —")
        print(f"C_total máx [mg/m³]: {summary.get('C_total_mg_m3_max', 0.0):.3f}")
        print(f"C_total prom [mg/m³]: {summary.get('C_total_mg_m3_avg', 0.0):.3f}")
        print(f"Masa capturada final en filtro [kg]: {summary.get('M_captured_kg_final', 0.0):.6f}")

        # 5) Evaluación contra criterios (si existen en el JSON)
        checks = evaluate_criteria(cfg, results)
        if checks:
            print("\n=== EVALUACIÓN DE CRITERIOS ===")
            for k, v in checks.items():
                if k == "V_face_obj":
                    print(f"Velocidad de cara — objetivo: {v['target_m_s']:.3f} m/s | "
                          f"mín obs: {v['min_observed_m_s']:.3f} m/s | OK: {v['ok']}")
                elif k == "v_duct_obj":
                    print(f"Velocidad en ducto — objetivo: {v['target_m_s']:.3f} m/s | "
                          f"mín obs: {v['min_observed_m_s']:.3f} m/s | OK: {v['ok']}")
                elif k == "C_limit_mg_m3":
                    print(f"Concentración — límite: {v['limit_mg_m3']:.3f} mg/m³ | "
                          f"máx obs: {v['max_observed_mg_m3']:.3f} mg/m³ | OK: {v['ok']}")

        # 6) Muestra de las primeras filas
        n = max(0, int(args.print_rows))
        if n > 0 and results:
            cols = ["time_s", "Q_m3_s", "V_face_m_s", "v_duct_m_s",
                    "C_tot_mg_m3", "DP_system_Pa", "P_e_W"]
            print("\n=== PRIMERAS FILAS ===")
            header = " | ".join(f"{c:>14}" for c in cols)
            print(header)
            print("-" * len(header))
            for row in results[:n]:
                print(" | ".join(f"{row.get(c, 0):14.6f}" if isinstance(row.get(c, 0), (int, float))
                                 else f"{str(row.get(c, '')):>14}" for c in cols))

    except FileNotFoundError:
        print(f"Error: no se encontró el archivo de configuración '{args.config}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("Error al ejecutar la simulación:", str(e), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()