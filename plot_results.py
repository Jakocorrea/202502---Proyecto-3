#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_results.py
Lee la configuración, ejecuta la simulación (sim_logic.Simulator) y:
  - Muestra TABLAS resumen en consola (promedios, picos, balances, criterios).
  - Muestra TABLAS por especie (h_m, W(L), J_pared al final; masa descargada).
  - Grafica series de tiempo (C_v, C_p, C_tot, ppm, Q, V_face, v_duct, DP, P).
  - Grafica por especie h_m(t), W_L(t) y J_pared(t).

Uso:
  python plot_results.py -c example_config.json
  python plot_results.py -c example_config.json --dt 0.5 --t_end 1200 --save_dir out --no_show
"""

from typing import Dict, List, Optional
import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim_logic import config_from_json, Simulator


# -------------------- Utilidades --------------------

def sdiv(num: float, den: float, default: float = 0.0) -> float:
    return num/den if den not in (0, 0.0) else default

def ensure_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return path

def df_save(df: pd.DataFrame, path: Optional[str], filename: str):
    if path:
        f = os.path.join(path, filename)
        df.to_csv(f, index=False)

def fig_save(fig: plt.Figure, path: Optional[str], filename: str, dpi: int):
    if path:
        f = os.path.join(path, filename)
        fig.savefig(f, dpi=dpi, bbox_inches="tight")

def print_table(title: str, df: pd.DataFrame, max_rows: int = 50):
    print("\n" + "="*80)
    print(title)
    print("-"*80)
    with pd.option_context("display.max_rows", max_rows,
                           "display.max_columns", 200,
                           "display.width", 120):
        print(df.to_string(index=False))


# -------------------- Gráficas --------------------

def plot_series(df: pd.DataFrame, x: str, y: List[str], ylabel: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in y:
        if col in df.columns:
            ax.plot(df[x], df[col], label=col)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(y) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_species_series(df: pd.DataFrame, species: List[str], base_key: str,
                        x_key: str, ylabel: str, title: str) -> plt.Figure:
    """base_key: 'hm_', 'W_L_', 'Jwall_', 'C_'"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nm in species:
        col = f"{base_key}{nm}" + ("_m_s" if base_key == "hm_" else "_mg_m3" if base_key in ("W_L_", "C_") else "_mg_m2_s")
        if col in df.columns:
            ax.plot(df[x_key], df[col], label=nm)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# -------------------- Principal --------------------

def main():
    parser = argparse.ArgumentParser(description="Graficador/Tabulador para la simulación de cabina.")
    parser.add_argument("-c", "--config", type=str, default="example_config.json", help="Ruta del JSON de configuración.")
    parser.add_argument("--dt", type=float, default=None, help="Paso de tiempo [s] (override).")
    parser.add_argument("--t_end", type=float, default=None, help="Tiempo final [s] (override).")
    parser.add_argument("--q_fixed", type=float, default=None, help="Caudal fijo [m3/s] (override).")
    parser.add_argument("--save_dir", type=str, default=None, help="Carpeta para guardar CSV/PNG (opcional).")
    parser.add_argument("--dpi", type=int, default=150, help="DPI para guardar figuras.")
    parser.add_argument("--no_show", action="store_true", help="No mostrar ventanas interactivas (solo guardar).")
    args = parser.parse_args()

    out_dir = ensure_dir(args.save_dir)

    # Cargar config y ajustar overrides
    cfg = config_from_json(args.config)
    if args.dt is not None:      cfg.sim.dt_s = args.dt
    if args.t_end is not None:   cfg.sim.t_end_s = args.t_end
    if args.q_fixed is not None: cfg.sim.Q_fixed_m3_s = args.q_fixed

    sim = Simulator(cfg)
    rows: List[Dict[str, float]] = sim.run()  # usa dt/t_end de cfg

    if not rows:
        # Si por alguna razón no hubo pasos, tomar un snapshot estático
        rows = [sim.step(0.0)]

    df = pd.DataFrame(rows)
    # calcular dt por fila para integrales
    df["dt"] = df["t_s"].diff().fillna(df["t_s"].clip(lower=cfg.sim.dt_s))
    df["dt"] = df["dt"].clip(lower=0.0)

    # ------------------ TABLAS RESUMEN ------------------

    # Promedios
    avg_Q      = df["Q_m3_s"].mean() if "Q_m3_s" in df else 0.0
    avg_vface  = df["V_face_m_s"].mean() if "V_face_m_s" in df else 0.0
    avg_vduct  = df["v_duct_m_s"].mean() if "v_duct_m_s" in df else 0.0

    # Picos
    peak_Ctot  = df["C_tot_mg_m3"].max() if "C_tot_mg_m3" in df else 0.0
    t_peak_C   = float(df.loc[df["C_tot_mg_m3"].idxmax(), "t_s"]) if "C_tot_mg_m3" in df else 0.0
    peak_ppm   = df["ppm_voc"].max() if "ppm_voc" in df else 0.0
    t_peak_ppm = float(df.loc[df["ppm_voc"].idxmax(), "t_s"]) if "ppm_voc" in df else 0.0

    # Energía (Wh)
    e_Wh = (df["P_electrica_W"] * df["dt"] / 3600.0).sum() if "P_electrica_W" in df else 0.0

    # Balances integrados
    eta_f = max(0.0, min(1.0, cfg.filter.eta_filter))
    ex_vap_kg = (df["Q_m3_s"] * (df["C_v_mg_m3"]/1e6) * df["dt"]).sum() if {"Q_m3_s","C_v_mg_m3"}.issubset(df.columns) else 0.0
    ex_aer_kg = ((1.0 - eta_f) * df["Q_m3_s"] * (df["C_p_mg_m3"]/1e6) * df["dt"]).sum() if {"Q_m3_s","C_p_mg_m3"}.issubset(df.columns) else 0.0
    cap_filt_kg = (eta_f * df["Q_m3_s"] * (df["C_p_mg_m3"]/1e6) * df["dt"]).sum() if {"Q_m3_s","C_p_mg_m3"}.issubset(df.columns) else 0.0

    # Tabla resumen general
    summary_df = pd.DataFrame([{
        "Q_prom [m3/s]": avg_Q,
        "V_face_prom [m/s]": avg_vface,
        "V_duct_prom [m/s]": avg_vduct,
        "C_tot_pico [mg/m3]": peak_Ctot,
        "t_pico_C_tot [s]": t_peak_C,
        "VOC_pico [ppm]": peak_ppm,
        "t_pico_ppm [s]": t_peak_ppm,
        "E_electrica [Wh]": e_Wh,
        "Vapor_descarga [kg]": ex_vap_kg,
        "Aerosol_descarga [kg]": ex_aer_kg,
        "Capturado_filtro [kg]": cap_filt_kg
    }])
    print_table("RESUMEN GENERAL", summary_df.round(4))
    df_save(summary_df, out_dir, "summary.csv")

    # ------------------ TABLAS POR ESPECIE ------------------

    sp_names = [sp.name for sp in cfg.mixture.species]

    # (1) Masa descargada a ducto (vapor) por especie
    mass_rows = []
    for nm in sp_names:
        col = f"C_{nm}_mg_m3"
        if col in df:
            m_kg = (df["Q_m3_s"] * (df[col]/1e6) * df["dt"]).sum()
            mass_rows.append({"Especie": nm, "Masa_vapor_descargada [kg]": m_kg})
    mass_df = pd.DataFrame(mass_rows)
    print_table("MASA DESCARGADA POR ESPECIE (vapor)", mass_df.round(6))
    df_save(mass_df, out_dir, "species_mass.csv")

    # (2) h_m, W(L) y J_pared en el último paso
    last = df.iloc[-1]
    xfer_rows = []
    for nm in sp_names:
        row = {
            "Especie": nm,
            "h_m [m/s]": float(last.get(f"hm_{nm}_m_s", np.nan)),
            "W(L) [mg/m3]": float(last.get(f"W_L_{nm}_mg_m3", np.nan)),
            "J_pared [mg/m2/s]": float(last.get(f"Jwall_{nm}_mg_m2_s", np.nan))
        }
        xfer_rows.append(row)
    xfer_df = pd.DataFrame(xfer_rows)
    print_table("TRANSFERENCIA (snapshot final)", xfer_df.round(6))
    df_save(xfer_df, out_dir, "species_transfer_final.csv")

    # (3) Criterios (si existen flags)
    crit_cols = [c for c in df.columns if c.startswith("ok_")]
    if crit_cols:
        crit_last = pd.DataFrame([{"criterio": c, "cumple": bool(last[c])} for c in crit_cols])
        print_table("CRITERIOS (snapshot final)", crit_last)
        df_save(crit_last, out_dir, "criteria_snapshot.csv")

    # Guardar el timeseries completo (opcional)
    df_save(df, out_dir, "timeseries.csv")

    # ------------------ GRÁFICAS ------------------

    figs: List[plt.Figure] = []

    # Concentraciones
    if {"C_v_mg_m3","C_p_mg_m3","C_tot_mg_m3"}.issubset(df.columns):
        f = plot_series(df, "t_s", ["C_v_mg_m3", "C_p_mg_m3", "C_tot_mg_m3"],
                        "Concentración [mg/m3]", "Concentraciones en el tiempo")
        figs.append(("concentraciones.png", f))

    # ppm VOC
    if "ppm_voc" in df:
        f = plot_series(df, "t_s", ["ppm_voc"], "VOC [ppm]", "VOC (ppmv) en el tiempo")
        figs.append(("voc_ppm.png", f))

    # Caudal
    if "Q_m3_s" in df:
        f = plot_series(df, "t_s", ["Q_m3_s"], "Q [m3/s]", "Caudal en el tiempo")
        figs.append(("Q.png", f))

    # Velocidades
    cols_v = [c for c in ["V_face_m_s", "v_duct_m_s"] if c in df]
    if cols_v:
        f = plot_series(df, "t_s", cols_v, "Velocidad [m/s]", "Velocidades características")
        figs.append(("velocidades.png", f))

    # Presiones y Potencia
    if "DP_system_Pa" in df:
        f = plot_series(df, "t_s", ["DP_system_Pa"], "ΔP sistema [Pa]", "Pérdida de presión del sistema")
        figs.append(("dp_system.png", f))
    if "P_electrica_W" in df:
        f = plot_series(df, "t_s", ["P_electrica_W"], "Potencia [W]", "Consumo eléctrico")
        figs.append(("power.png", f))

    # Por especie: h_m(t), W_L(t), J_pared(t)
    if sp_names:
        f = plot_species_series(df, sp_names, "hm_", "t_s", "h_m [m/s]", "Coeficientes de transferencia de masa")
        figs.append(("hm_species.png", f))
        f = plot_species_series(df, sp_names, "W_L_", "t_s", "W(L) [mg/m3]", "Concentración a la salida por especie")
        figs.append(("WL_species.png", f))
        f = plot_species_series(df, sp_names, "Jwall_", "t_s", "J_pared [mg/m2/s]", "Flujo a pared por especie (aprox.)")
        figs.append(("Jwall_species.png", f))

    # Guardar
    for fname, fig in figs:
        fig_save(fig, out_dir, fname, dpi=args.dpi)

    # Mostrar
    if not args.no_show:
        plt.show()
    else:
        # Cerrar figuras si no se muestran
        for _, fig in figs:
            plt.close(fig)

if __name__ == "__main__":
    main()
