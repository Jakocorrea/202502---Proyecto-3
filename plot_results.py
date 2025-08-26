#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_results.py (robusto)
- Ejecuta la simulación y arma un DataFrame con todas las series.
- Imprime tablas en consola.
- GUARDA CSV y PNG de forma robusta (por defecto en ./out).
- Mensajes claros de éxito/fracaso al guardar.

Uso:
  python plot_results.py -c example_config.json
  python plot_results.py -c example_config.json --save_dir resultados --no_show
  python plot_results.py -c example_config.json --dt 0.5 --t_end 1200
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import argparse
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim_logic import config_from_json, Simulator


# -------------------- Utilidades de guardado --------------------

def ensure_writable_dir(path: Path) -> Path:
    """Crea la carpeta si no existe y verifica que sea escribible."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".write_test.tmp"
        with open(test, "w", encoding="utf-8") as f:
            f.write("ok")
        test.unlink(missing_ok=True)
        return path
    except Exception:
        print(f"[WARN] No se pudo escribir en: {path.resolve()}")
        fallback = Path.cwd() / "out"
        print(f"[INFO] Usando carpeta fallback: {fallback.resolve()}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

def safe_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        name = name.replace(ch, "_")
    name = name.replace(" ", "_")
    return name

def df_save(df: pd.DataFrame, path: Optional[Path], filename: str) -> Optional[Path]:
    if path is None:
        return None
    try:
        f = path / safe_filename(filename)
        df.to_csv(f, index=False, encoding="utf-8", float_format="%.6g")
        print(f"[OK] CSV guardado: {f.resolve()}")
        return f
    except Exception as e:
        print(f"[ERROR] No se pudo guardar CSV '{filename}': {e}")
        traceback.print_exc()
        return None

def fig_save(fig: plt.Figure, path: Optional[Path], filename: str, dpi: int) -> Optional[Path]:
    if path is None:
        return None
    try:
        f = path / safe_filename(filename)
        fig.savefig(f, dpi=dpi, bbox_inches="tight")
        print(f"[OK] PNG guardado: {f.resolve()}")
        return f
    except Exception as e:
        print(f"[ERROR] No se pudo guardar figura '{filename}': {e}")
        traceback.print_exc()
        return None

def print_table(title: str, df: pd.DataFrame, max_rows: int = 50):
    print("\n" + "="*80)
    print(title)
    print("-"*80)
    with pd.option_context("display.max_rows", max_rows,
                           "display.max_columns", 200,
                           "display.width", 140):
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
    def colname(nm: str) -> str:
        if base_key == "hm_":
            return f"hm_{nm}_m_s"
        if base_key in ("W_L_", "C_"):
            return f"{base_key}{nm}_mg_m3"
        return f"Jwall_{nm}_mg_m2_s"
    for nm in species:
        col = colname(nm)
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
    parser.add_argument("-c", "--config", type=str, default="example_config_A_no_filter.json", help="Ruta del JSON de configuración.")
    parser.add_argument("--dt", type=float, default=None, help="Paso de tiempo [s] (override).")
    parser.add_argument("--t_end", type=float, default=None, help="Tiempo final [s] (override).")
    parser.add_argument("--q_fixed", type=float, default=None, help="Caudal fijo [m3/s] (override).")
    parser.add_argument("--save_dir", type=str, default="out", help="Carpeta para guardar CSV/PNG. Use --no_save para no guardar.")
    parser.add_argument("--no_save", action="store_true", help="No guardar archivos, solo mostrar.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI para guardar figuras.")
    parser.add_argument("--no_show", action="store_true", help="No mostrar ventanas interactivas (solo guardar/tabla).")
    args = parser.parse_args()

    # Carpeta de guardado (si aplica)
    out_dir: Optional[Path] = None
    if not args.no_save:
        out_dir = ensure_writable_dir(Path(args.save_dir))

    # Cargar config y overrides
    cfg = config_from_json(args.config)
    if args.dt is not None:      cfg.sim.dt_s = args.dt
    if args.t_end is not None:   cfg.sim.t_end_s = args.t_end
    if args.q_fixed is not None: cfg.sim.Q_fixed_m3_s = args.q_fixed

    # Ejecutar simulación
    sim = Simulator(cfg)
    rows: List[Dict[str, float]] = sim.run()

    if not rows:
        rows = [sim.step(0.0)]

    df = pd.DataFrame(rows)
    if "t_s" not in df.columns:
        df["t_s"] = np.arange(len(df)) * cfg.sim.dt_s

    # dt por fila para integrales
    dt_series = df["t_s"].diff().fillna(cfg.sim.dt_s).clip(lower=0.0)
    df["dt"] = dt_series

    # ------------------ TABLAS RESUMEN ------------------

    avg_Q      = df["Q_m3_s"].mean() if "Q_m3_s" in df else 0.0
    avg_vface  = df["V_face_m_s"].mean() if "V_face_m_s" in df else 0.0
    avg_vduct  = df["v_duct_m_s"].mean() if "v_duct_m_s" in df else 0.0

    peak_Ctot  = df["C_tot_mg_m3"].max() if "C_tot_mg_m3" in df else 0.0
    t_peak_C   = float(df.loc[df["C_tot_mg_m3"].idxmax(), "t_s"]) if "C_tot_mg_m3" in df else 0.0
    peak_ppm   = df["ppm_voc"].max() if "ppm_voc" in df else 0.0
    t_peak_ppm = float(df.loc[df["ppm_voc"].idxmax(), "t_s"]) if "ppm_voc" in df else 0.0

    e_Wh = (df.get("P_electrica_W", 0.0) * df["dt"] / 3600.0).sum() if "P_electrica_W" in df else 0.0

    eta_f = float(max(0.0, min(1.0, cfg.filter.eta_filter)))
    ex_vap_kg = (df["Q_m3_s"] * (df["C_v_mg_m3"]/1e6) * df["dt"]).sum() if {"Q_m3_s","C_v_mg_m3"}.issubset(df.columns) else 0.0
    ex_aer_kg = ((1.0 - eta_f) * df["Q_m3_s"] * (df["C_p_mg_m3"]/1e6) * df["dt"]).sum() if {"Q_m3_s","C_p_mg_m3"}.issubset(df.columns) else 0.0
    cap_filt_kg = (eta_f * df["Q_m3_s"] * (df["C_p_mg_m3"]/1e6) * df["dt"]).sum() if {"Q_m3_s","C_p_mg_m3"}.issubset(df.columns) else 0.0

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
    }]).round(6)
    print_table("RESUMEN GENERAL", summary_df)
    df_save(summary_df, out_dir, "summary.csv")

    # ------------------ TABLAS POR ESPECIE ------------------

    sp_names = [sp.name for sp in cfg.mixture.species]

    mass_rows = []
    for nm in sp_names:
        col = f"C_{nm}_mg_m3"
        if col in df:
            m_kg = (df["Q_m3_s"] * (df[col]/1e6) * df["dt"]).sum()
            mass_rows.append({"Especie": nm, "Masa_vapor_descargada [kg]": m_kg})
    mass_df = pd.DataFrame(mass_rows).round(6)
    if not mass_df.empty:
        print_table("MASA DESCARGADA POR ESPECIE (vapor)", mass_df)
        df_save(mass_df, out_dir, "species_mass.csv")

    last = df.iloc[-1]
    xfer_rows = []
    for nm in sp_names:
        xfer_rows.append({
            "Especie": nm,
            "h_m [m/s]": float(last.get(f"hm_{nm}_m_s", np.nan)),
            "W(L) [mg/m3]": float(last.get(f"W_L_{nm}_mg_m3", np.nan)),
            "J_pared [mg/m2/s]": float(last.get(f"Jwall_{nm}_mg_m2_s", np.nan))
        })
    xfer_df = pd.DataFrame(xfer_rows).round(6)
    print_table("TRANSFERENCIA (snapshot final)", xfer_df)
    df_save(xfer_df, out_dir, "species_transfer_final.csv")

    crit_cols = [c for c in df.columns if c.startswith("ok_")]
    if crit_cols:
        crit_last = pd.DataFrame([{"criterio": c, "cumple": bool(last[c])} for c in crit_cols])
        print_table("CRITERIOS (snapshot final)", crit_last)
        df_save(crit_last, out_dir, "criteria_snapshot.csv")

    # Guardar timeseries completo
    df_save(df, out_dir, "timeseries.csv")

    # ------------------ GRÁFICAS ------------------

    figs: List[Tuple[str, plt.Figure]] = []

    if {"C_v_mg_m3","C_p_mg_m3","C_tot_mg_m3"}.issubset(df.columns):
        figs.append((
            "concentraciones.png",
            plot_series(df, "t_s", ["C_v_mg_m3", "C_p_mg_m3", "C_tot_mg_m3"],
                        "Concentración [mg/m3]", "Concentraciones en el tiempo")
        ))

    if "ppm_voc" in df:
        figs.append((
            "voc_ppm.png",
            plot_series(df, "t_s", ["ppm_voc"], "VOC [ppm]", "VOC (ppmv) en el tiempo")
        ))

    if "Q_m3_s" in df:
        figs.append((
            "Q.png",
            plot_series(df, "t_s", ["Q_m3_s"], "Q [m3/s]", "Caudal en el tiempo")
        ))

    cols_v = [c for c in ["V_face_m_s", "v_duct_m_s"] if c in df]
    if cols_v:
        figs.append((
            "velocidades.png",
            plot_series(df, "t_s", cols_v, "Velocidad [m/s]", "Velocidades características")
        ))

    if "DP_system_Pa" in df:
        figs.append((
            "dp_system.png",
            plot_series(df, "t_s", ["DP_system_Pa"], "ΔP sistema [Pa]", "Pérdida de presión del sistema")
        ))

    if "P_electrica_W" in df:
        figs.append((
            "power.png",
            plot_series(df, "t_s", ["P_electrica_W"], "Potencia [W]", "Consumo eléctrico")
        ))

    if sp_names:
        figs.append((
            "hm_species.png",
            plot_species_series(df, sp_names, "hm_", "t_s", "h_m [m/s]", "Coeficientes de transferencia de masa")
        ))
        figs.append((
            "WL_species.png",
            plot_species_series(df, sp_names, "W_L_", "t_s", "W(L) [mg/m3]", "Concentración a la salida por especie")
        ))
        figs.append((
            "Jwall_species.png",
            plot_species_series(df, sp_names, "Jwall_", "t_s", "J_pared [mg/m2/s]", "Flujo a pared por especie (aprox.)")
        ))

    # Guardar figuras
    for fname, fig in figs:
        fig_save(fig, out_dir, fname, dpi=args.dpi)
        if args.no_show:
            plt.close(fig)

    # Mostrar
    if not args.no_show:
        plt.show()

    if out_dir is None:
        print("\n[INFO] No se guardaron archivos (usar --save_dir o quitar --no_save).")

if __name__ == "__main__":
    main()
