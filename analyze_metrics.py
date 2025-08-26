#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_metrics.py
Comparativos y barridos de métricas para la cabina:

(1) Caudal Q vs. concentración de partículas en aire (C_p)  → scatter + promedio por bins
(2) Potencia eléctrica vs. caudal Q                         → scatter + promedio por bins
(3) Transferencia de masa (h_m por especie) vs. caudal Q    → líneas por especie
(4) Consumo eléctrico vs. número de ventiladores (n)        → barrido con fans en PARALELO
    - Se reescala la curva del ventilador: DP(Q) = DP_base(Q/n) → equivale a n fans en paralelo
    - Para cada n se corre la simulación completa y se integra energía (Wh)

Uso:
  python analyze_metrics.py -c example_config.json --save_dir out_metrics --no_show
  python analyze_metrics.py -c example_config_B_cleaning_vfd.json -nmax 4
  python analyze_metrics.py -c example_config.json --dt 0.5 --t_end 1200 --nlist 1 2 3
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import argparse
import copy
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from sim_logic import config_from_json, Simulator


# -------------------- Estilo --------------------

def setup_matplotlib_style():
    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "grid.color": "0.5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.3,
    })

def beautify_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str, legend: bool = True):
    ax.set_title(title, pad=10, fontweight="semibold")
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.18)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        h, l = ax.get_legend_handles_labels()
        if h:
            ncol = 1 if len(l) <= 5 else 2
            ax.legend(loc="best", frameon=False, ncol=ncol)


# -------------------- Utilidades de guardado --------------------

def ensure_writable_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".write_test.tmp"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return path
    except Exception:
        print(f"[WARN] No se pudo escribir en: {path.resolve()}")
        fallback = Path.cwd() / "out_metrics"
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


# -------------------- Core helpers --------------------

def run_to_df(cfg) -> pd.DataFrame:
    sim = Simulator(cfg)
    rows: List[Dict[str, float]] = sim.run()
    if not rows:
        rows = [sim.step(0.0)]
    df = pd.DataFrame(rows)
    if "t_s" not in df:
        df["t_s"] = np.arange(len(df)) * cfg.sim.dt_s
    dt = df["t_s"].diff().fillna(cfg.sim.dt_s).clip(lower=0.0)
    df["dt"] = dt
    return df

def bins_avg(x: np.ndarray, y: np.ndarray, nbins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Promedio por bins en x (útil para ver tendencia)."""
    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([])
    x = np.asarray(x); y = np.asarray(y)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.array([]), np.array([])
    edges = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    yb = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (x >= edges[i]) & (x < edges[i+1])
        if mask.any():
            yb[i] = np.nanmean(y[mask])
    return centers, yb

def scale_fan_curve_parallel(curve: Dict[str, List[float]], n: int) -> Dict[str, List[float]]:
    """
    Fans en paralelo:
      - A igual ΔP, los caudales se suman: Q_total = n * Q_fan(ΔP)
      - Equivalente: DP_n(Q) = DP_1(Q/n)  => escalar eje Q por n
    """
    if n <= 1:
        return {"Q": list(curve["Q"]), "DP": list(curve["DP"])}
    return {"Q": [q * n for q in curve["Q"]], "DP": list(curve["DP"])}


# -------------------- Plots solicitados --------------------

def plot_Q_vs_Cp(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if {"Q_m3_s", "C_p_mg_m3"}.issubset(df.columns):
        ax.scatter(df["Q_m3_s"], df["C_p_mg_m3"], s=14, alpha=0.55, label="datos (t)")
        # Tendencia por bins
        xc, yc = bins_avg(df["Q_m3_s"].values, df["C_p_mg_m3"].values, nbins=24)
        if len(xc):
            ax.plot(xc, yc, linewidth=2.0, label="promedio por bins")
    beautify_axes(ax, "Caudal vs. Concentración de Partículas", "Q [m³/s]", "C_p [mg/m³]", legend=True)
    return fig

def plot_P_vs_Q(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if {"Q_m3_s", "P_electrica_W"}.issubset(df.columns):
        ax.scatter(df["Q_m3_s"], df["P_electrica_W"], s=14, alpha=0.55, label="datos (t)")
        xc, yc = bins_avg(df["Q_m3_s"].values, df["P_electrica_W"].values, nbins=24)
        if len(xc):
            ax.plot(xc, yc, linewidth=2.0, label="promedio por bins")
    beautify_axes(ax, "Potencia eléctrica vs. Caudal", "Q [m³/s]", "P_electrica [W]", legend=True)
    return fig

def plot_hm_vs_Q(df: pd.DataFrame, species: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    plotted = False
    for nm in species:
        col = f"hm_{nm}_m_s"
        if col in df.columns:
            ax.plot(df["Q_m3_s"], df[col], label=nm, linewidth=1.8, alpha=0.9)
            plotted = True
    beautify_axes(ax, "Transferencia de masa (h_m) vs. Caudal", "Q [m³/s]", "h_m [m/s]", legend=plotted)
    return fig

def sweep_power_vs_n_fans(cfg, n_list: List[int]) -> pd.DataFrame:
    rows = []
    for n in n_list:
        cfg_n = copy.deepcopy(cfg)
        cfg_n.fan.curve = scale_fan_curve_parallel(cfg.fan.curve, n)
        # Mantener mismas eficiencias; VFD/cleaning permanecen si están en el JSON
        df_n = run_to_df(cfg_n)
        # Energía y promedios
        e_Wh = (df_n.get("P_electrica_W", 0.0) * df_n["dt"] / 3600.0).sum() if "P_electrica_W" in df_n else 0.0
        avg_P = df_n["P_electrica_W"].mean() if "P_electrica_W" in df_n else 0.0
        avg_Q = df_n["Q_m3_s"].mean() if "Q_m3_s" in df_n else 0.0
        rows.append({"n_fans": n, "E_Wh": e_Wh, "P_avg_W": avg_P, "Q_avg_m3_s": avg_Q})
    return pd.DataFrame(rows)

def plot_energy_vs_n(df_n: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(df_n["n_fans"], df_n["E_Wh"], marker="o", linewidth=2.0, label="Energía [Wh]")
    beautify_axes(ax, "Consumo eléctrico vs. número de ventiladores (paralelo)", "n (ventiladores)", "Energía [Wh]", legend=True)
    return fig

def plot_poweravg_vs_n(df_n: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(df_n["n_fans"], df_n["P_avg_W"], marker="o", linewidth=2.0, label="Potencia promedio [W]")
    beautify_axes(ax, "Potencia promedio vs. número de ventiladores (paralelo)", "n (ventiladores)", "P_prom [W]", legend=True)
    return fig


# -------------------- Main --------------------

def main():
    setup_matplotlib_style()

    parser = argparse.ArgumentParser(description="Análisis comparativo de métricas de la cabina.")
    parser.add_argument("-c", "--config", type=str, default="example_config_B_cleaning_vfd.json",
                        help="Ruta del JSON de configuración.")
    parser.add_argument("--dt", type=float, default=None, help="Paso de tiempo [s] (override).")
    parser.add_argument("--t_end", type=float, default=None, help="Tiempo final [s] (override).")
    parser.add_argument("--q_fixed", type=float, default=None, help="Caudal fijo [m3/s] (override).")
    parser.add_argument("--save_dir", type=str, default="out_metrics", help="Carpeta para guardar CSV/PNG.")
    parser.add_argument("--no_save", action="store_true", help="No guardar archivos, solo mostrar.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI para guardar figuras.")
    parser.add_argument("--no_show", action="store_true", help="No mostrar ventanas interactivas.")
    parser.add_argument("--nmin", type=int, default=1, help="n mínimo de ventiladores (paralelo).")
    parser.add_argument("--nmax", type=int, default=4, help="n máximo de ventiladores (paralelo).")
    parser.add_argument("--nlist", type=int, nargs="*", default=None, help="Lista explícita de n (e.g., --nlist 1 2 3 4).")
    args = parser.parse_args()

    out_dir: Optional[Path] = None
    if not args.no_save:
        out_dir = ensure_writable_dir(Path(args.save_dir))

    # Cargar config base + overrides
    cfg = config_from_json(args.config)
    if args.dt is not None:      cfg.sim.dt_s = args.dt
    if args.t_end is not None:   cfg.sim.t_end_s = args.t_end
    if args.q_fixed is not None: cfg.sim.Q_fixed_m3_s = args.q_fixed

    # --- Corrida base para las 3 primeras gráficas ---
    df = run_to_df(cfg)
    species = [sp.name for sp in cfg.mixture.species]

    # 1) Q vs C_p
    fig_q_cp = plot_Q_vs_Cp(df)
    fig_save(fig_q_cp, out_dir, "Q_vs_Cp.png", dpi=args.dpi)
    df_qcp = df[["t_s", "Q_m3_s", "C_p_mg_m3"]].copy() if {"Q_m3_s","C_p_mg_m3"}.issubset(df.columns) else pd.DataFrame()
    if not df_qcp.empty:
        df_save(df_qcp, out_dir, "Q_vs_Cp_timeseries.csv")

    # 2) P vs Q
    fig_p_q = plot_P_vs_Q(df)
    fig_save(fig_p_q, out_dir, "P_vs_Q.png", dpi=args.dpi)
    df_pq = df[["t_s", "Q_m3_s", "P_electrica_W"]].copy() if {"Q_m3_s","P_electrica_W"}.issubset(df.columns) else pd.DataFrame()
    if not df_pq.empty:
        df_save(df_pq, out_dir, "P_vs_Q_timeseries.csv")

    # 3) h_m vs Q (por especie)
    fig_hm_q = plot_hm_vs_Q(df, species)
    fig_save(fig_hm_q, out_dir, "hm_vs_Q.png", dpi=args.dpi)
    # Guardar h_m vs Q (pivotado)
    hm_cols = [f"hm_{nm}_m_s" for nm in species if f"hm_{nm}_m_s" in df.columns]
    if hm_cols and "Q_m3_s" in df.columns:
        df_hm_q = df[["t_s", "Q_m3_s"] + hm_cols].copy()
        df_save(df_hm_q, out_dir, "hm_vs_Q_timeseries.csv")

    # --- 4) Barrido: consumo vs número de ventiladores (paralelo) ---
    if args.nlist is not None and len(args.nlist) > 0:
        n_list = sorted(set(int(n) for n in args.nlist if int(n) >= 1))
    else:
        n_list = list(range(max(1, args.nmin), max(args.nmin, args.nmax) + 1))

    df_n = sweep_power_vs_n_fans(cfg, n_list)
    if not df_n.empty:
        # Guardar tabla
        df_save(df_n, out_dir, "consumo_vs_n_fans.csv")
        # Gráficas
        fig_en = plot_energy_vs_n(df_n)
        fig_save(fig_en, out_dir, "energia_vs_n_fans.png", dpi=args.dpi)
        fig_pw = plot_poweravg_vs_n(df_n)
        fig_save(fig_pw, out_dir, "potencia_prom_vs_n_fans.png", dpi=args.dpi)

    # Mostrar / cerrar
    if not args.no_show:
        plt.show()
    else:
        plt.close("all")

    if out_dir is None:
        print("\n[INFO] No se guardaron archivos (usar --save_dir o quitar --no_save).")

if __name__ == "__main__":
    main()
