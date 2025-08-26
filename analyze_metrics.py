#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_metrics.py
Comparativos y barridos de métricas con DOS juegos de gráficas:
 - Gráfica BASE (datos + promedios por bins / líneas)
 - Gráfica POLI (ajuste por regresión polinómica en figura separada, SIN datos originales)

Métricas:
(1) Caudal Q vs. concentración de partículas en aire (C_p)
(2) Potencia eléctrica vs. caudal Q
(3) Transferencia de masa (h_m por especie) vs. caudal Q
(4) Consumo eléctrico vs. número de ventiladores (n) [paralelo]
    - DP_n(Q) = DP_base(Q/n)  (equivale a n ventiladores en paralelo)
    - Para cada n se corre la simulación completa e integra energía (Wh)

Uso:
  python analyze_metrics.py -c example_config.json --save_dir out_metrics --no_show
  python analyze_metrics.py -c example_config_B_cleaning_vfd.json -nmax 4
  python analyze_metrics.py -c example_config.json --dt 0.5 --t_end 1200 --nlist 1 2 3 --polydeg 3
  python analyze_metrics.py -c example_config.json --no_poly   # desactiva las figuras polinómicas
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
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size == 0:
        return np.array([]), np.array([])
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

def poly_curve(x: np.ndarray, y: np.ndarray, deg: int = 2, npoints: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Ajuste polinómico (con normalización de x); devuelve (xgrid, yhat)."""
    x = np.asarray(x); y = np.asarray(y)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size <= max(deg, 1):
        return np.array([]), np.array([])
    xmean = float(np.nanmean(x))
    xstd  = float(np.nanstd(x))
    if not np.isfinite(xstd) or xstd == 0.0:
        return np.array([]), np.array([])
    xn = (x - xmean) / xstd
    coeffs = np.polyfit(xn, y, deg=deg)
    xgrid = np.linspace(x.min(), x.max(), npoints)
    yhat  = np.polyval(coeffs, (xgrid - xmean) / xstd)
    return xgrid, yhat

def scale_fan_curve_parallel(curve: Dict[str, List[float]], n: int) -> Dict[str, List[float]]:
    """
    Fans en paralelo:
      - A igual ΔP, los caudales se suman: Q_total = n * Q_fan(ΔP)
      - Equivalente: DP_n(Q) = DP_1(Q/n)  => escalar eje Q por n
    """
    if n <= 1:
        return {"Q": list(curve["Q"]), "DP": list(curve["DP"])}
    return {"Q": [q * n for q in curve["Q"]], "DP": list(curve["DP"])}


# -------------------- Plots BASE + POLI (figuras separadas) --------------------

def plot_Q_vs_Cp_base(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if {"Q_m3_s", "C_p_mg_m3"}.issubset(df.columns):
        ax.scatter(df["Q_m3_s"], df["C_p_mg_m3"], s=14, alpha=0.55, label="datos (t)")
        xc, yc = bins_avg(df["Q_m3_s"].values, df["C_p_mg_m3"].values, nbins=24)
        if len(xc):
            ax.plot(xc, yc, linewidth=2.0, label="promedio por bins")
    beautify_axes(ax, "Caudal vs Cencentración", "Q [m³/s]", "C_p [mg/m³]", legend=True)
    return fig

def plot_Q_vs_Cp_poly(df: pd.DataFrame, polydeg: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if {"Q_m3_s", "C_p_mg_m3"}.issubset(df.columns):
        xg, yh = poly_curve(df["Q_m3_s"].values, df["C_p_mg_m3"].values, deg=polydeg)
        if len(xg):
            ax.plot(xg, yh, linewidth=2.4, label=f"regresión polinómica (deg={polydeg})")
    beautify_axes(ax, "Caudal vs Concentración (POLI)", "Q [m³/s]", "C_p [mg/m³]", legend=True)
    return fig

def plot_P_vs_Q_base(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if {"Q_m3_s", "P_electrica_W"}.issubset(df.columns):
        ax.scatter(df["Q_m3_s"], df["P_electrica_W"], s=14, alpha=0.55, label="datos (t)")
        xc, yc = bins_avg(df["Q_m3_s"].values, df["P_electrica_W"].values, nbins=24)
        if len(xc):
            ax.plot(xc, yc, linewidth=2.0, label="promedio por bins")
    beautify_axes(ax, "P eléctrica vs Caudal (BASE)", "Q [m³/s]", "P_electrica [W]", legend=True)
    return fig

def plot_P_vs_Q_poly(df: pd.DataFrame, polydeg: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if {"Q_m3_s", "P_electrica_W"}.issubset(df.columns):
        xg, yh = poly_curve(df["Q_m3_s"].values, df["P_electrica_W"].values, deg=polydeg)
        if len(xg):
            ax.plot(xg, yh, linewidth=2.4, label=f"regresión polinómica (deg={polydeg})")
    beautify_axes(ax, "P eléctrica vs Causal (POLI)", "Q [m³/s]", "P_electrica [W]", legend=True)
    return fig

def plot_hm_vs_Q_base(df: pd.DataFrame, species: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    plotted = False
    for nm in species:
        col = f"hm_{nm}_m_s"
        if col in df.columns:
            ax.plot(df["Q_m3_s"], df[col], label=f"{nm}", linewidth=1.4, alpha=0.85)
            plotted = True
    beautify_axes(ax, "Flujo de masa vs Caudal", "Q [m³/s]", "h_m [m/s]", legend=plotted)
    return fig

def plot_hm_vs_Q_poly(df: pd.DataFrame, species: List[str], polydeg: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    plotted = False
    for nm in species:
        col = f"hm_{nm}_m_s"
        if col in df.columns:
            xg, yh = poly_curve(df["Q_m3_s"].values, df[col].values, deg=polydeg)
            if len(xg):
                ax.plot(xg, yh, linewidth=2.0, label=f"{nm} (deg={polydeg})")
                plotted = True
    beautify_axes(ax, "Flujo de masa vs Caudal (POLI)", "Q [m³/s]", "h_m [m/s]", legend=plotted)
    return fig

def sweep_power_vs_n_fans(cfg, n_list: List[int]) -> pd.DataFrame:
    rows = []
    for n in n_list:
        cfg_n = copy.deepcopy(cfg)
        cfg_n.fan.curve = scale_fan_curve_parallel(cfg.fan.curve, n)
        df_n = run_to_df(cfg_n)
        e_Wh = (df_n.get("P_electrica_W", 0.0) * df_n["dt"] / 3600.0).sum() if "P_electrica_W" in df_n else 0.0
        avg_P = df_n["P_electrica_W"].mean() if "P_electrica_W" in df_n else 0.0
        avg_Q = df_n["Q_m3_s"].mean() if "Q_m3_s" in df_n else 0.0
        rows.append({"n_fans": n, "E_Wh": e_Wh, "P_avg_W": avg_P, "Q_avg_m3_s": avg_Q})
    return pd.DataFrame(rows)

def plot_energy_vs_n_base(df_n: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(df_n["n_fans"], df_n["E_Wh"], marker="o", linewidth=2.0, label="Energía [Wh]")
    beautify_axes(ax, "Energía vs n (BASE, paralelo)", "n (ventiladores)", "Energía [Wh]", legend=True)
    return fig

def plot_energy_vs_n_poly(df_n: pd.DataFrame, polydeg: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    if len(df_n) > max(polydeg, 1):
        xg, yh = poly_curve(df_n["n_fans"].values.astype(float), df_n["E_Wh"].values, deg=polydeg, npoints=300)
        if len(xg):
            ax.plot(xg, yh, linewidth=2.4, label=f"polinomio (deg={polydeg})")
    beautify_axes(ax, "Energía vs n (POLI, paralelo)", "n (ventiladores)", "Energía [Wh]", legend=True)
    return fig

def plot_poweravg_vs_n_base(df_n: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(df_n["n_fans"], df_n["P_avg_W"], marker="o", linewidth=2.0, label="Potencia promedio [W]")
    beautify_axes(ax, "Potencia promedio vs n (BASE, paralelo)", "n (ventiladores)", "P_prom [W]", legend=True)
    return fig

def plot_poweravg_vs_n_poly(df_n: pd.DataFrame, polydeg: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    if len(df_n) > max(polydeg, 1):
        xg, yh = poly_curve(df_n["n_fans"].values.astype(float), df_n["P_avg_W"].values, deg=polydeg, npoints=300)
        if len(xg):
            ax.plot(xg, yh, linewidth=2.4, label=f"polinomio (deg={polydeg})")
    beautify_axes(ax, "Potencia promedio vs n (POLI, paralelo)", "n (ventiladores)", "P_prom [W]", legend=True)
    return fig


# -------------------- Main --------------------

def main():
    setup_matplotlib_style()

    parser = argparse.ArgumentParser(description="Análisis comparativo de métricas (BASE y POLI en figuras separadas).")
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

    parser.add_argument("--polydeg", type=int, default=2, help="Grado de la regresión polinómica (>=1).")
    parser.add_argument("--no_poly", action="store_true", help="Desactiva las figuras polinómicas.")
    args = parser.parse_args()

    polydeg = max(1, int(args.polydeg))
    make_poly = not args.no_poly

    out_dir: Optional[Path] = None
    if not args.no_save:
        out_dir = ensure_writable_dir(Path(args.save_dir))

    # Cargar config base + overrides
    cfg = config_from_json(args.config)
    if args.dt is not None:      cfg.sim.dt_s = args.dt
    if args.t_end is not None:   cfg.sim.t_end_s = args.t_end
    if args.q_fixed is not None: cfg.sim.Q_fixed_m3_s = args.q_fixed

    # --- Corrida base para las 3 primeras métricas ---
    df = run_to_df(cfg)
    species = [sp.name for sp in cfg.mixture.species]

    # ---------- (1) Q vs C_p ----------
    fig_qcp_base = plot_Q_vs_Cp_base(df)
    fig_save(fig_qcp_base, out_dir, "Q_vs_Cp_BASE.png", dpi=args.dpi)
    if make_poly:
        fig_qcp_poly = plot_Q_vs_Cp_poly(df, polydeg=polydeg)
        fig_save(fig_qcp_poly, out_dir, "Q_vs_Cp_POLI.png", dpi=args.dpi)
    if {"Q_m3_s","C_p_mg_m3"}.issubset(df.columns):
        df_save(df[["t_s", "Q_m3_s", "C_p_mg_m3"]], out_dir, "Q_vs_Cp_timeseries.csv")

    # ---------- (2) P vs Q ----------
    fig_pq_base = plot_P_vs_Q_base(df)
    fig_save(fig_pq_base, out_dir, "P_vs_Q_BASE.png", dpi=args.dpi)
    if make_poly:
        fig_pq_poly = plot_P_vs_Q_poly(df, polydeg=polydeg)
        fig_save(fig_pq_poly, out_dir, "P_vs_Q_POLI.png", dpi=args.dpi)
    if {"Q_m3_s","P_electrica_W"}.issubset(df.columns):
        df_save(df[["t_s", "Q_m3_s", "P_electrica_W"]], out_dir, "P_vs_Q_timeseries.csv")

    # ---------- (3) h_m vs Q ----------
    fig_hm_base = plot_hm_vs_Q_base(df, species)
    fig_save(fig_hm_base, out_dir, "hm_vs_Q_BASE.png", dpi=args.dpi)
    if make_poly:
        fig_hm_poly = plot_hm_vs_Q_poly(df, species, polydeg=polydeg)
        fig_save(fig_hm_poly, out_dir, "hm_vs_Q_POLI.png", dpi=args.dpi)
    hm_cols = [f"hm_{nm}_m_s" for nm in species if f"hm_{nm}_m_s" in df.columns]
    if hm_cols and "Q_m3_s" in df.columns:
        df_save(df[["t_s", "Q_m3_s"] + hm_cols], out_dir, "hm_vs_Q_timeseries.csv")

    # ---------- (4) Consumo vs número de ventiladores ----------
    if args.nlist is not None and len(args.nlist) > 0:
        n_list = sorted(set(int(n) for n in args.nlist if int(n) >= 1))
    else:
        n_list = list(range(max(1, args.nmin), max(args.nmin, args.nmax) + 1))

    df_n = sweep_power_vs_n_fans(cfg, n_list)
    if not df_n.empty:
        df_save(df_n, out_dir, "consumo_vs_n_fans.csv")

        fig_en_base = plot_energy_vs_n_base(df_n)
        fig_save(fig_en_base, out_dir, "energia_vs_n_fans_BASE.png", dpi=args.dpi)
        if make_poly:
            fig_en_poly = plot_energy_vs_n_poly(df_n, polydeg=polydeg)
            fig_save(fig_en_poly, out_dir, "energia_vs_n_fans_POLI.png", dpi=args.dpi)

        fig_pw_base = plot_poweravg_vs_n_base(df_n)
        fig_save(fig_pw_base, out_dir, "potencia_prom_vs_n_fans_BASE.png", dpi=args.dpi)
        if make_poly:
            fig_pw_poly = plot_poweravg_vs_n_poly(df_n, polydeg=polydeg)
            fig_save(fig_pw_poly, out_dir, "potencia_prom_vs_n_fans_POLI.png", dpi=args.dpi)

    # Mostrar / cerrar
    if not args.no_show:
        plt.show()
    else:
        plt.close("all")

    if out_dir is None:
        print("\n[INFO] No se guardaron archivos (usar --save_dir o quitar --no_save).")

if __name__ == "__main__":
    main()
