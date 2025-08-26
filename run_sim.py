#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Dict, List
from sim_logic import config_from_json, Simulator



def fmt_bool(b: object) -> str:
    return "OK" if bool(b) else "NO"


def sdiv(num: float, den: float, default: float = 0.0) -> float:
    """División segura para evitar ZeroDivisionError."""
    return num/den if den not in (0, 0.0) else default

def main():
    parser = argparse.ArgumentParser(description="Simulación de cabina de pintura (dinámico + especies).")
    parser.add_argument("--config", "-c", type=str, default="example_config_B_cleaning_vfd.json",
                        help="Ruta al archivo JSON de configuración.")
    parser.add_argument("--dt", type=float, default=None, help="Paso de tiempo [s] (override).")
    parser.add_argument("--t_end", type=float, default=None, help="Tiempo final [s] (override).")
    parser.add_argument("--q_fixed", type=float, default=None, help="Caudal fijo [m3/s] (override).")
    parser.add_argument("--print_every", type=int, default=60, help="Frecuencia de impresión [s].")
    args = parser.parse_args()

    cfg = config_from_json(args.config)
    if args.dt is not None:      cfg.sim.dt_s = args.dt
    if args.t_end is not None:   cfg.sim.t_end_s = args.t_end
    if args.q_fixed is not None: cfg.sim.Q_fixed_m3_s = args.q_fixed

    # Normaliza algunos parámetros defensivamente
    if cfg.sim.dt_s <= 0:
        cfg.sim.dt_s = 1.0
    if cfg.sim.t_end_s < 0:
        cfg.sim.t_end_s = 0.0
    if args.print_every <= 0:
        args.print_every = 60

    sim = Simulator(cfg)

    sp_names = [sp.name for sp in cfg.mixture.species]
    eta_filter = max(0.0, min(1.0, cfg.filter.eta_filter))

    # Integrales y acumuladores
    t = 0.0
    dt = cfg.sim.dt_s
    t_end = cfg.sim.t_end_s

    m_exhaust_vap_kg = 0.0
    m_exhaust_aer_kg = 0.0
    m_captured_filter_kg = 0.0
    e_electrica_Wh = 0.0
    m_exhaust_species_kg = {nm: 0.0 for nm in sp_names}

    sum_Q = 0.0
    sum_v_face = 0.0
    sum_v_duct = 0.0
    n_steps = 0

    peak_C_tot = 0.0
    peak_ppm = 0.0
    peak_dict: Dict[str, float] = {}

    # Encabezado
    print("== Simulación iniciada ==")
    print(f"dt = {dt:.3f} s, t_end = {t_end:.1f} s, q_fixed = {cfg.sim.Q_fixed_m3_s}")
    print("-"*80)
    print("t[s]   Q[m3/s]  Vface[m/s]  Vduct[m/s]  C_v[mg/m3]  C_p[mg/m3]  C_tot[mg/m3]  ppm_VOC  P[W]")

    next_print = 0.0
    last_step = None  # ← clave: snapshot del último paso

    # Bucle de simulación
    while t < t_end - 1e-12:
        step = sim.step(dt)
        last_step = step  # ← siempre actualiza
        t = step["t_s"]
        Q = step["Q_m3_s"]
        Vface = step["V_face_m_s"]
        Vduct = step.get("v_duct_m_s", 0.0)
        C_v = step["C_v_mg_m3"]
        C_p = step["C_p_mg_m3"]
        C_tot = step["C_tot_mg_m3"]
        ppm = step["ppm_voc"]
        P = step["P_electrica_W"]

        if t >= next_print - 1e-9:
            print(f"{t:6.1f}  {Q:7.3f}    {Vface:7.3f}    {Vduct:7.3f}   {C_v:9.2f}   {C_p:9.2f}   {C_tot:11.2f}  {ppm:7.1f}  {P:7.0f}")
            next_print += args.print_every

        # Integrales
        m_exhaust_vap_kg       += Q * (C_v/1e6) * dt
        m_exhaust_aer_kg       += (1.0 - eta_filter) * Q * (C_p/1e6) * dt
        m_captured_filter_kg   += eta_filter * Q * (C_p/1e6) * dt

        # Por especie (vapor)
        for nm in sp_names:
            Ci = step.get(f"C_{nm}_mg_m3", 0.0)
            m_exhaust_species_kg[nm] += Q * (Ci/1e6) * dt

        # Energía eléctrica
        e_electrica_Wh += (P * dt) / 3600.0

        # Promedios
        sum_Q += Q
        sum_v_face += Vface
        sum_v_duct += Vduct
        n_steps += 1

        # Picos
        if C_tot > peak_C_tot:
            peak_C_tot = C_tot
            peak_dict["peak_C_tot_t"] = t
        if ppm > peak_ppm:
            peak_ppm = ppm
            peak_dict["peak_ppm_t"] = t

    # Si no hubo pasos (p. ej., t_end = 0), generamos un snapshot sin avanzar tiempo:
    if last_step is None:
        last_step = sim.step(0.0)  # dt=0 → no avanza el reloj interno
        # No acumulamos integrales ni promedios; sólo usamos para diagnóstico/imprimir

    # Resumen
    print("\n" + "="*80)
    print("RESUMEN")
    print("-"*80)
    avg_Q      = sdiv(sum_Q, n_steps, default=last_step.get("Q_m3_s", 0.0))
    avg_v_face = sdiv(sum_v_face, n_steps, default=last_step.get("V_face_m_s", 0.0))
    avg_v_duct = sdiv(sum_v_duct, n_steps, default=last_step.get("v_duct_m_s", 0.0))
    print(f"Promedios:   Q = {avg_Q:.3f} m3/s | V_face = {avg_v_face:.3f} m/s | V_duct = {avg_v_duct:.3f} m/s")

    if n_steps > 0:
        print(f"Concentración pico: C_tot = {peak_C_tot:.2f} mg/m3 @ t = {peak_dict.get('peak_C_tot_t', 0):.1f} s")
        print(f"VOC pico: {peak_ppm:.1f} ppm @ t = {peak_dict.get('peak_ppm_t', 0):.1f} s")
    else:
        # Con t_end=0 no hay evolución; reportamos el snapshot
        print("Concentración pico: (sin evolución temporal; usando snapshot t=0)")
        print(f"C_tot = {last_step.get('C_tot_mg_m3', 0.0):.2f} mg/m3, VOC = {last_step.get('ppm_voc', 0.0):.1f} ppm")

    print(f"Energía eléctrica ≈ {e_electrica_Wh:.1f} Wh")
    print("-"*80)
    print("Balances de emisión (integrados):")
    print(f"  Vapor a descarga (kg):   {m_exhaust_vap_kg:.6f}")
    print(f"  Aerosol a descarga (kg): {m_exhaust_aer_kg:.6f}")
    print(f"  Capturado en filtro (kg):{m_captured_filter_kg:.6f}   (sim state = {sim.state['M_captured_kg']:.6f})")
    print("-"*80)
    print("Especies (vapor) → masa descargada [kg]:")
    for nm in sp_names:
        print(f"  - {nm:18s}: {m_exhaust_species_kg[nm]:.6f}")

    # Criterios (del último snapshot disponible)
    crit_lines: List[str] = []
    if "ok_V_face" in last_step:
        crit_lines.append(f"V_face ≥ obj: {fmt_bool(last_step['ok_V_face'])}")
    if "ok_v_duct" in last_step:
        crit_lines.append(f"V_duct ≥ obj: {fmt_bool(last_step['ok_v_duct'])}")
    if "ok_C_limit" in last_step:
        crit_lines.append(f"C_v ≤ límite: {fmt_bool(last_step['ok_C_limit'])}")
    if "ok_LFL" in last_step:
        crit_lines.append(f"PPM ≤ LFL:    {fmt_bool(last_step['ok_LFL'])}")

    if crit_lines:
        print("-"*80)
        print("Criterios (snapshot final):")
        for line in crit_lines:
            print(" ", line)

    # Diagnóstico de transferencia y perfiles al final
    print("-"*80)
    print("Coeficientes h_m y perfiles a la salida (snapshot final):")
    for nm in sp_names:
        hm = last_step.get(f"hm_{nm}_m_s", 0.0)
        WL = last_step.get(f"W_L_{nm}_mg_m3", 0.0)
        Jw = last_step.get(f"Jwall_{nm}_mg_m2_s", 0.0)
        print(f"  {nm:18s}  h_m={hm:.5e} m/s | W(L)={WL:.2f} mg/m3 | J_pared≈{Jw:.3f} mg/m2/s")

    print("="*80)
    print("== Fin de la simulación ==")

if __name__ == "__main__":
    main()
