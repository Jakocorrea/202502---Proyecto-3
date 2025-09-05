# sim_logic.py
# =============================================================================
# 0-D mezcla con dos ENTRADAS (aire y pintura, densidades distintas) y un VENTILADOR
# - Mantiene balances de MASA total y de ESPECIE (pintura).
# - Ventilador con caudal Q_fan(t); velocidad de salida U_fan = Q_fan/Ly.
# - NUEVO: Velocidad "bulk" U_bulk(t) que arranca en U_air_in (=0.2) y
#          relaja de 1er orden hacia U_fan con constante mix_tau_s.
# - Reporta: c(t), U_bulk(t) (para tu gráfica), U_fan(t), rho(t), potencia de fan.
# =============================================================================

import json
import numpy as np

def load_params(path="params.json"):
    with open(path, "r") as f:
        return json.load(f)

def _to_Q_m2s_from_fan_cfg(fan_cfg, Ly):
    """
    Lee caudal del ventilador:
      - CONSTANTE:  'Q_fan' (m^2/s)  o 'Q_fan_m2_per_h' (m^2/h)  o 'U_fan' (m/s -> Q=U*Ly)
      - RAMPA 1er orden: 'Q0','Q_target' (m^2/s) o versiones *_m2_per_h, o 'U0','U_target' (m/s)
        dQ/dt = (Q_target - Q)/tau_s
    Devuelve: step(Q_prev, dt, t) y Q_init (para t=0).
    """
    mode = fan_cfg.get("type", "constant").lower()

    def _read_Q_key(key, default=None):
        if key in fan_cfg:
            return float(fan_cfg[key])
        h_key = f"{key}_m2_per_h"
        if h_key in fan_cfg:
            return float(fan_cfg[h_key]) / 3600.0
        return default

    if mode == "constant":
        Qc = _read_Q_key("Q_fan", default=None)
        if Qc is None:
            if "U_fan" in fan_cfg:
                Qc = float(fan_cfg["U_fan"]) * Ly
            else:
                raise ValueError("fan.type='constant' requiere Q_fan (m^2/s) o Q_fan_m2_per_h o U_fan (m/s).")
        def step(Q_prev, dt, t):
            return max(0.0, Qc)
        return step, Qc

    # first_order
    Q0 = _read_Q_key("Q0", default=None)
    Qt = _read_Q_key("Q_target", default=None)
    if Q0 is None and "U0" in fan_cfg:
        Q0 = float(fan_cfg["U0"]) * Ly
    if Qt is None and "U_target" in fan_cfg:
        Qt = float(fan_cfg["U_target"]) * Ly
    if Q0 is None: Q0 = 0.0
    if Qt is None: Qt = Q0
    tau = max(1e-9, float(fan_cfg.get("tau_s", 3.0)))

    def step(Q_prev, dt, t):
        Q = Q_prev if Q_prev is not None else Q0
        dQdt = (Qt - Q) / tau
        Q = Q + dt * dQdt
        return max(0.0, Q)
    return step, Q0

def simulate(params_path="params.json"):
    P = load_params(params_path)

    # Geometría y volumen
    Lx = float(P["Lx"]); Ly = float(P["Ly"])
    V  = Lx * Ly

    # Tiempo
    dt    = float(P["time"]["dt"])
    t_end = float(P["time"]["t_end"])
    nsteps = int(np.ceil(t_end / dt))

    # Entradas
    rho_air   = float(P["rho_air_in"])
    U_air     = float(P["U_air_in"])
    Q_air     = U_air * Ly

    rho_paint = float(P["rho_paint_in"])
    Q_paint   = float(P["Q_paint_in"])
    U_paint   = Q_paint / Ly

    # Ventilador
    fan_cfg = P.get("fan", {"type": "constant", "Q_fan": Q_air})
    fan_step, Q_fan_init = _to_Q_m2s_from_fan_cfg(fan_cfg, Ly)
    Q_fan = None  # se setea con fan_step al inicio

    # Dinámica de velocidad "bulk" (la que graficamos): U_bulk
    mix_tau_s = float(P.get("mix_tau_s", 3.0))
    U_bulk = U_air  # ARRANCA EXACTAMENTE en 0.2 m/s

    # Estado inicial de masas
    c0   = float(P.get("init", {}).get("c0", 0.0))
    rho0 = float(P.get("init", {}).get("rho0", rho_air))
    M  = rho0 * V
    Ms = rho0 * V * c0

    # Históricos
    t_hist    = np.zeros(nsteps + 1)
    c_hist    = np.zeros(nsteps + 1)
    rho_hist  = np.zeros(nsteps + 1)
    ubulk_hist= np.zeros(nsteps + 1)  # <<— lo que pediste graficar
    ufan_hist = np.zeros(nsteps + 1)
    Qfan_hist = np.zeros(nsteps + 1)
    Pfan_hist = np.zeros(nsteps + 1)
    dEk_hist  = np.zeros(nsteps + 1)

    # t=0
    t_hist[0]    = 0.0
    rho_hist[0]  = rho0
    c_hist[0]    = c0
    Q_fan        = fan_step(None, 0.0, 0.0)
    U_fan        = Q_fan / Ly
    ubulk_hist[0]= U_bulk          # = 0.2 m/s exacto
    ufan_hist[0] = U_fan
    Qfan_hist[0] = Q_fan

    if U_fan <= U_air:
        print("[AVISO] U_fan(0) <= U_air_in. La salida no es mayor que la entrada. "
              "Para ver un aumento claro, usa Ly más pequeño o Q_fan más grande.")

    # Bucle temporal
    for n in range(1, nsteps + 1):
        t = n * dt

        # 1) Ventilador (posible rampa)
        Q_fan = fan_step(Q_fan, dt, t)
        U_fan = Q_fan / Ly

        # 2) Velocidad 'bulk' que RELAJA desde U_air hacia U_fan
        dUbulk_dt = (U_fan - U_bulk) / max(1e-9, mix_tau_s)
        U_bulk = max(0.0, U_bulk + dt * dUbulk_dt)

        # 3) Variables actuales para balances
        rho = M / V
        c   = Ms / M if M > 1e-12 else 0.0

        # 4) Balances de masa total y especie
        m_in_air   = rho_air   * Q_air
        m_in_paint = rho_paint * Q_paint
        m_out      = rho       * Q_fan

        dM_dt  = m_in_air + m_in_paint - m_out
        dMs_dt = (rho_paint * Q_paint) - (rho * c * Q_fan)

        # 5) Avance explícito
        M  = max(1e-12, M  + dt * dM_dt)
        Ms = max(0.0,   Ms + dt * dMs_dt)

        # 6) Energía (análisis)
        m_in = m_in_air + m_in_paint
        e_k_in  = 0.0
        if m_in > 1e-12:
            e_k_in = (m_in_air * 0.5 * U_air**2 + m_in_paint * 0.5 * U_paint**2) / m_in
        e_k_out = 0.5 * U_fan**2
        dEk     = e_k_out - e_k_in
        P_fan   = m_out * dEk  # W por unidad de profundidad

        # 7) Guardados
        t_hist[n]     = t
        rho_hist[n]   = M / V
        c_hist[n]     = Ms / M if M > 1e-12 else 0.0
        ubulk_hist[n] = U_bulk
        ufan_hist[n]  = U_fan
        Qfan_hist[n]  = Q_fan
        Pfan_hist[n]  = P_fan
        dEk_hist[n]   = dEk

    return {
        "t": t_hist,
        "c_mean": c_hist,
        "u_bulk": ubulk_hist,   # <<— velocidad que arranca en 0.2 y sube
        "u_fan": ufan_hist,     # velocidad en el ventilador (referencia)
        "rho": rho_hist,
        "Q_fan": Qfan_hist,
        "dEk_spec": dEk_hist,
        "P_fan": Pfan_hist,
        "params": {
            "Lx": Lx, "Ly": Ly, "V": V,
            "rho_air_in": rho_air, "U_air_in": U_air, "Q_air": Q_air,
            "rho_paint_in": rho_paint, "Q_paint_in": Q_paint, "U_paint_in": U_paint,
            "fan": fan_cfg, "mix_tau_s": mix_tau_s, "dt": dt, "t_end": t_end
        }
    }