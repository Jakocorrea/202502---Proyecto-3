# sim_logic.py
# Modelo 0-D (mezcla bien agitada) con ventilador dinámico en velocidad:
#   V = Lx*Ly (por unidad de profundidad)
#   Q(t) = U(t) * Ly
#   dU/dt = (U_target - U)/tau_s     (si hay bloque "fan")
#   c'(t) = (Q(t)/V)*(c_in - c) + m_dot(t)/(rho*V)

import json
import numpy as np

def load_params(path="Entrega 3/params.json"):
    with open(path, "r") as f:
        return json.load(f)

def gate_spray(params, t):
    """1 si el spray está ON en t; 0 si OFF. Si no hay 'spray', está siempre ON."""
    sp = params.get("spray", None)
    if sp is None:
        return 1.0
    t_on = float(sp.get("t_on", 0.0))
    t_off = sp.get("t_off", None)
    if t < t_on:
        return 0.0
    if t_off is None:
        return 1.0
    return 1.0 if t < float(t_off) else 0.0

def simulate(params_path="params.json"):
    P = load_params(params_path)

    # Geometría y propiedades
    Lx = float(P["Lx"]); Ly = float(P["Ly"])
    V  = Lx * Ly                         # m^3 por unidad de profundidad (2D)
    rho = float(P.get("rho", 1.2))       # kg/m^3 (aire aprox.)

    # Entrada / fuente
    c_in  = float(P.get("c_in", 0.0))
    m_dot = float(P.get("m_dot", 0.0))   # kg/s (2D)

    # Tiempo
    dt    = float(P["time"]["dt"])
    t_end = float(P["time"]["t_end"])
    nsteps = int(np.ceil(t_end / dt))

    # Dinámica del ventilador: preferir U0/U_target (m/s); fallback a Q0/Q_target (m^2/s)
    if "fan" in P:
        fan = P["fan"]
        if "U0" in fan:       U = float(fan["U0"])
        elif "Q0" in fan:     U = float(fan["Q0"]) / Ly
        else:                 U = float(P.get("Q", 0.0)) / Ly  # si hubiera Q fijo

        if "U_target" in fan: U_target = float(fan["U_target"])
        elif "Q_target" in fan: U_target = float(fan["Q_target"]) / Ly
        else:                 U_target = U

        tau_s = max(1e-9, float(fan.get("tau_s", 3.0)))
        dynamic_fan = True
    else:
        # Sin bloque 'fan': velocidad fija a partir de Q
        U = float(P["Q"]) / Ly
        U_target = U
        tau_s = None
        dynamic_fan = False

    # Estado inicial
    c = 0.0

    # Históricos
    t_hist = np.zeros(nsteps + 1)
    c_hist = np.zeros(nsteps + 1)
    u_hist = np.zeros(nsteps + 1)

    # Guardar t=0
    t_hist[0] = 0.0
    c_hist[0] = c
    u_hist[0] = U

    for n in range(1, nsteps + 1):
        t = n * dt

        # 1) Ventilador: rampa 1er orden de U(t) hacia U_target
        if dynamic_fan:
            dUdt = (U_target - U) / tau_s
            U = U + dt * dUdt
            U = max(0.0, U)  # sin velocidades negativas

        Q = U * Ly  # m^2/s

        # 2) Spray ON/OFF (evaluación centrada)
        gate = gate_spray(P, t - 0.5*dt)
        m_src = m_dot * gate

        # 3) ODE de mezcla bien agitada para c(t)
        dc_dt = (Q / V) * (c_in - c) + m_src / (rho * V)
        c = c + dt * dc_dt
        c = max(0.0, min(1.0, c))

        # 4) Guardar
        t_hist[n] = t
        c_hist[n] = c
        u_hist[n] = U

    return {
        "t": t_hist,         # s
        "c_mean": c_hist,    # fracción másica promedio
        "u_mean": u_hist,    # velocidad media [m/s]
        "params": P
    }
