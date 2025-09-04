# sim_logic.py
# Modelo 0-D (mezcla bien agitada) con ventilador dinámico:
#   V = Lx*Ly (por unidad de profundidad)
#   Q(t): caudal 2D del ventilador con rampa de 1er orden
#     dQ/dt = (Q_target - Q)/tau_s
#   U(t) = Q(t) / Ly
#   Ecuación para fracción másica promedio c(t):
#     d/dt (rho*V*c) = rho*Q(t)*(c_in - c) + m_dot(t)
#   => dc/dt = (Q(t)/V)*(c_in - c) + m_dot(t)/(rho*V)

import json
import numpy as np

def load_params(path="params.json"):
    with open(path, "r") as f:
        return json.load(f)

def gate_spray(params, t):
    """1 si el spray está ON en el tiempo t; 0 si está OFF."""
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
    rho = float(P.get("rho", 1.2))       # kg/m^3 (aire aprox)

    # Ingreso / fuente
    c_in  = float(P.get("c_in", 0.0))
    m_dot = float(P.get("m_dot", 0.0))   # kg/s (2D)

    # Tiempo
    dt    = float(P["time"]["dt"])
    t_end = float(P["time"]["t_end"])
    nsteps = int(np.ceil(t_end / dt))

    # Dinámica del ventilador (si no hay bloque "fan", Q es constante)
    if "fan" in P:
        Q  = float(P["fan"].get("Q0", 0.0))         # m^2/s (estado inicial)
        Q_target = float(P["fan"].get("Q_target", Q))
        tau_s = max(1e-9, float(P["fan"].get("tau_s", 3.0)))  # s
    else:
        Q = float(P["Q"])
        Q_target = Q
        tau_s = None  # sin dinámica: Q(t) = Q

    # Estado inicial
    c = 0.0

    # Históricos
    t_hist = np.zeros(nsteps + 1)
    c_hist = np.zeros(nsteps + 1)
    u_hist = np.zeros(nsteps + 1)

    # Guardar t=0
    t_hist[0] = 0.0
    c_hist[0] = c
    u_hist[0] = Q / Ly

    for n in range(1, nsteps + 1):
        t = n * dt

        # 1) Actualizar caudal por dinámica del ventilador
        if tau_s is not None:
            # 1er orden hacia Q_target
            dQdt = (Q_target - Q) / tau_s
            Q = Q + dt * dQdt
            Q = max(0.0, Q)  # sin flujo negativo

        U = Q / Ly  # velocidad media

        # 2) Fuente (spray) ON/OFF (evaluación centrada)
        gate = gate_spray(P, t - 0.5*dt)
        m_src = m_dot * gate

        # 3) ODE para c(t)
        dc_dt = (Q / V) * (c_in - c) + m_src / (rho * V)
        c = c + dt * dc_dt
        c = max(0.0, min(1.0, c))  # límites físicos

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
