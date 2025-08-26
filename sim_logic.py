from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import json

# -------------------------
# Utilidades numéricas
# -------------------------
def lin_interp(x: float, xs: List[float], ys: List[float]) -> float:
    """Interpolación lineal con manejo robusto de bordes."""
    if len(xs) != len(ys):
        raise ValueError("Xs y Ys deben tener la misma longitud.")
    if len(xs) < 2:
        raise ValueError("Se requieren al menos dos puntos para interpolar.")

    # Asumimos xs ordenado ascendente; si no lo está, el llamador debe ordenar.
    if x <= xs[0]:
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    if x >= xs[-1]:
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
        return y1 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


# -------------------------
# Propiedades del aire y fricción
# -------------------------
def air_props_from_altitude(altitude_m: float, T_C: float) -> Tuple[float, float]:
    """Densidad (kg/m3) y viscosidad (Pa·s) a altura ~ISA y T con Sutherland."""
    g = 9.80665
    R = 287.05287
    T0 = 288.15
    p0 = 101325.0
    L = 0.0065  # gradiente térmico
    T = T_C + 273.15
    if altitude_m < 0:
        altitude_m = 0.0
    h = altitude_m
    factor = (1 - L * h / T0)
    if factor <= 0:
        factor = 1e-6
    exponent = g / (R * L)
    p = p0 * (factor ** exponent)
    rho = p / (R * T)

    # Sutherland
    T0_suth = 273.15
    mu0 = 1.716e-5
    S = 110.4
    mu = mu0 * ((T0_suth + S) / (T + S)) * (T / T0_suth) ** 1.5
    return rho, mu


def haaland_friction_factor(Re: float, eps_over_D: float) -> float:
    """Factor de fricción de Haaland (turbulento) + laminar simple."""
    if Re <= 0:
        return 0.0
    if Re < 2100:
        return 64.0 / Re
    if eps_over_D < 1e-12:
        eps_over_D = 1e-12
    inv_sqrt_f = -1.8 * math.log10((eps_over_D / 3.7) ** 1.11 + 6.9 / Re)
    f = 1.0 / (inv_sqrt_f * inv_sqrt_f)
    return f


# -------------------------
# Elementos del sistema
# -------------------------
@dataclass
class DuctSegment:
    L: float
    D: float
    eps: float
    K_minor: float

    def dp(self, Q: float, rho: float, mu: float) -> float:
        if self.D <= 0:
            return 0.0
        area = math.pi * (self.D ** 2) / 4.0
        if area <= 0:
            return 0.0
        v = Q / area
        Re = rho * v * self.D / (mu if mu > 0 else 1e-9)
        f = haaland_friction_factor(Re, self.eps / self.D if self.D > 0 else 0.0)
        dp_major = f * (self.L / self.D) * (rho * v * v / 2.0)
        dp_minor = self.K_minor * (rho * v * v / 2.0)
        return dp_major + dp_minor


@dataclass
class FanCurve:
    Q_points: List[float]
    DP_points: List[float]
    def dp(self, Q: float) -> float:
        return lin_interp(Q, self.Q_points, self.DP_points)


@dataclass
class FilterCurve:
    Q_points: List[float]
    DP0_points: List[float]
    def dp0(self, Q: float) -> float:
        return lin_interp(Q, self.Q_points, self.DP0_points)


@dataclass
class FilterModel:
    curve: FilterCurve
    alpha_Pa_per_kg: float
    eta_filter: float
    dp_max_Pa: float
    def dp(self, Q: float, M_captured_kg: float) -> float:
        # Aumento de ΔP con carga capturada; límite superior opcional
        dp_val = self.curve.dp0(Q) + self.alpha_Pa_per_kg * max(0.0, M_captured_kg)
        return min(dp_val, self.dp_max_Pa) if self.dp_max_Pa > 0 else dp_val


@dataclass
class Geometry:
    L: float
    A: float
    H: float
    area_face: float
    volume: Optional[float] = None
    def get_volume(self) -> float:
        return self.volume if self.volume and self.volume > 0 else self.L * self.A * self.H


@dataclass
class Air:
    altitude_m: float
    T_C: float
    RH: float
    rho: Optional[float] = None
    mu: Optional[float] = None
    def props(self) -> Tuple[float, float]:
        if (self.rho is not None) and (self.mu is not None):
            return self.rho, self.mu
        rho, mu = air_props_from_altitude(self.altitude_m, self.T_C)
        rho = self.rho if self.rho is not None else rho
        mu = self.mu if self.mu is not None else mu
        return rho, mu


@dataclass
class Pistol:
    q_l_mL_min: float
    rho_l_kg_L: float
    w_s: float
    TE: float
    f_evap_fast: float
    f_aer: float
    k_ev_s_inv: float = 0.0
    name: str = ""
    M_liq_reservoir_kg: float = 0.0

    def mass_flows(self, is_on: bool) -> Tuple[float, float, float]:
        """
        Retorna (m_vapor_kg_s, m_aer_kg_s, m_reservorio_add_kg_s)
        Considera evaporación rápida (on) + lenta (reservorio) y overspray -> aerosol.
        """
        m_vapor = 0.0
        m_aer = 0.0
        m_res_add = 0.0

        # Evaporación lenta desde reservorio
        m_vapor_slow = self.k_ev_s_inv * self.M_liq_reservoir_kg

        if is_on:
            # Flujo líquido alimentado
            m_liq = (self.q_l_mL_min * 1e-3 / 60.0) * self.rho_l_kg_L  # kg/s
            m_solvent = m_liq * max(0.0, min(1.0, self.w_s))
            m_vapor_fast = m_solvent * max(0.0, min(1.0, self.f_evap_fast))

            # Overspray (no depositado en sustrato)
            m_overspray = m_liq * (1.0 - max(0.0, min(1.0, self.TE)))
            m_nonvapor_overspray = m_overspray * (1.0 - max(0.0, min(1.0, self.f_evap_fast)))
            m_aer = m_nonvapor_overspray * max(0.0, min(1.0, self.f_aer))

            m_vapor = m_vapor_fast + m_vapor_slow
            # Lo que no se evapora ni pasa a aerosol, se acumula en reservorio
            m_res_add = max(0.0, m_liq - (m_vapor_fast + m_aer))
            self.M_liq_reservoir_kg += m_res_add - m_vapor_slow
        else:
            # Solo hay evaporación lenta del reservorio
            m_vapor = m_vapor_slow
            self.M_liq_reservoir_kg -= m_vapor_slow

        if self.M_liq_reservoir_kg < 0:
            self.M_liq_reservoir_kg = 0.0

        return m_vapor, m_aer, m_res_add


@dataclass
class Task:
    t_on_s: float
    t_off_s: float
    repetitions: int
    active_pistols: List[int]


@dataclass
class Schedule:
    tasks: List[Task]
    n_tasks_day: int = 1
    purge_s: float = 0.0

    def total_duration_s(self) -> float:
        base = 0.0
        for t in self.tasks:
            base += (t.t_on_s + t.t_off_s) * t.repetitions
        base *= max(1, self.n_tasks_day)
        base += max(0.0, self.purge_s)
        return base

    def is_on(self, t: float) -> Tuple[bool, List[int]]:
        day_cycle = 0.0
        expanded: List[Tuple[str, float, List[int]]] = []
        for _ in range(max(1, self.n_tasks_day)):
            for task in self.tasks:
                expanded.append(("on", task.t_on_s, task.active_pistols))
                expanded.append(("off", task.t_off_s, []))
                day_cycle += task.t_on_s + task.t_off_s
        purge = self.purge_s
        total = day_cycle + purge
        if total <= 0:
            return False, []

        t_mod = t % total
        elapsed = 0.0
        for kind, dur, pist in expanded:
            if t_mod < elapsed + dur:
                return (True, pist) if kind == "on" else (False, [])
            elapsed += dur
        return False, []


@dataclass
class Criteria:
    V_face_obj: Optional[float] = None
    ACH_obj: Optional[float] = None
    C_limit_mg_m3: Optional[float] = None
    LFL_ppm: Optional[float] = None
    v_duct_obj: Optional[float] = None


@dataclass
class CaptureFiltering:
    eta_capt_source: float = 0.0
    k_dep_s_inv: float = 0.0


@dataclass
class FilterState:
    M_captured_kg: float = 0.0


@dataclass
class SystemConfig:
    geometry: Geometry
    air: Air
    ducts: List[DuctSegment]
    fan: FanCurve
    filter_model: FilterModel
    schedule: Schedule
    pistols: List[Pistol]
    criteria: Criteria
    capture_filtering: CaptureFiltering
    K_extra: float = 0.0
    Q_fixed_m3_s: Optional[float] = None
    # Compat: eficiencias del conjunto ventilador+motor
    eta_fan: Optional[float] = None
    eta_motor: Optional[float] = None
    # Para soportar JSONs antiguos que tenían bloque separado
    fan_eff: Optional[Dict[str, float]] = None


# -------------------------
# Núcleo de simulación
# -------------------------
class Simulator:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.filter_state = FilterState()
        self.rho, self.mu = self.cfg.air.props()
        self.V = self.cfg.geometry.get_volume()

    # --- hidráulica ---
    def dp_network(self, Q: float) -> float:
        rho, mu = self.rho, self.mu
        dp = 0.0
        for seg in self.cfg.ducts:
            dp += seg.dp(Q, rho, mu)
        if self.cfg.ducts:
            D = self.cfg.ducts[0].D
            area = math.pi * (D ** 2) / 4.0
            v = Q / area if area > 0 else 0.0
            dp += self.cfg.K_extra * (rho * v * v / 2.0)
        return dp

    def dp_filter(self, Q: float) -> float:
        return self.cfg.filter_model.dp(Q, self.filter_state.M_captured_kg)

    def dp_system(self, Q: float) -> float:
        return self.dp_network(Q) + self.dp_filter(Q)

    def fan_dp(self, Q: float) -> float:
        return self.cfg.fan.dp(Q)

    def find_operating_Q(self) -> float:
        """Resuelve la intersección curva ventilador vs. sistema (bisección con fallback)."""
        Qmin = 0.0
        Qmax = max(self.cfg.fan.Q_points) * 1.2

        def f(Q):
            return self.fan_dp(Q) - self.dp_system(Q)

        fmin = f(Qmin)
        fmax = f(Qmax)
        # Si no hay cambio de signo, probamos muestreo y tomamos el mínimo residuo.
        if fmin * fmax > 0:
            samples = 50
            best_Q, best_val = Qmin, abs(fmin)
            for i in range(1, samples + 1):
                q = Qmin + (Qmax - Qmin) * i / samples
                val = abs(f(q))
                if val < best_val:
                    best_Q, best_val = q, val
            return best_Q

        # Bisección
        a, b = Qmin, Qmax
        for _ in range(60):
            c = 0.5 * (a + b)
            fc = f(c)
            if abs(fc) < 1e-3:
                return c
            if f(a) * fc <= 0:
                b = c
            else:
                a = c
        return 0.5 * (a + b)

    # --- emisiones y dinámica de concentraciones ---
    def emissions_step(self, t: float) -> Tuple[float, float]:
        """
        Devuelve (S_vapor_mg_s, S_aerosol_mg_s).
        Aplica captura en la fuente (eta_capt_source).
        """
        on, active = self.cfg.schedule.is_on(t)
        S_vapor_kg_s = 0.0
        S_aer_kg_s = 0.0

        if on:
            for idx in active:
                if 0 <= idx < len(self.cfg.pistols):
                    m_vap, m_aer, _ = self.cfg.pistols[idx].mass_flows(True)
                    S_vapor_kg_s += m_vap
                    S_aer_kg_s += m_aer
        else:
            # off: solo evaporación lenta desde los reservorios
            for p in self.cfg.pistols:
                m_vap, m_aer, _ = p.mass_flows(False)
                S_vapor_kg_s += m_vap

        # Captura en la fuente reduce el aerosol que entra al recinto
        eta_capt = max(0.0, min(1.0, self.cfg.capture_filtering.eta_capt_source))
        S_aer_kg_s *= (1.0 - eta_capt)

        # Convertir a mg/s
        return 1e6 * S_vapor_kg_s, 1e6 * S_aer_kg_s

    def step_dynamics(self, state: Dict[str, float], dt: float, t: float) -> Dict[str, float]:
        V = self.V
        # Caudal
        if self.cfg.Q_fixed_m3_s is not None and self.cfg.Q_fixed_m3_s > 0:
            Q = self.cfg.Q_fixed_m3_s
        else:
            Q = self.find_operating_Q()

        # Fuentes (mg/s)
        S_v_mg_s, S_p_mg_s = self.emissions_step(t)

        # Balance (C_v: vapor, C_p: partículas/aerosol)
        lambda_v = Q / V
        lambda_p = Q / V + max(0.0, self.cfg.capture_filtering.k_dep_s_inv)
        C_v = state["C_v_mg_m3"] + dt * ((S_v_mg_s / V) - lambda_v * state["C_v_mg_m3"])
        C_p = state["C_p_mg_m3"] + dt * ((S_p_mg_s / V) - lambda_p * state["C_p_mg_m3"])
        if C_v < 0:
            C_v = 0.0
        if C_p < 0:
            C_p = 0.0

        # Captura en filtro (solo de la fracción particulada C_p)
        eta_filter = max(0.0, min(1.0, self.cfg.filter_model.eta_filter))
        m_to_filter_mg_s = Q * state["C_p_mg_m3"]  # mg/s que pasan por el filtro
        m_captured_kg = state["M_captured_kg"] + dt * (eta_filter * m_to_filter_mg_s) * 1e-6

        # Hidráulica y potencias
        dp_filt = self.dp_filter(Q)
        dp_net = self.dp_network(Q)
        dp_sys = dp_net + dp_filt

        eta_fan = self.cfg.eta_fan if (self.cfg.eta_fan is not None) else 0.6
        eta_motor = self.cfg.eta_motor if (self.cfg.eta_motor is not None) else 0.9
        denom = max(1e-6, (eta_fan * eta_motor))
        P_e_W = (Q * dp_sys) / denom

        # Velocidades de interés
        V_face = Q / max(1e-9, self.cfg.geometry.area_face)
        if self.cfg.ducts:
            D0 = self.cfg.ducts[0].D
            A0 = math.pi * D0 * D0 / 4.0
            v_duct = Q / max(1e-9, A0)
        else:
            v_duct = 0.0

        return {
            "Q_m3_s": Q,
            "V_face_m_s": V_face,
            "v_duct_m_s": v_duct,
            "C_v_mg_m3": C_v,
            "C_p_mg_m3": C_p,
            "C_tot_mg_m3": C_v + C_p,
            "DP_filter_Pa": dp_filt,
            "DP_network_Pa": dp_net,
            "DP_system_Pa": dp_sys,
            "P_e_W": P_e_W,
            "M_captured_kg": m_captured_kg,
        }

    def run(self, t_end_s: float, dt_s: float) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        # Compat con JSONs que traen bloque 'fan_eff'
        if self.cfg.fan_eff:
            self.cfg.eta_fan = self.cfg.fan_eff.get("eta_fan", 0.6)
            self.cfg.eta_motor = self.cfg.fan_eff.get("eta_motor", 0.9)

        state = {"C_v_mg_m3": 0.0, "C_p_mg_m3": 0.0, "M_captured_kg": 0.0}
        ts: List[Dict[str, float]] = []
        t = 0.0
        energy_Wh = 0.0

        while t <= t_end_s + 1e-9:
            out = self.step_dynamics(state, dt_s, t)
            energy_Wh += out["P_e_W"] * dt_s / 3600.0
            ts.append(dict(time_s=t, **out))
            # Avanzar estado
            state["C_v_mg_m3"] = out["C_v_mg_m3"]
            state["C_p_mg_m3"] = out["C_p_mg_m3"]
            state["M_captured_kg"] = out["M_captured_kg"]
            t += dt_s

        # Resumen
        max_C = max((r["C_tot_mg_m3"] for r in ts), default=0.0)
        avg_C = sum((r["C_tot_mg_m3"] for r in ts), 0.0) / (len(ts) if ts else 1.0)
        max_DP = max((r["DP_system_Pa"] for r in ts), default=0.0)
        final_DP_filter = ts[-1]["DP_filter_Pa"] if ts else 0.0

        summary = {
            "t_end_s": t_end_s,
            "dt_s": dt_s,
            "energy_kWh": energy_Wh / 1000.0,
            "C_total_mg_m3_max": max_C,
            "C_total_mg_m3_avg": avg_C,
            "DP_system_Pa_max": max_DP,
            "DP_filter_Pa_final": final_DP_filter,
            "M_captured_kg_final": state["M_captured_kg"],
        }
        return ts, summary


# -------------------------
# Carga de configuración
# -------------------------
def _read_curve(curve: Dict) -> Tuple[List[float], List[float]]:
    Q = list(curve.get("Q", []))
    DP = list(curve.get("DP", []))
    if not Q or not DP or len(Q) != len(DP):
        raise ValueError("Curve requiere arreglos 'Q' y 'DP' de igual longitud.")
    pairs = sorted(zip(Q, DP), key=lambda x: x[0])
    Qs, DPs = [p[0] for p in pairs], [p[1] for p in pairs]
    return Qs, DPs


def config_from_dict(d: Dict) -> SystemConfig:
    # Geometría / aire
    geomd = d["geometry"]
    geometry = Geometry(L=geomd["L"], A=geomd["A"], H=geomd["H"],
                        area_face=geomd["area_face"],
                        volume=geomd.get("volume"))
    aird = d["air"]
    air = Air(altitude_m=aird["altitude_m"], T_C=aird["T_C"],
              RH=aird.get("RH", 50.0),
              rho=aird.get("rho"), mu=aird.get("mu"))

    # Ductos
    ducts: List[DuctSegment] = []
    for seg in d.get("ducts", []):
        ducts.append(DuctSegment(L=seg["L"], D=seg["D"], eps=seg["eps"], K_minor=seg.get("K", 0.0)))

    # Curvas
    Qf, DPf = _read_curve(d["fan"]["curve"])
    fan = FanCurve(Q_points=Qf, DP_points=DPf)
    Qc, DP0c = _read_curve(d["filter"]["dp0_curve"])
    flt_curve = FilterCurve(Q_points=Qc, DP0_points=DP0c)
    filter_model = FilterModel(
        curve=flt_curve,
        alpha_Pa_per_kg=d["filter"].get("alpha_Pa_per_kg", 0.0),
        eta_filter=d["filter"].get("eta_filter", 0.9),
        dp_max_Pa=d["filter"].get("dp_max_Pa", 1200.0),
    )

    # Programa (schedule)
    tasks: List[Task] = []
    for t in d["schedule"]["tasks"]:
        tasks.append(Task(
            t_on_s=t["t_on_s"],
            t_off_s=t["t_off_s"],
            repetitions=t["repetitions"],
            active_pistols=list(t.get("active_pistols", []))
        ))
    schedule = Schedule(
        tasks=tasks,
        n_tasks_day=d["schedule"].get("n_tasks_day", 1),
        purge_s=d["schedule"].get("purge_s", 0.0),
    )

    # Pistolas
    pistols: List[Pistol] = []
    for i, p in enumerate(d["source"]["pistols"]):
        pistols.append(Pistol(
            q_l_mL_min=p["q_l_mL_min"],
            rho_l_kg_L=p["rho_l_kg_L"],
            w_s=p["w_s"],
            TE=p["TE"],
            f_evap_fast=p["f_evap_fast"],
            f_aer=p["f_aer"],
            k_ev_s_inv=p.get("k_ev_s_inv", 0.0),
            name=p.get("name", f"pistol_{i+1}"),
        ))

    # Criterios y captura
    crd = d.get("criteria", {})
    criteria = Criteria(
        V_face_obj=crd.get("V_face_obj"),
        ACH_obj=crd.get("ACH_obj"),
        C_limit_mg_m3=crd.get("C_limit_mg_m3"),
        LFL_ppm=crd.get("LFL_ppm"),
        v_duct_obj=crd.get("v_duct_obj"),
    )
    capd = d.get("capture_filtering", {})
    capture_filtering = CaptureFiltering(
        eta_capt_source=capd.get("eta_capt_source", 0.0),
        k_dep_s_inv=capd.get("k_dep_s_inv", 0.0),
    )

    # Varios
    K_extra = d.get("inlets_outlets", {}).get("K_extra", 0.0)
    Q_fixed = d.get("sim", {}).get("Q_fixed_m3_s", None)

    # Eficiencias ventilador/motor (soporta bloque legado fan_eff)
    fan_eff = d.get("fan_eff", {"eta_fan": d["fan"].get("eta_fan", 0.6),
                                "eta_motor": d["fan"].get("eta_motor", 0.9)})

    cfg = SystemConfig(
        geometry=geometry,
        air=air,
        ducts=ducts,
        fan=fan,
        filter_model=filter_model,
        schedule=schedule,
        pistols=pistols,
        criteria=criteria,
        capture_filtering=capture_filtering,
        K_extra=K_extra,
        Q_fixed_m3_s=Q_fixed,
        eta_fan=fan_eff.get("eta_fan", 0.6),
        eta_motor=fan_eff.get("eta_motor", 0.9),
        fan_eff=fan_eff,
    )
    return cfg


def config_from_json(path: str) -> SystemConfig:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return config_from_dict(d)
