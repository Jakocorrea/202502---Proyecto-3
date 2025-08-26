from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import json

# -----------------------------
# Helpers
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def lin_interp(x: float, xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("lin_interp: listas mal formadas")
    if x <= xs[0]:
        # extrapola linealmente con el primer segmento
        x0,x1 = xs[0], xs[1]
        y0,y1 = ys[0], ys[1]
        return y0 + (y1-y0)*(x - x0)/(x1 - x0)
    if x >= xs[-1]:
        x0,x1 = xs[-2], xs[-1]
        y0,y1 = ys[-2], ys[-1]
        return y0 + (y1-y0)*(x - x0)/(x1 - x0)
    # búsqueda binaria simple
    lo, hi = 0, len(xs)-1
    while hi-lo > 1:
        mid = (lo+hi)//2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0,x1 = xs[lo], xs[hi]
    y0,y1 = ys[lo], ys[hi]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0)*(x - x0)/(x1 - x0)

# -----------------------------
# Aire y propiedades
# -----------------------------

R_UNIVERSAL = 8.314462618  # J/mol/K
R_DRY = 287.058            # J/kg/K
R_WV = 461.495             # J/kg/K
P0_PA = 101325.0

def pressure_at_altitude(alt_m: float, T_K: float) -> float:
    """Modelo ISA simplificado (capa troposférica)"""
    # si T_K no se conoce suficientemente, use 288.15 K nominal
    T0 = 288.15
    L = 0.0065  # K/m
    g0 = 9.80665
    M = 0.0289644
    R = R_UNIVERSAL
    if alt_m < 11000:
        return P0_PA * (1.0 - L*alt_m/T0)**(g0*M/(R*L))
    else:
        # aproximación grosera para >11 km
        return P0_PA * math.exp(-g0*M*(alt_m-11000)/(R*T0)) * (1.0 - L*11000/T0)**(g0*M/(R*L))

def sat_vapor_pressure_water(T_C: float) -> float:
    """Tetens (Pa)"""
    return 610.78 * math.exp((17.27*T_C)/(T_C + 237.3))

def air_density_moist(P_Pa: float, T_K: float, RH: float) -> float:
    """Densidad de aire húmedo (kg/m3) separando componentes seco/vapor."""
    e = clamp(RH, 0.0, 1.0) * sat_vapor_pressure_water(T_K - 273.15)
    Pd = P_Pa - e
    rho = Pd/(R_DRY*T_K) + e/(R_WV*T_K)
    return rho

def mu_air_sutherland(T_K: float) -> float:
    """Viscosidad dinámica del aire (Pa·s) usando Sutherland."""
    T0 = 273.15
    mu0 = 1.716e-5
    C = 111.0
    return mu0 * ((T_K/T0)**1.5) * (T0 + C)/(T_K + C)

# -----------------------------
# Difusión y Antoine
# -----------------------------

def D_Fuller_cm2_s(T_K: float, P_atm: float, Mi_g_mol: float, Mj_g_mol: float,
                   vi_cm3_mol: float, vj_cm3_mol: float) -> float:
    """
    Fuller (gas-gas) coeficiente binario (cm2/s).
    T en K, P en atm, M en g/mol, v (volúmenes de difusión) en cm3/mol.
    """
    return 0.001 * (T_K**1.75) * math.sqrt(1.0/Mi_g_mol + 1.0/Mj_g_mol) / (
        P_atm * ((vi_cm3_mol**(1.0/3.0) + vj_cm3_mol**(1.0/3.0))**2)
    )

def antoine_P_bar(T_K: float, A: float, B: float, C: float) -> float:
    """Forma: log10(P_bar) = A - B/(T_K + C)."""
    return 10.0**(A - B/(T_K + C))

# -----------------------------
# Estructuras de configuración
# -----------------------------

@dataclass
class Geometry:
    L: float
    A: float
    H: float
    area_face: float
    volume: Optional[float] = None
    type: str = "cross-draft"
    def get_V(self) -> float:
        return self.volume if (self.volume and self.volume > 0) else (self.A * self.H)

@dataclass
class AirConfig:
    altitude_m: float
    T_C: float
    RH: float
    rho: Optional[float] = None
    mu: Optional[float] = None

@dataclass
class Duct:
    L: float
    D: float
    eps: float
    K: float

@dataclass
class InOutExtras:
    K_extra: float = 0.0

@dataclass
class FilterConfig:
    dp0_curve: Dict[str, List[float]]  # keys: Q, DP (Pa)
    alpha_Pa_per_kg: float = 0.0       # aumento de ΔP por masa capturada
    eta_filter: float = 0.9            # eficiencia para aerosol
    dp_max_Pa: float = 0.0

    def dp0(self, Q_m3_s: float) -> float:
        return lin_interp(Q_m3_s, self.dp0_curve["Q"], self.dp0_curve["DP"])

@dataclass
class FanConfig:
    curve: Dict[str, List[float]]      # keys: Q, DP (Pa)
    eta_fan: float = 0.6
    eta_motor: float = 0.9
    def dp(self, Q_m3_s: float) -> float:
        return lin_interp(Q_m3_s, self.curve["Q"], self.curve["DP"])

@dataclass
class Pistol:
    name: str
    q_l_mL_min: float       # flujo líquido
    rho_l_kg_L: float       # densidad líquido
    w_s: float              # fracción másica solventes del líquido
    TE: float               # transfer efficiency (a la pieza)
    f_evap_fast: float      # fracción de solvente que volatiliza inmediato
    f_aer: float            # fracción que sale como aerosol (del total emitido al aire)
    k_ev_s_inv: float       # cinética 1er orden de evaporación remanente

@dataclass
class SourceConfig:
    pistols: List[Pistol]

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

@dataclass
class Criteria:
    V_face_obj: Optional[float] = None
    ACH_obj: Optional[float] = None
    C_limit_mg_m3: Optional[float] = None
    LFL_ppm: Optional[float] = None
    v_duct_obj: Optional[float] = None

@dataclass
class CaptureFiltering:
    eta_capt_source: float = 0.0  # captura local de aerosol
    k_dep_s_inv: float = 0.0      # depósito (paredes/suelo), 1/s

@dataclass
class SimSetup:
    dt_s: float = 1.0
    t_end_s: float = 3600.0
    Q_fixed_m3_s: Optional[float] = None

# ---------- Mezcla / Especies (nuevo) -------------

@dataclass
class Species:
    name: str
    M_g_mol: float
    v_Fuller_cm3_mol: float
    Antoine: Tuple[float, float, float]  # (A,B,C) bar-form
    x_liq: float                         # fracción molar en líquido
    LFL_vol_frac: Optional[float] = None # como fracción (e.g., 0.03 para 3% v/v)

@dataclass
class MixtureConfig:
    species: List[Species] = field(default_factory=list)

# Paquete de config

@dataclass
class SystemConfig:
    geometry: Geometry
    air: AirConfig
    ducts: List[Duct]
    inout: InOutExtras
    filter: FilterConfig
    fan: FanConfig
    source: SourceConfig
    schedule: Schedule
    criteria: Criteria
    capture: CaptureFiltering
    sim: SimSetup
    mixture: MixtureConfig

# -----------------------------
# Utilidades hidráulicas
# -----------------------------

def reynolds(rho: float, v: float, D: float, mu: float) -> float:
    if mu <= 0 or D <= 0:
        return 0.0
    return rho * v * D / mu

def friction_factor(Re: float, eps: float, D: float) -> float:
    if Re <= 0:
        return 0.0
    if Re < 2300:
        return 64.0/Re
    # Swamee-Jain
    return 0.25 / (math.log10(eps/(3.7*D) + 5.74/(Re**0.9))**2)

def duct_dp(Q: float, rho: float, mu: float, duct: Duct) -> Tuple[float, float, float]:
    A = math.pi * (duct.D**2) / 4.0
    v = Q / A if A > 0 else 0.0
    Re = reynolds(rho, v, duct.D, mu)
    f = friction_factor(Re, duct.eps, duct.D)
    dp = (f * duct.L / duct.D + duct.K) * 0.5 * rho * v * v
    return dp, v, Re

# -----------------------------
# Simulador
# -----------------------------

class Simulator:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        # estado dinámico
        self.state = {
            "t": 0.0,
            "C_v_mg_m3": 0.0,
            "C_p_mg_m3": 0.0,
            "M_captured_kg": 0.0,
        }
        # masa remanente líquida por pistola (que evapora lentamente)
        self.residual_liq_kg = [0.0 for _ in cfg.source.pistols]

        # propiedades de aire (si no fueron dadas)
        self.P_Pa, self.T_K, self.rho_air, self.mu_air = self._init_air()

        # preparar constantes especie/mezcla
        self._prep_mixture()

    # -------- aire --------
    def _init_air(self) -> Tuple[float, float, float, float]:
        ac = self.cfg.air
        T_K = ac.T_C + 273.15
        P = pressure_at_altitude(ac.altitude_m, T_K)
        rho = ac.rho if (ac.rho and ac.rho>0) else air_density_moist(P, T_K, ac.RH/100.0)
        mu = ac.mu if (ac.mu and ac.mu>0) else mu_air_sutherland(T_K)
        return P, T_K, rho, mu

    # -------- mezcla --------
    def _default_species(self) -> List[Species]:
        # Defaults razonables (A,B,C en bar) -> estos deben revisarse/ajustarse si se dispone de datos exactos
        # Nombres: Etanol, Acetato de propilo, 1-propanol, Isopropanol
        return [
            Species("Ethanol", 46.07, 41.0, (5.24677, 1598.673, -46.424), 0.40, 0.033),
            Species("Propyl acetate", 102.13, 92.0, (5.02029, 1474.76, -39.27), 0.25, 0.018),
            Species("1-Propanol", 60.10, 54.0, (5.20184, 1731.53, -46.29), 0.25, 0.022),
            Species("Isopropanol", 60.10, 56.0, (5.12705, 1683.125, -46.86), 0.10, 0.020),
        ]

    def _prep_mixture(self):
        if not self.cfg.mixture.species:
            self.cfg.mixture.species = self._default_species()
        # normaliza x_liq
        tot = sum(sp.x_liq for sp in self.cfg.mixture.species)
        if tot <= 0:
            n = len(self.cfg.mixture.species)
            for sp in self.cfg.mixture.species:
                sp.x_liq = 1.0/n
        else:
            for sp in self.cfg.mixture.species:
                sp.x_liq /= tot

    # -------- filtros/ventilador --------
    def filter_dp(self, Q: float) -> float:
        fc = self.cfg.filter
        base = fc.dp0(Q)
        dp = base + fc.alpha_Pa_per_kg * max(0.0, self.state["M_captured_kg"])
        return min(dp, fc.dp_max_Pa) if fc.dp_max_Pa and fc.dp_max_Pa>0 else dp

    def system_dp_without_fan(self, Q: float) -> Tuple[float, float, float, List[Dict[str,float]]]:
        """Retorna ΔP total (ductos + extras + filtro) y diagnósticos"""
        rho, mu = self.rho_air, self.mu_air
        total_dp = 0.0
        v_re_list = []
        for d in self.cfg.ducts:
            dp_i, v_i, Re_i = duct_dp(Q, rho, mu, d)
            total_dp += dp_i
            v_re_list.append({"D": d.D, "v": v_i, "Re": Re_i, "dp": dp_i})
        # extras (inlets/outlets)
        Ke = self.cfg.inout.K_extra if self.cfg.inout else 0.0
        if Ke and Ke > 0:
            # usar diámetro de la primera sección para velocidad de referencia
            if self.cfg.ducts:
                D0 = self.cfg.ducts[0].D
                A0 = math.pi*D0*D0/4
                v0 = Q/A0 if A0>0 else 0.0
                total_dp += 0.5*self.rho_air * v0*v0 * Ke
        # filtro
        total_dp += self.filter_dp(Q)
        return total_dp, (v_re_list[0]["v"] if v_re_list else 0.0), (v_re_list[0]["Re"] if v_re_list else 0.0), v_re_list

    def find_operating_Q(self) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
        """Encuentra Q tal que DP_fan(Q) = DP_sistema(Q)."""
        fan = self.cfg.fan

        # límites de búsqueda según curvas
        Qmin = max(0.0, min(fan.curve["Q"]))
        Qmax = max(fan.curve["Q"])
        # ampliar un poco el rango por robustez
        Qlo = 0.0
        Qhi = Qmax*1.2

        def F(Q: float) -> float:
            sys_dp, _, _, _ = self.system_dp_without_fan(Q)
            return fan.dp(Q) - sys_dp

        # bisección simple
        f_lo = F(Qlo)
        f_hi = F(Qhi)
        # si no cambian de signo, trata un punto medio de la curva
        if f_lo*f_hi > 0:
            Q_try = 0.5*(Qmin + Qmax)
            return Q_try, {"DP_fan_Pa": fan.dp(Q_try), "DP_sys_Pa": self.system_dp_without_fan(Q_try)[0]}, []

        for _ in range  (60):
            mid = 0.5*(Qlo + Qhi)
            f_mid = F(mid)
            if abs(f_mid) < 1e-3:
                break
            if f_lo * f_mid <= 0:
                Qhi = mid
                f_hi = f_mid
            else:
                Qlo = mid
                f_lo = f_mid
        Q = 0.5*(Qlo + Qhi)
        sys_dp, v0, Re0, v_re_list = self.system_dp_without_fan(Q)
        return Q, {"DP_fan_Pa": fan.dp(Q), "DP_sys_Pa": sys_dp, "v_duct0_m_s": v0, "Re_duct0": Re0}, v_re_list

    # -------- emisiones --------

    def _task_active(self, t: float, task: Task) -> bool:
        """Devuelve si en el ciclo actual de 'task' el sistema está 'on'."""
        period = task.t_on_s + task.t_off_s
        if period <= 0 or task.repetitions <= 0:
            return False
        # tiempo relativo dentro del bloque total de repeticiones
        t_mod = t % (period * task.repetitions)
        # en cada periodo, activo si t_mod % period < t_on_s
        return (t_mod % period) < task.t_on_s

    def emissions_step(self, t: float) -> Tuple[float, float]:
        """Retorna (S_vapor_kg_s, S_aer_kg_s). Maneja evaporación lenta remanente."""
        S_vapor = 0.0
        S_aer = 0.0
        # contribuciones de pistolas activas
        active_any = False
        for ti, task in enumerate(self.cfg.schedule.tasks):
            if self._task_active(t, task):
                active_any = True
                for pidx in task.active_pistols:
                    p = self.cfg.source.pistols[pidx]
                    q_L_s = (p.q_l_mL_min/1000.0)/60.0  # L/s
                    m_liq_kg_s = q_L_s * p.rho_l_kg_L
                    m_solvent_kg_s = m_liq_kg_s * p.w_s * (1.0 - p.TE)  # fracción que NO se queda en pieza
                    # división inmediata
                    S_vapor += p.f_evap_fast * m_solvent_kg_s
                    S_aer   += p.f_aer * m_solvent_kg_s
                    # remanente a "piso/superficie" que luego evapora
                    rem = max(0.0, m_solvent_kg_s - (p.f_evap_fast + p.f_aer)*m_solvent_kg_s)
                    self.residual_liq_kg[pidx] += rem
        # evaporación lenta de remanente
        for i, p in enumerate(self.cfg.source.pistols):
            if self.residual_liq_kg[i] > 0 and p.k_ev_s_inv > 0:
                evap = min(self.residual_liq_kg[i], self.residual_liq_kg[i]*p.k_ev_s_inv)
                self.residual_liq_kg[i] -= evap
                S_vapor += evap

        # captura en la fuente para el aerosol
        eta_capt = clamp(self.cfg.capture.eta_capt_source, 0.0, 1.0)
        S_aer_air = S_aer * (1.0 - eta_capt)
        return S_vapor, S_aer_air

    # -------- especies/equilibrio + coef. transferencia --------

    def species_equilibrium_gas(self) -> Dict[str, float]:
        """Fracciones molares en gas por Raoult ideal: y_i = x_i P_sat / P."""
        P_bar = self.P_Pa/1e5
        T = self.T_K
        y = {}
        tot = 0.0
        for sp in self.cfg.mixture.species:
            A,B,C = sp.Antoine
            Psat_bar = antoine_P_bar(T, A,B,C)
            yi = (sp.x_liq * Psat_bar) / (P_bar if P_bar>0 else 1.0)
            # evitar negativos/rangos raros si la ecuación no aplica
            yi = max(0.0, yi)
            y[sp.name] = yi
            tot += yi
        # normalizar si la suma supera 1 (casos fuera de rango)
        if tot > 1.0:
            for k in y:
                y[k] /= tot
            tot = 1.0
        y["AIR"] = max(0.0, 1.0 - tot)
        return y

    def diffusion_matrix_fuller(self) -> Dict[Tuple[str,str], float]:
        """D_ij (cm^2/s) para todas las parejas de especies y aire."""
        P_atm = self.P_Pa / 101325.0
        T = self.T_K
        # añadimos "Air" con M y v de Fuller típicos
        M_air = 28.97
        v_air = 20.1
        species = list(self.cfg.mixture.species)
        # matriz contra aire y entre solventes
        names = [sp.name for sp in species] + ["AIR"]
        Ms = [sp.M_g_mol for sp in species] + [M_air]
        vs = [sp.v_Fuller_cm3_mol for sp in species] + [v_air]
        D = {}
        for i in range(len(names)):
            for j in range(len(names)):
                if i==j:
                    continue
                Dij = D_Fuller_cm2_s(T, P_atm, Ms[i], Ms[j], vs[i], vs[j])
                D[(names[i], names[j])] = Dij
        return D

    def D_i_mix(self, y: Dict[str,float], D: Dict[Tuple[str,str], float]) -> Dict[str, float]:
        """Difusividad efectiva de cada especie (cm2/s) ignorando término con sí misma."""
        names = [sp.name for sp in self.cfg.mixture.species]
        out = {}
        for i in names:
            denom = 0.0
            for j in names + ["AIR"]]:
                if j == i:
                    continue
                yj = y.get(j, 0.0)
                Dij = D.get((i,j), 1e3)  # fallback grande
                if Dij <= 0:
                    Dij = 1e-12
                denom += yj / Dij
            if denom <= 0:
                out[i] = 1.0  # cm2/s
            else:
                out[i] = 1.0/denom
        return out

    def mass_transfer_coeffs(self, Q: float) -> Dict[str, float]:
        """h_m por especie (m/s) vía Sherwood en placa plana a partir del face velocity."""
        geom = self.cfg.geometry
        u = Q / (geom.area_face if geom.area_face>0 else 1.0)  # m/s
        nu = self.mu_air / self.rho_air  # m2/s
        ReL = u * geom.L / (nu if nu>0 else 1e-9)
        y = self.species_equilibrium_gas()
        Dmat = self.diffusion_matrix_fuller()
        D_i = self.D_i_mix(y, Dmat)  # cm2/s
        hm = {}
        for sp in self.cfg.mixture.species:
            Di_m2_s = D_i[sp.name] * 1e-4  # cm2/s -> m2/s
            Sc = nu/Di_m2_s if Di_m2_s>0 else 1e12
            if ReL < 5e5:
                Sh = 0.664 * (ReL**0.5) * (Sc**(1.0/3.0))
            else:
                Sh = 0.037 * (ReL**0.8) * (Sc**(1.0/3.0)) - 871.0 * (Sc**(1.0/3.0))
            hm[sp.name] = Sh * Di_m2_s / geom.L  # m/s
        return hm

    # -------- paso dinámico --------

    def step(self, dt: float) -> Dict[str, float]:
        cfg = self.cfg
        geom = cfg.geometry
        V = geom.get_V()

        # Caudal
        if cfg.sim.Q_fixed_m3_s and cfg.sim.Q_fixed_m3_s > 0:
            Q = cfg.sim.Q_fixed_m3_s
            diag_fan = {"DP_fan_Pa": self.cfg.fan.dp(Q), "DP_sys_Pa": self.system_dp_without_fan(Q)[0],
                        "v_duct0_m_s": (self.system_dp_without_fan(Q)[1])}
            v_re_list = self.system_dp_without_fan(Q)[3]
        else:
            Q, diag_fan, v_re_list = self.find_operating_Q()

        # Emisiones (kg/s)
        S_v_kg_s, S_p_kg_s = self.emissions_step(self.state["t"])

        # ODEs para concentraciones (mg/m3)
        C_v = self.state["C_v_mg_m3"]
        C_p = self.state["C_p_mg_m3"]
        sink_dep = cfg.capture.k_dep_s_inv

        dC_v = (S_v_kg_s*1e6)/V - (Q/V)*C_v - sink_dep*C_v
        dC_p = (S_p_kg_s*1e6)/V - (Q/V)*C_p - sink_dep*C_p

        C_v_next = max(0.0, C_v + dt*dC_v)
        C_p_next = max(0.0, C_p + dt*dC_p)

        # captura en filtro (aerosol en ducto)
        eta_f = clamp(cfg.filter.eta_filter, 0.0, 1.0)
        captured_kg_s = eta_f * Q * (C_p_next/1e6)  # kg/s
        self.state["M_captured_kg"] += captured_kg_s * dt

        # actualizar estado
        self.state["C_v_mg_m3"] = C_v_next
        self.state["C_p_mg_m3"] = C_p_next
        self.state["t"] += dt

        # diagnósticos extra (especies, hm, perfiles 1D simplificados)
        y_eq = self.species_equilibrium_gas()
        # masa molar promedio para ppm y partición de C_v
        names = [sp.name for sp in cfg.mixture.species]
        Mavg = sum(y_eq[n]*sp.M_g_mol for n,sp in zip(names, cfg.mixture.species))
        Mavg = Mavg if Mavg>0 else 60.0
        n_tot = self.P_Pa/(R_UNIVERSAL*self.T_K)  # mol/m3
        # ppmv total solvente usando C_v
        y_voc = ((C_v_next/1000.0)/Mavg) / n_tot  # mol/mol
        ppm_voc = max(0.0, y_voc*1e6)

        # reparto por especie proporcional a y_eq (solo para diagnóstico de perfiles)
        tot_y_solvs = sum(y_eq.get(sp.name,0.0) for sp in cfg.mixture.species)
        C_i = {}
        if tot_y_solvs > 0:
            for sp in cfg.mixture.species:
                frac = y_eq.get(sp.name,0.0)/tot_y_solvs
                C_i[sp.name] = frac * C_v_next
        else:
            for sp in cfg.mixture.species:
                C_i[sp.name] = 0.0

        hm = self.mass_transfer_coeffs(Q)
        u_face = Q/(geom.area_face if geom.area_face>0 else 1.0)

        # perfil simplificado: W(L) ≈ W0 * exp(-h_m*L/u)
        W_L = {}
        J0 = {}
        for sp in cfg.mixture.species:
            h = max(0.0, hm.get(sp.name, 0.0))
            if u_face > 0:
                W_L[sp.name] = C_i[sp.name] * math.exp(-h * geom.L / u_face)
            else:
                W_L[sp.name] = C_i[sp.name]
            # flujo de pared aproximado (mg/m2/s) usando gradiente película ~ h * C
            J0[sp.name] = h * C_i[sp.name]  # mg/(m2·s) si interpretamos 'C' como 'concentración másica'

        # Potencia eléctrica estimada (W)
        dp = diag_fan.get("DP_sys_Pa", 0.0)  # en equilibrio con el ventilador
        P_shaft = Q * dp / max(cfg.fan.eta_fan, 1e-6)
        P_elec = P_shaft / max(cfg.fan.eta_motor, 1e-6)

        # velocidades representativas
        v_duct0 = diag_fan.get("v_duct0_m_s", 0.0)

        # salida por paso
        out = {
            "t_s": self.state["t"],
            "Q_m3_s": Q,
            "V_face_m_s": u_face,
            "v_duct_m_s": v_duct0,
            "DP_system_Pa": diag_fan.get("DP_sys_Pa", 0.0),
            "DP_fan_Pa": diag_fan.get("DP_fan_Pa", 0.0),
            "C_v_mg_m3": C_v_next,
            "C_p_mg_m3": C_p_next,
            "C_tot_mg_m3": C_v_next + C_p_next,
            "ppm_voc": ppm_voc,
            "P_electrica_W": P_elec,
            "M_captured_kg": self.state["M_captured_kg"],
        }

        # añadir especies/hm/perfil (claves extendidas)
        for sp in cfg.mixture.species:
            nm = sp.name
            out[f"C_{nm}_mg_m3"] = C_i[nm]
            out[f"W_L_{nm}_mg_m3"] = W_L[nm]
            out[f"hm_{nm}_m_s"] = hm.get(nm, 0.0)
            out[f"Jwall_{nm}_mg_m2_s"] = J0[nm]

        # criterios (banderas)
        if cfg.criteria.V_face_obj is not None:
            out["ok_V_face"] = (u_face >= cfg.criteria.V_face_obj)
        if cfg.criteria.v_duct_obj is not None:
            out["ok_v_duct"] = (v_duct0 >= cfg.criteria.v_duct_obj)
        if cfg.criteria.C_limit_mg_m3 is not None:
            out["ok_C_limit"] = (C_v_next <= cfg.criteria.C_limit_mg_m3)
        if cfg.criteria.LFL_ppm is not None:
            out["ok_LFL"] = (ppm_voc <= cfg.criteria.LFL_ppm)

        # guarda algunos diagnósticos de ductos (primer segmento)
        if v_re_list:
            out["Re_duct0"] = v_re_list[0]["Re"]
            out["dp_duct0_Pa"] = v_re_list[0]["dp"]

        return out

    def run(self, dt: Optional[float]=None, t_end: Optional[float]=None) -> List[Dict[str, float]]:
        dt = dt if (dt is not None) else self.cfg.sim.dt_s
        t_end = t_end if (t_end is not None) else self.cfg.sim.t_end_s
        out = []
        while self.state["t"] < t_end - 1e-9:
            out.append(self.step(dt))
        return out

# -----------------------------
# Lectura de configuración
# -----------------------------

def config_from_dict(d: Dict) -> SystemConfig:
    # geometry
    g = d.get("geometry", {})
    geometry = Geometry(
        L = g.get("L", 2.0),
        A = g.get("A", 3.0),
        H = g.get("H", 2.5),
        area_face = g.get("area_face", 1.8),
        volume = g.get("volume", None),
        type = g.get("type", "cross-draft"),
    )
    # air
    a = d.get("air", {})
    air = AirConfig(
        altitude_m = a.get("altitude_m", 0.0),
        T_C = a.get("T_C", 20.0),
        RH = a.get("RH", 50.0),
        rho = a.get("rho", None),
        mu = a.get("mu", None),
    )
    # ducts
    ducts = [Duct(**dd) for dd in d.get("ducts", [])]
    # in/out
    inout = InOutExtras(**d.get("inlets_outlets", {"K_extra": 0.0}))
    # filter
    filt = d.get("filter", {})
    filter_cfg = FilterConfig(
        dp0_curve = filt.get("dp0_curve", {"Q":[0.0, 0.5, 1.0], "DP":[0.0, 50.0, 150.0]}),
        alpha_Pa_per_kg = filt.get("alpha_Pa_per_kg", 0.0),
        eta_filter = filt.get("eta_filter", 0.9),
        dp_max_Pa = filt.get("dp_max_Pa", 0.0),
    )
    # fan
    fan = d.get("fan", {})
    fan_cfg = FanConfig(
        curve = fan.get("curve", {"Q":[0.0, 0.5, 1.0, 1.5], "DP":[800.0, 700.0, 400.0, 150.0]}),
        eta_fan = fan.get("eta_fan", 0.6),
        eta_motor = fan.get("eta_motor", 0.9),
    )
    # source
    ps = [Pistol(**pp) for pp in d.get("source", {}).get("pistols", [])]
    source = SourceConfig(ps)
    # schedule
    tasks = [Task(**tt) for tt in d.get("schedule", {}).get("tasks", [])]
    schedule = Schedule(tasks=tasks,
                        n_tasks_day=d.get("schedule",{}).get("n_tasks_day",1),
                        purge_s=d.get("schedule",{}).get("purge_s",0.0))
    # criteria
    cr = d.get("criteria", {})
    criteria = Criteria(
        V_face_obj = cr.get("V_face_obj", None),
        ACH_obj = cr.get("ACH_obj", None),
        C_limit_mg_m3 = cr.get("C_limit_mg_m3", None),
        LFL_ppm = cr.get("LFL_ppm", None),
        v_duct_obj = cr.get("v_duct_obj", None),
    )
    # capture
    cap = d.get("capture_filtering", {})
    capture = CaptureFiltering(
        eta_capt_source = cap.get("eta_capt_source", 0.0),
        k_dep_s_inv = cap.get("k_dep_s_inv", 0.0),
    )
    # sim
    sm = d.get("sim", {})
    sim = SimSetup(
        dt_s = sm.get("dt_s", 1.0),
        t_end_s = sm.get("t_end_s", 3600.0),
        Q_fixed_m3_s = sm.get("Q_fixed_m3_s", None),
    )
    # mixture
    mix = d.get("mixture", None)
    if mix and "species" in mix:
        sp_list = []
        for sd in mix["species"]:
            sp_list.append(Species(
                name=sd["name"],
                M_g_mol=sd["M_g_mol"],
                v_Fuller_cm3_mol=sd["v_Fuller_cm3_mol"],
                Antoine=tuple(sd["Antoine"]),
                x_liq=sd.get("x_liq", 0.0),
                LFL_vol_frac=sd.get("LFL_vol_frac", None)
            ))
        mixture = MixtureConfig(sp_list)
    else:
        mixture = MixtureConfig([])

    return SystemConfig(
        geometry=geometry, air=air, ducts=ducts, inout=inout,
        filter=filter_cfg, fan=fan_cfg, source=source, schedule=schedule,
        criteria=criteria, capture=capture, sim=sim, mixture=mixture
    )

def config_from_json(path: str) -> SystemConfig:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return config_from_dict(d)
