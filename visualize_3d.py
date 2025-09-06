#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_3d.py
Visualización 3D dinámica aire–vapor con recirculación y sliders.
Fixes:
- Sliders: usar tuplas en plt.axes((...)) para evitar el error de tipo.
- Mantener referencia a la animación (anim = FuncAnimation(...)) para que no la recoleccione el GC.
- Siembra inicial de vapor para que se vea desde el inicio.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import argparse
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim_logic2 import config_from_json, Simulator


# -------------------- Geometría / utilidades --------------------

def booth_dims_from_cfg(cfg) -> Tuple[float, float, float]:
    L = float(cfg.geometry.L)
    H = float(cfg.geometry.H)
    A_floor = float(cfg.geometry.A)
    W = max(1e-9, A_floor / max(L, 1e-9))
    return L, W, H

def cross_section_area(cfg, W: float, H: float) -> float:
    A_face = float(getattr(cfg.geometry, "area_face", 0.0) or 0.0)
    return A_face if A_face > 0 else W * H

def make_transparent_box(ax, L: float, W: float, H: float, alpha=0.07, edge_alpha=0.25):
    v = np.array([
        [0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],
        [0, 0, H], [L, 0, H], [L, W, H], [0, W, H],
    ])
    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[0], v[3], v[7], v[4]],
    ]
    coll = Poly3DCollection(faces, alpha=alpha, facecolor="C0")
    coll.set_edgecolor((0, 0, 0, edge_alpha))
    ax.add_collection3d(coll)

def outlet_ring_coords(L: float, W: float, H: float, radius: float, n=100):
    cy, cz = W/2.0, H/2.0
    th = np.linspace(0, 2*np.pi, n)
    y = cy + radius*np.cos(th)
    z = cz + radius*np.sin(th)
    x = np.full_like(y, L)
    return x, y, z

def lin_interp(t: float, ts: np.ndarray, ys: np.ndarray) -> float:
    if len(ts) == 0:
        return float(ys[0]) if len(ys) else 0.0
    return float(np.interp(t, ts, ys, left=ys[0], right=ys[-1]))


# -------------------- Campo de velocidad (mezcla) --------------------

def velocity_field(pos: np.ndarray, t: float, U_mean: float,
                   L: float, W: float, H: float, swirl_scale: float) -> np.ndarray:
    if len(pos) == 0:
        return np.zeros((0, 3))
    x = pos[:, 0]; y = pos[:, 1]; z = pos[:, 2]
    mod = 0.12*np.sin(2*np.pi*y/max(W,1e-9)) * np.cos(2*np.pi*z/max(H,1e-9))
    Ux = U_mean * (1.0 + mod)
    omega_t = 0.5 + 0.3*np.sin(2*np.pi*t / max(L/max(U_mean,1e-9), 1.0))
    amp = swirl_scale * U_mean
    Uy = amp * ( np.sin(2*np.pi*x/max(L,1e-9) + 0.6*np.sin(0.4*t)) *
                 np.sin(np.pi*z/max(H,1e-9)) )
    Uz = amp * ( np.cos(2*np.pi*x/max(L,1e-9) + 0.4*np.cos(0.5*t)) *
                 np.sin(np.pi*y/max(W,1e-9)) )
    Uy += amp * 0.5 * np.cos(2*np.pi*y/max(W,1e-9) + omega_t*t) * np.cos(np.pi*z/max(H,1e-9))
    Uz += amp * 0.5 * np.sin(2*np.pi*z/max(H,1e-9) + omega_t*t) * np.cos(np.pi*y/max(W,1e-9))
    return np.stack([Ux, Uy, Uz], axis=1)


# -------------------- Sistema de partículas --------------------

class ParticleSystem:
    def __init__(self, max_particles: int, m_per_particle_kg: float, k_dep_s: float, rng: np.random.Generator):
        self.max_particles = int(max_particles)
        self.m_per_particle = float(m_per_particle_kg)
        self.k_dep = float(k_dep_s)
        self.rng = rng
        self.pos = np.empty((0, 3), dtype=float)
        self.age = np.empty((0,), dtype=float)

    def emit(self, n: int, center: np.ndarray, sigma: np.ndarray):
        if n <= 0:
            return
        new_pos = self.rng.normal(loc=center, scale=sigma, size=(n, 3))
        new_age = np.zeros((n,), dtype=float)
        self.pos = np.vstack([self.pos, new_pos])
        self.age = np.concatenate([self.age, new_age])
        if len(self.pos) > self.max_particles:
            over = len(self.pos) - self.max_particles
            self.pos = self.pos[over:, :]
            self.age = self.age[over:]

    def seed_uniform(self, n: int, L: float, W: float, H: float):
        if n <= 0:
            return
        x = self.rng.random(n) * L
        y = self.rng.random(n) * W
        z = self.rng.random(n) * H
        self.pos = np.vstack([self.pos, np.column_stack([x, y, z])])
        self.age = np.concatenate([self.age, np.zeros((n,), dtype=float)])
        if len(self.pos) > self.max_particles:
            over = len(self.pos) - self.max_particles
            self.pos = self.pos[over:, :]
            self.age = self.age[over:]

    def step(self, dt: float, t: float, LWH: Tuple[float, float, float],
             vel_fn, U_mean: float, swirl_scale: float, diff_sigma_m: float,
             outlet_mode: str = "tube", outlet_center: Tuple[float, float] = (0.0, 0.0),
             outlet_radius: float = 0.1):
        if len(self.pos) == 0:
            return
        L, W, H = LWH
        vel = vel_fn(self.pos, t, U_mean, L, W, H, swirl_scale)
        self.pos += vel * dt
        sigma = diff_sigma_m * np.sqrt(max(dt, 1e-9))
        self.pos += self.rng.normal(0.0, sigma, size=self.pos.shape)
        y = self.pos[:, 1]; z = self.pos[:, 2]
        y[y < 0] = -y[y < 0]; y[y > W] = 2*W - y[y > W]
        z[z < 0] = -z[z < 0]; z[z > H] = 2*H - z[z > H]
        self.pos[:, 1] = y; self.pos[:, 2] = z
        x = self.pos[:, 0]
        mask_in = x < 0
        if np.any(mask_in):
            x[mask_in] = -x[mask_in]
        mask_out = x > L
        keep = np.ones(len(self.pos), dtype=bool)
        if np.any(mask_out):
            if outlet_mode == "plane":
                keep &= ~mask_out
            else:
                cy, cz = outlet_center
                r2 = outlet_radius * outlet_radius
                inside_hole = ((self.pos[:, 1] - cy)**2 + (self.pos[:, 2] - cz)**2) <= r2
                exit_mask = mask_out & inside_hole
                keep &= ~exit_mask
                reflect_mask = mask_out & (~inside_hole)
                if np.any(reflect_mask):
                    x[reflect_mask] = 2*L - x[reflect_mask]
        self.pos[:, 0] = x
        if not np.all(keep):
            self.pos = self.pos[keep, :]
            self.age = self.age[keep]
        if self.k_dep > 0 and len(self.pos) > 0:
            dmin = np.minimum.reduce([
                self.pos[:, 1], W - self.pos[:, 1],
                self.pos[:, 2], H - self.pos[:, 2]
            ])
            base_p = 1.0 - np.exp(-self.k_dep * dt)
            wall_boost = np.clip(0.02 / np.maximum(dmin, 1e-6), 0.0, 4.0)
            p_local = np.clip(base_p * (1.0 + wall_boost), 0.0, 1.0)
            keep2 = (self.rng.random(len(self.pos)) > p_local)
            self.pos = self.pos[keep2, :]
            self.age = self.age[keep2]
        self.age += dt


# -------------------- Simulación de fondo --------------------

def run_sim_timeseries(cfg, dt_override: Optional[float], t_end_override: Optional[float]) -> pd.DataFrame:
    if dt_override is not None:
        cfg.sim.dt_s = float(dt_override)
    if t_end_override is not None:
        cfg.sim.t_end_s = float(t_end_override)
    sim = Simulator(cfg)
    rows = sim.run()
    if not rows:
        rows = [sim.step(0.0)]
    df = pd.DataFrame(rows)
    if "t_s" not in df:
        df["t_s"] = np.arange(len(df)) * cfg.sim.dt_s
    return df


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Visualización 3D dinámica aire–vapor con recirculación y sliders.")
    parser.add_argument("-c", "--config", type=str, default="example_config_B_cleaning_vfd.json")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t_end", type=float, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt_vis", type=float, default=None)
    parser.add_argument("--speed_scale", type=float, default=0.5)
    parser.add_argument("--swirl_scale", type=float, default=0.30)
    parser.add_argument("--diff_sigma", type=float, default=0.018)
    parser.add_argument("--max_particles_vapor", type=int, default=3000)
    parser.add_argument("--m_particle_vapor", type=float, default=2.2e-9)
    parser.add_argument("--emit_scale", type=float, default=0.85)
    parser.add_argument("--max_particles_air", type=int, default=1800)
    parser.add_argument("--air_seed", type=int, default=1400)
    parser.add_argument("--air_reseed_rate", type=int, default=70)
    parser.add_argument("--tube_radius", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    cfg = config_from_json(args.config)
    df = run_sim_timeseries(copy.deepcopy(cfg), args.dt, args.t_end)
    t_series = df["t_s"].values
    Q_series = df["Q_m3_s"].values if "Q_m3_s" in df.columns else np.zeros_like(t_series)
    dt_sim = float(cfg.sim.dt_s if args.dt is None else args.dt)

    L, W, H = booth_dims_from_cfg(cfg)
    A_cs = cross_section_area(cfg, W, H)
    outlet_center = (W/2.0, H/2.0)

    emitter_sim = Simulator(copy.deepcopy(cfg))
    t_emitter = 0.0

    k_dep = 0.0
    if hasattr(cfg, "capture") and hasattr(cfg.capture, "k_dep_s_inv"):
        k_dep = float(cfg.capture.k_dep_s_inv or 0.0)

    vap = ParticleSystem(max_particles=args.max_particles_vapor,
                         m_per_particle_kg=args.m_particle_vapor,
                         k_dep_s=k_dep,
                         rng=rng)
    air = ParticleSystem(max_particles=args.max_particles_air,
                         m_per_particle_kg=1.0,
                         k_dep_s=0.0,
                         rng=rng)

    air.seed_uniform(args.air_seed, L, W, H)

    emit_center = np.array([0.18 * L, 0.52 * W, 0.65 * H], dtype=float)
    emit_sigma  = np.array([0.02 * L, 0.05 * W, 0.05 * H], dtype=float)

    fps = max(1, int(args.fps))
    nframes = max(1, int(args.duration * fps))
    dt_vis = float(dt_sim if args.dt_vis is None else args.dt_vis)

    fig = plt.figure(figsize=(11.0, 7.8))
    plt.subplots_adjust(bottom=0.28)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((L, W, H))
    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Cabina de pintura · Mezcla aire–vapor (3D) · SOLO salida por ducto")

    make_transparent_box(ax, L, W, H, alpha=0.07, edge_alpha=0.25)
    x_ring, y_ring, z_ring = outlet_ring_coords(L, W, H, radius=args.tube_radius)
    [outlet_line] = ax.plot3D(x_ring, y_ring, z_ring, linewidth=2.0, alpha=0.7)
    src_handle = ax.scatter([emit_center[0]], [emit_center[1]], [emit_center[2]],
                            s=80, color=(1.0, 0.45, 0.1, 0.9), depthshade=False)
    air_scat = ax.scatter([], [], [], s=6,  color=(0.15, 0.65, 1.0, 0.50), depthshade=False)
    vap_scat = ax.scatter([], [], [], s=8,  color=(1.0, 0.40, 0.05, 0.85), depthshade=False)
    hud = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    # --- SLIDERS (usar TUPLAS, no listas) ---
    ax_speed  = plt.axes((0.12, 0.20, 0.33, 0.025))
    ax_swirl  = plt.axes((0.12, 0.16, 0.33, 0.025))
    ax_diff   = plt.axes((0.12, 0.12, 0.33, 0.025))
    ax_emit   = plt.axes((0.55, 0.20, 0.33, 0.025))
    ax_radius = plt.axes((0.55, 0.16, 0.33, 0.025))
    ax_reseed = plt.axes((0.55, 0.12, 0.33, 0.025))

    s_speed  = Slider(ax_speed,  "speed_scale",  0.05, 1.50, valinit=args.speed_scale,  valstep=0.01)
    s_swirl  = Slider(ax_swirl,  "swirl_scale",  0.00, 1.00, valinit=args.swirl_scale,  valstep=0.01)
    s_diff   = Slider(ax_diff,   "diff_sigma",   0.001, 0.08, valinit=args.diff_sigma,   valstep=0.001)
    s_emit   = Slider(ax_emit,   "emit_scale",   0.10, 2.50, valinit=args.emit_scale,    valstep=0.05)
    s_radius = Slider(ax_radius, "tube_radius",  0.05, min(W,H)/2.0, valinit=args.tube_radius, valstep=0.01)
    s_reseed = Slider(ax_reseed, "air_reseed/s", 0.0,  300.0, valinit=float(args.air_reseed_rate), valstep=1.0)

    # Interpolador
    def Q_of_t(t: float) -> float:
        return lin_interp(t, t_series, Q_series)

    t_anim = 0.0

    def reseed_air(n: int):
        if n <= 0:
            return
        center = np.array([0.01 * L, 0.5 * W, 0.5 * H])
        sigma  = np.array([0.004 * L, 0.25 * W, 0.25 * H])
        air.emit(n, center=center, sigma=sigma)

    # Siembra inicial de VAPOR para que se vea desde el primer frame
    vap.emit(250, center=emit_center, sigma=emit_sigma)

    def update(frame_idx: int):
        nonlocal t_anim, t_emitter

        speed_scale  = float(s_speed.val)
        swirl_scale  = float(s_swirl.val)
        diff_sigma   = float(s_diff.val)
        emit_scale   = float(s_emit.val)
        tube_radius  = float(s_radius.val)
        air_reseed   = float(s_reseed.val)

        xr, yr, zr = outlet_ring_coords(L, W, H, radius=tube_radius)
        outlet_line.set_data_3d(xr, yr, zr)

        Q_now = Q_of_t(t_anim)
        U_mean = (Q_now / max(A_cs, 1e-12)) * speed_scale

        S_v, S_aer = emitter_sim.emissions_step(t_emitter)
        t_emitter += dt_vis

        n_emit_vap = int((S_aer * dt_vis / max(vap.m_per_particle, 1e-18)) * emit_scale)
        n_emit_vap = int(min(n_emit_vap, max(1, vap.max_particles // 12)))
        vap.emit(n_emit_vap, center=emit_center, sigma=emit_sigma)

        vap.step(dt=dt_vis, t=t_anim, LWH=(L, W, H),
                 vel_fn=velocity_field, U_mean=U_mean, swirl_scale=swirl_scale,
                 diff_sigma_m=diff_sigma,
                 outlet_mode="tube", outlet_center=outlet_center, outlet_radius=tube_radius)

        air.step(dt=dt_vis, t=t_anim, LWH=(L, W, H),
                 vel_fn=velocity_field, U_mean=U_mean, swirl_scale=swirl_scale,
                 diff_sigma_m=diff_sigma*0.8,
                 outlet_mode="tube", outlet_center=outlet_center, outlet_radius=tube_radius)

        if len(air.pos) < air.max_particles:
            deficit = air.max_particles - len(air.pos)
            n_reseed = min(deficit, int(air_reseed * dt_vis + 0.5))
            reseed_air(n_reseed)

        if len(air.pos) > 0:
            ax_air = air.pos
            air_scat._offsets3d = (ax_air[:, 0], ax_air[:, 1], ax_air[:, 2])
        else:
            air_scat._offsets3d = ([], [], [])

        if len(vap.pos) > 0:
            ax_vap = vap.pos
            vap_scat._offsets3d = (ax_vap[:, 0], ax_vap[:, 1], ax_vap[:, 2])
        else:
            vap_scat._offsets3d = ([], [], [])

        hud.set_text(
            f"t = {t_anim:6.1f} s   |   Q = {Q_now:5.2f} m³/s   |   Ū = {U_mean:4.2f} m/s\n"
            f"Aire: {len(air.pos):4d}  ·  Vapor: {len(vap.pos):4d}  ·  R_tubo = {tube_radius:.2f} m"
        )

        t_anim += dt_vis
        return air_scat, vap_scat, hud, src_handle, outlet_line

    interval_ms = 1000.0 / max(1, int(args.fps))
    # Mantener referencia para que no sea recolectado
    anim = FuncAnimation(fig, update, frames=nframes, interval=interval_ms, blit=False, repeat=False)

    plt.show()


if __name__ == "__main__":
    main()
