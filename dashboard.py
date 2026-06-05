"""Interactive Streamlit dashboard for the depot RL agent.

    streamlit run dashboard.py

Loads the CURRENT model for the chosen experiment (via runs.py), lets you feed the yard
inbound/outbound demand tick by tick, and watch the trained agent place/retrieve containers
in a 3D yard view. Containers are green and solidify (opacity + colour) with dwell time; the
crane position is marked in red. Live metrics (total reward, reshuffle count, crane travel)
and a scrollable move log are shown alongside.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from depot_env import DepotEnv
from train import EXPERIMENTS
from evaluate import load_eval_config
import runs


# ──────────────────────────────────────────────────────────────────────────
# Controller — drive a DepotEnv interactively with user-supplied per-tick demand.
# Reuses the env's transitions, masking, observation and reward; only the
# tick/demand orchestration lives here, so the env file stays untouched.
# ──────────────────────────────────────────────────────────────────────────
class DashboardRun:
    def __init__(self, depot_cfg, reward_cfg, model, vecnorm):
        self.depot_cfg = depot_cfg
        self.reward_cfg = reward_cfg
        self.model = model
        self.vecnorm = vecnorm
        self.reset()

    def reset(self):
        self.env = DepotEnv(self.depot_cfg, reward_config=self.reward_cfg)
        self.env.reset(seed=0)
        self.env.grid[:] = -1                 # empty yard; we drive demand ourselves
        self.env.tick = 0
        self.env.crane_pos = np.array([0, 0, 0], dtype=np.int32)
        self.env.n_in_remaining = 0
        self.env.n_out_remaining = 0
        self.total_reward = 0.0
        self.total_travel = 0.0
        self.total_reshuffles = 0
        self.log = []

    def _predict(self, mask):
        obs = self.env._get_obs()
        nobs = self.vecnorm.normalize_obs(obs.astype(np.float32))
        a, _ = self.model.predict(nobs, action_masks=mask, deterministic=True)
        return int(np.asarray(a).reshape(-1)[0])

    def _mode_step(self):
        """Mirror DepotEnv._update_mode_or_advance, minus the (auto) tick advance.
        Returns True while the current tick still has an actionable move."""
        env = self.env
        if env.mode == 0:
            if env.n_in_remaining > 0 and not env._yard_full():
                return True
            env.n_in_remaining = 0
            if env.n_out_remaining > 0 and env._yard_occupancy() > 0:
                env.mode = 1
                return True
            env.n_out_remaining = 0
            return False
        if env.n_out_remaining > 0 and env._yard_occupancy() > 0:
            return True
        env.n_out_remaining = 0
        if env.n_in_remaining > 0 and not env._yard_full():
            env.mode = 0
            return True
        env.n_in_remaining = 0
        return False

    def step_tick(self, n_in, n_out):
        """Run the agent through one tick of the given demand, then age + advance."""
        env = self.env
        tick = env.tick
        env.n_in_remaining = int(n_in)
        env.n_out_remaining = int(n_out)
        moves = 0
        if env._resolve_mode():               # drop infeasible demand, pick start mode
            while True:
                mask = env.action_masks()
                if not mask.any():
                    break
                self._apply_and_log(self._predict(mask), tick)
                moves += 1
                if not self._mode_step():
                    break
        # end of tick: age occupied containers, advance, apply idle penalty
        env.grid[env.grid >= 0] += 1
        env.tick += 1
        tick_pen = env.reward_calc.tick_penalty()
        self.total_reward += tick_pen
        self.log.append(dict(tick=tick, op="— tick end —", coord="",
                             reward=round(tick_pen, 2),
                             travel=np.nan, reshuffles=np.nan, dwell=np.nan))
        return moves

    def _apply_and_log(self, action, tick):
        env = self.env
        decoded = env.decode_action(action)
        if decoded[0] == "inbound":
            _, bl, ba, r = decoded
            h = env._stack_height(bl, ba, r)            # tier it will land on
            env._apply_inbound(bl, ba, r)
            reward, _ = env.reward_calc.compute_inbound_step(bl, ba, r)
            self.total_reward += reward
            self.log.append(dict(tick=tick, op="inbound", coord=f"b{bl} y{ba} r{r} t{h}",
                                 reward=round(reward, 2), travel=0, reshuffles=0, dwell=np.nan))
        else:
            _, bl, ba, r, h = decoded
            retrieved = int(env.grid[bl, ba, r, h])      # read dwell BEFORE removal
            dist, rd = env._apply_outbound(bl, ba, r, h)
            reward, _ = env.reward_calc.compute_outbound_step(retrieved, dist, rd)
            self.total_reward += reward
            self.total_travel += dist
            self.total_reshuffles += len(rd)
            self.log.append(dict(tick=tick, op="outbound", coord=f"b{bl} y{ba} r{r} t{h}",
                                 reward=round(reward, 2), travel=dist,
                                 reshuffles=len(rd), dwell=retrieved))


# ──────────────────────────────────────────────────────────────────────────
# 3D yard rendering
# ──────────────────────────────────────────────────────────────────────────
BLOCK_GAP = 1        # empty columns between blocks along the x axis
BOX_W, BOX_H = 0.82, 0.9   # cuboid footprint / height (< 1 leaves seams between boxes)
N_OPACITY_BUCKETS = 5      # Mesh3d opacity is per-trace, so dwell-opacity is bucketed

# 12 triangles (2 per face) into the 8 corners of a box: bottom, top, then 4 sides.
_CUBE_TRIS = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
              (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
              (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]


def _cube_corners(cx, cy, cz):
    x0, x1 = cx - BOX_W / 2, cx + BOX_W / 2
    y0, y1 = cy - BOX_W / 2, cy + BOX_W / 2
    z0, z1 = cz, cz + BOX_H
    return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]


def _green_shade(t):
    """Fresh (t=0) -> light green, old (t=1) -> dark saturated green."""
    return f"rgb({int(150 * (1 - t))},{int(230 - 110 * t)},{int(150 * (1 - t))})"


def render_yard(env, show_numbers=True):
    """3-D yard: each container is a shaded cuboid, coloured + opacity-graded by dwell
    (fresh = light/translucent, old = dark/solid). Crane marked in red over its column."""
    B, Ba, R, H = env.B, env.Ba, env.R, env.H
    grid = env.grid
    occ = grid[grid >= 0]
    maxd = max(1, int(occ.max())) if occ.size else 1

    buckets = {}                       # opacity bucket -> {verts, tris, fc}
    lx, ly, lz, lt = [], [], [], []    # dwell-number label positions/text
    for bl in range(B):
        for ba in range(Ba):
            for r in range(R):
                for h in range(H):
                    d = int(grid[bl, ba, r, h])
                    if d < 0:
                        continue
                    t = min(1.0, d / maxd)
                    bk = int(round(t * (N_OPACITY_BUCKETS - 1)))
                    cx, cy, cz = bl * (Ba + BLOCK_GAP) + ba, r, h
                    data = buckets.setdefault(bk, dict(verts=[], tris=[], fc=[]))
                    base = len(data["verts"])
                    data["verts"].extend(_cube_corners(cx, cy, cz))
                    for tri in _CUBE_TRIS:
                        data["tris"].append((base + tri[0], base + tri[1], base + tri[2]))
                        data["fc"].append(_green_shade(t))
                    if show_numbers:
                        lx.append(cx); ly.append(cy); lz.append(cz + BOX_H / 2); lt.append(str(d))

    fig = go.Figure()

    # ground plane
    gx1 = (B - 1) * (Ba + BLOCK_GAP) + (Ba - 1) + 0.7
    fig.add_trace(go.Mesh3d(
        x=[-0.7, gx1, gx1, -0.7], y=[-0.7, -0.7, R - 0.3, R - 0.3], z=[-0.05] * 4,
        i=[0, 0], j=[1, 2], k=[2, 3], color="rgb(140,205,230)", opacity=0.5,
        hoverinfo="skip", showscale=False))

    # container cuboids — one Mesh3d per opacity bucket (Mesh3d opacity is per-trace)
    for bk, data in sorted(buckets.items()):
        op = 0.30 + 0.70 * (bk / (N_OPACITY_BUCKETS - 1))
        verts = data["verts"]
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in verts], y=[v[1] for v in verts], z=[v[2] for v in verts],
            i=[t[0] for t in data["tris"]], j=[t[1] for t in data["tris"]],
            k=[t[2] for t in data["tris"]],
            facecolor=data["fc"], opacity=op, flatshading=True, hoverinfo="skip",
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.1),
            lightposition=dict(x=120, y=200, z=300)))

    if show_numbers and lx:
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="text", text=lt,
                                   textfont=dict(size=9, color="black"), hoverinfo="skip"))

    # crane indicator over its column
    cb, cba, cr = (int(v) for v in env.crane_pos)
    cx, cy, cz = cb * (Ba + BLOCK_GAP) + cba, cr, H + 0.7
    fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[cz], mode="markers+text",
                               marker=dict(size=10, color="red", symbol="diamond"),
                               text=["crane"], textposition="top center",
                               textfont=dict(size=11, color="red")))
    fig.add_trace(go.Scatter3d(x=[cx, cx], y=[cy, cy], z=[0, cz], mode="lines",
                               line=dict(color="red", width=5, dash="dot"), hoverinfo="skip"))

    fig.update_layout(
        scene=dict(xaxis=dict(title="block · bay"), yaxis=dict(title="row"),
                   zaxis=dict(title="tier", range=[0, H + 1.5]),
                   aspectmode="data", camera=dict(eye=dict(x=1.6, y=1.6, z=1.05))),
        margin=dict(l=0, r=0, t=10, b=0), height=580, showlegend=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Streamlit app
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent(experiment):
    run_dir = runs.resolve_run_dir(experiment)
    depot_cfg, reward_cfg = load_eval_config(experiment, run_dir)
    model_path, vn_path = runs.resolve_model(run_dir)
    if model_path is None:
        return None
    vecnorm = VecNormalize.load(vn_path, DummyVecEnv([lambda: DepotEnv(depot_cfg)]))
    vecnorm.training = False
    model = MaskablePPO.load(model_path)
    return dict(run_dir=run_dir, depot_cfg=depot_cfg, reward_cfg=reward_cfg,
                model=model, vecnorm=vecnorm, model_name=os.path.basename(model_path))


def main():
    st.set_page_config(page_title="Depot RL Dashboard", layout="wide")
    st.title("🟩 Container Depot — RL Agent Dashboard")

    exps = list(EXPERIMENTS)
    exp = st.sidebar.selectbox("Experiment", exps,
                               index=exps.index("full") if "full" in exps else 0)
    agent = load_agent(exp)
    if agent is None:
        st.error(f"No trained model found for '{exp}'. Train one first: `python train.py {exp}`")
        st.stop()

    cfg = agent["depot_cfg"]
    st.sidebar.caption(f"run: `{os.path.basename(agent['run_dir'])}` · model: `{agent['model_name']}`")
    st.sidebar.caption(f"yard {cfg.n_blocks}×{cfg.n_bays}×{cfg.n_rows}×{cfg.n_tiers} "
                       f"(B×Ba×R×H) · n_ticks={cfg.n_ticks}")

    # (re)create the interactive run when the experiment changes
    if st.session_state.get("exp") != exp or "run" not in st.session_state:
        st.session_state.run = DashboardRun(cfg, agent["reward_cfg"], agent["model"], agent["vecnorm"])
        st.session_state.exp = exp
    run = st.session_state.run
    env = run.env

    capacity = env.grid.size
    occ = env._yard_occupancy()

    # ---- controls (sidebar) ----
    st.sidebar.subheader("Demand for next tick")
    n_in = st.sidebar.number_input("Inbound containers", 0, max(0, capacity - occ),
                                   value=min(3, max(0, capacity - occ)))
    n_out = st.sidebar.number_input("Outbound containers", 0, occ, value=min(2, occ))
    cstep, creset = st.sidebar.columns(2)
    if cstep.button("▶ Step tick", width="stretch"):
        run.step_tick(n_in, n_out)
        st.rerun()
    if creset.button("↺ Reset", width="stretch"):
        run.reset()
        st.rerun()
    show_numbers = st.sidebar.checkbox("Show dwell numbers", value=True)

    # ---- metrics ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total reward", f"{run.total_reward:.1f}")
    m2.metric("Reshuffles", run.total_reshuffles)
    m3.metric("Crane travel", f"{run.total_travel:.0f}")
    m4.metric("Tick", f"{env.tick} / {cfg.n_ticks}")

    # ---- yard + log ----
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Yard")
        st.plotly_chart(render_yard(env, show_numbers), width="stretch")
        st.caption("Each cube = a container; darker/more opaque = longer dwell (number on cube). "
                   "Red ◆ = crane. Blocks are spaced apart along the x axis.")
    with right:
        st.subheader("Move log (newest first)")
        if run.log:
            st.dataframe(pd.DataFrame(run.log[::-1]), height=520,
                         width="stretch", hide_index=True)
        else:
            st.info("Set the demand in the sidebar and press **Step tick** to start.")


if __name__ == "__main__":
    main()
