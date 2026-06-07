"""Operational dashboard — step-by-step crane instructions.

One random inbound/outbound order arrives at a time (inbound more likely). A random
valid crane action is chosen for each order; the dashboard shows a step-by-step
instruction sequence. Press **Next** to step through reshuffles then the final
retrieve/place (each press applies the step and highlights the next). Press
**Complete** to finish the current order and pull the next one.

    streamlit run dashboard_ops.py
"""
import dataclasses
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from depot_env import DepotEnv
from train import EXPERIMENTS
from evaluate import load_eval_config
from dashboard import (
    load_agent, render_yard,
    BLOCK_GAP, _CUBE_TRIS, _cube_corners,
)
import runs


STEP_ICON = {"inbound": "📦", "reshuffle": "🔄", "retrieve": "🚢"}
ORDER_ICON = {"inbound": "📦", "outbound": "🚢"}


# ─────────────────────────────────────────────────────────────────────────────
# 3-D highlight overlay
# ─────────────────────────────────────────────────────────────────────────────

def _highlight_traces(env, orange=(), blue=(), red=()):
    """Return Mesh3d traces to overlay coloured boxes on specific (bl,ba,r,h) slots."""
    Ba = env.Ba
    out = []
    for positions, color, opacity in [
        (orange, "orange", 0.80),
        (blue,   "blue",   0.35),
        (red,    "red",    0.80),
    ]:
        for bl, ba, r, h in positions:
            cx = bl * (Ba + BLOCK_GAP) + ba
            corners = _cube_corners(cx, r, h)
            out.append(go.Mesh3d(
                x=[v[0] for v in corners],
                y=[v[1] for v in corners],
                z=[v[2] for v in corners],
                i=[t[0] for t in _CUBE_TRIS],
                j=[t[1] for t in _CUBE_TRIS],
                k=[t[2] for t in _CUBE_TRIS],
                color=color, opacity=opacity, flatshading=True,
                hoverinfo="skip", showscale=False,
                lighting=dict(ambient=0.6, diffuse=0.8),
            ))
    return out


def render_yard_ops(env, step=None, show_numbers=True):
    """render_yard + coloured highlights for the current pending step."""
    fig = render_yard(env, show_numbers=show_numbers)
    if step is None:
        return fig

    orange, blue, red = [], [], []
    t = step["type"]
    if t == "inbound":
        blue.append(step["pos"])                # ghost box for the target (empty) slot
    elif t == "reshuffle":
        orange.append(step["from"])             # source container
        blue.append(step["to"])                 # destination slot (empty)
    elif t == "retrieve":
        red.append(step["pos"])                 # container about to be taken

    for trace in _highlight_traces(env, orange=orange, blue=blue, red=red):
        fig.add_trace(trace)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LiveOrderRun — drives the env one order at a time
# ─────────────────────────────────────────────────────────────────────────────

class LiveOrderRun:
    """
    Manages a DepotEnv in order-at-a-time mode.

    step_idx = index of the NEXT UNAPPLIED step in self.steps.
      -1          → no active order yet
      0..N-1      → current_step = steps[step_idx] (shown, not applied)
      >= N        → order complete
    """

    def __init__(self, depot_cfg, reward_cfg, model, vecnorm, warmstart=0.4, seed=42):
        self._depot_cfg = depot_cfg
        self._reward_cfg = reward_cfg
        self.model = model
        self.vecnorm = vecnorm
        self._warmstart = warmstart
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        ws_cfg = dataclasses.replace(self._depot_cfg, warmstart=self._warmstart)
        self.env = DepotEnv(ws_cfg, reward_config=self._reward_cfg)
        self.env.reset(seed=0)
        self.env.n_in_remaining = 0
        self.env.n_out_remaining = 0
        self.steps: list[dict] = []
        self.step_idx: int = -1
        self.order_text: str = ""
        self.order_num: int = 0
        self.history: list[dict] = []
        self.total_reshuffles: int = 0
        self.total_travel: float = 0.0
        self.orders_done: int = 0

    # ── Agent ─────────────────────────────────────────────────────────────────

    def _predict(self, mask) -> int:
        obs = self.env._get_obs()
        nobs = self.vecnorm.normalize_obs(obs.astype(np.float32))
        a, _ = self.model.predict(nobs, action_masks=mask, deterministic=True)
        return int(np.asarray(a).reshape(-1)[0])

    # ── Order generation — inbound preferred ─────────────────────────────────

    def _random_order_type(self) -> str | None:
        occ = self.env._yard_occupancy()
        cap = self.env.grid.size
        can_in, can_out = occ < cap, occ > 0
        if can_in and can_out:
            return "inbound" if self.rng.random() < 0.7 else "outbound"
        return "inbound" if can_in else ("outbound" if can_out else None)

    # ── Step planning — runs reshuffler on a copy, never touches the real grid ─

    def _plan(self, action) -> list[dict]:
        decoded = self.env.decode_action(action)

        if decoded[0] == "inbound":
            _, bl, ba, r = decoded
            h = self.env._stack_height(bl, ba, r)
            return [{"type": "inbound", "pos": (bl, ba, r, h),
                     "text": (f"Place incoming container at  "
                              f"Block {bl+1} · Bay {ba+1} · Row {r+1} · Tier {h+1}")}]

        _, bl, ba, r, h = decoded
        # Run reshuffler on a grid copy to capture the move sequence
        saved = self.env.grid
        self.env.grid = saved.copy()
        moves = self.env.reshuffler.relocate_above(bl, ba, r, h)
        self.env.grid = saved   # restore real grid

        steps: list[dict] = []
        for mv in moves:
            _, sb, sba, sr, sh, db, dba, dr, dh = mv
            dwell = int(saved[sb, sba, sr, sh])
            steps.append({
                "type": "reshuffle", "dwell": dwell,
                "from": (sb, sba, sr, sh),
                "to":   (db, dba, dr, dh),
                "text": (f"Move container (dwell {dwell})  "
                         f"B{sb+1}·Bay{sba+1}·R{sr+1}·T{sh+1}"
                         f"  →  "
                         f"B{db+1}·Bay{dba+1}·R{dr+1}·T{dh+1}"),
            })

        ret_dwell = int(saved[bl, ba, r, h])
        steps.append({
            "type": "retrieve", "pos": (bl, ba, r, h), "dwell": ret_dwell,
            "text": (f"Take container (dwell {ret_dwell} ticks) from  "
                     f"Block {bl+1} · Bay {ba+1} · Row {r+1} · Tier {h+1}"),
        })
        return steps

    # ── Step application ──────────────────────────────────────────────────────

    def _apply(self, step: dict):
        t = step["type"]
        if t == "inbound":
            bl, ba, r, h = step["pos"]
            self.env.grid[bl, ba, r, h] = 0
        elif t == "reshuffle":
            sb, sba, sr, sh = step["from"]
            db, dba, dr, dh = step["to"]
            self.env.grid[sb, sba, sr, sh] = -1
            self.env.grid[db, dba, dr, dh] = step["dwell"]
            self.total_reshuffles += 1
        elif t == "retrieve":
            bl, ba, r, h = step["pos"]
            dist = int(np.abs(self.env.crane_pos - np.array([bl, ba, r])).sum())
            self.env.grid[bl, ba, r, h] = -1
            self.env.crane_pos = np.array([bl, ba, r], dtype=np.int32)
            self.total_travel += dist
        self.history.append(step)

    def _flush(self):
        """Apply all remaining unapplied steps (used when Complete skips ahead)."""
        while self.step_idx < len(self.steps):
            self._apply(self.steps[self.step_idx])
            self.step_idx += 1

    # ── Public API ────────────────────────────────────────────────────────────

    def press_complete(self) -> bool:
        """Finish the current order (apply remaining steps) and start the next."""
        if self.step_idx >= 0:
            self._flush()
            self.env.grid[self.env.grid >= 0] += 1   # age containers at order end
            self.orders_done += 1

        order_type = self._random_order_type()
        if order_type is None:
            return False

        self.env.mode = 0 if order_type == "inbound" else 1
        self.env.n_in_remaining  = 1 if order_type == "inbound" else 0
        self.env.n_out_remaining = 0 if order_type == "inbound" else 1
        mask = self.env.action_masks()
        if not mask.any():
            return False

        action = self._predict(mask)
        self.order_text = order_type
        self.order_num += 1
        self.steps = self._plan(action)
        self.step_idx = 0          # show first step (not yet applied)
        return True

    def press_next(self):
        """Apply the current pending step and advance to the next."""
        if self.step_idx < len(self.steps):
            self._apply(self.steps[self.step_idx])
            self.step_idx += 1

    # ── View properties ───────────────────────────────────────────────────────

    @property
    def current_step(self) -> dict | None:
        """Step currently being shown (pending, not yet applied), or None."""
        if self.step_idx < 0 or self.step_idx >= len(self.steps):
            return None
        return self.steps[self.step_idx]

    @property
    def order_done(self) -> bool:
        return bool(self.steps) and self.step_idx >= len(self.steps)

    @property
    def has_next(self) -> bool:
        return 0 <= self.step_idx < len(self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def _instruction_box(step: dict):
    """Render the current instruction in a coloured callout box."""
    icon = STEP_ICON[step["type"]]
    t = step["type"]
    if t == "inbound":
        st.info(f"{icon}  {step['text']}", icon=None)
    elif t == "reshuffle":
        st.warning(f"{icon}  {step['text']}", icon=None)
    elif t == "retrieve":
        st.success(f"{icon}  {step['text']}", icon=None)


def main():
    st.set_page_config(page_title="Depot Ops", layout="wide")
    st.title("🏗️ Container Depot — Operational View")

    exps = list(EXPERIMENTS)
    exp = st.sidebar.selectbox("Experiment", exps,
                               index=exps.index("full") if "full" in exps else 0)
    agent = load_agent(exp)
    if agent is None:
        st.error(f"No trained model for '{exp}'. Run: python train.py {exp}")
        st.stop()

    cfg = agent["depot_cfg"]
    st.sidebar.caption(
        f"run: `{os.path.basename(agent['run_dir'])}` · model: `{agent['model_name']}`")
    st.sidebar.caption(
        f"yard {cfg.n_blocks}×{cfg.n_bays}×{cfg.n_rows}×{cfg.n_tiers} (B×Ba×R×H)")

    warmstart = st.sidebar.slider(
        "Yard pre-fill", min_value=0.0, max_value=0.8, value=0.4, step=0.05,
        help="Fraction of yard capacity filled at reset (0 = empty, 0.8 = near-full)")
    show_nums = st.sidebar.checkbox("Show dwell numbers", value=True)

    # Reset when experiment or warmstart level changes
    key = f"ops_{exp}_{warmstart}"
    if st.sidebar.button("↺ Reset yard"):
        st.session_state.pop(key, None)
        st.rerun()

    # ── Session state ─────────────────────────────────────────────────────────
    if key not in st.session_state:
        st.session_state[key] = LiveOrderRun(
            cfg, agent["reward_cfg"], agent["model"], agent["vecnorm"],
            warmstart=warmstart)
    run: LiveOrderRun = st.session_state[key]

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Orders done", run.orders_done)
    m2.metric("Reshuffles", run.total_reshuffles)
    m3.metric("Crane travel", f"{run.total_travel:.0f}")
    occ = run.env._yard_occupancy()
    m4.metric("Yard occ.", f"{100 * occ / run.env.grid.size:.0f}%")

    # ── Main layout ───────────────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Yard")
        st.plotly_chart(render_yard_ops(run.env, run.current_step, show_nums),
                        width="stretch")
        st.caption(
            "Green cubes = containers (darker / more opaque → older dwell). "
            "🟠 Orange = container to move away. "
            "🔵 Blue ghost = target slot. "
            "🔴 Red = container to retrieve. "
            "Red ◆ = crane position.")

    with right:
        # ── Order header ───────────────────────────────────────────────────
        if run.step_idx < 0:
            st.info("Press **Complete** to receive the first order.")
        else:
            icon = ORDER_ICON.get(run.order_text, "📋")
            st.markdown(f"### {icon} {run.order_text.upper()} — Order #{run.order_num}")

            step = run.current_step
            if step is not None:
                # Step progress
                shown = run.step_idx + 1
                total = len(run.steps)
                st.caption(f"Step {shown} of {total}")
                st.progress(shown / total)
                _instruction_box(step)
            elif run.order_done:
                st.success("✅ Order complete — press **Complete** for the next order.")

        # ── Buttons ────────────────────────────────────────────────────────
        st.divider()
        b_next, b_complete = st.columns(2)

        if b_next.button("⏭ Next", disabled=not run.has_next,
                         use_container_width=True):
            run.press_next()
            st.rerun()

        if b_complete.button("✅ Complete", use_container_width=True):
            ok = run.press_complete()
            if not ok:
                st.error("Cannot generate an order (yard full or empty).")
            st.rerun()

        # ── Instruction history ────────────────────────────────────────────
        st.subheader("Instruction history")
        if run.history:
            for step in reversed(run.history[-20:]):
                icon = STEP_ICON.get(step["type"], "•")
                st.markdown(f"- {icon} {step['text']}")
        else:
            st.caption("No instructions completed yet.")


if __name__ == "__main__":
    main()
