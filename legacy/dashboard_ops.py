"""Operational dashboard — step-by-step crane instructions.

One random inbound/outbound order arrives at a time (inbound more likely). A random
valid crane action is chosen for each order; the dashboard shows a step-by-step
instruction sequence. Press **Next** to step through reshuffles then the final
retrieve/place (each press applies the step and highlights the next). Press
**Complete** to finish the current order and pull the next one.

    streamlit run legacy/dashboard_ops.py

LEGACY — to be removed once the new frontend (deploy/) replaces it. The order/step
logic lives in legacy/ops.py.
"""
import os
import streamlit as st
import plotly.graph_objects as go

from ops import LiveOrderRun
from dashboard import (
    load_agent, render_yard,
    slot_xy, _CUBE_TRIS, _cube_corners,
)


STEP_ICON = {"inbound": "📦", "reshuffle": "🔄", "retrieve": "🚢"}
ORDER_ICON = {"inbound": "📦", "outbound": "🚢"}


# ─────────────────────────────────────────────────────────────────────────────
# 3-D highlight overlay
# ─────────────────────────────────────────────────────────────────────────────

def _highlight_traces(env, orange=(), blue=(), red=()):
    """Return Mesh3d traces to overlay coloured boxes on specific (bl,ba,r,h) slots."""
    out = []
    for positions, color, opacity in [
        (orange, "orange", 0.80),
        (blue,   "blue",   0.35),
        (red,    "red",    0.80),
    ]:
        for bl, ba, r, h in positions:
            cx, cy = slot_xy(env, bl, ba, r)
            corners = _cube_corners(cx, cy, h)
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

    agent = load_agent()
    if agent is None:
        st.error("No trained model found. Run: python training/train.py")
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

    # Reset when the warmstart level changes
    key = f"ops_{warmstart}"
    st.sidebar.divider()
    if st.sidebar.button("📅 Advance day", use_container_width=True,
                         help="Age all containers by one tick"):
        if key in st.session_state:
            st.session_state[key].advance_tick()
        st.rerun()
    if st.sidebar.button("↺ Reset yard", use_container_width=True):
        st.session_state.pop(key, None)
        st.rerun()

    # ── Session state ─────────────────────────────────────────────────────────
    if key not in st.session_state:
        st.session_state[key] = LiveOrderRun(
            cfg, agent["reward_cfg"], agent["policy"], warmstart=warmstart)
    run: LiveOrderRun = st.session_state[key]

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Day", run.env.tick)
    m2.metric("Orders done", run.orders_done)
    m3.metric("Reshuffles", run.total_reshuffles)
    m4.metric("Crane travel", f"{run.total_travel:.0f}")
    occ = run.env._yard_occupancy()
    m5.metric("Yard occ.", f"{100 * occ / run.env.grid.size:.0f}%")

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
