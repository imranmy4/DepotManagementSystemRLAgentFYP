"""Depot demo: the deployed system (Postgres, Redis, worker + model) driven from one page.

    cd deploy && docker compose up -d          # postgres + redis
    streamlit run demo/depot_demo.py

The page runs the order worker itself, so do NOT also start `python -m app.worker`.

Sidebar: build a yard (fill level; ages "structured" = long-tailed, oldest at the bottom as in
training, or "uniform" = random per container), set the daily gate traffic, skip time, and
autoplay. Autoplay repeats: confirm the crane's next move -> plan the next queued order with
the model -> when the queue is empty, end the day (time +1 day) and generate the next day's
orders. The 3-D yard view and instruction box are the legacy dashboard's.
"""
import random
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim  # noqa: E402
from dashboard_ops import _instruction_box, render_yard_ops  # noqa: E402

STEP_TYPE = {"store": "inbound", "relocate": "reshuffle", "retrieve": "retrieve"}
STEP_ICON = {"store": "📦", "relocate": "🔄", "retrieve": "🚚"}
STATUS_ICON = {"queued": "⏳", "in_progress": "🏗️", "done": "✅", "rejected": "⛔"}


@st.cache_resource(show_spinner="Loading the deployed model…")
def system() -> sim.System:
    return sim.System.connect()


def _where(p) -> str:
    return f"block {p[0]} · bay {p[1]} · row {p[2]} · tier {p[3]}"


def legacy_step(m: dict, env) -> dict:
    """A move in the legacy dashboard's step format (type, positions, text)."""
    t = STEP_TYPE[m["type"]]
    if m["type"] == "store":
        text = f"Store **{m['container_no']}** at {_where(m['dst'])}"
        return dict(type=t, pos=m["dst"], text=text)
    age = int(env.grid[m["src"]])
    if m["type"] == "relocate":
        text = (f"Move **{m['container_no']}** ({age} d) out of the way: "
                f"{_where(m['src'])} → {_where(m['dst'])}")
        return dict(type=t, **{"from": m["src"], "to": m["dst"]}, text=text)
    text = (f"Retrieve **{m['container_no']}** ({age} d in yard) from {_where(m['src'])} "
            f"→ truck · crane travel {m['travel']}")
    return dict(type=t, pos=m["src"], text=text)


def sidebar(sys_: sim.System) -> dict:
    cfg, rc = sys_.policy.depot_config, sys_.policy.reward_config
    st.sidebar.caption(f"model `{sys_.decided_by}` · yard "
                       f"{cfg.n_blocks}×{cfg.n_bays}×{cfg.n_rows}×{cfg.n_tiers}")

    with st.sidebar.expander("🏗️ Yard conditions", expanded=True):
        fill = st.slider("Fill", 0, 95, 50, 5, format="%d%%")
        pattern = st.radio("Container ages", ["structured", "uniform"], horizontal=True,
                           help="structured: long-tailed ages, oldest at the bottom of each "
                                "stack (as in training). uniform: random age per container, "
                                "in any order (a messy yard).")
        age_max = st.slider("Max age (days)", 10, 365, 180)
        age_mean = st.slider("Mean age (days)", 1, 120, 40,
                             disabled=pattern == "uniform",
                             help="uniform ages average half the max age")
        seed = st.number_input("Seed", 0, 9999, 0)
        if st.button("Build yard (wipes everything)", width="stretch", type="primary"):
            with st.spinner("Placing containers through inbound orders…"):
                n = sim.build_yard(sys_, fill / 100, pattern, age_mean, age_max, int(seed))
            st.session_state.day = 0
            st.session_state.rng = random.Random(int(seed))
            st.session_state.flash = f"Yard built: {n} containers ({pattern} ages)"
            st.rerun()

    with st.sidebar.expander("🚚 Daily gate traffic", expanded=True):
        max_in = st.slider("Max inbound / day", 0, 30, cfg.max_in)
        max_out = st.slider("Max outbound / day", 0, 30, cfg.max_out)
        c1, c2 = st.columns(2)
        if c1.button("New orders", width="stretch", help="One day of random orders, now"):
            n_in, n_out = sim.new_day(sys_, max_in, max_out, st.session_state.rng)
            st.session_state.flash = f"Queued {n_in} inbound and {n_out} outbound orders"
            st.rerun()
        if c2.button("Clear queue", width="stretch", help="Reject every queued order"):
            sim.reject_queued()
            st.rerun()

    with st.sidebar.expander("⏩ Time", expanded=True):
        days = st.number_input("Days", 1, 365, 7)
        if st.button(f"Skip {days} day(s)", width="stretch"):
            sim.skip_time(days)
            st.session_state.day += days
            st.rerun()

    with st.sidebar.expander("▶ Autoplay", expanded=True):
        play = st.toggle("Run", value=False)
        auto_days = st.checkbox("Start a new day when the queue is empty", value=True)
        speed = st.select_slider("Actions per refresh", [1, 2, 5, 10, 25], value=2)
        every = st.select_slider("Refresh every", [0.5, 1.0, 2.0], value=1.0,
                                 format_func=lambda s: f"{s:g} s")
    show_nums = st.sidebar.checkbox("Show ages on containers", value=True)
    return dict(play=play, auto_days=auto_days, speed=speed, every=every, max_in=max_in,
                max_out=max_out, show_nums=show_nums, deadline=rc.max_age)


def live(sys_: sim.System, ui: dict):
    if ui["play"]:
        for _ in range(ui["speed"]):
            done = sim.auto_step(sys_, ui["auto_days"], ui["max_in"], ui["max_out"],
                                 st.session_state.rng)
            if done == "day":
                st.session_state.day += 1
            if done in ("day", "idle"):
                break

    env, current, moves, orders, totals = sim.snapshot(sys_)
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    grid = env.grid
    ages = grid[grid >= 0]
    n, cap = int(ages.size), int(grid.size)
    per = max(totals["retrieved"], 1)
    m = st.columns(8)
    m[0].metric("Day", st.session_state.day)
    m[1].metric("Containers", f"{n} / {cap}", f"{100 * n / cap:.0f}%", delta_color="off")
    m[2].metric("Mean age", f"{ages.mean():.0f} d" if n else "—")
    m[3].metric(f"Overdue >{ui['deadline']:.0f} d", int((ages > ui["deadline"]).sum()))
    m[4].metric("Queued", totals["queued"])
    m[5].metric("Released", totals["retrieved"])
    m[6].metric("Reshuffles / 100", f"{100 * totals['reshuffles'] / per:.0f}")
    m[7].metric("Travel / release", f"{totals['travel'] / per:.1f}")

    nxt = next((mv for mv in moves if mv["status"] == "planned"), None)
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(render_yard_ops(env, legacy_step(nxt, env) if nxt else None,
                                        ui["show_nums"]), width="stretch")
        st.caption("Green = containers, darker = older (number = days in yard). "
                   "🟠 container to move · 🔵 where it goes · 🔴 container leaving · "
                   "red ◆ = crane.")
    with right:
        crane_panel(sys_, env, current, moves, nxt, totals)
        st.subheader("Orders")
        if orders:
            for o in orders:
                o[""] = STATUS_ICON.get(o["status"], "")
            st.dataframe(orders, hide_index=True, width="stretch", height=260,
                         column_order=["", "order", "type", "status", "container", "note"])
        else:
            st.caption("No orders yet: build a yard, then queue orders or start autoplay.")


def crane_panel(sys_, env, current, moves, nxt, totals):
    if current is None:
        st.info("Crane free." + (f" {totals['queued']} order(s) queued." if totals["queued"]
                                 else " No orders waiting."))
        if st.button("🧠 Plan next order (model)", width="stretch",
                     disabled=not totals["queued"]):
            sim.plan_next(sys_)
            st.rerun()
        return
    kind = "📦 INBOUND" if current["type"] == "inbound" else "🚚 OUTBOUND"
    st.markdown(f"### {kind} — order #{current['order_id']}")
    done = sum(mv["status"] == "done" for mv in moves)
    st.progress(done / max(len(moves), 1),
                text=f"step {min(done + 1, len(moves))} of {len(moves)}")
    if nxt:
        _instruction_box(legacy_step(nxt, env))
    for mv in moves:
        mark = "✅" if mv["status"] == "done" else ("👉" if mv is nxt else "▫️")
        st.markdown(f"{mark} {STEP_ICON[mv['type']]} {mv['container_no']} · {mv['type']}")
    if current["type"] == "outbound":
        n_resh = sum(mv["type"] == "relocate" for mv in moves)
        st.caption(f"The model chose a container needing {n_resh} reshuffle(s).")
    if st.button("✔ Confirm move", type="primary", width="stretch", disabled=nxt is None):
        sim.confirm_next()
        st.rerun()


def main():
    st.set_page_config(page_title="Depot demo", layout="wide")
    st.title("🏗️ Container depot — live demo")
    try:
        sys_ = system()
    except Exception as e:                    # database / Redis / model not available
        st.error(f"Cannot start the system: {e}")
        st.info("Start Postgres and Redis (`cd deploy && docker compose up -d`), make sure the "
                "yard is seeded (`cd deploy/api && python -m app.db.seed`) and a model is "
                "promoted (`python training/promote.py`).")
        st.stop()
    st.session_state.setdefault("day", 0)
    st.session_state.setdefault("rng", random.Random(0))
    ui = sidebar(sys_)
    st.fragment(run_every=ui["every"] if ui["play"] else None)(live)(sys_, ui)


if __name__ == "__main__":
    main()
