"""Streamlit visualiser for rollout traces.

Renders canonical rollout JSONL as click-through (or auto-playing) snapshots of
game state + the agent's decision and reasoning. Pure rendering over
sts_ai.rollout_view.

Run:
    .venv/bin/python -m pip install -e '.[viz]'   # once
    PYTHONPATH=src .venv/bin/streamlit run scripts/visualize_rollout.py
"""
from __future__ import annotations

import glob
import time
from collections import Counter

import streamlit as st

from sts_ai.rollout_view import DecisionView, load_rollout

st.set_page_config(page_title="StS rollout viewer", layout="wide")


def discover_files() -> list[str]:
    files = [f for f in glob.glob("data/**/*.jsonl", recursive=True) if ".error." not in f]
    return sorted(files)


def deck_summary(deck: list[str]) -> str:
    if not deck:
        return "_(empty)_"
    counts = Counter(deck)
    return ", ".join(f"{name} ×{n}" if n > 1 else name for name, n in sorted(counts.items()))


def render_decision(dv: DecisionView) -> None:
    hp_frac = dv.cur_hp / dv.max_hp if dv.max_hp else 0.0
    hp_color = "🟥" if hp_frac < 1 / 3 else ("🟨" if hp_frac < 2 / 3 else "🟩")

    st.subheader(f"Floor {dv.floor} · Act {dv.act} · {dv.screen}  —  boss: {dv.boss}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("HP", f"{dv.cur_hp} / {dv.max_hp}", help="current / max")
    m2.metric("Gold", dv.gold)
    m3.metric("Deck size", len(dv.deck))
    m4.metric("Room", dv.room or "—")
    st.progress(min(max(hp_frac, 0.0), 1.0), text=f"{hp_color} HP {dv.cur_hp}/{dv.max_hp}")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Deck**")
        st.markdown(deck_summary(dv.deck))
        st.markdown("**Relics**")
        st.markdown(", ".join(dv.relics) if dv.relics else "_(none)_")
        st.markdown("**Potions**")
        st.markdown(", ".join(dv.potions) if dv.potions else "_(none)_")
        with st.expander("Raw state text"):
            st.code(dv.state_text or "(none)")

    with right:
        validity = "✅ valid" if dv.valid else f"⚠️ invalid (fell back to 0)"
        retries = f" · {dv.retries} retr" if dv.retries else ""
        st.markdown(f"**Legal actions** · {validity}{retries}")
        for a in dv.actions:
            if a.chosen:
                st.markdown(f"➡️ **`{a.index}` {a.description}**")
            else:
                st.markdown(f"<span style='color:gray'>&nbsp;&nbsp;&nbsp;`{a.index}` {a.description}</span>",
                            unsafe_allow_html=True)

        if dv.reasoning:
            st.markdown("**Reasoning (action JSON)**")
            st.info(dv.reasoning)

        # consequence of this decision
        if dv.hp_after is not None:
            delta = dv.hp_after - dv.cur_hp
            arrow = "→"
            note = f" ({'+' if delta > 0 else ''}{delta} HP)" if delta else ""
            st.caption(f"after: HP {dv.cur_hp} {arrow} {dv.hp_after}{note}, floor {dv.floor} {arrow} {dv.floor_after}")

    if dv.thinking:
        with st.expander(f"🧠 Thinking trace ({len(dv.thinking)} chars)", expanded=False):
            st.markdown(dv.thinking)


def main() -> None:
    ss = st.session_state
    ss.setdefault("idx", 0)
    ss.setdefault("playing", False)
    ss.setdefault("loaded_path", None)

    st.sidebar.title("StS rollout viewer")
    files = discover_files()
    default_choice = files[0] if files else ""
    chosen = st.sidebar.selectbox("Rollout file", options=files or ["(no .jsonl under data/)"], index=0)
    custom = st.sidebar.text_input("…or path/glob", value="")
    path = custom.strip() or chosen
    if custom.strip() and ("*" in custom):
        matches = sorted(glob.glob(custom, recursive=True))
        if matches:
            path = st.sidebar.selectbox("matches", matches)

    if path != ss.loaded_path:
        ss.loaded_path = path
        ss.idx = 0
        ss.playing = False

    rollout = load_rollout(path)
    n = len(rollout.decisions)

    if rollout.error is not None:
        st.sidebar.error(f"error sidecar: {rollout.error.get('error', rollout.error)}")

    if n == 0:
        st.warning(f"No decisions in `{path}`. Pick a rollout file in the sidebar.")
        return

    ss.idx = max(0, min(ss.idx, n - 1))

    st.sidebar.caption(f"seed {rollout.seed} · {n} decisions · boss {rollout.boss}")

    def step(delta: int) -> None:
        ss.playing = False
        ss.idx = max(0, min(n - 1, ss.idx + delta))

    def toggle_play() -> None:
        ss.playing = not ss.playing

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    c1.button("⏮ Prev", on_click=step, args=(-1,), use_container_width=True, disabled=ss.idx == 0)
    c2.button("Next ⏭", on_click=step, args=(1,), use_container_width=True, disabled=ss.idx >= n - 1)
    c3.button("⏸ Pause" if ss.playing else "▶ Play", on_click=toggle_play, use_container_width=True)
    speed = c4.slider("seconds / step", 0.25, 3.0, 1.0, 0.25)

    sel = st.slider("decision", 0, n - 1, ss.idx)
    if sel != ss.idx and not ss.playing:
        ss.idx = sel

    st.caption(f"decision {ss.idx + 1} / {n}")
    render_decision(rollout.decisions[ss.idx])

    # autoplay: advance one step per `speed` seconds. ss.idx is not a widget key,
    # so mutating it here and rerunning is safe.
    if ss.playing:
        if ss.idx >= n - 1:
            ss.playing = False
        else:
            time.sleep(speed)
            ss.idx += 1
            st.rerun()


if __name__ == "__main__":
    main()
