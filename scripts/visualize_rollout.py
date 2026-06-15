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
import html
import time
from collections import Counter

import streamlit as st

from sts_ai.rollout_view import CardView, CombatView, DecisionView, EnemyView, load_rollout

st.set_page_config(page_title="StS rollout viewer", layout="wide")


def discover_files() -> list[str]:
    files = [f for f in glob.glob("data/**/*.jsonl", recursive=True) if ".error." not in f]
    return sorted(files)


# Tiles are drawn as raw HTML; each st.markdown call below emits a single-line
# HTML string (no leading indentation) so Streamlit's markdown pass doesn't treat
# it as a code block. The <style> block is re-emitted every run because a rerun
# re-renders the page from scratch. Neutral card tiles: card *type* (attack/skill/
# power) isn't in the records, so tiles are coloured by state (chosen / unplayable),
# not by type — see the "records only" data scope.
CSS = """
<style>
.sts-wrap{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 8px 0;align-items:flex-end;}
.sts-col{display:flex;flex-direction:column;gap:8px;margin:2px 0;}
.sts-card{position:relative;width:82px;height:110px;border-radius:9px;
  background:#262b38;border:1px solid #4b5266;box-shadow:0 1px 3px rgba(0,0,0,.4);
  color:#eef;overflow:hidden;flex:0 0 auto;}
.sts-card .sts-nm{position:absolute;bottom:7px;left:0;right:0;text-align:center;
  font-size:12px;font-weight:600;padding:0 4px;line-height:1.15;}
.sts-card .sts-cost{position:absolute;top:-1px;left:-1px;width:22px;height:22px;
  border-radius:50% 0 50% 50%;background:#3b6ea5;color:#fff;font-size:13px;font-weight:700;
  display:flex;align-items:center;justify-content:center;}
.sts-card .sts-count{position:absolute;top:4px;right:7px;font-size:11px;color:#cdd;}
.sts-card.upg .sts-nm{color:#9be29b;}
.sts-card.chosen{border:2px solid #f5c518;box-shadow:0 0 10px 2px rgba(245,197,24,.55);}
.sts-card.dim{opacity:.4;filter:grayscale(.6);}
.sts-panel{border-radius:8px;padding:7px 10px;border:1px solid #555;background:rgba(127,127,127,.10);}
.sts-enemy{border-color:#b0594a;background:rgba(176,74,58,.12);}
.sts-enemy.target{border:2px solid #f5c518;box-shadow:0 0 8px 1px rgba(245,197,24,.45);}
.sts-enemy.dead{opacity:.4;filter:grayscale(.8);}
.sts-player{border-color:#4a86b0;background:rgba(74,134,176,.12);}
.sts-erow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:5px;}
.sts-prow{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;}
.sts-enm{font-weight:700;font-size:14px;}
.sts-hp{position:relative;height:16px;border-radius:8px;background:#3a3a3a;overflow:hidden;max-width:100%;}
.sts-hpfill{position:absolute;top:0;left:0;bottom:0;}
.sts-hptext{position:relative;font-size:11px;color:#fff;font-weight:700;line-height:16px;
  padding-left:8px;text-shadow:0 1px 2px rgba(0,0,0,.85);}
.sts-pill{font-size:11px;padding:2px 7px;border-radius:10px;white-space:nowrap;
  background:rgba(127,127,127,.2);border:1px solid rgba(127,127,127,.45);}
.sts-pill.block{background:rgba(120,150,210,.28);}
.sts-pill.energy{background:rgba(220,180,40,.24);}
.sts-pill.tgt{background:rgba(245,197,24,.25);border-color:#f5c518;}
.sts-pill.pwr{background:rgba(150,90,200,.24);}
.sts-chip{display:inline-block;font-size:12px;padding:3px 9px;border-radius:11px;
  background:rgba(127,127,127,.16);border:1px solid rgba(127,127,127,.4);}
.sts-dim{color:#999;font-size:12px;}
</style>
"""


def _hp_hex(frac: float) -> str:
    if frac < 1 / 3:
        return "#d9534f"
    if frac < 2 / 3:
        return "#e0a800"
    return "#4caf50"


def _hp_bar_html(cur: int, mx: int, width: int) -> str:
    frac = max(0.0, min(1.0, cur / mx if mx else 0.0))
    return (f'<div class="sts-hp" style="width:{width}px">'
            f'<div class="sts-hpfill" style="width:{frac * 100:.0f}%;background:{_hp_hex(frac)}"></div>'
            f'<span class="sts-hptext">{cur}/{mx}</span></div>')


def _card_tile_html(card: CardView) -> str:
    cls = "sts-card"
    cls += " upg" if card.upgraded else ""
    cls += " chosen" if card.chosen else ""
    cls += "" if card.playable else " dim"
    cost = f'<div class="sts-cost">{html.escape(card.cost_text)}</div>' if card.cost_text is not None else ""
    count = f'<div class="sts-count">×{card.count}</div>' if card.count > 1 else ""
    return f'<div class="{cls}">{cost}{count}<div class="sts-nm">{html.escape(card.name)}</div></div>'


def _enemy_panel_html(e: EnemyView) -> str:
    cls = "sts-panel sts-enemy"
    cls += " dead" if not e.alive else ""
    cls += " target" if e.targeted else ""
    name = html.escape(e.display_name) + (" 💀" if not e.alive else "")
    intent = f'<span class="sts-pill">↪ {html.escape(e.intent_label)}</span>' if e.intent_label else ""
    block = f'<span class="sts-pill block">🛡 {e.block}</span>' if e.block else ""
    tgt = '<span class="sts-pill tgt">🎯 target</span>' if e.targeted else ""
    return (f'<div class="{cls}"><div class="sts-erow"><span class="sts-enm">{name}</span>'
            f'{intent}{block}{tgt}</div>{_hp_bar_html(e.cur_hp, e.max_hp, 170)}</div>')


def _player_panel_html(cv: CombatView) -> str:
    energy = f'<span class="sts-pill energy">⚡ {cv.player_energy}/{cv.player_energy_max}</span>'
    block = f'<span class="sts-pill block">🛡 {cv.player_block}</span>'
    powers = ("".join(f'<span class="sts-pill pwr">{html.escape(p)}</span>' for p in cv.powers)
              or '<span class="sts-dim">no powers</span>')
    return (f'<div class="sts-panel sts-player"><div class="sts-erow">'
            f'<span class="sts-enm">🧍 You</span>{energy}{block}</div>'
            f'{_hp_bar_html(cv.player_cur_hp, cv.player_max_hp, 220)}'
            f'<div class="sts-prow">{powers}</div></div>')


def _chips_html(items: list[str], labeler=None) -> str:
    if not items:
        return ""
    fn = labeler or (lambda x: x)
    chips = "".join(f'<span class="sts-chip">{html.escape(fn(i))}</span>' for i in items)
    return f'<div class="sts-wrap">{chips}</div>'


def _relic_label(relic: str) -> str:
    """'Burning Blood:0' -> 'Burning Blood'; keep a non-zero counter as '(n)'."""
    name, sep, cnt = relic.rpartition(":")
    if sep and cnt.isdigit():
        return f"{name} ({cnt})" if int(cnt) > 0 else name
    return relic


def _deck_cards(deck: list[str]) -> list[CardView]:
    counts = Counter(deck)
    return [CardView(name=n, upgraded=n.endswith("+"), count=c) for n, c in sorted(counts.items())]


def render_combat_board(cv: CombatView) -> None:
    col_e, col_p = st.columns([3, 2])
    with col_e:
        st.markdown(f"**⚔️ Enemies** · turn {cv.turn}")
        if cv.enemies:
            st.markdown('<div class="sts-col">' + "".join(_enemy_panel_html(e) for e in cv.enemies) + "</div>",
                        unsafe_allow_html=True)
        else:
            st.caption("(no enemies)")
    with col_p:
        st.markdown("**🧍 Player**")
        st.markdown(_player_panel_html(cv), unsafe_allow_html=True)

    target = next((e.display_name for e in cv.enemies if e.index == cv.chosen_target_index), None)
    if cv.chosen_card_name:
        played = f"played **{html.escape(cv.chosen_card_name)}**" + (f" → **{html.escape(target)}**" if target else "")
    elif cv.chosen_is_end_turn:
        played = "**ended turn** ⏹"
    else:
        played = ""
    st.markdown("**🃏 Hand**" + (f" · {played}" if played else ""))
    if cv.hand:
        st.markdown('<div class="sts-wrap">' + "".join(_card_tile_html(c) for c in cv.hand) + "</div>",
                    unsafe_allow_html=True)
    else:
        st.caption("(empty hand)")
    potions = ", ".join(cv.potions) if cv.potions else "—"
    st.caption(f"draw {cv.draw_count} · discard {cv.discard_count} · exhaust {cv.exhaust_count}"
               f"   ·   potions: {potions}   ·   dimmed cards = unaffordable")


def render_actions_and_reasoning(dv: DecisionView) -> None:
    validity = "✅ valid" if dv.valid else "⚠️ invalid (fell back to 0)"
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

    # consequence of this decision. Suppressed in combat: `hp_after` is the
    # map-level HP, which is stale within a turn, so the delta would be misleading.
    if dv.combat is None and dv.hp_after is not None:
        delta = dv.hp_after - dv.cur_hp
        note = f" ({'+' if delta > 0 else ''}{delta} HP)" if delta else ""
        st.caption(f"after: HP {dv.cur_hp} → {dv.hp_after}{note}, floor {dv.floor} → {dv.floor_after}")


def render_decision(dv: DecisionView) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    cv = dv.combat
    # In combat the live player HP lives in state["combat"]; state.cur_hp is the
    # (stale) map HP, so prefer the combat value.
    cur_hp = cv.player_cur_hp if cv else dv.cur_hp
    max_hp = cv.player_max_hp if cv else dv.max_hp
    hp_frac = cur_hp / max_hp if max_hp else 0.0
    hp_color = "🟥" if hp_frac < 1 / 3 else ("🟨" if hp_frac < 2 / 3 else "🟩")

    tag = "⚔️ Combat" if cv else dv.screen
    st.subheader(f"Floor {dv.floor} · Act {dv.act} · {tag}  —  boss: {dv.boss}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("HP", f"{cur_hp} / {max_hp}", help="current / max")
    m2.metric("Gold", dv.gold)
    if cv:
        m3.metric("Energy", f"{cv.player_energy}/{cv.player_energy_max}")
        m4.metric("Turn", cv.turn)
    else:
        m3.metric("Deck size", len(dv.deck))
        m4.metric("Room", dv.room or "—")
    st.progress(min(max(hp_frac, 0.0), 1.0), text=f"{hp_color} HP {cur_hp}/{max_hp}")

    if cv:
        render_combat_board(cv)
        with st.expander("Raw state text"):
            st.code(dv.state_text or "(none)")
        render_actions_and_reasoning(dv)
    else:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Deck**")
            if dv.deck:
                st.markdown('<div class="sts-wrap">' + "".join(_card_tile_html(c) for c in _deck_cards(dv.deck)) + "</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("_(empty)_")
            st.markdown("**Relics**")
            st.markdown(_chips_html(dv.relics, _relic_label) or "_(none)_", unsafe_allow_html=True)
            st.markdown("**Potions**")
            st.markdown(_chips_html(dv.potions) or "_(none)_", unsafe_allow_html=True)
            with st.expander("Raw state text"):
                st.code(dv.state_text or "(none)")
        with right:
            render_actions_and_reasoning(dv)

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
