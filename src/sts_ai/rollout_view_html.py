"""Pure HTML rendering for rollout decisions (no UI-framework dependency).

The combat-board / out-of-combat tiles were first written inline in the Streamlit
viewer (`scripts/visualize_rollout.py`); the *builders* there import nothing from
Streamlit, so they live here as a shared, unit-testable module. The Interactive
Studio server injects `render_decision_html` directly; the Streamlit viewer can
import the same builders (dedup).

`BOARD_CSS` carries the tile styles **plus** standalone layout classes (the
Streamlit app used `st.columns`/`st.metric` for layout, which the browser SPA does
not have). Card tiles are coloured by **state** (chosen / unaffordable), not by
card type — type isn't in the records (the "records only" data scope).
"""
from __future__ import annotations

import html
from collections import Counter

from sts_ai.rollout_view import CardView, CombatView, DecisionView, EnemyView

# Tile styles (lifted verbatim from scripts/visualize_rollout.py) + standalone
# layout classes for the framework-free SPA. Emitted once into the page <head>.
BOARD_CSS = """
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
/* standalone layout (replaces st.columns / st.metric) */
.sts-board{color:#dfe3ee;}
.sts-headline{font-size:18px;font-weight:700;margin:0 0 6px 0;}
.sts-metrics{display:flex;gap:18px;flex-wrap:wrap;margin:4px 0 10px 0;}
.sts-metric{display:flex;flex-direction:column;}
.sts-metric .k{font-size:11px;color:#9aa3b8;text-transform:uppercase;letter-spacing:.04em;}
.sts-metric .v{font-size:17px;font-weight:700;}
.sts-two{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;}
.sts-two>div{flex:1 1 280px;min-width:0;}
.sts-secttitle{font-weight:700;margin:10px 0 4px 0;}
.sts-actions{display:flex;flex-direction:column;gap:3px;}
.sts-action{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#aab;
  padding:2px 6px;border-radius:6px;}
.sts-action.chosen{color:#fff;background:rgba(245,197,24,.16);border:1px solid rgba(245,197,24,.5);font-weight:700;}
.sts-reasoning{background:rgba(74,134,176,.14);border:1px solid rgba(74,134,176,.4);
  border-radius:8px;padding:8px 11px;margin:8px 0;white-space:pre-wrap;}
.sts-thinking{margin:8px 0;}
.sts-thinking summary{cursor:pointer;color:#b794e0;font-weight:600;}
.sts-thinking .body{white-space:pre-wrap;font-size:13px;color:#cbd;margin-top:6px;
  max-height:280px;overflow:auto;border-left:2px solid #6b4f93;padding-left:10px;}
.sts-after{color:#9aa3b8;font-size:12px;margin-top:6px;}
.sts-raw{margin-top:8px;}
.sts-raw summary{cursor:pointer;color:#9aa3b8;font-size:12px;}
.sts-raw pre{white-space:pre-wrap;font-size:12px;color:#bcc;background:rgba(0,0,0,.25);
  border-radius:6px;padding:8px;max-height:320px;overflow:auto;}
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


def _metric(label: str, value: object) -> str:
    return f'<div class="sts-metric"><span class="k">{html.escape(label)}</span><span class="v">{html.escape(str(value))}</span></div>'


def _actions_html(dv: DecisionView) -> str:
    rows = []
    for a in dv.actions:
        cls = "sts-action chosen" if a.chosen else "sts-action"
        prefix = "➡️ " if a.chosen else ""
        rows.append(f'<div class="{cls}">{prefix}{a.index}: {html.escape(a.description)}</div>')
    return f'<div class="sts-actions">{"".join(rows)}</div>'


def _reasoning_and_thinking_html(dv: DecisionView) -> str:
    out = ""
    if dv.reasoning:
        out += f'<div class="sts-secttitle">Reasoning</div><div class="sts-reasoning">{html.escape(dv.reasoning)}</div>'
    if dv.thinking:
        out += (f'<details class="sts-thinking"><summary>🧠 Thinking ({len(dv.thinking)} chars)</summary>'
                f'<div class="body">{html.escape(dv.thinking)}</div></details>')
    if dv.raw_response and not dv.valid:
        out += (f'<details class="sts-raw"><summary>raw response (invalid)</summary>'
                f'<pre>{html.escape(dv.raw_response)}</pre></details>')
    return out


def render_combat_board_html(cv: CombatView) -> str:
    enemies = ("".join(_enemy_panel_html(e) for e in cv.enemies)
               or '<span class="sts-dim">(no enemies)</span>')
    hand = ("".join(_card_tile_html(c) for c in cv.hand)
            or '<span class="sts-dim">(empty hand)</span>')
    potions = ", ".join(cv.potions) if cv.potions else "—"
    return (
        '<div class="sts-two">'
        f'<div><div class="sts-secttitle">⚔️ Enemies · turn {cv.turn}</div>'
        f'<div class="sts-col">{enemies}</div></div>'
        f'<div><div class="sts-secttitle">🧍 Player</div>{_player_panel_html(cv)}</div>'
        '</div>'
        f'<div class="sts-secttitle">🃏 Hand</div><div class="sts-wrap">{hand}</div>'
        f'<div class="sts-after">draw {cv.draw_count} · discard {cv.discard_count} · '
        f'exhaust {cv.exhaust_count}  ·  potions: {html.escape(potions)}  ·  dimmed = unaffordable</div>'
    )


def render_ooc_board_html(dv: DecisionView) -> str:
    deck = ("".join(_card_tile_html(c) for c in _deck_cards(dv.deck))
            or '<span class="sts-dim">(empty)</span>')
    relics = _chips_html(dv.relics, _relic_label) or '<span class="sts-dim">(none)</span>'
    potions = _chips_html(dv.potions) or '<span class="sts-dim">(none)</span>'
    return (
        f'<div class="sts-secttitle">Deck ({len(dv.deck)})</div><div class="sts-wrap">{deck}</div>'
        f'<div class="sts-secttitle">Relics</div>{relics}'
        f'<div class="sts-secttitle">Potions</div>{potions}'
    )


def render_decision_html(dv: DecisionView) -> str:
    """Full board for one decision (combat or out-of-combat) as a single HTML
    string. The caller must include ``BOARD_CSS`` once in the page <head>."""
    cv = dv.combat
    cur_hp = cv.player_cur_hp if cv else dv.cur_hp
    max_hp = cv.player_max_hp if cv else dv.max_hp
    tag = "⚔️ Combat" if cv else (dv.screen or "decision")

    metrics = _metric("HP", f"{cur_hp} / {max_hp}") + _metric("Gold", dv.gold)
    if cv:
        metrics += _metric("Energy", f"{cv.player_energy}/{cv.player_energy_max}") + _metric("Turn", cv.turn)
    else:
        metrics += _metric("Deck", len(dv.deck)) + _metric("Room", dv.room or "—")

    board = render_combat_board_html(cv) if cv else render_ooc_board_html(dv)

    after = ""
    if dv.combat is None and dv.hp_after is not None:
        delta = dv.hp_after - dv.cur_hp
        note = f" ({'+' if delta > 0 else ''}{delta} HP)" if delta else ""
        after = (f'<div class="sts-after">after: HP {dv.cur_hp} → {dv.hp_after}{note}, '
                 f'floor {dv.floor} → {dv.floor_after}</div>')

    return (
        '<div class="sts-board">'
        f'<div class="sts-headline">Floor {dv.floor} · Act {dv.act} · {html.escape(tag)} '
        f'— boss: {html.escape(dv.boss)}</div>'
        f'<div class="sts-metrics">{metrics}</div>'
        f'{_hp_bar_html(cur_hp, max_hp, 320)}'
        f'{board}'
        f'<div class="sts-secttitle">Legal actions</div>{_actions_html(dv)}'
        f'{_reasoning_and_thinking_html(dv)}'
        f'{after}'
        '</div>'
    )
