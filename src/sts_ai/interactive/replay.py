"""Replay helpers for checkpoint/branch.

The C++ `GameContext`/`BattleContext` are opaque — there is no binary state
snapshot. The only way to reach a mid-game position is to replay from
`world_seed` + the recorded action sequence. These helpers re-apply a recorded
action sequence to a *fresh* env, matching each action by its `bits` +
`description` against the freshly built display list (robust to the combat
display-index dedup described in `src/sts_ai/CLAUDE.md`), then `env.step`.

No agent/model call happens during replay: we re-execute the exact recorded
actions, so replay is pure simulator stepping. See `tests/integration/
test_interactive_replay.py` for the determinism check this design relies on.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class ReplayError(RuntimeError):
    """A recorded action could not be re-resolved against the live display list.

    Indicates the replayed state diverged from the one the action was recorded in
    (a determinism break, a serializer change, or a corrupt/edited record)."""


def resolve_action_index(env: Any, bits: int | None, description: str) -> int:
    """Find the display index of the action matching (bits, description) in the
    env's current legal-action list.

    Match precedence:
      1. exact (bits, description) — the normal, unambiguous case;
      2. description-only — tolerates a `bits` representation drift while the
         human-readable action is unchanged (the dedup key in combat is the
         description, so it is unique in the display list).
    Raises ReplayError if there is no match or an ambiguous description-only one.
    """
    legal = env.legal_actions()
    if bits is not None:
        exact = [a for a in legal if int(a.bits) == int(bits) and a.description == description]
        if len(exact) == 1:
            return exact[0].index
    by_desc = [a for a in legal if a.description == description]
    if len(by_desc) == 1:
        return by_desc[0].index
    available = ", ".join(f"[{a.index}] bits={a.bits} {a.description!r}" for a in legal)
    raise ReplayError(
        f"cannot re-resolve recorded action bits={bits} description={description!r}; "
        f"{len(by_desc)} description matches among legal actions: {available or '(none)'}"
    )


def replay_actions(env: Any, actions: Sequence[Mapping[str, Any]]) -> int:
    """Re-apply a recorded action sequence to ``env``, advancing it to the
    frontier after the last action. ``actions`` items are `selected_action`
    dicts (`{"index", "bits", "description"}`). Returns the number of actions
    applied. Stops by raising ReplayError on the first action it cannot resolve
    (e.g. the env reached a terminal/divergent state early)."""
    applied = 0
    for action in actions:
        env.advance_to_decision()
        if env.is_terminal():
            raise ReplayError(
                f"env reached a terminal state after {applied} of {len(actions)} replayed "
                f"actions; cannot apply {action.get('description')!r}"
            )
        idx = resolve_action_index(env, action.get("bits"), str(action.get("description", "")))
        env.step(idx)
        applied += 1
    env.advance_to_decision()
    return applied
