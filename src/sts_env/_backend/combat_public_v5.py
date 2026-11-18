"""V5 public presentation over the unchanged v3 simulator/replay interface."""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from sts_env._copies import copy_tree
from sts_env._backend.combat_public_v4 import CombatPublicV4Env
from sts_env._backend.schemas import LegalAction


_SELECTION = re.compile(r"^\{ (EXHAUST_MANY|GAMBLE)\b")


def selection_label(action: LegalAction, hand: list[dict[str, Any]]) -> str:
    match = _SELECTION.match(action.description)
    if not match:
        return action.description
    task = match.group(1)
    # Action::getSelectedIdxs uses exactly the low ten bits. Never reinterpret
    # the native printed indices (its nonempty-mask formatter prints them wrong).
    indices = [i for i in range(10) if action.bits & (1 << i)]
    if not indices:
        return ("exhaust no cards" if task == "EXHAUST_MANY"
                else "keep current hand (discard no cards)")
    by_index = {c["index"]: c["label"] for c in hand}
    chosen = "; ".join(f"hand [{i}] {by_index[i]}" for i in indices)
    verb = "exhaust" if task == "EXHAUST_MANY" else "discard and redraw"
    return f"{verb}: {chosen}"


class CombatPublicV5Env(CombatPublicV4Env):
    combat_observation = "combat_public_v5"

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._selection_labels: dict[str, str] = {}
        self._raw_history: list[str] = []
        self._public_history: list[str] = []

    def legal_actions(self) -> list[LegalAction]:
        actions = self._env.legal_actions()
        if self._env.phase() != "combat":
            return actions
        result = list(actions)
        hand = None
        for i, action in enumerate(actions):
            if _SELECTION.match(action.description):
                if hand is None:
                    hand = self._env.public_combat_cards().get("hand", [])
                label = selection_label(action, hand)
                self._selection_labels[action.description] = label
                result[i] = replace(action, description=label)
        return result

    def _history_context(self) -> tuple[int | None, list[str]]:
        if getattr(type(self._env), "combat_history_context", None) is not None:
            return self._env.combat_history_context()
        # Older v3 backends expose the same fields through their full summary.
        state = self._env.summary()
        return state.get("ascension"), list(state.get("combat", {}).get("recent_actions", []))

    def _label(self, text: str) -> str:
        return self._selection_labels.get(text, {
            "{ EXHAUST_MANY none": "exhaust no cards",
            "{ GAMBLE none": "keep current hand (discard no cards)",
        }.get(text, text))

    def step(self, action_index: int) -> LegalAction:
        chosen = self.legal_actions()[action_index]
        _, before = self._history_context()
        public_before = self._history(before)
        executed = self._env.step(action_index)
        if (executed.index, executed.bits) != (chosen.index, chosen.bits):
            raise RuntimeError("v5 selection identity changed during execution")
        # Freeze the visible card names at execution. A later draw can reuse the
        # same native mask text for different cards; translating historical raw
        # masks through the current menu would rewrite the player's past action.
        # Store only wrapper-owned history; the underlying frozen v3 view must
        # remain byte-for-byte unchanged. The backend clears its list at turn or
        # battle boundaries, so append only when the old raw prefix survives.
        _, after = self._history_context()
        if after == [*before, executed.description]:
            self._raw_history = list(after)
            self._public_history = [*public_before, chosen.description]
        else:
            self._history(after)
        return chosen

    def _history(self, raw: list[str]) -> list[str]:
        if raw != self._raw_history:
            self._raw_history = list(raw)
            self._public_history = [self._label(action) for action in raw]
        return list(self._public_history)

    def _cards(self, cards: dict[str, Any]) -> dict[str, Any]:
        data = copy_tree(cards)
        self._label_selection(data)
        return data

    def _label_selection(self, cards: dict[str, Any]) -> None:
        selection = cards.get("selection")
        if selection:
            self.legal_actions()  # refresh labels for all currently legal masks
            selection["candidates"] = [self._label(x) for x in selection["candidates"]]

    def public_combat_cards(self) -> dict[str, Any]:
        return self._cards(self._env.public_combat_cards())

    def summary(self) -> dict[str, Any]:
        data = copy_tree(self._env.summary())
        data["combat_observation"] = self.combat_observation
        combat = data.get("combat")
        if combat:
            # Already owned by this summary; only standalone card views need a
            # separate copy before relabelling their selection candidates.
            self._label_selection(combat["cards"])
            combat["recent_actions"] = self._history(combat["recent_actions"])
        return data

    def describe_state(self) -> str:
        if self._env.phase() != "combat":
            return self._env.describe_state()
        self.legal_actions()
        text = self._env.describe_state()
        # Keep the numeric energy syntax readable by existing board parsers;
        # its denominator is explicitly defined by the v5 glossary.
        for old, new in sorted(self._selection_labels.items(), key=lambda x: -len(x[0])):
            if old != new:
                text = text.replace(old, new)
        ascension, raw_history = self._history_context()
        history = self._history(raw_history)
        text = re.sub(r"^Recent actions this turn:.*$",
                      lambda _: "Recent actions this turn: " + (" -> ".join(history) or "none"),
                      text, flags=re.MULTILINE)
        return text + f"\nAscension: {int(ascension)}"
