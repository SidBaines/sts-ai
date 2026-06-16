from __future__ import annotations

from sts_ai.schemas import LegalAction


NEUTRAL_FRAME = (
    "You are playing Slay the Spire. Choose one legal action from the list. "
    "Use the game state and action descriptions to make the strongest choice you can."
)


def render_action_prompt(
    state_text: str,
    legal_actions: list[LegalAction],
    framing: str = NEUTRAL_FRAME,
    induce_reasoning: bool = False,
) -> str:
    action_lines = "\n".join(
        f"{action.index}: {action.description}" for action in legal_actions
    )
    valid_indices = ", ".join(str(action.index) for action in legal_actions)
    reasoning_instruction = (
        "Before the JSON, think step by step inside a single <think>...</think> "
        "block. Put the final JSON object after the closing </think>. Do not use "
        "markdown fences.\n\n"
        if induce_reasoning
        else ""
    )
    return (
        f"{framing}\n\n"
        "Return exactly one JSON object with this schema:\n"
        '{"reasoning": "brief private reasoning", "action_index": 0}\n\n'
        f"Valid action_index values are: {valid_indices}. Use only these LEGAL ACTIONS indices; "
        "do not use hand, enemy, deck, or map indices as action_index.\n\n"
        f"{reasoning_instruction}"
        f"GAME STATE\n{state_text}\n\n"
        f"LEGAL ACTIONS\n{action_lines}\n"
    )
