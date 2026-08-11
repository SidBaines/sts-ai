from __future__ import annotations

from sts_ai.schemas import LegalAction


NEUTRAL_FRAME = (
    "You are playing Slay the Spire. Choose one legal action from the list. "
    "Use the game state and action descriptions to make the strongest choice you can."
)

REASONING_ACTION_OUTPUT = "reasoning_action"
ACTION_ONLY_OUTPUT = "action_only"
ACTION_TEXT_OUTPUT = "action_text"
TURN_PLAN_OUTPUT = "turn_plan"
OUTPUT_CONTRACTS = (
    REASONING_ACTION_OUTPUT,
    ACTION_ONLY_OUTPUT,
    ACTION_TEXT_OUTPUT,
    TURN_PLAN_OUTPUT,
)

ACTION_TEXT_INSTRUCTION = (
    "Return exactly one JSON object with this schema:\n"
    '{"action": "<the exact text of one legal action>"}\n'
    "Copy the action text exactly as it appears in LEGAL ACTIONS."
)
TURN_PLAN_INSTRUCTION = (
    "Return exactly one JSON object with this schema:\n"
    '{"plan": ["<action text>", "..."], "action": "<the first entry of plan>"}\n'
    '"plan" lists, in order, the exact texts of the actions you intend to take '
    'this turn (end it with "end turn"). "action" repeats the first entry.\n'
    "Copy action texts exactly as they appear in LEGAL ACTIONS."
)


RETRY_INSTRUCTION = (
    "\n\nYour previous response was invalid. Return only one JSON object "
    "with a legal integer action_index from the listed actions. Do not include "
    "a <think> block, markdown fence, or any other text."
)
ACTION_TEXT_RETRY_INSTRUCTION = (
    "\n\nYour previous response was invalid. Return only one JSON object of the "
    'form {"action": "<the exact text of one legal action>"}, with the action '
    "text copied exactly from LEGAL ACTIONS. Do not include a <think> block, "
    "markdown fence, or any other text."
)
TURN_PLAN_RETRY_INSTRUCTION = (
    "\n\nYour previous response was invalid. Return only one JSON object of the "
    'form {"plan": ["<action text>", "..."], "action": "<the first entry of '
    'plan>"}, with every action text copied exactly from LEGAL ACTIONS. Do not '
    "include a <think> block, markdown fence, or any other text."
)


def validate_output_contract(output_contract: str) -> None:
    if output_contract not in OUTPUT_CONTRACTS:
        raise ValueError(
            "output_contract must be one of "
            + ", ".join(repr(value) for value in OUTPUT_CONTRACTS)
        )


def retry_instruction(output_contract: str = REASONING_ACTION_OUTPUT) -> str:
    """Contract-appropriate invalid-response repair suffix.

    The default literal is the frozen historical retry prompt and must stay
    byte-identical for the index contracts; the semantic contracts get schema-
    matched wording instead of the misleading ``action_index`` phrasing.
    """
    validate_output_contract(output_contract)
    if output_contract == ACTION_TEXT_OUTPUT:
        return ACTION_TEXT_RETRY_INSTRUCTION
    if output_contract == TURN_PLAN_OUTPUT:
        return TURN_PLAN_RETRY_INSTRUCTION
    return RETRY_INSTRUCTION


def render_action_prompt(
    state_text: str,
    legal_actions: list[LegalAction],
    framing: str = NEUTRAL_FRAME,
    induce_reasoning: bool = False,
    output_contract: str = REASONING_ACTION_OUTPUT,
) -> str:
    validate_output_contract(output_contract)
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
    # Keep the historical default literal in its own branch: this function is a
    # frozen policy interface, so opting into the compact action-only contract
    # must not perturb existing prompts by even one byte.
    if output_contract in (REASONING_ACTION_OUTPUT, ACTION_ONLY_OUTPUT):
        output_schema = (
            '{"reasoning": "brief private reasoning", "action_index": 0}'
            if output_contract == REASONING_ACTION_OUTPUT
            else '{"action_index": 0}'
        )
        output_instruction = (
            "Return exactly one JSON object with this schema:\n"
            f"{output_schema}"
        )
        choice_instruction = (
            f"Valid action_index values are: {valid_indices}. Use only these LEGAL ACTIONS indices; "
            "do not use hand, enemy, deck, or map indices as action_index."
        )
    else:
        output_instruction = (
            ACTION_TEXT_INSTRUCTION
            if output_contract == ACTION_TEXT_OUTPUT
            else TURN_PLAN_INSTRUCTION
        )
        choice_instruction = "Choose from the LEGAL ACTIONS list below."
    return (
        f"{framing}\n\n"
        f"{output_instruction}\n\n"
        f"{choice_instruction}\n\n"
        f"{reasoning_instruction}"
        f"GAME STATE\n{state_text}\n\n"
        f"LEGAL ACTIONS\n{action_lines}\n"
    )
