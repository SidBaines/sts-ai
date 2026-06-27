"""Framing + prompt-template management for the Interactive Studio.

Two editable surfaces:

* **Framing** — the study's independent variable, a short instruction prepended
  to the prompt (default ``NEUTRAL_FRAME``). Editing framing alone keeps the
  prompt byte-identical to the harness (`render_action_prompt`), so framing-only
  Studio rollouts are directly comparable to batch rollouts.
* **Prompt template** — the *whole* prompt layout, exposed for free-form
  experimentation. ``DEFAULT_PROMPT_TEMPLATE`` reproduces ``render_action_prompt``
  exactly (locked by a parity test); editing it may diverge from the training/
  eval prompt (that's the point of the "advanced" editor).

`render_template` substitutes a fixed set of ``{placeholder}`` tokens in a single
pass (so literal braces in the JSON schema example or in ``state_text`` are left
untouched, and substituted values are never re-scanned).

`TemplateStore` persists named user templates as plain ``.txt`` files and always
surfaces the built-in defaults.
"""
from __future__ import annotations

import re
from pathlib import Path

from sts_ai.prompting import NEUTRAL_FRAME, render_action_prompt
from sts_ai.schemas import LegalAction

__all__ = [
    "NEUTRAL_FRAME",
    "DEFAULT_PROMPT_TEMPLATE",
    "BUILTIN_FRAMINGS",
    "BUILTIN_PROMPT_TEMPLATES",
    "reasoning_instruction_text",
    "compute_template_fields",
    "render_template",
    "TemplateStore",
]

# Mirrors prompting.render_action_prompt's induce_reasoning branch verbatim; the
# parity test (test_interactive_templates) fails if prompting.py drifts from it.
_REASONING_INSTRUCTION = (
    "Before the JSON, think step by step inside a single <think>...</think> "
    "block. Put the final JSON object after the closing </think>. Do not use "
    "markdown fences.\n\n"
)

# Placeholder string that reproduces render_action_prompt's output byte-for-byte
# once `compute_template_fields` supplies valid_indices / action_lines /
# reasoning_instruction. Shown as the starting point in the advanced editor.
DEFAULT_PROMPT_TEMPLATE = (
    "{framing}\n\n"
    "Return exactly one JSON object with this schema:\n"
    '{"reasoning": "brief private reasoning", "action_index": 0}\n\n'
    "Valid action_index values are: {valid_indices}. Use only these LEGAL ACTIONS indices; "
    "do not use hand, enemy, deck, or map indices as action_index.\n\n"
    "{reasoning_instruction}"
    "GAME STATE\n{state_text}\n\n"
    "LEGAL ACTIONS\n{action_lines}\n"
)

BUILTIN_FRAMINGS: dict[str, str] = {"neutral": NEUTRAL_FRAME}
BUILTIN_PROMPT_TEMPLATES: dict[str, str] = {"default": DEFAULT_PROMPT_TEMPLATE}

_PLACEHOLDER_RE = re.compile(
    r"\{(framing|valid_indices|reasoning_instruction|state_text|action_lines)\}"
)
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")


def reasoning_instruction_text(induce_reasoning: bool) -> str:
    """The <think>-block instruction (when induce_reasoning) or ``""``."""
    return _REASONING_INSTRUCTION if induce_reasoning else ""


def compute_template_fields(
    legal_actions: list[LegalAction], *, induce_reasoning: bool = False
) -> dict[str, str]:
    """The derived (non-framing) placeholder values for a decision, matching
    render_action_prompt's construction of valid_indices/action_lines."""
    return {
        "valid_indices": ", ".join(str(a.index) for a in legal_actions),
        "action_lines": "\n".join(f"{a.index}: {a.description}" for a in legal_actions),
        "reasoning_instruction": reasoning_instruction_text(induce_reasoning),
    }


def render_template(
    template: str,
    *,
    framing: str,
    state_text: str,
    legal_actions: list[LegalAction],
    induce_reasoning: bool = False,
) -> str:
    """Fill the placeholder tokens in ``template`` in one pass. Unknown ``{...}``
    and literal braces (e.g. the JSON schema example) are left as-is."""
    subs = {"framing": framing, "state_text": state_text}
    subs.update(compute_template_fields(legal_actions, induce_reasoning=induce_reasoning))
    return _PLACEHOLDER_RE.sub(lambda m: subs[m.group(1)], template)


def _safe_name(name: str) -> str:
    name = name.strip()
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"invalid template name {name!r}: use letters, digits, space, dash, underscore (max 64)"
        )
    return name


class TemplateStore:
    """Named framings + prompt templates on disk, merged with built-in defaults.

    Layout under ``base_dir`` (default ``data/interactive/templates``):
        framings/<name>.txt
        prompt_templates/<name>.txt
    Built-ins are always present in list/get; a user file of the same name
    overrides the built-in. Deleting a user file lets the built-in reappear.
    """

    def __init__(self, base_dir: str | Path = "data/interactive/templates") -> None:
        self.base_dir = Path(base_dir)
        self._framings_dir = self.base_dir / "framings"
        self._prompts_dir = self.base_dir / "prompt_templates"

    # --- framings ---------------------------------------------------------
    def list_framings(self) -> dict[str, str]:
        return self._list(self._framings_dir, BUILTIN_FRAMINGS)

    def get_framing(self, name: str) -> str:
        return self._get(self._framings_dir, BUILTIN_FRAMINGS, name, kind="framing")

    def save_framing(self, name: str, text: str) -> None:
        self._save(self._framings_dir, name, text)

    def delete_framing(self, name: str) -> bool:
        return self._delete(self._framings_dir, name)

    # --- prompt templates -------------------------------------------------
    def list_prompt_templates(self) -> dict[str, str]:
        return self._list(self._prompts_dir, BUILTIN_PROMPT_TEMPLATES)

    def get_prompt_template(self, name: str) -> str:
        return self._get(self._prompts_dir, BUILTIN_PROMPT_TEMPLATES, name, kind="prompt template")

    def save_prompt_template(self, name: str, text: str) -> None:
        self._save(self._prompts_dir, name, text)

    def delete_prompt_template(self, name: str) -> bool:
        return self._delete(self._prompts_dir, name)

    # --- shared -----------------------------------------------------------
    @staticmethod
    def _list(directory: Path, builtins: dict[str, str]) -> dict[str, str]:
        merged = dict(builtins)
        if directory.is_dir():
            for path in sorted(directory.glob("*.txt")):
                merged[path.stem] = path.read_text(encoding="utf-8")
        return merged

    def _get(self, directory: Path, builtins: dict[str, str], name: str, *, kind: str) -> str:
        name = _safe_name(name)
        path = directory / f"{name}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        if name in builtins:
            return builtins[name]
        raise KeyError(f"no {kind} named {name!r}")

    @staticmethod
    def _save(directory: Path, name: str, text: str) -> None:
        name = _safe_name(name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.txt").write_text(text, encoding="utf-8")

    @staticmethod
    def _delete(directory: Path, name: str) -> bool:
        name = _safe_name(name)
        path = directory / f"{name}.txt"
        if path.is_file():
            path.unlink()
            return True
        return False
