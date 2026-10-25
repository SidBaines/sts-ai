"""Versioned prompts shared by saved-state training and full-game agents.

Version 1 lists the menu by native display index. Version 2 (``labelled_prompt``)
gives each action a letter label, so menu choices cannot be confused with the
``[0]``-style indices of hand cards, enemies and potions in the state text, and
presents the menu in a caller-chosen order, so a fixed menu position cannot be
learned as a shortcut. The state text is identical in both versions.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import string
from typing import Sequence

PROMPT_VERSION = 'public_text_menu_v1'
SYSTEM_MESSAGE = ('You play Slay the Spire. Choose one legal action from current menu. '
                  'Reason as needed, then output only final JSON {"action_index": integer}.')
LABELLED_PROMPT_VERSION = 'public_text_labelled_menu_v2'
LABELLED_SYSTEM_MESSAGE = ('You play Slay the Spire as the Ironclad. You are shown the current game state '
                           'and a menu of legal actions, each with a letter label. Choose exactly one action. '
                           'Reason as needed, then output only final JSON {"action": "<label>"}, using the '
                           "chosen action's label.")


def legal_indices(observation: dict) -> list[int]:
    actions = observation.get('legal_actions')
    if not isinstance(actions, list) or not actions:
        raise ValueError('A public observation needs a nonempty current legal menu')
    indices = [a.get('index') for a in actions]
    if any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('Legal display indices must be unique integers')
    if any(not isinstance(a.get('description'), str) for a in actions):
        raise ValueError('Every legal action needs its public description')
    return indices


def prompt(observation: dict) -> list[dict[str, str]]:
    """Only public text and the raw current menu enter model messages.

    Deliberately do not serialize the observation dictionary: provenance, teacher
    labels, numeric feature adapters and accidentally supplied info are ignored.
    The environment owns text/glossary coverage. No hidden facts are reconstructed.
    """
    legal_indices(observation)
    text = observation.get('text')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('A public observation needs its complete rendered text')
    menu = '\n'.join(f"[{a['index']}] {a['description']}" for a in observation['legal_actions'])
    return [{'role': 'system', 'content': SYSTEM_MESSAGE}, {'role': 'user',
             'content': text + '\n\nLegal actions (choose exactly one current action index):\n' + menu}]


def menu_labels(count: int) -> list[str]:
    """A, B, ..., Z, AA, AB, ...: distinct and never numeric."""
    if type(count) is not int or count < 1:
        raise ValueError('A menu needs at least one action')
    labels = []
    for number in range(count):
        label, number = '', number + 1
        while number:
            number, remainder = divmod(number - 1, 26)
            label = string.ascii_uppercase[remainder] + label
        labels.append(label)
    return labels


@dataclass(frozen=True)
class LabelledPrompt:
    messages: list[dict[str, str]]
    labels: dict[str, int]
    order: list[int]


def labelled_prompt(observation: dict, order: Sequence[int] | None = None) -> LabelledPrompt:
    """Version-2 prompt: the public state text, then the menu in ``order``.

    ``order`` is a permutation of the native display indices (default: native
    order). Labels follow display position, so ``labels`` maps each label to
    the native index it presents; it never enters the messages.
    """
    indices = legal_indices(observation)
    text = observation.get('text')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('A public observation needs its complete rendered text')
    order = list(indices if order is None else order)
    if sorted(order) != sorted(indices) or len(order) != len(indices):
        raise ValueError('Menu order must be a permutation of the legal display indices')
    descriptions = {a['index']: a['description'] for a in observation['legal_actions']}
    labels = menu_labels(len(order))
    menu = '\n'.join(f'{label}) {descriptions[index]}' for label, index in zip(labels, order))
    return LabelledPrompt(
        messages=[{'role': 'system', 'content': LABELLED_SYSTEM_MESSAGE}, {'role': 'user',
                  'content': text + '\n\nActions (choose exactly one; answer with its label):\n' + menu}],
        labels=dict(zip(labels, order)), order=order)


def observation_identity(observation: dict) -> str:
    """Identity of the public input, excluding any unrequested adapter fields."""
    data = {'phase': observation.get('phase'), 'text': observation['text'],
            'legal_actions': observation['legal_actions']}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()
