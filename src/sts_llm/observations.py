"""One versioned prompt shared by saved-state training and full-game agents."""
from __future__ import annotations
import hashlib
import json

PROMPT_VERSION = 'public_text_menu_v1'
SYSTEM_MESSAGE = ('You play Slay the Spire. Choose one legal action from current menu. '
                  'Reason as needed, then output only final JSON {"action_index": integer}.')


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


def observation_identity(observation: dict) -> str:
    """Identity of the public input, excluding any unrequested adapter fields."""
    data = {'phase': observation.get('phase'), 'text': observation['text'],
            'legal_actions': observation['legal_actions']}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()
