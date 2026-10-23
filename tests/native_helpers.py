from __future__ import annotations
from dataclasses import asdict


def prepare_decision(env):
    env.advance_to_decision()
    if env.is_terminal():
        return "terminal", None
    actions = env.legal_actions()
    if not actions:
        return "no_legal_actions", None
    from sts_env._backend.glossary import augment
    version = env.combat_observation
    if version == "combat_public_v4":
        from sts_env._backend.glossary_v4 import augment
    elif version == "combat_public_v5":
        from sts_env._backend.glossary_v5 import augment
    action_dicts = [asdict(action) for action in actions]
    text = augment(env.describe_state(), action_dicts, env.phase(), map_graph=env.map_graph(),
                   turn_math_lines=env._turn_math_lines() if version in ("combat_public_v3", "combat_public_v4", "combat_public_v5") else None)
    return "ok", {"phase": env.phase(), "state": env.summary(), "state_text": text,
                  "legal_actions": actions, "legal_action_dicts": action_dicts}
