"""Research-child native exception boundaries; Python defects still propagate."""
from functools import wraps

from sts_env.env import SimulatorError


def native_call(operation, function, *args, **kwargs):
    """Convert only exceptions raised at a C binding, not Python callbacks.

    A binding exception has just this Python frame. A Python implementation or
    callback adds a frame, and remains a programming error. TypeError (bad API
    arguments) also remains a programming error. No inference code is wrapped.
    """
    try:
        return function(*args, **kwargs)
    except (RuntimeError, ValueError, IndexError, OverflowError) as exc:
        if exc.__traceback__.tb_next is not None:
            raise
        # Preserve the canonical teacher's explicit draw-order abstention.
        if operation == "BattleContext.search_best_action" and isinstance(exc, ValueError) and "draw_order_seed" in str(exc):
            raise
        raise SimulatorError(f"{operation}: {type(exc).__name__}: {exc}") from exc


def install_native_guards(sts):
    """Wrap the explicit binding calls used by reset/replay/step/search/observe.

    Installed once, only inside FreshPanelEpisode's disposable child. Objects
    retain their real native types, including GameContext.copy and init inputs.
    """
    if getattr(sts, "_fresh_native_guards", False):
        return
    methods = {
        "GameContext": ("__init__", "copy", "legal_actions", "describe_state", "keys", "map_graph", "public_event"),
        "BattleContext": ("__init__", "copy", "init", "exit_battle", "legal_actions", "describe_state",
                          "public_cards", "public_player", "public_relics", "public_potions", "enemies",
                          "reseed_rngs", "search_best_action"),
        "GameAction": ("execute", "describe"),
        "BattleAction": ("execute", "describe"),
        "Agent": ("__init__", "seed", "select_out_of_combat_action", "preview_out_of_combat_action"),
    }
    for name, names in methods.items():
        cls = getattr(sts, name)
        for method in names:
            original = getattr(cls, method)

            def guarded(*args, _original=original, _operation=f"{name}.{method}", **kwargs):
                return native_call(_operation, _original, *args, **kwargs)

            setattr(cls, method, wraps(original)(guarded))
    sts._fresh_native_guards = True
