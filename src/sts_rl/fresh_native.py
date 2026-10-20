"""Research-child native exception boundaries; Python defects still propagate."""
from functools import wraps
from contextlib import contextmanager

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


_GUARDED_METHODS = {
    "GameContext": ("__init__", "copy", "legal_actions", "describe_state", "keys", "map_graph", "public_event"),
    "BattleContext": ("__init__", "copy", "init", "exit_battle", "legal_actions", "describe_state",
                      "public_cards", "public_player", "public_relics", "public_potions", "enemies",
                      "reseed_rngs", "search_best_action"),
    "GameAction": ("execute", "describe"),
    "BattleAction": ("execute", "describe"),
    "Agent": ("__init__", "seed", "select_out_of_combat_action", "preview_out_of_combat_action"),
}


def install_native_guards(sts):
    """Wrap explicit native boundaries, retaining the real native object types.

    Disposable children install once. Direct runners use native_guard_scope so
    callers outside a run retain the native module's original exception API.
    """
    if getattr(sts, "_fresh_native_guards", False):
        return
    for name, names in _GUARDED_METHODS.items():
        cls = getattr(sts, name)
        for method in names:
            original = getattr(cls, method)

            def guarded(*args, _original=original, _operation=f"{name}.{method}", **kwargs):
                return native_call(_operation, _original, *args, **kwargs)

            setattr(cls, method, wraps(original)(guarded))
    sts._fresh_native_guards = True


@contextmanager
def native_guard_scope(sts):
    """Temporarily install guards when sharing an interpreter with other code."""
    if getattr(sts, "_fresh_native_guards", False):
        yield
        return
    originals = {(getattr(sts, cls), name): getattr(getattr(sts, cls), name)
                 for cls, names in _GUARDED_METHODS.items() for name in names}
    had_flag = hasattr(sts, "_fresh_native_guards")
    previous_flag = getattr(sts, "_fresh_native_guards", None)
    try:
        install_native_guards(sts)
        yield
    finally:
        for (cls, name), original in originals.items():
            setattr(cls, name, original)
        if had_flag:
            sts._fresh_native_guards = previous_flag
        elif hasattr(sts, "_fresh_native_guards"):
            del sts._fresh_native_guards
