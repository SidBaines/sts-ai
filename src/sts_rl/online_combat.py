"""On-policy combat samples and phase-specific policy updates from whole games."""

from __future__ import annotations

from collections import defaultdict
import random

import torch

from sts_policy.encoding import collate_observations
from sts_policy.losses import PPOConfig, masked_kl_divergence
from sts_policy.model_agent import collate_public, is_act3_boss_combat
from sts_policy.rollout_training import update_ppo
from .common import phase_forward
from .fresh_statistics import paired_win_normal_interval


ROLES = ("combat", "boss_specialist")


def samples_from_game(episode, ordinary_limit=32, boss_limit=16, *, specialist=True):
    """Uniformly sample decisions within each role; retain complete-game credit."""
    result = {role: [] for role in ROLES}
    if not episode or not episode[-1].terminated or episode[-1].truncated:
        return result
    rows = defaultdict(list)
    remaining = 0.0
    for t in reversed(episode):
        remaining += t.reward
        obs = t.observation
        if obs["phase"] == "combat" and len(obs["legal_actions"]) > 1:
            role = (
                "boss_specialist"
                if specialist and is_act3_boss_combat(obs)
                else "combat"
            )
            rows[role].append((t, remaining))
    rng = random.Random(episode[-1].world_seed)
    for role, limit in zip(ROLES, (ordinary_limit, boss_limit)):
        choices = rows[role]
        chosen = sorted(rng.sample(range(len(choices)), min(limit, len(choices))))
        for index in chosen:
            t, target = choices[index]
            state = t.observation["state"]
            battle = state.get("combat") or {}
            hp = battle.get("player_cur_hp", state.get("cur_hp", 0))
            max_hp = battle.get("player_max_hp", state.get("max_hp", 1))
            t.observation = {
                "phase": "combat",
                "state": {
                    "act": state.get("act"),
                    "room": str(state.get("room", "")),
                    "hp_quarter": min(3, int(4 * (hp or 0) / max(1, max_hp or 1))),
                },
                "legal_actions": [
                    {"index": a["index"]} for a in t.observation["legal_actions"]
                ],
            }
            if t.encoded_observation is None:
                raise ValueError("online combat requires sampling-time encodings")
            t.encoding_observation_id = id(t.observation)
            t.returns = target
            result[role].append(t)
    return result


def center_combat_returns(rows):
    """A world-held-out baseline for a return; rows from the same game stay together."""
    buckets = defaultdict(lambda: [0.0, 0])

    def keys(t):
        s = t.observation["state"]
        return (
            (),
            (s["act"],),
            (s["act"], s["room"]),
            (s["act"], s["room"], s["hp_quarter"]),
        )

    for t in rows:
        for key in keys(t):
            b = buckets[t.world_seed % 2, key]
            b[0] += t.returns
            b[1] += 1
    for t in rows:
        expected = 0.0
        for key in reversed(keys(t)):
            total, count = buckets[1 - t.world_seed % 2, key]
            if count >= 8:
                expected = total / count
                break
        t.advantage = t.returns - expected
        t.encoding_observation_id = id(t.observation)


def combat_optimizers(model, rate):
    optimizers = {}
    for role in ROLES:
        head = getattr(model, role, None)
        if head is None:
            continue
        for name, p in head.named_parameters():
            p.requires_grad_(not name.startswith("critic."))
        optimizers[role] = torch.optim.AdamW(
            (p for p in head.parameters() if p.requires_grad), lr=rate, weight_decay=0.0
        )
    return optimizers


def update_fighters(
    model, rows, optimizers, temperatures, *, rate, epochs, device, seed, minibatch=1024
):
    """Independent steps avoid weighting the boss by its much rarer occurrence."""
    from .ooc_trainer import _move_optimizer_state

    record = {}
    for role, opt in optimizers.items():
        batch = rows[role]
        if not batch:
            raise ValueError(f"no completed-game combat samples for {role}")
        center_combat_returns(batch)
        _move_optimizer_state(opt, device)
        head = getattr(model, role)
        try:
            cfg = PPOConfig(
                learning_rate=rate,
                epochs=epochs,
                minibatch_size=minibatch,
                value_weight=0.0,
                entropy_weight=0.0,
                clip_ratio=0.2,
                target_kl=0.025,
                kl_guard_before_update=True,
            )
            metrics = update_ppo(
                head,
                batch,
                collate=collate_public,
                forward=phase_forward(temperature=temperatures[role]),
                config=cfg,
                device=device,
                seed=seed,
                optimizer=opt,
                micro_batch=256,
            )
        finally:
            head.cpu()
            _move_optimizer_state(opt, "cpu")
            opt.zero_grad(set_to_none=True)
        record[role] = {
            k: metrics.get(k)
            for k in (
                "steps",
                "samples",
                "early_stop",
                "approx_kl",
                "clip_fraction",
                "policy_loss",
            )
        }
    return record


@torch.no_grad()
def action_diagnostics(head, rows, temperatures, candidate=None):
    """Equal top scores count as the same preference; report actual and expected exploration."""
    total = actual = 0
    expected = [0.0] * len(temperatures)
    changed = divergence = old_probability_error = 0.0
    if not rows:
        raise ValueError("cannot measure exploration without decisions")
    for i in range(0, len(rows), 256):
        part = rows[i : i + 256]
        x, a, mask = collate_observations(
            [t.encoded_observation for t in part], device="cpu"
        )
        logits = head.actor_logits(x, a, mask).masked_fill(~mask, -1e9)
        lower = (logits < logits.max(-1, keepdim=True).values - 1e-5) & mask
        meaningful = lower.any(-1)
        picked = torch.tensor([t.action_position for t in part])
        actual += int(lower.gather(1, picked[:, None]).sum())
        total += int(meaningful.sum())
        for j, temp in enumerate(temperatures):
            scaled = (logits / temp).masked_fill(~mask, -1e9)
            expected[j] += float((scaled.softmax(-1) * lower).sum())
        if candidate is not None:
            other = candidate.actor_logits(x, a, mask).masked_fill(~mask, -1e9)
            changed += float(lower.gather(1, other.argmax(-1)[:, None]).sum())
            divergence += float(
                masked_kl_divergence(
                    logits / temperatures[0], other / temperatures[0], mask
                ).sum()
            )
        if candidate is None and len(temperatures) == 1:
            observed = torch.tensor([t.behavior_log_prob for t in part])
            reconstructed = scaled.log_softmax(-1).gather(1, picked[:, None]).flatten()
            old_probability_error = max(
                old_probability_error, float((reconstructed - observed).abs().max())
            )
    return {
        "rows": len(rows),
        "choices": total,
        "actual": actual,
        "expected_rates": [v / max(1, total) for v in expected],
        "actual_rate": actual / max(1, total),
        "changed_choices": int(changed),
        "changed_rate": changed / max(1, total),
        "mean_kl": divergence / len(rows),
        "max_log_prob_error": old_probability_error
        if candidate is None and len(temperatures) == 1
        else None,
    }


def select_exploration(
    baseline, conditions, preference, *, max_win_drop=0.05, max_reach_drop=0.08
):
    """Apply both noninferiority screens to the least favourable missing outcomes."""
    n = len(baseline)
    if not n:
        raise ValueError("empty exploration panel")
    summary = {}
    for name, worlds in conditions.items():
        if worlds.keys() != baseline.keys():
            raise ValueError("exploration arms must play exactly the same worlds")
        result = {}
        for key in ("win", "act3"):
            lo = hi = up = down = valid = 0
            for seed, a in baseline.items():
                x, y = a[key], worlds[seed][key]
                lo += int(y is True) - int(x is not False)
                hi += int(y is not False) - int(x is True)
                if x is not None and y is not None:
                    valid += 1
                    up += int(y and not x)
                    down += int(x and not y)
            result[key] = {
                "bounds": [lo / n, hi / n],
                "complete": valid,
                "gains": up,
                "losses": down,
                "ci95": list(paired_win_normal_interval(valid, up, down))
                if valid > 1
                else None,
            }
        summary[name] = result
    choice = next(
        (
            name
            for name in preference
            if name in summary
            and summary[name]["win"]["bounds"][0] >= -max_win_drop
            and summary[name]["act3"]["bounds"][0] >= -max_reach_drop
        ),
        None,
    )
    return choice, summary


def viable_combat_fit(movement, *, min_changed=0.002, max_kl=0.05):
    return all(
        movement[r]["changed_rate"] >= min_changed and movement[r]["mean_kl"] <= max_kl
        for r in ROLES
    )


def nominate_combat_challenger(results, reference):
    """Choose a distinct eligible fit by worst-case paired gain, with stated tie breaks."""
    old_target = results[reference]["setting"]["target"]
    viable = [
        key
        for key, result in results.items()
        if key != reference and result["eligible"]
    ]
    if not viable:
        return None

    def order(key):
        result = results[key]
        setting = result["setting"]
        return (
            -result["comparison"]["full_cohort_gain_bounds_pp"][0],
            setting["epochs"],
            setting["rate"],
            abs(setting["target"] - old_target),
            setting["target"],
        )

    return min(viable, key=order)


def confirmed_combat_challenger(screen, independent):
    """Incomplete games may count against a change in both independent cohorts."""
    return (
        screen["full_cohort_gain_bounds_pp"][0] > 0
        and independent["worst_case_ci95_pp"][0] > 0
    )
