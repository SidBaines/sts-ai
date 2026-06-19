# Reward Specification & Anti-Hacking

**Status:** working spec for the offline PG (signed-advantage) and on-policy GRPO arms.
**Audience:** anyone training a policy on this harness or reading its eval.
See [`rl_and_framing_design.md`](rl_and_framing_design.md) (§"Reward design: keep it
trait-neutral", §gotcha 1) and [`research_plan.md`](research_plan.md) for the surrounding
commitments. This is deliverable 5 of [`grpo_readiness_handoff.md`](grpo_readiness_handoff.md).

## 1. The reward

**Primary reward = `final_floor`** (the deepest floor the run reached), read per trajectory
from the `*.meta.json` sidecar. It is a dense-enough, monotone competence signal: deeper =
played better, and it discriminates policies even when nobody wins (the current regime —
the model rarely clears Act 1). `outcome` (VICTORY) and `final_act` are recorded alongside
and define the binary `is_act_boss_clear` milestone (`reward.is_act_boss_clear`), but the
*scalar* the optimizer sees is the floor.

How the floor enters training:
- **Offline signed-advantage PG (deliverable 3):** `A = final_floor − baseline` (median over
  the run pool), broadcast to every decision in the trajectory (`advantage.offline_advantages`).
- **On-policy GRPO (deliverable 4):** `A = (final_floor − group_mean) / (group_std + ε)`,
  the group = K rollouts of one world seed, broadcast to the trajectory's decisions
  (`advantage.group_relative_advantages`). The group mean is the baseline that tames the
  sparse terminal-reward variance **without a value network**.

We do **not** broadcast a per-decision shaped reward. Credit assignment is the scalar
advantage attached to each decision by the training loop, not something the (Markovian,
fresh-context-per-turn) policy reasons about — see `rl_and_framing_design.md` §1.

## 2. The framing invariant: the reward MUST stay trait-neutral

The experiment asks *which latent trait absorbs the same competence update under different
framings*. **If the reward itself rewarded risk** (e.g. bonus for elite paths, low-HP play,
high-variance cards), the manipulation would be confounded — a trait shift could no longer
be attributed to the frame rather than to the reward. So:

- Reward stays **pure competence** (floor; optionally VICTORY/`final_act` milestones; at most
  light, trait-neutral shaping like combats-won). **No risk/adventure/caution term, ever.**
- Variance is handled with a **baseline** (GRPO group mean), never with risk-laden reward terms.
- This mirrors the prompt-neutrality invariant (`prompt-neutrality` memory; CLAUDE.md): the
  base prompt is comprehension-only and the reward is competence-only; framing is the *only*
  thing that varies between conditions, so the data + reward stay reusable across frames.

Tactical hints (`hinting.py`) obey the same rule — they correct only **tactical-truth**
mistakes (lethal-not-taken, full-block-not-taken-under-heavy-incoming), never risk/strategy.

## 3. Failure modes an on-policy optimizer will find (and the guardrail for each)

A sparse terminal reward over a hundreds-of-decisions episode invites reward hacking. Each
mode below is paired with the metric that surfaces it (so the eval can *see* the hack) and a
guardrail. The metrics are reported by `compare_paired.py` (per-arm, meta-level) and
`eval_metrics.py` (decision-level, from the JSONL traces).

| Failure mode | Why the optimizer likes it | Surfaced by | Guardrail |
| --- | --- | --- | --- |
| **Stalling to delay death** — dragging the episode out to avoid the terminal loss without progressing the floor. | Postpones the negative signal. | `mean_decisions`, `budget_truncated_rate` (meta); `longest_repeat_streak` (decisions). | Floor (not survival-time) is the reward; `--max-decisions` caps the episode and `budget_truncated` is flagged so a truncation is never read as a real ending. |
| **Degenerate loops / sim quirks** — repeating an identical (state, action) the engine permits, exploiting a simulator edge. | Free reward or indefinite survival. | `repeated_consecutive`, `longest_repeat_streak`, `distinct_state_fraction` (`eval_metrics.stall_metrics`). | Watch these per-iteration; a spike is the early warning. Fail-closed simulator policy already rejects illegal actions. |
| **Length blowup** — ever-longer reasoning to (spuriously) raise completion likelihood. | Token-level loss can reward verbosity. | `avg_completion_tokens` (meta/compare). | KL-to-frozen-reference penalty in the PG loss; watch completion length per iteration. |
| **Invalid-as-loss conflation** — an unrecoverable invalid decision *ends* the episode (`stopped_reason=agent_invalid`), so a truncated run scores as a low-floor "loss" for FORMAT reasons, not policy reasons. | Not chosen by the optimizer, but it pollutes the reward signal and the thinking-vs-no-thinking comparison. | `agent_invalid_rate` reported **beside** every outcome metric (compare_paired). | **Always read outcome next to `agent_invalid`.** Mask or apply an explicit format penalty to the terminal invalid record rather than scoring it as a plain low-floor loss (`rl_and_framing_design.md` §gotcha 1, 5). |
| **Hybrid-combat confound** — in `combat_control="search"` the floor partly reflects the *search agent's* competence, masking bad out-of-combat choices. | Not a hack, but dilutes credit on the risk-relevant OOC decisions. | n/a (design choice). | Start RL on the hybrid OOC action space deliberately (the risk-relevant decisions); move to `combat_control="llm"` when full-trajectory credit is wanted (`rl_and_framing_design.md` §gotcha 3). |

## 4. Decision

Start both PG arms on **`final_floor` with no shaping**, variance handled by the baseline
(offline median / GRPO group mean), terminal-invalid records masked or format-penalized, and
**every outcome reported beside `agent_invalid_rate`** plus the stall/length metrics above.
Revisit shaping only if the eval shows the floor signal is too sparse to move the policy —
and even then, only with trait-neutral terms. This keeps the data and reward reusable across
the later framing conditions.
