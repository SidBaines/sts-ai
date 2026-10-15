# Next-study options — 15 October 2026

This review of the non-LLM workstream was written after the joint sparse-win run.
It is an analysis for Sid to choose from, not a launch instruction: **no training
is authorized until Sid picks a study and its budget.** Read it after the
[15 October handoff](handoff/joint_sparse_win_20261015.md). Unless stated
otherwise, numbers come from saved run records, measured without new training.

## Where things stand

- **Best player:** the round-2 OOC PPO update 20 champion
  (`/Users/sidbaines/Documents/SlayTheSpireAI/.claude/worktrees/slay-spire-rl-debug-da6d22/data/rl_runs/ooc_r2_lam1/iter_0020.pt`).
  It wins 5.2–5.7% of games: 213/4,096 fresh worlds and 58/1,024 DEV worlds.
- **Combat head never improved:** the champion's combat head is the original
  from the first campaign. Every later gain came from the out-of-combat (OOC)
  head, through the macro-credit PPO recipe and two distilled rules.
- **Joint sparse-win run:** 62 vs 58 of 1,024, p=0.68; no change.
- **Play-time search is the largest measured improvement:**
  - Act 2–3 boss and elite search: 333 vs 213 of 4,096 fresh worlds.
  - Adding ordinary hallway fights: 98 vs 29 of 512. That pilot reused DEV
    worlds and is not confirmed.

## What the evidence says

1. **The joint run could barely move the policy, so its null result says little
   about sparse-win RL.**
   - Per-round approximate KL was 0.0002–0.0005, against 0.006–0.016 per update
     in the OOC recipe that did improve (`ooc_r2_lam1`); that is about 30× smaller.
   - Entropy and top-action probability were flat across all 19 rounds.
   - By the end, argmax choices differed from the original on only 1.6–3.1% of
     combat states and 0.8% of OOC states (`final_review.json`).
   - OOC sampling entropy was 0.10 (top-action probability 0.957), so it barely
     explored.
   - The cause was the settings: learning rate 1e-5/3e-5, one epoch, clip 0.1,
     and rehearsal toward the original.
2. **Critic calibration is a weak lead.**
   - Advantages are mean-centred and scaled per batch
     (`src/sts_rl/joint_sparse.py:886-905`), so a uniform shift in predicted win
     probability cancels exactly.
   - With λ=1 the critic is only a baseline. At explained variance 0.05–0.11,
     even a much better critic would cut advantage variance by about 10% at
     most.
   - The handoff's suggestion to make calibration the next topic is therefore
     low value.
3. **The 1,024-world evaluation can't see the gains we hope for.** Rough paired
   estimate, assuming about 5% discordant worlds:
   - 1,024 worlds: about 26% power for +1 point (an 18% relative gain), about
     68% for +2 points.
   - 4,096 worlds: about 74% for +1 point.

   Fight-level panels with resampled RNG give far more power for combat changes.
4. **Where runs end (1,024-world DEV baseline):**
   - 42% of all runs end at the Act 2 boss: 428 of 1,024, a pass rate of 349/777 = 45%.
   - Act 3 boss pass: 58/201 = 29%. Act 1 boss pass: 95%.
   - Act 2 boss pass by entry HP (confounded with deck strength):

     | Entry HP | Pass rate |
     |---|---|
     | 90+ | 74% |
     | 80s | 62% |
     | 70s | 45% |
     | 60s | 35% |
     | 50s | 22% |
5. **Combat is the biggest lever.**
   - On the 512-world pilot, searching every Act 2–3 combat roughly triples wins
     (29 → 98).
   - Hallways appear to matter most: Act 3 boss entry HP rises from 59 to 71, and
     Act 3 boss pass from 27% to 41%.
   - Search at the Act 3 boss itself does not help: 6 → 6 wins in 32 teacher
     fights.
6. **The earlier search-imitation attempts were not real tests.**
   - The canonical-teacher pilots fitted 2,460 labels with 32 optimizer steps.
   - The 18.7k-label run used the older sampler on bosses only.
   - The 51k-label DAgger run used privileged search, and about two-thirds of
     that search's edge was knowing the hidden future.
   - Teacher noise is not the whole story. In the 3,217 canonical labels
     (`/Users/sidbaines/.codex/worktrees/public-search-canonical/SlayTheSpireAI/data/canonical_20261014/expert/late128/`),
     individual samples often disagree: median top vote share 0.50, and only 5%
     of states are unanimous. But the teacher's plurality choice is about 78%
     self-consistent (an optimistic bootstrap), against the student's 44%
     agreement.
7. **The combat state representation loses a lot** (`src/sts_policy/encoding.py`).
   - Everything text-derived (hand, all piles, enemies, powers, relics, potions,
     menu) shares one 192-slot signed hash bag (`hashed = features[64:]`, line
     168). On one real state, 56 of 123 tokens collided.
   - Enemy Strength, Vulnerable, Weak and Poison are summed across enemies (line
     208).
   - Enemy block and max HP are missing from the state vector.
   - Same-name enemies merge, and actions already taken this turn are not
     encoded.
   - The network is a plain MLP with no per-entity structure. The linear probes
     of its embedding proposed on 13 October were never run.
   - The `archive/nonllm/critique` tag has typed per-entity public features and
     residual policies that are relevant here.
8. **Costs:**
   - Canonical search costs about 0.26 s per decision (24 samples × 1,000
     simulations), so 100k labels is roughly 5–7 thread-hours.
   - Resampling a fight from an out-of-combat state costs about 0.07 s. Card
     rewards and events cannot be re-randomized without a new binding.
   - Learned-policy rollouts ran at about 5 games/s. The
     [training speedups](joint_training_speedups.md) give 1.62× on the
     benchmark fixture, opt-in.

## Avenues, ranked

1. **Combat expert iteration at scale, judged on fight panels.** This is the top
   pick: the largest measured headroom, never properly tested, and affordable on
   the laptop.
   - **Data:**
     - collect canonical-search labels on states the student actually visits,
       in all combat rooms, including Act 1 and hallways;
     - collect 50–100k labels and keep the full vote distributions;
     - train on soft targets for thousands of optimizer steps, with rehearsal.
   - **Learning curve:** fit at 5k, 20k and 80k labels and score on held-out
     worlds.
   - **Evaluation:** resampled, paired fight panels from the 512-world bank
     (`.../public-search-canonical/SlayTheSpireAI/data/canonical_20261014/expert/train512_bank.jsonl`:
     368 Act 2 bosses, 2,326 Act 2 hallway fights and so on), scoring win rate
     and HP lost. Run full 4,096-world games only for candidates that pass.
   - **Decision rule:** if held-out agreement plateaus well below the teacher's
     own consistency even with much more data, the representation is the limit;
     go to option 2.
   - **Risks:** each search sample plans as if one sampled future were known, so
     some votes are "hope" plays. Nothing here fixes the Act 3 boss.
2. **An entity-structured combat encoder.**
   - Per-card and per-enemy tokens (HP, max HP, block, intent damage × hits,
     each power), a record of this turn's actions, and a small attention model.
     Candidates attend to their target.
   - The cleanest test trains the old and new encoders on the same labels from
     option 1.
   - Run the cheap embedding probes first.
3. **OOC: return to the recipe that worked, with less noisy credit at the boss
   gates.**
   - The macro-credit recipe: floor reward, λ=1, learning rate 3e-4, 3 epochs,
     about 0.01 KL per update.
   - New ingredient: replace each boss's single win/lose outcome with
     P(pass | deck, HP), estimated from about 16 resampled fights, roughly 1 s
     per gate. Use it as the value target at the gate or as a sub-goal reward for
     Acts 1–2. This targets the Act 2 boss, which ends 42% of runs.
   - Mix in the full-game outcome to avoid myopia.
   - Do this after combat improves, because a deck's value depends on how well
     the agent fights.
4. **Throughput.** The training speedups are merged. Further gains would
   multiply every option, so measure first.

**Deprioritize:**
- more critic calibration work (point 2);
- continuing the sparse-win condition as is (689 wins in 14.6k games is too
  little signal);
- more hand-written rules (most of those tested failed);
- the small-LLM line (around floor 17–19, against 35 here).

## Decisions needed before the next study

1. **Which study, and its budget.**
   - Related question: does a player that uses search count as "the model"?
   - If yes, the fastest route is search everywhere, with learning focused on
     guiding it.
   - If the network must play alone, options 1 and 2 are the critical path.
2. **Environment version: stay on 0.1.4 or adopt 0.1.5 first.**
   - **What 0.1.5 changes:**
     - free-once and retained flags in action and hand text;
     - out-of-combat potion actions, which add new legal choices the champion
       has never seen;
     - the burning-elite marker and keys;
     - live combat HP and gold;
     - event fixes;
     - Curiosity, which makes the Awakened One harder, plus Intangible,
       Lagavulin, Flight and Angry fixes.
   - **What that implies:** text-hashed features shift, so all baselines must be
     recomputed.
   - **Merge effort:**
     - Merging `env-v0.1.5` into `ws/nonllm` conflicts in about 12 files. Most
       are parallel copies of the same 0.1.2–0.1.4 changes, so taking `main`'s
       version resolves them; `pyproject.toml` and `docs/research_plan.md` need
       this branch's content kept.
     - The research overlay (`patches/sts_lightspeed_rl_panel.patch`) passes
       `git apply --check` on top of the 0.1.5 core patch; it has not been
       built or tested there.
   - **Recommendation:** adopt 0.1.5 as the first task of the next study, since
     any new study needs fresh baselines anyway.
3. **Backups.**
   - Models, datasets and native builds exist only on this laptop, spread across
     several worktrees listed in the handoff.
   - The 3.2 GB joint run has no off-machine copy.
   - New studies will add more, so decide on a backup before the next study
     creates more.
