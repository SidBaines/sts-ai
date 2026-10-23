# A better search teacher for combat

**Status:** running; findings so far, next steps under discussion (23 October).
**Verdict:** pooling the 24 searches makes a much stronger boss player (429
against 331 wins of about 1,020 worlds; more simulations help further), but
students distilled from it do not beat the champion, because they imitate it
poorly at bosses (46 to 66% agreement against the teacher's own 77 to 87%).

## Question

Does a teacher that plays boss fights better give a better student? The
teacher caps what expert iteration can reach
([expert iteration](../strategies.md#expert-iteration-for-combat)); the
champion loses about 75% of its games at the Act 2 and Act 3 bosses, and every
distilled student has won fewer boss fights than the search. The native source
(read 22 October) shows four weaknesses. Each of the 24 searches plans in one
sampled future, with the draw order and random streams fixed once. Each votes
for the first move of its best-scoring winning playout, else its most-visited
move; only 4.6% of round 2 labels are unanimous, and in 57% the top move has
12 or fewer of 24 votes. The score, 100 × (35 + HP + 4 × potions − 0.01 ×
turns) for a win, values HP that is worthless after a boss at Ascension 0, and
once any playout wins its range swamps the tree's value term (about 0.07
against an exploration term of about 1), so visits spread evenly. Finally,
samples forget cards known to be on top of the draw pile (Headbutt, Warcry,
the Frozen Eye relic), which the network's observation also lacks.

## Design

Each teacher variant is benchmarked as a boss player before it labels
anything, and the best then labels a distillation round (planned with Sid, 22
October). Step 0 measures the noise in today's votes and how often known top
cards could matter.

- **Benchmark:** the first 1,024 second-DEV worlds under the study protocol,
  run in the label collector: the champion plays all but the boss fights,
  which the teacher plays from public information; wins and boss pass rates
  by act, paired by world. (`eval_search_boss.py` shows out-of-combat potion
  actions, so its numbers are not comparable.)
- **Teacher changes:** (1) a best-mean vote with score fixes (no HP term at
  bosses, wins scored by estimated run-win probability, no potion or turn
  terms in the loss credit), of which a win-first score and an exploration
  constant were built as native options; (2) pooled searches, adding visits
  and values per root move across the 24 searches; (3) information-set tree
  search at the same 24,000-simulation budget; (4) network playouts and the
  critic's value. Steps 3 and 4 are not started.
- **Known top cards (deferred; Sid, 22 October):** after the vote, score and
  budget fixes, through simulator tracking, an observation change in its own
  pull request to main, and a new encoder version.

**Distillation with the pooled teacher (fixed 23 October, 00:40, before any
labels existed).** The champion (`af2fb9f1f260`) plays 2,000 new worlds from
90,501,000,000; the teacher labels them with round 2's room probabilities and
24 pooled searches of 1,000 simulations, recording per-search detail so that
luckiest-line labels come from the same searches. Two fits per label rule, on
this corpus only, spliced into the champion; probes are 8 worlds from
93,000,000,300. Decision rule: a pooled-label student is *Better* if it beats
the champion on second DEV (paired with its 899 wins) with exact McNemar p <
0.05, and is then confirmed on 4,096 unused worlds from 106,000,000,000
before any promotion (Sid's decision).

**Other decision rules (fixed 22 October).** A teacher variant replaces the
current one if it wins more benchmark boss fights (McNemar p < 0.05) and no
fewer Act 1 boss fights. Each new teacher gets two students, compared with the
champion on second DEV (4,096 worlds) and confirmed on unused worlds;
promotion is Sid's decision. A known-top-cards encoder is adopted if it is no
worse than the current one on the same labels, and better in decks with
Headbutt, Warcry or Frozen Eye. Labelling costs about 1.7 hours per round on
8 lanes.

## Results

**Noise: today's vote is mostly search noise that more search does not fix;
the best-mean vote improves with search, leaving disagreement between
futures.** How often two searches choose the same move, with one sampled
future and different search seeds, or different futures and one seed (134
benchmark decisions, about 45 per act, rather than the planned 500 corpus
positions; 8 searches per condition; chance is about 23%):

| Act | Vote rule | 1,000 simulations | 4,000 | 16,000 |
|---|---|---|---|---|
| 1 | Luckiest line (today) | 46% same future / 43% different | 50 / 45 | 50 / 48 |
| 1 | Best mean result | 70 / 59 | 78 / 70 | 91 / 74 |
| 2 | Luckiest line | 37 / 35 | 37 / 35 | 38 / 34 |
| 2 | Best mean result | 53 / 43 | 64 / 50 | 76 / 56 |
| 3 | Luckiest line | 50 / 43 | 53 / 44 | 55 / 46 |
| 3 | Best mean result | 58 / 49 | 66 / 56 | 71 / 60 |

Visits spread almost evenly over root moves (entropy 0.91 to 0.94 of the
maximum), and at 1,000 simulations a winning playout appears in about 95% of
Act 1 and 2 searches and 60% of Act 3 ones.

**Known top cards matter rarely.** In 174,862 round 2 combat decisions, the
deck holds Headbutt or Warcry in 3.7% of hallway, 2.9% of elite and 4.2% of
boss decisions, and Frozen Eye in 0.1 to 0.2%; the case the teacher gets
wrong occurs in about 0.03%.

**Benchmark: the teacher wins far more boss fights than the champion's combat
network** (1,024 worlds, 24 searches of 1,000 simulations):

| Boss fights played by | Wins | Paired difference in wins | Act 1 / Act 2 / Act 3 boss pass, paired difference |
|---|---:|---|---|
| Champion's combat network | 205 (20.0%) | | 95.1% / 67.9% / 38.4% |
| Teacher, today's vote (luckiest line) | 333 (32.5%) | +12.5 points (discordant 44/172, p < 0.001) | +0.3 [−0.9, +1.5] / **+12.9 [+9.9, +15.8]** / **+16.2 [+12.0, +20.4]** |
| Teacher, best mean result | 345 (33.7%) | +13.7 points (38/178, p < 0.001) | +0.2 [−1.0, +1.4] / **+14.7 [+12.0, +17.5]** / **+16.2 [+11.8, +20.5]** |

**Pooled searches are the best vote rule** (same worlds, 1,020 to 1,024
completed per arm, paired with today's teacher):

| Vote rule (24 searches × 1,000 simulations) | Wins | Paired difference vs today's vote | Act 1 / 2 / 3 boss pass vs today's vote |
|---|---:|---|---|
| Today's: first move of each search's luckiest playout | 331–333 | | |
| **Pooled searches: add up root visits and values across the 24 searches, take the best mean** | **429** | **+9.6 points (discordant 42/140, p < 0.001)** | +1.9 [+0.9, +3.0] / **+5.3 [+3.1, +7.6]** / **+11.1 [+7.5, +14.7]** |
| Most visited move per search | 361 | +2.9 (50/80, p = 0.011) | −0.5 / +0.6 / +4.0 [+1.0, +7.1] |
| Best mean per search, win-first score | 362 | +3.0 (68/98, p = 0.024) | −0.1 / +2.1 / +5.1 [+1.4, +8.8] |
| Best mean per search | 345 | +1.2 (61/73, p = 0.34) | −0.1 / +1.9 / −0.2 |
| Best mean per search, win-first score, exploration 1.0 | 329 | −0.4 (81/77, p = 0.81) | −0.9 / −3.3 [−5.8, −0.8] / +1.9 |

- With a Holm correction, pooled and most visited stay significant; the
  win-first arm does not. Best mean against today's is +1.2 points [−1.0,
  +3.4], no clear difference. Pooled bosses lift the champion's games to 429
  of 1,020 wins (42%), against 203 (20%) alone.
- Paired with pooled at 1,000 simulations: 2,000 simulations won 454 against
  429 (55/80, p = 0.038; Act 3 boss pass +4.1 points [+1.1, +7.1]); the
  win-first score won 404 against 429 (97/72, p = 0.065; Act 2 boss pass
  −2.0 [−4.1, −0.1]).

**Distillation round (rule fixed in advance): no pooled-label student beats
the champion.** The corpus (`combat_pooled_20261023`) has 134,376 fit rows,
about 98 labels per world, with 23% of the 2,000 games won. Second DEV,
paired with the champion's 899 wins:

| Student | Wins | McNemar p | Mean floor | Act 2 boss pass |
|---|---:|---:|---|---|
| Pooled labels, seed 1 | 873 | 0.39 | −1.70 [−2.09, −1.31] | −0.5 [−2.3, +1.3] |
| Pooled labels, seed 2 | 718 | < 0.001 | −2.92 [−3.33, −2.51] | −1.6 [−3.5, +0.3] |
| Luckiest-line labels from the same searches, seed 1 | 717 | < 0.001 | −1.92 [−2.27, −1.56] | −6.0 [−7.8, −4.3] |
| Luckiest-line labels, seed 2 | 843 | 0.059 | −0.32 [−0.68, +0.04] | +0.6 [−1.1, +2.4] |

Pooled against luckiest-line, same seed: 873 against 717 and 718 against 843
(both p < 0.001), so the seeds disagree. Three of the four fits lose 1.7 to
2.9 floors, mostly before the bosses.

**Imitation: students reproduce the teacher's boss choices poorly**
(exploratory). Agreement of the student's top move with the pooled teacher's
choice, on validation rows:

| | Act 1 boss | Act 2 boss | Act 3 boss | All rows |
|---|---:|---:|---:|---:|
| Two independent pooled teachers (24 different sampled futures each), 60 boss decisions per act | 87% | 78% | 77% | |
| Student on all 134,376 rows (two seeds) | 66% | 54–55% | 45–46% | 57% |
| Student on 20,000 rows | 63% | 49% | 40% | 51% |
| Student on 5,000 rows | 54% | 45% | 38% | 47% |

Students fall 21 to 31 points short, most at the Act 3 boss. Agreement rises
about 5 points per 7-fold increase in data, so halving the gap with data alone
would take on the order of 50 times more labels. The two full-data seeds agree
equally often (57%) but win 873 and 718.

Longer training does not close the gap: with patience 30 instead of 6 (up to
30,000 steps), validation agreement is 57.5% at learning rate 0.001 and 57.7%
at 0.0003, against 57.1%. Agreement on the training rows keeps rising (to 65
to 69%) while validation levels off, so at this data size the student starts
to overfit; and even its training agreement stays far below what one-hot
labels allow, so the hand-featured network also fits these choices slowly.
The structured entity transformer does no better on the same labels:
validation agreement 57.2% (width 128, 2 layers) and 57.9% (width 256, 4
layers), with the same per-act pattern. Two different model classes level off
at the same place, which points to the information both read (the same public
observation) or to data volume, not to network capacity.

**Where the champion and the pooled teacher disagree at bosses: potions
most of all** (exploratory; the corpus records both the champion's own move
and the teacher's choice at 96,260 labelled boss decisions). When the teacher
chooses to drink a potion (11% of boss labels), the champion does the same 12%
of the time, against about 53% for card plays and ending the turn.
Disagreement is highest in the first turns of a fight (44 to 46% agreement on
turns 0 to 2, 55% from turn 6) and at the Act 3 bosses (41 to 47%, Awakened
One lowest). In the benchmark the pooled teacher drinks 3.9 potions per game
at bosses; games with it playing the bosses use 10.4 potions in combat against
8.7 when the champion plays everything, so it spends about 1.7 more per game.
**About half of the teacher's boss advantage is potion use.** The pooled
teacher with potion actions kept out of its vote won 316 games, against 429
for the full pooled teacher (paired on the same worlds: discordant 167/53, p <
0.001) and 205 for the champion alone; its Act 1, 2 and 3 boss pass rates fall
by 3.0, 9.7 and 10.3 points against the full teacher. Of the full teacher's
224 extra wins over the champion, 113 vanish without potions. Its searches
still plan potion use deeper in the tree, which it then cannot carry out, so
this likely understates the no-potion teacher; "about half" is an upper
estimate of the potions' share.

## Conclusion

The teacher's ceiling is not what holds the student back; imitation is.
Established: the teacher plays bosses far better than the champion's combat
network (333 against 205 wins); pooling its searches is the best vote rule
(429 against 331, meeting the replacement rule), 2,000 simulations per search
beat 1,000 (454 against 429), and the win-first score shows no clear
difference once searches are pooled; no pooled-label student beat the
champion under the rule fixed in advance. Not established: whether pooled
labels make better students than luckiest-line ones, or whether
information-set search, network playouts or known top cards would help.

Caveats:
- Teacher benchmarks are exploratory, on the reused first quarter of second
  DEV, and a search-played boss fight is not a pure policy.
- Agreement measures consistency, not strength, counts equivalent move
  orderings as different, and was measured on different boss states for the
  teacher and the students.
- The champion's combat head is the best of five fits; fit-to-fit variance
  dominates any label effect.
- 23 of 1,024 benchmark worlds left the champion's own game before the
  teacher's first move (near-tied scores break differently in batched
  inference), which adds noise, not bias. The 0.03% figure is approximate.

## Next

Candidates, under discussion with Sid: a much larger pooled-label corpus, or
boss-weighted labelling; better student capacity or training (only about +5
points of agreement per 7-fold data so far); the pooled teacher inside the
joint loop's labelling; information-set search and network playouts, as
planned; known top cards later, as Sid decided. A hypothesis for the imitation gap
worth testing first: neither encoder sees the enemies' move history (only the
current intent and powers; the hand-featured one also sees the turn number),
while the teacher's simulations follow the true enemy AI, including its
history rules; a student with a short history of enemy moves, or a recurrent
student, could close part of the gap.

## Log

- 2026-10-22: planned with Sid (steps 0 to 3, score fixes, known top cards with encoder; step 4 follow-on); Secret Technique and Secret Weapon still unlabelled (1 abstention in 35,054 labels), no fix planned.
- 2026-10-22: vote rules need no native work; the benchmark runs in the label collector.
- 2026-10-22, 17:43: four benchmark arms queued; the first 110 worlds matched the champion's game up to the teacher's first move.
- 2026-10-22, 17:51: noise measured; vote rule and budget now come before information-set search.
- 2026-10-22: known top cards occur in about 0.03% of decisions; Sid deferred them until after the vote, score and budget fixes.
- 2026-10-22, 19:35: the teacher at bosses wins 32.5 to 33.7% against the champion's 20.0%.
- 2026-10-22, 22:16: six arms analysed; pooled and most-visited arms stopped by low disk space, to be rerun.
- 2026-10-23, 00:21: full reruns: pooled searches win 429 against 331 (p < 0.001).
- 2026-10-23, 04:09: pooled distillation round negative; pooled at 2,000 simulations beats 1,000 (454 against 429).
- 2026-10-23, 04:24: imitation gap: 46 to 66% student agreement against 77 to 87% teacher self-agreement.

Code: `experiments/2026-10-22-better-teacher/` at bad8eb2; native search
options at ec3f72e. Data: run `combat_pooled_20261023` (local).
