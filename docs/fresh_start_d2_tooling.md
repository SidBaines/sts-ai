# D2 supervised warm-up tooling

This is Step C tooling only. No PPO, expert iteration, study optimizer run or
native job was launched. The vectorized `entity_model.py` is preserved.

`scripts/rl/fresh_train.py --probe PATH [PATH ...]` accepts immutable unlabelled
C3 world shards in the separate `[93e9,94e9)` namespace. Use the same paths for
old and new phase models: selection is independent of architecture, model seed
and FIT subset. `probe_selection.json` freezes the smallest stable row IDs for
up to 64 nontrivial filtered choices per ACT in the configured phase, with exact
worlds, public/menu hashes, source receipts/archive hashes and B3 coverage. The
selection file and raw probe shards are launch inputs; the receipt and every
checkpoint identify the selection separately from label data.

The dedicated probe loader accepts only completed worlds, verifies original
prefixes/actions, public hashes, menu routing, archived protocol/filter/public
context/inference identities, public interface/version and native snapshot bytes.
Compatibility with labels requires the same public environment/native/input
context, rather than equality of entire archives containing different encoders.
Only the public observation enters encoding. `source_action` verifies the
trajectory and is never a target. Probe examples have no target field, split,
teacher outcome or training sampler entry.

Every supervised evaluation block logs `fixed93e9_probes`: pooled and per-ACT
CPU singleton actor-only KL(previous || current), KL(initial || current), exact
logit argmax changes, entropy, top probability, phase/count and source worlds.
Accelerator training uses a detached CPU deepcopy refreshed from weights;
optimizer parameters stay on their training device. Scoring preserves RNG and
training mode and never executes the critic. Only VALIDATION loss selects best
checkpoints or triggers patience. TEST remains a final selected-model CPU pass.

`--probe` remains optional for synthetic tooling fixtures. Without it, only the
explicitly named fixed FIT diagnostics are emitted. They are never called 93e9
probes. Actual study commands must supply `--probe`. The dashboard distinguishes
probe KL from FIT diagnostic KL.

The v2 label manifest retains intentional canonical-teacher abstentions
(`label=None,votes={}`) as raw rows. It counts raw combat/OOC rows, informative
combat labels, abstentions, target rows, and OOC choice/forced targets separately.
Abstentions have no target checksum and are excluded from all target ID lists,
FIT subset orders, held-out datasets and optimization. Corrupt nonempty votes
still fail validation; collector informative/abstention counters are checked
when present. Choice-only agreement stays the primary accuracy statistic.
Repeat-query statistics count and exclude abstained pairs.

Real training calls strict `begin_run` and requires its validated native identity
to match the label dataset. There is no production preparation bypass. Tests
with synthetic native bytes replace only the recording wrapper. Teachers remain
data lineage inputs, with initialization from scratch and no parent checkpoint.
Minibatches bucket entity and action counts independently into powers of two,
retain every optimization choice example, and checkpoint RNG and remaining batch
order. D5 excludes forced OOC targets from optimization; they remain in the common
manifest and in VALIDATION/TEST, including all-row and choice-only metrics.

D2 validation output and the read-only actual probe audit are in separate
`d2_*` artifacts; D1 outputs remain unchanged. No device speed claim follows
from tiny synthetic plumbing measurements. This sandbox cannot access MPS or
bind localhost; host execution is needed for those conditional checks.
