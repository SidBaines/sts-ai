# Separate OOC potion action-space candidate

The accepted current profile remains environment 0.3/public-v3/API-3. This
independent native candidate is **not adopted into its training or benchmarks**.
Its full scoped patch is `patches/sts_lightspeed_ooc_potions_candidate.patch`;
source/build are `external/sts_lightspeed_v040`. The patch includes the accepted
API-3 repairs plus the changes below, so apply it to clean pinned upstream rather
than on top of the API-3 patch.

The native validator accepts Blood Potion, Fruit Juice and Entropic Brew outside
combat, and permits discarding occupied potion slots. Yet the ordinary legal-menu
enumerator never supplied these actions. Direct potion execution also continued
into the event/reward/map/rest switch, using the potion slot as a screen choice.
That can silently make two decisions or trigger an unrelated callback.

The candidate appends valid potion actions after all ordinary choices, retaining
old prefix indices; preserves unsupported/empty-screen behavior; returns
immediately after potion consumption/discard; and rejects invalid game actions
in Release rather than relying on disabled assertions. Typed metadata exposes
the public potion slot, current Blood Potion healing and Fruit Juice max-HP gain.

The disposable native regression verifies ordinary indices remain stable, Blood
Potion heals exactly 16 at max HP80, Fruit Juice increases max HP by5, a combat-only
Fire Potion cannot be drunk OOC, discard preserves the current screen/world,
repeated invalid discard throws, and drinking at a rest screen does not also
rest. Results are `data/nonllm_v4/preflight/ooc_potions_candidate_acceptance.json`;
fresh and reverse application checks pass. Native module digest is recorded in
that acceptance output. Reproduction:

```
.venv/bin/python scripts/investigations/probe_ooc_potions_candidate.py \
  --build external/sts_lightspeed_v040/build \
  --output /private/tmp/ooc_potions_candidate_check.json
```

Adoption still needs an explicit native/interface revision, full integration
tests across additional screens, and policy transfer handling for genuinely new
legal choices. Existing old policies have no calibrated prior for those choices;
silently expanding their menu would compromise the baseline comparison. New
compatible banks and learning gates must follow. This is a concrete tested repair
candidate, not a claim that the current deployed action space already includes it.
