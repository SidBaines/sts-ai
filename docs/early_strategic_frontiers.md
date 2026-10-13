# Earlier strategic start collection

13 October 2026. `scripts/collect_early_strategic_frontiers.py` records exact
executed prefixes from fresh TRAIN games controlled by the frozen original
public OOC policy and explicitly privileged native combat. This is a source
curriculum, not a public-agent benchmark or a teacher target bank.

The parent worker records a choice before executing its current action, so the
saved prefix contains only preceding actions and their actual preview markers.
Only completed, valid source worlds are eligible. A later source fault removes
every candidate from that world; unfinished/faulted candidates stay preserved
in quarantine. Every accepted start then reconstructs the exact public
observation and legal menu in a separate native process. A replay failure
excludes its whole source world. Deadline-limited unverified starts remain
separate from accepted data.

The initial v1 probe completes all 16 fresh TRAIN games in **20.49 seconds**
using two workers, native combat at 1,000 simulations, no faults/cuts and no
source victories. It records and replay-verifies **89/89** starts from 14 source
worlds: 84 Act 2 starts from 14 worlds, five Act 3 starts from **one world**.
Different screens and acts from one world are correlated. This is successful
collection and reconstruction, not broad Act 3 coverage or a useful teacher.

The source is `data/nonllm_v4/banks/early_original_native1000_probe16`; its
manifest, complete source outcomes, exact candidates, accepted starts and
quarantine/unverified files are preserved. Latest v1 validation runs 553
host/native tests with one optional artifact skip. The accepted environment
remains 0.3/public-v3/native-API-3; no core or model weights change.

The probe exposes a selection limitation: the first reward menu still contains
gold and can omit the actual later card-only decision. Next versions should
distinguish public menu-kind/selection-purpose stages and include the Act 1
boss-relic choice, which shapes later deck viability. More frontier rows cannot
substitute for more independent late-act worlds. Future sampling must choose
worlds before starts so a world with many recorded screens does not dominate
the training mix.
