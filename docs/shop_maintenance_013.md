# Shop maintenance and observation migration

Release 0.1.3 repairs reached upstream shop defects. They were found on the
non-LLM workstream: on TRAIN world 40000000040, Courier restocks produced
1-gold potion offers, and the policy repeatedly purchased them even with a full
inventory. The old pricing function discarded its base-price argument, discount
results were unused, and the action validator permitted ineffective purchases.
Trajectories recorded before the repair keep their original identities.

The tracked native patch now multiplies restock base prices by the existing merchant variance and applies Courier and Membership discounts. It assigns the replacement relic's computed price, removes the duplicate Courier half-discount on replacement cards, applies Courier when acquired in the current shop, and checks the actual acquired egg relic when upgrading shop offers. Potion purchases require space and absence of Sozu; invalid native execution throws before changing gold. No unrelated high-ascension pricing or unsupported character rules are changed.

Package/environment version advances to 0.1.3. The observation label advances to `sts_env_public_v2`: legal shop menus now exclude ineffective potion purchases, so indices can differ. The JSON trace container format remains unchanged, but native/interface identities change. Replay old traces and banks with their original source/build; do not strip their expected provenance. Models trained on `sts_env_public_v1` can be loaded if their encoders do not depend on the removed menu entries, but their actions and outcomes must be re-evaluated on 0.1.3.

Five constructed native shop fixtures in `tests/integration/test_native_shop_maintenance.py` exercise actual setup through a real map shop, then check restock prices/discounts, full inventory, Sozu, unchanged gold on invalid purchase, card price discounts and new relic charges. They are mechanics regressions, not player performance. [The release record](release_v0.1.3.md) has the build, patch and suite results. Experiments need new full-run baselines before comparing results with 0.1.2 or earlier.
