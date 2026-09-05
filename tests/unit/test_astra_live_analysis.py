from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.astra_analyze_live import (
    E4B_SUFFIX, IncompleteCohort, PROBE_HASH, PUBLIC_KEYS, compare, load_cohort,
    validate_comparison, validate_config, validate_episode,
)
from sts_ai.seeding import derive_policy_seed


class TestAstraLiveAnalysis(unittest.TestCase):
    def fixture(self, seed: int = 1000) -> tuple[dict, list[dict], dict]:
        config = {"mode":"rollout","seeds":str(seed),"model":"/cache/"+E4B_SUFFIX,
                  "contract":"action_text","constrained":False,"batch_size":16,
                  "max_tokens":256,"temperature":0.0,"max_retries":1,"max_decisions":250,
                  "script_sha256":PROBE_HASH,"adapter":None,"adapter_identity":None}
        interface = {key:"b"*64 for key in PUBLIC_KEYS[:5]}
        interface.update(combat_observation="combat_public_v2",competence_interface_version="combat_public_v2")
        ac = {"model_id":config["model"],"output_contract":"action_text","temperature":0.0,
              "max_tokens":256,"max_retries":1,"adapter_path":None,"experimental_decoder":"free_v1","thinking":False}
        meta = {"schema_version":3,"world_seed":seed,"rollout_index":0,"policy_seed":derive_policy_seed(seed,0),
                "ascension":0,"combat_control":"search","battle_simulations":None,
                "extra":{"interface_provenance":interface,"agent_config":ac},
                "n_decisions":1,"n_invalid":0,"stopped_reason":"terminal","error":None,
                "final_floor":10,"outcome":"GameOutcome.PLAYER_LOSS",
                "model_id":config["model"],"temperature":0.0,"max_tokens":256,"max_retries":1}
        action = {"index":0,"bits":0,"description":"legal"}
        record = {"world_seed":seed,"rollout_index":0,"policy_seed":meta["policy_seed"],"decision_index":0,
                  "phase":"out_of_combat","legal_actions":[action],"selected_action":action,
                  "agent":{"valid":True,"retries":0},"action_executed":True,"after_state":{"floor":10}}
        return meta,[record],config

    def arm(self, floor: int | None = 10, name: str = "arm") -> dict:
        meta,records,config=self.fixture();episode=validate_episode(meta,records,1000,config)
        episode.update(floor=floor,scored=floor is not None)
        return {"name":name,"config":config,"episodes":[episode],"interface":episode["interface"]}

    def test_valid_frozen_configuration_and_episode(self) -> None:
        meta,rows,cfg=self.fixture();validate_config(cfg,(1000,))
        self.assertTrue(validate_episode(meta,rows,1000,cfg)["scored"])

    def test_fatal_invalid_episode_retains_floor_in_quality_metric(self) -> None:
        meta,rows,cfg=self.fixture();meta.update(stopped_reason="agent_invalid",n_invalid=1)
        rows[0].update(agent={"valid":False},action_executed=False,selected_action={})
        result=validate_episode(meta,rows,1000,cfg)
        self.assertTrue(result["scored"]);self.assertEqual(result["floor"],10)

    def test_simulator_error_is_unscored_and_bounded_without_silent_loss(self) -> None:
        meta,rows,cfg=self.fixture();meta.update(stopped_reason="simulator_error",error={"type":"NativeError"})
        result=validate_episode(meta,rows,1000,cfg)
        self.assertFalse(result["scored"]);self.assertIsNone(result["floor"])
        report=compare(self.arm(10,"control"),self.arm(None,"error"))
        self.assertEqual(report["n_planned_worlds"],1);self.assertEqual(report["n_scored_pairs"],0)
        self.assertIsNone(report["mean_floor_delta"])
        self.assertEqual(report["all_world_capped_floor_bounds"],{"metric":"min(final_floor,50)","lower":-10,"upper":40})

    def test_ub_and_budget_truncation_are_unscored(self) -> None:
        for changes in ({"undefined_behavior_evoked":True},{"stopped_reason":"max_decisions"}):
            meta,rows,cfg=self.fixture();meta.update(changes)
            self.assertFalse(validate_episode(meta,rows,1000,cfg)["scored"])

    def test_schema_identity_and_trace_contiguity_fail_closed(self) -> None:
        for changes in ({"schema_version":2},{"policy_seed":1},{"world_seed":1001}):
            meta,rows,cfg=self.fixture();meta.update(changes)
            with self.assertRaises(ValueError):validate_episode(meta,rows,1000,cfg)
        meta,rows,cfg=self.fixture();rows.append(copy.deepcopy(rows[0]));meta["n_decisions"]=2
        with self.assertRaisesRegex(ValueError,"contiguous"):validate_episode(meta,rows,1000,cfg)

    def test_executed_fallback_for_invalid_is_rejected(self) -> None:
        meta,rows,cfg=self.fixture();rows[0]["agent"]["valid"]=False
        with self.assertRaisesRegex(ValueError,"invalid action was executed"):validate_episode(meta,rows,1000,cfg)

    def test_missing_or_changed_provenance_is_rejected(self) -> None:
        meta,rows,cfg=self.fixture();del meta["extra"]["interface_provenance"]["simulator_binary_sha256"]
        with self.assertRaisesRegex(ValueError,"missing interface"):validate_episode(meta,rows,1000,cfg)
        a,b=self.arm(),self.arm();b["interface"]["simulator_binary_sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"simulator/interface"):validate_comparison(a,b)

    def test_temperature_and_batch_comparison_mismatch_fail(self) -> None:
        for key,value in (("temperature",.7),("batch_size",8),("max_retries",2)):
            a,b=self.arm(),self.arm();b["config"][key]=value
            with self.assertRaisesRegex(ValueError,"generation configuration"):validate_comparison(a,b)
        _meta,_rows,cfg=self.fixture();cfg["temperature"]=.7
        with self.assertRaisesRegex(ValueError,"temperature0"):validate_config(cfg,(1000,))

    def test_decoder_intervention_is_explicit_and_family_baseline_not_temperature_matched(self) -> None:
        a,b=self.arm(),self.arm();b["config"]["constrained"]=True
        validation=validate_comparison(a,b)
        self.assertIn("constrained",validation["observed_changes"])
        a["config"]=None
        self.assertEqual(validate_comparison(a,b)["kind"],"different_policy_families_same_worlds_and_simulator")

    def test_partial_cohort_never_produces_outcome_summary(self) -> None:
        meta,rows,cfg=self.fixture();cfg["seeds"]="1000,1001"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/"config.json").write_text(json.dumps(cfg))
            (p/"seed_1000_r0.meta.json").write_text(json.dumps(meta))
            (p/"seed_1000_r0.jsonl").write_text(json.dumps(rows[0])+"\n")
            with self.assertRaises(IncompleteCohort):load_cohort(p,seeds=(1000,1001))

    def test_extra_model_world_cannot_be_silently_intersected(self) -> None:
        meta,rows,cfg=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/"config.json").write_text(json.dumps(cfg))
            for seed in (1000,1001):
                (p/f"seed_{seed}_r0.meta.json").write_text(json.dumps(meta))
                (p/f"seed_{seed}_r0.jsonl").write_text(json.dumps(rows[0])+"\n")
            with self.assertRaisesRegex(ValueError,"unexpected rollout"):load_cohort(p,seeds=(1000,))

    def test_stale_runner_hash_cannot_supply_missing_simulation_budget(self) -> None:
        meta,rows,cfg=self.fixture();cfg["script_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"search budget"):validate_episode(meta,rows,1000,cfg)

    def test_unknown_snapshot_is_not_an_implicitly_allowed_model_bundle(self) -> None:
        _meta,_rows,cfg=self.fixture();cfg["model"]="/cache/snapshots/"+"f"*40
        with self.assertRaisesRegex(ValueError,"unrecognized model bundle"):validate_config(cfg,(1000,))

    def test_declared_model_bundle_change_keeps_temperature_guard(self) -> None:
        a,b=self.arm(),self.arm();b["config"]["model"]="declared26";b["config"]["constrained"]=True
        b["interface"]["chat_template_probe_hash"]="different"
        a["interface"]["chat_template_probe_hash"]="original"
        def bundle(c):
            return {"kind":"gemma26_ollama_nvfp4_mxfp8" if c["model"]=="declared26" else "e4b_hf_snapshot"}
        with patch("scripts.experiments.astra_analyze_live.model_bundle",side_effect=bundle):
            result=validate_comparison(a,b)
            self.assertEqual(result["kind"],"explicit_model_bundle_intervention")
            b["config"]["temperature"]=.7
            with self.assertRaisesRegex(ValueError,"generation configuration"):validate_comparison(a,b)


if __name__=="__main__":unittest.main()
