"""Fail-closed analysis of completed September live-policy cohorts."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
import statistics

from sts_ai.eval_stats import bootstrap_ci
from sts_ai.provenance import adapter_provenance, file_sha256
from sts_ai.seeding import derive_policy_seed

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(1000, 1016))
PROBE_HASH = "484b216f36c287e7f483280d1fb16f2334fc51ad3b1ab491da4bd9f13402e56d"
E4B_SUFFIX = "models--mlx-community--gemma-4-e4b-it-bf16/snapshots/eec12d0899edea9b738ab1009af9159cdfd70d71"
GEMMA26_DIR = ROOT / "data/experiments/astra_5sep/gemma26_ollama"
GEMMA26_MANIFEST_HASH = "f0fc7e0ae4947d382989b1f57db3d098a86b87c5b6e9523b9561ba81a2a64879"
PUBLIC_KEYS = ("simulator_binary_sha256", "simulator_patch_sha256", "python_serializer_sha256",
               "glossary_sha256", "prompting_sha256", "combat_observation", "competence_interface_version")
GENERATION_KEYS = ("model", "contract", "batch_size", "max_tokens", "temperature", "max_retries", "max_decisions", "script_sha256")


class IncompleteCohort(ValueError):
    pass


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def model_bundle(config: dict) -> dict:
    model = config["model"]
    if model.endswith(E4B_SUFFIX):
        return {"kind":"e4b_hf_snapshot","path":model,"revision":E4B_SUFFIX.split("/")[-1]}
    if Path(model).resolve() != GEMMA26_DIR.resolve():
        raise ValueError("unrecognized model bundle")
    if config["adapter"] is not None or config["constrained"] is not True:
        raise ValueError("Gemma26 arm must be constrained base without an adapter")
    directory=GEMMA26_DIR
    audit=json.loads((directory/"staging_report.json").read_text())
    strict=json.loads((directory/"strict_load_report.json").read_text())
    if audit.get("source_manifest_sha256") != GEMMA26_MANIFEST_HASH or not strict.get("strict_lazy_load_passed"):
        raise ValueError("Gemma26 source/strict-loader audit mismatch")
    checks={"config.json":"config_sha256","ollama_model.py":"custom_model_sha256",
            "chat_template.jinja":"chat_template_sha256","strict_load_report.json":"strict_load_report_sha256"}
    for filename,key in checks.items():
        if file_sha256(directory/filename) != audit.get(key):
            raise ValueError(f"Gemma26 audited file changed: {filename}")
    if len(audit.get("sources",[])) != 657 or audit.get("source_blobs_modified") is not False:
        raise ValueError("Gemma26 weight-source inventory mismatch")
    for record in audit["sources"]:
        path=directory/record["symlink"]
        if path.resolve() != Path(record["source"]).resolve() or path.stat().st_size != record["size"] or path.resolve().name != "sha256-"+record["sha256"]:
            raise ValueError("Gemma26 audited weight link changed")
    for filename,record in audit["tokenizer_sources"].items():
        if file_sha256(directory/filename) != record["sha256"]:
            raise ValueError("Gemma26 tokenizer provenance mismatch")
    return {"kind":"gemma26_ollama_nvfp4_mxfp8","path":str(directory.relative_to(ROOT)),
            "source_manifest_sha256":GEMMA26_MANIFEST_HASH,
            "staging_report_sha256":file_sha256(directory/"staging_report.json"),
            "strict_load_report_sha256":file_sha256(directory/"strict_load_report.json"),
            "custom_model_sha256":audit["custom_model_sha256"],"config_sha256":audit["config_sha256"],
            "chat_template_sha256":audit["chat_template_sha256"],"quantization_modes":audit["modes"],
            "weight_verification":"Source hashes were verified by staging audit; analyzer rechecks audited symlink targets and sizes, not all large weight bytes."}


def validate_config(config: dict, seeds: tuple[int, ...]) -> None:
    required = set(GENERATION_KEYS) | {"mode", "seeds", "constrained", "adapter", "adapter_identity"}
    if required - config.keys():
        raise ValueError(f"missing generation provenance: {sorted(required-config.keys())}")
    actual = [int(s) for s in config["seeds"].split(",")]
    if len(actual) != len(set(actual)) or set(actual) != set(seeds):
        raise ValueError("configured seed cohort differs from planned identities")
    if config["mode"] != "rollout" or config["script_sha256"] != PROBE_HASH:
        raise ValueError("unrecognized rollout script provenance")
    if config["contract"] != "action_text" or config["temperature"] != 0.0:
        raise ValueError("live study requires action_text and temperature0")
    if config["max_decisions"] != 250:
        raise ValueError("decision budget differs from planned250")
    model_bundle(config)
    if adapter_provenance(config["adapter"]) != config["adapter_identity"]:
        raise ValueError("adapter identity mismatch")


def validate_episode(meta: dict, records: list[dict], seed: int, config: dict | None) -> dict:
    if meta.get("schema_version") != 3:
        raise ValueError("schema_version mismatch")
    identity = (seed, 0, derive_policy_seed(seed, 0))
    if tuple(meta.get(k) for k in ("world_seed", "rollout_index", "policy_seed")) != identity:
        raise ValueError("metadata rollout identity mismatch")
    if meta.get("ascension") != 0 or meta.get("combat_control") != "search":
        raise ValueError("environment configuration mismatch")
    extra = meta.get("extra", {})
    provenance = extra.get("interface_provenance", {})
    if any(k not in provenance for k in PUBLIC_KEYS):
        raise ValueError("missing interface provenance")
    for key in PUBLIC_KEYS[:5]:
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance[key])):
            raise ValueError(f"malformed interface hash: {key}")
    if provenance["combat_observation"] != "combat_public_v2" or provenance["competence_interface_version"] != "combat_public_v2":
        raise ValueError("public observation mismatch")
    if config is not None:
        # The pinned runner omitted this metadata field; its audited source fixes
        # search50, default max_act3 and boss multiplier2. Reject other code hashes.
        if meta.get("battle_simulations") not in (None, 50) or config["script_sha256"] != PROBE_HASH:
            raise ValueError("search budget cannot be verified")
        ac = extra.get("agent_config", {})
        mapping = {"model_id": "model", "output_contract": "contract", "temperature": "temperature",
                   "max_tokens": "max_tokens", "max_retries": "max_retries", "adapter_path": "adapter"}
        if any(ac.get(k) != config[v] for k,v in mapping.items()):
            raise ValueError("agent generation configuration differs from arm config")
        expected_decoder = "canonical_trie_v1" if config["constrained"] else "free_v1"
        if ac.get("experimental_decoder") != expected_decoder or ac.get("thinking") is not False:
            raise ValueError("decoder/thinking provenance mismatch")
        if any(meta.get(k) != config[v] for k,v in mapping.items() if k != "output_contract" and k != "adapter_path"):
            raise ValueError("metadata generation configuration mismatch")
    else:
        if meta.get("battle_simulations") != 50 or extra.get("max_act") != 3 or extra.get("max_decisions") != 250 or extra.get("boss_simulation_multiplier") != 2.0:
            raise ValueError("baseline environment/budget mismatch")
    if len(records) != meta.get("n_decisions") or len(records) > 250:
        raise ValueError("trace length differs from metadata")
    invalid = 0
    for i,row in enumerate(records):
        if row.get("decision_index") != i:
            raise ValueError("trace decision indices not contiguous")
        if tuple(row.get(k) for k in ("world_seed", "rollout_index", "policy_seed")) != identity:
            raise ValueError("trace rollout identity mismatch")
        if row.get("phase") != "out_of_combat":
            raise ValueError("unexpected combat decision")
        legal = row.get("legal_actions", [])
        if not row.get("agent", {}).get("valid", True):
            invalid += 1
            if row.get("action_executed") or i != len(records)-1:
                raise ValueError("invalid action was executed or not final")
        elif row.get("action_executed") is not True or row.get("selected_action") not in legal:
            raise ValueError("executed action is not exactly in recorded menu")
    if invalid != meta.get("n_invalid"):
        raise ValueError("invalid-count mismatch")
    stop = meta.get("stopped_reason")
    if stop == "agent_invalid" and (not invalid or not records):
        raise ValueError("agent_invalid outcome lacks fatal invalid trace")
    if stop == "terminal" and (invalid or meta.get("error")):
        raise ValueError("terminal metadata contradicts invalid/error trace")
    scored = stop in ("terminal", "agent_invalid") and not meta.get("error") and not meta.get("undefined_behavior_evoked", False)
    if scored and records and records[-1]["after_state"]["floor"] != meta["final_floor"]:
        raise ValueError("final floor inconsistent with trace")
    return {"world_seed": seed, "stopped_reason": stop, "scored": scored,
            "floor": meta["final_floor"] if scored else None, "n_decisions": len(records),
            "n_invalid": invalid, "n_retry_decisions": sum(r["agent"].get("retries",0)>0 for r in records),
            "outcome": meta.get("outcome"), "error": meta.get("error"),
            "undefined_behavior_evoked": meta.get("undefined_behavior_evoked",False),
            "interface": provenance}


def load_cohort(directory: Path, *, seeds: tuple[int, ...] = SEEDS, model: bool = True) -> dict:
    config = json.loads((directory / "config.json").read_text()) if model else None
    if config is not None:
        validate_config(config,seeds)
        if Path(config["model"]).resolve() == GEMMA26_DIR.resolve() and directory.name != "gemma26_text_constrained":
            raise ValueError("Gemma26 requires the explicitly planned arm name")
    expected = {f"seed_{s}_r0" for s in seeds}
    metas = {p.name.removesuffix(".meta.json") for p in directory.glob("*.meta.json")}
    traces = {p.stem for p in directory.glob("seed_*_r*.jsonl") if not p.name.endswith(".audit.jsonl")}
    if not model:
        # The baseline experiment intentionally contains32worlds; select the
        # prespecified16 comparison worlds, never a data-dependent intersection.
        metas &= expected; traces &= expected
    if metas != expected or traces != expected:
        if (metas|traces)-expected:
            raise ValueError("unexpected rollout identities")
        raise IncompleteCohort(f"needs all{len(seeds)} trace+meta pairs; metadata={len(metas)},traces={len(traces)}")
    episodes=[]; artifact_hashes={}
    for seed in seeds:
        stem=directory/f"seed_{seed}_r0"
        meta=json.loads(stem.with_suffix(".meta.json").read_text())
        episodes.append(validate_episode(meta,read_rows(stem.with_suffix(".jsonl")),seed,config))
        for suffix in (".meta.json", ".jsonl"):
            path=stem.with_suffix(suffix);artifact_hashes[path.name]=file_sha256(path)
    first=episodes[0]["interface"]
    if any(e["interface"] != first for e in episodes):
        raise ValueError("within-arm interface provenance mismatch")
    digest=hashlib.sha256(json.dumps(artifact_hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"name":directory.name,"directory":str(directory),"config":config,"episodes":episodes,"interface":first,
            "cohort_artifact_sha256":digest,"config_sha256":file_sha256(directory/"config.json") if model else None,
            "model_bundle":model_bundle(config) if model else None}


def validate_comparison(first: dict, second: dict) -> dict:
    if {e["world_seed"] for e in first["episodes"]} != {e["world_seed"] for e in second["episodes"]}:
        raise ValueError("comparison requires exact complete world identities")
    if any(first["interface"][k] != second["interface"][k] for k in PUBLIC_KEYS):
        raise ValueError("comparison public simulator/interface mismatch")
    a,b=first["config"],second["config"]
    if a is not None and b is not None:
        cross_model=a["model"] != b["model"]
        if any(a[k] != b[k] for k in GENERATION_KEYS if k != "model"):
            raise ValueError("comparison generation configuration mismatch")
        if cross_model:
            bundles=[model_bundle(c) for c in (a,b)]
            if {x["kind"] for x in bundles} != {"e4b_hf_snapshot","gemma26_ollama_nvfp4_mxfp8"}:
                raise ValueError("undeclared model bundle intervention")
            keys=set(first["interface"])|set(second["interface"])
            if any(first["interface"].get(k) != second["interface"].get(k) for k in keys if k != "chat_template_probe_hash"):
                raise ValueError("cross-model public prompt/interface mismatch")
            return {"kind":"explicit_model_bundle_intervention","bundles":bundles,
                    "changes":"Architecture,weights,quantization,tokenizer and chat template change together; parameter-count effects are not isolated.",
                    "additional_decoder_change":a["constrained"] != b["constrained"],
                    "additional_adapter_change":a["adapter_identity"] != b["adapter_identity"]}
        if first["interface"] != second["interface"]:
            raise ValueError("comparison model prompt/interface mismatch")
        return {"kind":"matched_model_interventions","allowed_changes":["constrained","adapter_identity"],
                "observed_changes":{k:{"first":a[k],"second":b[k]} for k in ("constrained","adapter_identity") if a[k]!=b[k]}}
    return {"kind":"different_policy_families_same_worlds_and_simulator",
            "note":"Model temperature0 is not a matched sampling-temperature intervention against RandomLegalAgent or deterministic heuristic."}


def compare(first: dict, second: dict) -> dict:
    validation=validate_comparison(first,second)
    a={e["world_seed"]:e for e in first["episodes"]};b={e["world_seed"]:e for e in second["episodes"]}
    seeds=sorted(a);paired=[s for s in seeds if a[s]["scored"] and b[s]["scored"]]
    deltas=[b[s]["floor"]-a[s]["floor"] for s in paired]
    low=[];high=[]
    for s in seeds:
        av=min(a[s]["floor"],50) if a[s]["scored"] else None
        bv=min(b[s]["floor"],50) if b[s]["scored"] else None
        low.append((bv if bv is not None else 0)-(av if av is not None else 50))
        high.append((bv if bv is not None else 50)-(av if av is not None else 0))
    return {"first":first["name"],"second":second["name"],"direction":"second minus first",
            "validation":validation,"n_planned_worlds":len(seeds),"n_scored_pairs":len(paired),
            "unscored_pair_worlds":sorted(set(seeds)-set(paired)),
            "mean_floor_delta":statistics.mean(deltas) if deltas else None,
            "bootstrap_ci_95":list(bootstrap_ci(deltas)) if deltas else None,
            "all_world_capped_floor_bounds":{"metric":"min(final_floor,50)","lower":statistics.mean(low),"upper":statistics.mean(high)}}


def summarize(arm: dict) -> dict:
    episodes=arm["episodes"];scored=[e for e in episodes if e["scored"]]
    return {"status":"complete_validated","source_directory":arm["directory"],"config":arm["config"],
            "cohort_artifact_sha256":arm["cohort_artifact_sha256"],"config_sha256":arm["config_sha256"],
            "model_bundle":arm["model_bundle"],
            "n_planned":len(episodes),"n_artifact_complete":len(episodes),"n_scored":len(scored),
            "stop_reasons":dict(Counter(e["stopped_reason"] for e in episodes)),
            "mean_floor_scored":statistics.mean(e["floor"] for e in scored) if scored else None,
            "n_invalid_episodes":sum(e["stopped_reason"]=="agent_invalid" for e in episodes),
            "n_decisions":sum(e["n_decisions"] for e in episodes),"n_invalid_decisions":sum(e["n_invalid"] for e in episodes),
            "n_retry_decisions":sum(e["n_retry_decisions"] for e in episodes),
            "n_victories":sum(e["outcome"]=="GameOutcome.PLAYER_VICTORY" for e in scored),
            "unscored_worlds":[e["world_seed"] for e in episodes if not e["scored"]],
            "episodes":[{k:v for k,v in e.items() if k!="interface"} for e in episodes]}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live",type=Path,default=Path("data/experiments/astra_5sep/live"))
    p.add_argument("--baselines",type=Path,default=Path("data/experiments/astra_5sep/baselines"))
    p.add_argument("--out",type=Path,default=Path("docs/astra_5sep_live_results.json"))
    args=p.parse_args()
    if file_sha256(ROOT/"scripts/experiments/astra_policy_probe.py") != PROBE_HASH:
        raise ValueError("audited probe source hash changed; cannot infer missing environment metadata")
    ready=[];arms={};rejected=False
    for directory in sorted(args.live.iterdir()):
        if not directory.is_dir() or not (directory/"config.json").exists():continue
        try:
            arm=load_cohort(directory);ready.append(arm);arms[directory.name]=summarize(arm)
        except IncompleteCohort as exc:
            arms[directory.name]={"status":"incomplete_unscored","reason":str(exc)}
        except (ValueError,KeyError) as exc:
            rejected=True;arms[directory.name]={"status":"rejected_unscored","reason":str(exc)}
    baselines=[load_cohort(args.baselines/name,model=False) for name in ("random_search50","heuristic_search50")]
    comparisons=[compare(base,arm) for arm in ready for base in baselines]
    comparisons += [compare(a,b) for a,b in combinations(ready,2)]
    report={"kind":"astra_matched_live_v1","world_seeds":list(SEEDS),"arms":arms,
            "baselines":{a["name"]:summarize(a) for a in baselines},"comparisons":comparisons,
            "provenance_note":"Probe metadata omits battle_simulations/git_sha; exact audited runner hash pins search50,maxact3,boss multiplier2. Public interface and native binary hashes must match baseline artifacts.",
            "scoring":"Terminal and agent_invalid stops retain observed final floor. Simulator errors, UB flags and other truncations are unscored and bounded0–50 under the capped metric. Incomplete arms have no outcome summary.",
            "comparability":"Live configs use temperature0,max_tokens256. Free decoding uses that256 cap; constrained decoding instead uses longest canonical candidate token count+1. Historical temperature0.7 cohorts are not compared. Random/heuristic comparisons change policy family; the explicitly registered Gemma26 comparison changes the whole model bundle.",
            "script_sha256":file_sha256(__file__)}
    args.out.parent.mkdir(parents=True,exist_ok=True);tmp=args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");tmp.replace(args.out)
    print(json.dumps({"out":str(args.out),"arms":{k:v['status'] for k,v in arms.items()},"n_comparisons":len(comparisons)}))
    if rejected:raise SystemExit(2)


if __name__=="__main__":main()
