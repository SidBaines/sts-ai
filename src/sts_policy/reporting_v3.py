"""Curated, read-only study reporting without importing learning dependencies."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

REPORT_VERSION = "nonllm_learning_report_v3_v1"
MAX_DOCUMENT_BYTES = 32 * 1024**2


def finite_document(value, path: str, unavailable: list[str]):
    if isinstance(value,float) and not math.isfinite(value):
        unavailable.append(path)
        return None
    if isinstance(value,dict):
        return {k:finite_document(v,f"{path}.{k}",unavailable) for k,v in value.items()}
    if isinstance(value,list):
        return [finite_document(v,f"{path}[{i}]",unavailable) for i,v in enumerate(value)]
    return value


def read_json(path: Path, root: Path) -> dict:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("report source escapes study root")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError("report document exceeds size limit")
    return json.loads(path.read_text())


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    scale = 1+z*z/total
    center = (p+z*z/(2*total))/scale
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/scale
    return [max(0.,center-radius),min(1.,center+radius)]


def outcome_summary(result: dict) -> dict:
    valid = [e for e in result.get("episodes",[]) if e.get("terminated") and not e.get("truncated")]
    fresh = [e for e in valid if e.get("start_family")=="fresh"]
    conditional = [e for e in valid if e.get("start_family")!="fresh"]
    wins = lambda rows: sum(e.get("task_outcome")=="victory" for e in rows)
    unique = len({e["world_seed"] for e in fresh})==len(fresh)
    identity = hashlib.sha256(json.dumps(sorted(result.get("task_ids",[]))).encode()).hexdigest()
    return {"cohort_identity":identity,"requested":result.get("requested_tasks"),"finished":len(result.get("episodes",[])),
        "complete":result.get("cohort_completed",False),"faults":len(result.get("faults",[])),
        "truncated":sum(bool(e.get("truncated")) for e in result.get("episodes",[])),"unstarted":result.get("unstarted_tasks"),
        "fresh_completed":len(fresh),"fresh_wins":wins(fresh),"fresh_interval_95":wilson(wins(fresh),len(fresh)) if unique else None,
        "act1":sum(e.get("max_act_reached",0)>=2 for e in fresh),"act2":sum(e.get("max_act_reached",0)>=3 for e in fresh),
        "fresh_mean_floor":sum(e["floor"] for e in fresh)/len(fresh) if fresh else None,
        "conditional_completed":len(conditional),"conditional_wins":wins(conditional),
        "conditional_actor_bearing_wins":sum(e.get("task_outcome")=="victory" and e.get("actor_choice_rows",0)>0 for e in conditional),
        "survival":[{"floor":floor,"reached":sum(e["floor"]>=floor for e in fresh),"denominator":len(fresh)} for floor in range(1,52)],
        "loss_enemies":dict(Counter(", ".join(n for n in e["fights"][-1].get("enemies",[]) if not n.startswith("INVALID")) for e in valid
            if e.get("task_outcome")=="defeat" and e.get("fights"))),
        "elapsed_seconds":result.get("elapsed_seconds"),"primitive_steps":result.get("primitive_steps"),
        "steps_per_second":result.get("primitive_steps_per_second"),"timings":result.get("timings",{}),
        "controller":result.get("combat_controller"),"collector_version":result.get("collector_version")}


def paired(reference: dict, candidate: dict) -> dict:
    def valid(rows):
        return {e["task_id"]:e for e in rows.get("episodes",[]) if e.get("terminated") and not e.get("truncated")}
    old,new = valid(reference),valid(candidate)
    common = sorted(set(old)&set(new))
    result = {"common_valid":len(common),"reference_valid":len(old),"candidate_valid":len(new)}
    for key,predicate in (("wins",lambda e:e.get("task_outcome")=="victory"),
                          ("act1",lambda e:e.get("max_act_reached",0)>=2),
                          ("act2",lambda e:e.get("max_act_reached",0)>=3)):
        gains = sum(not predicate(old[k]) and predicate(new[k]) for k in common)
        losses = sum(predicate(old[k]) and not predicate(new[k]) for k in common)
        result[key] = {"gains":gains,"losses":losses,"net":gains-losses}
    result["mean_floor_delta"] = sum(new[k]["floor"]-old[k]["floor"] for k in common)/len(common) if common else None
    return result


def build_report(root: Path) -> dict:
    root = root.resolve()
    errors,sources,evaluations,runs,nonfinite,teacher_controls = [],[],[],[],[],[]
    raw_results = {}
    def load(path):
        try:
            data = read_json(path,root)
            unavailable = []
            data = finite_document(data,"",unavailable)
            if unavailable:
                nonfinite.append({"path":str(path.relative_to(root)),"fields":unavailable,
                                  "display":"unavailable; original source preserved"})
            sources.append({"path":str(path.relative_to(root)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            return data
        except (OSError,ValueError) as exc:
            errors.append({"path":str(path.relative_to(root)),"error":str(exc)})
            return None
    paths = list((root/"evals").glob("*.json"))+list((root/"pilots").glob("*/evaluation_*.json"))
    for path in sorted(paths):
        if "final" in path.name.lower() or path.name.endswith("_paired.json"):
            continue
        doc = load(path)
        if doc is None:
            continue
        results = doc.get("results",[doc.get("result",doc)])
        for i,result in enumerate(results):
            if "episodes" not in result:
                continue
            name = str(path.relative_to(root))+ (f":{i}" if len(results)>1 else "")
            raw_results[name] = result
            evaluations.append({"name":name,"checkpoint":doc.get("checkpoint",result.get("checkpoint")),
                "checkpoint_sha256":doc.get("checkpoint_sha256",result.get("checkpoint_sha256")),
                "summary":outcome_summary(result)})
    references = {}
    for row in evaluations:
        if row["name"] in ("evals/common_initial_context_v2_dev192.json","evals/confirmation331_baseline_dev192.json"):
            references[row["summary"]["cohort_identity"]] = row["name"]
    for row in evaluations:
        baseline = references.get(row["summary"]["cohort_identity"])
        if baseline:
            row.update(reference=baseline,paired=paired(raw_results[baseline],raw_results[row["name"]]))
    for folder in sorted((root/"pilots").glob("*")):
        if not folder.is_dir() or not (folder/"status.json").exists():
            continue
        status = load(folder/"status.json")
        if status is None:
            continue
        updates = []
        for path in sorted(folder.glob("update_*.json")):
            row = load(path)
            if row is not None:
                updates.append({k:row.get(k) for k in ("chunk","trainer_version","actor_exposures","critic_exposures",
                    "actor_steps","critic_steps","unique_actor_rows_used","unique_critic_rows_used","elapsed_seconds",
                    "actor_grad_norm_before_clip","actor_grad_norm_after_clip","critic_grad_norm_before_clip",
                    "critic_grad_norm_after_clip","buffer_drift","rehearsal_drift","calibration_before","calibration_after",
                    "rows_by_phase_act_choice","advantage_before_normalization","collection_summary","stop_reason")})
        def total(key):
            values = [u[key] for u in updates]
            return sum(values) if values and all(v is not None for v in values) else None
        runs.append({"name":str(folder.relative_to(root)),"kind":"strategic PPO","status":status.get("status"),
            "updated_utc":status.get("updated_utc"),"learner":status.get("learner_checkpoint"),
            "learner_sha256":status.get("learner_sha256"),"validated_champion":status.get("validated_champion"),
            "config":status.get("config"),"trainer_config":status.get("trainer_config"),"updates":updates,
            "actor_exposures":total("actor_exposures"),"critic_exposures":total("critic_exposures"),
            "actor_steps":total("actor_steps"),"critic_steps":total("critic_steps"),
            "unique_note":"Unique-row counts apply within each collected buffer; sums are not globally unique identities."})
    for folder in sorted((root/"combat").glob("*")):
        metrics = []
        for path in sorted(folder.glob("metrics_*.json")):
            row = load(path)
            if row is not None:
                metrics.append({"step":int(path.stem.split("_")[-1]),**row})
        if metrics:
            last = metrics[-1]
            complete = load(folder/"completion.json") if (folder/"completion.json").exists() else None
            launch = load(folder/"launch.json") if (folder/"launch.json").exists() else None
            runs.append({"name":str(folder.relative_to(root)),"kind":"native combat BC",
                "status":"fit complete / frozen base verified" if complete else "saved checkpoint; completion marker unavailable",
                "config":launch.get("config") if launch else last.get("config"),
                "metrics":metrics,"actor_exposures":sum(last["actual_exposures_by_act"].values()),
                "unique_rows_used":last["unique_rows_used"],"elapsed_seconds":last["elapsed_seconds"],
                "note":"Teacher likelihood and agreement are supervision diagnostics, not game competence."})
    for folder in sorted((root/"teacher_controls").glob("*")):
        baseline_path = folder/"learned_baseline.json"
        if not baseline_path.exists():
            continue
        reference = load(baseline_path)
        if reference is None:
            continue
        rows = [{"name":"learned combat", "summary":outcome_summary(reference)}]
        for path in sorted(folder.glob("native_*.json")):
            result = load(path)
            if result is not None:
                rows.append({"name":path.stem,"summary":outcome_summary(result),"paired":paired(reference,result)})
        teacher_controls.append({"name":str(folder.relative_to(root)),"rows":rows})
    teacher_audits = []
    for name in ("policy_dagger_original_68/label_audit.json","policy_teacher_panel_68/stability_audit.json"):
        path = root/"banks"/name
        if path.exists():
            audit = load(path)
            if audit is not None:
                teacher_audits.append({"name":name,"audit":audit})
    return {"report_version":REPORT_VERSION,"generated_utc":datetime.now(timezone.utc).isoformat(),"root":str(root),
        "selection_note":"Promotion is explicit in condition status. Repeated DEV tuning is descriptive; final cohorts stay unopened.",
        "evaluations":evaluations,"runs":runs,"sources":sources,"errors":errors,"nonfinite_metrics":nonfinite,
        "teacher_controls":teacher_controls,"teacher_audits":teacher_audits,
        "coverage_note":"Curated eval, learner and teacher-control documents plus named audit summaries only; partial journals, teacher rows, checkpoints and final data are not read."}
