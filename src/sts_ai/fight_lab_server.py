"""Local collection browser and isolated model fights using the Studio board."""
from __future__ import annotations

from pathlib import Path
from contextlib import asynccontextmanager

from sts_ai import fight_lab as lab
from sts_ai.rollout_view import to_view
from sts_ai.rollout_view_html import BOARD_CSS, render_decision_html


def create_app(root: Path, *, jobs=None):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, Response
    from sts_ai.fight_lab_jobs import FightJobs, RunBusyError

    root = root.resolve()
    collection = lab.read_json(root / "collection.json")
    runner = jobs if jobs is not None else FightJobs(root, collection)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            runner.close()

    app = FastAPI(title="Single-fight lab", lifespan=lifespan)

    @app.post("/api/runs", status_code=202)
    def start_run(body: dict = Body(...)):
        state_id = body.get("state_id")
        if not isinstance(state_id, str):
            raise HTTPException(400, "state_id is required")
        try:
            return runner.start(state_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RunBusyError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/runs")
    def list_runs():
        return runner.list()

    @app.get("/api/runs/{job_id}")
    def get_run(job_id: str):
        try:
            return runner.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown run") from exc

    @app.post("/api/runs/{job_id}/cancel")
    def cancel_run(job_id: str):
        try:
            return runner.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown run") from exc

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "interactive/static/fight_lab.html")

    @app.get("/app.js")
    def javascript():
        return FileResponse(Path(__file__).parent / "interactive/static/fight_lab.js")

    @app.get("/api/collection")
    def info():
        return {"sources": collection["sources"], "reports": collection["reports"],
                "config": collection["config"], "n_states": len(collection["states"]),
                "enemies": sorted({e for s in collection["states"] for e in s["enemies"]}),
                "board_css": BOARD_CSS}

    @app.get("/api/states")
    def states(phase: str = "combat", enemy: str = "", battle_start: bool = False,
               hp_below: float | None = None, act: int | None = None,
               floor_min: int | None = None, floor_max: int | None = None,
               turn: int | None = None, split: str = "", download: bool = False,
               state_id: str | None = None):
        try:
            filters = lab.StateFilter(phase, enemy, battle_start, hp_below, act,
                                      floor_min, floor_max, turn, split)
            selected = lab.selection(collection, filters, [state_id] if state_id else None)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if download:
            import json
            return Response(json.dumps(selected, indent=2), media_type="application/json",
                            headers={"Content-Disposition": 'attachment; filename="selection.json"'})
        return selected

    def trajectories():
        found = {}
        for path in sorted((root / "runs").glob("*/*.jsonl")):
            meta_path = path.with_suffix(".meta.json")
            if (not meta_path.exists() or root not in path.resolve().parents
                    or root not in meta_path.resolve().parents):
                continue
            meta = lab.read_json(meta_path)
            identity = meta.get("extra", {}).get("fight_lab", {})
            source = next((s for s in collection["states"] if s["state_id"] == identity.get("state_id")), None)
            if source is None or source["public_sha256"] != identity.get("start_public_sha256"):
                continue
            key = str(path.relative_to(root))
            found[key] = {"path": key, "state_id": source["state_id"],
                          "label": path.parent.name, "stopped_reason": meta["stopped_reason"],
                          "metrics": meta["extra"]["local_task"]["metrics"]}
        return found

    @app.get("/api/trajectories")
    def list_trajectories():
        return list(trajectories().values())

    @app.get("/api/view")
    def view(source: str, index: int = 0):
        try:
            if source in {s["path"] for s in collection["sources"]}:
                records = lab.source_records(root, collection, source)
                outcome = next(s["stopped_reason"] for s in collection["sources"] if s["path"] == source)
            elif source in trajectories():
                path = (root / source).resolve()
                if root not in path.parents:
                    raise ValueError("trajectory path escapes collection")
                records = lab.read_records(path)
                outcome = trajectories()[source]["stopped_reason"]
            else:
                raise ValueError("unknown trace")
            if not 0 <= index < len(records):
                raise ValueError("decision index outside trace")
            record = records[index]
            return {"count": len(records), "index": index, "record": record,
                    "outcome": outcome, "board_html": render_decision_html(to_view(record)),
                    "source_state": next((s for s in collection["states"]
                                          if s["source"] == source and s["index"] == index), None)}
        except (ValueError, StopIteration) as exc:
            raise HTTPException(400, str(exc)) from exc

    return app
