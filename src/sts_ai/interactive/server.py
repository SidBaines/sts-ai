"""FastAPI backend for the Interactive Rollout Studio.

Thin HTTP/SSE shell over `SessionRegistry` (session.py). FastAPI is imported
lazily inside `create_app` so `sts_ai.interactive` imports without the optional
`app` extra. The static SPA is served from `static/` via FileResponse (no extra
deps). Model generation streams over SSE; Starlette iterates a *sync* generator
in a threadpool, so the blocking model call never stalls the event loop and a
client disconnect closes the generator (cancel).

Run via `scripts/interactive_app.py` (uvicorn).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sts_ai.interactive.session import ALL_METHODS, SessionRegistry
from sts_ai.rollout_view_html import BOARD_CSS

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    *,
    cache_dir: str = "data/interactive",
    model_backend: str = "mlx",
    model_id: str | None = None,
    registry: SessionRegistry | None = None,
):
    """Build the FastAPI app. ``registry`` may be injected (tests); otherwise one
    is constructed over ``cache_dir`` with the given default model backend."""
    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, StreamingResponse

    reg = registry or SessionRegistry(
        cache_dir=cache_dir, default_model_backend=model_backend, default_model_id=model_id
    )

    app = FastAPI(title="StS Interactive Rollout Studio")

    def _session(session_id: str):
        try:
            return reg.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")

    # --- static SPA -------------------------------------------------------
    @app.get("/")
    def index():
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/static/{name}")
    def static_file(name: str):
        target = (_STATIC_DIR / name).resolve()
        if _STATIC_DIR.resolve() not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(target)

    # --- config -----------------------------------------------------------
    @app.get("/api/config")
    def config():
        return {
            "methods": list(ALL_METHODS),
            "combat_controls": ["llm", "search"],
            "default_model_backend": reg.default_model_backend,
            "default_model_id": reg.default_model_id,
            "mlx_available": _module_installed("mlx_lm"),
            "vllm_available": _module_installed("vllm"),
            "cache_dir": str(reg.store.root),
            "board_css": BOARD_CSS,
        }

    # --- sessions ---------------------------------------------------------
    @app.get("/api/sessions")
    def list_sessions():
        return reg.list_sessions()

    @app.post("/api/sessions")
    def create_session(body: dict = Body(default={})):
        if "world_seed" not in body:
            raise HTTPException(status_code=400, detail="world_seed is required")
        try:
            session = reg.create(
                world_seed=int(body["world_seed"]),
                ascension=int(body.get("ascension", 0)),
                combat_control=body.get("combat_control", "llm"),
                max_act=int(body.get("max_act", 3)),
                battle_simulations=int(body.get("battle_simulations", 2000)),
                framing=body.get("framing"),
                prompt_template=body.get("prompt_template"),
                use_advanced_template=bool(body.get("use_advanced_template", False)),
                model_backend=body.get("model_backend"),
                model_id=body.get("model_id"),
                temperature=float(body.get("temperature", 0.2)),
                max_tokens=int(body.get("max_tokens", 4096)),
                thinking=bool(body.get("thinking", False)),
                label=body.get("label", ""),
            )
        except Exception as exc:  # noqa: BLE001 - surface sim/build errors to the client
            raise HTTPException(status_code=500, detail=f"could not create session: {exc}")
        return session.current_view()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        return _session(session_id).current_view()

    @app.get("/api/sessions/{session_id}/view")
    def view_at(session_id: str, at: int = Query(default=-1)):
        session = _session(session_id)
        return session.current_view() if at < 0 else session.view_at(at)

    @app.post("/api/sessions/{session_id}/step")
    def step(session_id: str, body: dict = Body(default={})):
        session = _session(session_id)
        method = body.get("method", "first")
        action_index = body.get("action_index")
        action_index = None if action_index is None else int(action_index)
        n = int(body.get("n", 1))
        if n > 1:
            return session.sample_n(method, n=n, action_index=action_index)
        return session.step(method, action_index=action_index)

    @app.get("/api/sessions/{session_id}/stream_step")
    def stream_step(
        session_id: str,
        method: str = Query(default="model"),
        action_index: int | None = Query(default=None),
    ):
        session = _session(session_id)

        def _events():
            try:
                for event in session.stream_step(method, action_index=action_index):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'type': 'done', 'status': 'error', 'error': str(exc)})}\n\n"

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{session_id}/sample")
    def sample(session_id: str, body: dict = Body(default={})):
        session = _session(session_id)
        return session.sample_candidates(
            body.get("method", "model"),
            k=int(body.get("k", 1)),
            temperature=body.get("temperature"),
        )

    @app.post("/api/sessions/{session_id}/branch")
    def branch(session_id: str, body: dict = Body(default={})):
        if "at" not in body:
            raise HTTPException(status_code=400, detail="branch point 'at' is required")
        try:
            child = reg.branch(session_id, int(body["at"]), label=body.get("label"))
        except (KeyError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return child.current_view()

    @app.post("/api/sessions/load")
    def load_session(body: dict = Body(default={})):
        sid = body.get("session_id")
        if not sid:
            raise HTTPException(status_code=400, detail="session_id is required")
        try:
            return reg.load(sid).current_view()
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no session {sid!r}")

    @app.post("/api/sessions/{session_id}/save")
    def save_session(session_id: str):
        _session(session_id)._save()
        return {"status": "saved"}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        return {"deleted": reg.delete(session_id)}

    # --- framing / prompt editing ----------------------------------------
    @app.put("/api/sessions/{session_id}/framing")
    def set_framing(session_id: str, body: dict = Body(default={})):
        session = _session(session_id)
        session.set_framing(
            framing=body.get("framing"),
            template=body.get("template"),
            use_advanced=body.get("use_advanced"),
        )
        return session.current_view()

    @app.put("/api/sessions/{session_id}/config")
    def set_model_config(session_id: str, body: dict = Body(default={})):
        session = _session(session_id)
        session.set_model_config(
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            thinking=body.get("thinking"),
            model_id=body.get("model_id"),
            model_backend=body.get("model_backend"),
        )
        return session.current_view()

    @app.post("/api/sessions/{session_id}/preview_prompt")
    def preview_prompt(session_id: str, body: dict = Body(default={})):
        session = _session(session_id)
        return session.preview_prompt(
            framing=body.get("framing"),
            template=body.get("template"),
            use_advanced=body.get("use_advanced"),
        )

    # --- templates --------------------------------------------------------
    @app.get("/api/templates/framings")
    def list_framings():
        return reg.templates.list_framings()

    @app.post("/api/templates/framings")
    def save_framing(body: dict = Body(default={})):
        try:
            reg.templates.save_framing(body["name"], body.get("text", ""))
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"status": "saved", "framings": reg.templates.list_framings()}

    @app.delete("/api/templates/framings/{name}")
    def delete_framing(name: str):
        try:
            return {"deleted": reg.templates.delete_framing(name)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/templates/prompts")
    def list_prompts():
        return reg.templates.list_prompt_templates()

    @app.post("/api/templates/prompts")
    def save_prompt(body: dict = Body(default={})):
        try:
            reg.templates.save_prompt_template(body["name"], body.get("text", ""))
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"status": "saved", "prompts": reg.templates.list_prompt_templates()}

    @app.delete("/api/templates/prompts/{name}")
    def delete_prompt(name: str):
        try:
            return {"deleted": reg.templates.delete_prompt_template(name)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def _module_installed(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False
