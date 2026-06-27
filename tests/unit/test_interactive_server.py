"""API tests for the Interactive Studio server.

Uses FastAPI's TestClient over a SessionRegistry wired with the fake env + fake
model agent from test_interactive_session, so these run without the simulator or
MLX — gated only on fastapi being installed (`.[app]`). The TestClient import is
deferred to setUp so module collection never needs fastapi.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.support import requires_fastapi
from tests.unit.test_interactive_session import _fake_agent_builder, _fake_env_factory


@requires_fastapi
class InteractiveServerApiTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from sts_ai.interactive.server import create_app
        from sts_ai.interactive.session import SessionRegistry

        self._tmp = tempfile.TemporaryDirectory()
        registry = SessionRegistry(
            cache_dir=str(Path(self._tmp.name) / "interactive"),
            env_factory=_fake_env_factory,
            agent_builder=_fake_agent_builder,
        )
        self.client = TestClient(create_app(registry=registry))

    def tearDown(self):
        self._tmp.cleanup()

    def _create(self, **kw):
        body = {"world_seed": 3, "combat_control": "llm", "framing": "F0"}
        body.update(kw)
        r = self.client.post("/api/sessions", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_config(self):
        cfg = self.client.get("/api/config").json()
        self.assertIn("user", cfg["methods"])
        self.assertIn("model", cfg["methods"])
        self.assertIn(".sts-card", cfg["board_css"])

    def test_create_requires_world_seed(self):
        r = self.client.post("/api/sessions", json={})
        self.assertEqual(r.status_code, 400)

    def test_create_step_and_list(self):
        view = self._create()
        sid = view["session_id"]
        self.assertEqual(view["status"], "ok")
        self.assertIn("board_html", view)

        r = self.client.post(f"/api/sessions/{sid}/step", json={"method": "user", "action_index": 0})
        out = r.json()
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["decision"]["selected"]["description"], "d0 action 0")

        sessions = self.client.get("/api/sessions").json()
        self.assertEqual([s["session_id"] for s in sessions], [sid])
        self.assertEqual(sessions[0]["n_decisions"], 1)

    def test_sample_does_not_commit(self):
        sid = self._create()["session_id"]
        out = self.client.post(f"/api/sessions/{sid}/sample", json={"method": "model", "k": 3}).json()
        self.assertEqual(len(out["candidates"]), 3)
        # not committed
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").json()["decision_index"], 0)

    def test_preview_prompt_reflects_edits(self):
        sid = self._create()["session_id"]
        out = self.client.post(f"/api/sessions/{sid}/preview_prompt", json={"framing": "EDITED FRAME"}).json()
        self.assertEqual(out["status"], "ok")
        self.assertIn("EDITED FRAME", out["prompt"])

    def test_set_framing_and_config(self):
        sid = self._create()["session_id"]
        v = self.client.put(f"/api/sessions/{sid}/framing", json={"framing": "RISKY"}).json()
        self.assertEqual(v["framing"], "RISKY")
        v2 = self.client.put(f"/api/sessions/{sid}/config", json={"temperature": 0.9, "thinking": True}).json()
        self.assertEqual(v2["temperature"], 0.9)
        self.assertTrue(v2["thinking"])

    def test_branch_creates_child(self):
        sid = self._create()["session_id"]
        self.client.post(f"/api/sessions/{sid}/step", json={"method": "first"})
        self.client.post(f"/api/sessions/{sid}/step", json={"method": "first"})
        child = self.client.post(f"/api/sessions/{sid}/branch", json={"at": 1}).json()
        self.assertEqual(child["parent_id"], sid)
        self.assertEqual(child["branch_point"], 1)
        self.assertEqual(child["decision_index"], 1)

    def test_stream_step_sse(self):
        sid = self._create()["session_id"]
        with self.client.stream("GET", f"/api/sessions/{sid}/stream_step?method=model") as resp:
            self.assertEqual(resp.status_code, 200)
            body = "".join(resp.iter_text())
        self.assertIn('"type": "token"', body)
        self.assertIn('"type": "done"', body)
        # the decision was committed
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").json()["decision_index"], 1)

    def test_templates_crud(self):
        r = self.client.post("/api/templates/framings", json={"name": "risky", "text": "Take risks."})
        self.assertEqual(r.status_code, 200, r.text)
        framings = self.client.get("/api/templates/framings").json()
        self.assertEqual(framings["risky"], "Take risks.")
        self.assertIn("neutral", framings)  # built-in present
        self.assertTrue(self.client.delete("/api/templates/framings/risky").json()["deleted"])

    def test_missing_session_404(self):
        self.assertEqual(self.client.get("/api/sessions/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
