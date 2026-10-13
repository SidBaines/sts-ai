"""Experimental privileged teacher stability RPC, separate from policy inputs."""
from __future__ import annotations

import multiprocessing

from .workers import CurriculumEpisode, CurriculumWorker

PANEL_VERSION = "search_draw_stability_v1"


class PanelEpisode(CurriculumEpisode):
    def teacher_panel(self, simulations: int = 1000) -> dict:
        if self.finished or self.observation["phase"] != "combat" or simulations < 1:
            raise ValueError("teacher stability panel requires a live combat and positive budget")
        results=[]
        conditions=[(s,None) for s in (1,2,3)]+[(1,d) for d in (1,2,3)]
        for search_seed,draw_seed in conditions:
            try:
                raw=self.episode.env.search_best_combat_action(simulations,
                    search_seed=search_seed,draw_order_seed=draw_seed)
            except ValueError as error:
                if draw_seed is not None and "unsupported for draw-index card selection" in str(error):
                    results.append({"search_seed":search_seed,"draw_order_seed":draw_seed,"status":"abstain",
                                    "reason":"draw-index identity cannot be preserved"})
                    continue
                raise
            index=int(self.episode.env.display_index_for_bits(int(raw["bits"])))
            roots=[]
            for edge in raw["root_edges"]:
                if edge["valid"]:
                    roots.append({"index":int(self.episode.env.display_index_for_bits(int(edge["bits"]))),
                        "visits":int(edge["visits"]),"mean_evaluation":float(edge["mean_evaluation"])})
            results.append({"status":"ok","index":index,"search_seed":search_seed,"draw_order_seed":draw_seed,
                "selection_method":str(raw["selection_method"]),"winning_sequence_found":bool(raw["winning_sequence_found"]),
                "predicted_player_hp":int(raw["predicted_player_hp"]),"root_simulations":int(raw["root_simulations"]),
                "root_edges":roots})
        return {"panel_version":PANEL_VERSION,"teacher_uses_private_state":True,"simulations":simulations,
                "primary_index":results[0]["index"],"results":results}


def serve_panel(connection,config):
    episode=PanelEpisode(config)
    try:
        while True:
            method,args=connection.recv()
            if method=="close":
                break
            try:
                connection.send(("ok",getattr(episode,method)(*args)))
            except Exception as error:
                connection.send(("error",{"type":type(error).__name__,"message":str(error)}))
                break
    except (EOFError,BrokenPipeError):
        pass
    finally:
        connection.close()


class PanelWorker(CurriculumWorker):
    """Same bounded/disposable worker contract with a teacher-only extra RPC."""
    def _launch(self):
        context=multiprocessing.get_context("spawn")
        self._connection,child=context.Pipe()
        self._process=context.Process(target=serve_panel,args=(child,self.config),name="sts-teacher-panel",daemon=True)
        self._process.start(); child.close()

    def teacher_panel(self,simulations=1000):
        return self._call("teacher_panel",simulations)
