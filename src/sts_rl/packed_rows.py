"""Lossless columnar storage for joint PPO's immutable choice rows.

The Sequence interface also accepts historical list-of-dict consumers. Only
public float32 features are stored; scalar likelihoods/advantages retain Python
float precision. Pickle/torch.save serialize a handful of arrays, not an array
and dictionary object for every decision. This is an internal buffer format,
not a change to public observations or episode receipts.
"""
from collections.abc import Sequence
import operator

import numpy as np


class PackedRows(Sequence):
    format = "joint_choice_columns_v1"
    integers = ("position", "seed", "act", "action_index")
    floats = ("logp", "value", "return", "advantage")

    def __init__(self, state, actions, offsets, columns, phases, rooms, floors):
        self.state, self.actions, self.offsets = state, actions, offsets
        self.columns, self.phases, self.rooms, self.floors = columns, phases, rooms, floors

    @classmethod
    def from_rows(cls, rows):
        if isinstance(rows, cls):
            return rows
        rows = list(rows)
        n = len(rows)
        state = np.stack([r["state"] for r in rows]) if n else np.empty((0, 256), np.float32)
        actions = np.concatenate([r["actions"] for r in rows]) if n else np.empty((0, 128), np.float32)
        if state.dtype != np.float32 or actions.dtype != np.float32:
            raise ValueError("Packed joint features must already be float32")
        if state.shape != (n, 256) or actions.ndim != 2 or actions.shape[1] != 128:
            raise ValueError("Invalid public feature dimensions")
        counts = np.array([len(r["actions"]) for r in rows], np.int64)
        offsets = np.concatenate([np.zeros(1, np.int64), counts.cumsum()])
        columns = {k: np.array([r[k] for r in rows], np.int64) for k in cls.integers}
        columns.update({k: np.array([r[k] for r in rows], np.float64) for k in cls.floats})
        if n and (np.any(counts < 2) or np.any(columns["position"] < 0)
                  or np.any(columns["position"] >= counts)):
            raise ValueError("Only legal choice rows may enter a joint buffer")
        phases = np.array([r["phase"] == "combat" for r in rows], bool)
        if any(r["phase"] not in ("combat", "ooc") for r in rows):
            raise ValueError("Invalid joint phase")
        return cls(state, actions, offsets, columns, phases,
                   tuple(r["room"] for r in rows), tuple(r["floor"] for r in rows))

    @classmethod
    def concatenate(cls, parts):
        parts = [cls.from_rows(p) for p in parts if len(p)]
        if not parts:
            return cls.from_rows([])
        if len(parts) == 1:
            return parts[0]
        starts = np.cumsum([0] + [len(p.actions) for p in parts[:-1]])
        offsets = np.concatenate([*(p.offsets[:-1] + start for p, start in zip(parts, starts)),
                                  np.array([sum(len(p.actions) for p in parts)], np.int64)])
        return cls(np.concatenate([p.state for p in parts]), np.concatenate([p.actions for p in parts]),
                   offsets, {k: np.concatenate([p.columns[k] for p in parts]) for k in parts[0].columns},
                   np.concatenate([p.phases for p in parts]),
                   sum((p.rooms for p in parts), ()), sum((p.floors for p in parts), ()))

    def __len__(self):
        return len(self.state)

    def __getstate__(self):
        # Torch's default pickle protocol encodes NumPy payloads as large
        # temporary byte strings. Tensor storages instead stream through
        # torch.save and use multiprocessing's shared-memory transport.
        # The in-process API remains NumPy and no numeric conversion occurs.
        import torch
        state = {"schema": self.format, **self.__dict__}
        for key in ("state", "actions", "offsets", "phases"):
            state[key] = torch.from_numpy(state[key])
        state["columns"] = {k: torch.from_numpy(v) for k, v in self.columns.items()}
        return state

    def __setstate__(self, state):
        import torch
        state = dict(state)
        # Accept the initial array-only v1 prototype as well as tensor storage.
        if state.pop("schema", self.format) != self.format:
            raise ValueError("Unsupported packed joint buffer schema")
        def array(value):
            return value.detach().cpu().numpy() if torch.is_tensor(value) else value
        for key in ("state", "actions", "offsets", "phases"):
            state[key] = array(state[key])
        state["columns"] = {k: array(v) for k, v in state["columns"].items()}
        self.__dict__.update(state)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return {"state": self.state[index],
                "actions": self.actions[self.offsets[index]:self.offsets[index + 1]],
                **{k: v[index].item() for k, v in self.columns.items()},
                "phase": "combat" if self.phases[index] else "ooc",
                "room": self.rooms[index], "floor": self.floors[index]}

    @property
    def nbytes(self):
        return sum(x.nbytes for x in [self.state, self.actions, self.offsets, self.phases,
                                     *self.columns.values()])


def concatenate_rows(parts):
    if any(isinstance(p, PackedRows) for p in parts):
        return PackedRows.concatenate(parts)
    return [r for p in parts for r in p]
