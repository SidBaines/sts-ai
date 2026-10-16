"""Controlled public entities plus the exact archived 128 action features.

The caller applies policy_observation once. Neither encoder filters the menu;
both read their existing public allowlists. Legacy state features are discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sts_policy import encoding, entity_encoding

ENCODER_VERSION = 'public_entities_legacy_actions_controlled_v1'
ENCODER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LEGACY_ENCODER_SHA256 = hashlib.sha256(Path(encoding.__file__).read_bytes()).hexdigest()
LEGACY_ACTION_DIM = 128


@dataclass(frozen=True)
class EncodedControlledObservation:
    entity: entity_encoding.EncodedEntityObservation
    legacy_action_features: np.ndarray

    @property
    def action_indices(self):
        return self.entity.action_indices

    @property
    def entity_ids(self):
        return self.entity.entity_ids

    @property
    def coverage(self):
        # Keep B3 coverage/bucket semantics identical in ON and OFF conditions.
        return self.entity.coverage


def encode_observation(public_obs: Mapping[str, Any]) -> EncodedControlledObservation:
    entity = entity_encoding.encode_observation(public_obs)
    legacy = encoding.encode_observation(public_obs)
    indices = np.asarray([a['index'] for a in public_obs['legal_actions']], dtype=np.int64)
    if (not np.array_equal(entity.action_indices, indices)
            or not np.array_equal(legacy.action_indices, indices)):
        raise ValueError('Controlled entity/legacy action/menu alignment failure')
    if legacy.action_features.shape != (len(indices), LEGACY_ACTION_DIM):
        raise ValueError('Controlled encoder requires exactly 128 legacy action features')
    return EncodedControlledObservation(entity, legacy.action_features)


def collate_observations(examples: list[EncodedControlledObservation], device='cpu') -> dict:
    import torch
    batch = entity_encoding.collate_observations([x.entity for x in examples], device=device)
    features = np.zeros((*batch['action_mask'].shape, LEGACY_ACTION_DIM), dtype=np.float32)
    for i, example in enumerate(examples):
        count = len(example.action_indices)
        if example.legacy_action_features.shape != (count, LEGACY_ACTION_DIM):
            raise ValueError('Controlled legacy features/menu shape mismatch')
        features[i, :count] = example.legacy_action_features
    batch['legacy_action_features'] = torch.as_tensor(features, device=device)
    return batch
