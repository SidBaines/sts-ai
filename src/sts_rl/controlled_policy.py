"""Controlled replacements under the unchanged archived StudyPolicy contract.

Only public, already-filtered observations reach the encoder. All imports of the
new model are lazy so original controls remain usable before integration.
"""
from collections.abc import Mapping

from sts_rl.fresh_study import StudyPolicy
from sts_rl.recording import file_identity

CONTROLLED_KIND = 'entity_controlled_v1'


def replacement_from_payload(payload, phase):
    """Load one phase head, enforcing its real kind, source hashes and phase."""
    if not isinstance(payload, dict) or not isinstance(payload.get('metadata'), dict):
        raise ValueError('Replacement requires checkpoint metadata')
    meta = payload['metadata']
    kind = meta.get('kind')
    if kind in (CONTROLLED_KIND, 'entity_fresh_v1'):
        config = meta.get('model_config')
        if not isinstance(config, Mapping) or config.get('phase') != phase:
            raise ValueError('Replacement requires matching explicit entity phase')
        if kind == CONTROLLED_KIND:
            if type(config.get('legacy_action_features')) is not bool:
                raise ValueError('Controlled replacement requires boolean legacy_action_features')
            from sts_policy.controlled_entity import policy_from_payload
        else:
            from sts_policy.entity_model import policy_from_payload
        model = policy_from_payload(payload)  # authoritative source/vocabulary hash checks
        actual = model.config.phase
        if meta.get('phase', actual) != actual:
            raise ValueError('Replacement metadata phase mismatch')
    elif kind in (None, 'candidate') and meta.get('encoder_version') == 'public_candidate_v2':
        from sts_policy.model import CandidateActorCritic
        from sts_policy.model_agent import policy_from_payload
        model = policy_from_payload(payload)
        if not isinstance(model, CandidateActorCritic):
            raise ValueError('Replacement must be a single candidate phase head')
        actual = meta.get('phase')
    else:
        raise ValueError('Unknown replacement kind/encoder_version')
    if actual != phase or meta.get('config', {}).get('phase', phase) != phase:
        raise ValueError('Replacement phase mismatch')
    StudyPolicy._validate_finite(model)
    return model.to('cpu').eval()


class ControlledStudyPolicy(StudyPolicy):
    """Original champion and filtering/argmax, with controlled phase inference."""
    def __init__(self, champion, combat=None, ooc=None):
        import torch
        super().__init__(champion)  # never ask the archived loader to parse a new kind
        self._controlled_phases = set()
        for phase, path in (('combat', combat), ('ooc', ooc)):
            if path is None:
                continue
            identity = file_identity(path)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            model = replacement_from_payload(payload, phase)
            if file_identity(path) != identity:
                raise RuntimeError('Replacement changed during load')
            self.replacements[phase] = model
            self.identity[phase] = identity
            if payload['metadata'].get('kind') == CONTROLLED_KIND:
                self._controlled_phases.add(phase)

    def _logits(self, observation):
        import torch
        phase = 'combat' if observation['phase'] == 'combat' else 'ooc'
        if phase not in self._controlled_phases:
            return super()._logits(observation)
        from sts_policy.controlled_encoding import encode_observation, collate_observations
        # choose_action/distribution already derived inventory and filtered once.
        encoded = encode_observation(observation)
        indices = [a['index'] for a in observation['legal_actions']]
        if (any(type(i) is not int for i in indices) or len(set(indices)) != len(indices) or
                encoded.action_indices.tolist() != indices):
            raise ValueError('Controlled encoder action/menu index mismatch')
        batch = collate_observations([encoded], device='cpu')
        n = len(indices)
        if (batch['legacy_action_features'].shape != (1, n, 128) or
                batch['action_mask'].shape != (1, n) or not bool(batch['action_mask'].all())):
            raise ValueError('Controlled singleton action feature/mask shape mismatch')
        if any(value.device.type != 'cpu' for value in batch.values() if torch.is_tensor(value)):
            raise ValueError('Controlled inference requires CPU')
        with self._lock, torch.inference_mode():
            logits = self.replacements[phase].actor_logits(batch)
            if logits.shape != (1, n) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Controlled singleton logits/menu shape or finite check failed')
            return logits[0].detach().clone()


def policy_factory(champion, combat=None, ooc=None, *, original_policy=StudyPolicy):
    """Use the same factory in panel parent and spawned native-child cache."""
    import torch
    identities, controlled = [], False
    for path in (combat, ooc):
        if path is None:
            continue
        identity = file_identity(path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if file_identity(path) != identity:
            raise RuntimeError('Replacement changed during dispatch')
        if not isinstance(payload, dict) or not isinstance(payload.get('metadata'), dict):
            raise ValueError('Replacement requires checkpoint metadata')
        controlled |= payload['metadata'].get('kind') == CONTROLLED_KIND
        identities.append(identity)
    policy = (ControlledStudyPolicy if controlled else original_policy)(champion, combat=combat, ooc=ooc)
    for identity in identities:
        if file_identity(identity['path']) != identity:
            raise RuntimeError('Replacement changed during policy construction')
    return policy
