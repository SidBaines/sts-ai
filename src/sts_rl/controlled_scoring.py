"""Shared strict checkpoint and frozen-selection validation; no fitting/inference."""

from sts_rl.controlled_policy import CONTROLLED_KIND, replacement_from_payload
from sts_rl.fresh_panel_comparison import _hash, _require, _same
from sts_rl.fresh_scoring import SELECTION_CRITERION


def checkpoint_details(payload, *, architecture=None, phase=None, deferred=False):
    """Validate actual metadata/model, never infer an arm from a filename."""
    from sts_policy.entity_encoding import VOCABULARY_SHA256
    from sts_rl.fresh_training import FILTERS, checkpoint_training_condition
    meta = payload['metadata']
    config = meta['config']
    actual = config['architecture']
    actual_phase = config['phase']
    _require(actual in ('old', 'new', 'controlled') and actual_phase in ('combat', 'ooc'),
             'Invalid checkpoint architecture/phase')
    _require((architecture is None or actual == architecture) and (phase is None or actual_phase == phase),
             'Checkpoint differs from frozen architecture/phase')
    kind = meta.get('kind')
    _require((actual == 'controlled' and kind == CONTROLLED_KIND) or
             (actual == 'new' and kind == 'entity_fresh_v1') or
             (actual == 'old' and kind in (None, 'candidate') and
              meta.get('encoder_version') == 'public_candidate_v2'), 'Checkpoint kind/architecture mismatch')
    _require(meta.get('initialization') == 'from_scratch' and 'parent_checkpoint' in meta and
             meta['parent_checkpoint'] is None and _hash(meta.get('data_identity')),
             'Require from-scratch checkpoint without a weight parent and with data identity')
    replacement_from_payload(payload, actual_phase)
    condition = checkpoint_training_condition(payload)
    _require(condition['phase'] == actual_phase and condition['fit_size'] in ('5000', '20000', 'full'),
             'Invalid checkpoint training condition')
    if actual == 'controlled':
        _require(type(condition.get('legacy_action_features')) is bool and
                 condition['legacy_action_features'] == meta['model_config']['legacy_action_features'],
                 'Controlled training condition/model feature mismatch')
    if deferred:
        _require(config.get('defer_test') is True and meta.get('filters') == FILTERS and
                 meta.get('vocabulary_sha256') == VOCABULARY_SHA256,
                 'Checkpoint is not a compatible deferred selection')
        _require(type(payload['step']) is int and payload['step'] == meta['training_state']['best_step'],
                 'Checkpoint is not the validation-selected best step')
    return meta, condition


def frozen_selection(path, inputs):
    """No raw labels are read here. Source file bytes are only hashed."""
    from sts_policy.training import load_checkpoint
    from sts_rl.fresh_training import FILTERS, digest
    selection, identity = inputs.read(path)
    _require(_same(selection.get('selection_criterion'), SELECTION_CRITERION),
             'Selection requires minimum validation_soft_ce with lower_lr tie break')
    checkpoint = inputs.verify(selection['selected_checkpoint'])
    manifest, _ = inputs.read(selection['manifest']['path'], selection['manifest'])
    _require(manifest.get('schema') == 'fresh_data_manifest_v2' and manifest.get('filters') == FILTERS and
             manifest.get('data_identity') == digest({k: v for k, v in manifest.items() if k != 'data_identity'}),
             'Frozen manifest identity/contract mismatch')
    payload = load_checkpoint(checkpoint['path'], expected_data_identity=manifest['data_identity'])
    meta, condition = checkpoint_details(payload, deferred=True)
    architecture = meta['config']['architecture']
    if architecture == 'controlled' or 'architecture' in selection or 'training_condition' in selection:
        _require(selection.get('architecture') == architecture and
                 _same(selection.get('training_condition'), condition),
                 'Frozen selection architecture/training condition mismatch')
    source = meta['source_receipt']
    _require(source['data_identity'] == manifest['data_identity'] and source.get('weight_parent') is None and
             _same(source['label_source_files'], manifest['source_files']) and
             source['filters'] == meta['filters'] and source['vocabulary_sha256'] == meta['vocabulary_sha256'],
             'Checkpoint source receipt/manifest binding mismatch')
    launch, _ = inputs.read(source['launch']['path'], source['launch'])
    _require(launch.get('parent_checkpoint') is None and _same(launch['effective_config'], meta['config']) and
             source.get('architecture') == architecture and source.get('defer_test') is True and
             _same(launch['native'], manifest['public_surface']['provenance']['native_provenance']) and
             not launch['native'].get('preparation_only') and _hash(launch['native'].get('module_sha256')),
             'Checkpoint training source/native identity mismatch')
    inputs.verify(launch['source_archive'])
    for item in manifest['source_files']:
        inputs.verify(item)
    inputs.recheck()
    return selection, identity, manifest, meta, condition
