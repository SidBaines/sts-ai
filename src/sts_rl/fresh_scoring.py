"""Finalize deferred TEST only after an external validation-only selection freeze.

This module never trains or selects models. File identities and the declared
selection criterion are checked before opening any raw label shard.
"""
from pathlib import Path
import json

import torch

from sts_policy.entity_encoding import VOCABULARY_SHA256
from sts_policy.training import load_checkpoint
from sts_rl.fresh_study import atomic_json
from sts_rl.fresh_training import (FILTERS, TrainingControl, _verify_file, digest,
    freeze_manifest, preencode, score_test_checkpoint)
from sts_rl.recording import begin_run, file_identity


SELECTION_CRITERION = {'metric':'validation_soft_ce','direction':'min','tie_break':'lower_lr'}


def score_deferred(selection_path, out, *, deadline_utc=None):
    """Score exactly the frozen checkpoint, using only singleton CPU actor calls.

    Selection JSON requires selected_checkpoint and manifest (recording.py file
    identities), plus selection_criterion equal to SELECTION_CRITERION. The
    orchestrator freezes that receipt after fixing all hyperparameters; this
    scorer does not compare candidates or infer a selection from TEST.
    """
    selection_identity = file_identity(selection_path)
    selection = json.loads(Path(selection_path).read_text())
    if selection.get('selection_criterion') != SELECTION_CRITERION:
        raise ValueError('Selection requires minimum validation_soft_ce with lower_lr tie break')
    for key in ('selected_checkpoint','manifest'):
        if key not in selection:
            raise ValueError('Selection missing '+key+' identity')
        _verify_file(selection[key])
    checkpoint = selection['selected_checkpoint']
    manifest = json.loads(Path(selection['manifest']['path']).read_text())
    if (manifest.get('schema') != 'fresh_data_manifest_v2' or manifest.get('filters') != FILTERS or
            manifest.get('data_identity') != digest({k:v for k,v in manifest.items() if k!='data_identity'})):
        raise ValueError('Frozen manifest identity/contract mismatch')
    payload = load_checkpoint(checkpoint['path'],expected_data_identity=manifest['data_identity'])
    metadata = payload['metadata']
    config = metadata['config']
    architecture, phase = config['architecture'], config['phase']
    controlled_inputs = None
    if architecture == 'controlled' or metadata.get('kind') == 'entity_controlled_v1':
        from sts_rl.controlled_scoring import frozen_selection
        from sts_rl.fresh_panel_comparison import _Inputs
        controlled_inputs = _Inputs()
        checked, checked_id, checked_manifest, _, _ = frozen_selection(selection_path, controlled_inputs)
        if (checked != selection or checked_id != selection_identity or checked_manifest != manifest or
                architecture != 'controlled'):
            raise ValueError('Controlled selection changed or architecture/kind mismatch')
        del checked_manifest
    if (not config.get('defer_test',False) or metadata.get('initialization') != 'from_scratch' or
            metadata.get('parent_checkpoint') is not None or metadata.get('filters') != FILTERS or
            metadata.get('vocabulary_sha256') != VOCABULARY_SHA256 or
            architecture not in ('new','old','controlled') or phase not in ('combat','ooc')):
        raise ValueError('Checkpoint is not a compatible deferred warm-up selection')
    if payload['step'] != metadata['training_state']['best_step']:
        raise ValueError('Checkpoint is not the validation-selected best step')
    if architecture=='old' and (payload['model_config']['size'] != config.get('old_size','large') or
                               payload['model_config']['context'] != config.get('old_context','none')):
        raise ValueError('Checkpoint old control architecture/config mismatch')
    _verify_file(checkpoint)
    _verify_file(selection['manifest'])
    _verify_file(selection_identity)
    # No raw TEST access has occurred above this boundary.
    sources = [f['path'] for f in manifest['source_files']]
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Fresh-only scoring output required')
    (out/'.fresh-score-claim').touch(exist_ok=False)
    torch.set_num_threads(1)
    control = TrainingControl(out,deadline_utc)
    control.start()
    try:
        control.check()
        if freeze_manifest(sources,control=control,mixed_sources=manifest.get('mixed_sources',False),
                           mixed_sampling=manifest.get('mixed_sampling',False)) != manifest:
            raise ValueError('Frozen manifest/source identities differ')
        effective = {'stage':'deferred_test_scoring','training':False,'weight_updates':0,
                     'architecture':architecture,'phase':phase,'device':'cpu',
                     'selection':selection_identity,'selected_checkpoint':checkpoint,
                     'manifest':selection['manifest'],'selection_criterion':selection['selection_criterion'],
                     'data_identity':manifest['data_identity']}
        launch = begin_run(out,effective,checkpoint=checkpoint['path'],
                           inputs=[selection_path,selection['manifest']['path'],*sources])
        _verify_file(launch)
        recorded = json.loads(Path(launch['path']).read_text())
        if (recorded['native'].get('preparation_only') or not recorded['native'].get('module_sha256') or
                recorded['native'] != manifest['public_surface']['provenance']['native_provenance']):
            raise ValueError('Scoring requires validated native identity matching the label dataset')
        receipt = {**effective,'launch':launch,'filters':FILTERS,'vocabulary_sha256':VOCABULARY_SHA256,
                   'source_receipt':metadata['source_receipt'],'label_source_files':manifest['source_files']}
        atomic_json(out/'receipt.json',receipt)
        # Recheck the frozen inputs immediately before TEST encoding.
        for identity in (selection_identity,selection['manifest'],checkpoint):
            _verify_file(identity)
        if controlled_inputs is not None:
            controlled_inputs.recheck()
        dataset = preencode(sources,manifest,phase,architecture,control=control,splits=('TEST',))
        result = score_test_checkpoint(checkpoint['path'],dataset['TEST'],architecture,out,control=control)
        for identity in (selection_identity,selection['manifest'],checkpoint):
            _verify_file(identity)
        if controlled_inputs is not None:
            controlled_inputs.recheck()
        summary = {'status':'completed','training':False,'weight_updates':0,
                   'selected_checkpoint':checkpoint,'selection':selection_identity,
                   'data_identity':manifest['data_identity'],**result,**control.stats()}
        if controlled_inputs is not None:
            summary['test_rows_file'] = file_identity(out/'test_rows.jsonl')
        atomic_json(out/'summary.json',summary)
        control.status.update(summary)
        control.emit()
        return summary
    except Exception as exc:
        control.status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        control.emit()
        raise
    finally:
        control.close()
