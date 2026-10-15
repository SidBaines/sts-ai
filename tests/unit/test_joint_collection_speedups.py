import copy
import gc
import gzip
import pickle
import random
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import weakref
from concurrent.futures import Future
from pathlib import Path

try:
    import numpy as np
    import torch

    from sts_policy.curriculum import TaskSpec
    from sts_policy.model import CandidateActorCritic
    from sts_policy.model_agent import CompositeActorCritic, collate_public
    from sts_policy.rollout_training import collect_rollouts
    from sts_rl.joint_sparse import (JointConfig, SparseCollectors, SparseEpisodeSink, WinCritics, actor_forward,
        actor_update, build_anchors, compact_chunk, compact_stream_chunk, critic_update, evaluation_forward,
        freeze_legacy_critics, identity_state, save_buffer)
    from sts_rl.packed_rows import PackedRows, concatenate_rows
except ModuleNotFoundError as error:
    if error.name not in {"torch", "numpy"}:
        raise
    torch = None


class Observation(dict):
    pass


class Worker:
    fault_exception = TimeoutError
    def __init__(self):
        self.refs = []

    def observe(self, done=False):
        obs = Observation(phase='combat' if self.step % 2 == 0 else 'out_of_combat',
            state={'act': 3, 'floor': 40 + self.step, 'cur_hp': 20, 'max_hp': 80,
                   'room': 'BOSS', 'unused_large_field': 'x' * 10000},
            text='Deck: Strike, Defend', legal_actions=[] if done else [
                {'index': 0, 'description': 'Play Strike'}, {'index': 1, 'description': 'End turn'}])
        self.refs.append(weakref.ref(obs))
        return obs

    def submit(self, method, *args):
        f = Future()
        try:
            if method == 'reset':
                self.seed, self.step = args[0].seed, 0
                f.set_result((self.observe(), {}))
            else:
                self.step += 1
                if self.seed == 7 and self.step == 2:
                    raise self.fault_exception('native operation timed out')
                cut = self.seed == 8 and self.step == 2
                done = self.step == 3
                win = done and self.seed == 5
                f.set_result((self.observe(done), float(win), done, cut,
                    {'task_outcome': 'victory' if win else 'defeat' if done else '',
                     'action_prefix': list(range(self.step)), 'ooc_previews': [False] * self.step}))
        except Exception as e:
            f.set_exception(e)
        return f


@unittest.skipIf(torch is None, "Optional RL dependencies absent")
class CollectionSpeedupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.actor = CompositeActorCritic(CandidateActorCritic('small', 'none'),
                                         CandidateActorCritic('small', 'none'))
        freeze_legacy_critics(self.actor)
        self.critics = WinCritics()
        self.cfg = JointConfig(out='/tmp/unused', checkpoint='/tmp/unused', microbatch=4, minibatch=8)

    def collect(self, mode, seeds=(5, 6, 7, 8), seconds=20, worker_cls=Worker):
        torch.manual_seed(19)
        worker = worker_cls()
        lean = mode == 'lean'
        stream = mode != 'legacy'
        r = collect_rollouts(self.actor, [worker], [TaskSpec(kind='run', seed=s, max_act=3) for s in seeds],
            collate=collate_public, forward=evaluation_forward if lean else actor_forward(self.actor, self.critics, self.cfg, lean),
            deterministic=lean, deferred_bootstrap=True, max_seconds=seconds,
            reuse_encodings=not lean, actor_only=lean,
            episode_sink_factory=(lambda: SparseEpisodeSink(not lean)) if stream else None)
        compact = compact_stream_chunk(r) if stream else compact_chunk(r)
        return r, compact, worker

    def test_training_stream_exact_rows_receipts_and_fault_quarantine(self):
        old, expected, _ = self.collect('legacy')
        new, actual, worker = self.collect('compact')
        self.assertEqual(expected[1:], actual[1:])
        self.assertEqual([{k:v for k,v in f.items() if k != 'wait_seconds'} for f in old['faults']],
                         [{k:v for k,v in f.items() if k != 'wait_seconds'} for f in new['faults']])
        self.assertEqual(new['transitions'], [])
        self.assertEqual(new['excluded_transitions'], 1)
        self.assertEqual(len(actual[0]), 6)  # two genuine whole games only
        for a, b in zip(expected[0], actual[0]):
            for key in a:
                if isinstance(a[key], np.ndarray):
                    np.testing.assert_array_equal(a[key], b[key])
                else:
                    self.assertEqual(a[key], b[key], key)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in worker.refs))
        self.assertEqual(old['retained_encoding_bytes'], new['retained_encoding_bytes'])

    def test_lean_eval_matches_all_episode_diagnostics(self):
        worker = Worker()
        torch.manual_seed(19)
        old = collect_rollouts(self.actor, [worker], [TaskSpec(kind='run', seed=s, max_act=3) for s in (5,6,7,8)],
            collate=collate_public, forward=actor_forward(self.actor, self.critics, self.cfg, True),
            deterministic=True, deferred_bootstrap=True)
        expected = compact_chunk(old, keep_rows=False)
        r, actual, _ = self.collect('lean')
        self.assertEqual(expected[1:], actual[1:])
        self.assertEqual([{k:v for k,v in f.items() if k != 'wait_seconds'} for f in old['faults']],
                         [{k:v for k,v in f.items() if k != 'wait_seconds'} for f in r['faults']])
        self.assertEqual(len(actual[0]), 0)
        self.assertEqual(r['retained_encoding_bytes'], 0)
        self.assertEqual(r['transitions'], [])

    def test_budget_censoring_and_sink_guards(self):
        _, old, _ = self.collect('legacy', seeds=(5,), seconds=0)
        _, new, _ = self.collect('compact', seeds=(5,), seconds=0)
        self.assertEqual(old[1:], new[1:])
        self.assertEqual(len(new[0]), 0)
        with self.assertRaises(ValueError):
            collect_rollouts(self.actor, [], [], collate=collate_public,
                forward=evaluation_forward, actor_only=True)
        with self.assertRaises(ValueError):
            collect_rollouts(self.actor, [], [], collate=collate_public,
                forward=evaluation_forward, episode_sink_factory=SparseEpisodeSink)
        with self.assertRaisesRegex(ValueError, 'Reward contract'):
            SparseEpisodeSink().finish(dict(terminated=True, task_outcome='victory', reward=0))

    def test_actor_only_logits_match_mixed_and_homogeneous_heads(self):
        for context in ('none', 'mean'):
            actor = CompositeActorCritic(CandidateActorCritic('small', context), CandidateActorCritic('small', context))
            for phase in ([0,1,0,1], [0]*4, [1]*4):
                state, actions = torch.randn(4,256), torch.randn(4,5,128)
                state[:,7] = torch.tensor(phase)
                mask = torch.tensor([[1,1,0,0,0],[1]*5,[1,1,1,0,0],[1]*5], dtype=torch.bool)
                with torch.no_grad():
                    torch.testing.assert_close(actor(state,actions,mask)[0], actor.actor_logits(state,actions,mask), rtol=0, atol=0)

    def test_packed_anchors_own_only_selected_arrays_and_schema_is_checked(self):
        _, data, _ = self.collect('legacy', seeds=(5,6))
        rows=[]
        for phase in ('combat','ooc'):
            row=next(r for r in data[0] if r['phase']==phase)
            for act in (1,2,3):
                rows.extend({**row, 'act':act} for _ in range(20))
        packed=PackedRows.from_rows(rows)
        anchors=build_anchors(packed,16,19)
        self.assertEqual(len(anchors),96)
        for row in anchors:
            self.assertFalse(np.shares_memory(row['state'],packed.state))
            self.assertFalse(np.shares_memory(row['actions'],packed.actions))
        state=packed.__getstate__()
        self.assertTrue(torch.is_tensor(state['state']))
        self.assertEqual(state['columns']['logp'].dtype,torch.float64)
        state['schema']='future unsupported format'
        with self.assertRaisesRegex(ValueError,'schema'):
            PackedRows.__new__(PackedRows).__setstate__(state)

    def test_programming_errors_propagate_instead_of_becoming_losses(self):
        class BrokenWorker(Worker):
            fault_exception=RuntimeError
        with self.assertRaisesRegex(RuntimeError,'Unexpected collector'):
            self.collect('compact',seeds=(7,),worker_cls=BrokenWorker)

    def test_worker_budget_and_empty_tasks(self):
        context = MagicMock()
        context.Pipe.side_effect = [(MagicMock(), MagicMock()) for _ in range(4)]
        with patch('sts_rl.joint_sparse.mp.get_context', return_value=context):
            pool = SparseCollectors(self.cfg, collectors=4, workers=6)
            self.assertEqual(pool.worker_counts, [2,2,1,1])
            self.assertEqual(sum(pool.worker_counts), 6)
            assigned = [c.kwargs['args'][3] for c in context.Process.call_args_list]
            self.assertEqual(assigned, pool.worker_counts)
            with self.assertRaises(ValueError): SparseCollectors(self.cfg, collectors=0)
        packed = PackedRows.concatenate([[], []])
        self.assertEqual(len(packed), 0)
        self.assertEqual(packed.state.shape, (0,256))
        self.assertEqual(packed.actions.shape, (0,128))
        self.assertEqual(packed.offsets.tolist(), [0])

    def test_packed_serialization_concat_and_exact_optimizer_replay(self):
        _, expected, _ = self.collect('legacy', seeds=(5,6))
        rows = expected[0]
        packed = PackedRows.from_rows(rows)
        packed = PackedRows.concatenate([packed[:2], [], packed[2:]])
        self.assertIsInstance(concatenate_rows([[], packed]), PackedRows)
        for roundtrip in [pickle.loads(pickle.dumps(packed))]:
            for a, b in zip(rows, roundtrip):
                for k in a:
                    if isinstance(a[k],np.ndarray): np.testing.assert_array_equal(a[k],b[k])
                    else: self.assertEqual(a[k],b[k])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'buffer.pt.gz'
            save_buffer({'rows': packed}, path)
            with gzip.open(path,'rb') as f:
                loaded = torch.load(f, weights_only=False)['rows']
            self.assertIsInstance(loaded, PackedRows)
        states = []
        for data in (rows, loaded):
            actor, critics, ref = copy.deepcopy(self.actor), copy.deepcopy(self.critics), copy.deepcopy(self.actor)
            opts = {p: torch.optim.AdamW([v for v in getattr(actor,p).parameters() if v.requires_grad], lr=1e-5, weight_decay=0) for p in ('combat','ooc')}
            co = torch.optim.AdamW(critics.parameters(),lr=.0003)
            rng = random.Random(9)
            update = actor_update(actor,ref,opts,data,rows,self.cfg,rng,.1)
            critic_update(critics,co,data,self.cfg)
            # Test moments on a second step as well as the initial weights.
            actor_update(actor,ref,opts,data,rows,self.cfg,rng,.1)
            critic_update(critics,co,data,self.cfg)
            states.append((identity_state(actor),identity_state(critics),rng.getstate(),update,
                           copy.deepcopy({p:o.state_dict() for p,o in opts.items()}),
                           copy.deepcopy(co.state_dict())))
            if isinstance(data,PackedRows):
                before=identity_state(actor)
                local_rng=rng.getstate()
                bad=JointConfig(**{**self.cfg.__dict__, 'local_kl_limit':-1})
                rejected=actor_update(actor,ref,opts,data,rows,bad,rng,.1)
                self.assertFalse(rejected['accepted'])
                self.assertEqual(rejected['retained_steps'],0)
                self.assertEqual(identity_state(actor),before)
                self.assertEqual(local_rng,rng.getstate())
                for phase,optimizer in opts.items():
                    for key,old in states[-1][4][phase]['state'].items():
                        for name,value in old.items():
                            self.assertTrue(torch.equal(value,optimizer.state_dict()['state'][key][name]))
        self.assertEqual(states[0][:4], states[1][:4])
        def compare(a,b):
            if torch.is_tensor(a): self.assertTrue(torch.equal(a,b))
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for k in a: compare(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                for x,y in zip(a,b): compare(x,y)
            else: self.assertEqual(a,b)
        compare(states[0][4],states[1][4])
        compare(states[0][5],states[1][5])
        for key in ('logp','value','return','advantage'):
            self.assertEqual(packed.columns[key].dtype,np.float64)
        self.assertEqual(packed[-1]['seed'], rows[-1]['seed'])
        with self.assertRaises(IndexError): packed[len(packed)]
        with self.assertRaises(ValueError):
            self.cfg.__class__(**{**self.cfg.__dict__, 'collection_backend':'unknown'}).validate()


if __name__ == '__main__':
    unittest.main()
