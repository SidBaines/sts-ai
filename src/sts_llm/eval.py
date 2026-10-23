"""Saved-state diagnostics and a clean-CLI agent with auditable responses."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from .generation import OpenAIChatClient, parse_action
from .observations import prompt, legal_indices, observation_identity
from .teacher import teacher_reward, validate_teacher

RESPONSE_LOG_SCHEMA = 'sts_llm_response_log_v1'


class InvalidModelAction(RuntimeError):
    """Model failure, deliberately not converted into an ordinary game loss."""


class LLMAgent:
    def __init__(self, client, *, response_log: str | None = None,
                 require_reasoning_boundary: bool = True, auto_forced: bool = True):
        self.client, self.require_reasoning_boundary = client, require_reasoning_boundary
        self.auto_forced = auto_forced
        self.log = Path(response_log).open('x') if response_log else None
        self.episode_counter = 0
        self.policy_seed = None
        self.rng = None
        self.decision = 0
        self._closed = False

    def reset(self, seed):
        if self._closed:
            raise RuntimeError('LLM agent is closed')
        self.policy_seed = int(seed)
        self.rng = random.Random(self.policy_seed)
        self.episode_counter += 1
        self.decision = 0
        self._record({'event': 'reset', 'policy_seed': self.policy_seed})

    def _record(self, record):
        if self.log:
            record = {'schema': RESPONSE_LOG_SCHEMA, 'episode_counter': self.episode_counter,
                      **record}
            self.log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            self.log.flush()

    def choose_action(self, observation):
        if self._closed:
            raise RuntimeError('LLM agent is closed')
        # Direct callers may omit reset; the clean CLI always supplies it. Lazy
        # initialization avoids creating an unused constructor episode in logs.
        if self.rng is None:
            self.reset(0)
        indices = legal_indices(observation)
        self.decision += 1
        identity = observation_identity(observation)
        if self.auto_forced and len(indices) == 1:
            self._record({'event': 'forced_action', 'decision': self.decision,
                          'public_observation_identity': identity, 'action_index': indices[0]})
            return indices[0]
        messages = prompt(observation)
        response = self.client.generate(messages, seed=self.rng.randrange(2**31))
        parsed = parse_action(response.text, indices, finish_reason=response.finish_reason,
                              reasoning_content=response.reasoning_content,
                              require_reasoning_boundary=self.require_reasoning_boundary)
        self._record({'event': 'model_action', 'decision': self.decision,
                      'public_observation_identity': identity, 'phase': observation.get('phase'),
                      'legal_indices': indices, 'messages': messages,
                      'parse': parsed.to_dict(), 'response': response.raw,
                      'elapsed_seconds': response.elapsed_seconds})
        if not parsed.valid:
            raise InvalidModelAction(f'Model action failure at decision{self.decision}: {parsed.error}')
        return parsed.action_index

    def close(self):
        if self.log:
            self.log.close()
        self._closed = True


def make_agent(*, base_url, model, max_tokens=4096, temperature=0., timeout=120., api_key=None,
               extra_body=None, response_log=None, require_reasoning_boundary=True,
               auto_forced=True):
    return LLMAgent(OpenAIChatClient(base_url=base_url, model=model, max_tokens=max_tokens,
        temperature=temperature, timeout=timeout, api_key=api_key, extra_body=extra_body),
        response_log=response_log, require_reasoning_boundary=require_reasoning_boundary,
        auto_forced=auto_forced)


def evaluate_saved(rows, client, *, output: str, require_reasoning_boundary=True):
    """Never calls the simulator. Agreement is a diagnostic, not correctness."""
    count = legal = truncated = agreement = excluded = 0
    reward = 0.
    started = time.monotonic()
    with Path(output).open('x') as stream:
        for row in rows:
            teacher = validate_teacher(row)
            response = client.generate(row['messages'])
            parsed = parse_action(response.text, row['legal_indices'], finish_reason=response.finish_reason,
                reasoning_content=response.reasoning_content, require_reasoning_boundary=require_reasoning_boundary)
            score = teacher_reward(row, parsed.action_index if parsed.valid else None)
            top = teacher['legal_indices'][max(range(len(teacher['probabilities'])),
                                              key=teacher['probabilities'].__getitem__)]
            matched = parsed.valid and parsed.action_index == top
            filtered = parsed.valid and parsed.action_index in teacher.get('excluded_indices', [])
            record = {'id': row['id'], 'phase': row['phase'], 'split': row.get('split'),
                      'parse': parsed.to_dict(), 'teacher_probability': score,
                      'teacher_top_agreement': matched, 'teacher_filtered_action': filtered,
                      'response': response.raw, 'elapsed_seconds': response.elapsed_seconds}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            count += 1
            legal += parsed.valid
            truncated += parsed.truncated
            agreement += matched
            excluded += filtered
            reward += score
    return {'n': count, 'legal_rate': legal/count if count else None,
            'truncation_rate': truncated/count if count else None,
            'mean_teacher_probability': reward/count if count else None,
            'teacher_top_agreement': agreement/count if count else None,
            'teacher_filtered_action_rate': excluded/count if count else None,
            'elapsed_seconds': time.monotonic()-started,
            'interpretation': 'Saved-state teacher preferences; not a game-win or reasoning-faithfulness evaluation.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--extra-body', type=json.loads, default={})
    parser.add_argument('--allow-final-channel-only', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row['split'] == args.split]
    if not rows:
        raise ValueError('Requested split is empty')
    client = OpenAIChatClient(base_url=args.base_url, model=args.model, max_tokens=args.max_tokens,
                             temperature=args.temperature, extra_body=args.extra_body)
    summary = evaluate_saved(rows, client, output=args.output,
                              require_reasoning_boundary=not args.allow_final_channel_only)
    Path(args.output + '.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
