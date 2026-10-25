"""K2 reasoning boundaries, strict final-action parsing and a small HTTP client."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import re
import time
from urllib.request import Request, urlopen

K2_REASONING_CLOSERS = ('</ifm|think>', '</ifm|think_fast>', '</ifm|think_faster>')
K2_REASONING_END_TOKEN_IDS = (64030, 64051, 64053)
K2_ASSISTANT_END_TOKEN_ID = 64019
_TRUNCATED = {'length', 'max_tokens', 'max_length', 'token_limit'}
_OPENERS = ('<ifm|think>', '<ifm|think_fast>', '<ifm|think_faster>', '<think>')
_ENDS = ('<|ifm|im_end|>', '<|endoftext|>', '<|im_end|>', '<ifm|im_end>', '</ifm|im_end>', '[im_end]')


@dataclass(frozen=True)
class ActionParse:
    action_index: int | None
    valid: bool
    error: str | None
    reasoning: str
    final_text: str
    truncated: bool

    def to_dict(self):
        return asdict(self)


def parse_action(text: str, legal_indices, *, finish_reason: str | None = None,
                 reasoning_content: str | None = None, final_channel: bool = False,
                 require_reasoning_boundary: bool = True, labels=None) -> ActionParse:
    """Parse only the final channel; never accept JSON inside unfinished thought.

    K2's chat template already supplies the opening reasoning tag, so raw generated
    completions normally have a closer with no opener. Any of the three effort
    closers can occur at any requested effort. A native API's separate reasoning
    field establishes an explicit final channel, even when that field is empty.

    With ``labels`` (label -> native index, from ``observations.labelled_prompt``)
    the final JSON must be exactly ``{"action": "<label>"}`` and the returned
    ``action_index`` is the native index; otherwise ``{"action_index": int}``.
    """
    if not isinstance(text, str):
        raise TypeError('Generated text must be a string')
    indices = list(legal_indices)
    if any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('Legal indices must be unique integers')
    if labels is not None and (not labels or set(labels.values()) - set(indices)
                               or any(not isinstance(label, str) for label in labels)):
        raise ValueError('Labels must map strings onto legal indices')
    truncated = finish_reason in _TRUNCATED
    boundary = [(text.find(tag), tag) for tag in (*K2_REASONING_CLOSERS, '</think>') if tag in text]
    reasoning = reasoning_content or ''
    final = text
    if boundary:
        position, tag = min(boundary)
        reasoning = text[:position]
        for opener in _OPENERS:
            reasoning = reasoning.removeprefix(opener)
        final = text[position + len(tag):]
    elif reasoning_content is None and not final_channel and require_reasoning_boundary:
        return ActionParse(None, False, 'truncated' if truncated else 'missing_reasoning_boundary',
                           text, '', truncated)
    final = final.strip()
    stripping = True
    while stripping:
        stripping = False
        for suffix in _ENDS:
            if final.endswith(suffix):
                final = final[:-len(suffix)].strip()
                stripping = True
                break
    if truncated:
        return ActionParse(None, False, 'truncated', reasoning, final, True)
    # A single Markdown fence is accepted; prose, duplicate JSON and nested
    # action objects are rejected rather than taking a convenient last match.
    match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', final, flags=re.S | re.I)
    if match:
        final = match.group(1).strip()
    try:
        def no_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate key')
                result[key] = value
            return result
        value = json.loads(final, object_pairs_hook=no_duplicates)
    except (ValueError, TypeError):
        return ActionParse(None, False, 'invalid_final_json', reasoning, final, False)
    if labels is not None:
        if not isinstance(value, dict) or set(value) != {'action'} or not isinstance(value['action'], str):
            return ActionParse(None, False, 'invalid_action_schema', reasoning, final, False)
        if value['action'] not in labels:
            return ActionParse(None, False, 'illegal_action_label', reasoning, final, False)
        return ActionParse(labels[value['action']], True, None, reasoning, final, False)
    if not isinstance(value, dict) or set(value) != {'action_index'} or type(value['action_index']) is not int:
        return ActionParse(None, False, 'invalid_action_schema', reasoning, final, False)
    index = value['action_index']
    if index not in indices:
        return ActionParse(index, False, 'illegal_action_index', reasoning, final, False)
    return ActionParse(index, True, None, reasoning, final, False)


@dataclass(frozen=True)
class Generation:
    text: str
    reasoning_content: str | None
    finish_reason: str | None
    usage: dict
    elapsed_seconds: float
    raw: dict


class OpenAIChatClient:
    """OpenAI-compatible endpoint; model-specific template options stay explicit."""
    def __init__(self, *, base_url: str, model: str, max_tokens: int = 4096,
                 temperature: float = 0., timeout: float = 120., api_key: str | None = None,
                 extra_body: dict | None = None):
        self.url = base_url.rstrip('/')
        if not self.url.endswith('/chat/completions'):
            self.url += '/chat/completions' if self.url.endswith('/v1') else '/v1/chat/completions'
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.timeout, self.api_key = timeout, api_key
        self.extra_body = dict(extra_body or {})
        if {'messages', 'model'} & self.extra_body.keys():
            raise ValueError('extra_body cannot replace model or messages')

    def generate(self, messages, *, seed: int | None = None) -> Generation:
        body = {'model': self.model, 'messages': messages, 'max_tokens': self.max_tokens,
                'temperature': self.temperature, 'stream': False, **self.extra_body}
        if seed is not None:
            body['seed'] = seed
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        started = time.monotonic()
        with urlopen(Request(self.url, data=json.dumps(body).encode(), headers=headers),
                     timeout=self.timeout) as response:
            raw = json.load(response)
        choice = raw['choices'][0]
        message = choice.get('message', {})
        text = message.get('content', choice.get('text', '')) or ''
        if not isinstance(text, str):
            raise ValueError('Endpoint returned non-string content')
        reasoning = message.get('reasoning_content', message.get('reasoning'))
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError('Endpoint returned non-string reasoning')
        return Generation(text, reasoning, choice.get('finish_reason'), raw.get('usage', {}),
                          time.monotonic() - started, raw)
