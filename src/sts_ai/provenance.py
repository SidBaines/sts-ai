"""Readable and content-addressed competence-interface provenance."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

from sts_ai.prompting import (
    NEUTRAL_FRAME,
    REASONING_ACTION_OUTPUT,
    render_action_prompt,
)
from sts_ai.schemas import LegalAction


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=32)
def _sha256_cached(path_text: str, size: int, mtime_ns: int) -> str:
    _ = size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    try:
        stat = path.stat()
        return _sha256_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return None


def adapter_provenance(path: str | Path | None) -> dict[str, Any] | None:
    """Content-address a LoRA adapter instead of trusting its mutable path."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"adapter path does not exist: {resolved}")
    if resolved.is_file():
        return {
            "path": str(resolved),
            "sha256": file_sha256(resolved),
        }
    files: dict[str, str] = {}
    for name in ("adapters.safetensors", "adapter_config.json"):
        candidate = resolved / name
        digest = file_sha256(candidate)
        if digest is not None:
            files[name] = digest
    if "adapters.safetensors" not in files:
        raise ValueError(f"adapter directory has no adapters.safetensors: {resolved}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(resolved),
        "files": files,
        "identity_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def competence_interface_provenance(env: Any, agent: Any) -> dict[str, Any]:
    """Fingerprint the actual local policy interface, including dirty files.

    A git SHA alone names only HEAD and is insufficient while an experiment runs
    from reviewed-but-uncommitted changes. These hashes identify the bytes that
    define state, glossary, prompt, native patch/build, and chat-template probe.
    """

    root = _repo_root()
    framing = str(getattr(agent, "framing", NEUTRAL_FRAME))
    output_contract = str(
        getattr(agent, "output_contract", REASONING_ACTION_OUTPUT)
    )
    prompt_probe = render_action_prompt(
        "__sts_state_probe__",
        [
            LegalAction(index=0, bits=0, description="probe action zero"),
            LegalAction(index=1, bits=1, description="probe action one"),
        ],
        framing,
        output_contract=output_contract,
    )
    result: dict[str, Any] = {
        "competence_interface_version": str(
            getattr(env, "combat_observation", "legacy")
        ),
        "combat_observation": str(getattr(env, "combat_observation", "legacy")),
        "output_contract": output_contract,
        "prompt_probe_sha256": hashlib.sha256(prompt_probe.encode("utf-8")).hexdigest(),
        "python_serializer_sha256": file_sha256(root / "src/sts_ai/lightspeed.py"),
        "glossary_sha256": file_sha256(root / "src/sts_ai/glossary.py"),
        "prompting_sha256": file_sha256(root / "src/sts_ai/prompting.py"),
        "simulator_patch_sha256": file_sha256(
            root / "patches/sts_lightspeed_python_api.patch"
        ),
    }
    module_path = getattr(getattr(env, "sts", None), "__file__", None)
    if module_path:
        result["simulator_binary_sha256"] = file_sha256(module_path)

    tokenizer = getattr(agent, "tokenizer", None)
    if tokenizer is not None:
        try:
            from sts_ai.train.sft_format import chat_template_probe_hash

            result["chat_template_probe_hash"] = chat_template_probe_hash(
                tokenizer,
                enable_thinking=bool(getattr(agent, "enable_thinking", False)),
            )
        except (AttributeError, TypeError, ValueError):
            # The rest of the provenance remains useful for simple/fake agents.
            pass
    return {key: value for key, value in result.items() if value is not None}
