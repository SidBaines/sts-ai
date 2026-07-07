"""Thin MLX LoRA trainer wrapper and data conversion helpers."""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import types
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

__all__ = ["build_lora_cmd", "prepare_mlx_data", "prepare_native_mlx_data", "train"]


_GEMMA_THOUGHT_MARKERS = ("<|channel>thought", "<channel|>")
_GEMMA_THOUGHT_OPEN = "<|channel>thought"
_GEMMA_THOUGHT_CLOSE = "<channel|>"
_GEMMA_ASSISTANT_TURN_END = "<turn|>\n"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _mlx_record(record: dict[str, Any]) -> dict[str, Any]:
    if "messages" not in record:
        raise ValueError(
            "dataset record is missing required 'messages' field for MLX chat "
            "training"
        )
    completion = str(record.get("completion", ""))
    if any(marker in completion for marker in _GEMMA_THOUGHT_MARKERS):
        raise ValueError(
            "MLX chat training would strip Gemma native thought-channel "
            "completions via the model chat template. Use a native-thinking-safe "
            "training path instead of train_mlx.prepare_mlx_data."
        )
    return {"messages": record["messages"]}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_mlx_record(record), ensure_ascii=False) + "\n")


def _write_raw_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _split_records(
    records: list[dict[str, Any]],
    *,
    valid_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in [0.0, 1.0)")
    if len(records) <= 1:
        return records, []

    n_valid = int(len(records) * valid_fraction)
    n_valid = max(1, min(n_valid, len(records) - 1))
    return records[n_valid:], records[:n_valid]


def prepare_mlx_data(
    dataset_path: Path,
    out_dir: Path,
    *,
    valid_fraction: float = 0.1,
    shuffle_seed: int = 0,
) -> Path:
    """Convert SFT JSONL to mlx-lm's chat JSONL layout.

    The input dataset can contain provenance keys such as ``world_seed`` and
    ``stem``; only ``messages`` is written so mlx-lm selects its ChatDataset.
    For a one-example dataset, the single example stays in train and
    ``valid.jsonl`` is empty so the trainer never sees a duplicated target.
    """
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(dataset_path)
    random.Random(shuffle_seed).shuffle(records)
    train_records, valid_records = _split_records(
        records,
        valid_fraction=valid_fraction,
    )

    _write_jsonl(out_dir / "train.jsonl", train_records)
    _write_jsonl(out_dir / "valid.jsonl", valid_records)
    return out_dir


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool = True) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return list(token_ids)


def _thinking_close_token_index(
    *,
    prompt_len: int,
    completion: str,
    tokenizer: Any,
) -> int | None:
    lower = completion.lower()
    open_idx = lower.find(_GEMMA_THOUGHT_OPEN)
    if open_idx == -1:
        return None
    close_idx = lower.find(_GEMMA_THOUGHT_CLOSE, open_idx)
    if close_idx == -1:
        return -1
    close_end = close_idx + len(_GEMMA_THOUGHT_CLOSE)
    return prompt_len + len(
        _encode(
            tokenizer,
            completion[:close_end],
            add_special_tokens=False,
        )
    )


def _native_token_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[dict[str, Any] | None, str | None]:
    prompt = str(record.get("prompt", ""))
    completion = str(record.get("completion", ""))
    if not prompt:
        return None, "missing_prompt"
    if not completion:
        return None, "missing_completion"

    prompt_ids = _encode(tokenizer, prompt)
    target_completion = (
        completion
        if completion.endswith(_GEMMA_ASSISTANT_TURN_END)
        else completion + _GEMMA_ASSISTANT_TURN_END
    )
    completion_ids = _encode(tokenizer, target_completion, add_special_tokens=False)
    if not completion_ids:
        return None, "missing_completion"

    thought_close = _thinking_close_token_index(
        prompt_len=len(prompt_ids),
        completion=completion,
        tokenizer=tokenizer,
    )
    if thought_close == -1:
        return None, "source_thinking_truncated"

    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_seq_length:
        if thought_close is not None and thought_close > max_seq_length:
            return None, "would_truncate_thinking"
        return None, "too_long"

    return (
        {
            "input_ids": input_ids,
            "offset": len(prompt_ids),
            "n_prompt_tokens": len(prompt_ids),
            "n_completion_tokens": len(completion_ids),
            "n_total_tokens": len(input_ids),
            "world_seed": record.get("world_seed"),
            "decision_index": record.get("decision_index"),
            "stem": record.get("stem"),
            "task_window_id": record.get("task_window_id"),
        },
        None,
    )


def prepare_native_mlx_data(
    dataset_path: Path,
    out_dir: Path,
    *,
    tokenizer: Any,
    max_seq_length: int = 8192,
    valid_fraction: float = 0.1,
    shuffle_seed: int = 0,
) -> dict[str, Any]:
    """Pre-tokenize prompt+completion SFT data for native-thinking MLX LoRA.

    This bypasses `mlx_lm`'s stock ChatDataset, which strips Gemma-4 native
    thought-channel text from assistant messages. Records are written as token
    ids plus the completion offset expected by mlx-lm's prompt-masked loss.
    """
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(dataset_path)
    skipped: Counter[str] = Counter()
    tokenized: list[dict[str, Any]] = []
    for record in records:
        token_record, skip_reason = _native_token_record(
            record,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        if skip_reason is not None:
            skipped[skip_reason] += 1
            continue
        assert token_record is not None
        tokenized.append(token_record)

    if not tokenized:
        raise ValueError(
            "native MLX data preparation kept zero examples; "
            f"skipped_record_counts={dict(skipped)}"
        )

    random.Random(shuffle_seed).shuffle(tokenized)
    train_records, valid_records = _split_records(
        tokenized,
        valid_fraction=valid_fraction,
    )
    _write_raw_jsonl(out_dir / "train.jsonl", train_records)
    _write_raw_jsonl(out_dir / "valid.jsonl", valid_records)

    lengths = [int(record["n_total_tokens"]) for record in tokenized]
    completion_lengths = [int(record["n_completion_tokens"]) for record in tokenized]
    report = {
        "format": "pretokenized_prompt_completion",
        "max_seq_length": max_seq_length,
        "n_input_records": len(records),
        "n_kept_records": len(tokenized),
        "n_train_records": len(train_records),
        "n_valid_records": len(valid_records),
        "skipped_record_counts": dict(skipped),
        "max_total_tokens": max(lengths),
        "max_completion_tokens": max(completion_lengths),
    }
    (out_dir / "native_mlx_data_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Native MLX data: "
        f"kept {len(tokenized)}/{len(records)} examples; "
        f"skipped {sum(skipped.values())}; "
        f"skipped_record_counts={dict(skipped)}; "
        f"max_total_tokens={report['max_total_tokens']}; "
        f"max_completion_tokens={report['max_completion_tokens']}",
        file=sys.stderr,
        flush=True,
    )
    return report


def _warn_if_tokenizer_mismatch(manifest_path: Path, base_model: str) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tokenizer_id = manifest.get("tokenizer_id")
        if tokenizer_id and str(tokenizer_id) != base_model:
            print(
                "WARNING: dataset manifest tokenizer_id "
                f"{tokenizer_id!r} differs from base_model {base_model!r}; "
                "continuing because compatible models can share tokenizers.",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001 - manifest checks are advisory for MLX.
        print(
            f"WARNING: could not check dataset manifest {manifest_path}: {exc}",
            file=sys.stderr,
        )


def build_lora_cmd(
    *,
    python_exe: str,
    base_model: str,
    data_dir: Path,
    out_adapter_dir: Path,
    num_layers: int,
    iters: int,
    batch_size: int,
    learning_rate: float,
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int | None = None,
    save_every: int | None = None,
    val_batches: int | None = None,
    mask_prompt: bool = True,
    max_seq_length: int = 8192,
) -> list[str]:
    cmd = [
        python_exe,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        base_model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(out_adapter_dir),
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(num_layers),
        "--learning-rate",
        str(learning_rate),
        "--max-seq-length",
        str(max_seq_length),
    ]

    if mask_prompt:
        cmd.append("--mask-prompt")

    if wandb_project is not None:
        cmd.extend(["--report-to", "wandb", "--project-name", wandb_project])

    # Confirm these flag names with `python -m mlx_lm lora --help` for the
    # installed mlx-lm; pass only when set so an unknown flag cannot break a
    # default run.
    if steps_per_eval is not None:
        cmd.extend(["--steps-per-eval", str(steps_per_eval)])
    if steps_per_report is not None:
        cmd.extend(["--steps-per-report", str(steps_per_report)])
    if save_every is not None:
        cmd.extend(["--save-every", str(save_every)])
    if val_batches is not None:
        cmd.extend(["--val-batches", str(val_batches)])

    return cmd


def _run_mlx_lora(
    *,
    data_dir: Path,
    base_model: str,
    out_adapter_dir: Path,
    num_layers: int,
    iters: int,
    batch_size: int,
    learning_rate: float,
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int | None = None,
    save_every: int | None = None,
    val_batches: int | None = None,
    mask_prompt: bool = True,
    max_seq_length: int = 8192,
) -> None:
    subprocess.run(
        build_lora_cmd(
            python_exe=sys.executable,
            base_model=base_model,
            data_dir=data_dir,
            out_adapter_dir=out_adapter_dir,
            num_layers=num_layers,
            iters=iters,
            batch_size=batch_size,
            learning_rate=learning_rate,
            wandb_project=wandb_project,
            steps_per_eval=steps_per_eval,
            steps_per_report=steps_per_report,
            save_every=save_every,
            val_batches=val_batches,
            mask_prompt=mask_prompt,
            max_seq_length=max_seq_length,
        ),
        check=True,
    )


class _PreTokenizedExample:
    def __init__(self, record: dict[str, Any]):
        self.record = record

    def __len__(self) -> int:
        return len(self.record["input_ids"])


class _PreTokenizedDataset:
    def __init__(self, records: list[dict[str, Any]], *, mask_prompt: bool = True):
        self._records = [_PreTokenizedExample(record) for record in records]
        self._mask_prompt = mask_prompt

    def process(self, example: _PreTokenizedExample) -> tuple[list[int], int]:
        record = example.record
        offset = int(record["offset"]) if self._mask_prompt else 0
        return list(record["input_ids"]), offset

    def __getitem__(self, index: int) -> _PreTokenizedExample:
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)


def _load_pretokenized_dataset(
    path: Path,
    *,
    mask_prompt: bool = True,
) -> _PreTokenizedDataset:
    if not path.exists():
        return _PreTokenizedDataset([], mask_prompt=mask_prompt)
    return _PreTokenizedDataset(_load_jsonl(path), mask_prompt=mask_prompt)


def _has_gemma_thought_completion(dataset_path: Path) -> bool:
    for record in _load_jsonl(dataset_path):
        completion = str(record.get("completion", ""))
        if any(marker in completion for marker in _GEMMA_THOUGHT_MARKERS):
            return True
    return False


def _run_mlx_lora_native(
    *,
    dataset_path: Path,
    data_dir: Path,
    base_model: str,
    out_adapter_dir: Path,
    num_layers: int,
    iters: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int | None = None,
    save_every: int | None = None,
    val_batches: int | None = None,
    valid_fraction: float = 0.1,
    mask_prompt: bool = True,
) -> None:
    del wandb_project  # Native path uses mlx-lm's trainer directly; logging is not wired yet.
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.utils import load

    print("Loading pretrained model")
    model, tokenizer = load(base_model, tokenizer_config={"trust_remote_code": True})
    prepare_native_mlx_data(
        dataset_path,
        data_dir,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        valid_fraction=valid_fraction,
    )
    train_set = _load_pretokenized_dataset(
        data_dir / "train.jsonl",
        mask_prompt=mask_prompt,
    )
    valid_set = _load_pretokenized_dataset(
        data_dir / "valid.jsonl",
        mask_prompt=mask_prompt,
    )

    cfg = deepcopy(CONFIG_DEFAULTS)
    cfg.update(
        {
            "model": base_model,
            "train": True,
            "fine_tune_type": "lora",
            "data": str(data_dir),
            "num_layers": num_layers,
            "batch_size": batch_size,
            "iters": iters,
            "learning_rate": learning_rate,
            "adapter_path": str(out_adapter_dir),
            "max_seq_length": max_seq_length,
            "mask_prompt": mask_prompt,
            "steps_per_eval": steps_per_eval
            if steps_per_eval is not None
            else CONFIG_DEFAULTS["steps_per_eval"],
            "steps_per_report": steps_per_report
            if steps_per_report is not None
            else CONFIG_DEFAULTS["steps_per_report"],
            "save_every": save_every
            if save_every is not None
            else CONFIG_DEFAULTS["save_every"],
            "val_batches": val_batches
            if val_batches is not None
            else CONFIG_DEFAULTS["val_batches"],
            "project_name": None,
            "report_to": None,
        }
    )
    args = types.SimpleNamespace(**cfg)

    print("Training")
    train_model(args, model, train_set, valid_set)


def train(
    dataset_path: Path,
    base_model: str,
    out_adapter_dir: Path,
    *,
    num_layers: int = 8,
    iters: int = 200,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    valid_fraction: float = 0.1,
    data_dir: Path | None = None,
    manifest_path: Path | None = None,
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int | None = None,
    save_every: int | None = None,
    val_batches: int | None = None,
    mask_prompt: bool = True,
    max_seq_length: int = 8192,
) -> Path:
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("install .[train-mlx]") from exc

    dataset_path = Path(dataset_path)
    out_adapter_dir = Path(out_adapter_dir)
    out_adapter_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is not None:
        _warn_if_tokenizer_mismatch(Path(manifest_path), base_model)

    if data_dir is not None:
        prepared_data_dir = Path(data_dir)
        if _has_gemma_thought_completion(dataset_path):
            _run_mlx_lora_native(
                dataset_path=dataset_path,
                data_dir=prepared_data_dir,
                base_model=base_model,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                wandb_project=wandb_project,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                valid_fraction=valid_fraction,
                mask_prompt=mask_prompt,
            )
        else:
            prepared_data_dir = prepare_mlx_data(
                dataset_path,
                prepared_data_dir,
                valid_fraction=valid_fraction,
            )
            _run_mlx_lora(
                data_dir=prepared_data_dir,
                base_model=base_model,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                wandb_project=wandb_project,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                mask_prompt=mask_prompt,
                max_seq_length=max_seq_length,
            )
        return out_adapter_dir

    with tempfile.TemporaryDirectory(prefix="mlx_data_", dir=out_adapter_dir) as tmp:
        prepared_data_dir = Path(tmp)
        if _has_gemma_thought_completion(dataset_path):
            _run_mlx_lora_native(
                dataset_path=dataset_path,
                data_dir=prepared_data_dir,
                base_model=base_model,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                wandb_project=wandb_project,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                valid_fraction=valid_fraction,
                mask_prompt=mask_prompt,
            )
        else:
            prepared_data_dir = prepare_mlx_data(
                dataset_path,
                prepared_data_dir,
                valid_fraction=valid_fraction,
            )
            _run_mlx_lora(
                data_dir=prepared_data_dir,
                base_model=base_model,
                out_adapter_dir=out_adapter_dir,
                num_layers=num_layers,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                wandb_project=wandb_project,
                steps_per_eval=steps_per_eval,
                steps_per_report=steps_per_report,
                save_every=save_every,
                val_batches=val_batches,
                mask_prompt=mask_prompt,
                max_seq_length=max_seq_length,
            )
    return out_adapter_dir
