"""Thin MLX LoRA trainer wrapper and data conversion helpers."""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["build_lora_cmd", "prepare_mlx_data", "train"]


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
    return {"messages": record["messages"]}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_mlx_record(record), ensure_ascii=False) + "\n")


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
        ),
        check=True,
    )


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
        prepared_data_dir = prepare_mlx_data(
            dataset_path,
            Path(data_dir),
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
        )
        return out_adapter_dir

    with tempfile.TemporaryDirectory(prefix="mlx_data_", dir=out_adapter_dir) as tmp:
        prepared_data_dir = prepare_mlx_data(
            dataset_path,
            Path(tmp),
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
        )
    return out_adapter_dir
