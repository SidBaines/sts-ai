"""Download the frozen HF model or verify local generation + one QLoRA update."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
from pathlib import Path

from sts_ai.fight_lab import MODEL_ID, MODEL_REVISION, read_json, write_new
from sts_ai.provenance import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "check"])
    parser.add_argument("--output", type=Path, required=True, help="Fresh manifest/report path")
    parser.add_argument("--adapter-output", type=Path,
                        help="Fresh directory for the disposable one-step smoke adapter")
    args = parser.parse_args()
    if args.output.exists() or (args.adapter_output and args.adapter_output.exists()):
        parser.error("outputs must be fresh")
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                    local_files_only=args.command == "check", max_workers=3))
    report = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(snapshot),
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                        for p in sorted(snapshot.iterdir()) if p.is_file()},
              "versions": {p: importlib.metadata.version(p) for p in
                           ("mlx", "mlx-lm", "huggingface-hub", "transformers")}}
    if args.command == "download":
        write_new(args.output, report)
        print(f"Downloaded and hashed {snapshot}", flush=True)
        return
    if args.adapter_output is None:
        parser.error("check requires --adapter-output")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers

    print("Loading frozen model", flush=True)
    model, tokenizer = load(str(snapshot))
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Reply with just the word ready."}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
    mx.random.seed(0)
    a = generate(model, tokenizer, prompt=prompt, max_tokens=16, sampler=make_sampler(temp=0))
    mx.random.seed(0)
    b = generate(model, tokenizer, prompt=prompt, max_tokens=16, sampler=make_sampler(temp=0))
    if a != b or not a.strip():
        raise RuntimeError("greedy generation repeat failed")
    print(f"Greedy repeat: {a!r}", flush=True)
    cfg = read_json(Path("configs/competence/gemma4_26b_local.json"))["initial_lora"]
    mx.random.seed(cfg["seed"])
    model.freeze()
    linear_to_lora_layers(model, cfg["num_layers"], cfg["lora_parameters"])
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if not trainable or any("lora_" not in k for k in trainable):
        raise RuntimeError("unexpected trainable parameters")
    before = {k: mx.array(v) for k, v in trainable.items()}
    mx.eval(before)
    ids = tokenizer.encode(prompt + "ready" + tokenizer.eos_token)
    inputs = mx.array([ids[:-1]])
    targets = mx.array([ids[1:]])
    start = max(0, len(tokenizer.encode(prompt)) - 1)

    def loss_fn(m):
        logits = m(inputs)
        return nn.losses.cross_entropy(logits[:, start:], targets[:, start:], reduction="mean")

    model.train()
    loss, grad = nn.value_and_grad(model, loss_fn)(model)
    optimizer = optim.Adam(learning_rate=cfg["learning_rate"])
    optimizer.update(model, grad)
    mx.eval(loss, model.trainable_parameters(), optimizer.state)
    changed = sum(bool(mx.any(v != before[k]).item()) for k, v in tree_flatten(model.trainable_parameters()))
    if not bool(mx.isfinite(loss).item()) or not changed:
        raise RuntimeError("QLoRA step failed: nonfinite loss or unchanged weights")
    args.adapter_output.mkdir(parents=True)
    mx.save_safetensors(str(args.adapter_output / "adapters.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())))
    write_new(args.adapter_output / "adapter_config.json", cfg)
    report.update({"greedy_response": a, "greedy_repeat_equal": True,
                   "smoke_loss": loss.item(), "changed_adapter_tensors": changed,
                   "trainable_parameters": sum(v.size for v in trainable.values()),
                   "peak_memory_gb": mx.get_peak_memory() / 1e9, "lora_config": cfg,
                   "smoke_sequence_tokens": len(ids), "smoke_gradient_checkpointing": False,
                   "note": "Synthetic one-step mechanics check; not a combat-trained adapter."})
    del model, optimizer, grad, trainable, before
    gc.collect()
    mx.clear_cache()
    print("Reloading smoke adapter", flush=True)
    model, tokenizer = load(str(snapshot), adapter_path=str(args.adapter_output))
    report["adapter_reload_response"] = generate(model, tokenizer, prompt=prompt, max_tokens=16,
                                                  sampler=make_sampler(temp=0))
    write_new(args.output, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
