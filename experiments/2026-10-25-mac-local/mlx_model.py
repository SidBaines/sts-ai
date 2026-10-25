"""Build an MLX-loadable K2-Horizon-0.9B directory from the pinned Hugging Face snapshot.

K2-Horizon-0.9B (this revision) is a dense Llama layout: grouped-query attention,
pre-norm RMSNorm, SwiGLU MLP, no biases, untied embeddings and YaRN RoPE. The
script checks those assumptions against the original config, writes a Llama
config for mlx-lm, and symlinks the unchanged BF16 weights and tokenizer files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "IFM/K2-Horizon-0.9B"
REVISION = "7c75f2f49e74ac97c1852e62b4c860091d6a00ea"
LINKED = ("model-00000-of-00001.safetensors", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "chat_template.jinja", "generation_config.json")


def llama_config(hf: dict) -> dict:
    layers = hf["num_hidden_layers"]
    checks = {
        "dense layers only": hf.get("num_experts", 0) == 0 and set(hf.get("mlp_only_layers", [])) == set(range(layers)),
        "no mixture-of-attention": hf.get("mova_num_experts", 0) in (0, None),
        "no query/key norm": not hf.get("query_key_norm", False),
        "ungrouped RMSNorm": hf.get("layernorm_num_groups", 1) == 1,
        "full-width RoPE": hf.get("rope_head_dim") in (None, hf["head_dim"]),
        "no attention bias": not hf.get("attention_bias", False),
        "untied embeddings": not hf.get("tie_word_embeddings", False),
        "SiLU MLP": hf.get("hidden_act") == "silu",
    }
    rope = hf["rope_parameters"]
    checks["YaRN RoPE"] = rope.get("rope_type") == "yarn" and rope.get("truncate", True)
    # mlx-lm derives the YaRN attention factor as 0.1*ln(factor)+1.
    checks["default YaRN attention factor"] = math.isclose(
        rope.get("attention_factor", 0.1 * math.log(rope["factor"]) + 1), 0.1 * math.log(rope["factor"]) + 1,
        rel_tol=0, abs_tol=1e-9)
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"config is not a plain Llama layout: {failed}")
    return {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "hidden_size": hf["hidden_size"], "num_hidden_layers": layers,
        "intermediate_size": hf["intermediate_size"], "num_attention_heads": hf["num_attention_heads"],
        "num_key_value_heads": hf["num_key_value_heads"], "head_dim": hf["head_dim"],
        "rms_norm_eps": hf["rms_norm_eps"], "vocab_size": hf["vocab_size"],
        "max_position_embeddings": hf["max_position_embeddings"], "rope_theta": hf["rope_theta"],
        "rope_traditional": False, "attention_bias": False, "mlp_bias": False,
        "tie_word_embeddings": False,
        "rope_scaling": {"type": "yarn", "factor": rope["factor"],
                         "original_max_position_embeddings": rope["original_max_position_embeddings"],
                         "beta_fast": rope["beta_fast"], "beta_slow": rope["beta_slow"]},
        "bos_token_id": hf["bos_token_id"], "eos_token_id": hf["eos_token_id"],
        "pad_token_id": hf["pad_token_id"],
        "source": {"repo": REPO, "revision": REVISION, "weights": "unchanged BF16 safetensors"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    source = Path(snapshot_download(REPO, revision=args.revision))
    config = llama_config(json.loads((source / "config.json").read_text()))
    config["source"]["revision"] = args.revision
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in LINKED:
        os.symlink((source / name).resolve(), args.output / name)
    print(json.dumps({"output": str(args.output), "source": str(source)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
