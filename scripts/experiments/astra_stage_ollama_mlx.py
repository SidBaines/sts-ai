"""Stage cached Ollama MLX tensors without modifying/repacking source blobs.

CPU-only staging and packed-array checks. Actual MLX strict loading is a separate
explicit smoke command once the active GPU experiment finishes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
from typing import Any


CUSTOM_MODEL = '''from mlx_lm.models.gemma4 import Model as BaseModel, ModelArgs
import mlx.core as mx

def remap_packed_weights(weights, split):
    out = {}
    for name, value in weights.items():
        if name.endswith(".experts.gate_up_proj") or name.endswith(".experts.gate_up_proj.scale"):
            scales = name.endswith(".scale")
            base = name.removesuffix(".scale").removesuffix(".gate_up_proj")
            gate, up = split(value, 2, axis=-2)
            suffix = "scales" if scales else "weight"
            out[f"{base}.switch_glu.gate_proj.{suffix}"] = gate
            out[f"{base}.switch_glu.up_proj.{suffix}"] = up
        elif name.endswith(".experts.down_proj") or name.endswith(".experts.down_proj.scale"):
            scales = name.endswith(".scale")
            base = name.removesuffix(".scale").removesuffix(".down_proj")
            out[f"{base}.switch_glu.down_proj.{('scales' if scales else 'weight')}"] = value
        else:
            target = name.removesuffix(".weight.scale") + ".scales" if name.endswith(".weight.scale") else name
            out[target] = value
    return out

class Model(BaseModel):
    def sanitize(self, weights):
        # Split both packed values and their scales in the same gate/up order as
        # the installed Gemma4 sanitizer. No dequantization or requantization.
        return super().sanitize(remap_packed_weights(weights, mx.split))
'''


def header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length)), 8 + length


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(name: str) -> str:
    return name.replace("model.language_model.", "language_model.model.", 1)


def quantization_from_header(name: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Validate one source tensor's exact supported packed representation."""
    info = metadata.get("__metadata__", {})
    mode = info.get("quant_type")
    if mode is None:
        return None
    if mode not in ("nvfp4", "mxfp8"):
        raise ValueError(f"Unsupported quantization mode: {mode}")
    bits, group_size = (4, 16) if mode == "nvfp4" else (8, 32)
    if int(info.get("group_size", -1)) != group_size:
        raise ValueError(f"Wrong group size for {mode}")
    if set(metadata) != {"__metadata__", name, name + ".scale"}:
        raise ValueError("Packed tensor must have exactly its weight and scale arrays")
    weight, scale = metadata[name], metadata[name + ".scale"]
    if weight["dtype"] != "U32" or scale["dtype"] != "U8":
        raise ValueError("Packed weights/scales must be U32/U8")
    w, s = weight["shape"], scale["shape"]
    if not w or not s or w[:-1] != s[:-1] or w[-1] * (32 // bits) != s[-1] * group_size:
        raise ValueError("Packed weight and scale dimensions disagree")
    return {"bits": bits, "group_size": group_size, "mode": mode}


def mapped_shapes(name: str, shape: list[int]) -> list[tuple[str, list[int]]]:
    if ".experts.gate_up_proj" in name:
        scales = name.endswith(".scale")
        base = name.removesuffix(".scale").removesuffix(".gate_up_proj")
        half = list(shape)
        assert half[-2] % 2 == 0
        half[-2] //= 2
        return [(normalized(f"{base}.switch_glu.{part}.{('scales' if scales else 'weight')}"), half)
                for part in ("gate_proj", "up_proj")]
    if ".experts.down_proj" in name:
        scales = name.endswith(".scale")
        base = name.removesuffix(".scale").removesuffix(".down_proj")
        name = f"{base}.switch_glu.down_proj.{('scales' if scales else 'weight')}"
    elif name.endswith(".weight.scale"):
        name = name.removesuffix(".weight.scale") + ".scales"
    return [(normalized(name), list(shape))]


def expected_shapes(config: dict[str, Any]) -> dict[str, list[int]]:
    c = config["text_config"]
    d, m, e, f = c["hidden_size"], c["intermediate_size"], c["num_experts"], c["moe_intermediate_size"]
    expected = {"language_model.model.embed_tokens.weight": [c["vocab_size"], d],
                "language_model.model.norm.weight": [d]}
    for i, layer_type in enumerate(c["layer_types"]):
        p = f"language_model.model.layers.{i}."
        head = c["global_head_dim"] if layer_type == "full_attention" else c["head_dim"]
        kv = c["num_global_key_value_heads"] if layer_type == "full_attention" else c["num_key_value_heads"]
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
                     "post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"):
            expected[p + norm + ".weight"] = [d]
        expected.update({p + "layer_scalar": [1], p + "router.scale": [d], p + "router.per_expert_scale": [e],
                         p + "router.proj.weight": [e, d], p + "self_attn.q_norm.weight": [head],
                         p + "self_attn.k_norm.weight": [head], p + "self_attn.q_proj.weight": [c["num_attention_heads"] * head, d],
                         p + "self_attn.k_proj.weight": [kv * head, d],
                         p + "self_attn.o_proj.weight": [d, c["num_attention_heads"] * head],
                         p + "mlp.gate_proj.weight": [m, d], p + "mlp.up_proj.weight": [m, d], p + "mlp.down_proj.weight": [d, m],
                         p + "experts.switch_glu.gate_proj.weight": [e, f, d],
                         p + "experts.switch_glu.up_proj.weight": [e, f, d],
                         p + "experts.switch_glu.down_proj.weight": [e, d, f]})
        if layer_type != "full_attention":
            expected[p + "self_attn.v_proj.weight"] = [kv * head, d]
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/experiments/astra_5sep/gemma26_ollama"))
    parser.add_argument("--manifest", type=Path, default=Path.home() / ".ollama/models/manifests/registry.ollama.ai/library/gemma4/26b-mlx")
    parser.add_argument("--chat-template", type=Path, default=Path.home() / ".cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d/chat_template.jinja")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    blobs = Path.home() / ".ollama/models/blobs"
    manifest = json.loads(args.manifest.read_text())
    layers = manifest["layers"]
    files = {l.get("name"): blobs / l["digest"].replace(":", "-") for l in layers if "name" in l}
    config = json.loads(files["config.json"].read_text())
    quantization = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    shape_rows = {}; sources = []; modes = Counter(); raw_headers = {}
    for index, layer in enumerate(layers):
        name = layer.get("name", "")
        if layer["mediaType"] != "application/vnd.ollama.image.tensor" or not name.startswith("model.language_model."):
            continue
        source = files[name]
        metadata, data_start = header(source)
        raw_headers[name] = (source, metadata, data_start)
        q = metadata.get("__metadata__", {})
        representation = quantization_from_header(name, metadata)
        mode = representation["mode"] if representation else None
        if representation:
            w = metadata[name]["shape"]
            for target, _ in mapped_shapes(name, w):
                quantization[target.removesuffix(".weight")] = dict(representation)
        modes[mode or "unquantized"] += 1
        for key, entry in metadata.items():
            if key == "__metadata__":
                continue
            for target, shape in mapped_shapes(key, entry["shape"]):
                assert target not in shape_rows
                shape_rows[target] = {"shape": shape, "dtype": entry["dtype"], "source_name": key}
        filename = f"model-{index:05d}.safetensors"
        (args.out / filename).symlink_to(source)
        actual = sha256(source)
        assert actual == layer["digest"].split(":")[1], f"Source hash mismatch: {name}"
        sources.append({"name": name, "source": str(source), "symlink": filename, "sha256": actual,
                        "size": source.stat().st_size, "quantization_metadata": q})
    expected = expected_shapes(config)
    got_weights = set(shape_rows) - {key for key in shape_rows if key.endswith(".scales")}
    assert got_weights == set(expected), {"missing": sorted(set(expected) - got_weights), "extra": sorted(got_weights - set(expected))}
    for name, expected_shape in expected.items():
        actual = list(shape_rows[name]["shape"])
        q = quantization.get(name.removesuffix(".weight"))
        if q:
            actual[-1] *= 32 // q["bits"]
        assert actual == expected_shape, (name, actual, expected_shape)
    # Numerically compare the pure remapper with raw packed memmaps, touching
    # only sampled coordinates. No mlx import or GPU model allocation here.
    import ast
    import numpy as np
    function = next(node for node in ast.parse(CUSTOM_MODEL).body if isinstance(node, ast.FunctionDef))
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "remap_check", "exec"), namespace)
    samples = []
    for i in (0, 15, 29):
        name = f"model.language_model.layers.{i}.experts.gate_up_proj"
        source, metadata, data_start = raw_headers[name]
        arrays = {}
        for key in (name, name + ".scale"):
            entry = metadata[key]
            arrays[key] = np.memmap(source, dtype="<u4" if entry["dtype"] == "U32" else "u1", mode="r",
                offset=data_start + entry["data_offsets"][0], shape=tuple(entry["shape"]))
        mapped = namespace["remap_packed_weights"](arrays, np.split)
        for key, value in arrays.items():
            scales = key.endswith(".scale"); half = value.shape[-2] // 2
            for part, offset in (("gate_proj", 0), ("up_proj", half)):
                target = name.removesuffix(".gate_up_proj") + f".switch_glu.{part}." + ("scales" if scales else "weight")
                split = mapped[target]
                assert np.shares_memory(value, split)
                for expert, row, column in ((0, 0, 0), (64, half // 2, value.shape[-1] // 2), (127, half - 1, value.shape[-1] - 1)):
                    original = int(value[expert, row + offset, column]); observed = int(split[expert, row, column])
                    assert original == observed
                    samples.append({"target": normalized(target), "coordinate": [expert, row, column], "raw_packed_value": original, "split_value": observed})
        del arrays, mapped
    config["quantization"] = quantization
    config["model_file"] = "ollama_model.py"
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "ollama_model.py").write_text(CUSTOM_MODEL)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (args.out / name).symlink_to(files[name])
    (args.out / "chat_template.jinja").symlink_to(args.chat_template.resolve())
    report = {"kind": "ollama_cached_mlx_staging_v1", "source_manifest": str(args.manifest), "source_manifest_sha256": sha256(args.manifest),
        "sources": sources, "modes": dict(modes), "text_weight_files": len(sources), "sanitized_arrays": len(shape_rows),
        "shape_validation": "all normalized parameter names and unpacked shapes match installed gemma4 architecture", "numeric_packed_split_samples": samples,
        "custom_model_sha256": sha256(args.out / "ollama_model.py"), "config_sha256": sha256(args.out / "config.json"),
        "chat_template_source": str(args.chat_template), "chat_template_sha256": sha256(args.chat_template),
        "tokenizer_sources": {name: {"source": str(files[name]), "sha256": sha256(files[name])} for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json")},
        "strict_mlx_load": "NOT YET RUN: wait for active GPU suite to finish", "source_blobs_modified": False,
        "limitations": ["No dequantization/requantization; staging uses symlinks. Runtime split views may cause packed-array contiguity copies in kernels.", "Numeric packed-array equality verifies split/rename correctness, not end-to-end model output parity with Ollama.", "This mixed NVFP4/MXFP8 model is a distinct quantization arm from HF affine4bit.", "Draft and vision tensors excluded; text-only model interface."]}
    (args.out / "staging_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "sanitized_shapes.json").write_text(json.dumps(shape_rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "text_weight_files": len(sources), "modes": dict(modes), "shape_checks": len(expected), "numeric_samples": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
