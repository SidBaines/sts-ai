"""Packed importer contracts, without MLX imports, weights, or GPU allocation."""
from __future__ import annotations

import ast
from copy import deepcopy
import unittest

from scripts.experiments.astra_stage_ollama_mlx import (
    CUSTOM_MODEL,
    mapped_shapes,
    quantization_from_header,
)


class QuantizationMetadataTest(unittest.TestCase):
    name = "model.language_model.layers.0.mlp.gate_proj.weight"

    def header(self, mode="nvfp4", group_size="16", scale_columns=4):
        return {
            "__metadata__": {"quant_type": mode, "group_size": group_size},
            self.name: {"dtype": "U32", "shape": [3, 8]},
            self.name + ".scale": {"dtype": "U8", "shape": [3, scale_columns]},
        }

    def test_nvfp4_and_mxfp8_keep_distinct_modes_and_group_sizes(self):
        self.assertEqual(quantization_from_header(self.name, self.header()),
                         {"mode": "nvfp4", "bits": 4, "group_size": 16})
        self.assertEqual(quantization_from_header(self.name, self.header("mxfp8", "32", 1)),
                         {"mode": "mxfp8", "bits": 8, "group_size": 32})

    def test_unquantized_header_has_no_quantization_override(self):
        self.assertIsNone(quantization_from_header(self.name,
            {self.name: {"dtype": "BF16", "shape": [3, 8]}}))

    def test_unknown_mode_or_wrong_group_size_fail_closed(self):
        for header in (self.header("affine"), self.header(group_size="32")):
            with self.subTest(header=header), self.assertRaises(ValueError):
                quantization_from_header(self.name, header)

    def test_wrong_scale_dtype_and_incompatible_shape_fail_closed(self):
        header = self.header()
        header[self.name + ".scale"]["dtype"] = "BF16"
        for candidate in (header, self.header(scale_columns=2)):
            with self.subTest(header=candidate), self.assertRaises(ValueError):
                quantization_from_header(self.name, candidate)

    def test_missing_scale_or_unexpected_bias_fail_closed(self):
        missing = self.header()
        del missing[self.name + ".scale"]
        bias = self.header()
        bias[self.name + ".bias"] = {"dtype": "BF16", "shape": [3, 4]}
        for candidate in (missing, bias):
            with self.subTest(header=candidate), self.assertRaises(ValueError):
                quantization_from_header(self.name, candidate)


class FusedPackedRemapTest(unittest.TestCase):
    @staticmethod
    def remap_function():
        # Execute only the generic pure function, not generated model imports.
        node = next(n for n in ast.parse(CUSTOM_MODEL).body if isinstance(n, ast.FunctionDef))
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "packed_remap_test", "exec"), namespace)
        return namespace["remap_packed_weights"]

    @staticmethod
    def split_rows(array, sections, axis):
        if sections != 2 or axis != -2:
            raise AssertionError("Expected two output-axis halves")
        half = len(array[0]) // 2
        return ([expert[:half] for expert in array], [expert[half:] for expert in array])

    def test_weights_and_scales_split_in_matching_gate_then_up_order(self):
        base = "model.language_model.layers.0.experts"
        weight = [[[11, 12], [13, 14], [21, 22], [23, 24]],
                  [[31, 32], [33, 34], [41, 42], [43, 44]]]
        scales = [[[101], [102], [201], [202]], [[301], [302], [401], [402]]]
        source = {base + ".gate_up_proj": weight, base + ".gate_up_proj.scale": scales}
        before = deepcopy(source)
        result = self.remap_function()(source, self.split_rows)
        self.assertEqual(result[base + ".switch_glu.gate_proj.weight"], [weight[0][:2], weight[1][:2]])
        self.assertEqual(result[base + ".switch_glu.up_proj.weight"], [weight[0][2:], weight[1][2:]])
        self.assertEqual(result[base + ".switch_glu.gate_proj.scales"], [scales[0][:2], scales[1][:2]])
        self.assertEqual(result[base + ".switch_glu.up_proj.scales"], [scales[0][2:], scales[1][2:]])
        self.assertEqual(source, before)

    def test_down_and_regular_projection_scale_names_preserve_arrays(self):
        names = {"model.language_model.layers.0.experts.down_proj": [1, 2],
                 "model.language_model.layers.0.experts.down_proj.scale": [3, 4],
                 "model.language_model.layers.0.mlp.gate_proj.weight.scale": [5, 6],
                 "model.language_model.layers.0.input_layernorm.weight": [7, 8]}
        result = self.remap_function()(names, self.split_rows)
        self.assertIs(result["model.language_model.layers.0.experts.switch_glu.down_proj.weight"], names[next(iter(names))])
        self.assertIs(result["model.language_model.layers.0.experts.switch_glu.down_proj.scales"], names["model.language_model.layers.0.experts.down_proj.scale"])
        self.assertEqual(result["model.language_model.layers.0.mlp.gate_proj.scales"], [5, 6])
        self.assertEqual(result["model.language_model.layers.0.input_layernorm.weight"], [7, 8])

    def test_packed_output_and_scale_shapes_split_only_output_axis(self):
        base = "model.language_model.layers.0.experts.gate_up_proj"
        weights = mapped_shapes(base, [128, 1408, 352])
        scales = mapped_shapes(base + ".scale", [128, 1408, 176])
        self.assertEqual([shape for _, shape in weights], [[128, 704, 352]] * 2)
        self.assertEqual([shape for _, shape in scales], [[128, 704, 176]] * 2)
        self.assertTrue(weights[0][0].endswith("switch_glu.gate_proj.weight"))
        self.assertTrue(scales[1][0].endswith("switch_glu.up_proj.scales"))


if __name__ == "__main__":
    unittest.main()
