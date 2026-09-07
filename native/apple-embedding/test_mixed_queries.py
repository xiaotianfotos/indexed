from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from validate_mixed_queries import (
    checked_embedding,
    normalize,
    parse_weights,
    rank_row,
    response_space,
    summarize,
)


class MixedQueryMetricTests(unittest.TestCase):
    def test_dataset_references_are_complete(self) -> None:
        root = Path(__file__).with_name("mixed-query-set")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        image_ids = {item["id"] for item in manifest["images"]}
        self.assertEqual(len(image_ids), 7)
        self.assertEqual(len(manifest["queries"]), 20)
        for item in manifest["images"]:
            self.assertTrue((root / item["path"]).is_file(), item["path"])
        for query in manifest["queries"]:
            self.assertIn(query["reference_image"], image_ids)
            self.assertIn(query["target_image"], image_ids)
            self.assertNotEqual(query["reference_image"], query["target_image"])

    def test_rank_and_group_summaries(self) -> None:
        corpus = np.eye(3, dtype=np.float32)
        query = {
            "id": "q",
            "edit_type": "color",
            "language": "zh",
            "reference_image": "source",
            "target_image": "target",
        }
        row = rank_row(
            query,
            np.asarray([0.1, 0.9, 0.0], dtype=np.float32),
            corpus,
            ["source", "target", "other"],
        )
        self.assertTrue(row["correct"])
        self.assertEqual(row["target_rank"], 1)
        self.assertGreater(row["target_minus_source"], 0)
        summary = summarize([row])
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["by_edit_type"]["color"]["correct"], 1)
        self.assertEqual(summary["by_language"]["zh"]["correct"], 1)

    def test_fusion_helpers(self) -> None:
        self.assertEqual(parse_weights("0.75,0.25,0.75"), [0.25, 0.75])
        vector = normalize(np.asarray([3.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_vllm_response_contract(self) -> None:
        response = {
            "model": "wemm-embedding-9b",
            "data": [{"embedding": [1.0, 0.0]}],
        }
        vector = checked_embedding(
            response,
            model="wemm-embedding-9b",
            dimensions=2,
        )
        np.testing.assert_array_equal(vector, np.asarray([1.0, 0.0]))
        self.assertEqual(
            response_space(response, "wemm-9b-bf16-2-fingerprint"),
            "wemm-9b-bf16-2-fingerprint",
        )
        with self.assertRaises(RuntimeError):
            response_space(response, None)
        with self.assertRaises(RuntimeError):
            checked_embedding(response, model="different-model", dimensions=2)
        with self.assertRaises(RuntimeError):
            checked_embedding(response, model="wemm-embedding-9b", dimensions=4)


if __name__ == "__main__":
    unittest.main()
