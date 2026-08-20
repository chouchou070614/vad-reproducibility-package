from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ncbaf_ops_v2"))

from trajectory_sampling import DeterministicPrioritySampler
from production_contract import (
    EVIDENCE_FAMILY,
    TASK_ID,
    tag_evidence_identity,
    validate_outer_manifest_gate,
)
from ncbaf_v2.ncbaf_pipeline import discover_records, infer_family
from production_ncbaf_v2_runner import find_valid_trajectory, verify_trajectory_task


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(start, count, spatial_width=4):
    index = np.arange(start, start + count, dtype=np.int64)
    return {
        "global_query_index": index,
        "dataset_index": index // 16,
        "batch_index": np.full(count, start // max(count, 1), dtype=np.int64),
        "batch_item_index": (index // 16) % 4,
        "spatial_y": (index % 16) // spatial_width,
        "spatial_x": (index % 16) % spatial_width,
    }


class SamplingTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_chunk_invariant(self):
        rng = np.random.RandomState(7)
        queries = rng.randn(100, 8).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        one = DeterministicPrioritySampler(17, 8, 123)
        one.observe(queries, metadata(0, 100))
        chunked = DeterministicPrioritySampler(17, 8, 123)
        chunked.observe(queries[:31], metadata(0, 31))
        chunked.observe(queries[31:], metadata(31, 69))
        centers = queries[:5]
        first = one.finalize(centers)
        second = chunked.finalize(centers)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_nearest_slot_and_distances_use_actual_centers(self):
        queries = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        centers = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        sampler = DeterministicPrioritySampler(4, 2, 9)
        sampler.observe(queries, metadata(0, 4, spatial_width=2))
        result = sampler.finalize(centers)
        expected = np.argmax(np.matmul(result["queries"], centers.T), axis=1)
        np.testing.assert_array_equal(result["nearest_slot"], expected)
        row = np.arange(4)
        expected_cosine = 1.0 - np.matmul(result["queries"], centers.T)[row, expected]
        np.testing.assert_allclose(result["nearest_cosine_distance"], expected_cosine)
        np.testing.assert_allclose(
            result["nearest_l2_distance"],
            np.linalg.norm(result["queries"] - centers[expected], axis=1),
        )


class ContractTextTests(unittest.TestCase):
    def test_training_contract_has_every_real_stage(self):
        source = (HERE / "ped2_Train_Pred_trajectory.py").read_text(encoding="utf-8")
        for token in (
            "01_random_initial_memory.pt",
            "02_warmup_%d_pre_kmeans_memory.pt",
            "03_warmup_%d_actual_kmeans_centers.pt",
            "03b_warmup_%d_raw_kmeans_centers_before_normalization.pt",
            "04_warmup_%d_post_replacement_memory.pt",
            "05_final_memory.pt",
            "warmup_%d_normal_query_trace.npz",
            '"final_checkpoint": "../model.pth"',
        ):
            self.assertIn(token, source)
        self.assertIn("must be KMemory-Continue, not Freeze", source)
        self.assertIn('"family": "KMemory"', source)
        self.assertIn('"memory_update_policy": "Continue"', source)
        self.assertIn('"query_source_split": "normal_training"', source)
        self.assertIn('"kmeans_uses_test_labels": False', source)
        self.assertIn('trajectory_state["sha256"]', source)

    def test_queue_has_both_b03_gates_and_exclusive_lock(self):
        source = (HERE / "queue_after_b03.sh").read_text(encoding="utf-8")
        self.assertIn("B03_shanghai_baseline_R_seed2022", source)
        self.assertIn("B03_shanghai_baseline_P_seed2022", source)
        self.assertIn("flock -n 9", source)
        self.assertIn("nvidia-smi --query-compute-apps=pid", source)
        self.assertNotIn("freeze_memory_after_kmeans", source)

    def test_exact_conditional_task_identity_is_used_everywhere(self):
        self.assertEqual(
            TASK_ID, "COND01_ped2_P_kmemory_continue_trajectory_seed2020"
        )
        card = json.loads(
            (HERE / "dashboard_card.template.json").read_text(encoding="utf-8")
        )
        self.assertEqual(card["task_id"], TASK_ID)
        queue = (HERE / "queue_after_b03.sh").read_text(encoding="utf-8")
        self.assertIn(TASK_ID, queue)
        self.assertNotIn("S05_ped2_P_kmemory_continue_trajectory_seed2020", queue)


class ProductionCompatibilityTests(unittest.TestCase):
    def test_exported_evidence_is_explicitly_kmemory_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / TASK_ID
            scores = task / "scores"
            scores.mkdir(parents=True)
            for split in ("training", "testing"):
                path = scores / (split + "_raw_evidence.npz")
                numeric_evidence = np.asarray([0.125, 0.5, 0.875], dtype=np.float64)
                np.savez_compressed(
                    path,
                    schema_version=np.asarray("raw-evidence-v1"),
                    dataset=np.asarray("ped2"),
                    branch=np.asarray("P"),
                    split=np.asarray(split),
                    seed=np.asarray(2020, dtype=np.int64),
                    pixel_mse=numeric_evidence,
                )
                path.with_suffix(".profile.json").write_text(
                    json.dumps({"split": split}) + "\n", encoding="utf-8"
                )
                tag_evidence_identity(path, TASK_ID)
                with np.load(path, allow_pickle=False) as evidence:
                    metadata = {name: evidence[name] for name in evidence.files}
                self.assertEqual(str(metadata["family"].item()), EVIDENCE_FAMILY)
                self.assertEqual(infer_family(path, metadata), "KMemoryTrajectory")
                self.assertEqual(str(metadata["task_id"].item()), TASK_ID)
                np.testing.assert_array_equal(metadata["pixel_mse"], numeric_evidence)

            records, rows = discover_records(Path(directory))
            self.assertEqual(records, {})  # trajectory evidence is auxiliary, not a core R/P pair
            discovered = [row for row in rows if row.get("status") == "DISCOVERED"]
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0]["family"], "KMemoryTrajectory")

    def test_manifest_and_validation_pass_the_real_outer_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "11_supplement" / TASK_ID
            trajectory = task / "artifacts" / "memory_trajectory"
            trajectory.mkdir(parents=True)
            (task / "metrics").mkdir(parents=True)

            relative_files = {
                "warmup_pre_kmeans_memory": "02_warmup_26_pre_kmeans_memory.pt",
                "actual_kmeans_centers": "03_warmup_26_actual_kmeans_centers.pt",
                "post_replacement_memory": "04_warmup_26_post_replacement_memory.pt",
                "final_memory": "05_final_memory.pt",
                "warmup_query_trace": "warmup_26_normal_query_trace.npz",
                "final_checkpoint": "../model.pth",
            }
            hashes = {}
            for index, (key, relative) in enumerate(relative_files.items()):
                artifact = (trajectory / relative).resolve()
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(("artifact-%d-%s" % (index, key)).encode("ascii"))
                hashes[key] = file_hash(artifact)

            state_path = trajectory / "trajectory_state.json"
            state_path.write_text(
                json.dumps({
                    "schema_version": "memory-trajectory-v2",
                    "task_id": TASK_ID,
                    "dataset": "ped2",
                    "family": "KMemory",
                    "branch": "P",
                    "seed": 2020,
                    "memory_update_policy": "Continue",
                    "warmup_epoch": 26,
                    "query_source_split": "normal_training",
                    "kmeans_uses_test_labels": False,
                    "same_training_trajectory": True,
                    "training_complete": True,
                    "files": relative_files,
                    "sha256": hashes,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            (task / "status.json").write_text(
                json.dumps({"task_id": TASK_ID, "state": "completed"}) + "\n",
                encoding="utf-8",
            )
            (task / "completeness_report.json").write_text(
                json.dumps({"complete": True}) + "\n", encoding="utf-8"
            )
            (task / "metrics/trajectory_validation.json").write_text(
                json.dumps({"state": "passed"}) + "\n", encoding="utf-8"
            )

            manifest_path = task / "manifest_sha256.csv"
            files = sorted(path for path in task.rglob("*") if path.is_file())
            with manifest_path.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(
                    target, fieldnames=("relative_path", "bytes", "sha256")
                )
                writer.writeheader()
                for path in files:
                    writer.writerow({
                        "relative_path": str(path.relative_to(task)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "sha256": file_hash(path),
                    })

            validate_outer_manifest_gate(task, manifest_path)
            found, errors = find_valid_trajectory(root)
            self.assertEqual(found, state_path)
            self.assertFalse([error for error in errors if "invalid:" in error])
            passed, evidence = verify_trajectory_task(found, root)
            self.assertTrue(passed, evidence)
            self.assertEqual(evidence["trajectory_validation"], "passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
