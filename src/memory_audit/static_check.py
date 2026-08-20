from __future__ import annotations

import ast
import hashlib
import json
import py_compile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON_FILES = (
    "production_contract.py",
    "trajectory_sampling.py",
    "ped2_kmeans_memory_trajectory.py",
    "ped2_Train_Pred_trajectory.py",
    "ped2_run_memory_trajectory_task.py",
    "test_memory_trajectory_contract.py",
)

for name in PYTHON_FILES:
    path = HERE / name
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    py_compile.compile(str(path), doraise=True)

card = json.loads((HERE / "dashboard_card.template.json").read_text(encoding="utf-8"))
assert card["task_id"] == "COND01_ped2_P_kmemory_continue_trajectory_seed2020"
assert card["dataset"] == "ped2"
assert card["branch"] == "P"
assert card["family"] == "KMemory"
assert card["memory_update_policy"] == "Continue"
assert card["warmup"] == 26
assert card["epochs"] == 60

baselines = json.loads((HERE / "SOURCE_BASELINES.json").read_text(encoding="utf-8"))
assert set(baselines["sha256"]) == {
    "../ped2_Train_Pred.py",
    "../ped2_kmeans_memory.py",
    "../ped2_run_train_task.py",
}
for relative, expected in baselines["sha256"].items():
    source = (HERE / relative).resolve()
    if source.is_file():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected

runner = (HERE / "ped2_run_memory_trajectory_task.py").read_text(encoding="utf-8")
contract = (HERE / "production_contract.py").read_text(encoding="utf-8")
assert "EPOCHS = 60" in runner
assert "WARMUP = 26" in runner
assert '"query_source_split": "normal_training"' in runner
assert '"kmeans_uses_test_labels": False' in runner
assert "SEED = 2020" in runner
assert 'EVIDENCE_FAMILY = "KMemoryTrajectory"' in contract
assert "tag_evidence_identity(raw_output, args.task_id)" in runner
assert "validate_outer_manifest_gate(task_dir, manifest_path)" in runner
assert "--freeze_memory_after_kmeans" not in runner

queue = (HERE / "queue_after_b03.sh").read_text(encoding="utf-8")
assert 'TASK_ID="${TASK_ID:-COND01_ped2_P_kmemory_continue_trajectory_seed2020}"' in queue
assert "S05_ped2_P_kmemory_continue_trajectory_seed2020" not in queue

print("STATIC_CHECK_OK: %d Python files and dashboard contract passed" % len(PYTHON_FILES))
