"""Bounded CPU checks for the new Nsight contract. Run on allocated compute."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from profiling import run_pretrain_ep_overlap as adapter
from profiling.ampa_profile.overlap_trace import OverlapMethodLabels
from tools import ep_overlap_nsys_posthoc as posthoc


OWNER = (1 << 48) + (123 << 24)
UUID = "GPU-" + "a" * 32


def execution():
    from types import SimpleNamespace
    return adapter.execution_contract(SimpleNamespace(execution_steps=10, overlap_capture="nsys",
        case="uccl_ordinary", communication_backend="uccl_ep", communication_sms=20, overlap_source_sha256="b" * 64))


def journal(rank=0, step=5):
    observer = OverlapMethodLabels(None, [], communication_backend="uccl_ep")
    return {"schema_version": 1, "kind": "ampa_ep_overlap_calls", "rank": rank, "step": step,
        "capture_mode": "nsys", "skipped_iter": False, "layers": list(range(48)),
        "observation_contract": observer.contract_record(4), "counts": dict(observer.expected_counts(4)),
        "payload_bytes": {}, "payload_dtypes": {}, "uccl_autograd_pairs": [
            {"call_id": index, "layer": index // 8, "family": "dispatch" if index % 2 == 0 else "combine",
             "context_id": index + 1, "epoch": 1, "forward_returned": True, "backward_returned": True}
            for index in range(384)]}


def api_database(*, complete=True):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
    db.executemany("INSERT INTO StringIds VALUES(?,?)", [(1, "uccl::internode::dispatch<test>"), (2, "cuLaunchKernelEx")])
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER)")
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(20,30,?,7,1)", (OWNER,))
    fields = "start INTEGER,globalTid INTEGER,correlationId INTEGER" + (",end INTEGER,nameId INTEGER,returnValue INTEGER" if complete else "")
    db.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER({fields})")
    if complete:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(10,?,7,15,2,0)", (OWNER + 1,))
    else:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(10,?,7)", (OWNER + 1,))
    return db


def full_node_fixture(path):
    """Synthetic event identity only, with the complete required node grid."""
    training = {"run_dir": str(path.parent / "training"), "communication_sms": 20,
        "training_slurm_job_id": "123456", "worker_source_path": str(posthoc.ROOT / "jobs/training/ep_overlap_worker.sh"),
        "worker_source_sha256": "e" * 64, "rank_placement": []}
    parent = {"schema_version": 1, "kind": "ampa_nsight_torchrun_parent", "job": "123456",
        "node_rank": 0, "role": "torchrun_launcher", "source_path": training["worker_source_path"],
        "source_sha256": training["worker_source_sha256"], "pid": 100,
        "command": ["/opt/venvs/megatron-bridge/bin/python", "-m", "torch.distributed.run", "--nnodes=4",
            "--nproc_per_node=4", "--node_rank=0", "--master_addr=host", "--master_port=12345", "--max_restarts=0",
            f"--log_dir={Path(training['run_dir']) / 'logs/torchrun-node-0'}", "--redirects=3", "--no-python",
            "bash", training["worker_source_path"], "--worker"]}
    expected, nvtx, kernels, launches, targets = {}, [], [], [], []
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
        db.executemany("INSERT INTO StringIds VALUES(?,?)", [(1, "uccl::internode::dispatch<test>"),
            (2, "uccl::internode::combine<test>"), (3, "cuLaunchKernelEx")])
        db.execute("CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,deviceId INTEGER,contextId INTEGER,correlationId INTEGER,demangledName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER,returnValue INTEGER)")
        db.execute("CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER,cudaId INTEGER,pid INTEGER,uuid TEXT)")
        db.execute("CREATE TABLE TARGET_INFO_GPU(id INTEGER,uuid TEXT)")
        for rank in range(4):
            uuid = f"GPU-{rank + 1:032x}"
            training["rank_placement"].append({"rank": rank, "slurm_node_id": 0, "physical_gpu_uuid": uuid})
            targets.append((rank, rank, 101 + rank, uuid))
            owner = (1 << 48) + ((101 + rank) << 24)
            expected[rank] = {}
            for step in range(5, 10):
                record = journal(rank, step)
                expected[rank][step] = posthoc.checked_journal(record, rank=rank, step=step, execution=execution())
                start, end = step * 10_000_000, step * 10_000_000 + 9_000_000
                nvtx.append((start, end, owner + 1, f"AMPA/STEP/{step}"))
                cursor = start + 10
                for name, count in record["counts"].items():
                    for _ in range(count):
                        nvtx.append((cursor, cursor + 1, owner + 1, name))
                        cursor += 2
                for pair in record["uccl_autograd_pairs"]:
                    for direction in ("FWD", "BWD"):
                        native_family = ("dispatch" if (pair["family"] == "dispatch") == (direction == "FWD") else "combine")
                        correlation = step * 10000 + pair["call_id"] * 2 + (direction == "BWD")
                        nvtx.append((cursor, cursor + 8, owner + 1, f"AMPA/UCCL_CALL/{pair['call_id']}/{direction}"))
                        launches.append((cursor + 1, cursor + 2, owner + 1, correlation, 3, 0))
                        kernels.append((cursor + 2, cursor + 3, owner, rank, 1, correlation,
                            1 if native_family == "dispatch" else 2, 20, 1, 1, 256, 1, 1))
                        cursor += 10
        db.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)", nvtx)
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", kernels)
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(?,?,?,?,?,?)", launches)
        db.executemany("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(?,?,?,?)", targets)
        db.executemany("INSERT INTO TARGET_INFO_GPU VALUES(?,?)", [(row[0], row[3]) for row in targets])
    return training, parent, expected


class OverlapNsightPosthocTests(unittest.TestCase):
    def test_unchanged_sql_reader_accepts_new_full_node_grid_and_rejects_missing_node_scope(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "node.sqlite"
            training, parent, expected = full_node_fixture(path)
            # Ordinary-schedule journal/count construction is forbidden here.
            with patch.object(posthoc.sql, "journal", side_effect=AssertionError("old journal called")):
                result = posthoc.sql.inspect_sqlite(path, 0, training, expected, parent)
            self.assertTrue(result["valid_timeline_contents"], result.get("timeline_error"))
            self.assertEqual(result["payload_kernel_count"], 4 * 5 * 768)
            self.assertEqual(result["exact_paired_payload_directions"], 4 * 5 * 768)
            self.assertEqual(result["whole_session_event_completeness"], "unresolved")
            with sqlite3.connect(path) as db:
                db.execute("DELETE FROM NVTX_EVENTS WHERE rowid=(SELECT rowid FROM NVTX_EVENTS WHERE text LIKE 'AMPA/OVERLAP/%' LIMIT 1)")
            failed = posthoc.sql.inspect_sqlite(path, 0, training, expected, parent)
            self.assertFalse(failed["valid_timeline_contents"])
            self.assertIn("NVTX scopes", failed["timeline_error"])

    def test_new_node_contract_constructs_all_768_pair_scopes(self):
        record = journal()
        result = posthoc.checked_journal(record, rank=0, step=5, execution=execution())
        expected = Counter(record["counts"])
        expected.update({f"AMPA/UCCL_CALL/{index}/{direction}": 1 for index in range(384) for direction in ("FWD", "BWD")})
        self.assertEqual(result["nvtx_counts"], expected)
        self.assertEqual(len(result["call_families"]), 384)
        self.assertTrue(any(name.startswith("AMPA/OVERLAP/") for name in expected))

    def test_old_schedule_missing_node_and_bad_pair_return_are_rejected(self):
        original = journal()
        changes = []
        old = deepcopy(original)
        old["counts"] = {"AMPA/TRAIN/microbatch_forward": 4}
        changes.append(old)
        missing = deepcopy(original)
        missing["counts"].pop(next(name for name in missing["counts"] if name.startswith("AMPA/OVERLAP/")))
        changes.append(missing)
        failed = deepcopy(original)
        failed["uccl_autograd_pairs"][1]["backward_returned"] = False
        changes.append(failed)
        for record in changes:
            with self.subTest(counts=len(record["counts"])), self.assertRaises((ValueError, RuntimeError)):
                posthoc.checked_journal(record, rank=0, step=5, execution=execution())

    def test_execution_prerequisite_cannot_be_replaced_by_nonempty_reports(self):
        proof = {"valid_execution_and_accounting": False}
        inputs = Mock()
        inputs.read.return_value = proof
        with self.assertRaises((ValueError, RuntimeError)):
            posthoc.checked_training("/proof.json", "/acquisition", inputs)
        inputs.bind.assert_not_called()
        claimed = {key: True for key in ("valid_execution_and_accounting", "valid_training_health",
            "valid_send_receive_audit", "valid_all_rank_startup_and_completion")}
        claimed.update(kind="ampa_ep_overlap_execution_accounting_validation", orchestration_dir="/acquisition",
            qualification_only=False, case="uccl_ordinary", capture="nsys", rank_placement=[{"rank": 9}])
        inputs.read.return_value = claimed
        with self.assertRaisesRegex((ValueError, RuntimeError), "fresh strict execution"):
            posthoc.checked_training("/proof.json", "/acquisition", inputs)
        actual = {**claimed, "rank_placement": [{"rank": 1}]}
        with self.assertRaisesRegex((ValueError, RuntimeError), "rank_placement"):
            posthoc.checked_training("/proof.json", "/acquisition", inputs, revalidated_record=actual)
        for key in ("training_slurm_job_id", "run_dir", "source_manifest_sha256", "training_worker_sha256"):
            actual = {**claimed, key: "raw-actual"}
            with self.subTest(key=key), self.assertRaisesRegex((ValueError, RuntimeError), key):
                posthoc.checked_training("/proof.json", "/acquisition", inputs, revalidated_record=actual)

    def test_launch_success_and_end_fields_are_bound_to_payload_correlations(self):
        with api_database() as db:
            result = posthoc.launch_success(db, {"trainer_global_pids": [OWNER]})
        self.assertEqual(result["payload_launches"], 1)
        self.assertTrue(result["all_payload_launch_success_verified"])
        self.assertIsNone(result["unavailable_reason"])

    def test_failed_duplicate_and_nonlaunch_api_witnesses_are_rejected(self):
        for statement in ["UPDATE CUPTI_ACTIVITY_KIND_DRIVER SET returnValue=1",
                          "INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER SELECT * FROM CUPTI_ACTIVITY_KIND_DRIVER",
                          "UPDATE CUPTI_ACTIVITY_KIND_DRIVER SET end=start",
                          "UPDATE CUPTI_ACTIVITY_KIND_DRIVER SET start=25,end=26",
                          "UPDATE StringIds SET value='cuStreamWaitEvent' WHERE id=2"]:
            with self.subTest(statement=statement), api_database() as db:
                db.execute(statement)
                with self.assertRaises((ValueError, RuntimeError)):
                    posthoc.launch_success(db, {"trainer_global_pids": [OWNER]})

    def test_missing_api_status_fields_remain_unavailable(self):
        with api_database(complete=False) as db:
            result = posthoc.launch_success(db, {"trainer_global_pids": [OWNER]})
        self.assertFalse(result["all_payload_launch_success_verified"])
        self.assertEqual(result["successful_launches_with_end_fields"], 0)
        self.assertTrue(result["unavailable_reason"])

    def test_cuda_identity_uses_runtime_uuid_and_process_device_mapping(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER,cudaId INTEGER,pid INTEGER,uuid TEXT)")
            db.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(5,0,123,?)", (UUID,))
            db.execute("CREATE TABLE TARGET_INFO_GPU(id INTEGER,uuid TEXT)")
            db.execute("INSERT INTO TARGET_INFO_GPU VALUES(5,?)", (UUID,))
            db.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
            db.execute("INSERT INTO StringIds VALUES(1,'uccl::internode::dispatch<test>')")
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid INTEGER,deviceId INTEGER,demangledName INTEGER)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,0,1)", (OWNER,))
            # A process can enumerate another visible device without using it
            # for payload kernels. Match the observed device, not row count.
            db.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(8,1,123,?)", ("GPU-" + "d" * 32,))
            proof = {"rank_by_global_pid": {str(OWNER): 9}}
            placement = [{"rank": 9, "slurm_node_id": 2, "physical_gpu_uuid": UUID}]
            result = posthoc.cuda_identities(db, proof, placement, 2)
            self.assertEqual(result[OWNER]["rank"], 9)
            self.assertEqual(result[OWNER]["physical_gpu_id"], 5)
            self.assertEqual(result[OWNER]["uuid"], "a" * 32)
            placement[0]["physical_gpu_uuid"] = "GPU-" + "c" * 32
            with self.assertRaises((ValueError, RuntimeError)):
                posthoc.cuda_identities(db, proof, placement, 2)
            placement[0]["physical_gpu_uuid"] = UUID
            db.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE SELECT * FROM TARGET_INFO_CUDA_DEVICE WHERE cudaId=0")
            with self.assertRaises((ValueError, RuntimeError)):
                posthoc.cuda_identities(db, proof, placement, 2)


if __name__ == "__main__":
    unittest.main()
