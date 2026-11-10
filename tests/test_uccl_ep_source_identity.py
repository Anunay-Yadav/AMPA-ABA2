"""CPU checks that a manifest cannot authorize arbitrary UCCL source edits."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import uccl_ep_source_identity as identity


APPROVED_PATCH = (Path(__file__).resolve().parents[1] / "ai/qwen-uccl-ep-comparison/iteration-01"
                  / "02-implement/uccl-borrowed-scratch.patch")


class UcclSourceIdentityTest(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with TemporaryDirectory() as temporary:
            build = Path(temporary)
            archive = build / identity.PATCH_ARCHIVE
            archive.parent.mkdir()
            archive.write_bytes(APPROVED_PATCH.read_bytes())
            state = {"base": identity.BASE_REVISION, "diff": b"fixed approved diff\n",
                     "status": "\n".join(" M " + path for path in identity.PATCH_FILES) + "\n"}
            bases = {}
            for relative, hunks in identity._patch_hunks(archive.read_bytes()).items():
                separator = b"\nindependent unchanged separator\n"
                bases[relative] = b"prefix\n" + separator.join(before for before, _ in hunks) + b"suffix\n"
                actual = b"prefix\n" + separator.join(after for _, after in hunks) + b"suffix\n"
                path = build / "uccl" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(actual)

            def git(source, *arguments):
                self.assertEqual(source, build / "uccl")
                if arguments == ("rev-parse", "HEAD"):
                    return (state["base"] + "\n").encode()
                if arguments[0] == "status":
                    return state["status"].encode()
                if arguments[0] == "show":
                    return bases[arguments[1].split(":", 1)[1]]
                if arguments[0] == "diff":
                    return state["diff"]
                self.fail(f"unexpected Git operation: {arguments}")

            with patch.object(identity, "_git", side_effect=git):
                yield build, archive, state, bases

    def test_exact_postimage_can_be_recorded_and_verified_without_overwriting(self):
        with self.fixture() as (build, archive, state, bases):
            result = identity.write_uccl_source_identity(build)
            self.assertEqual(set(result), {"base_revision", "patch_sha256", "source_diff_sha256", "modified_files"})
            self.assertEqual(result["base_revision"], identity.BASE_REVISION)
            self.assertEqual(result["patch_sha256"], identity.PATCH_SHA256)
            self.assertEqual(set(result["modified_files"]), set(identity.PATCH_FILES))
            self.assertEqual(identity.verify_uccl_source(build), result)
            with self.assertRaises(FileExistsError):
                identity.write_uccl_source_identity(build)

    def test_wrong_base_is_rejected(self):
        with self.fixture() as (build, archive, state, bases):
            identity.write_uccl_source_identity(build)
            state["base"] = "0" * 40
            with self.assertRaisesRegex(ValueError, "base revision"):
                identity.verify_uccl_source(build)

    def test_changed_archive_is_rejected_even_with_updated_manifest(self):
        with self.fixture() as (build, archive, state, bases):
            record = identity.write_uccl_source_identity(build)
            archive.write_bytes(archive.read_bytes() + b"extra\n")
            record["patch_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            (build / "source_identity.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "approved patch"):
                identity.verify_uccl_source(build)

    def test_extra_untracked_staged_or_missing_modifications_are_rejected(self):
        for fault in ("extra", "staged", "missing"):
            with self.subTest(fault=fault), self.fixture() as (build, archive, state, bases):
                identity.write_uccl_source_identity(build)
                if fault == "extra":
                    state["status"] += "?? ep/unreviewed.py\n"
                elif fault == "staged":
                    state["status"] = state["status"].replace(" M ", "M  ", 1)
                else:
                    state["status"] = "\n".join(state["status"].splitlines()[1:])
                with self.assertRaisesRegex(ValueError, "exactly the four"):
                    identity.verify_uccl_source(build)

    def test_regenerated_manifest_cannot_authorize_an_extra_edit_in_allowed_file(self):
        with self.fixture() as (build, archive, state, bases):
            record = identity.write_uccl_source_identity(build)
            relative = identity.PATCH_FILES[0]
            path = build / "uccl" / relative
            path.write_bytes(path.read_bytes() + b"unapproved extra edit\n")
            state["diff"] = b"forged larger diff\n"
            record["source_diff_sha256"] = hashlib.sha256(state["diff"]).hexdigest()
            record["modified_files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            (build / "source_identity.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "exact approved patch"):
                identity.verify_uccl_source(build)

    def test_manifest_or_actual_diff_binding_changes_are_rejected(self):
        for fault in ("manifest", "diff"):
            with self.subTest(fault=fault), self.fixture() as (build, archive, state, bases):
                record = identity.write_uccl_source_identity(build)
                if fault == "manifest":
                    record["modified_files"][identity.PATCH_FILES[0]] = "0" * 64
                    (build / "source_identity.json").write_text(json.dumps(record))
                else:
                    state["diff"] += b"different\n"
                with self.assertRaisesRegex(ValueError, "manifest differs"):
                    identity.verify_uccl_source(build)

    def test_ambiguous_base_context_is_not_guessed(self):
        with self.fixture() as (build, archive, state, bases):
            relative = identity.PATCH_FILES[0]
            bases[relative] *= 2
            with self.assertRaisesRegex(ValueError, "not unique"):
                identity.write_uccl_source_identity(build)

    def test_git_reads_disable_optional_index_writes_and_replace_objects(self):
        with patch.object(identity.subprocess, "check_output", return_value=b"value") as command:
            self.assertEqual(identity._git(Path("/source"), "status", "--porcelain=v1"), b"value")
            arguments = command.call_args
            self.assertEqual(arguments.args[0][-2:], ["status", "--porcelain=v1"])
            self.assertEqual(arguments.kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
            self.assertEqual(arguments.kwargs["env"]["GIT_NO_REPLACE_OBJECTS"], "1")


if __name__ == "__main__":
    unittest.main()
