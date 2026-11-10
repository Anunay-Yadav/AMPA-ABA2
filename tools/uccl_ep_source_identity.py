"""Verify only the approved UCCL borrowed-scratch change on its fixed base.

Verification reads Git and files without updating the checkout or index.
The build manifest binds the result; it does not authorize source changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


BASE_REVISION = "c8fd14e93f2106b69a0d30e0f78c12a52e07a181"
PATCH_SHA256 = "749a28bdb82d25ba92bace55ad6d99bafc1075280bb5466e12f378f5a53a21b2"
PATCH_FILES = (
    "ep/bench/utils.py", "ep/include/uccl_proxy.hpp",
    "ep/src/uccl_ep.cc", "ep/src/uccl_proxy.cpp",
)
PATCH_ARCHIVE = "source/uccl-borrowed-scratch.patch"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _git(source, *arguments):
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_NO_REPLACE_OBJECTS="1")
    return subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "-C", str(source), *arguments], env=environment)


def _patch_hunks(payload):
    """Read full-context hunks from the one fixed-hash approved text patch."""
    files, current, old, new = {}, None, None, None
    expected_old = expected_new = 0

    def finish():
        if old is not None:
            _require(len(old) == expected_old and len(new) == expected_new, "invalid approved hunk lengths")
            files[current].append((b"".join(old), b"".join(new)))

    for line in payload.splitlines(keepends=True):
        if line.startswith(b"diff --git "):
            finish()
            match = re.fullmatch(rb"diff --git a/(\S+) b/(\S+)\n", line)
            _require(match is not None and match[1] == match[2], "unsupported approved patch path")
            current = match[1].decode("utf-8")
            _require(current in PATCH_FILES and current not in files, "unexpected approved patch file")
            files[current], old, new = [], None, None
        elif line.startswith(b"@@ "):
            finish()
            match = re.match(rb"@@ -\d+,(\d+) \+\d+,(\d+) @@", line)
            _require(current is not None and match is not None, "unsupported approved hunk")
            expected_old, expected_new = int(match[1]), int(match[2])
            old, new = [], []
        elif old is None:
            _require(line.startswith((b"--- a/", b"+++ b/")), "unexpected approved patch metadata")
        else:
            _require(line[:1] in (b" ", b"-", b"+"), "unsupported approved patch line")
            if line[:1] != b"+":
                old.append(line[1:])
            if line[:1] != b"-":
                new.append(line[1:])
    finish()
    _require(set(files) == set(PATCH_FILES) and all(files.values()), "approved patch file grid differs")
    return files


def _source_identity(build_dir):
    build = Path(build_dir).resolve()
    source, archive = build / "uccl", build / PATCH_ARCHIVE
    _require(source.is_dir() and not source.is_symlink(), "source checkout must be a regular directory")
    _require(archive.is_file() and not archive.is_symlink(), "approved patch archive is unavailable")
    patch = archive.read_bytes()
    _require(_sha256(patch) == PATCH_SHA256, "archived patch differs from the approved patch")
    _require(_git(source, "rev-parse", "HEAD").decode().strip() == BASE_REVISION,
             "UCCL base revision differs from the approved base")
    status = _git(source, "status", "--porcelain=v1", "--untracked-files=all").decode().splitlines()
    _require(sorted(status) == sorted(" M " + path for path in PATCH_FILES),
             "UCCL source must contain exactly the four approved unstaged modifications")
    modified = {}
    for relative, hunks in _patch_hunks(patch).items():
        expected = _git(source, "show", f"{BASE_REVISION}:{relative}")
        for before, after in hunks:
            _require(before and expected.count(before) == 1, "approved patch context is not unique in base source")
            expected = expected.replace(before, after, 1)
        path = source / relative
        _require(path.is_file() and not path.is_symlink(), "modified source must be a regular file")
        actual = path.read_bytes()
        _require(actual == expected, f"source differs from exact approved patch: {relative}")
        modified[relative] = _sha256(actual)
    diff = _git(source, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--", *PATCH_FILES)
    _require(diff, "approved source diff is empty")
    return {"base_revision": BASE_REVISION, "patch_sha256": PATCH_SHA256,
            "source_diff_sha256": _sha256(diff), "modified_files": dict(sorted(modified.items()))}


def write_uccl_source_identity(build_dir):
    """Build only: record source after exact application on a fresh clean clone."""
    identity = _source_identity(build_dir)
    with (Path(build_dir) / "source_identity.json").open("x", encoding="utf-8") as handle:
        json.dump(identity, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return identity


def verify_uccl_source(build_dir):
    """Read-only verification of the fixed base, approved postimage and manifest."""
    identity = _source_identity(build_dir)
    manifest = Path(build_dir) / "source_identity.json"
    _require(manifest.is_file() and not manifest.is_symlink(), "source identity manifest is unavailable")
    _require(json.loads(manifest.read_text()) == identity, "source identity manifest differs from actual source")
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="Build only: create the manifest once.")
    args = parser.parse_args()
    function = write_uccl_source_identity if args.write else verify_uccl_source
    print(json.dumps(function(args.build_dir), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
