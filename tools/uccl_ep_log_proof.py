"""Bind native CXI endpoint messages to the four torchrun worker logs.

Run inside allocated compute. A requested transport environment variable does
not substitute for these native initialization records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect_node(log_dir, node_rank):
    rows = []
    for local_rank in range(4):
        streams = {}
        contents = []
        for name in ("stdout", "stderr"):
            candidates = list(Path(log_dir).glob(f"**/attempt_0/{local_rank}/{name}.log"))
            if len(candidates) != 1:
                raise ValueError(f"expected one {name} log for worker {local_rank}")
            path = candidates[0]
            payload = path.read_bytes()
            streams[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}
            contents.append(payload.decode("utf-8", errors="replace"))
        content = "\n".join(contents)
        domains = re.findall(r"Selected CXI domain (\S+) for local_rank (\d+)", content)
        if not domains or set(domains) != {(f"cxi{local_rank}", str(local_rank))}:
            raise ValueError(f"worker {local_rank}: native CXI domain/local-rank selection differs")
        endpoints = content.count("[CXI] libfabric/CXI endpoint initialized")
        if endpoints < 1:
            raise ValueError(f"worker {local_rank}: no completed native CXI endpoint initialization")
        rows.append({"rank": node_rank * 4 + local_rank, "node_rank": node_rank,
                     "local_rank": local_rank, "native_cxi_domain": f"cxi{local_rank}",
                     "domain_messages": len(domains), "endpoint_messages": endpoints, "logs": streams})
    return {"valid_native_cxi_log_binding": True, "workers": rows,
            "claim_limit": "Native initialization proves selected domains, not physical bytes or achieved RDMA bandwidth."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=range(4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_node(args.log_dir, args.node_rank)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
