# SM work in communication

Goal: Explain which communication operations use GPU streaming multiprocessors (SMs) in the measured UCCL normal path.

State: COMPLETE. Read-only exploration; no training, source changes, jobs or large trace parsing.

Checked: The pinned build under runs/uccl-ep-owned-build-20261101T1713Z; internode.cu dispatch warp roles at555–587, token packing near924, RDMA/NVLink forwarding at1169 onward, receive-buffer copies near1559, and combine reduction near2200. cxi_transport.cpp registers the CUDA main buffer at218 and posts fi_write from its offset at294. The project author's architecture description confirms the GPU/CPU/NIC split: https://uccl-project.github.io/posts/uccl-ep/.

Findings: SM kernels pack activations and routing metadata into GPU communication buffers, forward received data over NVLink, place data for expert computation, combine/reduce outputs, and poll flags/credits. The payload buffers are GPU memory registered for NIC access. NIC DMA performs the cross-node payload transfer; the UCCL CPU proxy submits network operations and handles completions. Hardware memory/link engines perform transfers initiated by GPU kernels; some pinned paths use TMA. Polling can keep communication warps resident while waiting.

Recorded setting: Rank0 runtime identity reports num_sms=20 in both UCCL control routing modes. This is a configured launch budget, not measured physical SM occupancy or active-copy utilization.

Limits: The current traces do not separate useful copying from waiting, report physical wire bytes, or quantify actual occupied/active SMs. No byte-for-byte GPU/CPU payload-staging or always-busy SM claim is made.

Changed: This status, task index and root finding index only. No behavior change; tests unnecessary.

Next action: None for this explanatory question.
