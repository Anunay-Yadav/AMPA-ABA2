"""Reviewed Qwen workload variants; checkpoint save identity is separate."""

def workload(variant: str) -> dict[str, int]:
    variants = {"cp1_ep8": (1, 8), "cp2_ep8": (2, 8),
                "cp1_ep16": (1, 16), "cp2_ep16": (2, 16)}
    if variant not in variants:
        raise ValueError(f"unsupported topology variant: {variant!r}")
    cp, ep = variants[variant]
    dp = ep // cp
    return {"cp": cp, "ep": ep, "world_size": ep, "dense_dp": dp,
            "accumulation": 64 // dp, "token_positions": 4096 // cp}
