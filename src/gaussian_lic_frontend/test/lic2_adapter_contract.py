#!/usr/bin/env python3
"""Static regression guard for adapter validation and fail-closed transforms."""

from pathlib import Path
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: lic2_adapter_contract.py ADAPTER_SOURCE")
    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    required = (
        "validate_pointcloud_xyz_layout(*msg)",
        "validate_pointcloud_xyz_layout(out)",
        "for (size_t row = 0U; row < out.height; ++row)",
        "for (size_t column = 0U; column < out.width; ++column)",
        "cloud.is_bigendian",
        "failed to validate/transform/filter pointcloud; dropping cloud",
        "IMU world-frame pointcloud transform requested but no orientation is available",
    )
    for token in required:
        if token not in text:
            raise AssertionError(f"missing adapter contract token: {token}")
    if "forwarding original cloud" in text:
        raise AssertionError("transform failures must not fail open to the original cloud")
    print("lic2_adapter_contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
