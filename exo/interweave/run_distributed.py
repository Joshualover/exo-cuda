#!/usr/bin/env python3
"""
Run distributed inference test across multiple Interweave nodes.
"""

import asyncio
import sys
from .distributed_server import run_distributed_test


async def main():
    # Default nodes - adjust as needed
    nodes = [
        "192.168.0.161:8089",  # Dell C4130 (V100 CUDA)
        "192.168.0.153:8089",  # Mac Pro (FirePro OpenCL)
        "192.168.0.50:8089",   # Power8 (CPU)
    ]

    # Allow override from command line
    if len(sys.argv) > 1:
        nodes = sys.argv[1:]

    print(f"Testing distributed inference across: {nodes}")
    result = await run_distributed_test(nodes)

    if result:
        print(f"\nSUCCESS! Total time: {result['total_time_ms']:.1f}ms")
    else:
        print("\nFAILED!")


if __name__ == '__main__':
    asyncio.run(main())
