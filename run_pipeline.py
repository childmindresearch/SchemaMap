#!/usr/bin/env python3
"""
End-to-End Extraction & Aggregation Pipeline Runner

Runs the complete structured data extraction workflow followed by relational 
data aggregation based on configuration parameters defined in config.yaml.
"""

import argparse
import asyncio
import os
import sys
import time

# Ensure src/ directory is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import extract_workflow
import aggregate_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end extraction and aggregation pipeline based on config.yaml."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction phase and run only data aggregation."
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip data aggregation phase and run only extraction."
    )

    args = parser.parse_args()

    # Pass configuration file argument to sys.argv for sub-modules
    sys.argv = [sys.argv[0], "--config", args.config]

    start_time = time.perf_counter()
    print("=====================================================================")
    print("🚀 Starting End-to-End Pipeline Execution")
    print(f"Config File: {args.config}")
    print("=====================================================================\n")

    # 1. Step 1: Extraction Phase
    if not args.skip_extraction:
        print("---------------------------------------------------------------------")
        print("Phase 1: Running Asynchronous Structured Data Extraction")
        print("---------------------------------------------------------------------")
        try:
            asyncio.run(extract_workflow.async_main())
        except Exception as e:
            print(f"❌ Error during extraction phase: {e}", file=sys.stderr)
            sys.exit(1)
        print("✅ Phase 1 Complete!\n")
    else:
        print("⏩ Phase 1 (Extraction) Skipped.\n")

    # 2. Step 2: Data Aggregation Phase
    if not args.skip_aggregation:
        print("---------------------------------------------------------------------")
        print("Phase 2: Running Relational Table Data Aggregator")
        print("---------------------------------------------------------------------")
        try:
            aggregate_outputs.main()
        except Exception as e:
            print(f"❌ Error during aggregation phase: {e}", file=sys.stderr)
            sys.exit(1)
        print("✅ Phase 2 Complete!\n")
    else:
        print("⏩ Phase 2 (Aggregation) Skipped.\n")

    elapsed = time.perf_counter() - start_time
    print("=====================================================================")
    print(f"🎉 Pipeline Execution Complete! Total Elapsed Time: {elapsed:.2f}s")
    print("=====================================================================")


if __name__ == "__main__":
    main()
