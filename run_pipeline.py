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
import extract_multifile_workflow
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
        "--schema-dir",
        type=str,
        default=None,
        help="Directory path to multi-file Pydantic schema files (overrides config.yaml)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save extracted JSON output files (overrides config.yaml)."
    )
    parser.add_argument(
        "--agg-dir",
        type=str,
        default=None,
        help="Directory to save aggregated CSV and SQLite tables (overrides config.yaml)."
    )
    parser.add_argument(
        "--multifile",
        action="store_true",
        help="Force multi-file workflow mode for schema directories."
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

    start_time = time.perf_counter()
    print("=====================================================================")
    print("🚀 Starting End-to-End Pipeline Execution")
    print(f"Config File: {args.config}")
    if args.schema_dir:
        print(f"Schema Directory: {args.schema_dir}")
    if args.output_dir:
        print(f"Extraction Output Directory: {args.output_dir}")
    if args.agg_dir:
        print(f"Aggregation Directory: {args.agg_dir}")
    print("=====================================================================\n")

    # Determine if schema file setting points to a directory (Multi-File workflow)
    cfg = extract_workflow.load_config_file(args.config)
    schema_path = args.schema_dir or cfg.get("schema", {}).get("file", "")
    is_dir_schema = os.path.isdir(schema_path) if schema_path else False
    use_multifile_wf = args.multifile or is_dir_schema


    # 1. Step 1: Extraction Phase (Default Workflow)
    if not args.skip_extraction:
        print("---------------------------------------------------------------------")
        print("Phase 1: Running Multi-File DAG-Classified Structured Extraction Workflow")
        print("---------------------------------------------------------------------")
        
        phase1_args = [sys.argv[0], "--config", args.config]
        if args.schema_dir:
            phase1_args.extend(["--schema-dir", args.schema_dir])
        if args.output_dir:
            phase1_args.extend(["--output-dir", args.output_dir])
        sys.argv = phase1_args

        try:
            asyncio.run(extract_multifile_workflow.async_main())
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
        
        phase2_args = [sys.argv[0], "--config", args.config]
        if args.output_dir:
            phase2_args.extend(["--input-dir", args.output_dir])
        if args.agg_dir:
            phase2_args.extend(["--output-dir", args.agg_dir])
        sys.argv = phase2_args

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
