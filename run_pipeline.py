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
        "--schema-path", "--schema-dir", "--schema-file",
        dest="schema_path",
        type=str,
        default=None,
        help="Path to Pydantic schema file (.py) or multi-file directory (overrides config.yaml)."
    )
    parser.add_argument(
        "--domain-mapping-file", "--mapping-file",
        dest="domain_mapping_file",
        type=str,
        default=None,
        help="Path to domain mapping python file (.py) containing DOMAIN_EXTRACTION_MAP (overrides config.yaml)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save extracted JSON output files (overrides config.yaml)."
    )
    parser.add_argument(
        "--prompt-payloads-dir", "--payloads-dir",
        dest="prompt_payloads_dir",
        type=str,
        default=None,
        help="Directory to save first LLM prompt payload per schema class (overrides config.yaml)."
    )
    parser.add_argument(
        "--agg-dir",
        type=str,
        default=None,
        help="Directory to save aggregated CSV and SQLite tables (overrides config.yaml)."
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
    if args.schema_path:
        print(f"Schema Path: {args.schema_path}")
    if args.domain_mapping_file:
        print(f"Domain Mapping File: {args.domain_mapping_file}")
    if args.output_dir:
        print(f"Extraction Output Directory: {args.output_dir}")
    if args.prompt_payloads_dir:
        print(f"Prompt Payloads Directory: {args.prompt_payloads_dir}")
    if args.agg_dir:
        print(f"Aggregation Directory: {args.agg_dir}")
    print("=====================================================================\n")

    # Step 1: Structured Extraction Phase
    if not args.skip_extraction:
        print("---------------------------------------------------------------------")
        print("Phase 1: Running Structured Extraction Workflow")
        print("---------------------------------------------------------------------")
        try:
            asyncio.run(
                extract_workflow.run_extraction(
                    config_file=args.config,
                    schema_path=args.schema_path,
                    domain_mapping_file=args.domain_mapping_file,
                    output_dir=args.output_dir,
                    prompt_payloads_dir=args.prompt_payloads_dir
                )
            )
        except Exception as e:
            print(f"❌ Error during extraction phase: {e}", file=sys.stderr)
            sys.exit(1)
        print("✅ Phase 1 Complete!\n")
    else:
        print("⏩ Phase 1 (Extraction) Skipped.\n")

    # Step 2: Data Aggregation Phase
    if not args.skip_aggregation:
        print("---------------------------------------------------------------------")
        print("Phase 2: Running Relational Table Data Aggregator")
        print("---------------------------------------------------------------------")
        try:
            aggregate_outputs.run_aggregation(
                config_file=args.config,
                input_dir=args.output_dir,
                output_dir=args.agg_dir
            )
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
