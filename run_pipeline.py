#!/usr/bin/env python3
"""
End-to-End Extraction & Aggregation Pipeline Runner

Runs the complete structured data extraction workflow followed by relational
data aggregation based on configuration parameters defined in config.yaml.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure src/ directory is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.config import PipelineConfig
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run SchemaMap end-to-end extraction and aggregation pipeline."
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
        help="Path to Pydantic schema file (.py) or multi-file directory."
    )
    parser.add_argument(
        "--domain-mapping-file", "--mapping-file",
        dest="domain_mapping_file",
        type=str,
        default=None,
        help="Path to domain mapping python file (.py) containing DOMAIN_EXTRACTION_MAP."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save extracted JSON output files."
    )
    parser.add_argument(
        "--prompt-payloads-dir", "--payloads-dir",
        dest="prompt_payloads_dir",
        type=str,
        default=None,
        help="Directory to save first LLM prompt payload per schema class."
    )
    parser.add_argument(
        "--agg-dir",
        type=str,
        default=None,
        help="Directory to save aggregated CSV and SQLite tables."
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume workflow and skip previously completed documents/schemas."
    )
    parser.add_argument(
        "--extract-mode",
        type=str,
        choices=["top_level", "all", "child"],
        default=None,
        help="Dynamic extraction mode: top_level (default), all, or child."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Ollama model identifier (e.g. qwen2.5:14b)."
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="Ollama / OpenAI-compatible API base URL."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing input documents."
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["txt", "parquet"],
        default=None,
        help="Format of input documents."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process a specific input file only."
    )
    parser.add_argument(
        "--triage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable schema relevance triage pre-filtering."
    )
    parser.add_argument(
        "--wide-table",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate an additional consolidated wide table (1 row per document)."
    )
    parser.add_argument(
        "--source-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include raw document text as 'source_text' in the consolidated wide table."
    )
    parser.add_argument(
        "--drop-empty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Discard empty extraction records where all data fields are null or empty."
    )
    parser.add_argument(
        "--drop-empty-tables",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip generating CSV/SQLite tables for schemas with 0 non-empty rows."
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

    # Load base configuration
    cfg = PipelineConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.schema_path:
        cfg.schema_settings.path = args.schema_path
    if args.domain_mapping_file:
        cfg.schema_settings.domain_mapping_file = args.domain_mapping_file
    if args.model_name:
        cfg.model.name = args.model_name
    if args.model_url:
        cfg.model.url = args.model_url
    if args.output_dir:
        cfg.execution.output_dir = args.output_dir
        if not args.agg_dir:
            cfg.aggregation.input_dir = args.output_dir
    if args.prompt_payloads_dir:
        cfg.execution.prompt_payloads_dir = args.prompt_payloads_dir
    if args.agg_dir:
        cfg.aggregation.output_dir = args.agg_dir
    if args.resume is not None:
        cfg.execution.resume = args.resume
    if args.extract_mode:
        cfg.schema_settings.extract_mode = args.extract_mode
    if args.input_dir:
        cfg.input.path = args.input_dir
    if args.input_format:
        cfg.input.format = args.input_format
    if args.triage is not None:
        cfg.execution.enable_triage = args.triage
    if args.wide_table is not None:
        cfg.aggregation.consolidated_wide_table = args.wide_table
    if args.source_text is not None:
        cfg.aggregation.include_source_text = args.source_text
    if args.drop_empty is not None:
        cfg.aggregation.drop_empty_records = args.drop_empty
    if args.drop_empty_tables is not None:
        cfg.aggregation.drop_empty_tables = args.drop_empty_tables

    # Run pipeline
    pipeline = Pipeline(cfg)
    pipeline.run(
        skip_extraction=args.skip_extraction,
        skip_aggregation=args.skip_aggregation,
        file_override=args.file
    )


if __name__ == "__main__":
    main()
