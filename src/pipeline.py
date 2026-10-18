"""
End-to-End Pipeline Coordinator

Integrates document ingestion, schema discovery, asynchronous extraction,
and relational aggregation into a unified execution flow.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

from .config import PipelineConfig, resolve_path
from .chunker import Document, load_input_documents
from .schemas import (
    SchemaMeta,
    load_schemas,
    get_top_level_schemas,
    get_child_schemas,
    load_domain_mapping,
)
from .extractor import ExtractionEngine, setup_logger
from .aggregator import DataAggregator
from .tracker import ResumeTracker


class Pipeline:
    """Master workflow orchestrator for SchemaMap."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = setup_logger(self.config.execution.log_file)
        self.tracker = ResumeTracker(resolve_path(self.config.execution.output_dir))

    def select_target_schemas(self, schemas_meta: List[SchemaMeta]) -> List[SchemaMeta]:
        """Filters schema list according to configured extraction mode and class name."""
        target = schemas_meta
        target_class = self.config.schema_settings.class_name
        if target_class:
            target = [m for m in schemas_meta if m.class_name == target_class]
            if not target:
                raise ValueError(f"Specified class_name '{target_class}' not found in loaded schemas.")
            return target

        mode = self.config.schema_settings.extract_mode
        if mode == "top_level":
            return get_top_level_schemas(schemas_meta)
        elif mode == "child":
            return get_child_schemas(schemas_meta)
        return schemas_meta

    async def run_extraction_async(self, file_override: Optional[str] = None) -> None:
        """Executes the asynchronous extraction phase across input documents."""
        schema_path = resolve_path(self.config.schema_settings.path)
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema path not found: '{schema_path}'. Check 'schema.path' in config.yaml.")

        all_schemas = load_schemas(schema_path)
        target_schemas = self.select_target_schemas(all_schemas)

        domain_mapping = {}
        mapping_file = self.config.schema_settings.domain_mapping_file
        if mapping_file:
            resolved_map = resolve_path(mapping_file)
            domain_mapping = load_domain_mapping(resolved_map)
            if domain_mapping:
                self.logger.info(f"Loaded Domain Extraction Mapping from '{mapping_file}' ({len(domain_mapping)} domains).")

        self.logger.info("=== SchemaMap Pipeline Configuration ===")
        self.logger.info(f"Schema Path: '{schema_path}'")
        self.logger.info(f"Extraction Mode: '{self.config.schema_settings.extract_mode}'")
        self.logger.info(f"Active Schemas ({len(target_schemas)}): {[m.qualified_name for m in target_schemas]}")

        input_path = resolve_path(self.config.input.path)
        docs = load_input_documents(
            input_dir=input_path,
            input_format=self.config.input.format,
            file_override=file_override,
            id_col=self.config.input.id_column,
            text_col=self.config.input.text_column,
        )

        if not docs:
            self.logger.warning(f"No input documents found at '{input_path}'.")
            return

        self.logger.info(f"Loaded {len(docs)} input document(s) for extraction.")
        engine = ExtractionEngine(self.config, logger=self.logger)

        try:
            processed_count = 0
            skipped_count = 0
            total_docs = len(docs)

            for doc_idx, doc in enumerate(docs, 1):
                if self.config.execution.resume and self.tracker.is_document_completed(doc.id, target_schemas):
                    self.logger.info(f"=== [Document {doc_idx}/{total_docs} Skipped] Record ID: '{doc.id}' already completed. ===")
                    skipped_count += 1
                    continue

                await engine.extract_document(
                    doc=doc,
                    doc_idx=doc_idx,
                    total_docs=total_docs,
                    schemas_meta=target_schemas,
                    domain_mapping=domain_mapping
                )
                processed_count += 1

            self.logger.info(f"Extraction Phase Complete! Total: {total_docs} | Processed: {processed_count} | Skipped: {skipped_count}")
        finally:
            await engine.close()

    def run_aggregation(self) -> None:
        """Executes relational data aggregation phase."""
        input_dir = resolve_path(self.config.aggregation.input_dir or self.config.execution.output_dir)
        output_dir = resolve_path(self.config.aggregation.output_dir)
        aggregator = DataAggregator(self.config.aggregation, input_config=self.config.input)
        aggregator.aggregate(
            input_dir=input_dir,
            output_dir=output_dir
        )

    def run(
        self,
        skip_extraction: bool = False,
        skip_aggregation: bool = False,
        file_override: Optional[str] = None
    ) -> None:
        """Runs the complete pipeline end-to-end."""
        start_time = time.perf_counter()
        print("=====================================================================")
        print("🚀 Starting SchemaMap Pipeline")
        print(f"Schema Source: {self.config.schema_settings.path}")
        print(f"Extraction Mode: {self.config.schema_settings.extract_mode}")
        print(f"Outputs Directory: {self.config.execution.output_dir}")
        print(f"Aggregation Directory: {self.config.aggregation.output_dir}")
        print("=====================================================================\n")

        if not skip_extraction:
            print("---------------------------------------------------------------------")
            print("Phase 1: Running Structured Extraction Workflow")
            print("---------------------------------------------------------------------")
            asyncio.run(self.run_extraction_async(file_override=file_override))
            print("✅ Phase 1 (Extraction) Complete!\n")
        else:
            print("⏩ Phase 1 (Extraction) Skipped.\n")

        if not skip_aggregation:
            print("---------------------------------------------------------------------")
            print("Phase 2: Running Relational Table Data Aggregator")
            print("---------------------------------------------------------------------")
            self.run_aggregation()
            print("✅ Phase 2 (Aggregation) Complete!\n")
        else:
            print("⏩ Phase 2 (Aggregation) Skipped.\n")

        elapsed = time.perf_counter() - start_time
        print("=====================================================================")
        print(f"🎉 Pipeline Execution Complete! Total Time: {elapsed:.2f}s")
        print("=====================================================================")
