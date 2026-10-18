"""
SchemaMap Package

A domain-agnostic, schema-driven framework for dynamic structured text extraction
and relational data aggregation.
"""

from .config import PipelineConfig, load_config
from .chunker import Document, chunk_text, load_input_documents
from .schemas import (
    SchemaMeta,
    load_schemas,
    get_top_level_schemas,
    get_child_schemas,
    load_domain_mapping,
)
from .prompts import (
    EXTRACTION_RULES,
    TRIAGE_RULES,
    format_schema_field_instructions,
    build_triage_prompt,
    build_extraction_prompt,
)
from .merger import SafeJSONEncoder, merge_pydantic_instances
from .tracker import ResumeTracker, sanitize_doc_id
from .extractor import ExtractionEngine
from .aggregator import (
    DataAggregator,
    has_extracted_content,
    build_consolidated_wide_table,
    load_source_text_map,
)
from .pipeline import Pipeline

__all__ = [
    "PipelineConfig",
    "load_config",
    "Document",
    "chunk_text",
    "load_input_documents",
    "SchemaMeta",
    "load_schemas",
    "get_top_level_schemas",
    "get_child_schemas",
    "load_domain_mapping",
    "format_schema_field_instructions",
    "build_triage_prompt",
    "build_extraction_prompt",
    "EXTRACTION_RULES",
    "TRIAGE_RULES",
    "SafeJSONEncoder",
    "merge_pydantic_instances",
    "ResumeTracker",
    "sanitize_doc_id",
    "ExtractionEngine",
    "DataAggregator",
    "has_extracted_content",
    "build_consolidated_wide_table",
    "load_source_text_map",
    "Pipeline",
]
