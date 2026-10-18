"""
Pipeline Configuration Manager

Provides strongly typed configuration classes with Pydantic validation,
path resolution relative to project root or CWD, and YAML loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, model_validator
import yaml


def find_project_root(start_dir: Optional[str] = None) -> Path:
    """Locates project root by searching upwards for config.yaml or .git."""
    curr = Path(start_dir or Path.cwd()).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "config.yaml").exists() or (parent / ".git").exists():
            return parent
    return curr


PROJECT_ROOT = find_project_root()


def resolve_path(path_str: Optional[str], base_dir: Optional[Path] = None) -> str:
    """Resolves relative paths against project root or base_dir."""
    if not path_str:
        return ""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    base = base_dir or PROJECT_ROOT
    candidate = base / p
    if candidate.exists():
        return str(candidate.resolve())
    # Return normalized path under base even if it doesn't exist yet
    return str(candidate.resolve())


class InputConfig(BaseModel):
    path: str = Field(default="inputs", description="Directory or file path (.txt or .parquet)")
    format: Literal["txt", "parquet"] = Field(default="txt", description="Input document format")
    id_column: str = Field(default="identifier", description="Parquet ID column")
    text_column: str = Field(default="anonymized", description="Parquet raw text column")


class SchemaConfig(BaseModel):
    path: str = Field(
        default="schemas/test_schemas",
        description="Path to single Python schema file (.py) or multi-file directory"
    )
    class_name: Optional[str] = Field(
        default=None,
        description="Target specific schema class name (null for all / auto-detect)"
    )
    domain_mapping_file: Optional[str] = Field(
        default=None,
        description="Optional path to domain mapping python file containing DOMAIN_EXTRACTION_MAP"
    )
    extract_mode: Literal["top_level", "all", "child"] = Field(
        default="top_level",
        description="Extraction mode: top_level (default auto-hierarchy), all, or child"
    )

    @model_validator(mode="before")
    @classmethod
    def _support_file_or_path(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "path" not in data and "file" in data:
                data["path"] = data.pop("file")
        return data

    @property
    def file(self) -> str:
        """Backwards compatibility alias for path."""
        return self.path

    @file.setter
    def file(self, val: str) -> None:
        self.path = val


class ModelConfig(BaseModel):
    name: str = Field(default="qwen2.5:14b", description="Ollama model identifier")
    url: str = Field(default="http://localhost:11434/v1", description="Ollama / OpenAI-compatible base URL")
    api_key: str = Field(default="ollama", description="API key for LLM endpoint")
    num_async_calls: int = Field(default=4, ge=1, description="Max concurrent async calls")
    timeout: float = Field(default=300.0, ge=5.0, description="HTTP request timeout in seconds")
    retries: int = Field(default=3, ge=0, description="Max retry attempts per chunk on extraction error")


class ChunkingConfig(BaseModel):
    size: int = Field(default=300, ge=50, description="Word count per chunk")
    overlap: int = Field(default=30, ge=0, description="Word overlap between chunks")


class ExecutionConfig(BaseModel):
    output_dir: str = Field(default="outputs/extracts", description="Output directory for extracted JSONs")
    prompt_payloads_dir: Optional[str] = Field(
        default="outputs/prompt_payloads",
        description="Directory to save first LLM prompt payload per schema class"
    )
    resume: bool = Field(default=True, description="Skip already completed document schemas")
    enable_triage: bool = Field(default=True, description="Filter schemas per chunk via fast triage pass to skip unmentioned entities")
    triage_threshold: int = Field(default=3, ge=1, description="Minimum candidate schemas before triage filtering activates")
    sync_mode: bool = Field(default=False, description="Run synchronously instead of asynchronously")
    log_file: str = Field(default="extraction_workflow.log", description="Log file path")


class AggregationConfig(BaseModel):
    input_dir: Optional[str] = Field(
        default=None,
        description="Directory containing JSON outputs (defaults to execution.output_dir if null)"
    )
    output_dir: str = Field(default="outputs/aggregated_tables", description="Directory for CSV/SQLite tables")
    db_file: str = Field(default="aggregated_data.db", description="SQLite database filename")
    csv: bool = Field(default=True, description="Enable CSV table generation")
    sqlite: bool = Field(default=True, description="Enable SQLite database generation")
    multi_table: bool = Field(default=True, description="Relational multi-table (true) or denormalized (false)")
    drop_empty_records: bool = Field(default=True, description="Discard records where all extracted data fields are null or empty")
    drop_empty_tables: bool = Field(default=True, description="Skip generating CSV/SQLite tables that have 0 non-empty rows")
    consolidated_wide_table: bool = Field(
        default=True,
        description="Generate an additional consolidated wide table (1 row per document, all schema fields as columns)"
    )
    wide_table_name: str = Field(
        default="consolidated_wide",
        description="Table and file name for the consolidated wide table"
    )
    include_source_text: bool = Field(
        default=True,
        description="Include raw document text as 'source_text' column in consolidated wide table"
    )


class PipelineConfig(BaseModel):
    input: InputConfig = Field(default_factory=InputConfig)
    schema_settings: SchemaConfig = Field(default_factory=SchemaConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)

    @model_validator(mode="before")
    @classmethod
    def _map_schema_key(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Map YAML 'schema' key to 'schema_settings' field
            if "schema" in data and "schema_settings" not in data:
                data["schema_settings"] = data.pop("schema")
        return data

    @model_validator(mode="after")
    def _sync_aggregation_defaults(self) -> PipelineConfig:
        if self.aggregation.input_dir is None:
            self.aggregation.input_dir = self.execution.output_dir
        return self

    @property
    def schema(self) -> SchemaConfig:
        """Alias for schema_settings."""
        return self.schema_settings

    @classmethod
    def from_yaml(cls, filepath: str = "config.yaml") -> PipelineConfig:
        """Loads and validates configuration from a YAML file."""
        resolved = resolve_path(filepath)
        if not os.path.exists(resolved):
            # Fall back to default config if file not found
            return cls()
        with open(resolved, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def to_yaml(self, filepath: str) -> None:
        """Dumps current configuration to a YAML file."""
        dumped = self.model_dump(mode="json")
        # Rename schema_settings back to schema for readability
        if "schema_settings" in dumped:
            dumped["schema"] = dumped.pop("schema_settings")
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(dumped, f, sort_keys=False, default_flow_style=False)


def load_config(filepath: str = "config.yaml") -> PipelineConfig:
    """Convenience helper to load PipelineConfig from YAML."""
    return PipelineConfig.from_yaml(filepath)
