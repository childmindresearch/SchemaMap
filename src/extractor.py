"""
Optimized LLM Extraction Engine with Schema Relevance Triage

Performs direct, single-pass structured data extraction using local LLMs (via Ollama)
and pydantic-ai. Features a fast Schema Relevance Triage filter to eliminate 80-90%
of redundant LLM calls on irrelevant schemas per chunk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.output import PromptedOutput
except ImportError as e:
    Agent = None
    OllamaModel = None
    OllamaProvider = None
    PromptedOutput = None

from .config import PipelineConfig, resolve_path
from .chunker import Document, chunk_text
from .schemas import (
    SchemaMeta,
    get_domain_info_for_schema,
)
from .prompts import (
    EXTRACTION_RULES,
    TRIAGE_RULES,
    format_schema_field_instructions,
    build_triage_prompt,
    build_extraction_prompt,
)
from .merger import merge_pydantic_instances
from .tracker import ResumeTracker, sanitize_doc_id


class TriageResult(BaseModel):
    """Output model for fast schema relevance triage classification."""
    relevant_schemas: List[str] = Field(
        default_factory=list,
        description="List of exact schema class names that have factual evidence present in the text excerpt."
    )


def setup_logger(log_filepath: str = "extraction_workflow.log") -> logging.Logger:
    """Configures centralized execution logger with file and console handlers."""
    logger = logging.getLogger("schemamap")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if log_filepath:
        log_dir = os.path.dirname(log_filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_filepath, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger






class ExtractionEngine:
    """Orchestrates single-schema LLM extraction with triage filtering and concurrency control."""

    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        if Agent is None:
            raise ImportError("Required dependency missing: pydantic-ai. Run: pip install pydantic-ai")

        self.config = config
        self.logger = logger or setup_logger(config.execution.log_file)
        self.tracker = ResumeTracker(resolve_path(config.execution.output_dir))
        self.saved_payload_classes: Set[str] = set()

        # Concurrency & Clients
        self.semaphore = asyncio.Semaphore(config.model.num_async_calls)
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=config.model.timeout, connect=60.0),
            limits=httpx.Limits(
                max_connections=config.model.num_async_calls,
                max_keepalive_connections=config.model.num_async_calls
            )
        )
        self.openai_client = AsyncOpenAI(
            base_url=config.model.url,
            api_key=config.model.api_key,
            http_client=self.http_client
        )
        self.provider = OllamaProvider(openai_client=self.openai_client)
        self.model = OllamaModel(config.model.name, provider=self.provider)

        # Agent Cache: (schema_cls, system_prompt) -> Agent
        self._agents: Dict[Tuple[type[BaseModel], str], Any] = {}

    async def close(self) -> None:
        """Cleans up HTTP client connections."""
        await self.http_client.aclose()

    def get_or_create_agent(self, output_schema: type[BaseModel], system_prompt: str) -> Any:
        """Retrieves or creates a cached pydantic-ai Agent for a specific schema."""
        cache_key = (output_schema, system_prompt)
        if cache_key not in self._agents:
            self._agents[cache_key] = Agent(
                self.model,
                output_type=PromptedOutput(output_schema),
                system_prompt=system_prompt,
                retries=self.config.model.retries
            )
        return self._agents[cache_key]

    def save_prompt_payload(
        self,
        meta: SchemaMeta,
        system_prompt: str,
        user_chunk_text: str
    ) -> None:
        """Saves first prompt payload per schema class for inspection."""
        payload_dir = self.config.execution.prompt_payloads_dir
        if not payload_dir or meta.qualified_name in self.saved_payload_classes:
            return

        payload_path = Path(resolve_path(payload_dir)) / f"{meta.class_name}.txt"
        payload_path.parent.mkdir(parents=True, exist_ok=True)

        if payload_path.exists():
            self.saved_payload_classes.add(meta.qualified_name)
            return

        content = (
            "================================================================================\n"
            "LLM EXTRACTION PROMPT PAYLOAD (FIRST CALL FOR SCHEMA)\n"
            f"Target Class: {meta.class_name}\n"
            f"Qualified Schema: {meta.qualified_name}\n"
            f"Module File: {meta.module_name}.py\n"
            f"Filepath: {meta.filepath}\n"
            "================================================================================\n\n"
            "--------------------------------------------------------------------------------\n"
            "1. SYSTEM PROMPT & FIELD INSTRUCTIONS\n"
            "--------------------------------------------------------------------------------\n"
            f"{system_prompt}\n\n"
            "--------------------------------------------------------------------------------\n"
            "2. USER INPUT (DOCUMENT CHUNK EXCERPT)\n"
            "--------------------------------------------------------------------------------\n"
            f"{user_chunk_text}\n\n"
            "================================================================================\n"
            "3. FULL EXACT TEXT CONCATENATED TO LLM\n"
            "================================================================================\n"
            f"{system_prompt}\n{user_chunk_text}\n"
        )
        try:
            payload_path.write_text(content, encoding="utf-8")
            self.saved_payload_classes.add(meta.qualified_name)
            self.logger.info(f"  [PROMPT PAYLOAD SAVED] First LLM payload for '{meta.class_name}' saved -> {payload_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save prompt payload for '{meta.class_name}': {e}")

    def save_triage_payload(
        self,
        candidate_schemas: List[SchemaMeta],
        system_prompt: str,
        user_chunk_text: str
    ) -> None:
        """Saves first triage prompt payload for human inspection."""
        payload_dir = self.config.execution.prompt_payloads_dir
        if not payload_dir or "_triage" in self.saved_payload_classes:
            return

        payload_path = Path(resolve_path(payload_dir)) / "_SCHEMA_RELEVANCE_TRIAGE.txt"
        payload_path.parent.mkdir(parents=True, exist_ok=True)

        if payload_path.exists():
            self.saved_payload_classes.add("_triage")
            return

        candidate_list = "\n".join([f"- {m.class_name} ({m.module_name}.py)" for m in candidate_schemas])
        content = (
            "================================================================================\n"
            "SCHEMA RELEVANCE TRIAGE PROMPT PAYLOAD (FIRST CALL)\n"
            f"Screening Candidate Schemas ({len(candidate_schemas)} total)\n"
            f"Active Threshold: {self.config.execution.triage_threshold}\n"
            "================================================================================\n\n"
            "--------------------------------------------------------------------------------\n"
            "1. CANDIDATE SCHEMA ROSTER\n"
            "--------------------------------------------------------------------------------\n"
            f"{candidate_list}\n\n"
            "--------------------------------------------------------------------------------\n"
            "2. SYSTEM PROMPT & FIELD INSTRUCTIONS SENT TO LLM\n"
            "--------------------------------------------------------------------------------\n"
            f"{system_prompt}\n\n"
            "--------------------------------------------------------------------------------\n"
            "3. SAMPLE USER INPUT (DOCUMENT CHUNK EXCERPT)\n"
            "--------------------------------------------------------------------------------\n"
            f"{user_chunk_text}\n\n"
            "================================================================================\n"
            "4. FULL EXACT TEXT CONCATENATED TO LLM\n"
            "================================================================================\n"
            f"{system_prompt}\n{user_chunk_text}\n"
        )
        try:
            payload_path.write_text(content, encoding="utf-8")
            self.saved_payload_classes.add("_triage")
            self.logger.info(f"  [PROMPT PAYLOAD SAVED] First Triage prompt payload saved -> {payload_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save triage prompt payload: {e}")

    async def triage_chunk(
        self,
        chunk_text_data: str,
        candidate_schemas: List[SchemaMeta],
        doc_id: str,
        chunk_idx: int,
        total_chunks: int
    ) -> List[SchemaMeta]:
        """
        Fast triage pass: classifies which schemas are relevant to this chunk.
        Skips 80-90% of redundant calls on unmentioned schemas.
        """
        if not self.config.execution.enable_triage or len(candidate_schemas) < self.config.execution.triage_threshold:
            return candidate_schemas

        prompt = build_triage_prompt(candidate_schemas)
        self.save_triage_payload(candidate_schemas, prompt, chunk_text_data)
        agent = self.get_or_create_agent(TriageResult, prompt)
        word_count = len(chunk_text_data.split())

        async with self.semaphore:
            start_time = time.perf_counter()
            self.logger.info(
                f"  [TRIAGE DISPATCHED] Doc: '{doc_id}' | Chunk: {chunk_idx}/{total_chunks} ({word_count} w) | Screening {len(candidate_schemas)} candidate schemas..."
            )
            try:
                if self.config.execution.sync_mode:
                    result = await asyncio.to_thread(agent.run_sync, chunk_text_data)
                else:
                    result = await agent.run(chunk_text_data)

                elapsed = time.perf_counter() - start_time
                output: Optional[TriageResult] = result.output
                relevant_names = set(output.relevant_schemas if output else [])

                # Match class names (case-insensitive)
                schema_map = {m.class_name.lower(): m for m in candidate_schemas}
                selected = [
                    schema_map[name.lower()]
                    for name in relevant_names
                    if name.lower() in schema_map
                ]

                # Safety fallback: if triage matched none, extract all candidates
                if not selected:
                    self.logger.info(
                        f"  [TRIAGE EMPTY] No candidate matched explicitly ({elapsed:.2f}s). Falling back to all {len(candidate_schemas)} schemas."
                    )
                    return candidate_schemas

                skipped_count = len(candidate_schemas) - len(selected)
                self.logger.info(
                    f"  [TRIAGE SUCCESS] Selected {len(selected)}/{len(candidate_schemas)} schemas: {[s.class_name for s in selected]} "
                    f"(Skipped {skipped_count} irrelevant schemas! Saved {skipped_count} LLM calls) | Elapsed: {elapsed:.2f}s"
                )
                return selected
            except Exception as e:
                self.logger.warning(f"  [TRIAGE FAILED] {e}. Extracting all candidate schemas as fallback.")
                return candidate_schemas

    async def _extract_chunk_schema(
        self,
        meta: SchemaMeta,
        agent: Any,
        chunk_text_data: str,
        doc_id: str,
        chunk_idx: int,
        total_chunks: int
    ) -> Any:
        """Executes a single-schema extraction on a text chunk under semaphore."""
        word_count = len(chunk_text_data.split())
        async with self.semaphore:
            start_time = time.perf_counter()
            self.logger.info(
                f"  [API DISPATCHED] Doc: '{doc_id}' | Chunk: {chunk_idx}/{total_chunks} ({word_count} w) | Schema: '{meta.class_name}'"
            )
            try:
                if self.config.execution.sync_mode:
                    result = await asyncio.to_thread(agent.run_sync, chunk_text_data)
                else:
                    result = await agent.run(chunk_text_data)

                elapsed = time.perf_counter() - start_time
                output = result.output
                if output is not None:
                    self.logger.info(
                        f"  [API SUCCESS] Doc: '{doc_id}' | Chunk: {chunk_idx}/{total_chunks} | Schema: '{meta.class_name}' | Elapsed: {elapsed:.2f}s"
                    )
                    return output
                else:
                    self.logger.warning(
                        f"  [API EMPTY] Doc: '{doc_id}' | Chunk: {chunk_idx}/{total_chunks} | Schema: '{meta.class_name}'"
                    )
                    return None
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                self.logger.error(
                    f"  [API FAILED] Doc: '{doc_id}' | Chunk: {chunk_idx}/{total_chunks} | Schema: '{meta.class_name}' | Error: {e}"
                )
                return None

    async def extract_document(
        self,
        doc: Document,
        doc_idx: int,
        total_docs: int,
        schemas_meta: List[SchemaMeta],
        domain_mapping: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """
        Extracts structured data for each target schema from a document.
        Uses Schema Relevance Triage to filter schemas per chunk, then extracts
        only relevant schemas.
        """
        self.logger.info(f"=== [Document {doc_idx}/{total_docs} Started] ID: {doc.id} ({doc.source}) ===")
        if not doc.text:
            self.logger.warning(f"Skipping document ID '{doc.id}': Input text is empty.")
            return

        chunks = chunk_text(doc.text, self.config.chunking.size, self.config.chunking.overlap)
        doc_dir = self.tracker.get_document_dir(doc.id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        chunks_cache_dir = doc_dir / "chunks"
        chunks_cache_dir.mkdir(parents=True, exist_ok=True)

        base_name = sanitize_doc_id(doc.id)

        # Granular Resume: Identify pending schemas for this document
        if self.config.execution.resume:
            pending_schemas = self.tracker.get_pending_schemas(doc.id, schemas_meta)
            if not pending_schemas:
                self.tracker.mark_completed(doc.id, schemas_meta)
                self.logger.info(f"=== [Document {doc_idx}/{total_docs} Skipped] All {len(schemas_meta)} schemas already completed for ID: '{doc.id}'. ===")
                return
            if len(pending_schemas) < len(schemas_meta):
                completed_count = len(schemas_meta) - len(pending_schemas)
                self.logger.info(
                    f"Document '{doc.id}' resuming: {completed_count}/{len(schemas_meta)} schemas already complete on disk. "
                    f"Processing {len(pending_schemas)} pending schema(s)."
                )
            active_target_schemas = pending_schemas
        else:
            active_target_schemas = schemas_meta

        self.logger.info(f"Document '{doc.id}' split into {len(chunks)} chunk(s). Target schemas to extract: {len(active_target_schemas)}")

        # Prepare prompts and agents per active target schema
        schema_agents = {}
        schema_prompts = {}
        for meta in active_target_schemas:
            domain_info = get_domain_info_for_schema(meta, domain_mapping) if domain_mapping else None
            prompt = build_extraction_prompt(meta.schema_cls, domain_info)
            schema_prompts[meta.qualified_name] = prompt
            schema_agents[meta.qualified_name] = self.get_or_create_agent(meta.schema_cls, prompt)

        schema_chunk_outputs: Dict[str, List[Any]] = {meta.qualified_name: [] for meta in active_target_schemas}

        # Process each chunk with triage filtering
        for chunk_idx, chunk_text_data in enumerate(chunks, 1):
            # Run fast triage screening pass on active target schemas
            active_schemas = await self.triage_chunk(
                chunk_text_data, active_target_schemas, doc.id, chunk_idx, len(chunks)
            )

            self.logger.info(
                f"--- [Chunk {chunk_idx}/{len(chunks)}] Extracting {len(active_schemas)} targeted schema(s) ---"
            )

            async def process_schema(meta: SchemaMeta):
                schema_cache = chunks_cache_dir / meta.qualified_name
                schema_cache.mkdir(parents=True, exist_ok=True)
                chunk_file = schema_cache / f"{base_name}_chunk_{chunk_idx}.json"

                # Chunk-level resume: Hit disk cache if already extracted during interrupted run
                if self.config.execution.resume and chunk_file.exists():
                    try:
                        cached_raw = json.loads(chunk_file.read_text(encoding="utf-8"))
                        if cached_raw is not None:
                            cached_output = meta.schema_cls.model_validate(cached_raw)
                            schema_chunk_outputs[meta.qualified_name].append(cached_output)
                        self.logger.info(
                            f"  [CHUNK CACHE HIT] Doc: '{doc.id}' | Chunk: {chunk_idx}/{len(chunks)} | Schema: '{meta.class_name}' loaded from cache (LLM call skipped)."
                        )
                        return
                    except Exception:
                        pass

                agent = schema_agents[meta.qualified_name]
                prompt = schema_prompts[meta.qualified_name]

                self.save_prompt_payload(meta, prompt, chunk_text_data)
                output = await self._extract_chunk_schema(
                    meta, agent, chunk_text_data, doc.id, chunk_idx, len(chunks)
                )

                if output is not None:
                    schema_chunk_outputs[meta.qualified_name].append(output)
                    # Cache chunk JSON to disk
                    dumped = output.model_dump(mode="json") if hasattr(output, "model_dump") else output
                    chunk_file.write_text(json.dumps(dumped, indent=2), encoding="utf-8")

            # Concurrently extract only active schemas for this chunk
            gather_results = await asyncio.gather(
                *[process_schema(meta) for meta in active_schemas],
                return_exceptions=True
            )
            for meta, res in zip(active_schemas, gather_results):
                if isinstance(res, Exception):
                    self.logger.error(
                        f"  [CHUNK ERROR] Doc: '{doc.id}' | Chunk: {chunk_idx}/{len(chunks)} | Schema: '{meta.class_name}' failed: {res}"
                    )

        # Consolidation Pass per pending target schema
        for meta in active_target_schemas:
            chunk_outputs = schema_chunk_outputs.get(meta.qualified_name, [])
            out_path = self.tracker.get_schema_output_path(doc.id, meta)
            try:
                if not chunk_outputs:
                    out_path.write_text(json.dumps([], indent=2), encoding="utf-8")
                    self.logger.info(f"  [CONSOLIDATION EMPTY] No findings for '{meta.qualified_name}'. Saved empty list.")
                    continue

                flattened: List[Any] = []
                for item in chunk_outputs:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)

                if not flattened:
                    out_path.write_text(json.dumps([], indent=2), encoding="utf-8")
                    continue

                if len(flattened) == 1:
                    merged_output = flattened[0]
                else:
                    merged_output = merge_pydantic_instances(meta.schema_cls, flattened)

                if hasattr(merged_output, "model_dump"):
                    dump_data = merged_output.model_dump(mode="json")
                elif isinstance(merged_output, list):
                    dump_data = [it.model_dump(mode="json") if hasattr(it, "model_dump") else it for it in merged_output]
                else:
                    dump_data = merged_output

                out_path.write_text(json.dumps(dump_data, indent=2), encoding="utf-8")
                self.logger.info(f"  [CONSOLIDATION SUCCESS] Doc: '{doc.id}' | Schema: '{meta.qualified_name}' -> {out_path}")
            except Exception as e:
                self.logger.error(f"  [CONSOLIDATION FAILED] Doc: '{doc.id}' | Schema: '{meta.qualified_name}' | Error: {e}")

        # Mark document completed in tracker if all schemas have been completed
        if self.tracker.is_document_completed(doc.id, schemas_meta):
            self.tracker.mark_completed(doc.id, schemas_meta)
        self.logger.info(f"=== [Document {doc_idx}/{total_docs} Finished] ID: {doc.id} ===\n")
