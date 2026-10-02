#!/usr/bin/env python3
"""
Dynamic Pydantic AI Extraction Workflow

Domain-agnostic, schema-driven extraction engine supporting single Pydantic schema files (.py)
and multi-file schema directories using AST/DAG dependency classification, dynamic domain guidance,
and domain-grouped container extractions for zero-loss LLM performance.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

# Ensure script directory and project root are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.output import PromptedOutput
except ImportError as e:
    print(f"Error: Required dependency missing. Run: pip install pydantic-ai\nDetail: {e}", file=sys.stderr)
    sys.exit(1)

import schema_loader

DEFAULT_INPUT_DIR = "inputs"
DEFAULT_INPUT_FORMAT = "txt"
DEFAULT_SCHEMA_PATH = "schemas/core_master"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_PROMPT_PAYLOADS_DIR = "prompt_payloads"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_URL = "http://localhost:11434/v1"
DEFAULT_NUM_ASYNC = 4
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 30
DEFAULT_LOG_FILE = "extraction_workflow.log"

saved_payload_classes: Set[str] = set()


def build_system_prompt(domain_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Constructs a tailored system prompt for the extraction task.
    If domain_info is provided, dynamically injects domain title, target schemas,
    and specific domain extraction instructions into the prompt context.
    """
    prompt = (
        "You are an expert clinical data extraction assistant.\n\n"
        "### Task\n"
        "Extract structured data from the provided document excerpt into the required JSON schema.\n\n"
    )

    if domain_info:
        title = domain_info.get("domain_title")
        target_classes = domain_info.get("target_classes")
        instructions = domain_info.get("extraction_instructions")

        prompt += "### Focus Area & Domain Guidance\n"
        if title:
            prompt += f"**Domain:** {title}\n"
        if target_classes:
            classes_str = ", ".join(target_classes) if isinstance(target_classes, list) else str(target_classes)
            prompt += f"**Target Schemas:** {classes_str}\n"
        if instructions:
            prompt += f"**Domain Instructions:** {instructions}\n"
        prompt += "\n"

    prompt += (
        "### Constraints & Accuracy Rules\n"
        "1. **Strict Context Adherence:** Rely ONLY on clear facts directly mentioned in the text.\n"
        "2. **Anti-Hallucination:** If a field is not explicitly mentioned, set its value to `null` (or default).\n"
        "3. **Strict Date & Enum Formatting:** Standard ISO format for dates. Exact enum values only.\n\n"
        "### Document Excerpt\n"
    )
    return prompt


def save_first_prompt_payload(
    identifier: str,
    class_name: str,
    module_name: str,
    filepath: str,
    system_prompt: str,
    user_chunk_text: str,
    payload_dir: str,
    logger: logging.Logger
):
    """
    Saves ONLY the first LLM request payload for each individual schema class or domain container
    to the prompt_payloads directory for manual human inspection and review.
    """
    if identifier in saved_payload_classes:
        return

    os.makedirs(payload_dir, exist_ok=True)
    payload_filename = f"{class_name}.txt"
    payload_filepath = os.path.join(payload_dir, payload_filename)

    if os.path.exists(payload_filepath):
        saved_payload_classes.add(identifier)
        return

    payload_content = (
        "================================================================================\n"
        "LLM EXTRACTION PROMPT PAYLOAD (FIRST CALL FOR CLASS / DOMAIN)\n"
        f"Target Class / Container: {class_name}\n"
        f"Identifier: {identifier}\n"
        f"Module File: {module_name}.py\n"
        f"Filepath: {filepath}\n"
        "================================================================================\n\n"
        "--------------------------------------------------------------------------------\n"
        "1. SYSTEM PROMPT\n"
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
        with open(payload_filepath, "w", encoding="utf-8") as f:
            f.write(payload_content)
        saved_payload_classes.add(identifier)
        logger.info(f"  [PROMPT PAYLOAD SAVED] First LLM payload for '{class_name}' saved -> {payload_filepath}")
    except Exception as e:
        logger.warning(f"Failed to save prompt payload for '{class_name}': {e}")


def resolve_config_path(config_filepath: str) -> str:
    """Resolves config path relative to CWD or project root."""
    if os.path.exists(config_filepath):
        return config_filepath
    root_config = os.path.join(project_root, config_filepath)
    if os.path.exists(root_config):
        return root_config
    return config_filepath


def load_config_file(config_filepath: str) -> dict:
    """Loads workflow configuration from a YAML file if it exists."""
    config_filepath = resolve_config_path(config_filepath)
    if not config_filepath or not os.path.exists(config_filepath):
        return {}
    try:
        import yaml
        with open(config_filepath, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        print(f"Warning: Could not parse config file '{config_filepath}': {e}", file=sys.stderr, flush=True)
        return {}


def setup_logger(log_filepath: str) -> logging.Logger:
    logger = logging.getLogger("extraction_workflow")
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


def create_agent(model: OllamaModel, output_schema: type[BaseModel], system_prompt: Optional[str] = None) -> Agent:
    prompt = system_prompt or build_system_prompt()
    return Agent(
        model,
        output_type=PromptedOutput(output_schema),
        system_prompt=prompt,
        retries=3
    )


def is_document_processed(doc_id: str, output_dir: str, schemas_meta: list[schema_loader.SchemaMeta]) -> bool:
    base_name = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in doc_id])
    doc_out_dir = os.path.join(output_dir, base_name)
    if not os.path.exists(doc_out_dir):
        return False
    for meta in schemas_meta:
        out_filename = f"{base_name}__{meta.module_name}__{meta.class_name}.json"
        out_filepath = os.path.join(doc_out_dir, out_filename)
        if not os.path.exists(out_filepath):
            return False
    return True


async def process_document(
    doc_id: str,
    input_text: str,
    source_info: str,
    doc_idx: int,
    total_docs: int,
    schemas_meta: list[schema_loader.SchemaMeta],
    extract_mode: str,
    ollama_model: OllamaModel,
    chunk_size: int,
    chunk_overlap: int,
    sync_mode: bool,
    output_dir: str,
    prompt_payloads_dir: Optional[str],
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    domain_mapping: Optional[Dict[str, Dict[str, Any]]] = None
):
    logger.info(f"=== [Document {doc_idx}/{total_docs} Started] ID: {doc_id} ({source_info}) ===")
    try:
        if not input_text:
            logger.warning(f"Skipping document ID '{doc_id}': Input text is empty.")
            return

        chunks = schema_loader.chunk_text(input_text, chunk_size, chunk_overlap)
        base_name = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in doc_id])
        doc_out_dir = os.path.join(output_dir, base_name)
        os.makedirs(doc_out_dir, exist_ok=True)

        schema_chunk_outputs: Dict[str, List[Any]] = {meta.qualified_name: [] for meta in schemas_meta}

        for meta in schemas_meta:
            chunk_dir = os.path.join(doc_out_dir, "chunks", meta.qualified_name)
            os.makedirs(chunk_dir, exist_ok=True)

        if extract_mode == "domain_grouped":
            domain_containers = schema_loader.build_domain_containers(schemas_meta)
            logger.info(f"Document '{doc_id}' split into {len(chunks)} chunk(s). Active Domain Containers ({len(domain_containers)} domains across {len(schemas_meta)} classes).")

            domain_agents = {}
            domain_prompts = {}
            for d_meta in domain_containers:
                sample_meta = d_meta.member_schemas[0] if d_meta.member_schemas else None
                domain_info = schema_loader.get_domain_info_for_schema(sample_meta, domain_mapping) if sample_meta and domain_mapping else None
                sys_prompt = build_system_prompt(domain_info)
                domain_prompts[d_meta.module_name] = sys_prompt
                domain_agents[d_meta.module_name] = create_agent(ollama_model, d_meta.container_cls, system_prompt=sys_prompt)

            async def extract_domain_for_chunk(d_meta: schema_loader.DomainContainerMeta, chunk_idx: int, chunk_text_data: str):
                agent = domain_agents[d_meta.module_name]
                sys_prompt = domain_prompts[d_meta.module_name]

                # Save prompt payload for domain container
                if prompt_payloads_dir:
                    save_first_prompt_payload(
                        identifier=d_meta.module_name,
                        class_name=d_meta.container_cls.__name__,
                        module_name=d_meta.module_name,
                        filepath=d_meta.filepath,
                        system_prompt=sys_prompt,
                        user_chunk_text=chunk_text_data,
                        payload_dir=prompt_payloads_dir,
                        logger=logger
                    )

                word_count = len(chunk_text_data.split())
                async with semaphore:
                    start_time = time.perf_counter()
                    logger.info(f"  [API DISPATCHED] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} ({word_count} w) | Domain: '{d_meta.module_name}' ({len(d_meta.member_schemas)} classes)")
                    try:
                        if sync_mode:
                            result = await asyncio.to_thread(agent.run_sync, chunk_text_data)
                        else:
                            result = await agent.run(chunk_text_data)

                        elapsed = time.perf_counter() - start_time
                        output = result.output
                        if output is not None:
                            # Unpack member list outputs from domain container
                            for m_meta in d_meta.member_schemas:
                                field_name = f"{m_meta.class_name.lower()}_list"
                                extracted_items = getattr(output, field_name, []) or []
                                if extracted_items:
                                    schema_chunk_outputs[m_meta.qualified_name].extend(extracted_items)

                                    # Save chunk JSON for member schema
                                    schema_out_dir = os.path.join(doc_out_dir, "chunks", m_meta.qualified_name)
                                    chunk_filename = f"{base_name}_chunk_{chunk_idx}.json"
                                    chunk_filepath = os.path.join(schema_out_dir, chunk_filename)
                                    dumped_items = [it.model_dump(mode="json") if hasattr(it, "model_dump") else it for it in extracted_items]
                                    with open(chunk_filepath, "w", encoding="utf-8") as chunk_f:
                                        json.dump(dumped_items, chunk_f, indent=4)

                            logger.info(f"  [API SUCCESS] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} | Domain: '{d_meta.module_name}' | Elapsed: {elapsed:.2f}s")
                        else:
                            logger.warning(f"  [API EMPTY] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} | Domain: '{d_meta.module_name}'")
                    except Exception as e:
                        elapsed = time.perf_counter() - start_time
                        logger.error(f"  [API FAILED] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} | Domain: '{d_meta.module_name}' | Error: {e}")

            for chunk_idx, chunk_text_data in enumerate(chunks, 1):
                logger.info(f"--- [Chunk {chunk_idx}/{len(chunks)}] Dispatching {len(domain_containers)} domain container(s) ---")
                tasks = [
                    extract_domain_for_chunk(d_meta, chunk_idx, chunk_text_data)
                    for d_meta in domain_containers
                ]
                await asyncio.gather(*tasks)

        else:
            logger.info(f"Document '{doc_id}' split into {len(chunks)} chunk(s). Active schemas to extract: {len(schemas_meta)}")
            agents = {}
            system_prompts = {}
            for meta in schemas_meta:
                domain_info = schema_loader.get_domain_info_for_schema(meta, domain_mapping) if domain_mapping else None
                sys_prompt = build_system_prompt(domain_info)
                system_prompts[meta.qualified_name] = sys_prompt
                agents[meta.qualified_name] = create_agent(ollama_model, meta.schema_cls, system_prompt=sys_prompt)

            async def extract_schema_for_chunk(meta: schema_loader.SchemaMeta, chunk_idx: int, chunk_text_data: str):
                agent = agents[meta.qualified_name]
                sys_prompt = system_prompts[meta.qualified_name]

                if prompt_payloads_dir:
                    save_first_prompt_payload(
                        identifier=meta.qualified_name,
                        class_name=meta.class_name,
                        module_name=meta.module_name,
                        filepath=meta.filepath,
                        system_prompt=sys_prompt,
                        user_chunk_text=chunk_text_data,
                        payload_dir=prompt_payloads_dir,
                        logger=logger
                    )

                schema_out_dir = os.path.join(doc_out_dir, "chunks", meta.qualified_name)
                word_count = len(chunk_text_data.split())

                async with semaphore:
                    start_time = time.perf_counter()
                    logger.info(f"  [API DISPATCHED] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} ({word_count} w) | Schema: '{meta.qualified_name}'")
                    try:
                        if sync_mode:
                            result = await asyncio.to_thread(agent.run_sync, chunk_text_data)
                        else:
                            result = await agent.run(chunk_text_data)

                        elapsed = time.perf_counter() - start_time
                        output = result.output
                        if output is not None:
                            schema_chunk_outputs[meta.qualified_name].append(output)

                            chunk_filename = f"{base_name}_chunk_{chunk_idx}.json"
                            chunk_filepath = os.path.join(schema_out_dir, chunk_filename)
                            with open(chunk_filepath, "w", encoding="utf-8") as chunk_f:
                                json.dump(output.model_dump(mode="json"), chunk_f, indent=4)
                            logger.info(f"  [API SUCCESS] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} | Schema: '{meta.qualified_name}' | Elapsed: {elapsed:.2f}s")
                        else:
                            logger.warning(f"  [API EMPTY] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} | Schema: '{meta.qualified_name}'")
                    except Exception as e:
                        elapsed = time.perf_counter() - start_time
                        logger.error(f"  [API FAILED] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} | Schema: '{meta.qualified_name}' | Error: {e}")

            for chunk_idx, chunk_text_data in enumerate(chunks, 1):
                logger.info(f"--- [Chunk {chunk_idx}/{len(chunks)}] Dispatching {len(schemas_meta)} top-level schema(s) ---")
                tasks = [
                    extract_schema_for_chunk(meta, chunk_idx, chunk_text_data)
                    for meta in schemas_meta
                ]
                await asyncio.gather(*tasks)

        # Instance Consolidation Pass per schema
        for meta in schemas_meta:
            chunk_outputs = schema_chunk_outputs[meta.qualified_name]
            try:
                if not chunk_outputs:
                    logger.warning(f"  [CONSOLIDATION SKIPPED] No outputs for '{meta.qualified_name}' in Doc '{doc_id}'.")
                    continue

                if len(chunk_outputs) == 1 and isinstance(chunk_outputs[0], list):
                    flattened_outputs = chunk_outputs[0]
                elif any(isinstance(co, list) for co in chunk_outputs):
                    flattened_outputs = []
                    for co in chunk_outputs:
                        if isinstance(co, list):
                            flattened_outputs.extend(co)
                        else:
                            flattened_outputs.append(co)
                else:
                    flattened_outputs = chunk_outputs

                if not flattened_outputs:
                    continue

                if len(flattened_outputs) == 1:
                    merged_output = flattened_outputs[0]
                else:
                    merged_output = schema_loader.merge_pydantic_instances(meta.schema_cls, flattened_outputs)

                out_filename = f"{base_name}__{meta.module_name}__{meta.class_name}.json"
                out_filepath = os.path.join(doc_out_dir, out_filename)

                dump_data = merged_output.model_dump(mode="json") if hasattr(merged_output, "model_dump") else [
                    it.model_dump(mode="json") if hasattr(it, "model_dump") else it for it in merged_output
                ]

                with open(out_filepath, "w", encoding="utf-8") as out_f:
                    json.dump(dump_data, out_f, indent=4)
                logger.info(f"  [CONSOLIDATION SUCCESS] Doc: '{doc_id}' | Schema: '{meta.qualified_name}' -> {out_filepath}")
            except Exception as e:
                logger.error(f"  [CONSOLIDATION FAILED] Doc: '{doc_id}' | Schema: '{meta.qualified_name}' | Error: {e}")

    except Exception as e:
        logger.error(f"Error processing Doc '{doc_id}': {e}")
    logger.info(f"=== [Document {doc_idx}/{total_docs} Finished] ID: {doc_id} ===\n")


def load_input_documents(
    input_dir: str,
    input_format: str,
    file_override: Optional[str] = None,
    id_col: str = "identifier",
    text_col: str = "anonymized"
) -> List[Dict[str, str]]:
    """Loads input documents from text files or Parquet files."""
    docs = []
    input_fmt = input_format.lower()
    target_path = file_override or input_dir

    if target_path and (target_path.endswith(".parquet") or target_path.endswith(".pq")):
        input_fmt = "parquet"

    if input_fmt == "parquet":
        import pandas as pd
        p_files = [file_override] if file_override else (
            [input_dir] if os.path.isfile(input_dir) else [
                os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))
                if f.endswith(".parquet") or f.endswith(".pq")
            ]
        )
        for p_file in p_files:
            df = pd.read_parquet(p_file)
            for idx, row in df.iterrows():
                raw_id = row[id_col]
                text_val = row[text_col]
                doc_id = str(int(raw_id)) if isinstance(raw_id, float) and raw_id.is_integer() else str(raw_id).strip()
                docs.append({
                    "id": doc_id,
                    "text": str(text_val).strip() if pd.notna(text_val) else "",
                    "source": f"{p_file} [row {idx}]"
                })
    else:
        txt_files = [file_override] if file_override else (
            [input_dir] if os.path.isfile(input_dir) else [
                os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))
                if f.endswith(".txt")
            ]
        )
        for filepath in txt_files:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            with open(filepath, "r", encoding="utf-8") as f:
                text_val = f.read().strip()
            docs.append({
                "id": base_name,
                "text": text_val,
                "source": filepath
            })

    return docs


async def run_extraction(
    config_file: str = "config.yaml",
    schema_path: Optional[str] = None,
    domain_mapping_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    prompt_payloads_dir: Optional[str] = None,
    extract_mode: Optional[str] = None,
    input_dir: Optional[str] = None,
    input_format: Optional[str] = None,
    file_override: Optional[str] = None,
    model: Optional[str] = None,
    url: Optional[str] = None,
    num_async: Optional[int] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    sync_mode: Optional[bool] = None,
    resume: Optional[bool] = None,
    log_file: Optional[str] = None
):
    """Programmatic entry point for running the structured extraction engine."""
    cfg = load_config_file(config_file)
    input_cfg = cfg.get("input", {})
    schema_cfg = cfg.get("schema", {})
    model_cfg = cfg.get("model", {})
    chunk_cfg = cfg.get("chunking", {})
    exec_cfg = cfg.get("execution", {})

    schema_path = schema_path or schema_cfg.get("file", schema_cfg.get("path", DEFAULT_SCHEMA_PATH))
    domain_mapping_file = domain_mapping_file or schema_cfg.get("domain_mapping_file", schema_cfg.get("mapping_file"))
    output_dir = output_dir or exec_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    prompt_payloads_dir = prompt_payloads_dir or exec_cfg.get("prompt_payloads_dir", DEFAULT_PROMPT_PAYLOADS_DIR)
    extract_mode = extract_mode or schema_cfg.get("extract_mode", "domain_grouped")
    input_dir = input_dir or input_cfg.get("path", DEFAULT_INPUT_DIR)
    input_format = input_format or input_cfg.get("format", DEFAULT_INPUT_FORMAT)
    model = model or model_cfg.get("name", DEFAULT_MODEL)
    url = url or model_cfg.get("url", DEFAULT_URL)
    num_async = num_async if num_async is not None else model_cfg.get("num_async_calls", DEFAULT_NUM_ASYNC)
    chunk_size = chunk_size if chunk_size is not None else chunk_cfg.get("size", DEFAULT_CHUNK_SIZE)
    chunk_overlap = chunk_overlap if chunk_overlap is not None else chunk_cfg.get("overlap", DEFAULT_CHUNK_OVERLAP)
    sync_mode = sync_mode if sync_mode is not None else exec_cfg.get("sync_mode", False)
    resume = resume if resume is not None else exec_cfg.get("resume", True)
    log_file = log_file or exec_cfg.get("log_file", DEFAULT_LOG_FILE)

    if prompt_payloads_dir:
        os.makedirs(prompt_payloads_dir, exist_ok=True)

    schema_meta_list = schema_loader.load_schemas(schema_path)

    domain_mapping = {}
    if domain_mapping_file:
        resolved_mapping_path = resolve_config_path(domain_mapping_file)
        domain_mapping = schema_loader.load_domain_mapping(resolved_mapping_path)
        if domain_mapping:
            print(f"Loaded Domain Extraction Mapping from '{domain_mapping_file}' ({len(domain_mapping)} domains defined).")
        else:
            print(f"Warning: Domain mapping file '{domain_mapping_file}' could not be loaded or was empty.")

    if extract_mode == "top_level":
        target_schemas = schema_loader.get_top_level_schemas(schema_meta_list)
    elif extract_mode == "child":
        target_schemas = schema_loader.get_child_schemas(schema_meta_list)
    else:
        target_schemas = schema_meta_list

    print(f"Schema Classification Complete:")
    print(f"  - Total Schemas Discovered: {len(schema_meta_list)}")
    print(f"  - Top-Level Root Schemas: {len(schema_loader.get_top_level_schemas(schema_meta_list))}")
    print(f"  - Embedded Child Schemas: {len(schema_loader.get_child_schemas(schema_meta_list))}")
    print(f"  - Active Target Schemas Selected ({extract_mode} mode): {len(target_schemas)}\n")

    if not target_schemas:
        raise ValueError("No schemas selected for execution.")

    docs_to_process = load_input_documents(
        input_dir=input_dir,
        input_format=input_format,
        file_override=file_override,
        id_col=input_cfg.get("id_column", "identifier"),
        text_col=input_cfg.get("text_column", "anonymized")
    )

    logger = setup_logger(log_file)
    logger.info(f"Starting Structured Data Extraction Workflow")
    logger.info(f"Schema Path: '{schema_path}' | Active Schemas: {len(target_schemas)}")
    if domain_mapping_file:
        logger.info(f"Domain Mapping Context: '{domain_mapping_file}'")
    if prompt_payloads_dir:
        logger.info(f"Prompt Payloads Directory: '{prompt_payloads_dir}'")
    logger.info(f"Target Documents: {len(docs_to_process)}")

    os.makedirs(output_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(num_async)

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=num_async, max_keepalive_connections=num_async)
    )
    openai_client = AsyncOpenAI(base_url=url, api_key="ollama", http_client=http_client)
    shared_provider = OllamaProvider(openai_client=openai_client)
    shared_model = OllamaModel(model, provider=shared_provider)

    try:
        processed_count = 0
        skipped_count = 0
        total_docs = len(docs_to_process)

        for doc_idx, doc_item in enumerate(docs_to_process, 1):
            doc_id = doc_item["id"]
            if resume and is_document_processed(doc_id, output_dir, target_schemas):
                logger.info(f"=== [Document {doc_idx}/{total_docs} Skipped] Record ID: '{doc_id}' already fully processed. ===")
                skipped_count += 1
                continue

            await process_document(
                doc_id=doc_id,
                input_text=doc_item["text"],
                source_info=doc_item["source"],
                doc_idx=doc_idx,
                total_docs=total_docs,
                schemas_meta=target_schemas,
                extract_mode=extract_mode,
                ollama_model=shared_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                sync_mode=sync_mode,
                output_dir=output_dir,
                prompt_payloads_dir=prompt_payloads_dir,
                semaphore=semaphore,
                logger=logger,
                domain_mapping=domain_mapping
            )
            processed_count += 1
        logger.info(f"Workflow Complete! Total: {total_docs} | Processed: {processed_count} | Skipped: {skipped_count}")
    finally:
        await http_client.aclose()


async def async_main():
    parser = argparse.ArgumentParser(
        description="Extract structured text dynamically using Pydantic schema files or directories."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--schema-path", "--schema-dir", "--schema-file", dest="schema_path", type=str, default=None, help="Path to single schema file (.py) or schema directory")
    parser.add_argument("--domain-mapping-file", "--mapping-file", dest="domain_mapping_file", type=str, default=None, help="Path to domain mapping python file (.py) containing DOMAIN_EXTRACTION_MAP")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for extracted JSON files")
    parser.add_argument("--prompt-payloads-dir", "--payloads-dir", dest="prompt_payloads_dir", type=str, default=None, help="Directory to save first LLM prompt payload per schema class")
    parser.add_argument("--extract-mode", type=str, choices=["domain_grouped", "top_level", "all", "child"], default=None, help="DAG extraction mode (default: domain_grouped)")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory")
    parser.add_argument("--input-format", type=str, choices=["txt", "parquet"], default=None, help="Input format")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    parser.add_argument("--num-async", type=int, default=None, help="Async concurrency limit")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name")
    parser.add_argument("--url", type=str, default=None, help="Ollama API base URL")
    parser.add_argument("--sync-mode", action="store_true", default=None, help="Synchronous mode")
    parser.add_argument("--chunk-size", type=int, default=None, help="Word count chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap words")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None, help="Resume mode")
    parser.add_argument("--file", type=str, default=None, help="Process specific input file")

    args = parser.parse_args()
    await run_extraction(
        config_file=args.config,
        schema_path=args.schema_path,
        domain_mapping_file=args.domain_mapping_file,
        output_dir=args.output_dir,
        prompt_payloads_dir=args.prompt_payloads_dir,
        extract_mode=args.extract_mode,
        input_dir=args.input_dir,
        input_format=args.input_format,
        file_override=args.file,
        model=args.model,
        url=args.url,
        num_async=args.num_async,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sync_mode=args.sync_mode,
        resume=args.resume,
        log_file=args.log_file
    )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
