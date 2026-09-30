#!/usr/bin/env python3
"""
Dedicated Multi-File Pydantic AI Extraction Workflow

This script handles complex multi-file schema directories containing mixed types 
(parent container models, standalone models, and embedded child models across multiple .py files).

Key Capabilities:
1. Dynamically imports all .py schema files in a target directory into a global namespace.
2. Applies DAG field-reflection to isolate top-level root models from embedded child models.
3. Dispatches async extraction tasks for top-level schemas per text chunk.
4. Performs field-level model consolidation across chunk outputs.
5. Saves namespace-qualified JSON outputs ('<doc_id>__<module_name>__<class_name>.json')
   to prevent filename and table collisions.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

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
import extract_workflow
import multifile_schema_loader

DEFAULT_INPUT_DIR = "inputs"
DEFAULT_INPUT_FORMAT = "txt"
DEFAULT_SCHEMA_DIR = "schemas/test_schemas/multifile/V1"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_URL = "http://localhost:11434/v1"
DEFAULT_NUM_ASYNC = 4
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 30
DEFAULT_LOG_FILE = "multifile_extraction_workflow.log"

DEFAULT_SYSTEM_PROMPT = (
"""
You are an expert clinical data extraction assistant. 

### Task
Extract structured data from the provided document excerpt into the required JSON schema.

### Constraints & Accuracy Rules
1. **Strict Context Adherence:** Rely ONLY on clear facts directly mentioned in the text.
2. **Anti-Hallucination:** If a field is not explicitly mentioned, set its value to `null` (or default).
3. **Strict Date & Enum Formatting:** Standard ISO format for dates. Exact enum values only.

### Document Excerpt
"""
)


def setup_logger(log_filepath: str) -> logging.Logger:
    logger = logging.getLogger("multifile_extraction_workflow")
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


def create_agent(model: OllamaModel, output_schema: type[BaseModel]) -> Agent:
    return Agent(
        model,
        output_type=PromptedOutput(output_schema),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        retries=3
    )


def is_document_processed(doc_id: str, output_dir: str, schemas_meta: list[multifile_schema_loader.SchemaMeta]) -> bool:
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


async def process_document_multifile(
    doc_id: str,
    input_text: str,
    source_info: str,
    doc_idx: int,
    total_docs: int,
    schemas_meta: list[multifile_schema_loader.SchemaMeta],
    ollama_model: OllamaModel,
    chunk_size: int,
    chunk_overlap: int,
    sync_mode: bool,
    output_dir: str,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger
):
    logger.info(f"=== [Multi-File Document {doc_idx}/{total_docs} Started] ID: {doc_id} ({source_info}) ===")
    try:
        if not input_text:
            logger.warning(f"Skipping document ID '{doc_id}': Input text is empty.")
            return

        chunks = schema_loader.chunk_text(input_text, chunk_size, chunk_overlap)
        logger.info(f"Document '{doc_id}' split into {len(chunks)} chunk(s). Active schemas to extract: {len(schemas_meta)}")

        agents = {
            meta.qualified_name: create_agent(ollama_model, meta.schema_cls)
            for meta in schemas_meta
        }

        base_name = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in doc_id])
        doc_out_dir = os.path.join(output_dir, base_name)
        os.makedirs(doc_out_dir, exist_ok=True)

        schema_chunk_outputs = {meta.qualified_name: [] for meta in schemas_meta}

        for meta in schemas_meta:
            chunk_dir = os.path.join(doc_out_dir, "chunks", meta.qualified_name)
            os.makedirs(chunk_dir, exist_ok=True)

        async def extract_schema_for_chunk(meta: multifile_schema_loader.SchemaMeta, chunk_idx: int, chunk_text_data: str):
            agent = agents[meta.qualified_name]
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

                merged_output = schema_loader.merge_pydantic_instances(meta.schema_cls, chunk_outputs)

                out_filename = f"{base_name}__{meta.module_name}__{meta.class_name}.json"
                out_filepath = os.path.join(doc_out_dir, out_filename)
                with open(out_filepath, "w", encoding="utf-8") as out_f:
                    json.dump(merged_output.model_dump(mode="json"), out_f, indent=4)
                logger.info(f"  [CONSOLIDATION SUCCESS] Doc: '{doc_id}' | Schema: '{meta.qualified_name}' -> {out_filepath}")
            except Exception as e:
                logger.error(f"  [CONSOLIDATION FAILED] Doc: '{doc_id}' | Schema: '{meta.qualified_name}' | Error: {e}")

    except Exception as e:
        logger.error(f"Error processing Doc '{doc_id}': {e}")
    logger.info(f"=== [Multi-File Document {doc_idx}/{total_docs} Finished] ID: {doc_id} ===\n")


async def async_main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file.")
    pre_args, _ = pre_parser.parse_known_args()

    cfg = extract_workflow.load_config_file(pre_args.config)

    input_cfg = cfg.get("input", {})
    schema_cfg = cfg.get("schema", {})
    model_cfg = cfg.get("model", {})
    chunk_cfg = cfg.get("chunking", {})
    exec_cfg = cfg.get("execution", {})

    parser = argparse.ArgumentParser(
        description="Extract structured text dynamically using multi-file Pydantic schema directories."
    )
    parser.add_argument("--config", type=str, default=pre_args.config, help="Path to config.yaml")
    parser.add_argument("--schema-dir", type=str, default=schema_cfg.get("file", DEFAULT_SCHEMA_DIR), help="Multi-file schema directory path")
    parser.add_argument("--extract-mode", type=str, choices=["top_level", "all", "child"], default="top_level", help="Extraction mode for multi-file directory")
    parser.add_argument("--input-dir", type=str, default=input_cfg.get("path", DEFAULT_INPUT_DIR), help="Input directory")
    parser.add_argument("--input-format", type=str, choices=["txt", "parquet"], default=input_cfg.get("format", DEFAULT_INPUT_FORMAT), help="Input format")
    parser.add_argument("--id-col", type=str, default=input_cfg.get("id_column", "identifier"), help="ID column for parquet")
    parser.add_argument("--text-col", type=str, default=input_cfg.get("text_column", "anonymized"), help="Text column for parquet")
    parser.add_argument("--output-dir", type=str, default=exec_cfg.get("output_dir", DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--log-file", type=str, default=exec_cfg.get("log_file", DEFAULT_LOG_FILE), help="Log file path")
    parser.add_argument("--num-async", type=int, default=model_cfg.get("num_async_calls", DEFAULT_NUM_ASYNC), help="Async concurrency limit")
    parser.add_argument("--model", type=str, default=model_cfg.get("name", DEFAULT_MODEL), help="Ollama model name")
    parser.add_argument("--url", type=str, default=model_cfg.get("url", DEFAULT_URL), help="Ollama API base URL")
    parser.add_argument("--sync-mode", action="store_true", default=exec_cfg.get("sync_mode", False), help="Synchronous mode")
    parser.add_argument("--chunk-size", type=int, default=chunk_cfg.get("size", DEFAULT_CHUNK_SIZE), help="Word count chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=chunk_cfg.get("overlap", DEFAULT_CHUNK_OVERLAP), help="Chunk overlap words")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=exec_cfg.get("resume", True), help="Resume mode")
    parser.add_argument("--file", type=str, default=None, help="Process specific input file")

    args = parser.parse_args()

    # Load and classify multi-file schemas
    print(f"Loading and classifying multi-file schema directory: '{args.schema_dir}'...", flush=True)
    schema_meta_list = multifile_schema_loader.load_multifile_schemas(args.schema_dir)

    if args.extract_mode == "top_level":
        target_schemas = multifile_schema_loader.get_top_level_schemas(schema_meta_list)
    elif args.extract_mode == "child":
        target_schemas = multifile_schema_loader.get_child_schemas(schema_meta_list)
    else:
        target_schemas = schema_meta_list

    print(f"Multi-File Schema Classification Complete:")
    print(f"  - Total Schemas Discovered: {len(schema_meta_list)}")
    print(f"  - Top-Level Root Schemas: {len(multifile_schema_loader.get_top_level_schemas(schema_meta_list))}")
    print(f"  - Embedded Child Schemas: {len(multifile_schema_loader.get_child_schemas(schema_meta_list))}")
    print(f"  - Active Target Schemas Selected ({args.extract_mode} mode): {len(target_schemas)}\n")

    if not target_schemas:
        print("Error: No schemas selected for execution.", file=sys.stderr)
        sys.exit(1)

    # Load input documents (supporting txt and parquet)
    docs_to_process = []
    input_fmt = args.input_format.lower()
    target_path = args.file or args.input_dir

    if target_path and (target_path.endswith(".parquet") or target_path.endswith(".pq")) and input_fmt == "txt":
        input_fmt = "parquet"

    if input_fmt == "parquet":
        import pandas as pd
        p_files = [args.file] if args.file else (
            [args.input_dir] if os.path.isfile(args.input_dir) else [
                os.path.join(args.input_dir, f) for f in sorted(os.listdir(args.input_dir))
                if f.endswith(".parquet") or f.endswith(".pq")
            ]
        )
        for p_file in p_files:
            df = pd.read_parquet(p_file)
            for idx, row in df.iterrows():
                raw_id = row[args.id_col]
                text_val = row[args.text_col]
                doc_id = str(int(raw_id)) if isinstance(raw_id, float) and raw_id.is_integer() else str(raw_id).strip()
                docs_to_process.append({
                    "id": doc_id,
                    "text": str(text_val).strip() if pd.notna(text_val) else "",
                    "source": f"{p_file} [row {idx}]"
                })
    else:
        txt_files = [args.file] if args.file else (
            [args.input_dir] if os.path.isfile(args.input_dir) else [
                os.path.join(args.input_dir, f) for f in sorted(os.listdir(args.input_dir))
                if f.endswith(".txt")
            ]
        )
        for filepath in txt_files:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            with open(filepath, "r", encoding="utf-8") as f:
                text_val = f.read().strip()
            docs_to_process.append({
                "id": base_name,
                "text": text_val,
                "source": filepath
            })

    logger = setup_logger(args.log_file)
    logger.info(f"Starting Multi-File Structured Data Extraction Workflow")
    logger.info(f"Schema Directory: '{args.schema_dir}' | Active Schemas: {len(target_schemas)}")
    logger.info(f"Target Documents: {len(docs_to_process)}")

    os.makedirs(args.output_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(args.num_async)

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.num_async, max_keepalive_connections=args.num_async)
    )
    openai_client = AsyncOpenAI(base_url=args.url, api_key="ollama", http_client=http_client)
    shared_provider = OllamaProvider(openai_client=openai_client)
    shared_model = OllamaModel(args.model, provider=shared_provider)

    try:
        processed_count = 0
        skipped_count = 0
        total_docs = len(docs_to_process)

        for doc_idx, doc_item in enumerate(docs_to_process, 1):
            doc_id = doc_item["id"]
            if args.resume and is_document_processed(doc_id, args.output_dir, target_schemas):
                logger.info(f"=== [Document {doc_idx}/{total_docs} Skipped] Record ID: '{doc_id}' already fully processed. ===")
                skipped_count += 1
                continue

            await process_document_multifile(
                doc_id=doc_id,
                input_text=doc_item["text"],
                source_info=doc_item["source"],
                doc_idx=doc_idx,
                total_docs=total_docs,
                schemas_meta=target_schemas,
                ollama_model=shared_model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                sync_mode=args.sync_mode,
                output_dir=args.output_dir,
                semaphore=semaphore,
                logger=logger
            )
            processed_count += 1
        logger.info(f"Multi-File Workflow Complete! Total: {total_docs} | Processed: {processed_count} | Skipped: {skipped_count}")
    finally:
        await http_client.aclose()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
