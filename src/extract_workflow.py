#!/usr/bin/env python3
"""
Dynamic Pydantic AI Extraction Workflow
This script demonstrates how to extract structured data from unstructured text
using pydantic-ai and a local LLM via Ollama. It is completely domain-agnostic:
1. Dynamically loads a Pydantic schema from an external Python file (.py).
2. Supports extracting via a single root container model OR looping over all individual child schemas.
3. Performs dynamic text chunking (word-based) with customizable size/overlap.
4. Applies a schema-agnostic Map-Reduce merging algorithm.
5. Executes synchronously or asynchronously.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

# Ensure script directory and project root are in sys.path for relative imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

# Ensure pydantic-ai is installed
try:
    from pydantic_ai import Agent
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.output import PromptedOutput
except ImportError as e:
    print(f"Error: Required dependency missing. Please run: pip install pydantic-ai\nDetail: {e}", file=sys.stderr)
    sys.exit(1)

# Import dynamic schema loading and merging helpers
import schema_loader

# =====================================================================
# Global Configuration Default Constants
# Adjust these default settings to customize the extraction workflow.
# All defaults can also be overridden at runtime via CLI arguments.
# =====================================================================

# ---------------------------------------------------------------------
# 1. Input & Output Paths
# ---------------------------------------------------------------------
# Directory path (e.g. "reports_25") OR direct file path (e.g. "inputs/data.parquet") to process
DEFAULT_INPUT_DIR = "inputs"
DEFAULT_INPUT_FORMAT = "txt"  # Options: "txt" | "parquet"
DEFAULT_SCHEMA_FILE = "schemas/test_schemas/root_container_schema_example.py"
DEFAULT_USE_ROOT_SCHEMA = True
DEFAULT_OUTPUT_DIR = "outputs"

# Resume / Overwrite Mode:
# True  : Resumes workflow by skipping records that already have complete output JSON files
# False : Overwrites existing outputs and re-processes all records
DEFAULT_RESUME = True

# Log file path for detailed request and lifecycle tracking
DEFAULT_LOG_FILE = "extraction_workflow.log"


def setup_logger(log_filepath: str) -> logging.Logger:
    """Sets up a file and console logger for execution and request status tracking."""
    logger = logging.getLogger("extraction_workflow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File Handler
    if log_filepath:
        log_dir = os.path.dirname(log_filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_filepath, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ---------------------------------------------------------------------
# 2. Parquet Field Mapping (Used when DEFAULT_INPUT_FORMAT = "parquet")
# ---------------------------------------------------------------------
# Column in the Parquet file representing the document ID (e.g. "identifier", "patient_id")
DEFAULT_PARQUET_ID_COL = "identifier"

# Column in the Parquet file containing raw report text (e.g. "anonymized", "report_text")
DEFAULT_PARQUET_TEXT_COL = "anonymized"


# ---------------------------------------------------------------------
# 3. Model & Execution Parameters
# ---------------------------------------------------------------------
# Ollama model identifier to use for extraction (e.g. "SemP16k-Qwen122:latest")
DEFAULT_MODEL = "SemP32k-Qwen:v1"

# Maximum number of concurrent asynchronous LLM extraction requests
NUM_ASYNC_CALLS = 8


# ---------------------------------------------------------------------
# 4. Text Chunking Configuration
# ---------------------------------------------------------------------
# Target word count per chunk. Set to 0 to disable chunking for short reports.
DEFAULT_CHUNK_SIZE = 1800

# Number of overlapping words between consecutive text chunks
DEFAULT_CHUNK_OVERLAP = 30


# ---------------------------------------------------------------------
# 5. System Prompt & Extraction Guidelines
# ---------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
"""
You are an expert clinical data extraction assistant. 

### Task
Extract structured clinical data from the provided medical report excerpt into the required JSON schema.

### Constraints & Accuracy Rules
1. **Strict Context Adherence:** Rely ONLY on the clear facts directly mentioned in the report. Do not assume, extrapolate, or bring in outside medical knowledge.
2. **Anti-Hallucination:** If a schema field is not explicitly mentioned or cannot be directly inferred from the text, set its value to `null` (or the schema's designated default). Never invent dates, values, or diagnoses.
3. **No Speculation:** If a diagnosis is listed as "ruled out" or "suspected," do not extract it as a confirmed condition unless the schema explicitly asks for suspected cases.
4. **Utmost Accuracy**  Maintain strict adherence to accuracy when identifying content in text, ensure relative dates (past,present,future) and other references are handled accurately.
5. **Patient Focus**  If the reports are anonymized or make unclear references, maintain a patient centric focus to preserve the integrity of content extracts.
6. **Strict Date Formatting:** All date and datetime fields MUST be in standard ISO format (e.g. YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). If a date is missing, incomplete, or invalid, set the field value to `null`. NEVER use placeholder strings like "unknown", "N/A", "present", or empty strings `""` for date fields.
7. **Strict Enum Values:** For fields that restrict options to specific enums, you MUST only output one of the allowed enum values. If the value is not in the allowed list, set the field to `null`.

### Clinical Report Excerpt
"""
)


# =====================================================================
# 1. Agent Configuration
# =====================================================================

def create_agent(model: OllamaModel, output_schema: type[BaseModel]) -> Agent:
    """Configures and returns a Pydantic AI agent for structured extraction using a shared model/provider connection pool."""
    agent = Agent(
        model,
        output_type=PromptedOutput(output_schema),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        retries=3
    )
    return agent


# =====================================================================
# 2. Loop Core Helper Functions
# =====================================================================

def is_document_processed(doc_id: str, output_dir: str, schemas_to_run: list[type[BaseModel]]) -> bool:
    """Checks whether a document has already been fully processed and stored in output_dir."""
    base_name = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in doc_id])
    doc_out_dir = os.path.join(output_dir, base_name)
    if not os.path.exists(doc_out_dir):
        return False
    for schema_cls in schemas_to_run:
        schema_name = schema_cls.__name__
        out_filepath = os.path.join(doc_out_dir, f"{base_name}_{schema_name}.json")
        if not os.path.exists(out_filepath):
            return False
    return True

async def process_document_sequentially(
    doc_id: str,
    input_text: str,
    source_info: str,
    doc_idx: int,
    total_docs: int,
    schemas_to_run: list[type[BaseModel]],
    ollama_model: OllamaModel,
    chunk_size: int,
    chunk_overlap: int,
    sync_mode: bool,
    output_dir: str,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger
):
    """Processes a single document/item: chunks, runs chunk-by-chunk sequentially,
    extracting all schemas concurrently per chunk, then merges and saves results.
    """
    logger.info(f"=== [Document {doc_idx}/{total_docs} Started] processing: {source_info} (ID: {doc_id}) ===")
    try:
        if not input_text:
            logger.warning(f"Skipping document ID '{doc_id}': Input text is empty.")
            return

        # 1. Chunk text
        chunks = schema_loader.chunk_text(input_text, chunk_size, chunk_overlap)
        logger.info(f"Document '{doc_id}' split into {len(chunks)} chunk(s)")

        # 2. Create Agents for all schemas using shared model/client pool
        agents = {
            schema_cls: create_agent(ollama_model, schema_cls)
            for schema_cls in schemas_to_run
        }

        # Sanitize base_name for valid file paths
        base_name = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in doc_id])

        # Create a dedicated output folder for this report ID
        doc_out_dir = os.path.join(output_dir, base_name)
        os.makedirs(doc_out_dir, exist_ok=True)

        # Track outputs for each schema across chunks: {schema_cls: [output1, output2, ...]}
        schema_chunk_outputs = {schema_cls: [] for schema_cls in schemas_to_run}

        # Create designated output folders for all schemas under this report's directory
        for schema_cls in schemas_to_run:
            schema_out_dir = os.path.join(doc_out_dir, "chunks", schema_cls.__name__)
            os.makedirs(schema_out_dir, exist_ok=True)

        # Helper coroutine for executing a single schema extraction task
        async def extract_schema_for_chunk(schema_cls, chunk_idx, chunk_text_data):
            agent = agents[schema_cls]
            schema_name = schema_cls.__name__
            schema_out_dir = os.path.join(doc_out_dir, "chunks", schema_name)
            word_count = len(chunk_text_data.split())
            
            async with semaphore:
                start_time = time.perf_counter()
                logger.info(f"  [API DISPATCHED] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} ({word_count} words) | Schema: '{schema_name}'")
                try:
                    if sync_mode:
                        result = await asyncio.to_thread(agent.run_sync, chunk_text_data)
                    else:
                        result = await agent.run(chunk_text_data)
                    
                    elapsed = time.perf_counter() - start_time
                    output = result.output
                    if output is not None:
                        schema_chunk_outputs[schema_cls].append(output)
                        
                        # Save intermediary chunk output as a JSON file in the report's chunk folder
                        chunk_filename = f"{base_name}_chunk_{chunk_idx}.json"
                        chunk_filepath = os.path.join(schema_out_dir, chunk_filename)
                        with open(chunk_filepath, "w", encoding="utf-8") as chunk_f:
                            json.dump(output.model_dump(mode="json"), chunk_f, indent=4)
                        logger.info(f"  [API SUCCESS] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} | Schema: '{schema_name}' | Elapsed: {elapsed:.2f}s | Saved: {chunk_filepath}")
                    else:
                        logger.warning(f"  [API EMPTY] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} | Schema: '{schema_name}' | Elapsed: {elapsed:.2f}s | Status: Returned empty output")
                except Exception as e:
                    elapsed = time.perf_counter() - start_time
                    logger.error(f"  [API FAILED] Doc: '{doc_id}' | Chunk: {chunk_idx}/{len(chunks)} | Schema: '{schema_name}' | Elapsed: {elapsed:.2f}s | Error: {e}")

        # Map Phase: Process each chunk across ALL schemas CONCURRENTLY (bounded by semaphore)
        for chunk_idx, chunk_text_data in enumerate(chunks, 1):
            logger.info(f"--- [Chunk {chunk_idx}/{len(chunks)} Started] Dispatching {len(schemas_to_run)} schema(s) for Doc '{doc_id}' ---")
            
            schema_tasks = [
                extract_schema_for_chunk(schema_cls, chunk_idx, chunk_text_data)
                for schema_cls in schemas_to_run
            ]
            await asyncio.gather(*schema_tasks)

        # Reduce Phase: Merge and save output for each schema in the report's dedicated folder
        for schema_cls in schemas_to_run:
            schema_name = schema_cls.__name__
            chunk_outputs = schema_chunk_outputs[schema_cls]
            try:
                if not chunk_outputs:
                    logger.warning(f"  [REDUCE SKIPPED] No successful extractions for Schema '{schema_name}' in Doc '{doc_id}'.")
                    continue

                merged_output = schema_loader.merge_pydantic_instances(schema_cls, chunk_outputs)
                
                # Save final output as a JSON file in the report's dedicated directory
                out_filename = f"{base_name}_{schema_name}.json"
                out_filepath = os.path.join(doc_out_dir, out_filename)
                with open(out_filepath, "w", encoding="utf-8") as out_f:
                    json.dump(merged_output.model_dump(mode="json"), out_f, indent=4)
                logger.info(f"  [REDUCE SUCCESS] Doc: '{doc_id}' | Schema: '{schema_name}' | Saved final output: {out_filepath}")
            except Exception as e:
                logger.error(f"  [REDUCE FAILED] Doc: '{doc_id}' | Schema: '{schema_name}' | Error: {e}")
            
    except Exception as e:
        logger.error(f"Error processing Doc '{doc_id}' ({source_info}): {e}")
    logger.info(f"=== [Document {doc_idx}/{total_docs} Finished] processing: {source_info} ===\n")


# =====================================================================
# 3. Main CLI Entrypoint
# =====================================================================

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


async def async_main():
    # 0. Pre-parse --config option to load YAML settings as default fallbacks
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file.")
    pre_args, _ = pre_parser.parse_known_args()

    cfg = load_config_file(pre_args.config)

    # Extract settings from config file (falling back to module DEFAULT constants)
    input_cfg = cfg.get("input", {})
    schema_cfg = cfg.get("schema", {})
    model_cfg = cfg.get("model", {})
    chunk_cfg = cfg.get("chunking", {})
    exec_cfg = cfg.get("execution", {})

    cfg_input_dir = input_cfg.get("path", DEFAULT_INPUT_DIR)
    cfg_input_fmt = input_cfg.get("format", DEFAULT_INPUT_FORMAT)
    cfg_id_col = input_cfg.get("id_column", DEFAULT_PARQUET_ID_COL)
    cfg_text_col = input_cfg.get("text_column", DEFAULT_PARQUET_TEXT_COL)

    cfg_schema_file = schema_cfg.get("file", DEFAULT_SCHEMA_FILE)
    cfg_schema_class = schema_cfg.get("class_name", None)
    cfg_use_root = schema_cfg.get("use_root_schema", DEFAULT_USE_ROOT_SCHEMA)

    cfg_model = model_cfg.get("name", DEFAULT_MODEL)
    cfg_url = model_cfg.get("url", "http://localhost:11434/v1")
    cfg_num_async = model_cfg.get("num_async_calls", NUM_ASYNC_CALLS)

    cfg_chunk_size = chunk_cfg.get("size", DEFAULT_CHUNK_SIZE)
    cfg_chunk_overlap = chunk_cfg.get("overlap", DEFAULT_CHUNK_OVERLAP)

    cfg_output_dir = exec_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    cfg_resume = exec_cfg.get("resume", DEFAULT_RESUME)
    cfg_sync_mode = exec_cfg.get("sync_mode", False)
    cfg_log_file = exec_cfg.get("log_file", DEFAULT_LOG_FILE)

    parser = argparse.ArgumentParser(
        description="Extract structured text dynamically using Python-defined Pydantic schemas."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help="Path to YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=cfg_input_dir,
        help=f"Directory containing input files (default: {cfg_input_dir})."
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["txt", "parquet"],
        default=cfg_input_fmt,
        help=f"Format of input files: 'txt' or 'parquet' (default: {cfg_input_fmt})."
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default=cfg_id_col,
        help=f"Column name for document ID in parquet files (default: {cfg_id_col})."
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=cfg_text_col,
        help=f"Column name for document text content in parquet files (default: {cfg_text_col})."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=cfg_output_dir,
        help=f"Directory to save output JSON files (default: {cfg_output_dir})."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=cfg_log_file,
        help=f"File path for detailed request and status logging (default: {cfg_log_file})."
    )
    parser.add_argument(
        "--num-async",
        type=int,
        default=cfg_num_async,
        help=f"Number of concurrent async calls (default: {cfg_num_async})."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process a specific input file. If omitted, all matching format files in the input directory are processed."
    )
    
    # Schema selection arguments
    parser.add_argument(
        "--schema-file",
        type=str,
        default=cfg_schema_file,
        help=f"Path to a Python file (.py) defining a Pydantic schema (default: {cfg_schema_file})."
    )
    parser.add_argument(
        "--schema-class",
        type=str,
        default=cfg_schema_class,
        help="Name of a specific Pydantic BaseModel class within --schema-file."
    )
    parser.add_argument(
        "--use-root-schema",
        action=argparse.BooleanOptionalAction,
        default=cfg_use_root,
        help=f"Target a single root container summary model for fast extraction (default: {cfg_use_root}). Set --no-use-root-schema to run all individual child schemas."
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=cfg_resume,
        help=f"Skip records that already have complete final output JSON files in output-dir (default: {cfg_resume}). Set --no-resume to force overwrite."
    )
    
    # Model configuration arguments
    parser.add_argument(
        "--model",
        type=str,
        default=cfg_model,
        help=f"Ollama model name (default: {cfg_model})."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=cfg_url,
        help=f"Ollama API base URL (default: {cfg_url})."
    )
    parser.add_argument(
        "--sync-mode",
        action="store_true",
        default=cfg_sync_mode,
        help="Run the extraction synchronously instead of asynchronously."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=cfg_chunk_size,
        help=f"Word-based chunk size for dynamic chunking (default: {cfg_chunk_size}). Set to 0 to disable chunking."
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=cfg_chunk_overlap,
        help=f"Number of overlapping words between chunks (default: {cfg_chunk_overlap})."
    )

    args = parser.parse_args()

    # 1. Load the Schema Class(es)
    try:
        schema_files = schema_loader.resolve_schema_files(args.schema_file)
        schemas_to_run = []

        if args.schema_class:
            for s_file in schema_files:
                try:
                    cls = schema_loader.load_schema_from_python(s_file, args.schema_class)
                    schemas_to_run.append(cls)
                except (AttributeError, TypeError):
                    pass
            if not schemas_to_run:
                print(f"Error: Class '{args.schema_class}' not found in specified schema file(s).", file=sys.stderr, flush=True)
                sys.exit(1)
            print(f"Loaded specific schema '{args.schema_class}' from {len(schema_files)} file(s).", flush=True)

        elif args.use_root_schema:
            print(f"Auto-detecting root container schema(s) from {len(schema_files)} python file(s)...", flush=True)
            for s_file in schema_files:
                try:
                    root_cls = schema_loader.find_root_schema(s_file)
                    schemas_to_run.append(root_cls)
                    print(f"  - [{os.path.basename(s_file)}] Root Schema: '{root_cls.__name__}'", flush=True)
                except ValueError as e:
                    print(f"  - [{os.path.basename(s_file)}] Skipped (no schema classes found)", flush=True)
        else:
            print(f"Loading all individual child schemas from {len(schema_files)} python file(s)...", flush=True)
            for s_file in schema_files:
                file_schemas = schema_loader.load_all_schemas_from_python(s_file)
                schemas_to_run.extend(file_schemas)
                print(f"  - [{os.path.basename(s_file)}] Loaded {len(file_schemas)} schema(s)", flush=True)

        # Deduplicate schemas while preserving definition order
        unique_schemas = []
        for s in schemas_to_run:
            if s not in unique_schemas:
                unique_schemas.append(s)
        schemas_to_run = unique_schemas

        if not schemas_to_run:
            print(f"Error: No valid Pydantic BaseModel schemas found in {args.schema_file}", file=sys.stderr, flush=True)
            sys.exit(1)
    except Exception as e:
        print(f"Error loading schemas: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 2. Determine documents/records to process based on input format
    docs_to_process = []
    input_fmt = args.input_format.lower()
    
    # Auto-detect parquet format if specified target file/path ends with .parquet or .pq
    target_path = args.file or args.input_dir
    if target_path and (target_path.endswith(".parquet") or target_path.endswith(".pq")) and input_fmt == "txt":
        input_fmt = "parquet"

    if input_fmt == "parquet":
        try:
            import pandas as pd
        except ImportError:
            print("Error: 'pandas' and 'pyarrow' (or 'fastparquet') are required for Parquet input support. Run: pip install pandas pyarrow", file=sys.stderr, flush=True)
            sys.exit(1)

        parquet_files = []
        if args.file:
            if os.path.exists(args.file):
                parquet_files.append(args.file)
            else:
                print(f"Error: Specified file does not exist: {args.file}", file=sys.stderr, flush=True)
                sys.exit(1)
        else:
            if os.path.isfile(args.input_dir):
                parquet_files.append(args.input_dir)
            elif os.path.isdir(args.input_dir):
                p_files = sorted([
                    os.path.join(args.input_dir, f)
                    for f in os.listdir(args.input_dir)
                    if f.endswith(".parquet") or f.endswith(".pq")
                ])
                parquet_files.extend(p_files)
                if not parquet_files:
                    print(f"Warning: No .parquet files found in directory '{args.input_dir}'", file=sys.stderr, flush=True)
            else:
                print(f"Error: Input path '{args.input_dir}' does not exist as a file or directory.", file=sys.stderr, flush=True)
                sys.exit(1)

        for p_file in parquet_files:
            try:
                df = pd.read_parquet(p_file)
            except Exception as e:
                print(f"Error reading Parquet file '{p_file}': {e}", file=sys.stderr, flush=True)
                sys.exit(1)

            missing_cols = []
            if args.id_col not in df.columns:
                missing_cols.append(args.id_col)
            if args.text_col not in df.columns:
                missing_cols.append(args.text_col)

            if missing_cols:
                print(
                    f"Error: Required column(s) {missing_cols} missing in '{p_file}'. "
                    f"Available columns: {list(df.columns)}",
                    file=sys.stderr,
                    flush=True
                )
                sys.exit(1)

            for idx, row in df.iterrows():
                raw_id = row[args.id_col]
                text_val = row[args.text_col]
                
                if pd.notna(raw_id):
                    doc_id = str(int(raw_id)) if isinstance(raw_id, float) and raw_id.is_integer() else str(raw_id).strip()
                else:
                    doc_id = f"row_{idx}"
                    
                input_text = str(text_val).strip() if pd.notna(text_val) else ""

                docs_to_process.append({
                    "id": doc_id,
                    "text": input_text,
                    "source": f"{p_file} [row {idx}, ID: {doc_id}]"
                })

    else:  # "txt" format
        files_to_process = []
        if args.file:
            if os.path.exists(args.file):
                files_to_process.append(args.file)
            else:
                print(f"Error: Specified file does not exist: {args.file}", file=sys.stderr, flush=True)
                sys.exit(1)
        else:
            if os.path.isfile(args.input_dir):
                files_to_process.append(args.input_dir)
            elif os.path.isdir(args.input_dir):
                txt_files = sorted([
                    os.path.join(args.input_dir, f)
                    for f in os.listdir(args.input_dir)
                    if f.endswith(".txt")
                ])
                files_to_process.extend(txt_files)
                if not files_to_process:
                    print(f"Warning: No .txt files found in directory '{args.input_dir}'", file=sys.stderr, flush=True)
            else:
                print(f"Error: Input path '{args.input_dir}' does not exist as a file or directory.", file=sys.stderr, flush=True)
                sys.exit(1)

        for filepath in files_to_process:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text_val = f.read().strip()
            except Exception as e:
                print(f"Error reading file '{filepath}': {e}", file=sys.stderr, flush=True)
                text_val = ""

            docs_to_process.append({
                "id": base_name,
                "text": text_val,
                "source": filepath
            })

    # Initialize logger
    logger = setup_logger(args.log_file)

    logger.info("=====================================================================")
    logger.info("Starting Dynamic Structured Extraction Workflow")
    logger.info(f"Log File: {args.log_file}")
    logger.info(f"Model: {args.model} | Server: {args.url} | Max Concurrency: {args.num_async}")
    logger.info(f"Input Format: {input_fmt} | Schema File: {args.schema_file}")
    logger.info(f"Root Schema Mode: {args.use_root_schema} | Resume Mode: {args.resume}")
    if input_fmt == "parquet":
        logger.info(f"Parquet Columns: ID='{args.id_col}', Text='{args.text_col}'")
    logger.info(f"Schemas to run ({len(schemas_to_run)}): {', '.join([s.__name__ for s in schemas_to_run])}")
    logger.info(f"Mode: {'Synchronous' if args.sync_mode else 'Asynchronous'} | Chunk Size: {args.chunk_size if args.chunk_size > 0 else 'Disabled'}")
    logger.info(f"Found {len(docs_to_process)} document(s)/record(s) to process.")
    logger.info("=====================================================================\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create the semaphore using the configured limit
    semaphore = asyncio.Semaphore(args.num_async)

    # Create a single unified HTTP connection pool strictly capped at max_connections=args.num_async
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=args.num_async,
            max_keepalive_connections=args.num_async
        )
    )
    openai_client = AsyncOpenAI(base_url=args.url, api_key="ollama", http_client=http_client)
    shared_provider = OllamaProvider(openai_client=openai_client)
    shared_model = OllamaModel(args.model, provider=shared_provider)

    try:
        # 3. Process each document sequentially
        logger.info(f"Processing {len(docs_to_process)} document(s)/record(s) sequentially...")
        total_docs = len(docs_to_process)
        processed_count = 0
        skipped_count = 0

        for doc_idx, doc_item in enumerate(docs_to_process, 1):
            doc_id = doc_item["id"]
            if args.resume and is_document_processed(doc_id, args.output_dir, schemas_to_run):
                logger.info(f"=== [Document {doc_idx}/{total_docs} Skipped] Record ID: '{doc_id}' already fully processed. ===")
                skipped_count += 1
                continue

            await process_document_sequentially(
                doc_id=doc_id,
                input_text=doc_item["text"],
                source_info=doc_item["source"],
                doc_idx=doc_idx,
                total_docs=total_docs,
                schemas_to_run=schemas_to_run,
                ollama_model=shared_model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                sync_mode=args.sync_mode,
                output_dir=args.output_dir,
                semaphore=semaphore,
                logger=logger
            )
            processed_count += 1
        logger.info(f"All tasks completed! Total: {total_docs} | Processed: {processed_count} | Skipped: {skipped_count}")
    finally:
        await http_client.aclose()


import extract_multifile_workflow


async def async_main():
    await extract_multifile_workflow.async_main()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

