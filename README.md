# Dynamic Structured Extraction & Aggregation Framework (SchemaMap)

A domain-agnostic, schema-driven framework designed to extract structured JSON data from unstructured clinical, medical, or general text documents using **`pydantic-ai`** and local Large Language Models (via **Ollama**).

The framework dynamically loads Pydantic schemas at runtime (supporting both single `.py` files and multi-file schema directories with mixed types), applies **AST/DAG field-reflection** to isolate root models or dynamically synthesizes **Domain Container Models** (`domain_grouped` mode) for zero-loss multi-class extraction. It segments long documents using sliding-window word chunking, dispatches concurrent LLM extraction requests, preserves **100% exact raw extracted values** across chunk outputs with zero string or numerical alteration, and normalizes JSON extractions into relational **CSV tables and SQLite databases**.

---

## Architecture Overview

```
[ Unstructured Text / Parquet ]
               │
               ▼
   1. Multi-File DAG & Domain Container Loader (src/schema_loader.py)
      - Resolves cross-file symbol namespaces
      - Maps schemas to domain instructions (core_domain_mapping.py / summary_domain_mapping.py)
      - Synthesizes domain container models (domain_grouped) for 1-pass multi-class extraction
               │
               ▼
   2. Text Segmentation & Word Chunking (Sliding Window)
               │
               ▼
   3. Chunk Extraction Pass: Concurrent LLM Extractions (pydantic-ai + Ollama)
      - Logs exact prompt payloads to prompt_payloads/ for human inspection
               │
               ▼
   4. Exact Data Integrity Consolidation (src/schema_loader.py)
      - Preserves verbatim raw text, dates, scores, and enums
      - Deduplicates list items via exact raw JSON equality
      - Zero text mutation, zero float averaging, zero majority voting
               │
               ▼
   5. Namespaced JSON Storage (outputs/<source_id>__<module>__<class>.json)
               │
               ▼
   6. Relational Table Aggregator (src/aggregate_outputs.py)
               │
       ┌────────┴────────┐
       ▼                 ▼
   CSV Tables     SQLite Database (.db)
```

---

## Key Features

- **Multi-File & Domain-Grouped Support**: Automatically process single `.py` schema files or multi-file schema directories (e.g., `schemas/core_master`, `schemas/summary_master`).
- **Domain Container Synthesis (`domain_grouped` mode)**: Automatically groups granular schema classes per domain file into container models, allowing full multi-class extraction in **1 LLM call per domain file** with zero information loss (3.3x–4.6x speedup).
- **Customized Domain Prompting**: Supports domain mapping dictionaries (`schemas/core_domain_mapping.py`, `schemas/summary_domain_mapping.py`) that inject specialized clinical extraction instructions and focus area guidance directly into LLM prompts.
- **Human Inspection Prompt Payloads**: Automatically saves the exact prompt payload of the first LLM request for each class/domain to `prompt_payloads/` for manual review.
- **AST / DAG Hierarchy Classification (`top_level` mode)**: Uses type-hint reflection across `BaseModel.model_fields` to isolate top-level root models from embedded child models when extracting un-grouped schemas.
- **Exact Raw Data Integrity (Zero Alteration Guarantee)**:
  - **Text & Numbers**: Extracted raw strings, standard scores, and ratings are preserved verbatim without lowercasing, truncation, or mathematical averaging.
  - **Lists**: Extended and deduplicated across chunk outputs using exact raw JSON equality.
  - **Enums & Booleans**: Preserved exactly as reported in primary clinical mentions without majority-vote alteration.
- **Collision-Free Namespace Isolation**: Formats output filenames using a double-underscore convention (`<source_id>__<module_name>__<schema_name>.json`), preventing filename collisions across modules.
- **Relational Data Aggregator**: Flattens nested JSON extractions into normalized parent-child relational tables linked by `source_id`, `source_file`, `parent_item_index`, and `item_index` foreign keys.
- **Resumable Execution**: Automatically skips already completed output JSON files.

---

## Extraction Modes

| Extract Mode | Description | Recommended Use Case |
| :--- | :--- | :--- |
| **`domain_grouped`** *(Default)* | Group granular schema classes by domain file into single container models. | Multi-file schema directories (`core_master`, `summary_master`) for maximum LLM efficiency & 100% extraction completeness. |
| **`top_level`** | Uses DAG introspection to extract top-level root container models only. | Standard nested single-root Pydantic model hierarchies. |
| **`all`** | Discovers and extracts all Pydantic schemas individually. | Exhaustive extraction across standalone schemas. |
| **`child`** | Extracts embedded child component schemas directly. | Targeted extraction on nested sub-models. |

---

## Directory Structure

```
SchemaMap/
├── run_pipeline.py                    # Unified end-to-end pipeline runner
├── config.yaml                        # Centralized workflow & aggregation configuration
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation
├── inputs/                            # Input text (.txt) or Parquet files
│   ├── sample_clinical_report.txt
│   └── sample_contact.txt
├── schemas/                           # Python Pydantic schema definitions
│   ├── core_master/                   # Core master schema directory (6 domain modules)
│   ├── core_domain_mapping.py         # Domain instructions & mapping for core_master
│   ├── summary_master/                # Summary master schema directory (6 domain modules)
│   ├── summary_domain_mapping.py      # Domain instructions & mapping for summary_master
│   └── test_schemas/                  # Unit test & example schemas
├── prompt_payloads/                   # Saved first-call LLM prompt payloads for human inspection
├── outputs/                           # Extracted structured JSON files
├── aggregated_tables/                 # Exported relational CSV tables and SQLite database (.db)
└── src/                               # Engine source code
    ├── extract_workflow.py            # Async LLM extraction workflow engine
    ├── schema_loader.py               # Dynamic schema loader, chunker, container builder & raw merger
    └── aggregate_outputs.py           # Relational multi-table aggregator & exporter
```

---

## Installation & Setup

### 1. Prerequisites
- **Python**: 3.11+
- **Ollama**: Installed and running locally (`http://localhost:11434/v1`)
- **LLM Model**: Pulled target model in Ollama (e.g., `ollama pull qwen2.5:7b`)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/childmindresearch/SchemaMap.git
cd SchemaMap

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart Guide

### 1. Configuration (`config.yaml`)
Customize extraction parameters in `config.yaml`:
```yaml
input:
  path: "inputs"
  format: "txt"

schema:
  file: "schemas/core_master"
  domain_mapping_file: "schemas/core_domain_mapping.py"
  extract_mode: "domain_grouped"

model:
  name: "qwen2.5:7b"
  url: "http://localhost:11434/v1"
  num_async_calls: 4

chunking:
  size: 800
  overlap: 30

execution:
  output_dir: "outputs"
  prompt_payloads_dir: "prompt_payloads"
  resume: true

aggregation:
  output_dir: "aggregated_tables"
  db_file: "aggregated_data.db"
  csv: true
  sqlite: true
  multi_table: true
```

### 2. Run Default Pipeline
Run the full extraction and aggregation pipeline in a single command based on `config.yaml`:
```bash
python3 run_pipeline.py
```

### 3. Run Pipeline with CLI Schema Overrides
You can execute targeted pipeline runs for different schema directories and domain mappings:

- **Core Master Schema**:
  ```bash
  python3 run_pipeline.py \
    --schema-path schemas/core_master \
    --domain-mapping-file schemas/core_domain_mapping.py \
    --output-dir outputs/core_master \
    --agg-dir aggregated_tables/core_master
  ```

- **Summary Master Schema**:
  ```bash
  python3 run_pipeline.py \
    --schema-path schemas/summary_master \
    --domain-mapping-file schemas/summary_domain_mapping.py \
    --output-dir outputs/summary_master \
    --agg-dir aggregated_tables/summary_master
  ```

### 4. Run Pipeline Components Independently

- **Extraction Phase Only**:
  ```bash
  python3 src/extract_workflow.py \
    --schema-path schemas/core_master \
    --domain-mapping-file schemas/core_domain_mapping.py
  ```

- **Aggregation Phase Only**:
  ```bash
  python3 src/aggregate_outputs.py \
    --input-dir outputs/core_master \
    --output-dir aggregated_tables/core_master
  ```

---

## Output Formats

1. **Extracted JSON Files** (`outputs/`):
   - Structured JSON records named `<source_id>__<module_name>__<schema_name>.json`.
2. **Prompt Payloads** (`prompt_payloads/`):
   - Human-readable text inspection files capturing system prompts, domain instructions, constraints, and raw text sent to the LLM.
3. **Relational CSV Tables** (`aggregated_tables/*.csv`):
   - Normalized relational tables with `source_id`, `source_file`, `parent_item_index`, and `item_index` primary/foreign keys.
4. **SQLite Database** (`aggregated_tables/aggregated_data.db`):
   - Fully loaded SQLite database with automatically inferred column types for downstream SQL querying.

---

## License
MIT License. See LICENSE file for details.
