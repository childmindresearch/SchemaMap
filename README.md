# Dynamic Structured Extraction & Aggregation Framework

A domain-agnostic, schema-driven framework designed to extract structured JSON data from unstructured clinical, medical, or general text documents using **`pydantic-ai`** and local Large Language Models (via **Ollama**).

The framework dynamically loads Pydantic schemas at runtime, segments long documents using sliding-window word chunking, dispatches concurrent LLM extraction requests, consolidates chunk extractions using a schema-agnostic **Map-Reduce merging engine**, and normalizes JSON outputs into relational **CSV tables and SQLite databases**.

---

## Architecture Overview

```
[ Unstructured Text / Parquet ]
               │
               ▼
   1. Dynamic Schema Loader (src/schema_loader.py)
               │
               ▼
   2. Text Segmentation & Word Chunking (Sliding Window)
               │
               ▼
   3. Map Phase: Concurrent LLM Extractions (pydantic-ai + Ollama)
               │
               ▼
   4. Reduce Phase: Schema-Agnostic Instance Merger
               │
               ▼
   5. Output JSON Storage (outputs/)
               │
               ▼
   6. Relational Table Aggregator (src/aggregate_outputs.py)
               │
      ┌────────┴────────┐
      ▼                 ▼
  CSV Tables     SQLite Database (.db)
```

---

## Features

- **Dynamic Schema Reflection**: Load any Pydantic model (`BaseModel`) from external `.py` files without modifying framework code.
- **Root Container vs Multi-Class Extraction**: Target a single summary container model for high throughput or extract child classes in sequence.
- **Sliding-Window Word Chunking**: Segment large documents into custom word counts with overlapping boundaries to preserve context across splits.
- **Schema-Agnostic Map-Reduce Engine**:
  - **Lists**: Extended and deduplicated across chunk outputs.
  - **Long Text**: Concatenated with delimiters.
  - **Numbers**: Calculated averages (e.g., scores/ratings).
  - **Enums/Booleans**: Merged via majority-vote consensus.
- **Relational Data Aggregator**: Automatically flattens nested JSON extractions into normalized parent-child relational tables linked by `source_id` and `item_index` foreign keys.
- **YAML Configuration System**: Manage dataset paths, schema selections, LLM model IDs, chunk sizes, and output locations via `config.yaml`.
- **Resumable Execution**: Automatically skips already processed documents to prevent redundant API calls.

---

> **Map-Reduce**: SchemaMap applies an in-memory Map-Reduce pattern to process long documents—extracting schema instances concurrently per text chunk (Map) and consolidating them into a single Pydantic object via field-level reduction rules (Reduce).

---

## Root Container vs. Multi-Class Extraction Guide

The framework supports two extraction paradigms to balance LLM API volume, speed, and extraction precision:

| Feature / Metric | Root Container Mode (`use_root_schema: true`) | Multi-Class Mode (`use_root_schema: false`) |
| :--- | :--- | :--- |
| **How it Works** | Targets **1 top-level container model** that nests child sub-models. | Loops through **all defined schema classes** in the `.py` file sequentially. |
| **API Requests per Chunk** | **1 request per text chunk** | **N requests per text chunk** (where N = number of classes in file) |
| **Throughput & Speed** | 🚀 **High Speed** (up to 10x-20x faster) | 🐢 **Slower** (high API volume) |
| **Schema Structure** | Requires a master model (e.g. ending in `Summary` or `Report`). | Works with any collection of independent Pydantic classes. |
| **Best For** | Full-document extractions, production runs, linked entities. | Targeted extractions, isolated model debugging. |

---

### 1. Root Container Mode (`use_root_schema: true`) — *Default & Recommended*

In Root Container mode, the framework auto-detects a master wrapper class (e.g. `SampleReportSummary` or `ClinicalRecord`) in your schema file that references child models as fields:

```python
# Master Root Container Class
class SampleReportSummary(BaseModel):
    patient: Optional[PatientInfo] = None
    diagnoses: List[ClinicalDiagnosis] = []
    recommended_interventions: List[str] = []
```

* **Execution Behavior**: For a document split into 3 chunks, the LLM is called **3 times** in total. In each call, the LLM populates the complete nested tree at once.
* **When to Use**:
  * Production data processing where speed and API efficiency are essential.
  * When child entities are contextualized together (e.g., patient info + diagnoses + interventions).
  * Your schema file has a top-level container class.

---

### 2. Multi-Class Mode (`use_root_schema: false`)

In Multi-Class mode, the framework ignores top-level container wrappers and runs a separate extraction pass for **every individual Pydantic class** defined in the `.py` file.

```python
# Standalone Schema Classes evaluated in separate LLM calls
class PatientInfo(BaseModel): ...
class ClinicalDiagnosis(BaseModel): ...
class MedicationRecord(BaseModel): ...
```

* **Execution Behavior**: If your schema file defines 20 classes and a document is split into 3 chunks, the framework dispatches **60 API requests** (20 classes × 3 chunks).
* **When to Use**:
  * You need to extract only one specific entity type using `--schema-class MedicationRecord`.
  * The schema models are large or complex, and asking the LLM to extract everything in one prompt causes context overflow or hallucination.
  * Inspecting or fine-tuning prompts for individual entity classes.

---

### How to Switch Between Modes

- **In `config.yaml`**:
  ```yaml
  schema:
    file: "schemas/test_schemas/root_container_schema_example.py"
    use_root_schema: true   # Set to true for Root Container, or false for Multi-Class
  ```

- **Via Command-Line**:
  ```bash
  # Scenario 1: Root Container Mode (Single Master Summary Model):
  python3 src/extract_workflow.py --schema-file schemas/test_schemas/root_container_schema_example.py --use-root-schema

  # Scenario 2: Multi-Class Mode (Standalone Entity Models):
  python3 src/extract_workflow.py --schema-file schemas/test_schemas/multi_class_schema_example.py --no-use-root-schema
  ```

### Multi-File / Directory Schema Loading

You can pass a **directory path** (e.g. `schemas/test_schemas/multi_file_example`) to `--schema-file` or `schema.file` in `config.yaml`. The engine will scan all `.py` files in that folder and run them in a **single pass**:

* **Root Container Mode (`use_root_schema: true`)**: Finds and extracts the root container schema for *each* `.py` file in the directory.
* **Multi-Class Mode (`use_root_schema: false`)**: Extracts all schema classes defined across *all* `.py` files in the directory.

```bash
# Scenario 3: Process all schema files in a directory in a single pass:
python3 src/extract_workflow.py --schema-file schemas/test_schemas/multi_file_example
```

## Directory Structure

```
ExtractFeatures/
├── run_pipeline.py        # End-to-end extraction and aggregation pipeline runner
├── config.yaml            # Centralized workflow & aggregation configuration
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore specification
├── README.md              # Project documentation
├── inputs/                # Input text or Parquet files to scan
│   ├── sample_clinical_report.txt
│   └── sample_contact.txt
├── schemas/               # Python schema definitions (Pydantic models)
│   └── test_schemas/      # Test and example schema scenarios
│       ├── root_container_schema_example.py   # Scenario 1: Root Container
│       ├── multi_class_schema_example.py      # Scenario 2: Multi-Class
│       └── multi_file_example/                # Scenario 3: Multi-File Directory
│           ├── 01_patient_intake_schema.py
│           └── 02_discharge_summary_schema.py
└── src/                   # Core Python modules
    ├── extract_workflow.py    # Main asynchronous extraction engine CLI
    ├── aggregate_outputs.py   # Relational table aggregator & exporter CLI
    └── schema_loader.py       # Dynamic schema loader & Map-Reduce merger
```

---

## Installation & Setup

### 1. Prerequisites
- **Python**: 3.11+
- **Ollama**: Installed and running locally (`http://localhost:11434/v1`)
- **LLM Model**: Pulled target model in Ollama (e.g., `ollama pull qwen2.5:7b` or custom model)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/ExtractFeatures.git
cd ExtractFeatures

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart Guide

### 1. Configuration (`config.yaml`)
Customize your extraction parameters in `config.yaml`:
```yaml
input:
  path: "inputs"
  format: "txt"

schema:
  file: "schemas/test_schemas/root_container_schema_example.py"
  use_root_schema: true

model:
  name: "qwen2.5:7b"
  url: "http://localhost:11434/v1"
  num_async_calls: 4

chunking:
  size: 1800
  overlap: 30

execution:
  output_dir: "outputs"
  resume: true
```

### 2. Run End-to-End Pipeline
Run the full extraction and aggregation pipeline in a single command based on `config.yaml`:
```bash
python3 run_pipeline.py
```

### 3. Run Individual Components (Optional)
You can also run extraction or aggregation phases independently:

* **Extraction Phase Only**:
  ```bash
  python3 src/extract_workflow.py
  ```

* **Aggregation Phase Only**:
  ```bash
  python3 src/aggregate_outputs.py
  ```

Outputs will be saved to `aggregated_tables/`:
- `aggregated_tables/*.csv`
- `aggregated_tables/aggregated_data.db`

---

## Defining Custom Schemas

Create a Python file in `schemas/` defining Pydantic models:

```python
# schemas/my_custom_schema.py
from typing import List, Optional
from pydantic import BaseModel, Field

class PatientDemographics(BaseModel):
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    age: Optional[int] = Field(None, description="Age in years")

class Diagnosis(BaseModel):
    condition: str = Field(..., description="Diagnosed condition name")
    severity: Optional[str] = Field(None, description="Mild, moderate, or severe")

class ClinicalSummaryReport(BaseModel):
    patient: Optional[PatientDemographics] = None
    diagnoses: List[Diagnosis] = []
```

Point `config.yaml` to your new schema:
```yaml
schema:
  file: "schemas/my_custom_schema.py"
```

---

## License
MIT License. See LICENSE file for details.
# SchemaMap
