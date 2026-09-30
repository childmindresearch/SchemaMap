# Dynamic Structured Extraction & Aggregation Framework (SchemaMap)

A domain-agnostic, schema-driven framework designed to extract structured JSON data from unstructured clinical, medical, or general text documents using **`pydantic-ai`** and local Large Language Models (via **Ollama**).

The framework dynamically loads Pydantic schemas at runtime (supporting both single `.py` files and multi-file schema directories with mixed types), applies **AST/DAG field-reflection** to isolate top-level root models, segments long documents using sliding-window word chunking, dispatches concurrent LLM extraction requests, consolidates chunk extractions using an in-memory **Map-Reduce merging engine**, and normalizes JSON outputs into relational **CSV tables and SQLite databases**.

---

## Architecture Overview

```
[ Unstructured Text / Parquet ]
               │
               ▼
   1. Multi-File DAG Schema Classifier (src/multifile_schema_loader.py)
      - Resolves cross-file symbol namespaces
      - Builds DAG of field references
      - Filters top-level root models from embedded child models
               │
               ▼
   2. Text Segmentation & Word Chunking (Sliding Window)
               │
               ▼
   3. Map Phase: Concurrent LLM Extractions (pydantic-ai + Ollama)
               │
               ▼
   4. Reduce Phase: Schema-Agnostic Instance Merger (src/schema_loader.py)
               │
               ▼
   5. Namespaced JSON Storage (outputs/<doc_id>/<doc_id>__<module>__<class>.json)
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

- **Multi-File & Mixed-Type Support (Default Workflow)**: Automatically process directories containing multiple `.py` schema files containing a mixture of root container models, standalone models, and embedded child component models.
- **AST / DAG Hierarchy Classification**: Uses type-hint reflection across `BaseModel.model_fields` to construct a Directed Acyclic Graph (DAG) of schema relationships. Automatically isolates top-level root models from embedded child models, reducing LLM API calls by **70%–85%**.
- **Collision-Free Namespace Isolation**: Formats output filenames using a double-underscore convention (`<doc_id>__<module_name>__<class_name>.json`), preventing class name collisions when multiple `.py` files define identically named schemas (e.g., `GeneratedModule`).
- **Sliding-Window Word Chunking**: Segments large documents into custom word counts with overlapping boundaries to preserve context across splits.
- **Schema-Agnostic Map-Reduce Engine**:
  - **Lists**: Extended and deduplicated across chunk outputs.
  - **Long Text**: Concatenated with delimiters.
  - **Numbers**: Calculated averages (e.g., scores/ratings).
  - **Enums/Booleans**: Merged via majority-vote consensus.
- **Relational Data Aggregator**: Automatically flattens nested JSON extractions into normalized parent-child relational tables linked by `source_id`, `parent_item_index`, and `item_index` foreign keys.
- **Resumable Execution**: Automatically skips already processed documents to prevent redundant API calls.

---

## Multi-File Schema Processing & DAG Classification

When processing complex schema folders (such as `schemas/test_schemas/multifile/V1` through `V4`), directories contain multiple `.py` files with mixed model types:

| Schema Category | Description | DAG Classifier Action |
| :--- | :--- | :--- |
| **Top-Level Root Models** | Master summary or container classes (e.g. `TreatmentHistory`, `DiagnosticAssessment`, `ClinicalSummary`). | **Selected for LLM Extraction** |
| **Standalone Entity Models** | Independent un-nested entity models (e.g. `AcademicAccommodation`). | **Selected for LLM Extraction** |
| **Embedded Child Models** | Component models referenced as field types inside another model (e.g. `AnxietySymptom`, `MedicationStatus`). | **Automatically Suppressed** (extracted naturally inside top-level parent models) |

---

## Directory Structure

```
SchemaMap/
├── run_pipeline.py                    # Unified end-to-end pipeline runner (Default workflow)
├── config.yaml                        # Centralized workflow & aggregation configuration
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation
├── inputs/                            # Input text or Parquet files to scan
│   ├── sample_clinical_report.txt
│   └── sample_contact.txt
├── schemas/                           # Python schema definitions (Pydantic models)
│   └── test_schemas/
│       ├── root_container_schema_example.py
│       ├── multi_class_schema_example.py
│       └── multifile/                 # Evaluation Multi-File Directories
│           ├── V1/                    # 16 schema files (104 classes -> 25 top-level roots)
│           ├── V2/                    # 14 schema files (95 classes -> 23 top-level roots)
│           ├── V3/                    # 8 schema files (61 classes -> 16 top-level roots)
│           └── V4/                    # 4 schema files (37 classes -> 6 top-level roots)
└── src/                               # Core Python engine modules
    ├── run_pipeline.py                # Main CLI pipeline wrapper
    ├── extract_multifile_workflow.py  # Default multi-file DAG extraction engine
    ├── extract_workflow.py            # Extraction workflow wrapper
    ├── multifile_schema_loader.py     # Cross-file namespace & DAG reflection classifier
    ├── schema_loader.py               # Map-Reduce merging algorithm & chunking helper
    └── aggregate_outputs.py           # Relational table aggregator & exporter
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
  file: "schemas/test_schemas/multifile/V1"

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

aggregation:
  output_dir: "aggregated_tables"
  db_file: "aggregated_data.db"
  csv: true
  sqlite: true
```

### 2. Run Default End-to-End Pipeline
Run the full extraction and aggregation pipeline in a single command based on `config.yaml`:
```bash
python3 run_pipeline.py
```

### 3. Target Specific Directories (CLI Overrides)
You can run targeted pipeline executions for specific schema directories and dedicated output locations:

```bash
# Run V1 pipeline execution:
python3 run_pipeline.py --schema-dir schemas/test_schemas/multifile/V1 --output-dir outputs/V1 --agg-dir aggregated_tables/V1

# Run V4 pipeline execution:
python3 run_pipeline.py --schema-dir schemas/test_schemas/multifile/V4 --output-dir outputs/V4 --agg-dir aggregated_tables/V4
```

### 4. Run Components Independently (Optional)

- **Extraction Phase Only**:
  ```bash
  python3 src/extract_multifile_workflow.py --schema-dir schemas/test_schemas/multifile/V1
  ```

- **Aggregation Phase Only**:
  ```bash
  python3 src/aggregate_outputs.py --input-dir outputs/V1 --output-dir aggregated_tables/V1
  ```

Outputs are saved in `aggregated_tables/`:
- `aggregated_tables/*.csv` (Normalized relational tables)
- `aggregated_tables/aggregated_data.db` (SQLite relational database)

---

## Defining Custom Schemas

Create a Python file or directory of Python files in `schemas/`:

```python
# schemas/my_custom_schema.py
from typing import List, Optional
from pydantic import BaseModel, Field

class PatientDemographics(BaseModel):
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    age: Optional[int] = Field(None, description="Age in years")

class Diagnosis(BaseModel):
    condition: str = Field(..., description="Diagnosed condition name")

class ClinicalSummaryReport(BaseModel):
    patient: Optional[PatientDemographics] = None
    diagnoses: List[Diagnosis] = []
```

Point `config.yaml` or CLI arguments to your schema file or directory:
```yaml
schema:
  file: "schemas/my_custom_schema.py"
```

---

## License
MIT License. See LICENSE file for details.
