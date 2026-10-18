"""
Unit tests for SchemaMap core components:
- Configuration Manager (config.py)
- Chunker & Document Loader (chunker.py)
- Schema Loader, DAG & Container Builder (schemas.py)
- Verbatim Instance Merger (merger.py)
- Resume Tracker (tracker.py)
- Relational Aggregator (aggregator.py)
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PipelineConfig, resolve_path
from src.chunker import chunk_text, load_input_documents, Document
from src.schemas import (
    load_schemas,
    get_top_level_schemas,
    get_child_schemas,
    extract_referenced_models,
    format_schema_field_instructions,
)
from src.extractor import build_extraction_prompt, build_triage_prompt, TriageResult
from src.merger import merge_pydantic_instances, SafeJSONEncoder
from src.tracker import ResumeTracker, sanitize_doc_id
from src.aggregator import (
    parse_filename,
    flatten_dict,
    process_json_data,
    extract_relational_tables,
    has_extracted_content,
    build_consolidated_wide_table,
    DataAggregator,
)


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.input.format, "txt")
        self.assertEqual(cfg.chunking.size, 300)
        self.assertEqual(cfg.chunking.overlap, 30)
        self.assertTrue(cfg.execution.resume)
        self.assertTrue(cfg.aggregation.multi_table)
        self.assertEqual(cfg.model.timeout, 300.0)
        self.assertEqual(cfg.model.retries, 3)
        self.assertEqual(cfg.aggregation.input_dir, cfg.execution.output_dir)

    def test_load_yaml(self):
        from src.config import load_config
        cfg = load_config("config.yaml")
        self.assertIsNotNone(cfg.schema.path)
        self.assertTrue(os.path.exists(resolve_path(cfg.schema.path)))
        self.assertEqual(cfg.schema.extract_mode, "top_level")
        self.assertEqual(cfg.aggregation.output_dir, "outputs/aggregated_tables")


class TestChunker(unittest.TestCase):
    def test_load_input_documents_alias(self):
        docs = load_input_documents(path="inputs", input_format="txt")
        self.assertTrue(len(docs) > 0)

    def test_chunk_text_basic(self):
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)
        self.assertTrue(len(chunks) >= 2)
        self.assertIn("word1", chunks[0])
        # Verify overlap: chunk 2 should contain the end of chunk 1
        self.assertIn("word4", chunks[0])
        self.assertIn("word4", chunks[1])

    def test_chunk_text_small(self):
        text = "hello world"
        chunks = chunk_text(text, chunk_size=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "hello world")

    def test_load_input_documents(self):
        docs = load_input_documents("inputs", input_format="txt")
        self.assertTrue(len(docs) > 0)
        self.assertTrue(all(isinstance(d, Document) for d in docs))
        self.assertTrue(all(len(d.text) > 0 for d in docs))


class TestSchemas(unittest.TestCase):
    def test_load_test_schemas(self):
        schema_path = "schemas/test_schemas"
        schemas = load_schemas(schema_path)
        self.assertTrue(len(schemas) >= 2)
        names = [s.class_name for s in schemas]
        self.assertIn("PatientIntakeSummary", names)
        self.assertIn("DischargeSummaryReport", names)

    def test_schema_field_instructions_and_prompt(self):
        schema_path = "schemas/test_schemas"
        schemas = load_schemas(schema_path)
        sample = next(s for s in schemas if s.class_name == "PatientIntakeSummary")

        # Test field guide generation
        field_guide = format_schema_field_instructions(sample.schema_cls)
        self.assertIn("patient", field_guide)
        self.assertIn("reason_for_referral", field_guide)

        # Test optimized extraction prompt generation
        prompt = build_extraction_prompt(sample.schema_cls, domain_info={"domain_title": "Intake"})
        self.assertIn("PatientIntakeSummary", prompt)
        self.assertIn("Verbatim Fidelity", prompt)
        self.assertIn("Strict Null Rule", prompt)
        self.assertIn("Intake", prompt)

    def test_triage_prompt_and_result(self):
        schema_path = "schemas/test_schemas"
        schemas = load_schemas(schema_path)
        triage_prompt = build_triage_prompt(schemas)
        self.assertIn("PatientIntakeSummary", triage_prompt)
        self.assertIn("DischargeSummaryReport", triage_prompt)
        self.assertIn("Candidate Schemas", triage_prompt)

        # Test TriageResult parsing
        res = TriageResult.model_validate({"relevant_schemas": ["PatientIntakeSummary"]})
        self.assertEqual(res.relevant_schemas, ["PatientIntakeSummary"])

    def test_prompts_module_direct_imports(self):
        from src.prompts import (
            EXTRACTION_RULES,
            TRIAGE_RULES,
            format_schema_field_instructions as format_fn,
            build_triage_prompt as triage_fn,
            build_extraction_prompt as extract_fn,
        )
        self.assertIn("Verbatim Fidelity", EXTRACTION_RULES)
        self.assertIn("Triage Rules", TRIAGE_RULES)
        self.assertTrue(callable(format_fn))
        self.assertTrue(callable(triage_fn))
        self.assertTrue(callable(extract_fn))


class TestMerger(unittest.TestCase):
    def test_merge_pydantic_instances(self):
        class DemoItem(BaseModel):
            name: str
            tags: list[str] = Field(default_factory=list)
            score: float = 0.0

        inst1 = DemoItem(name="Patient A", tags=["anxiety", "depression"], score=12.5)
        inst2 = DemoItem(name="Patient A", tags=["insomnia", "anxiety"], score=12.5)

        merged = merge_pydantic_instances(DemoItem, [inst1, inst2])
        self.assertEqual(merged.name, "Patient A")
        # Exact list deduplication:
        self.assertEqual(merged.tags, ["anxiety", "depression", "insomnia"])
        # No numerical averaging:
        self.assertEqual(merged.score, 12.5)


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resume_tracker(self):
        tracker = ResumeTracker(self.temp_dir)
        doc_id = "patient_123"

        schemas = load_schemas("schemas/test_schemas")
        self.assertFalse(tracker.is_document_completed(doc_id, schemas))

        pending = tracker.get_pending_schemas(doc_id, schemas)
        self.assertEqual(len(pending), len(schemas))

        # Simulate writing one schema file
        out_path = tracker.get_schema_output_path(doc_id, schemas[0])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"test": 1}), encoding="utf-8")

        # Now pending should be decreased by 1
        pending_after = tracker.get_pending_schemas(doc_id, schemas)
        self.assertEqual(len(pending_after), len(schemas) - 1)
        # Document still not completed because remaining schemas are pending!
        self.assertFalse(tracker.is_document_completed(doc_id, schemas))

        # Mark completed explicitly
        tracker.mark_completed(doc_id, schemas)
        self.assertTrue(tracker.is_document_completed(doc_id, schemas))

    def test_chunk_cache_and_pending_schemas(self):
        tracker = ResumeTracker(self.temp_dir)
        doc_id = "patient_456"
        schemas = load_schemas("schemas/test_schemas")

        doc_dir = tracker.get_document_dir(doc_id)
        chunk_dir = doc_dir / "chunks" / schemas[0].qualified_name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_file = chunk_dir / "patient_456_chunk_1.json"
        chunk_file.write_text(json.dumps({"test_chunk": 1}), encoding="utf-8")

        self.assertTrue(chunk_file.exists())
        self.assertEqual(len(tracker.get_pending_schemas(doc_id, schemas)), len(schemas))


class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_parse_filename(self):
        src, schema = parse_filename("doc1__module__PatientRecord.json")
        self.assertEqual(src, "doc1")
        self.assertEqual(schema, "PatientRecord")

    def test_relational_extraction_and_aggregation(self):
        # Create mock extracted JSON
        doc_dir = Path(self.temp_dir) / "doc1"
        doc_dir.mkdir(parents=True, exist_ok=True)
        json_file = doc_dir / "doc1__01_test__Intake.json"

        data = {
            "patient_name": "Jane Doe",
            "diagnoses": [
                {"code": "F41.1", "description": "Generalized anxiety disorder"},
                {"code": "F32.9", "description": "Major depressive disorder"}
            ]
        }
        json_file.write_text(json.dumps(data), encoding="utf-8")

        from src.config import AggregationConfig
        agg_cfg = AggregationConfig(
            input_dir=self.temp_dir,
            output_dir=self.out_dir,
            db_file="test.db",
            csv=True,
            sqlite=True,
            multi_table=True
        )
        aggregator = DataAggregator(agg_cfg)
        tables = aggregator.aggregate()

        self.assertIn("Intake", tables)
        self.assertIn("Intake_diagnoses", tables)

        # Check CSV generated
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "Intake.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "Intake_diagnoses.csv")))

        # Check SQLite generated
        db_path = os.path.join(self.out_dir, "test.db")
        self.assertTrue(os.path.exists(db_path))

    def test_has_extracted_content(self):
        # Empty cases
        self.assertFalse(has_extracted_content(None))
        self.assertFalse(has_extracted_content(""))
        self.assertFalse(has_extracted_content("   "))
        self.assertFalse(has_extracted_content([]))
        self.assertFalse(has_extracted_content({}))
        self.assertFalse(has_extracted_content({"source_id": "doc1", "source_file": "f.json"}))
        self.assertFalse(has_extracted_content({"source_id": "doc1", "field_a": None, "field_b": ""}))
        self.assertFalse(has_extracted_content([None, "", []]))

        # Non-empty cases
        self.assertTrue(has_extracted_content("valid text"))
        self.assertTrue(has_extracted_content(0))
        self.assertTrue(has_extracted_content(False))
        self.assertTrue(has_extracted_content({"source_id": "doc1", "field_a": "value"}))
        self.assertTrue(has_extracted_content([None, "found"]))

    def test_drop_empty_records_and_tables(self):
        doc_dir = Path(self.temp_dir) / "doc2"
        doc_dir.mkdir(parents=True, exist_ok=True)

        # File 1: completely null extract
        empty_json = doc_dir / "doc2__01__EmptyModel.json"
        empty_json.write_text(json.dumps({"col1": None, "col2": None, "col3": ""}), encoding="utf-8")

        # File 2: valid content extract
        valid_json = doc_dir / "doc2__01__ValidModel.json"
        valid_json.write_text(json.dumps({"col1": "Data present"}), encoding="utf-8")

        from src.config import AggregationConfig
        agg_cfg = AggregationConfig(
            input_dir=self.temp_dir,
            output_dir=self.out_dir,
            db_file="test2.db",
            csv=True,
            sqlite=True,
            multi_table=True,
            drop_empty_records=True,
            drop_empty_tables=True
        )
        aggregator = DataAggregator(agg_cfg)
        tables = aggregator.aggregate()

        # ValidModel must exist
        self.assertIn("ValidModel", tables)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "ValidModel.csv")))

        # EmptyModel must be safely pruned
        self.assertNotIn("EmptyModel", tables)
        self.assertFalse(os.path.exists(os.path.join(self.out_dir, "EmptyModel.csv")))

    def test_consolidated_wide_table_and_one_to_many(self):
        doc_dir = Path(self.temp_dir) / "patient_007"
        doc_dir.mkdir(parents=True, exist_ok=True)

        # File 1: Clinical Symptom Profile with list of scalars (1-to-many)
        f1 = doc_dir / "patient_007__01__ClinicalProfile.json"
        f1.write_text(json.dumps({
            "primary_diagnoses": ["Major Depression", "Generalized Anxiety"],
            "suicidality_risk": "moderate"
        }), encoding="utf-8")

        # File 2: Treatment Plan with list of child dicts (1-to-many objects)
        f2 = doc_dir / "patient_007__02__TreatmentPlan.json"
        f2.write_text(json.dumps({
            "medications": [
                {"name": "sertraline", "dose": "50mg"},
                {"name": "bupropion", "dose": "150mg"}
            ],
            "assessment": {"gad7_score": 14}
        }), encoding="utf-8")

        from src.config import AggregationConfig
        agg_cfg = AggregationConfig(
            input_dir=self.temp_dir,
            output_dir=self.out_dir,
            db_file="test_wide.db",
            csv=True,
            sqlite=True,
            multi_table=True,
            consolidated_wide_table=True,
            wide_table_name="consolidated_wide"
        )
        aggregator = DataAggregator(agg_cfg)
        tables = aggregator.aggregate(doc_text_map={"patient_007": "Patient presents with severe anxiety and depression."})

        self.assertIn("consolidated_wide", tables)
        wide_rows = tables["consolidated_wide"]
        self.assertEqual(len(wide_rows), 1)

        row = wide_rows[0]
        self.assertEqual(row["source_id"], "patient_007")
        self.assertEqual(row["source_text"], "Patient presents with severe anxiety and depression.")

        # Scalar field check
        self.assertEqual(row["ClinicalProfile__suicidality_risk"], "moderate")
        self.assertEqual(row["TreatmentPlan__assessment__gad7_score"], 14)

        # 1-to-many list of scalars: check JSON string and count column
        self.assertEqual(row["ClinicalProfile__primary_diagnoses__count"], 2)
        diagnoses = json.loads(row["ClinicalProfile__primary_diagnoses"])
        self.assertIn("Major Depression", diagnoses)
        self.assertIn("Generalized Anxiety", diagnoses)

        # 1-to-many list of objects: check count, full JSON, and exploded attribute columns
        self.assertEqual(row["TreatmentPlan__medications__count"], 2)
        med_names = json.loads(row["TreatmentPlan__medications__name"])
        self.assertEqual(med_names, ["sertraline", "bupropion"])
        med_doses = json.loads(row["TreatmentPlan__medications__dose"])
        self.assertEqual(med_doses, ["50mg", "150mg"])

        # Check CSV and SQLite exports for consolidated_wide including source_text
        csv_path = os.path.join(self.out_dir, "consolidated_wide.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            self.assertIn("source_text", header_line)

        import sqlite3
        conn = sqlite3.connect(os.path.join(self.out_dir, "test_wide.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT source_id, source_text, ClinicalProfile__suicidality_risk FROM consolidated_wide")
        res = cursor.fetchone()
        self.assertEqual(res[0], "patient_007")
        self.assertEqual(res[1], "Patient presents with severe anxiety and depression.")
        self.assertEqual(res[2], "moderate")
        conn.close()

        # Test with include_source_text = False
        agg_cfg.include_source_text = False
        aggregator_no_text = DataAggregator(agg_cfg)
        tables_no_text = aggregator_no_text.aggregate(doc_text_map={"patient_007": "Patient presents with severe anxiety and depression."})
        wide_row_no_text = tables_no_text["consolidated_wide"][0]
        self.assertNotIn("source_text", wide_row_no_text)

    def test_cli_argument_options(self):
        import subprocess
        result = subprocess.run(
            ["python3", "run_pipeline.py", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--wide-table", result.stdout)
        self.assertIn("--source-text", result.stdout)
        self.assertIn("--drop-empty", result.stdout)
        self.assertIn("--drop-empty-tables", result.stdout)


if __name__ == "__main__":
    unittest.main()

