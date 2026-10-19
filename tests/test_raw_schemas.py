"""
Unit tests for modernized schemas in schemas/raw
Verifies single root containers, shallow nesting, (str, Enum) inheritance,
and empty instantiation compatibility.
"""

from enum import Enum
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schemas import load_schemas, get_top_level_schemas, get_child_schemas
from src.prompts import build_triage_prompt, build_extraction_prompt
from src.merger import merge_pydantic_instances


class TestRawSchemas(unittest.TestCase):
    def setUp(self):
        self.raw_path = "schemas/raw"
        self.schemas = load_schemas(self.raw_path)
        self.top_schemas = get_top_level_schemas(self.schemas)
        self.child_schemas = get_child_schemas(self.schemas)

    def test_top_level_count_and_names(self):
        """Ensures exactly 5 unified root schemas exist with no duplicate names."""
        top_names = [m.class_name for m in self.top_schemas]
        self.assertEqual(len(top_names), 5)
        self.assertEqual(len(set(top_names)), 5, "Top-level class names must be unique")

        expected_roots = {
            "InterpersonalRelationalAssessment",
            "PsychiatricMedicationManagement",
            "ClinicalPsychopathologyAssessment",
            "FamilyDynamicsAssessment",
            "AffectiveDysregulationAssessment",
        }
        self.assertEqual(set(top_names), expected_roots)

    def test_empty_instantiation_all_models(self):
        """Verifies strict null compliance: every model must instantiate empty without ValidationError."""
        for m in self.schemas:
            try:
                inst = m.schema_cls()
                self.assertIsNotNone(inst)
            except Exception as e:
                self.fail(f"Schema {m.qualified_name} failed empty instantiation: {e}")

    def test_enum_subclass_str(self):
        """Verifies that all enums across schemas inherit from (str, Enum) for JSON/SQLite compatibility."""
        for m in self.schemas:
            mod = sys.modules.get(m.module_name)
            if not mod:
                continue
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, Enum) and obj is not Enum:
                    self.assertTrue(
                        issubclass(obj, str),
                        f"Enum {m.module_name}.{name} must inherit from (str, Enum)"
                    )

    def test_merger_compatibility(self):
        """Verifies that merge_pydantic_instances succeeds across chunk instances for every root schema."""
        for m in self.top_schemas:
            inst1 = m.schema_cls()
            inst2 = m.schema_cls()
            merged = merge_pydantic_instances(m.schema_cls, [inst1, inst2])
            self.assertIsInstance(merged, m.schema_cls)

    def test_triage_prompt_uniqueness(self):
        """Verifies triage prompt contains unique schema entries."""
        prompt = build_triage_prompt(self.top_schemas)
        for m in self.top_schemas:
            self.assertIn(m.class_name, prompt)
            self.assertEqual(prompt.count(f"- **`{m.class_name}`**"), 1)


if __name__ == "__main__":
    unittest.main()
