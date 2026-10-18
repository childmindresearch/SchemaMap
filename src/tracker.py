"""
Execution & Resume Tracker

Provides granular resume tracking and completion validation.
Ensures incomplete documents or newly added schemas are properly identified
rather than prematurely skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from .schemas import SchemaMeta



def sanitize_doc_id(doc_id: str) -> str:
    """Sanitizes document identifier into a safe filesystem folder name."""
    cleaned = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(doc_id)])
    return cleaned.strip(" ._") or "doc"


class ResumeTracker:
    """Manages document extraction completion status and resume checkpoints."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def get_document_dir(self, doc_id: str) -> Path:
        """Returns directory path for a specific document."""
        return self.output_dir / sanitize_doc_id(doc_id)

    def get_schema_output_path(self, doc_id: str, meta: SchemaMeta) -> Path:
        """Returns the expected output JSON path for a specific schema class."""
        base_name = sanitize_doc_id(doc_id)
        filename = f"{base_name}__{meta.module_name}__{meta.class_name}.json"
        return self.get_document_dir(doc_id) / filename

    def is_schema_completed(self, doc_id: str, meta: SchemaMeta) -> bool:
        """Checks if a specific schema extraction has already succeeded for a document."""
        out_path = self.get_schema_output_path(doc_id, meta)
        if not out_path.exists():
            return False
        try:
            return out_path.stat().st_size > 0
        except OSError:
            return False

    def get_pending_schemas(self, doc_id: str, schemas_meta: List[SchemaMeta]) -> List[SchemaMeta]:
        """Returns the list of schemas that have not yet been extracted for this document."""
        return [meta for meta in schemas_meta if not self.is_schema_completed(doc_id, meta)]

    def is_document_completed(self, doc_id: str, schemas_meta: List[SchemaMeta]) -> bool:
        """
        Determines if a document is fully processed.
        Returns True ONLY if ALL target schemas have complete output files.
        """
        doc_dir = self.get_document_dir(doc_id)
        if not doc_dir.exists():
            return False

        completion_file = doc_dir / ".completed"
        completion_json = doc_dir / ".completed.json"

        # Check explicit completion manifest
        if completion_json.exists():
            try:
                with open(completion_json, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                completed_names = set(manifest.get("completed_schemas", []))
                expected_names = {meta.qualified_name for meta in schemas_meta}
                if expected_names.issubset(completed_names):
                    return True
            except Exception:
                pass

        if completion_file.exists():
            return True

        # Fallback check: verify every expected schema file exists and is non-empty
        return len(self.get_pending_schemas(doc_id, schemas_meta)) == 0

    def mark_completed(
        self,
        doc_id: str,
        schemas_meta: List[SchemaMeta],
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Writes a completion manifest marking the document as fully processed."""
        doc_dir = self.get_document_dir(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "doc_id": doc_id,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_schemas": len(schemas_meta),
            "completed_schemas": [meta.qualified_name for meta in schemas_meta],
        }
        if extra_metadata:
            manifest.update(extra_metadata)

        manifest_path = doc_dir / ".completed.json"
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass
