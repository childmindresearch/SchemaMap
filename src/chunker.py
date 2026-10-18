"""
Text Chunking & Document Ingestion Engine

Handles sliding-window word chunking and document ingestion from text files (.txt)
and Apache Parquet datasets (.parquet / .pq).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    """Represents an ingested text document."""
    id: str
    text: str
    source: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def chunk_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[str]:
    """
    Chunks a text string into word-based segments of dynamic length with overlap.
    Preserves all word tokens and formatting.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        return [text]

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: List[str] = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks


def load_input_documents(
    input_dir: Optional[str] = None,
    input_format: str = "txt",
    file_override: Optional[str] = None,
    id_col: str = "identifier",
    text_col: str = "anonymized",
    path: Optional[str] = None,
) -> List[Document]:
    """
    Loads input documents from text files (.txt) or Parquet files (.parquet / .pq).
    Returns a list of Document objects with verified ID and text content.
    """
    docs: List[Document] = []
    target_path = file_override or path or input_dir or ""
    input_fmt = input_format.lower()

    if target_path and (target_path.endswith(".parquet") or target_path.endswith(".pq")):
        input_fmt = "parquet"

    if input_fmt == "parquet":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas and pyarrow are required for Parquet input. Install with: pip install pandas pyarrow")

        p_files: List[str] = []
        if os.path.isfile(target_path):
            p_files = [target_path]
        elif os.path.isdir(target_path):
            for root, _, files in os.walk(target_path):
                for f in sorted(files):
                    if (f.endswith(".parquet") or f.endswith(".pq")) and not f.startswith("."):
                        p_files.append(os.path.join(root, f))

        for p_file in p_files:
            df = pd.read_parquet(p_file)
            for idx, row in df.iterrows():
                raw_id = row.get(id_col, f"row_{idx}")
                text_val = row.get(text_col, "")
                if isinstance(raw_id, float) and raw_id.is_integer():
                    doc_id = str(int(raw_id))
                else:
                    doc_id = str(raw_id).strip()
                docs.append(Document(
                    id=doc_id,
                    text=str(text_val).strip() if pd.notna(text_val) else "",
                    source=f"{p_file} [row {idx}]"
                ))
    else:
        txt_files: List[str] = []
        if os.path.isfile(target_path):
            txt_files = [target_path]
        elif os.path.isdir(target_path):
            for root, _, files in os.walk(target_path):
                for f in sorted(files):
                    if f.endswith(".txt") and not f.startswith("."):
                        txt_files.append(os.path.join(root, f))

        for filepath in txt_files:
            base_name = Path(filepath).stem
            with open(filepath, "r", encoding="utf-8") as f:
                text_val = f.read().strip()
            docs.append(Document(
                id=base_name,
                text=text_val,
                source=filepath
            ))

    return docs
