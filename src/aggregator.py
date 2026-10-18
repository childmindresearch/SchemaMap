"""
Dynamic Relational Data Aggregator

Normalizes nested extraction JSON outputs into linked relational CSV tables
and a SQLite database, or denormalized flat tables.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
from .config import AggregationConfig, InputConfig, PipelineConfig, resolve_path

METADATA_KEYS = {"source_id", "source_text", "source_file", "parent_item_index", "item_index"}


def has_extracted_content(val: Any) -> bool:
    """
    Recursively inspects a data structure or table row to determine if it
    contains at least one non-null, non-empty extracted fact.
    Ignores structural metadata columns (source_id, source_file, item_index, parent_item_index).
    """
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip() != ""
    if isinstance(val, (list, tuple, set)):
        return any(has_extracted_content(item) for item in val)
    if isinstance(val, dict):
        content_items = {k: v for k, v in val.items() if k not in METADATA_KEYS}
        if not content_items:
            return False
        return any(has_extracted_content(v) for v in content_items.values())
    return True


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parses a filename using either the double-underscore convention:
      '<source_id>__<module_name>__<schema_name>.json' or '<source_id>__<qualified_name>.json'
    or the single-underscore fallback convention:
      '<source_id>_<schema_name>.json'
    """
    basename = Path(filename).stem
    if "__" in basename:
        parts = basename.split("__")
        if len(parts) >= 3:
            return parts[0], parts[-1]
        elif len(parts) == 2:
            return parts[0], parts[1]
    if "_" not in basename:
        return "unknown", basename
    source_id, schema_name = basename.rsplit("_", 1)
    return source_id, schema_name


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """Recursively flattens a nested dictionary."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat[new_key] = v
    return flat


def process_json_data(data: Any) -> List[Dict[str, Any]]:
    """Processes JSON content for single-table mode into a list of flat dictionaries (rows)."""
    if isinstance(data, list):
        rows = []
        for item in data:
            if isinstance(item, dict):
                rows.append(flatten_dict(item))
            else:
                rows.append({"value": item})
        return rows

    elif isinstance(data, dict):
        flat_parent = flatten_dict(data)
        list_fields = {}
        scalar_fields = {}
        for k, v in flat_parent.items():
            if isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v):
                list_fields[k] = v
            else:
                scalar_fields[k] = v

        if list_fields:
            rows = []
            for field_name, items in list_fields.items():
                for item in items:
                    flat_item = flatten_dict(item)
                    row = dict(scalar_fields)
                    row.update(flat_item)
                    rows.append(row)
            return rows
        else:
            return [flat_parent]
    else:
        return [{"value": data}]


def extract_relational_tables(
    data: Any,
    root_schema_name: str,
    source_id: str,
    source_file: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Recursively extracts normalized relational multi-table data from a JSON payload.
    Returns mapping of table_name -> list of row dictionaries with foreign keys.
    """
    tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def process_object(obj: Any, parent_table: str, path: str = "", foreign_keys: Optional[Dict[str, Any]] = None):
        if foreign_keys is None:
            foreign_keys = {}

        scalar_fields = dict(foreign_keys)
        scalar_fields.setdefault("source_id", source_id)
        scalar_fields.setdefault("source_file", source_file)

        if not isinstance(obj, dict):
            row = dict(scalar_fields)
            row["value"] = obj
            tables[parent_table].append(row)
            return

        for key, val in obj.items():
            field_path = f"{path}_{key}" if path else key

            if isinstance(val, list):
                child_table_name = f"{root_schema_name}_{field_path}"
                for idx, item in enumerate(val):
                    child_fk = dict(foreign_keys)
                    child_fk["source_id"] = source_id
                    child_fk["source_file"] = source_file
                    if "item_index" in child_fk:
                        child_fk["parent_item_index"] = child_fk.pop("item_index")
                    child_fk["item_index"] = idx

                    if isinstance(item, dict):
                        process_object(item, child_table_name, path=field_path, foreign_keys=child_fk)
                    else:
                        row = dict(child_fk)
                        row["value"] = item
                        tables[child_table_name].append(row)

            elif isinstance(val, dict):
                has_list = any(isinstance(v, list) for v in val.values())
                if has_list:
                    for sub_k, sub_v in val.items():
                        sub_path = f"{field_path}_{sub_k}"
                        if isinstance(sub_v, list):
                            child_table_name = f"{root_schema_name}_{sub_path}"
                            for idx, item in enumerate(sub_v):
                                child_fk = dict(foreign_keys)
                                child_fk["source_id"] = source_id
                                child_fk["source_file"] = source_file
                                if "item_index" in child_fk:
                                    child_fk["parent_item_index"] = child_fk.pop("item_index")
                                child_fk["item_index"] = idx

                                if isinstance(item, dict):
                                    process_object(item, child_table_name, path=sub_path, foreign_keys=child_fk)
                                else:
                                    row = dict(child_fk)
                                    row["value"] = item
                                    tables[child_table_name].append(row)
                        elif isinstance(sub_v, dict):
                            sub_flat = flatten_dict(sub_v, parent_key=sub_path)
                            scalar_fields.update(sub_flat)
                        else:
                            scalar_fields[sub_path] = sub_v
                else:
                    flat_sub = flatten_dict(val, parent_key=field_path)
                    scalar_fields.update(flat_sub)
            else:
                scalar_fields[field_path] = val

        tables[parent_table].append(scalar_fields)

    if isinstance(data, list):
        for idx, item in enumerate(data):
            fk = {"source_id": source_id, "source_file": source_file, "item_index": idx}
            if isinstance(item, dict):
                process_object(item, root_schema_name, foreign_keys=fk)
            else:
                row = dict(fk)
                row["value"] = item
                tables[root_schema_name].append(row)
    else:
        process_object(data, root_schema_name)

    return tables


def serialize_value(val: Any, for_csv: bool = False) -> Any:
    """Serializes values to appropriate representation for CSV or SQLite."""
    if val is None:
        return "" if for_csv else None
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, bool):
        return str(val) if for_csv else val
    return str(val) if for_csv else val


def infer_sqlite_type(values: List[Any]) -> str:
    """Infers SQLite data type from Python values."""
    non_null_vals = [v for v in values if v is not None]
    if not non_null_vals:
        return "TEXT"
    if all(isinstance(v, bool) for v in non_null_vals):
        return "INTEGER"
    if all(isinstance(v, int) for v in non_null_vals):
        return "INTEGER"
    if all(isinstance(v, (int, float)) for v in non_null_vals):
        return "REAL"
    return "TEXT"


def normalize_record_keys(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalizes keys case-insensitively across record dicts."""
    if not records:
        return records

    canonical_map = {}
    metadata_keys = {"source_id", "source_text", "source_file", "parent_item_index", "item_index"}

    for r in records:
        for k in r.keys():
            k_lower = k.lower()
            if k_lower not in canonical_map:
                if k_lower in metadata_keys:
                    canonical_map[k_lower] = k_lower
                else:
                    canonical_map[k_lower] = k

    normalized = []
    for r in records:
        norm_r = {}
        for k, v in r.items():
            norm_r[canonical_map[k.lower()]] = v
        normalized.append(norm_r)
    return normalized


def discover_json_files(input_dir: str) -> List[Tuple[str, str]]:
    """Recursively scans input_dir for output JSON files, skipping chunks and manifests."""
    results: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(input_dir):
        if "chunks" in root.split(os.sep):
            continue
        for f in sorted(files):
            if f.endswith(".json") and not f.startswith("."):
                results.append((os.path.join(root, f), f))
    return results


def load_source_text_map(input_config: Optional[InputConfig] = None) -> Dict[str, str]:
    """Loads a mapping of doc_id -> raw document text based on InputConfig or config.yaml."""
    cfg_input = input_config
    if cfg_input is None:
        if os.path.exists("config.yaml"):
            try:
                from .config import load_config
                cfg = load_config("config.yaml")
                cfg_input = cfg.input
            except Exception:
                pass
    if not cfg_input or not cfg_input.path:
        return {}

    in_path = resolve_path(cfg_input.path)
    if not os.path.exists(in_path):
        return {}

    try:
        from .chunker import load_input_documents
        from .tracker import sanitize_doc_id
        docs = load_input_documents(
            input_dir=in_path,
            input_format=cfg_input.format,
            id_col=cfg_input.id_column,
            text_col=cfg_input.text_column
        )
        text_map: Dict[str, str] = {}
        for doc in docs:
            raw_id = str(doc.id)
            clean_id = raw_id.strip()
            sanitized = sanitize_doc_id(raw_id)
            text_map[raw_id] = doc.text
            text_map[clean_id] = doc.text
            text_map[sanitized] = doc.text
        return text_map
    except Exception as e:
        print(f"Warning: Could not load source text for wide table: {e}", file=sys.stderr)
        return {}


def build_consolidated_wide_table(
    json_files: List[Tuple[str, str]],
    drop_empty_records: bool = True,
    doc_text_map: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Constructs a single consolidated wide table across all extracted JSON files
    where exactly 1 row represents 1 document (source_id).

    Specialized 1-to-Many Relationship Handling:
    - Lists of scalars (e.g. diagnoses, symptoms):
        Stores full JSON array in '<Schema>__<Field>'
        Stores integer count in '<Schema>__<Field>__count'
    - Lists of child objects (e.g. core_relationships, medications):
        Stores full JSON array of child objects in '<Schema>__<Field>'
        Stores integer count in '<Schema>__<Field>__count'
        Explodes each child attribute across items into '<Schema>__<Field>__<attr>'
    - Multi-record schemas (multiple instances per document):
        Stores integer count in '<Schema>__count'
        Stores raw record list in '<Schema>__records'
        Aggregates individual fields across records into '<Schema>__<Field>'
    - Disambiguates schema class names across different modules to prevent collisions.
    """
    cls_modules: Dict[str, Set[str]] = defaultdict(set)
    parsed_files = []
    for filepath, filename in json_files:
        basename = Path(filename).stem
        if "__" in basename:
            parts = basename.split("__")
            doc_id = parts[0]
            schema_name = parts[-1]
            mod_name = parts[1] if len(parts) >= 3 else ""
        elif "_" in basename:
            doc_id, schema_name = basename.rsplit("_", 1)
            mod_name = ""
        else:
            doc_id, schema_name = "unknown", basename
            mod_name = ""
        cls_modules[schema_name].add(mod_name)
        parsed_files.append((filepath, filename, doc_id, mod_name, schema_name))

    colliding_classes = {cls for cls, mods in cls_modules.items() if len(mods) > 1}

    doc_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for filepath, filename, doc_id, mod_name, schema_name in parsed_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if drop_empty_records and not has_extracted_content(data):
            continue

        schema_key = f"{mod_name}__{schema_name}" if (schema_name in colliding_classes and mod_name) else schema_name
        if schema_key not in doc_data[doc_id]:
            doc_data[doc_id][schema_key] = data
        else:
            existing = doc_data[doc_id][schema_key]
            if isinstance(existing, list) and isinstance(data, list):
                doc_data[doc_id][schema_key] = existing + data
            elif isinstance(existing, list):
                doc_data[doc_id][schema_key] = existing + [data]
            elif isinstance(data, list):
                doc_data[doc_id][schema_key] = [existing] + data
            elif isinstance(existing, dict) and isinstance(data, dict):
                merged = dict(existing)
                for k, v in data.items():
                    if k not in merged or merged[k] is None:
                        merged[k] = v
                    elif v is not None:
                        if isinstance(merged[k], list) and isinstance(v, list):
                            merged[k] = merged[k] + v
                        elif isinstance(merged[k], list):
                            merged[k] = merged[k] + [v]
                        elif isinstance(v, list):
                            merged[k] = [merged[k]] + v
                        elif merged[k] != v:
                            merged[k] = [merged[k], v]
                doc_data[doc_id][schema_key] = merged

    wide_rows: List[Dict[str, Any]] = []
    for doc_id, schemas in sorted(doc_data.items()):
        row: Dict[str, Any] = {"source_id": doc_id}
        if doc_text_map:
            text_val = (
                doc_text_map.get(doc_id)
                or doc_text_map.get(doc_id.strip())
                or doc_text_map.get(doc_id.replace(" ", "_"))
            )
            row["source_text"] = text_val
        for schema_key, data in schemas.items():
            def process_val(val: Any, prefix: str):
                if val is None:
                    return
                elif isinstance(val, (int, float, bool, str)):
                    if isinstance(val, str) and not val.strip():
                        return
                    row[prefix] = val
                elif isinstance(val, list):
                    if not val:
                        return
                    row[f"{prefix}__count"] = len(val)
                    if all(isinstance(x, (int, float, bool, str)) for x in val):
                        row[prefix] = json.dumps(val, ensure_ascii=False)
                    elif all(isinstance(x, dict) for x in val):
                        row[prefix] = json.dumps(val, ensure_ascii=False)
                        child_attrs: Set[str] = set()
                        for item in val:
                            child_attrs.update(item.keys())
                        for attr in sorted(child_attrs):
                            child_vals = [item[attr] for item in val if attr in item and item[attr] is not None]
                            if child_vals:
                                row[f"{prefix}__{attr}"] = json.dumps(child_vals, ensure_ascii=False)
                    else:
                        row[prefix] = json.dumps(val, ensure_ascii=False)
                elif isinstance(val, dict):
                    for k, v in val.items():
                        process_val(v, f"{prefix}__{k}")

            if isinstance(data, list):
                if not data:
                    continue
                row[f"{schema_key}__count"] = len(data)
                row[f"{schema_key}__records"] = json.dumps(data, ensure_ascii=False)
                child_attrs: Set[str] = set()
                for item in data:
                    if isinstance(item, dict):
                        child_attrs.update(item.keys())
                for attr in sorted(child_attrs):
                    vals = [item[attr] for item in data if isinstance(item, dict) and attr in item and item[attr] is not None]
                    if vals:
                        row[f"{schema_key}__{attr}"] = json.dumps(vals, ensure_ascii=False)
            elif isinstance(data, dict):
                for k, v in data.items():
                    process_val(v, f"{schema_key}__{k}")

        if not drop_empty_records or len(row) > 1:
            wide_rows.append(row)

    if drop_empty_records and wide_rows:
        all_cols: Set[str] = set()
        for r in wide_rows:
            all_cols.update(r.keys())
        populated_cols = {col for col in all_cols if col in ("source_id", "source_text") or any(r.get(col) is not None for r in wide_rows)}
        wide_rows = [{k: v for k, v in r.items() if k in populated_cols} for r in wide_rows]

    return wide_rows



class DataAggregator:
    """Relational table aggregator and exporter."""

    def __init__(self, config: AggregationConfig, input_config: Optional[InputConfig] = None):
        self.config = config
        self.input_config = input_config

    def aggregate(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        doc_text_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Runs end-to-end table aggregation and exports CSV and SQLite files."""
        in_dir = resolve_path(input_dir or self.config.input_dir or "outputs")
        out_dir = resolve_path(output_dir or self.config.output_dir)
        multi_table = self.config.multi_table

        if not os.path.exists(in_dir):
            print(f"Error: Input directory '{in_dir}' does not exist.", file=sys.stderr)
            return {}

        json_files = discover_json_files(in_dir)
        if not json_files:
            print(f"Warning: No JSON files found in input directory '{in_dir}'.", file=sys.stderr)
            return {}

        os.makedirs(out_dir, exist_ok=True)
        if self.config.csv:
            for old_csv in Path(out_dir).glob("*.csv"):
                try:
                    old_csv.unlink()
                except Exception:
                    pass
        all_tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        mode_label = "Multi-Table Relational" if multi_table else "Single-Table Denormalized"
        print(f"Aggregating {len(json_files)} JSON file(s) across '{in_dir}' [{mode_label}]...")

        skipped_empty_files = 0
        for filepath, filename in json_files:
            source_id, schema_name = parse_filename(filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading file '{filename}': {e}", file=sys.stderr)
                continue

            if self.config.drop_empty_records and not has_extracted_content(data):
                skipped_empty_files += 1
                continue

            if multi_table:
                file_tables = extract_relational_tables(data, schema_name, source_id, filename)
                for tbl_name, rows in file_tables.items():
                    if self.config.drop_empty_records:
                        rows = [r for r in rows if has_extracted_content(r)]
                    all_tables[tbl_name].extend(rows)
            else:
                rows = process_json_data(data)
                for r in rows:
                    r["source_id"] = source_id
                    r["source_file"] = filename
                    if not self.config.drop_empty_records or has_extracted_content(r):
                        all_tables[schema_name].append(r)

        if skipped_empty_files > 0:
            print(f"  [Filter] Discarded {skipped_empty_files} completely empty extract file(s).")

        if self.config.drop_empty_records:
            for tbl_name in list(all_tables.keys()):
                all_tables[tbl_name] = [r for r in all_tables[tbl_name] if has_extracted_content(r)]

        if self.config.drop_empty_tables:
            initial_count = len(all_tables)
            all_tables = {tbl_name: rows for tbl_name, rows in all_tables.items() if len(rows) > 0}
            pruned_count = initial_count - len(all_tables)
            if pruned_count > 0:
                print(f"  [Filter] Pruned {pruned_count} table(s) that contained 0 non-empty rows.")

        # Additional Output Option: Consolidated Wide Table
        if self.config.consolidated_wide_table and json_files:
            text_map_to_use = None
            if self.config.include_source_text:
                if doc_text_map is not None:
                    text_map_to_use = doc_text_map
                else:
                    text_map_to_use = load_source_text_map(self.input_config)

            wide_records = build_consolidated_wide_table(
                json_files,
                drop_empty_records=self.config.drop_empty_records,
                doc_text_map=text_map_to_use
            )
            if wide_records:
                all_tables[self.config.wide_table_name] = wide_records
                print(f"  [Wide Table] Generated consolidated wide table '{self.config.wide_table_name}': {len(wide_records)} row(s), {len(wide_records[0]) if wide_records else 0} column(s).")

        print(f"\nGrouped data into {len(all_tables)} distinct table(s):")
        for tbl_name, records in all_tables.items():
            print(f"  - {tbl_name}: {len(records)} row(s)")

        db_path = os.path.join(out_dir, self.config.db_file)
        conn = None
        if self.config.sqlite:
            if os.path.exists(db_path):
                os.remove(db_path)
            conn = sqlite3.connect(db_path)
            print(f"\nWriting SQLite database: {db_path}")

        try:
            for table_name, records in all_tables.items():
                if not records:
                    continue

                records = normalize_record_keys(records)
                all_keys = set()
                for r in records:
                    all_keys.update(r.keys())

                metadata_cols = [c for c in ["source_id", "source_text", "source_file", "parent_item_index", "item_index"] if c in all_keys]
                other_cols = sorted([k for k in all_keys if k not in metadata_cols])
                headers = metadata_cols + other_cols

                if not headers:
                    continue

                # CSV Export
                if self.config.csv:
                    csv_path = os.path.join(out_dir, f"{table_name}.csv")
                    try:
                        with open(csv_path, "w", encoding="utf-8", newline="") as csv_f:
                            writer = csv.DictWriter(csv_f, fieldnames=headers)
                            writer.writeheader()
                            for r in records:
                                writer.writerow({k: serialize_value(r.get(k), for_csv=True) for k in headers})
                        print(f"  - Saved CSV: {csv_path}")
                    except Exception as e:
                        print(f"Error writing CSV for '{table_name}': {e}", file=sys.stderr)

                # SQLite Export
                if conn is not None:
                    col_defs = ", ".join([f'"{col}" {infer_sqlite_type([r.get(col) for r in records])}' for col in headers])
                    db_table_name = "".join([c if c.isalnum() or c == "_" else "_" for c in table_name])
                    try:
                        cursor = conn.cursor()
                        cursor.execute(f'CREATE TABLE IF NOT EXISTS "{db_table_name}" ({col_defs});')
                        quoted_headers = [f'"{h}"' for h in headers]
                        headers_str = ", ".join(quoted_headers)
                        placeholders = ", ".join(["?" for _ in headers])
                        insert_sql = f'INSERT INTO "{db_table_name}" ({headers_str}) VALUES ({placeholders});'
                        rows_to_insert = [
                            [serialize_value(r.get(col), for_csv=False) for col in headers]
                            for r in records
                        ]
                        cursor.executemany(insert_sql, rows_to_insert)
                        conn.commit()
                        print(f"  - Database table '{db_table_name}' loaded successfully.")
                    except Exception as e:
                        print(f"Error writing SQLite table '{db_table_name}': {e}", file=sys.stderr)
        finally:
            if conn is not None:
                conn.close()

        print("\nAggregation complete!")
        return all_tables


def run_aggregation(
    config_file: str = "config.yaml",
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    db_file: Optional[str] = None,
    generate_csv: Optional[bool] = None,
    generate_sqlite: Optional[bool] = None,
    multi_table: Optional[bool] = None,
    drop_empty_records: Optional[bool] = None,
    drop_empty_tables: Optional[bool] = None,
    consolidated_wide_table: Optional[bool] = None,
    wide_table_name: Optional[str] = None,
    include_source_text: Optional[bool] = None
) -> None:
    """Entry point for programmatic invocation."""
    cfg = PipelineConfig.from_yaml(config_file)
    agg_cfg = cfg.aggregation
    in_dir = resolve_path(input_dir or agg_cfg.input_dir or cfg.execution.output_dir)
    out_dir = resolve_path(output_dir or agg_cfg.output_dir)

    agg_cfg.input_dir = in_dir
    agg_cfg.output_dir = out_dir
    if db_file:
        agg_cfg.db_file = db_file
    if generate_csv is not None:
        agg_cfg.csv = generate_csv
    if generate_sqlite is not None:
        agg_cfg.sqlite = generate_sqlite
    if multi_table is not None:
        agg_cfg.multi_table = multi_table
    if drop_empty_records is not None:
        agg_cfg.drop_empty_records = drop_empty_records
    if drop_empty_tables is not None:
        agg_cfg.drop_empty_tables = drop_empty_tables
    if consolidated_wide_table is not None:
        agg_cfg.consolidated_wide_table = consolidated_wide_table
    if wide_table_name is not None:
        agg_cfg.wide_table_name = wide_table_name
    if include_source_text is not None:
        agg_cfg.include_source_text = include_source_text

    aggregator = DataAggregator(agg_cfg, input_config=cfg.input)
    aggregator.aggregate(input_dir=in_dir, output_dir=out_dir)
