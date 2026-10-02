#!/usr/bin/env python3
"""
Dynamic Data Aggregator

This script dynamically processes the JSON output files of the extraction workflow,
agnostic of filenames and schemas. It parses filenames according to the convention:
  <source_id>__<module_name>__<schema_name>.json
and aggregates the results into distinct relational tables (CSV and SQLite).
Each row includes document-level identifiers (source_id and source_file)
to enable easy relational joining.

Supports two output modes:
  - Single table (MULTI_TABLE = False): Flattens schema lists into a single denormalized table per schema.
  - Multi-table (MULTI_TABLE = True): Extracts a normalized relational structure where parent models
    and array/child entities are stored in distinct linked tables.
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import defaultdict
from typing import Optional

# Ensure script directory and project root are in sys.path for relative imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Default Fallback Settings
INPUT_DIR = "outputs"
OUTPUT_DIR = "aggregated_tables"
OUTPUT_CSV = True
OUTPUT_SQLITE = True
MULTI_TABLE = True


def resolve_config_path(config_filepath: str = "config.yaml") -> str:
    """Resolves config path relative to CWD or project root."""
    if os.path.exists(config_filepath):
        return config_filepath
    root_config = os.path.join(project_root, config_filepath)
    if os.path.exists(root_config):
        return root_config
    return config_filepath


def load_config_file(config_filepath: str = "config.yaml") -> dict:
    """Loads workflow configuration from a YAML file if it exists."""
    config_filepath = resolve_config_path(config_filepath)
    if not config_filepath or not os.path.exists(config_filepath):
        return {}
    try:
        import yaml
        with open(config_filepath, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def get_aggregation_config(config_filepath: str = "config.yaml") -> dict:
    """Extracts aggregation configuration parameters with fallback defaults."""
    cfg = load_config_file(config_filepath)
    agg_cfg = cfg.get("aggregation", {})
    exec_cfg = cfg.get("execution", {})

    input_dir = agg_cfg.get("input_dir") or exec_cfg.get("output_dir") or INPUT_DIR
    output_dir = agg_cfg.get("output_dir", OUTPUT_DIR)
    db_file = agg_cfg.get("db_file", "aggregated_data.db")
    generate_csv = agg_cfg.get("csv", OUTPUT_CSV)
    generate_sqlite = agg_cfg.get("sqlite", OUTPUT_SQLITE)
    multi_table = agg_cfg.get("multi_table", MULTI_TABLE)

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "db_file": db_file,
        "csv": generate_csv,
        "sqlite": generate_sqlite,
        "multi_table": multi_table
    }



def parse_filename(filename: str):
    """
    Parses a filename using either the double-underscore convention:
      '<source_id>__<module_name>__<schema_name>.json' or '<source_id>__<qualified_name>.json'
    or the single-underscore fallback convention:
      '<source_id>_<schema_name>.json'
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    if '__' in basename:
        parts = basename.split('__')
        if len(parts) >= 3:
            source_id = parts[0]
            schema_name = parts[-1]
            return source_id, schema_name
        elif len(parts) == 2:
            return parts[0], parts[1]
    if '_' not in basename:
        return "unknown", basename
    source_id, schema_name = basename.rsplit('_', 1)
    return source_id, schema_name



def process_json_data(data):
    """Processes JSON content for single-table mode and yields a list of flat dictionaries (rows)."""
    def flatten_dict(d, parent_key='', sep='_'):
        flat = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flat.update(flatten_dict(v, new_key, sep=sep))
            else:
                flat[new_key] = v
        return flat

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
                    for ik, iv in flat_item.items():
                        row[ik] = iv
                    rows.append(row)
            return rows
        else:
            return [flat_parent]
    else:
        return [{"value": data}]


def extract_relational_tables(data, root_schema_name, source_id, source_file):
    """
    Recursively extracts normalized relational multi-table data from a JSON payload.
    Returns a dictionary mapping table_name (str) -> list of row dicts.
    """
    tables = defaultdict(list)

    def flatten_dict(d, parent_key='', sep='_'):
        flat = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flat.update(flatten_dict(v, new_key, sep=sep))
            else:
                flat[new_key] = v
        return flat

    def process_object(obj, parent_table, path="", foreign_keys=None):
        if foreign_keys is None:
            foreign_keys = {}

        scalar_fields = dict(foreign_keys)
        if "source_id" not in scalar_fields:
            scalar_fields["source_id"] = source_id
        if "source_file" not in scalar_fields:
            scalar_fields["source_file"] = source_file

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
                            for fk, fv in sub_flat.items():
                                scalar_fields[fk] = fv
                        else:
                            scalar_fields[sub_path] = sub_v
                else:
                    flat_sub = flatten_dict(val, parent_key=field_path)
                    for fk, fv in flat_sub.items():
                        scalar_fields[fk] = fv
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


def serialize_value(val, for_csv=False):
    """Serializes values to appropriate representation for CSV or SQLite."""
    if val is None:
        return "" if for_csv else None
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, bool):
        return str(val) if for_csv else val
    return str(val) if for_csv else val


def infer_sqlite_type(values):
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


def normalize_record_keys(records):
    """Normalizes keys case-insensitively across record dicts."""
    if not records:
        return records

    canonical_map = {}
    metadata_keys = {"source_id", "source_file", "parent_item_index", "item_index"}
    
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


def discover_json_files(input_dir: str):
    """Recursively scans input_dir for output JSON files, skipping intermediate 'chunks' folders."""
    results = []
    for root, dirs, files in os.walk(input_dir):
        if "chunks" in root.split(os.sep):
            continue
        for f in sorted(files):
            if f.endswith(".json"):
                results.append((os.path.join(root, f), f))
    return results


def load_as_dataframes(input_dir=None, multi_table=None, config_file="config.yaml"):
    """Programmatic helper to load and return aggregated tables as a dictionary of Pandas DataFrames."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to use load_as_dataframes(). Install it with: pip install pandas")
        
    cfg_defaults = get_aggregation_config(config_file)
    if input_dir is None:
        input_dir = cfg_defaults["input_dir"]
    if multi_table is None:
        multi_table = cfg_defaults["multi_table"]

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
        
    json_files = discover_json_files(input_dir)
    all_table_records = defaultdict(list)
    
    for filepath, filename in json_files:
        source_id, schema_name = parse_filename(filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if multi_table:
            file_tables = extract_relational_tables(data, schema_name, source_id, filename)
            for tbl_name, rows in file_tables.items():
                all_table_records[tbl_name].extend(rows)
        else:
            rows = process_json_data(data)
            for r in rows:
                r["source_id"] = source_id
                r["source_file"] = filename
                all_table_records[schema_name].append(r)
            
    dfs = {}
    for table_name, records in all_table_records.items():
        if not records:
            continue
        records = normalize_record_keys(records)
        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        metadata_cols = [c for c in ["source_id", "source_file", "parent_item_index", "item_index"] if c in all_keys]
        other_cols = sorted([k for k in all_keys if k not in metadata_cols])
        headers = metadata_cols + other_cols
        
        processed_records = []
        for r in records:
            processed_r = {k: serialize_value(r.get(k), for_csv=False) for k in headers}
            processed_records.append(processed_r)
            
        dfs[table_name] = pd.DataFrame(processed_records, columns=headers)
        
    return dfs


def run_aggregation(
    config_file: str = "config.yaml",
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    db_file: Optional[str] = None,
    generate_csv: Optional[bool] = None,
    generate_sqlite: Optional[bool] = None,
    multi_table: Optional[bool] = None
):
    """Programmatic entry point for running data aggregation."""
    cfg_defaults = get_aggregation_config(config_file)

    input_dir = input_dir or cfg_defaults["input_dir"]
    output_dir = output_dir or cfg_defaults["output_dir"]
    db_file = db_file or cfg_defaults["db_file"]
    generate_csv = generate_csv if generate_csv is not None else cfg_defaults["csv"]
    generate_sqlite = generate_sqlite if generate_sqlite is not None else cfg_defaults["sqlite"]
    multi_table = multi_table if multi_table is not None else cfg_defaults["multi_table"]

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        return

    if not generate_csv and not generate_sqlite:
        print("Warning: Both CSV and SQLite outputs are disabled. Nothing to do.", file=sys.stderr)
        return

    os.makedirs(output_dir, exist_ok=True)

    json_files = discover_json_files(input_dir)
    if not json_files:
        print(f"Warning: No JSON files found in input directory '{input_dir}'", file=sys.stderr)
        return

    mode_str = "Multi-Table Relational" if multi_table else "Single-Table Denormalized"
    print(f"Found {len(json_files)} JSON file(s) across '{input_dir}'. Mode: {mode_str}. Processing...")

    all_tables = defaultdict(list)

    for filepath, filename in json_files:
        source_id, schema_name = parse_filename(filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading file '{filename}': {e}", file=sys.stderr)
            continue

        if multi_table:
            file_tables = extract_relational_tables(data, schema_name, source_id, filename)
            for tbl_name, rows in file_tables.items():
                all_tables[tbl_name].extend(rows)
        else:
            rows = process_json_data(data)
            for r in rows:
                r["source_id"] = source_id
                r["source_file"] = filename
                all_tables[schema_name].append(r)

    print(f"\nGrouped data into {len(all_tables)} distinct table(s):")
    for tbl_name, records in all_tables.items():
        print(f"  - {tbl_name}: {len(records)} row(s)")

    db_path = os.path.join(output_dir, db_file)
    conn = None
    if generate_sqlite:
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        print(f"\nWriting to SQLite database: {db_path}")

    for table_name, records in all_tables.items():
        records = normalize_record_keys(records)
        all_keys = set()
        for r in records:
            all_keys.update(r.keys())

        metadata_cols = [c for c in ["source_id", "source_file", "parent_item_index", "item_index"] if c in all_keys]
        other_cols = sorted([k for k in all_keys if k not in metadata_cols])
        headers = metadata_cols + other_cols

        if generate_csv:
            csv_filepath = os.path.join(output_dir, f"{table_name}.csv")
            try:
                with open(csv_filepath, "w", encoding="utf-8", newline="") as csv_f:
                    writer = csv.DictWriter(csv_f, fieldnames=headers)
                    writer.writeheader()
                    for r in records:
                        row_data = {k: serialize_value(r.get(k), for_csv=True) for k in headers}
                        writer.writerow(row_data)
                print(f"  - Saved CSV: {csv_filepath}")
            except Exception as e:
                print(f"Error writing CSV for table '{table_name}': {e}", file=sys.stderr)

        if conn is not None:
            col_types = []
            for col in headers:
                vals = [r.get(col) for r in records]
                ctype = infer_sqlite_type(vals)
                col_types.append(f'"{col}" {ctype}')

            col_defs = ", ".join(col_types)
            db_table_name = "".join([c if c.isalnum() or c == "_" else "_" for c in table_name])

            try:
                cursor = conn.cursor()
                cursor.execute(f'CREATE TABLE IF NOT EXISTS "{db_table_name}" ({col_defs});')

                quoted_headers = [f'"{h}"' for h in headers]
                headers_str = ", ".join(quoted_headers)
                placeholders = ", ".join(["?" for _ in headers])
                insert_sql = f'INSERT INTO "{db_table_name}" ({headers_str}) VALUES ({placeholders});'

                rows_to_insert = []
                for r in records:
                    rows_to_insert.append([serialize_value(r.get(col), for_csv=False) for col in headers])

                cursor.executemany(insert_sql, rows_to_insert)
                conn.commit()
                print(f"  - Database table '{db_table_name}' loaded successfully.")
            except Exception as e:
                print(f"Error writing SQLite table '{db_table_name}': {e}", file=sys.stderr)

    if conn is not None:
        conn.close()

    print("\nAggregation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate extraction outputs into relational CSV and SQLite tables."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing extracted JSON output files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save aggregated tables.")
    parser.add_argument("--db-file", type=str, default=None, help="SQLite database filename.")
    parser.add_argument("--csv", action="store_true", default=None, help="Generate CSV files.")
    parser.add_argument("--no-csv", action="store_false", dest="csv", help="Do not generate CSV files.")
    parser.add_argument("--sqlite", action="store_true", default=None, help="Generate SQLite database.")
    parser.add_argument("--no-sqlite", action="store_false", dest="sqlite", help="Do not generate SQLite database.")
    parser.add_argument("--multi-table", action="store_true", dest="multi_table", default=None, help="Store outputs in normalized multi-table relational structure.")
    parser.add_argument("--single-table", action="store_false", dest="multi_table", help="Store outputs in single table per base model.")

    args = parser.parse_args()
    run_aggregation(
        config_file=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        db_file=args.db_file,
        generate_csv=args.csv,
        generate_sqlite=args.sqlite,
        multi_table=args.multi_table
    )


if __name__ == "__main__":
    main()
