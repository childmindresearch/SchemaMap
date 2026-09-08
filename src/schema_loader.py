"""
Schema Loader Module for dynamic structured text extraction.
Supports loading Pydantic schemas dynamically from Python (.py) files.
Also provides a generic Map-Reduce Pydantic model merging algorithm.
"""

import importlib.util
import inspect
import json
import os
import sys
from collections import Counter
from datetime import date, datetime
from enum import Enum
from pydantic import BaseModel


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to safely serialize dates, datetimes, enums,
    and Pydantic BaseModel instances.
    """
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def resolve_schema_files(schema_path: str) -> list[str]:
    """
    Resolves a schema path into a list of Python (.py) file paths.
    Supports single file paths or directory paths containing multiple schema files.
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file or directory not found: '{schema_path}'.")

    if os.path.isfile(schema_path):
        return [schema_path]
    elif os.path.isdir(schema_path):
        py_files = sorted([
            os.path.join(schema_path, f)
            for f in os.listdir(schema_path)
            if f.endswith(".py") and not f.startswith("__")
        ])
        if not py_files:
            raise FileNotFoundError(f"No Python schema files (.py) found in directory '{schema_path}'.")
        return py_files
    else:
        raise FileNotFoundError(f"Invalid schema path: '{schema_path}'.")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    """
    Chunks a text string into word-based segments of a dynamic length.
    """
    if chunk_size <= 0:
        return [text]
        
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size
        
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
            
    return chunks


def load_schema_from_python(filepath: str, class_name: str) -> type[BaseModel]:
    """Loads a Pydantic model class dynamically from a Python file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Python schema file not found: {filepath}")
        
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module {module_name} at {filepath}")
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, class_name):
        raise AttributeError(f"Module '{module_name}' has no class named '{class_name}'")
        
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseModel):
        raise TypeError(f"Class '{class_name}' is not a subclass of Pydantic BaseModel")
        
    # Resolve any postponed type annotations (e.g. from __future__ import annotations)
    cls.model_rebuild()
    
    return cls


def load_all_schemas_from_python(filepath: str) -> list[type[BaseModel]]:
    """Loads all Pydantic model classes dynamically in definition order from a Python file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Python schema file not found: {filepath}")
        
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module {module_name} at {filepath}")
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    schemas = []
    for obj in vars(module).values():
        if inspect.isclass(obj):
            if obj.__module__ == module_name and issubclass(obj, BaseModel) and obj is not BaseModel:
                try:
                    obj.model_rebuild()
                except Exception:
                    pass
                schemas.append(obj)
                
    return schemas


def find_root_schema(filepath: str) -> type[BaseModel]:
    """
    Finds and returns the primary root container model in a schema file.
    Prefers classes ending in 'Summary' or 'Report', defaulting to the final defined BaseModel.
    """
    schemas = load_all_schemas_from_python(filepath)
    if not schemas:
        raise ValueError(f"No Pydantic BaseModel schemas found in {filepath}")
        
    # 1. Match convention names ending in 'Summary' or 'Report'
    for cls in reversed(schemas):
        if cls.__name__.endswith("Summary") or cls.__name__.endswith("Report"):
            return cls
            
    # 2. Fallback to the last defined schema class in the file
    return schemas[-1]


def merge_pydantic_instances(schema_cls: type[BaseModel], instances: list[BaseModel]) -> BaseModel:
    """
    Consolidates multiple Pydantic model instances into a single merged instance.
    - Lists of primitives or nested sub-models are extended and deduplicated.
    - Dicts are updated and merged.
    - Nested sub-models are recursively merged.
    - Long string fields are concatenated.
    - Short strings, numbers, and booleans use majority-vote consolidation.
    """
    if not instances:
        raise ValueError("No instances to merge.")
    if len(instances) == 1:
        return instances[0]
        
    merged_data = {}
    for field_name, field_info in schema_cls.model_fields.items():
        # Gather all non-null values for this field across instances
        values = [getattr(inst, field_name) for inst in instances if getattr(inst, field_name) is not None]
        if not values:
            merged_data[field_name] = None
            continue
            
        first_val = values[0]
        
        if isinstance(first_val, list):
            merged_list = []
            seen = set()
            for val in values:
                for item in val:
                    # Deduplicate based on serialized JSON or string representation
                    if isinstance(item, (BaseModel, dict)):
                        item_repr = json.dumps(item, cls=SafeJSONEncoder, sort_keys=True)
                    else:
                        item_repr = str(item).strip().lower()
                        
                    if item_repr not in seen:
                        merged_list.append(item)
                        seen.add(item_repr)
            merged_data[field_name] = merged_list
            
        elif isinstance(first_val, dict):
            merged_dict = {}
            for val in values:
                merged_dict.update(val)
            merged_data[field_name] = merged_dict

        elif isinstance(first_val, BaseModel):
            # Recursively merge nested Pydantic sub-models
            sub_instances = [v for v in values if isinstance(v, BaseModel)]
            merged_data[field_name] = merge_pydantic_instances(type(first_val), sub_instances)
            
        elif isinstance(first_val, str) and any(len(v) > 30 for v in values):
            # Concatenate long text fields with delimiter
            non_empty_strings = [v for v in values if v.strip()]
            unique_strings = []
            for s in non_empty_strings:
                if s not in unique_strings:
                    unique_strings.append(s)
            merged_data[field_name] = " | ".join(unique_strings)
            
        elif isinstance(first_val, float):
            # Float averages are useful (e.g. ratings)
            floats = [float(v) for v in values]
            merged_data[field_name] = sum(floats) / len(floats)
            
        else:
            # Short strings, ints, booleans: majority vote
            try:
                counter = Counter(values)
                merged_data[field_name] = counter.most_common(1)[0][0]
            except TypeError:
                # Fallback if unhashable
                merged_data[field_name] = first_val
                
    return schema_cls(**merged_data)
