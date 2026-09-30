"""
Multi-File Schema Loader Module

Provides robust dynamic loading, cross-file namespace isolation, and AST/DAG
dependency classification for multi-file Pydantic schema directories.
"""

import importlib.util
import inspect
import os
import sys
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Type
from pydantic import BaseModel


@dataclass
class SchemaMeta:
    """Metadata container for a loaded Pydantic schema class."""
    module_name: str
    class_name: str
    qualified_name: str
    schema_cls: Type[BaseModel]
    is_top_level: bool
    is_child: bool
    filepath: str


def extract_referenced_models(annotation: Any) -> Set[Type[BaseModel]]:
    """
    Recursively inspects a type annotation (unwrap Optional, List, Union, Dict, etc.)
    and returns all referenced Pydantic BaseModel subclasses.
    """
    referenced = set()

    if annotation is None:
        return referenced

    # Direct subclass of BaseModel
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel) and annotation is not BaseModel:
        referenced.add(annotation)
        return referenced

    # Handle Generic Alias types (List[T], Optional[T], Union[T1, T2], Dict[K, V], list[T], etc.)
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if args:
        for arg in args:
            referenced.update(extract_referenced_models(arg))

    return referenced


def resolve_multifile_schema_files(schema_path: str) -> List[str]:
    """Resolves a directory or file path into a sorted list of Python schema files."""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema path not found: '{schema_path}'.")

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


def load_multifile_schemas(schema_dir_or_file: str) -> List[SchemaMeta]:
    """
    Dynamically loads all Pydantic schemas from a multi-file directory or file.
    Applies cross-file namespace resolution and DAG field-reflection to classify
    top-level root models vs child embedded models.
    """
    py_files = resolve_multifile_schema_files(schema_dir_or_file)
    
    # 1. First Pass: Import all modules and collect classes
    module_classes: Dict[str, List[Tuple[str, Type[BaseModel], str]]] = {}
    all_loaded_classes: Set[Type[BaseModel]] = set()

    for filepath in py_files:
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Warning: Error executing module '{module_name}' ({filepath}): {e}", file=sys.stderr)
            continue

        # Extract Pydantic BaseModel subclasses
        file_classes = []
        for obj in vars(module).values():
            if inspect.isclass(obj):
                if obj.__module__ == module_name and issubclass(obj, BaseModel) and obj is not BaseModel:
                    try:
                        obj.model_rebuild()
                    except Exception:
                        pass
                    class_name = obj.__name__
                    qualified_name = f"{module_name}_{class_name}"
                    file_classes.append((class_name, obj, filepath))
                    all_loaded_classes.add(obj)

        module_classes[module_name] = file_classes

    # 2. Second Pass: Deferred type rebuild across global namespace
    for module_name, classes in module_classes.items():
        for class_name, cls, _ in classes:
            try:
                cls.model_rebuild()
            except Exception:
                pass

    # 3. Third Pass: DAG Introspection to classify Top-Level vs Child models
    # Collect all child classes referenced by any field in any model within the directory
    child_classes_global: Set[Type[BaseModel]] = set()

    for module_name, classes in module_classes.items():
        for class_name, cls, _ in classes:
            if hasattr(cls, "model_fields"):
                for field_name, field_info in cls.model_fields.items():
                    referenced = extract_referenced_models(field_info.annotation)
                    for ref_cls in referenced:
                        if ref_cls != cls:  # Exclude self-references
                            child_classes_global.add(ref_cls)

    # 4. Construct final SchemaMeta records
    schema_meta_list: List[SchemaMeta] = []

    for module_name, classes in module_classes.items():
        for class_name, cls, filepath in classes:
            is_child = cls in child_classes_global
            is_top_level = not is_child
            qualified_name = f"{module_name}_{class_name}"

            schema_meta_list.append(SchemaMeta(
                module_name=module_name,
                class_name=class_name,
                qualified_name=qualified_name,
                schema_cls=cls,
                is_top_level=is_top_level,
                is_child=is_child,
                filepath=filepath
            ))

    return schema_meta_list


def get_top_level_schemas(schema_meta_list: List[SchemaMeta]) -> List[SchemaMeta]:
    """Filters metadata list to return only top-level root schemas."""
    return [meta for meta in schema_meta_list if meta.is_top_level]


def get_child_schemas(schema_meta_list: List[SchemaMeta]) -> List[SchemaMeta]:
    """Filters metadata list to return only child embedded schemas."""
    return [meta for meta in schema_meta_list if meta.is_child]
