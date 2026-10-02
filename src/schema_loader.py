"""
Schema Loader & Processing Engine

Provides Pydantic schema loading, cross-file namespace isolation, AST/DAG
field-reflection classification (top-level vs child models), text chunking,
and field-level Pydantic model instance consolidation.
"""

import importlib.util
import inspect
import json
import os
import sys
import typing
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type
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


def chunk_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[str]:
    """Chunks a text string into word-based segments of dynamic length with overlap."""
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

    # Handle Generic Alias types (List[T], Optional[T], Union[T1, T2], Dict[K, V], etc.)
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if args:
        for arg in args:
            referenced.update(extract_referenced_models(arg))

    return referenced


def resolve_schema_files(schema_path: str) -> List[str]:
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


def load_schemas(schema_dir_or_file: str) -> List[SchemaMeta]:
    """
    Dynamically loads all Pydantic schemas from a schema file or directory.
    Applies cross-file namespace resolution and DAG field-reflection to classify
    top-level root models vs child embedded models.
    """
    py_files = resolve_schema_files(schema_dir_or_file)
    
    # Pass 1: Import modules and collect classes
    module_classes: Dict[str, List[Tuple[str, Type[BaseModel], str]]] = {}
    all_loaded_classes: Set[Type[BaseModel]] = set()

    for filepath in py_files:
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        
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

        file_classes = []
        for obj in vars(module).values():
            if inspect.isclass(obj):
                if obj.__module__ == module_name and issubclass(obj, BaseModel) and obj is not BaseModel:
                    try:
                        obj.model_rebuild()
                    except Exception:
                        pass
                    class_name = obj.__name__
                    file_classes.append((class_name, obj, filepath))
                    all_loaded_classes.add(obj)

        module_classes[module_name] = file_classes

    # Pass 2: Deferred type rebuild across global namespace
    for module_name, classes in module_classes.items():
        for class_name, cls, _ in classes:
            try:
                cls.model_rebuild()
            except Exception:
                pass

    # Pass 3: DAG Introspection to classify Top-Level vs Child models
    child_classes_global: Set[Type[BaseModel]] = set()

    for module_name, classes in module_classes.items():
        for class_name, cls, _ in classes:
            if hasattr(cls, "model_fields"):
                for field_name, field_info in cls.model_fields.items():
                    referenced = extract_referenced_models(field_info.annotation)
                    for ref_cls in referenced:
                        if ref_cls != cls:  # Exclude self-references
                            child_classes_global.add(ref_cls)

    # Pass 4: Construct SchemaMeta records
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


def merge_pydantic_instances(schema_cls: Type[BaseModel], instances: List[BaseModel]) -> BaseModel:
    """
    Consolidates Pydantic model instances across chunks while preserving EXACT raw extracted values.
    - Zero mathematical alteration or averaging.
    - Zero text mutation, lowercasing, or artificial string concatenation.
    - List items deduplicated based on exact raw JSON equality.
    """
    if not instances:
        raise ValueError("No instances to merge.")
    if len(instances) == 1:
        return instances[0]
        
    merged_data = {}
    for field_name, field_info in schema_cls.model_fields.items():
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
                    if isinstance(item, BaseModel):
                        item_repr = json.dumps(item.model_dump(mode="json"), cls=SafeJSONEncoder, sort_keys=True)
                    elif isinstance(item, dict):
                        item_repr = json.dumps(item, cls=SafeJSONEncoder, sort_keys=True)
                    else:
                        item_repr = str(item)
                        
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
            sub_instances = [v for v in values if isinstance(v, BaseModel)]
            merged_data[field_name] = merge_pydantic_instances(type(first_val), sub_instances)

        else:
            # Preserve exact raw extracted value from primary mention without alteration
            merged_data[field_name] = first_val
                
    return schema_cls(**merged_data)



def load_domain_mapping(mapping_filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically loads a domain mapping dictionary (e.g. DOMAIN_EXTRACTION_MAP)
    from a specified Python file.
    """
    if not mapping_filepath or not os.path.exists(mapping_filepath):
        return {}

    module_name = os.path.splitext(os.path.basename(mapping_filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, mapping_filepath)
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Warning: Could not execute domain mapping module '{mapping_filepath}': {e}", file=sys.stderr)
        return {}

    mapping = getattr(module, "DOMAIN_EXTRACTION_MAP", getattr(module, "DOMAIN_MAPPING", {}))
    return mapping if isinstance(mapping, dict) else {}


def get_domain_info_for_schema(
    meta: Optional[SchemaMeta],
    domain_mapping: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Resolves domain metadata for a given schema by matching filename, module name,
    or target class names against the loaded domain mapping dictionary.
    """
    if not meta or not domain_mapping:
        return None


    filename = os.path.basename(meta.filepath)
    py_module_key = f"{meta.module_name}.py"

    if filename in domain_mapping:
        return domain_mapping[filename]

    if py_module_key in domain_mapping:
        return domain_mapping[py_module_key]

    if meta.module_name in domain_mapping:
        return domain_mapping[meta.module_name]

    for key, info in domain_mapping.items():
        if isinstance(info, dict):
            target_classes = info.get("target_classes", [])
            if meta.class_name in target_classes:
                return info

    return None


@dataclass
class DomainContainerMeta:
    """Metadata container for a dynamically synthesized domain extraction container model."""
    module_name: str
    container_cls: Type[BaseModel]
    member_schemas: List[SchemaMeta]
    filepath: str


def build_domain_containers(schema_meta_list: List[SchemaMeta]) -> List[DomainContainerMeta]:
    """
    Groups schemas by module (file) and dynamically synthesizes a Pydantic
    domain container model holding List[Model] fields for each class in the file.
    Enables zero-loss multi-class extraction in a single LLM pass per domain.
    """
    from pydantic import create_model, Field

    modules_map: Dict[str, List[SchemaMeta]] = {}
    for meta in schema_meta_list:
        modules_map.setdefault(meta.module_name, []).append(meta)

    domain_containers: List[DomainContainerMeta] = []

    for module_name, metas in modules_map.items():
        fields = {}
        for meta in metas:
            field_name = f"{meta.class_name.lower()}_list"
            fields[field_name] = (
                typing.Optional[List[meta.schema_cls]],
                Field(
                    default=None,
                    description=f"Optional list of extracted {meta.class_name} entity records. Leave null or empty [] if not explicitly mentioned."
                )
            )

        container_class_name = "".join(part.capitalize() for part in module_name.split("_")) + "DomainContainer"
        container_cls = create_model(
            container_class_name,
            **fields
        )


        filepath = metas[0].filepath if metas else ""
        domain_containers.append(DomainContainerMeta(
            module_name=module_name,
            container_cls=container_cls,
            member_schemas=metas,
            filepath=filepath
        ))

    return domain_containers


