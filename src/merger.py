"""
Exact Data Integrity Consolidation Engine

Consolidates Pydantic model instances across chunks while preserving
100% exact raw extracted values with zero string or numerical alteration.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
import json
from typing import Any, Dict, List, Type
from pydantic import BaseModel


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to safely serialize dates, datetimes, enums,
    and Pydantic BaseModel instances.
    """
    def default(self, obj: Any) -> Any:
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


def merge_pydantic_instances(schema_cls: Type[BaseModel], instances: List[BaseModel]) -> BaseModel:
    """
    Consolidates Pydantic model instances across chunks while preserving EXACT raw extracted values:
      - Zero mathematical alteration or averaging.
      - Zero text mutation, lowercasing, or artificial string concatenation.
      - List items deduplicated based on exact raw JSON equality.
      - Enums & Booleans preserved exactly as reported in primary clinical mentions.
    """
    if not instances:
        raise ValueError("No instances to merge.")
    if len(instances) == 1:
        return instances[0]

    merged_data: Dict[str, Any] = {}
    for field_name in schema_cls.model_fields:
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
