"""
Dedicated LLM Prompts & Instruction Templates for SchemaMap

Centralizes all system prompts, triage prompts, extraction rules,
and dynamic plain-English field instruction generation.
Edit this file to adjust or fine-tune prompt language, rules, or formatting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type
from pydantic import BaseModel

if TYPE_CHECKING:
    from .schemas import SchemaMeta


# =====================================================================
# 1. Core Prompt Rules & Text Blocks
# =====================================================================

EXTRACTION_RULES = """### Accuracy & Extraction Rules
1. **Verbatim Fidelity:** Extract names, numbers, scores, dates, and terms exactly as written in the text. Never paraphrase or modify raw values.
2. **Strict Null Rule:** If an entity or field is NOT explicitly mentioned in the text excerpt, leave its value as `null` (or empty list `[]`). Do not guess, assume, or invent values.
3. **Context Grounding:** Rely ONLY on clear facts mentioned directly in the document excerpt below."""

TRIAGE_RULES = """### Triage Rules
1. Include a schema class name in `relevant_schemas` ONLY if the text contains clear facts about it.
2. If a schema's topic is NOT discussed in the excerpt, do NOT include it.
3. Output valid JSON matching the schema."""


# =====================================================================
# 2. Field Instruction Formatter
# =====================================================================

def format_schema_field_instructions(schema_cls: Type[BaseModel]) -> str:
    """
    Inspects a Pydantic model's fields and generates clear, human-readable
    instructions for the LLM describing each field's name, type, and purpose.
    """
    lines: List[str] = []
    if not hasattr(schema_cls, "model_fields") or not schema_cls.model_fields:
        return "- No specific fields defined."

    for field_name, field_info in schema_cls.model_fields.items():
        desc = field_info.description or "Extract this value if mentioned."
        annotation = field_info.annotation

        # Format human-friendly type
        if hasattr(annotation, "__name__"):
            type_label = annotation.__name__
        else:
            type_label = str(annotation).replace("typing.", "")

        req_label = "Required" if field_info.is_required() else "Optional"
        lines.append(f"- **`{field_name}`** ({type_label}, {req_label}): {desc}")

    return "\n".join(lines)


# =====================================================================
# 3. Schema Relevance Triage Prompt Builder
# =====================================================================

def build_triage_prompt(candidate_schemas: Sequence[Any]) -> str:
    """
    Builds a lightweight triage prompt that describes candidate schemas concisely
    for rapid classification before deep extraction.
    """
    schema_lines: List[str] = []
    for m in candidate_schemas:
        doc = (m.schema_cls.__doc__ or "").strip()
        if doc:
            first_line = doc.split("\n")[0]
            summary = first_line[:120]
        else:
            # Introspect model fields for clear semantic signal when docstring is absent
            fields = list(getattr(m.schema_cls, "model_fields", {}).keys())[:4]
            if fields:
                summary = "Extracts: " + ", ".join(fields)
            else:
                summary = "Entity schema"
        schema_lines.append(f"- **`{m.class_name}`**: {summary}")

    return (
        "You are a fast document triage assistant.\n"
        "Your task is to inspect the document excerpt and identify ONLY which entity schemas have "
        "explicit, factual mentions or relevant data in the text.\n\n"
        "### Candidate Schemas\n"
        + "\n".join(schema_lines)
        + "\n\n"
        + TRIAGE_RULES
        + "\n\n"
        "### Document Excerpt\n"
    )


# =====================================================================
# 4. Single-Schema Deep Extraction Prompt Builder
# =====================================================================

def build_extraction_prompt(
    schema_cls: Type[BaseModel],
    domain_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Constructs an optimized, easy-to-follow extraction prompt tailored for
    a single Pydantic schema class.
    """
    class_name = schema_cls.__name__
    docstring = (schema_cls.__doc__ or "").strip()
    field_guide = format_schema_field_instructions(schema_cls)

    prompt = (
        "You are an expert structured data extraction assistant.\n"
        f"Your task is to extract all factual information from the text that matches the schema: **{class_name}**.\n\n"
    )

    if docstring:
        prompt += f"### Schema Description\n{docstring}\n\n"

    if domain_info:
        title = domain_info.get("domain_title")
        instructions = domain_info.get("extraction_instructions")
        if title or instructions:
            prompt += "### Domain Focus Area\n"
            if title:
                prompt += f"**Domain:** {title}\n"
            if instructions:
                prompt += f"**Specific Focus:** {instructions}\n"
            prompt += "\n"

    prompt += (
        f"### Target Fields to Extract for {class_name}\n"
        f"{field_guide}\n\n"
        + EXTRACTION_RULES
        + "\n\n"
        "### Document Excerpt\n"
    )
    return prompt
