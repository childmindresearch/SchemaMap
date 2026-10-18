"""
Clinical Evaluation and Diagnostic Assessment Schema

Captures formal clinical evaluation findings, diagnostic determinations, severity ratings,
and comorbid clinical conditions.
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class DiagnosticCertainty(str, Enum):
    CONFIRMED = "confirmed"
    PROVISIONAL = "provisional"
    RULED_OUT = "ruled_out"
    HISTORICAL = "historical"


class DiagnosticSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    IN_REMISSION = "in_remission"


class ClinicalDiagnosis(BaseModel):
    """Structured assessment of a specific clinical disorder or condition."""
    condition_name: str = Field(..., description="Name of diagnosed disorder e.g. Attention-Deficit/Hyperactivity Disorder, Generalized Anxiety Disorder")
    status: DiagnosticCertainty = Field(default=DiagnosticCertainty.CONFIRMED, description="Diagnostic status or certainty")
    severity: Optional[DiagnosticSeverity] = Field(None, description="Clinical severity level")
    specifiers: List[str] = Field(default_factory=list, description="Clinical specifiers e.g. Combined Type, Inattentive, Social Anxiety")


class ClinicalEvaluationSummary(BaseModel):
    """
    ROOT CONTAINER: Comprehensive Clinical Evaluation.
    Synthesizes diagnostic impressions, neuropsychological findings, and observational metrics.
    """
    evaluation_date: Optional[str] = Field(None, description="Date evaluation took place")
    diagnoses: List[ClinicalDiagnosis] = Field(default_factory=list, description="Formal clinical diagnostic impressions")
    cognitive_observations: List[str] = Field(default_factory=list, description="Observed cognitive, executive, or behavioral findings")
    behavioral_presentation: Optional[str] = Field(None, description="Clinician observation of mood, affect, and demeanor")
    risk_level: Optional[str] = Field(None, description="Assessed safety or clinical risk level e.g. low, moderate, none")
