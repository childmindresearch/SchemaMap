"""
Root Container Schema Example

This schema pattern uses a single top-level container model (ClinicalReportSummary)
that nests child models as fields. 

In Root Container Mode (`use_root_schema: true`), the framework auto-detects 
this container class and extracts all nested fields in 1 LLM request per text chunk.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Patient demographics and identification."""
    patient_id: Optional[str] = Field(None, description="Patient identifier e.g. PT-9042")
    age: Optional[int] = Field(None, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Gender or sex of the patient")


class ClinicalDiagnosis(BaseModel):
    """Diagnosed clinical condition."""
    condition_name: str = Field(..., description="Name of the clinical condition e.g. ADHD, Anxiety")
    status: Optional[str] = Field(None, description="Status e.g. confirmed, suspected, ruled out")
    severity: Optional[str] = Field(None, description="Severity e.g. mild, moderate, severe")


class RecommendedIntervention(BaseModel):
    """Prescribed therapeutic or academic intervention."""
    intervention_type: str = Field(..., description="Type of intervention e.g. IEP accommodation, CBT therapy")
    details: Optional[str] = Field(None, description="Specific details or accommodation parameters")


class ClinicalReportSummary(BaseModel):
    """
    ROOT CONTAINER MODEL:
    Top-level wrapper nesting all child entity models into a unified JSON structure.
    """
    patient: Optional[PatientInfo] = Field(None, description="Demographics and patient info")
    diagnoses: List[ClinicalDiagnosis] = Field(default_factory=list, description="List of clinical diagnoses")
    interventions: List[RecommendedIntervention] = Field(default_factory=list, description="List of recommended interventions")
    key_findings: List[str] = Field(default_factory=list, description="Key clinical observations or findings")
