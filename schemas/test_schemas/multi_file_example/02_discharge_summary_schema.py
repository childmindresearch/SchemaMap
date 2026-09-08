"""
Multi-File Schema Example - File 2: Discharge Summary Schema

Part of a multi-file schema directory test. In Root Container Mode,
the framework discovers DischargeSummaryReport from this file.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ClinicalDiagnosis(BaseModel):
    """Diagnosed clinical condition."""
    condition_name: str = Field(..., description="Name of diagnosed condition")
    status: Optional[str] = Field(None, description="Confirmed or resolved")


class DischargeSummaryReport(BaseModel):
    """
    ROOT CONTAINER MODEL FOR DISCHARGE:
    Auto-detected root container model for patient discharge reports.
    """
    diagnoses: List[ClinicalDiagnosis] = Field(default_factory=list, description="Final clinical diagnoses")
    discharge_status: Optional[str] = Field(None, description="Status at discharge e.g. improved, stable")
    aftercare_recommendations: List[str] = Field(default_factory=list, description="Recommended post-discharge interventions")
