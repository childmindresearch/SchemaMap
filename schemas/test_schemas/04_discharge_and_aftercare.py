"""
Discharge and Aftercare Summary Schema

Captures summary discharge status, resolved conditions, and post-episode aftercare plans.
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class DischargeOutcome(str, Enum):
    IMPROVED = "improved"
    STABLE = "stable"
    TRANSFERRED = "transferred"
    AGAINST_ADVICE = "against_advice"
    COMPLETED = "completed"


class DischargeDiagnosis(BaseModel):
    """Diagnostic state at conclusion of clinical service."""
    condition_name: str = Field(..., description="Name of clinical condition e.g. ADHD, Generalized Anxiety Disorder")
    status: Optional[str] = Field(None, description="Resolved, persistent, or in maintenance")


class DischargeSummaryReport(BaseModel):
    """
    ROOT CONTAINER: Patient Discharge Summary Report.
    Summarizes treatment response, disposition status, and continuity of care instructions.
    """
    discharge_date: Optional[str] = Field(None, description="Date of discharge or episode completion")
    discharge_status: Optional[DischargeOutcome] = Field(None, description="Clinical disposition or outcome status")
    diagnoses: List[DischargeDiagnosis] = Field(default_factory=list, description="Final clinical diagnoses at discharge")
    aftercare_recommendations: List[str] = Field(default_factory=list, description="Recommended post-discharge aftercare plan")
