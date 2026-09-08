"""
Multi-File Schema Example - File 1: Patient Intake Schema

Part of a multi-file schema directory test. In Root Container Mode,
the framework discovers PatientIntakeSummary from this file.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Patient demographics and identification."""
    patient_id: Optional[str] = Field(None, description="Patient identifier e.g. PT-9042")
    age: Optional[int] = Field(None, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Gender or sex of the patient")


class PatientIntakeSummary(BaseModel):
    """
    ROOT CONTAINER MODEL FOR INTAKE:
    Auto-detected root container model for patient intake evaluations.
    """
    patient: Optional[PatientInfo] = Field(None, description="Demographics and patient info")
    reason_for_referral: Optional[str] = Field(None, description="Primary concern or referral reason")
    intake_date: Optional[str] = Field(None, description="Date of intake evaluation")
