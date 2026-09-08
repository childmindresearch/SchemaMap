"""
Multi-Class Schema Example

This schema pattern defines standalone, un-nested Pydantic models. 

In Multi-Class Mode (`use_root_schema: false`), the framework loops through 
each defined class sequentially, dispatching separate LLM extraction requests 
for each individual class per text chunk.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PatientDemographics(BaseModel):
    """Standalone entity model for patient demographics."""
    patient_id: Optional[str] = Field(None, description="Patient identifier e.g. PT-9042")
    age: Optional[int] = Field(None, description="Age of the patient")
    gender: Optional[str] = Field(None, description="Gender or sex")


class DiagnosisRecord(BaseModel):
    """Standalone entity model for a single clinical diagnosis."""
    condition_name: str = Field(..., description="Name of diagnosed condition")
    status: Optional[str] = Field(None, description="Confirmed, suspected, or ruled out")
    severity: Optional[str] = Field(None, description="Mild, moderate, or severe")


class AcademicAccommodation(BaseModel):
    """Standalone entity model for school or classroom accommodations."""
    accommodation_name: str = Field(..., description="Name of accommodation e.g. extended time, front seating")
    setting: Optional[str] = Field(None, description="Classroom, examination, or school environment")


class ClinicalNoteDisclaimer(BaseModel):
    """Standalone entity model for note disclaimers."""
    disclaimer_present: bool = Field(False, description="Whether a confidentiality disclaimer is present")
    disclaimer_text: Optional[str] = Field(None, description="Exact disclaimer wording if present")
