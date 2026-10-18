"""
Patient Intake and Demographics Schema

General clinical intake schema capturing patient demographics, contact context,
primary referral rationale, and initial presenting concerns.
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class GenderIdentity(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    OTHER = "other"
    UNKNOWN = "unknown"


class ReferralSource(str, Enum):
    SELF = "self"
    PRIMARY_CARE = "primary_care"
    SCHOOL_EDUCATOR = "school_educator"
    SPECIALIST = "specialist"
    EMERGENCY = "emergency"
    OTHER = "other"


class PatientDemographics(BaseModel):
    """Demographic and identification attributes for an evaluated individual."""
    patient_id: Optional[str] = Field(None, description="Patient medical record or evaluation identifier e.g. PT-9042")
    age: Optional[int] = Field(None, description="Patient age in completed years")
    gender: Optional[GenderIdentity] = Field(None, description="Gender identity or administrative sex")
    grade_or_occupation: Optional[str] = Field(None, description="Current school grade or occupational status")


class PatientIntakeSummary(BaseModel):
    """
    ROOT CONTAINER: Patient Intake Summary.
    Captures baseline presentation, referral context, and initial clinical intake details.
    """
    patient: Optional[PatientDemographics] = Field(None, description="Demographic and identification details")
    referral_source: Optional[ReferralSource] = Field(None, description="Originating source of clinical referral")
    reason_for_referral: Optional[str] = Field(None, description="Primary clinical concern or presenting symptoms")
    intake_date: Optional[str] = Field(None, description="Date of initial clinical intake or evaluation")
    presenting_complaints: List[str] = Field(default_factory=list, description="Specific functional or clinical complaints reported")
