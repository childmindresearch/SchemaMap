"""
Treatment Plan and Interventions Schema

Captures multimodal therapeutic plans including behavioral therapies, educational accommodations,
and medical/pharmacotherapy regimens.
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class InterventionCategory(str, Enum):
    BEHAVIORAL_THERAPY = "behavioral_therapy"
    EDUCATIONAL_SUPPORT = "educational_support"
    PHARMACOTHERAPY = "pharmacotherapy"
    OCCUPATIONAL_THERAPY = "occupational_therapy"
    LIFESTYLE = "lifestyle"


class PlannedIntervention(BaseModel):
    """Specific therapeutic or supportive intervention item."""
    category: InterventionCategory = Field(..., description="Category of therapeutic modality")
    intervention_name: str = Field(..., description="Name of intervention e.g. Cognitive Behavioral Therapy, IEP, Methylphenidate")
    details: Optional[str] = Field(None, description="Specific parameters, dosage, or implementation strategies")
    frequency: Optional[str] = Field(None, description="Planned frequency or schedule e.g. weekly, daily, 1.5x time")


class TreatmentPlanReport(BaseModel):
    """
    ROOT CONTAINER: Multimodal Treatment Plan Report.
    Coordinates all therapeutic recommendations, accommodations, and follow-up timelines.
    """
    interventions: List[PlannedIntervention] = Field(default_factory=list, description="Prescribed therapeutic interventions")
    school_or_work_accommodations: List[str] = Field(default_factory=list, description="Environmental or educational accommodations")
    follow_up_interval: Optional[str] = Field(None, description="Recommended clinical follow-up timeframe")
