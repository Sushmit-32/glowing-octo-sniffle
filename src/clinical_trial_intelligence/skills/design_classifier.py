from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from clinical_trial_intelligence.observability import get_logger
from clinical_trial_intelligence.skills.base import Skill, SkillResult, traced_skill

logger = get_logger(__name__)


class StudyDesign(BaseModel):
    phase: str | None = None
    study_type: str | None = None
    randomization: str | None = None
    allocation: str | None = None
    masking: str | None = None
    who_masked: list[str] = []
    primary_purpose: str | None = None
    enrollment: int | None = None
    arms_info: list[dict[str, Any]] = []


class DesignClassifier(Skill[dict[str, Any], dict[str, Any]]):
    name = "design_classifier"
    version = "1.0.0"

    @traced_skill()
    async def execute(self, input: dict[str, Any]) -> SkillResult[dict[str, Any]]:
        design_module = input.get("protocolSection", {}).get("designModule", {})

        phases = design_module.get("phases", [])
        phase = phases[0] if phases else None

        study_type = design_module.get("studyType")

        design_info = design_module.get("designInfo", {})
        randomization = design_info.get("allocation")
        allocation = design_info.get("interventionModel")
        masking = design_info.get("maskingInfo", {}).get("masking")
        who_masked = design_info.get("maskingInfo", {}).get("whoMasked", [])
        primary_purpose = design_info.get("primaryPurpose")

        enrollment_info = design_module.get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count")

        arms = design_module.get("arms", [])
        arms_info = [{"type": a.get("type"), "label": a.get("label")} for a in arms]

        fields_extracted = sum(
            1
            for v in [phase, study_type, randomization, allocation, masking, primary_purpose, enrollment]
            if v is not None
        )

        return SkillResult(
            output={
                "phase": phase,
                "study_type": study_type,
                "randomization": randomization,
                "allocation": allocation,
                "masking": masking,
                "who_masked": who_masked,
                "primary_purpose": primary_purpose,
                "enrollment": enrollment,
                "arms_info": arms_info,
            },
            confidence=min(0.3 + fields_extracted * 0.1, 1.0),
            provenance=self._create_provenance(fields_extracted=fields_extracted),
        )