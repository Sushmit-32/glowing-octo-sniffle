from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from clinical_trial_intelligence.observability import get_logger
from clinical_trial_intelligence.skills.base import Skill, SkillResult, traced_skill

logger = get_logger(__name__)


@dataclass
class QualityScore:
    overall: float
    breakdown: dict[str, float]
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


class QualityScorerConfig(BaseModel):
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "completeness": 0.30,
            "consistency": 0.20,
            "provenance": 0.25,
            "results_availability": 0.15,
            "recency": 0.10,
        }
    )


class QualityScorer(Skill[dict[str, Any], dict[str, Any]]):
    name = "quality_scorer"
    version = "1.0.0"

    def __init__(self, config: QualityScorerConfig | None = None):
        self.config = config or QualityScorerConfig()

    @traced_skill()
    async def execute(self, input: dict[str, Any]) -> SkillResult[dict[str, Any]]:
        record = input

        completeness = self._score_completeness(record)
        consistency = self._score_consistency(record)
        provenance = self._score_provenance(record)
        results_availability = self._score_results_availability(record)
        recency = self._score_recency(record)

        weights = self.config.weights
        overall = (
            completeness * weights["completeness"]
            + consistency * weights["consistency"]
            + provenance * weights["provenance"]
            + results_availability * weights["results_availability"]
            + recency * weights["recency"]
        )

        breakdown = {
            "completeness": completeness,
            "consistency": consistency,
            "provenance": provenance,
            "results_availability": results_availability,
            "recency": recency,
        }

        confidence = self._calculate_confidence(breakdown)

        return SkillResult(
            output={
                "overall": overall,
                "breakdown": breakdown,
                "confidence": confidence,
                "details": {"weights_used": weights, "field_count": len(record)},
            },
            confidence=confidence,
            provenance=self._create_provenance(weights=weights),
        )

    def _score_completeness(self, record: dict[str, Any]) -> float:
        fields = [
            "brief_title", "official_title", "overall_status", "phase", "study_type",
            "enrollment", "start_date", "completion_date", "sponsor", "conditions",
            "interventions", "primary_outcomes", "secondary_outcomes", "eligibility_criteria",
            "inclusion_criteria", "exclusion_criteria",
        ]
        present = sum(1 for f in fields if record.get(f))
        return present / len(fields)

    def _score_consistency(self, record: dict[str, Any]) -> float:
        score = 1.0
        if record.get("phase") and "Phase" not in str(record.get("phase", "")):
            score -= 0.1
        if record.get("enrollment") and record.get("enrollment", 0) < 0:
            score -= 0.2
        return max(0.0, score)

    def _score_provenance(self, record: dict[str, Any]) -> float:
        prov = record.get("provenance", {})
        if not prov:
            return 0.0
        fields_with_prov = sum(1 for k in record if k in prov)
        return min(1.0, fields_with_prov / 10)

    def _score_results_availability(self, record: dict[str, Any]) -> float:
        return 1.0 if record.get("has_results") else 0.0

    def _score_recency(self, record: dict[str, Any]) -> float:
        return 0.5

    def _calculate_confidence(self, breakdown: dict[str, float]) -> float:
        return sum(breakdown.values()) / len(breakdown)