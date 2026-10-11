from __future__ import annotations

from pydantic import BaseModel

from clinical_trial_intelligence.observability import get_logger
from clinical_trial_intelligence.skills.base import Skill, SkillResult, traced_skill

logger = get_logger(__name__)


class Outcome(BaseModel):
    measure: str
    description: str | None = None
    time_frame: str | None = None
    safety_issue: bool = False


class ExtractedOutcomes(BaseModel):
    primary: list[Outcome] = []
    secondary: list[Outcome] = []
    other: list[Outcome] = []
    has_results: bool = False


class OutcomeExtractor(Skill[dict[str, Any], dict[str, Any]]):
    name = "outcome_extractor"
    version = "1.0.0"

    @traced_skill()
    async def execute(self, input: dict[str, Any]) -> SkillResult[dict[str, Any]]:
        study_record = input.get("study_record", {})
        study_results = input.get("study_results", {})

        outcomes_module = study_record.get("protocolSection", {}).get("outcomesModule", {})

        primary = self._extract_outcomes(outcomes_module.get("primaryOutcomes", []), "primary")
        secondary = self._extract_outcomes(outcomes_module.get("secondaryOutcomes", []), "secondary")
        other = self._extract_outcomes(outcomes_module.get("otherOutcomes", []), "other")

        has_results = bool(study_results.get("hasResults") or study_results.get("resultsSection"))

        if has_results:
            results_module = study_results.get("resultsSection", {})
            primary = self._enrich_with_results(primary, results_module.get("primaryOutcomes", []))
            secondary = self._enrich_with_results(secondary, results_module.get("secondaryOutcomes", []))

        return SkillResult(
            output={
                "primary": primary,
                "secondary": secondary,
                "other": other,
                "has_results": has_results,
            },
            confidence=0.9 if (primary or secondary) else 0.5,
            provenance=self._create_provenance(
                primary_count=len(primary),
                secondary_count=len(secondary),
                has_results=has_results,
            ),
        )

    def _extract_outcomes(self, outcomes: list[dict], category: str) -> list[dict]:
        extracted = []
        for o in outcomes:
            extracted.append({
                "measure": o.get("measure", ""),
                "description": o.get("description", ""),
                "time_frame": o.get("timeFrame", ""),
                "safety_issue": o.get("safetyIssue", False),
                "category": category,
            })
        return extracted

    def _enrich_with_results(self, outcomes: list[dict], results: list[dict]) -> list[dict]:
        for out in outcomes:
            for res in results:
                if out["measure"] == res.get("measure"):
                    out["result_value"] = res.get("value")
                    out["result_ci"] = res.get("confidenceInterval")
                    out["p_value"] = res.get("pValue")
        return outcomes