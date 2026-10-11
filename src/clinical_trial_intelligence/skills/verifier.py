from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jsonpath_ng import parse as jsonpath_parse

from clinical_trial_intelligence.observability import get_logger
from clinical_trial_intelligence.skills.base import Skill, SkillResult, traced_skill

logger = get_logger(__name__)


@dataclass
class FieldVerification:
    field: str
    expected: Any
    actual: Any
    match: bool
    confidence: float
    jsonpath: str


@dataclass
class VerificationResult:
    per_field: dict[str, FieldVerification] = field(default_factory=dict)
    overall_pass: bool = True
    mismatch_count: int = 0


FIELD_PATHS = {
    "nct_id": "$.nctId",
    "brief_title": "$.briefTitle",
    "official_title": "$.officialTitle",
    "phase": "$.phases[0]",
    "study_type": "$.studyType",
    "enrollment": "$.enrollmentInfo.count",
    "conditions": "$.conditions",
    "interventions": "$.arms[*].interventions[*].name",
    "sponsor": "$.sponsorCollaboratorsModule.leadSponsor.name",
    "start_date": "$.statusModule.startDateStruct.date",
    "completion_date": "$.statusModule.completionDateStruct.date",
    "eligibility_criteria": "$.eligibilityModule.eligibilityCriteria",
    "primary_outcomes": "$.outcomesModule.primaryOutcomes[*].measure",
    "secondary_outcomes": "$.outcomesModule.secondaryOutcomes[*].measure",
}


class Verifier(Skill[dict[str, Any], dict[str, Any]]):
    name = "verifier"
    version = "1.0.0"

    @traced_skill()
    async def execute(self, input: dict[str, Any]) -> SkillResult[dict[str, Any]]:
        extracted = input.get("extracted", {})
        raw_api = input.get("raw_api", {})

        per_field = {}
        mismatch_count = 0

        for field_name, jsonpath_expr in FIELD_PATHS.items():
            verification = self._verify_field(field_name, jsonpath_expr, raw_api, extracted)
            per_field[field_name] = verification
            if not verification.match:
                mismatch_count += 1

        overall_pass = mismatch_count == 0
        confidence = 1.0 - (mismatch_count / len(FIELD_PATHS)) if FIELD_PATHS else 1.0

        output = VerificationResult(
            per_field=per_field,
            overall_pass=overall_pass,
            mismatch_count=mismatch_count,
        )

        return SkillResult(
            output=output,
            confidence=confidence,
            provenance=self._create_provenance(
                fields_verified=len(FIELD_PATHS),
                mismatches=mismatch_count,
            ),
        )

    def _verify_field(
        self,
        field_name: str,
        jsonpath_expr: str,
        raw: dict[str, Any],
        extracted: dict[str, Any],
    ) -> FieldVerification:
        try:
            expected_matches = jsonpath_parse(jsonpath_expr).find(raw)
            expected = expected_matches[0].value if expected_matches else None
        except Exception:
            expected = None

        actual = extracted.get(field_name)

        match, confidence = self._compare_values(expected, actual)

        return FieldVerification(
            field=field_name,
            expected=expected,
            actual=actual,
            match=match,
            confidence=confidence,
            jsonpath=jsonpath_expr,
        )

    def _compare_values(self, expected: Any, actual: Any) -> tuple[bool, float]:
        if expected is None and actual is None:
            return True, 1.0

        if expected is None or actual is None:
            return False, 0.0

        if isinstance(expected, list) and isinstance(actual, list):
            return self._compare_lists(expected, actual)

        if isinstance(expected, dict) and isinstance(actual, dict):
            return self._compare_dicts(expected, actual)

        match = str(expected).lower().strip() == str(actual).lower().strip()
        return match, 1.0 if match else 0.0

    def _compare_lists(self, expected: list[Any], actual: list[Any]) -> tuple[bool, float]:
        if len(expected) != len(actual):
            return False, 0.5

        if not expected and not actual:
            return True, 1.0

        matches = 0
        for e, a in zip(expected, actual, strict=False):
            match, _ = self._compare_values(e, a)
            if match:
                matches += 1

        confidence = matches / len(expected) if expected else 1.0
        return matches == len(expected), confidence

    def _compare_dicts(self, expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, float]:
        all_keys = set(expected.keys()) | set(actual.keys())
        if not all_keys:
            return True, 1.0

        matches = 0
        for key in all_keys:
            match, _ = self._compare_values(expected.get(key), actual.get(key))
            if match:
                matches += 1

        confidence = matches / len(all_keys)
        return matches == len(all_keys), confidence