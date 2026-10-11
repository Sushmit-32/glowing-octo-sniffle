from __future__ import annotations

import re

from clinical_trial_intelligence.mcp_client import MCPClient
from clinical_trial_intelligence.observability import get_logger
from clinical_trial_intelligence.skills.base import Skill, SkillResult, traced_skill

logger = get_logger(__name__)

NCT_PATTERN = re.compile(r"^NCT\d{8}$", re.IGNORECASE)


class NCTValidator(Skill[list[str], dict[str, Any]]):
    name = "nct_validator"
    version = "1.0.0"

    def __init__(self, mcp_client: MCPClient | None = None):
        self._mcp_client = mcp_client

    @traced_skill()
    async def execute(self, input: list[str]) -> SkillResult[dict[str, Any]]:
        seen: set[str] = set()
        duplicates: list[str] = []
        valid: list[str] = []
        invalid: list[dict[str, str]] = []

        for code in input:
            normalized = code.strip().upper()
            if normalized in seen:
                duplicates.append(normalized)
                continue
            seen.add(normalized)

            if not NCT_PATTERN.match(normalized):
                invalid.append({"code": code, "reason": "Invalid format: must match ^NCT\\d{8}$"})
                continue

            valid.append(normalized)

        existence_results: dict[str, bool] = {}
        if self._mcp_client and valid:
            existence_results = await self._check_existence_batch(valid)

        final_valid: list[str] = []
        for nct_id in valid:
            if existence_results.get(nct_id, True):
                final_valid.append(nct_id)
            else:
                invalid.append({"code": nct_id, "reason": "Study not found in ClinicalTrials.gov"})

        confidence = 1.0 if not invalid and not duplicates else 0.8
        if invalid:
            confidence = 0.5
        if not valid and not invalid:
            confidence = 0.0

        output = {
            "valid": final_valid,
            "invalid": invalid,
            "duplicates": duplicates,
        }

        return SkillResult(
            output=output,
            confidence=confidence,
            provenance=self._create_provenance(
                total_input=len(input),
                format_valid=len(valid),
                existence_checked=bool(self._mcp_client),
            ),
        )

    async def _check_existence_batch(self, nct_ids: list[str]) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for nct_id in nct_ids:
            try:
                await self._mcp_client.get_study_record(nct_id)
                results[nct_id] = True
            except Exception:
                results[nct_id] = False
        return results