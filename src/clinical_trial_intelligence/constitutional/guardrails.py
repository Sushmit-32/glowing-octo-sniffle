from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from clinical_trial_intelligence.models import ConstitutionalFlag, EnforcementMode, Severity
from clinical_trial_intelligence.skills.constitutional_auditor import ConstitutionalAuditor

class ConstitutionalGuardrails:
    def __init__(self, constitution_path: Path, enforcement_mode: EnforcementMode = EnforcementMode.SOFT):
        self.constitution_path = constitution_path
        self.enforcement_mode = enforcement_mode
        self._constitution = self._load_constitution()

    def _load_constitution(self) -> dict[str, Any]:
        try:
            with open(self.constitution_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            return {"principles": []}

    def pre_flight(self, config: Any) -> list[ConstitutionalFlag]:
        flags = []
        if not self.constitution_path.exists():
            flags.append(ConstitutionalFlag(
                principle="constitution", flag="constitution_missing",
                field="config", evidence="Constitution file not found",
                severity=Severity.CRITICAL, nct_id="",
                suggestion="Ensure constitution.yaml exists at configured path"
            ))
        return flags

    def per_record(self, record: dict[str, Any]) -> list[ConstitutionalFlag]:
        flags = []
        auditor = ConstitutionalAuditor()
        import asyncio
        result = asyncio.run(auditor.execute((record, {"enforcement_mode": self.enforcement_mode.value})))
        return result.output.get("flags", [])

    def post_flight(self, all_records: list[dict[str, Any]], compliance_report: Any) -> None:
        pass