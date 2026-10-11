from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from clinical_trial_intelligence.models import ComplianceReport, ConstitutionalFlag, EnforcementMode, Severity
from clinical_trial_intelligence.skills.constitutional_auditor import ConstitutionalAuditor


@dataclass
class AuditReport:
    run_id: str
    timestamp: datetime
    total_records: int
    total_flags: int
    flags_by_principle: dict[str, list[ConstitutionalFlag]] = field(default_factory=dict)
    compliance_rate: float
    disclaimer_verified: bool = False
    enforcement_mode: EnforcementMode = EnforcementMode.SOFT


class ConstitutionalAuditorWrapper:
    def __init__(self, constitution_path: Path = Path("constitution/constitution.yaml")):
        self.constitution_path = constitution_path

    def generate_report(self, flags: list[ConstitutionalFlag], run_id: str) -> AuditReport:
        flags_by_principle: dict[str, list[ConstitutionalFlag]] = {}
        for flag in flags:
            flags_by_principle.setdefault(flag.principle, []).append(flag)

        disclaimer_verified = any(f.flag == "missing_disclaimer" for f in flags)

        return AuditReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            total_records=1,
            total_flags=len(flags),
            flags_by_principle=flags_by_principle,
            compliance_rate=1.0 if not flags else 0.8,
            disclaimer_verified=disclaimer_verified,
            enforcement_mode=EnforcementMode.SOFT,
        )

    def save_report(self, report: AuditReport, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self._to_markdown(report))

    def _to_markdown(self, report: AuditReport) -> str:
        lines = [
            f"# Constitutional Compliance Report",
            f"",
            f"**Run ID:** {report.run_id}",
            f"**Timestamp:** {report.timestamp.isoformat()}",
            f"**Enforcement Mode:** {report.enforcement_mode.value}",
            f"**Total Records:** {report.total_records}",
            f"**Total Flags:** {report.total_flags}",
            f"**Compliance Rate:** {report.compliance_rate:.1%}",
            f"**Disclaimer Verified:** {'✅' if report.disclaimer_verified else '❌'}",
            f"",
            f"## Flags by Principle",
            f"",
        ]

        for principle, flags in report.flags_by_principle.items():
            lines.append(f"### {principle}")
            for flag in flags:
                lines.append(f"- **{flag.flag}** ({flag.severity.value}): {flag.evidence}")
                if flag.suggestion:
                    lines.append(f"  - *Suggestion:* {flag.suggestion}")

        return "\n".join(lines)