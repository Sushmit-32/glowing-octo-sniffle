from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from clinical_trial_intelligence.models import ConstitutionalFlag, EnforcementMode, Severity
from clinical_trial_intelligence.observability import get_logger, record_constitution_flag
from clinical_trial_intelligence.skills.base import Skill, SkillResult, traced_skill

logger = get_logger(__name__)


@dataclass
class AuditResult:
    flags: list[ConstitutionalFlag] = field(default_factory=list)
    compliance_score: float = 1.0
    passed: bool = True


class ConstitutionalAuditor(Skill[tuple[dict[str, Any], dict[str, Any]], dict[str, Any]]):
    name = "constitutional_auditor"
    version = "1.0.0"

    PII_PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        (r"\b\d{10,}\b", "Phone Number"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email"),
        (r"\b\d{1,5}\s\w+\s\w+\b", "Address"),
        (r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b", "DOB"),
        (r"\b(MR|MRN|Medical Record)\s*\d+\b", "Medical Record Number"),
    ]

    DEMOGRAPHIC_KEYWORDS = [
        "race", "ethnicity", "gender", "sex", "age", "pregnan",
        "disabilit", "veteran", "religion", "sexual orientation",
    ]

    def __init__(self, constitution_path: Path | None = None, confidence_threshold: float = 0.7):
        self.constitution_path = constitution_path or settings.constitution_path
        self.confidence_threshold = confidence_threshold
        self._constitution = self._load_constitution()

    def _load_constitution(self) -> dict[str, Any]:
        try:
            with open(self.constitution_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("constitution_load_failed", path=str(self.constitution_path), error=str(e))
            return {"principles": []}

    @traced_skill()
    async def execute(self, input: tuple[dict[str, Any], dict[str, Any]]) -> SkillResult[dict[str, Any]]:
        record, config = input
        enforcement_mode = config.get("enforcement_mode", "soft")

        flags: list[ConstitutionalFlag] = []

        flags.extend(await self._check_safety_first(record))
        flags.extend(await self._check_honesty_transparency(record))
        flags.extend(await self._check_privacy_respect(record))
        flags.extend(await self._check_human_oversight(record))
        flags.extend(await self._check_fairness_bias(record))
        flags.extend(await self._check_helpfulness(record))

        for flag in flags:
            record_constitution_flag(flag.principle, flag.severity.value)

        compliance_score = self._calculate_compliance_score(flags)
        passed = len([f for f in flags if f.severity in (Severity.CRITICAL, Severity.HIGH)]) == 0

        if enforcement_mode == EnforcementMode.HARD and not passed:
            critical_flags = [f for f in flags if f.severity == Severity.CRITICAL]
            if critical_flags:
                raise RuntimeError(f"Constitutional violation: {critical_flags}")

        return SkillResult(
            output={
                "flags": flags,
                "compliance_score": compliance_score,
                "passed": passed,
            },
            confidence=compliance_score,
            provenance=self._create_provenance(flags_count=len(flags), compliance_score=compliance_score),
        )

    async def _check_safety_first(self, record: dict[str, Any]) -> list[ConstitutionalFlag]:
        flags = []
        has_disclaimer = bool(record.get("constitutional_flags"))
        if not has_disclaimer:
            flags.append(
                ConstitutionalFlag(
                    principle="safety_first",
                    flag="missing_disclaimer",
                    field="record",
                    evidence="No medical disclaimer present in output",
                    severity=Severity.CRITICAL,
                    nct_id=record.get("nct_id", ""),
                    suggestion="Add disclaimer: 'This report extracts data from ClinicalTrials.gov. It does not constitute medical advice.'",
                )
            )
        return flags

    async def _check_honesty_transparency(self, record: dict[str, Any]) -> list[ConstitutionalFlag]:
        flags = []
        prov = record.get("provenance", {})
        required_prov_fields = ["source", "endpoint", "fetched_at"]
        missing = [f for f in required_prov_fields if f not in prov or not prov[f]]

        if missing:
            flags.append(
                ConstitutionalFlag(
                    principle="honesty_transparency",
                    flag="incomplete_provenance",
                    field="provenance",
                    evidence=f"Missing provenance fields: {', '.join(missing)}",
                    severity=Severity.WARNING,
                    nct_id=record.get("nct_id", ""),
                    suggestion="Ensure all provenance fields are populated during extraction",
                )
            )

        extracted_field_names = [
            "nct_id", "brief_title", "official_title", "phase", "study_type",
            "enrollment", "conditions", "interventions", "sponsor",
            "start_date", "completion_date", "eligibility_criteria",
        ]

        for field_name in extracted_field_names:
            if field_name not in prov:
                flags.append(
                    ConstitutionalFlag(
                        principle="honesty_transparency",
                        flag="missing_field_provenance",
                        field=field_name,
                        evidence=f"Field '{field_name}' has no provenance",
                        severity=Severity.INFO,
                        nct_id=record.get("nct_id", ""),
                    )
                )
        return flags

    async def _check_privacy_respect(self, record: dict[str, Any]) -> list[ConstitutionalFlag]:
        flags = []

        for field_name, value in record.items():
            if isinstance(value, str):
                for pattern, pii_type in self.PII_PATTERNS:
                    if re.search(pattern, value, re.IGNORECASE):
                        flags.append(
                            ConstitutionalFlag(
                                principle="privacy_respect",
                                flag="pii_detected",
                                field=field_name,
                                evidence=f"Potential {pii_type} detected",
                                severity=Severity.CRITICAL,
                                nct_id=record.get("nct_id", ""),
                                suggestion="Remove or redact PII from extracted data",
                            )
                        )

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        for pattern, pii_type in self.PII_PATTERNS:
                            if re.search(pattern, item, re.IGNORECASE):
                                flags.append(
                                    ConstitutionalFlag(
                                        principle="privacy_respect",
                                        flag="pii_detected",
                                        field=f"{field_name}[{i}]",
                                        evidence=f"Potential {pii_type} detected in list field '{field_name}'",
                                        severity=Severity.CRITICAL,
                                        nct_id=record.get("nct_id", ""),
                                    )
                                )
        return flags

    async def _check_human_oversight(self, record: dict[str, Any]) -> list[ConstitutionalFlag]:
        flags = []
        confidence = record.get("extraction_confidence") or record.get("data_quality_score") or 1.0

        if confidence < self.confidence_threshold:
            flags.append(
                ConstitutionalFlag(
                    principle="human_oversight",
                    flag="low_confidence",
                    field="record",
                    evidence=f"Extraction confidence {confidence:.2f} below threshold {self.confidence_threshold}",
                    severity=Severity.WARNING,
                    nct_id=record.get("nct_id", ""),
                    suggestion="Flag for human review before clinical use",
                )
            )

        if not record.get("primary_outcomes") and not record.get("secondary_outcomes"):
            flags.append(
                ConstitutionalFlag(
                    principle="human_oversight",
                    flag="missing_outcomes",
                    field="outcomes",
                    evidence="No outcome measures extracted",
                    severity=Severity.HIGH,
                    nct_id=record.get("nct_id", ""),
                )
            )
        return flags

    async def _check_fairness_bias(self, record: dict[str, Any]) -> list[ConstitutionalFlag]:
        flags = []

        for field_name in ["conditions", "interventions", "locations"]:
            field_value = record.get(field_name, [])
            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, str):
                        item_lower = item.lower()
                        for kw in self.DEMOGRAPHIC_KEYWORDS:
                            if kw in item_lower:
                                flags.append(
                                    ConstitutionalFlag(
                                        principle="fairness_bias",
                                        flag="demographic_term_in_field",
                                        field=field_name,
                                        evidence=f"Demographic term '{kw}' found in {field_name}: {item}",
                                        severity=Severity.INFO,
                                        nct_id=record.get("nct_id", ""),
                                    )
                                )
        return flags

    async def _check_helpfulness(self, record: dict[str, Any]) -> list[ConstitutionalFlag]:
        flags = []

        has_summary = bool(record.get("brief_title") or record.get("official_title"))
        has_key_fields = all(
            record.get(f) is not None for f in ["phase", "study_type", "enrollment", "conditions"]
        )

        if not has_summary or not has_key_fields:
            flags.append(
                ConstitutionalFlag(
                    principle="helpfulness",
                    flag="incomplete_summary",
                    field="record",
                    evidence=f"Summary incomplete: has_title={has_summary}, has_key_fields={has_key_fields}",
                    severity=Severity.INFO,
                    nct_id=record.get("nct_id", ""),
                    suggestion=(
                        "Ensure trial records include title, phase, study type, enrollment, and conditions"
                    ),
                )
            )
        return flags

    def _calculate_compliance_score(self, flags: list[ConstitutionalFlag]) -> float:
        if not flags:
            return 1.0

        severity_weights = {
            Severity.CRITICAL: 0.3,
            Severity.HIGH: 0.2,
            Severity.MEDIUM: 0.1,
            Severity.WARNING: 0.05,
            Severity.INFO: 0.02,
        }

        penalty = sum(severity_weights.get(f.severity, 0.01) for f in flags)
        return max(0.0, 1.0 - penalty)

    @classmethod
    async def generate_report(cls, flags: list[ConstitutionalFlag], run_id: str) -> ComplianceReport:
        from clinical_trial_intelligence.models import ComplianceReport, EnforcementMode

        flags_by_principle: dict[str, list[ConstitutionalFlag]] = {}
        for flag in flags:
            flags_by_principle.setdefault(flag.principle, []).append(flag)

        disclaimer_verified = any(f.flag == "missing_disclaimer" for f in flags)

        return ComplianceReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            total_records=1,
            total_flags=len(flags),
            flags_by_principle=flags_by_principle,
            compliance_rate=1.0 if not flags else 0.8,
            disclaimer_verified=disclaimer_verified,
            enforcement_mode=EnforcementMode.SOFT,
        )