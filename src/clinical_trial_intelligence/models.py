from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EnforcementMode(str, Enum):
    SOFT = "soft"
    HARD = "hard"


class EligibilityMode(str, Enum):
    REGEX = "regex"
    LLM = "llm"


class ConstitutionalFlag(BaseModel):
    principle: str
    flag: str
    field: str
    evidence: str
    severity: Severity
    nct_id: str
    suggestion: str | None = None


@dataclass
class SkillResult:
    output: Any
    confidence: float
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ParsedEligibility(BaseModel):
    inclusion: list[str] = Field(default_factory=list)
    exclusion: list[str] = Field(default_factory=list)
    raw_text: str = ""
    parser_mode: str = "regex"


class Outcome(BaseModel):
    measure: str
    description: str | None = None
    time_frame: str | None = None
    safety_issue: bool = False


class ExtractedOutcomes(BaseModel):
    primary: list[Outcome] = Field(default_factory=list)
    secondary: list[Outcome] = Field(default_factory=list)
    other: list[Outcome] = Field(default_factory=list)
    has_results: bool = False


class StudyDesign(BaseModel):
    phase: str | None = None
    study_type: str | None = None
    randomization: str | None = None
    allocation: str | None = None
    masking: str | None = None
    who_masked: list[str] = Field(default_factory=list)
    primary_purpose: str | None = None
    enrollment: int | None = None
    arms_info: list[dict[str, Any]] = Field(default_factory=list)


class QualityScore(BaseModel):
    overall: float
    breakdown: dict[str, float]
    confidence: float
    details: dict[str, Any] = Field(default_factory=dict)


class FieldVerification(BaseModel):
    field: str
    expected: Any
    actual: Any
    match: bool
    confidence: float
    jsonpath: str


class VerificationResult(BaseModel):
    per_field: dict[str, FieldVerification] = Field(default_factory=dict)
    overall_pass: bool = True
    mismatch_count: int = 0


class AuditResult(BaseModel):
    flags: list[ConstitutionalFlag] = Field(default_factory=list)
    compliance_score: float = 1.0
    passed: bool = True


class ComplianceReport(BaseModel):
    run_id: str
    timestamp: datetime
    total_records: int
    total_flags: int
    flags_by_principle: dict[str, list[ConstitutionalFlag]] = Field(default_factory=dict)
    compliance_rate: float
    disclaimer_verified: bool = False
    enforcement_mode: EnforcementMode = EnforcementMode.SOFT


class TrialRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    nct_id: str
    brief_title: str | None = None
    official_title: str | None = None
    overall_status: str | None = None
    phase: str | None = None
    study_type: str | None = None
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    sponsor: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    primary_outcomes: list[str] = Field(default_factory=list)
    secondary_outcomes: list[str] = Field(default_factory=list)
    eligibility_criteria: str | None = None
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    data_quality_score: float | None = None
    extraction_confidence: float | None = None
    constitutional_flags: list[ConstitutionalFlag] = Field(default_factory=list)
    source_timestamp: str = Field(default_factory=utc_now)
    provenance: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    nct_id: str
    extracted: TrialRecord | None = None
    raw_api: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    provenance: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    constitutional_flags: list[ConstitutionalFlag] = Field(default_factory=list)
    quality_score: float | None = None


class ValidatedNCTs(BaseModel):
    valid: list[str] = Field(default_factory=list)
    invalid: list[dict[str, str]] = Field(default_factory=list)
    duplicates: list[str] = Field(default_factory=list)


class BatchResult(BaseModel):
    batch_id: str
    successful: list[ExtractionResult] = Field(default_factory=list)
    failed: list[dict[str, str]] = Field(default_factory=list)
    flags: list[ConstitutionalFlag] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    batch_size: int = 10
    parallel_workers: int = 3
    mcp_server_url: str = "https://clinicaltrials.caseyjhand.com/mcp"
    checkpoint_dir: Path = Field(default=Path("data/checkpoints"))
    csv_output_dir: Path = Field(default=Path("output"))
    episodic_log_dir: Path = Field(default=Path("data/episodic"))
    constitution_path: Path = Field(default=Path("constitution/constitution.yaml"))
    enforcement_mode: EnforcementMode = EnforcementMode.SOFT
    confidence_threshold: float = 0.7
    request_timeout_ms: int = 30000
    max_retries: int = 3
    checkpoint_enabled: bool = True
    resume: bool = False
    eligibility_mode: EligibilityMode = EligibilityMode.REGEX


class PipelineResult(BaseModel):
    run_id: str
    config: PipelineConfig
    start_time: datetime
    end_time: datetime | None = None
    total_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    batches: list[BatchResult] = Field(default_factory=list)
    compliance_report: ComplianceReport | None = None
    output_path: Path | None = None
    error: str | None = None


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    completed_batches: list[str] = Field(default_factory=list)
    pending_ncts: list[str] = Field(default_factory=list)
    failed_ncts: list[tuple[str, str]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def update_timestamp(self) -> None:
        self.updated_at = utc_now()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Checkpoint | None:
        if not path.exists():
            return None
        return cls.model_validate_json(path.read_text())