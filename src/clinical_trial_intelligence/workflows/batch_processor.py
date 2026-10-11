from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import SpanKind

from clinical_trial_intelligence.config import settings
from clinical_trial_intelligence.mcp_client import MCPClient
from clinical_trial_intelligence.models import (
    BatchResult,
    Checkpoint,
    ExtractionResult,
    PipelineConfig,
    TrialRecord,
)
from clinical_trial_intelligence.observability import get_logger, set_batch_id
from clinical_trial_intelligence.skills import (
    CSVExporter,
    ConstitutionalAuditor,
    DesignClassifier,
    EligibilityParser,
    EpisodicLogExporter,
    NCTValidator,
    OutcomeExtractor,
    QualityScorer,
    Verifier,
)

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class BatchProcessor:
    def __init__(
        self,
        config: PipelineConfig,
        mcp_client: MCPClient,
        eligibility_parser: EligibilityParser,
        outcome_extractor: OutcomeExtractor,
        design_classifier: DesignClassifier,
        verifier: Verifier,
        constitutional_auditor: ConstitutionalAuditor,
        quality_scorer: QualityScorer,
    ):
        self.config = config
        self.mcp_client = mcp_client
        self.eligibility_parser = eligibility_parser
        self.outcome_extractor = outcome_extractor
        self.design_classifier = design_classifier
        self.verifier = verifier
        self.constitutional_auditor = constitutional_auditor
        self.quality_scorer = quality_scorer
        self._checkpoint_path = config.checkpoint_dir / "checkpoints"
        self._checkpoint_path.mkdir(parents=True, exist_ok=True)
        self._shutdown_requested = False

    async def process_batch(self, nct_ids: list[str], batch_id: str, run_id: str) -> BatchResult:
        batch_result = BatchResult(batch_id=batch_id)

        for nct_id in nct_ids:
            if self._shutdown_requested:
                break

            set_batch_id(batch_id)
            nct_log = logger.bind(nct_id=nct_id, batch_id=batch_id)

            try:
                extraction_result = await self._process_nct(nct_id, batch_id, run_id)
                if extraction_result:
                    batch_result.successful.append(extraction_result)
                    nct_log.info("nct_processing_complete", confidence=extraction_result.confidence)
                else:
                    batch_result.failed.append({"nct_id": nct_id, "error": "No extraction result"})
                    nct_log.error("nct_processing_failed", error="No extraction result")

            except Exception as e:
                batch_result.failed.append({"nct_id": nct_id, "error": str(e)})
                nct_log.exception("nct_processing_failed", error=str(e))

        batch_result.success_count = len(batch_result.successful)
        batch_result.failure_count = len(batch_result.failed)

        for result in batch_result.successful:
            batch_result.flags.extend(result.constitutional_flags)

        await self.save_checkpoint(run_id, [batch_id], [], batch_result.failed, {})

        return batch_result

    async def _process_nct(self, nct_id: str, batch_id: str, run_id: str) -> ExtractionResult | None:
        with tracer.start_as_current_span("nct.process", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("nct_id", nct_id)
            span.set_attribute("batch_id", batch_id)

            nct_log = logger.bind(nct_id=nct_id, batch_id=batch_id)

            nct_log.info("fetching_study_record")
            study_record = await self.mcp_client.get_study_record(nct_id)
            nct_log.info("fetching_study_results")
            study_results = await self.mcp_client.get_study_results(nct_id)

            raw_api = {
                "study_record": study_record,
                "study_results": study_results,
            }

            protocol_section = study_record.get("protocolSection", {})

            nct_log.info("extracting_eligibility")
            eligibility_text = protocol_section.get("eligibilityModule", {}).get("eligibilityCriteria", "")
            eligibility_result = await self.eligibility_parser.execute(eligibility_text)

            nct_log.info("extracting_outcomes")
            outcome_result = await self.outcome_extractor.execute({
                "study_record": study_record,
                "study_results": study_results,
            })

            nct_log.info("classifying_design")
            design_result = await self.design_classifier.execute(study_record)

            extracted = TrialRecord(
                nct_id=nct_id,
                brief_title=protocol_section.get("identificationModule", {}).get("briefTitle"),
                official_title=protocol_section.get("identificationModule", {}).get("officialTitle"),
                overall_status=protocol_section.get("statusModule", {}).get("overallStatus"),
                phase=design_result.get("phase"),
                study_type=design_result.get("study_type"),
                enrollment=design_result.get("enrollment"),
                start_date=protocol_section.get("statusModule", {}).get("startDateStruct", {}).get("date"),
                completion_date=protocol_section.get("statusModule", {}).get("completionDateStruct", {}).get("date"),
                sponsor=protocol_section.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
                conditions=[c for c in protocol_section.get("conditionsModule", {}).get("conditions", [])],
                interventions=[i.get("name", "") for i in protocol_section.get("armsInterventionsModule", {}).get("interventions", [])],
                primary_outcomes=[o.get("measure", "") for o in outcome_result.get("primary", [])],
                secondary_outcomes=[o.get("measure", "") for o in outcome_result.get("secondary", [])],
                eligibility_criteria=eligibility_text,
                inclusion_criteria=eligibility_result.get("inclusion", []),
                exclusion_criteria=eligibility_result.get("exclusion", []),
                provenance={
                    "source": "ClinicalTrials.gov MCP",
                    "endpoint": settings.mcp_server_url,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "eligibility": eligibility_result.get("provenance", {}),
                    "outcomes": outcome_result.get("provenance", {}),
                    "design": design_result.get("provenance", {}),
                },
            )

            extraction_result = ExtractionResult(
                nct_id=nct_id,
                extracted=extracted,
                raw_api=raw_api,
                confidence=0.8,
            )

            nct_log.info("verifying_extraction")
            verification_result = await self.verifier.execute({
                "extracted": extracted.model_dump(),
                "raw_api": raw_api,
            })
            extraction_result.verification_results = [verification_result]

            nct_log.info("auditing_constitution")
            audit_result = await self.constitutional_auditor.execute((extracted.model_dump(), {
                "enforcement_mode": "soft",
            }))
            extraction_result.constitutional_flags = audit_result.get("flags", [])

            nct_log.info("scoring_quality")
            quality_result = await self.quality_scorer.execute(extracted.model_dump())
            extraction_result.quality_score = quality_result.get("overall")

            return extraction_result

    def request_shutdown(self) -> None:
        self._shutdown_requested = True
        logger.warning("shutdown_requested")

    async def save_checkpoint(
        self,
        run_id: str,
        completed_batches: list[str],
        pending_ncts: list[str],
        failed_ncts: list[dict[str, str]],
        stats: dict[str, Any],
    ) -> Path:
        checkpoint = Checkpoint(
            run_id=run_id,
            completed_batches=completed_batches,
            pending_ncts=pending_ncts,
            failed_ncts=[(f["nct_id"], f["error"]) for f in failed_ncts],
            stats=stats,
            updated_at=datetime.now(timezone.utc),
        )

        checkpoint_file = self._checkpoint_path / f"run_{run_id}.json"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file.write_text(checkpoint.model_dump_json(indent=2))

        logger.info("checkpoint_saved", run_id=run_id, path=str(checkpoint_file))
        return checkpoint_file

    @classmethod
    async def load_checkpoint(cls, run_id: str, checkpoint_dir: Path) -> Checkpoint | None:
        checkpoint_file = checkpoint_dir / f"run_{run_id}.json"
        if not checkpoint_file.exists():
            return None
        return Checkpoint.model_validate_json(checkpoint_file.read_text())


class BatchProcessorFactory:
    @staticmethod
    def create(config: PipelineConfig, mcp_client: MCPClient) -> BatchProcessor:
        eligibility_parser = EligibilityParser(mode=config.eligibility_mode.value)
        outcome_extractor = OutcomeExtractor()
        design_classifier = DesignClassifier()
        verifier = Verifier()
        constitutional_auditor = ConstitutionalAuditor()
        quality_scorer = QualityScorer()

        return BatchProcessor(
            config=config,
            mcp_client=mcp_client,
            eligibility_parser=eligibility_parser,
            outcome_extractor=outcome_extractor,
            design_classifier=design_classifier,
            verifier=verifier,
            constitutional_auditor=constitutional_auditor,
            quality_scorer=quality_scorer,
        )