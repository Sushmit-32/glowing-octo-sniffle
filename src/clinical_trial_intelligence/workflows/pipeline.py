from __future__ import annotations

import signal
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opentelemetry.trace import SpanKind, Status, StatusCode
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from clinical_trial_intelligence.config import settings
from clinical_trial_intelligence.mcp_client import MCPClient, create_mcp_client
from clinical_trial_intelligence.models import (
    BatchResult,
    PipelineConfig,
    PipelineResult,
    TrialRecord,
)
from clinical_trial_intelligence.observability import get_logger, set_batch_id, set_run_id, setup_observability
from clinical_trial_intelligence.skills import NCTValidator
from clinical_trial_intelligence.skills.base import tracer
from clinical_trial_intelligence.workflows.batch_processor import BatchProcessor, BatchProcessorFactory

logger = get_logger(__name__)


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._mcp_client: MCPClient | None = None
        self._batch_processor: BatchProcessor | None = None
        self._shutdown_requested = False
        self._run_id = config.run_id or uuid.uuid4().hex[:12]
        self._pipeline_result: PipelineResult | None = None

    async def run_pipeline(self, nct_codes: list[str]) -> PipelineResult:
        set_run_id(self._run_id)

        with tracer.start_as_current_span("workflow.pipeline.run", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("pipeline.run_id", self._run_id)
            span.set_attribute("pipeline.input_count", len(nct_codes))

            log = logger.bind(run_id=self._run_id, input_count=len(nct_codes))
            log.info("pipeline_start", nct_codes=nct_codes)

            self._pipeline_result = PipelineResult(
                run_id=self._run_id,
                config=self.config,
                start_time=datetime.now(timezone.utc),
                total_trials=0,
            )

            try:
                validated = await self._validate_ncts(nct_codes)
                self._pipeline_result.total_trials = len(validated["valid"])

                if not validated["valid"]:
                    log.warning("no_valid_ncts", invalid_count=len(validated["invalid"]))
                    self._finalize_pipeline()
                    return self._pipeline_result

                completed_batches: list[str] = []
                pending_ncts = validated["valid"].copy()
                failed_ncts: list[dict[str, str]] = []
                all_batch_results: list[BatchResult] = []

                if self.config.resume:
                    checkpoint = await BatchProcessor.load_checkpoint(self._run_id, Path(self.config.checkpoint_dir))
                    if checkpoint:
                        completed_batches = checkpoint.completed_batches
                        pending_ncts = checkpoint.pending_ncts
                        failed_ncts = [{"nct_id": n, "error": e} for n, e in checkpoint.failed_ncts]
                        log.info(
                            "checkpoint_loaded",
                            completed_batches=len(completed_batches),
                            pending_ncts=len(pending_ncts),
                            failed_ncts=len(failed_ncts),
                        )

                batches = self._chunk_batches(pending_ncts)
                log.info("batches_created", batch_count=len(batches))

                async with create_mcp_client() as mcp_client:
                    self._mcp_client = mcp_client
                    self._batch_processor = BatchProcessorFactory.create(self.config, mcp_client)

                    self._setup_signal_handlers()

                    progress = Progress(
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TextColumn("•"),
                        TimeElapsedColumn(),
                        TextColumn("•"),
                        TimeRemainingColumn(),
                    )

                    with progress:
                        batch_task = progress.add_task("Processing batches", total=len(batches))

                        for i, batch_ncts in enumerate(batches):
                            if self._shutdown_requested:
                                log.warning("shutdown_during_batch_processing")
                                break

                            batch_id = f"batch_{i:04d}_{uuid.uuid4().hex[:8]}"

                            if batch_id in completed_batches:
                                log.info("batch_skipped", batch_id=batch_id)
                                progress.advance(batch_task)
                                continue

                            set_batch_id(batch_id)

                            batch_result = await self._batch_processor.process_batch(batch_ncts, batch_id, self._run_id)

                            all_batch_results.append(batch_result)
                            completed_batches.append(batch_id)
                            failed_ncts.extend(batch_result.failed)

                            pending_ncts = [nct for nct in pending_ncts if nct not in batch_ncts]

                            await self._save_checkpoint(completed_batches, pending_ncts, failed_ncts)

                            progress.advance(batch_task)
                            progress.update(
                                batch_task,
                                description=f"Completed {i + 1}/{len(batches)} batches",
                            )

                            log.info(
                                "batch_completed",
                                batch_id=batch_id,
                                success_count=batch_result.success_count,
                                failure_count=batch_result.failure_count,
                            )

                            if self._shutdown_requested:
                                break

                        self._pipeline_result.batches = all_batch_results
                        self._pipeline_result.successful_trials = sum(b.success_count for b in all_batch_results)
                        self._pipeline_result.failed_trials = sum(b.failure_count for b in all_batch_results)

                        await self._generate_compliance_report(all_batch_results)
                        await self._export_results(all_batch_results)
                        await self._save_final_checkpoint(all_batch_results, failed_ncts)

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("pipeline.error", type(e).__name__)
                log.exception("pipeline_failed", error=str(e))
                raise
            else:
                span.set_status(Status(StatusCode.OK))
                log.info("pipeline_complete")
            finally:
                self._finalize_pipeline()

        return self._pipeline_result

    async def _validate_ncts(self, nct_codes: list[str]) -> dict[str, Any]:
        with tracer.start_as_current_span("skill.nct_validator", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("pipeline.run_id", self._run_id)
            validator = NCTValidator(mcp_client=self._mcp_client)
            result = await validator.execute(nct_codes)
            return result.output

    def _chunk_batches(self, nct_ids: list[str]) -> list[list[str]]:
        batch_size = self.config.batch_size
        return [nct_ids[i:i + batch_size] for i in range(0, len(nct_ids), batch_size)]

    async def _save_checkpoint(
        self,
        completed_batches: list[str],
        pending_ncts: list[str],
        failed_ncts: list[dict[str, str]],
    ) -> None:
        if not self.config.checkpoint_enabled or not self._batch_processor:
            return

        stats = {
            "completed_batches": len(completed_batches),
            "pending_ncts": len(pending_ncts),
            "failed_ncts": len(failed_ncts),
            "total_batches": len(completed_batches) + len(pending_ncts),
        }

        await self._batch_processor.save_checkpoint(
            self._run_id,
            completed_batches,
            pending_ncts,
            failed_ncts,
            stats,
        )

    async def _save_final_checkpoint(
        self,
        all_batch_results: list[BatchResult],
        failed_ncts: list[dict[str, str]],
    ) -> None:
        if not self.config.checkpoint_enabled or not self._batch_processor:
            return

        stats = {
            "completed_batches": len(all_batch_results),
            "total_trials": self._pipeline_result.total_trials if self._pipeline_result else 0,
            "successful_trials": self._pipeline_result.successful_trials if self._pipeline_result else 0,
            "failed_trials": self._pipeline_result.failed_trials if self._pipeline_result else 0,
        }

        await self._batch_processor.save_checkpoint(
            self._run_id,
            [b.batch_id for b in all_batch_results],
            [],
            failed_ncts,
            stats,
        )

    async def _generate_compliance_report(self, all_batch_results: list[BatchResult]) -> None:
        from clinical_trial_intelligence.skills import ConstitutionalAuditor

        all_flags = []
        for batch in all_batch_results:
            all_flags.extend(batch.flags)

        auditor = ConstitutionalAuditor()
        report = await auditor.generate_report(all_flags, self._run_id)

        report.run_id = self._run_id
        report.trials_reviewed = len(all_batch_results)
        report.enforcement_mode = self.config.enforcement_mode

        self._pipeline_result.compliance_report = report

        logger.info(
            "compliance_report_generated",
            run_id=self._run_id,
            total_flags=report.total_flags,
            trials_reviewed=report.trials_reviewed,
        )

    async def _export_results(self, all_batch_results: list[BatchResult]) -> None:
        with tracer.start_as_current_span("workflow.export_results", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("pipeline.run_id", self._run_id)

            records: list[TrialRecord] = []
            for batch in all_batch_results:
                for result in batch.successful:
                    if result.extracted:
                        records.append(result.extracted)

            if not records:
                logger.warning("no_records_to_export")
                return

            from clinical_trial_intelligence.skills import CSVExporter, EpisodicLogExporter

            self.config.csv_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.config.csv_output_dir / f"trials_{self._run_id}.csv"

            csv_exporter = CSVExporter()
            await csv_exporter.export(records, output_path)

            self.config.episodic_log_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = self.config.episodic_log_dir / "runs.parquet"

            episodic_exporter = EpisodicLogExporter()
            episodic_records = [r.model_dump() for r in records]
            await episodic_exporter.append(episodic_records, parquet_path)

            if self._pipeline_result:
                self._pipeline_result.output_path = output_path

            logger.info("results_exported", run_id=self._run_id, path=str(output_path), count=len(records))

    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum, frame):
            self._shutdown_requested = True
            logger.warning("shutdown_signal_received", signal=signum)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _finalize_pipeline(self) -> None:
        if self._pipeline_result:
            self._pipeline_result.end_time = datetime.now(timezone.utc)
            logger.info(
                "pipeline_finalized",
                run_id=self._run_id,
                total=self._pipeline_result.total_trials,
                successful=self._pipeline_result.successful_trials,
                failed=self._pipeline_result.failed_trials,
            )


async def run_pipeline(nct_codes: list[str], config: PipelineConfig) -> PipelineResult:
    setup_observability()
    pipeline = Pipeline(config)
    return await pipeline.run_pipeline(nct_codes)