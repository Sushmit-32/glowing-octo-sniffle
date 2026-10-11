"""Integration tests for pipeline components."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from clinical_trial_intelligence.models import PipelineConfig, Checkpoint
from clinical_trial_intelligence.mcp_client import MCPClient
from clinical_trial_intelligence.workflows.batch_processor import BatchProcessor, BatchProcessorFactory
from clinical_trial_intelligence.workflows.pipeline import run_pipeline


class TestMCPClient:
    @pytest.fixture
    def client(self):
        return MCPClient()

    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        async with client as c:
            assert c is not None


class TestBatchProcessor:
    @pytest.fixture
    def config(self, tmp_path):
        return PipelineConfig(
            batch_size=2,
            parallel_workers=1,
            mcp_server_url="https://clinicaltrials.caseyjhand.com/mcp",
            checkpoint_dir=tmp_path / "checkpoints",
            csv_output_dir=tmp_path / "output",
            episodic_log_dir=tmp_path / "episodic",
            constitution_path=Path("constitution/constitution.yaml"),
            enforcement_mode="soft",
            request_timeout_ms=10000,
            max_retries=1,
        )

    @pytest.fixture
    def skills(self):
        from clinical_trial_intelligence.skills import (
            ConstitutionalAuditor, DesignClassifier, EligibilityParser,
            NCTValidator, OutcomeExtractor, QualityScorer, Verifier
        )
        return {
            "validator": NCTValidator(),
            "eligibility_parser": EligibilityParser(),
            "outcome_extractor": OutcomeExtractor(),
            "design_classifier": DesignClassifier(),
            "quality_scorer": QualityScorer(),
            "verifier": Verifier(),
            "constitutional_auditor": ConstitutionalAuditor(),
        }

    @pytest.mark.asyncio
    async def test_batch_processor_creation(self, config, skills):
        processor = BatchProcessorFactory.create(config, MagicMock())
        assert processor.config == config


class TestPipeline:
    @pytest.fixture
    def config(self, tmp_path):
        return PipelineConfig(
            batch_size=2,
            parallel_workers=1,
            mcp_server_url="https://clinicaltrials.caseyjhand.com/mcp",
            checkpoint_dir=tmp_path / "checkpoints",
            csv_output_dir=tmp_path / "output",
            episodic_log_dir=tmp_path / "episodic",
            constitution_path=Path("constitution/constitution.yaml"),
            enforcement_mode="soft",
            request_timeout_ms=10000,
            max_retries=1,
        )

    @pytest.mark.asyncio
    async def test_pipeline_validation_only(self, config, tmp_path):
        config.checkpoint_dir = tmp_path / "checkpoints"
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        nct_codes = ["NCT00000001", "NCT00000002"]

        with patch("clinical_trial_intelligence.workflows.pipeline.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with patch("clinical_trial_intelligence.workflows.pipeline.NCTValidator") as mock_validator_class:
                mock_validator = AsyncMock()
                mock_validator.execute.return_value = MagicMock(
                    output=MagicMock(valid=nct_codes, invalid=[], duplicates=[]),
                    confidence=1.0
                )
                mock_validator_class.return_value = mock_validator

                result = await run_pipeline(nct_codes, config)
                assert result is not None


class TestCheckpointResume:
    @pytest.fixture
    def config(self, tmp_path):
        return PipelineConfig(
            batch_size=2,
            parallel_workers=1,
            mcp_server_url="https://clinicaltrials.caseyjhand.com/mcp",
            checkpoint_dir=tmp_path / "checkpoints",
            csv_output_dir=tmp_path / "output",
            episodic_log_dir=tmp_path / "episodic",
            constitution_path=Path("constitution/constitution.yaml"),
            enforcement_mode="soft",
            request_timeout_ms=10000,
            max_retries=1,
        )

    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, config, tmp_path):
        checkpoint = Checkpoint(
            run_id="test_run",
            completed_batches=["batch_0000_abc123", "batch_0001_def456"],
            pending_ncts=["NCT00000003", "NCT00000004"],
            failed_ncts=[("NCT00000005", "timeout")],
            stats={"success": 4, "failed": 1, "flagged": 0}
        )

        checkpoint_dir = config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "run_test_run.json"
        checkpoint_path.write_text(checkpoint.model_dump_json(indent=2))

        loaded = Checkpoint.model_validate_json(checkpoint_path.read_text())
        assert loaded.run_id == "test_run"
        assert loaded.completed_batches == ["batch_0000_abc123", "batch_0001_def456"]
        assert loaded.pending_ncts == ["NCT00000003", "NCT00000004"]
        assert loaded.failed_ncts == [("NCT00000005", "timeout")]