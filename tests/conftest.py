"""Pytest configuration and fixtures."""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil

from clinical_trial_intelligence.models import PipelineConfig, TrialRecord


@pytest.fixture
def temp_dir():
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    return PipelineConfig(
        batch_size=2,
        parallel_workers=1,
        mcp_server_url="https://clinicaltrials.caseyjhand.com/mcp",
        checkpoint_dir=temp_dir / "checkpoints",
        csv_output_dir=temp_dir / "output",
        episodic_log_dir=temp_dir / "episodic",
        constitution_path=Path("constitution/constitution.yaml"),
        enforcement_mode="soft",
        request_timeout_ms=10000,
        max_retries=1,
    )


@pytest.fixture
def sample_trial_record():
    return TrialRecord(
        nct_id="NCT00000001",
        brief_title="Test Study",
        official_title="Official Test Study Title",
        overall_status="Completed",
        phase="Phase 2",
        study_type="Interventional",
        enrollment=100,
        start_date="2020-01-15",
        completion_date="2022-06-30",
        sponsor="Test Sponsor",
        conditions=["Condition A", "Condition B"],
        interventions=["Drug X", "Placebo"],
        primary_outcomes=["Primary outcome at 12 weeks"],
        secondary_outcomes=["Secondary outcome 1", "Secondary outcome 2"],
        eligibility_criteria="Inclusion: Age 18-65. Exclusion: Pregnancy.",
        inclusion_criteria=["Age 18-65", "Condition A diagnosis"],
        exclusion_criteria=["Pregnancy", "Severe comorbidities"],
        data_quality_score=0.95,
        extraction_confidence=0.92,
        constitutional_flags=[],
        source_timestamp="2024-01-15T10:30:00Z",
        provenance={},
    )


@pytest.fixture
def sample_nct_codes():
    return ["NCT00000001", "NCT00000002", "NCT00000003"]


@pytest_asyncio.fixture
async def mock_mcp_client():
    from unittest.mock import AsyncMock, MagicMock
    from clinical_trial_intelligence.mcp_client import MCPClient

    client = MagicMock(spec=MCPClient)
    client.get_study_record = AsyncMock()
    client.get_study_results = AsyncMock()
    client.search_studies = AsyncMock()
    client.close = AsyncMock()
    return client