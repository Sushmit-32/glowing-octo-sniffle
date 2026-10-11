from __future__ import annotations

import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from clinical_trial_intelligence.config import PipelineConfig
from clinical_trial_intelligence.workflows.pipeline import run_pipeline


class RunTrialAnalysisInput(BaseModel):
    nct_codes: list[str] = Field(..., description="List of NCT IDs to analyze")
    output_format: str = Field(default="csv", description="Output format: csv or json")
    config_overrides: dict[str, Any] = Field(default_factory=dict, description="Pipeline config overrides")


class RunTrialAnalysisOutput(BaseModel):
    run_id: str
    csv_path: str | None = None
    compliance_report_path: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


mcp = FastMCP("clinical-trial-intelligence")

session_state = {
    "preferences": {"batch_size": 10, "output_format": "csv"},
    "history": [],
}


@mcp.tool()
async def run_trial_analysis(input: RunTrialAnalysisInput) -> RunTrialAnalysisOutput:
    run_id = uuid.uuid4().hex[:12]

    config = PipelineConfig(
        run_id=run_id,
        batch_size=input.config_overrides.get("batch_size", session_state["preferences"].get("batch_size", 10)),
        parallel_workers=input.config_overrides.get("parallel_workers", 3),
        checkpoint_dir=input.config_overrides.get("checkpoint_dir", "data/checkpoints"),
        csv_output_dir=input.config_overrides.get("csv_output_dir", "output"),
        episodic_log_dir=input.config_overrides.get("episodic_log_dir", "data/episodic"),
        constitution_path=input.config_overrides.get("constitution_path", "constitution/constitution.yaml"),
        enforcement_mode=input.config_overrides.get("enforcement_mode", "soft"),
        checkpoint_enabled=input.config_overrides.get("checkpoint_enabled", True),
        resume=input.config_overrides.get("resume", False),
    )

    result = await run_pipeline(input.nct_codes, config)

    session_state["history"].append({
        "run_id": run_id,
        "nct_codes": input.nct_codes,
        "timestamp": result.start_time.isoformat() if result.start_time else None,
    })
    session_state["history"] = session_state["history"][-10:]

    return RunTrialAnalysisOutput(
        run_id=run_id,
        csv_path=str(result.output_path) if result.output_path else None,
        compliance_report_path=None,
        stats={
            "total": result.total_trials,
            "successful": result.successful_trials,
            "failed": result.failed_trials,
        },
    )


@mcp.tool()
async def get_session_state() -> dict[str, Any]:
    return session_state


@mcp.tool()
async def update_preferences(preferences: dict[str, Any]) -> dict[str, Any]:
    session_state["preferences"].update(preferences)
    return session_state["preferences"]


if __name__ == "__main__":
    mcp.run()