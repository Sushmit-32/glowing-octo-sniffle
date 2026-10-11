#!/usr/bin/env python
"""Drift detector for scheduled evaluations."""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from clinical_trial_intelligence.config import PipelineConfig
from clinical_trial_intelligence.mcp_client import ClinicalTrialsMCPClient
from clinical_trial_intelligence.models import TrialRecord
from clinical_trial_intelligence.workflows.pipeline import run_pipeline


@dataclass
class DriftReport:
    timestamp: str
    nct_count: int
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    drift_detected: bool
    drift_details: Dict[str, Any]


async def fetch_random_ncts(client: ClinicalTrialsMCPClient, count: int) -> List[str]:
    result = await client.search_studies(query="", page_size=min(count * 3, 200))
    studies = result.get("studies", [])
    nct_ids = [s.get("protocolSection", {}).get("identificationModule", {}).get("nctId") for s in studies]
    nct_ids = [n for n in nct_ids if n]
    return random.sample(nct_ids, min(count, len(nct_ids)))


async def run_drift_check(nct_count: int, baseline_path: Path, output_path: Path):
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        baseline = {"macro_f1": 0.85, "compliance_rate": 1.0, "mean_quality_score": 0.87, "mean_extraction_confidence": 0.85, "flag_rate": 0.15}

    client = ClinicalTrialsMCPClient()
    nct_codes = await fetch_random_ncts(client, nct_count)
    await client.close()

    print(f"Testing {len(nct_codes)} random NCTs...")

    config = PipelineConfig(
        batch_size=10,
        parallel_workers=3,
        mcp_server_url="https://clinicaltrials.caseyjhand.com/mcp",
        checkpoint_dir=Path("data/checkpoints"),
        csv_output_dir=Path("output"),
        episodic_log_dir=Path("data/episodic"),
        constitution_path=Path("constitution/constitution.yaml"),
        enforcement_mode="soft"
    )

    result = await run_pipeline(nct_codes, config)

    quality_scores = [r.data_quality_score for r in result.records if r.data_quality_score is not None]
    extraction_confidences = [r.extraction_confidence for r in result.records if r.extraction_confidence is not None]
    compliance_count = sum(1 for r in result.records if not r.constitutional_flags or r.constitutional_flags.strip() == "")

    current_metrics = {
        "mean_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
        "mean_extraction_confidence": sum(extraction_confidences) / len(extraction_confidences) if extraction_confidences else 0,
        "compliance_rate": compliance_count / len(result.records) if result.records else 1.0,
        "total_records": len(result.records),
        "flagged_records": len(result.records) - compliance_count
    }

    drift_detected = False
    drift_details = {}

    for key, baseline_val in baseline.items():
        if key in current_metrics and baseline_val > 0:
            current_val = current_metrics[key]
            relative_change = (baseline_val - current_val) / baseline_val
            drift_details[key] = {
                "baseline": baseline_val,
                "current": current_val,
                "relative_change": relative_change,
                "threshold": 0.10
            }
            if relative_change > 0.10:
                drift_detected = True

    report = DriftReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        nct_count=len(nct_codes),
        current_metrics=current_metrics,
        baseline_metrics=baseline,
        drift_detected=drift_detected,
        drift_details=drift_details
    )

    output_path.write_text(json.dumps(asdict(report), indent=2, default=str))
    print(f"Drift report saved to {output_path}")
    print(f"Drift detected: {drift_detected}")

    if drift_detected:
        print("WARNING: Significant drift detected!")
        for key, details in drift_details.items():
            if details["relative_change"] > 0.10:
                print(f"  {key}: {details['baseline']:.4f} -> {details['current']:.4f} ({details['relative_change']:.1%} degradation)")

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run drift detection")
    parser.add_argument("--count", type=int, default=50, help="Number of random NCTs to test")
    parser.add_argument("--baseline", type=Path, default=Path("tests/evals/golden/baseline_metrics.json"), help="Baseline metrics JSON")
    parser.add_argument("--output", type=Path, default=Path("drift_report.json"), help="Output JSON path")
    args = parser.parse_args()

    import asyncio
    asyncio.run(run_drift_check(args.count, args.baseline, args.output))


if __name__ == "__main__":
    main()