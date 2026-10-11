#!/usr/bin/env python
"""Evaluation runner for Clinical Trial Intelligence Agent."""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from clinical_trial_intelligence.config import PipelineConfig
from clinical_trial_intelligence.models import TrialRecord
from clinical_trial_intelligence.workflows.pipeline import run_pipeline


@dataclass
class FieldMetrics:
    field: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalResult:
    run_id: str
    total_records: int
    field_metrics: List[FieldMetrics]
    macro_f1: float
    micro_f1: float
    compliance_rate: float
    flags_count: int
    passed_threshold: bool


EVAL_FIELDS = [
    "brief_title", "official_title", "overall_status", "phase", "study_type",
    "enrollment", "start_date", "completion_date", "sponsor", "conditions",
    "interventions", "primary_outcomes", "secondary_outcomes", "eligibility_criteria",
    "inclusion_criteria", "exclusion_criteria", "data_quality_score",
    "extraction_confidence", "constitutional_flags"
]


def load_golden_set(golden_dir: Path) -> Dict[str, TrialRecord]:
    expected_path = golden_dir / "expected_output.csv"
    if not expected_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {expected_path}")

    df = pd.read_csv(expected_path)
    records = {}
    for _, row in df.iterrows():
        record = TrialRecord(**row.to_dict())
        records[record.nct_id] = record
    return records


def load_expected_flags(golden_dir: Path) -> Dict[str, List[str]]:
    flags_path = golden_dir / "expected_flags.yaml"
    if not flags_path.exists():
        return {}
    import yaml
    with open(flags_path) as f:
        return yaml.safe_load(f) or {}


def normalize_value(val: Any) -> str:
    if pd.isna(val) or val is None:
        return ""
    if isinstance(val, list):
        return "|".join(sorted(str(v).strip() for v in val))
    return str(val).strip()


def compare_records(predicted: TrialRecord, expected: TrialRecord) -> Dict[str, bool]:
    results = {}
    for field in EVAL_FIELDS:
        pred_val = normalize_value(getattr(predicted, field, None))
        exp_val = normalize_value(getattr(expected, field, None))
        results[field] = pred_val == exp_val
    return results


def compute_metrics(predictions: Dict[str, TrialRecord], ground_truth: Dict[str, TrialRecord]) -> EvalResult:
    field_results = {field: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for field in EVAL_FIELDS}
    total_compliance = 0
    total_flags = 0

    expected_flags = load_expected_flags(Path("tests/evals/golden"))

    for nct_id, pred_record in predictions.items():
        exp_record = ground_truth.get(nct_id)
        if not exp_record:
            continue

        comparisons = compare_records(pred_record, exp_record)
        for field, match in comparisons.items():
            pred_val = normalize_value(getattr(pred_record, field, None))
            exp_val = normalize_value(getattr(exp_record, field, None))

            if pred_val and exp_val:
                if match:
                    field_results[field]["tp"] += 1
                else:
                    field_results[field]["fp"] += 1
                    field_results[field]["fn"] += 1
            elif pred_val and not exp_val:
                field_results[field]["fp"] += 1
            elif not pred_val and exp_val:
                field_results[field]["fn"] += 1
            else:
                field_results[field]["tn"] += 1

        flags = pred_record.constitutional_flags.split(";") if pred_record.constitutional_flags else []
        total_flags += len([f for f in flags if f])

        exp_flags = expected_flags.get(nct_id, [])
        if all(ef in flags for ef in exp_flags):
            total_compliance += 1

    field_metrics = []
    for field, counts in field_results.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        field_metrics.append(FieldMetrics(field, precision, recall, f1, tp + fn))

    macro_f1 = sum(m.f1 for m in field_metrics) / len(field_metrics)

    total_tp = sum(c["tp"] for c in field_results.values())
    total_fp = sum(c["fp"] for c in field_results.values())
    total_fn = sum(c["fn"] for c in field_results.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    compliance_rate = total_compliance / len(ground_truth) if ground_truth else 1.0

    return EvalResult(
        run_id="eval",
        total_records=len(predictions),
        field_metrics=field_metrics,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        compliance_rate=compliance_rate,
        flags_count=total_flags,
        passed_threshold=False
    )


async def run_evaluation(golden_dir: Path, threshold_f1: float, threshold_compliance: float, output_path: Path):
    print(f"Loading golden dataset from {golden_dir}")
    ground_truth = load_golden_set(golden_dir)
    nct_codes = list(ground_truth.keys())

    print(f"Running pipeline on {len(nct_codes)} NCT codes...")
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

    output_csv = Path(result.output_path)
    if output_csv.exists():
        pred_df = pd.read_csv(output_csv)
        predictions = {}
        for _, row in pred_df.iterrows():
            record = TrialRecord(**row.to_dict())
            predictions[record.nct_id] = record
    else:
        predictions = {r.nct_id: r for r in result.records}

    print("Computing metrics...")
    eval_result = compute_metrics(predictions, ground_truth)
    eval_result.passed_threshold = eval_result.macro_f1 >= threshold_f1 and eval_result.compliance_rate >= threshold_compliance

    print(f"\n=== Evaluation Results ===")
    print(f"Total records: {eval_result.total_records}")
    print(f"Macro F1: {eval_result.macro_f1:.4f} (threshold: {threshold_f1})")
    print(f"Micro F1: {eval_result.micro_f1:.4f}")
    print(f"Compliance rate: {eval_result.compliance_rate:.4f} (threshold: {threshold_compliance})")
    print(f"Total flags: {eval_result.flags_count}")
    print(f"PASSED: {eval_result.passed_threshold}")

    print("\nPer-field metrics:")
    for m in eval_result.field_metrics:
        print(f"  {m.field:30s} P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} (n={m.support})")

    output_path.write_text(json.dumps(asdict(eval_result), indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    if not eval_result.passed_threshold:
        raise SystemExit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--golden", type=Path, default=Path("tests/evals/golden"), help="Golden dataset directory")
    parser.add_argument("--threshold-f1", type=float, default=0.85, help="Minimum macro F1 threshold")
    parser.add_argument("--threshold-compliance", type=float, default=1.0, help="Minimum compliance rate")
    parser.add_argument("--output", type=Path, default=Path("evals_results.json"), help="Output JSON path")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.golden, args.threshold_f1, args.threshold_compliance, args.output))


if __name__ == "__main__":
    main()