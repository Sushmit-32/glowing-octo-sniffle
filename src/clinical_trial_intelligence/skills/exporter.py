from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from clinical_trial_intelligence.models import TrialRecord
from clinical_trial_intelligence.observability import get_logger
from clinical_trial_intelligence.skills.base import Skill, SkillResult, traced_skill

logger = get_logger(__name__)

CSV_COLUMNS = [
    "nct_id", "brief_title", "official_title", "overall_status", "phase", "study_type",
    "enrollment", "start_date", "completion_date", "sponsor", "conditions", "interventions",
    "primary_outcomes", "secondary_outcomes", "eligibility_criteria", "inclusion_criteria",
    "exclusion_criteria", "data_quality_score", "extraction_confidence",
    "constitutional_flags", "source_timestamp",
]

PROVENANCE_COLUMNS = [f"{c}_provenance" for c in CSV_COLUMNS]
ALL_COLUMNS = CSV_COLUMNS + PROVENANCE_COLUMNS


class CSVExporter(Skill[tuple[list[TrialRecord], Path], Path]):
    name = "csv_exporter"
    version = "1.0.0"

    def __init__(self, include_provenance: bool = True):
        self.include_provenance = include_provenance

    @traced_skill()
    async def execute(self, input: tuple[list[TrialRecord], Path]) -> SkillResult[Path]:
        records, output_path = input
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not records:
            logger.warning("export_empty_records", path=str(output_path))
            output_path.write_text("")
            return SkillResult(output=output_path, confidence=1.0, provenance=self._create_provenance(record_count=0))

        field_names = self._get_field_names(records)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_names, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(self._record_to_row(record))

        logger.info("csv_export_complete", path=str(output_path), record_count=len(records))
        return SkillResult(
            output=output_path,
            confidence=1.0,
            provenance=self._create_provenance(record_count=len(records), fields=field_names),
        )

    def _get_field_names(self, records: list[TrialRecord]) -> list[str]:
        return ALL_COLUMNS if self.include_provenance else CSV_COLUMNS

    def _record_to_row(self, record: TrialRecord) -> dict[str, Any]:
        data = record.model_dump(exclude={"provenance"})

        for key, value in data.items():
            if isinstance(value, list):
                data[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, datetime):
                data[key] = value.isoformat()

        if self.include_provenance and record.provenance:
            prov = record.provenance
            for col in CSV_COLUMNS:
                prov_key = f"{col}_raw" if col != "nct_id" else "source"
                data[f"{col}_provenance"] = prov.get(prov_key, prov.get(col, ""))

        return data


class EpisodicLogExporter(Skill[tuple[list[dict[str, Any]], Path], Path]):
    name = "episodic_log_exporter"
    version = "1.0.0"

    @traced_skill()
    async def execute(self, input: tuple[list[dict[str, Any]], Path]) -> SkillResult[Path]:
        records, output_path = input
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        if output_path.exists():
            existing = pd.read_parquet(output_path)
            df = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
        else:
            df = pd.DataFrame(records)

        df.to_parquet(output_path, index=False)
        logger.info("episodic_log_append", path=str(output_path), record_count=len(records))

        return SkillResult(
            output=output_path,
            confidence=1.0,
            provenance=self._create_provenance(record_count=len(records)),
        )