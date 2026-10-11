"""Unit tests for skills."""

import pytest
from clinical_trial_intelligence.skills import (
    NCTValidator,
    EligibilityParser,
    OutcomeExtractor,
    DesignClassifier,
    QualityScorer,
    Verifier,
    ConstitutionalAuditor,
)
from clinical_trial_intelligence.models import TrialRecord, ExtractionResult


class TestNCTValidator:
    @pytest.fixture
    def validator(self):
        return NCTValidator()

    @pytest.mark.asyncio
    async def test_valid_nct_format(self, validator):
        result = await validator.execute(["NCT00000001", "NCT12345678"])
        assert result.output["valid"] == ["NCT00000001", "NCT12345678"]
        assert result.output["invalid"] == []
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_invalid_nct_format(self, validator):
        result = await validator.execute(["NCT0000001", "INVALID", "NCT123456789"])
        assert len(result.output["invalid"]) == 3
        assert result.output["valid"] == []

    @pytest.mark.asyncio
    async def test_deduplication(self, validator):
        result = await validator.execute(["NCT00000001", "nct00000001", "NCT00000002"])
        assert len(result.output["valid"]) == 2
        assert "NCT00000001" in result.output["valid"]
        assert "NCT00000002" in result.output["valid"]
        assert len(result.output["duplicates"]) == 1


class TestEligibilityParser:
    @pytest.fixture
    def parser(self):
        return EligibilityParser()

    @pytest.mark.asyncio
    async def test_standard_format(self, parser):
        text = """Inclusion Criteria:
- Age 18-65
- Condition A diagnosis

Exclusion Criteria:
- Pregnancy
- Severe comorbidities"""
        result = await parser.execute(text)
        assert "Age 18-65" in result.output["inclusion"]
        assert "Condition A diagnosis" in result.output["inclusion"]
        assert "Pregnancy" in result.output["exclusion"]
        assert "Severe comorbidities" in result.output["exclusion"]
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_mixed_case_headers(self, parser):
        text = """inclusion criteria:
- Item 1
exclusion criteria:
- Item 2"""
        result = await parser.execute(text)
        assert "Item 1" in result.output["inclusion"]
        assert "Item 2" in result.output["exclusion"]

    @pytest.mark.asyncio
    async def test_no_headers(self, parser):
        text = "Age 18-65. No pregnancy. Condition A required."
        result = await parser.execute(text)
        assert result.output["raw_text"] == text


class TestOutcomeExtractor:
    @pytest.fixture
    def extractor(self):
        return OutcomeExtractor()

    @pytest.mark.asyncio
    async def test_primary_outcomes(self, extractor):
        study_record = {
            "protocolSection": {
                "outcomesModule": {
                    "primaryOutcomes": [
                        {"measure": "Primary outcome 1", "timeFrame": "12 weeks"},
                        {"measure": "Primary outcome 2", "timeFrame": "24 weeks"},
                    ],
                    "secondaryOutcomes": [{"measure": "Secondary outcome 1", "timeFrame": "12 weeks"}],
                }
            }
        }
        study_results = {}
        result = await extractor.execute({"study_record": study_record, "study_results": study_results})
        assert len(result.output["primary"]) == 2
        assert result.output["primary"][0]["measure"] == "Primary outcome 1"
        assert result.output["has_results"] is False

    @pytest.mark.asyncio
    async def test_with_results(self, extractor):
        study_record = {
            "protocolSection": {"outcomesModule": {"primaryOutcomes": [{"measure": "OS"}], "secondaryOutcomes": []}}
        }
        study_results = {"hasResults": True}
        result = await extractor.execute({"study_record": study_record, "study_results": study_results})
        assert result.output["has_results"] is True


class TestDesignClassifier:
    @pytest.fixture
    def classifier(self):
        return DesignClassifier()

    @pytest.mark.asyncio
    async def test_phase_classification(self, classifier):
        study_record = {
            "protocolSection": {
                "designModule": {
                    "phases": ["Phase 2"],
                    "studyType": "Interventional",
                    "designInfo": {
                        "allocation": "Randomized",
                        "interventionModel": "Parallel Assignment",
                        "maskingInfo": {"masking": "Double", "whoMasked": ["Participant", "Investigator"]},
                        "primaryPurpose": "Treatment",
                    },
                    "enrollmentInfo": {"count": 100},
                    "arms": [{"type": "Experimental"}, {"type": "Placebo"}],
                }
            }
        }
        result = await classifier.execute(study_record)
        assert result.output["study_type"] == "Interventional"
        assert result.output["enrollment"] == 100


class TestQualityScorer:
    @pytest.fixture
    def scorer(self):
        return QualityScorer()

    @pytest.mark.asyncio
    async def test_high_quality_record(self, scorer):
        record = TrialRecord(
            nct_id="NCT00000001",
            brief_title="Test",
            official_title="Test",
            overall_status="Completed",
            phase="Phase 2",
            study_type="Interventional",
            enrollment=100,
            start_date="2020-01-01",
            completion_date="2022-01-01",
            sponsor="Sponsor",
            conditions=["Cond"],
            interventions=["Drug"],
            primary_outcomes=["Outcome"],
            secondary_outcomes=["Outcome"],
            eligibility_criteria="Crit",
            inclusion_criteria=["Incl"],
            exclusion_criteria=["Excl"],
            data_quality_score=0.9,
            extraction_confidence=0.9,
            constitutional_flags=[],
            source_timestamp="2024-01-01T00:00:00Z",
            provenance={},
        )
        result = await scorer.execute(record)
        assert 0.0 <= result.output["overall"] <= 1.0
        assert "completeness" in result.output["breakdown"]


class TestVerifier:
    @pytest.fixture
    def verifier(self):
        return Verifier()

    @pytest.mark.asyncio
    async def test_matching_fields(self, verifier):
        extraction = ExtractionResult(
            nct_id="NCT00000001",
            extracted={"phase": "Phase 2", "enrollment": 100, "nct_id": "NCT00000001"},
            raw_api={"protocolSection": {"designModule": {"phases": ["Phase 2"], "enrollmentInfo": {"count": 100}}}},
            confidence=0.9,
            provenance={},
        )
        result = await verifier.execute(extraction)
        assert result.output["overall_pass"] is True
        assert result.output["mismatch_count"] == 0


class TestConstitutionalAuditor:
    @pytest.fixture
    def auditor(self):
        return ConstitutionalAuditor()

    @pytest.mark.asyncio
    async def test_audit_runs(self, auditor):
        record = TrialRecord(
            nct_id="NCT00000001",
            brief_title="Test",
            official_title="Test",
            overall_status="Completed",
            phase="Phase 2",
            study_type="Interventional",
            enrollment=100,
            start_date="2020-01-01",
            completion_date="2022-01-01",
            sponsor="Sponsor",
            conditions=["Cond"],
            interventions=["Drug"],
            primary_outcomes=["Outcome"],
            secondary_outcomes=["Outcome"],
            eligibility_criteria="Crit",
            inclusion_criteria=["Incl"],
            exclusion_criteria=["Excl"],
            data_quality_score=0.9,
            extraction_confidence=0.9,
            constitutional_flags=[],
            source_timestamp="2024-01-01T00:00:00Z",
            provenance={},
        )
        result = await auditor.execute((record.model_dump(), {}))
        assert isinstance(result.output["flags"], list)
        assert 0.0 <= result.output["compliance_score"] <= 1.0