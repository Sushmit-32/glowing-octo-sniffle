from __future__ import annotations

import re

from clinical_trial_intelligence.config import settings
from clinical_trial_intelligence.observability import get_logger
from clinical_trial_intelligence.skills.base import Skill, SkillResult, traced_skill

logger = get_logger(__name__)


class EligibilityParser(Skill[str, dict[str, Any]]):
    name = "eligibility_parser"
    version = "1.0.0"

    def __init__(self, mode: str = "regex"):
        self.mode = mode

    @traced_skill()
    async def execute(self, input: str) -> SkillResult[dict[str, Any]]:
        if self.mode == "llm" and settings.anthropic_api_key:
            return await self._parse_llm(input)
        return await self._parse_regex(input)

    async def _parse_regex(self, text: str) -> SkillResult[dict[str, Any]]:
        inclusion: list[str] = []
        exclusion: list[str] = []

        text = re.sub(r"\s+", " ", text.strip())

        inclusion_patterns = [
            r"(?:inclusion criteria?)[:\s]*(.*?)(?=(?:exclusion criteria?)|$)",
            r"(?:inclusion)[:\s]*(.*?)(?=(?:exclusion)|$)",
        ]

        exclusion_patterns = [
            r"(?:exclusion criteria?)[:\s]*(.*?)(?=(?:inclusion criteria?)|$)",
            r"(?:exclusion)[:\s]*(.*?)(?=(?:inclusion)|$)",
        ]

        for pattern in inclusion_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                inclusion.extend(self._extract_items(match.group(1)))
                break

        for pattern in exclusion_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                exclusion.extend(self._extract_items(match.group(1)))
                break

        if not inclusion and not exclusion:
            sentences = re.split(r"[.!?]+", text)
            for sent in sentences:
                sent = sent.strip()
                if any(kw in sent.lower() for kw in ["must", "required", "age", "diagnos"]):
                    inclusion.append(sent)
                elif any(kw in sent.lower() for kw in ["not", "exclude", "pregnan", "unable"]):
                    exclusion.append(sent)

        confidence = 0.8 if (inclusion or exclusion) else 0.3

        return SkillResult(
            output={
                "inclusion": inclusion,
                "exclusion": exclusion,
                "raw_text": text,
                "parser_mode": "regex",
            },
            confidence=confidence,
            provenance=self._create_provenance(inclusion_count=len(inclusion), exclusion_count=len(exclusion)),
        )

    async def _parse_llm(self, text: str) -> SkillResult[dict[str, Any]]:
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

            prompt = (
                "Parse the following clinical trial eligibility criteria into "
                "inclusion and exclusion criteria.\n"
                'Return a JSON object with two keys: "inclusion" and "exclusion", '
                "each containing a list of strings.\n\n"
                f"Criteria text:\n{text}\n\nJSON response:"
            )

            response = await client.messages.create(
                model=settings.anthropic_model,
                max_tokens=1000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            import json
            result = json.loads(response.content[0].text)

            return SkillResult(
                output={
                    "inclusion": result.get("inclusion", []),
                    "exclusion": result.get("exclusion", []),
                    "raw_text": text,
                    "parser_mode": "llm",
                },
                confidence=0.9,
                provenance=self._create_provenance(parser_mode="llm"),
            )

        except Exception as e:
            logger.warning("llm_parse_failed", error=str(e))
            return await self._parse_regex(text)

    def _extract_items(self, text: str) -> list[str]:
        items = []
        for line in re.split(r"[\n\r;•\-\*]+", text):
            line = line.strip()
            line = re.sub(r"^\d+[.)]\s*", "", line)
            if line and len(line) > 3:
                items.append(line)
        return items