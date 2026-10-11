from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Generic, TypeVar

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from clinical_trial_intelligence.observability import get_logger, record_skill_confidence

T = TypeVar("T")
R = TypeVar("R")

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class SkillResult(Generic[R]):
    output: R
    confidence: float
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


class Skill(ABC, Generic[T, R]):
    name: str = "base_skill"
    version: str = "1.0.0"

    @abstractmethod
    async def execute(self, input: T) -> SkillResult[R]:
        pass

    def _create_provenance(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "skill": self.name,
            "version": self.version,
            "timestamp": time.time(),
            "run_id": uuid.uuid4().hex[:8],
            **kwargs,
        }

    def _record_confidence(self, confidence: float, nct_id: str | None = None) -> None:
        record_skill_confidence(confidence, self.name, nct_id or "unknown")


def traced_skill(skill_name: str | None = None):
    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            name = skill_name or getattr(self, "name", func.__name__)

            with tracer.start_as_current_span(f"skill.{name}", kind=SpanKind.INTERNAL) as span:
                span.set_attribute("skill.name", name)
                span.set_attribute("skill.version", getattr(self, "version", "1.0.0"))

                start_time = time.monotonic()
                log = get_logger(func.__module__)

                log.info("skill_start", input_type=type(args[0]).__name__ if args else "none")

                try:
                    result = await func(self, *args, **kwargs)
                    duration_ms = (time.monotonic() - start_time) * 1000

                    span.set_attribute("skill.duration_ms", duration_ms)
                    span.set_attribute("skill.confidence", result.confidence)
                    span.set_status(Status(StatusCode.OK))

                    log.info("skill_complete", confidence=result.confidence, duration_ms=duration_ms, provenance=result.provenance)

                    if hasattr(result, "confidence"):
                        record_skill_confidence(result.confidence, name, getattr(args[0], "nct_id", "unknown") if args else "unknown")

                    return result

                except Exception as e:
                    duration_ms = (time.monotonic() - start_time) * 1000
                    span.set_attribute("skill.duration_ms", duration_ms)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    log.exception("skill_failed", duration_ms=duration_ms, error=str(e))
                    raise

        return wrapper

    return decorator