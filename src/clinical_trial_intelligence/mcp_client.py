from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from typing_extensions import Self

from clinical_trial_intelligence.config import settings
from clinical_trial_intelligence.models import ExtractionResult, TrialRecord

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class MCPError(Exception):
    def __init__(self, message: str, code: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class NotFoundError(MCPError):
    def __init__(self, nct_id: str):
        super().__init__(f"Study not found: {nct_id}", "NOT_FOUND", {"nct_id": nct_id})


class RateLimitError(MCPError):
    def __init__(self, retry_after: float | None = None):
        super().__init__("Rate limited", "RATE_LIMITED", {"retry_after": retry_after})


class MalformedResponseError(MCPError):
    def __init__(self, message: str = "Malformed response from MCP server"):
        super().__init__(message, "MALFORMED_RESPONSE")


class MCPClient:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._request_semaphore = asyncio.Semaphore(1)
        self._last_request_time = 0.0
        self._min_request_interval = 1.0

    async def __aenter__(self) -> Self:
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _ensure_client(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=settings.mcp_server_url,
                timeout=httpx.Timeout(settings.request_timeout_ms / 1000),
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            await self._initialize_session()

    async def _initialize_session(self) -> None:
        try:
            response = await self._client.post(
                "/sessions",
                json={"capabilities": {}},
                timeout=10,
            )
            if response.status_code == 200:
                self._session_id = response.json().get("session_id")
                logger.info("mcp_session_initialized", session_id=self._session_id)
        except Exception as e:
            logger.warning("mcp_session_init_failed", error=str(e))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            self._session_id = None

    async def _rate_limit(self) -> None:
        async with self._request_semaphore:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
            self._last_request_time = time.monotonic()

    async def _make_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        await self._rate_limit()
        await self._ensure_client()

        request_id = uuid.uuid4().hex[:8]
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}

        for attempt in range(settings.max_retries + 1):
            try:
                if not self._client:
                    await self._ensure_client()

                response = await self._client.post("/", json=payload, timeout=settings.request_timeout_ms / 1000)

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 1))
                    if attempt < settings.max_retries:
                        wait = min(settings.retry_backoff_base * (2**attempt), settings.retry_backoff_max)
                        logger.warning("rate_limited", attempt=attempt, wait=wait)
                        await asyncio.sleep(wait)
                        continue
                    raise RateLimitError(retry_after)

                if response.status_code == 404:
                    raise NotFoundError(params.get("nct_id", "unknown"))

                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    raise MCPError(data["error"].get("message", "Unknown error"), data["error"].get("code", "UNKNOWN"))

                return data.get("result", {})

            except httpx.TimeoutException:
                if attempt < settings.max_retries:
                    wait = min(settings.retry_backoff_base * (2**attempt), settings.retry_backoff_max)
                    logger.warning("request_timeout", attempt=attempt, wait=wait)
                    await asyncio.sleep(wait)
                    continue
                raise

            except Exception as e:
                logger.exception("mcp_request_failed", method=method, attempt=attempt, error=str(e))
                if attempt < settings.max_retries:
                    wait = min(settings.retry_backoff_base * (2**attempt), settings.retry_backoff_max)
                    await asyncio.sleep(wait)
                    continue
                raise

        raise MCPError("Max retries exceeded", "MAX_RETRIES")

    async def get_study_record(self, nct_id: str) -> dict[str, Any]:
        with tracer.start_as_current_span("mcp.get_study_record", kind=SpanKind.CLIENT) as span:
            span.set_attribute("nct_id", nct_id)
            logger.info("fetching_study_record", nct_id=nct_id)
            return await self._make_request("tools/call", {"name": "clinicaltrials_get_study_record", "arguments": {"nct_id": nct_id}})

    async def get_study_results(self, nct_id: str) -> dict[str, Any]:
        with tracer.start_as_current_span("mcp.get_study_results", kind=SpanKind.CLIENT) as span:
            span.set_attribute("nct_id", nct_id)
            logger.info("fetching_study_results", nct_id=nct_id)
            return await self._make_request("tools/call", {"name": "clinicaltrials_get_study_results", "arguments": {"nct_id": nct_id}})

    async def search_studies(self, query: str, page_size: int = 10) -> dict[str, Any]:
        with tracer.start_as_current_span("mcp.search_studies", kind=SpanKind.CLIENT) as span:
            span.set_attribute("query", query)
            logger.info("searching_studies", query=query, page_size=page_size)
            return await self._make_request("tools/call", {"name": "clinicaltrials_search_studies", "arguments": {"query": query, "page_size": page_size}})


async def create_mcp_client() -> MCPClient:
    client = MCPClient()
    await client.__aenter__()
    return client