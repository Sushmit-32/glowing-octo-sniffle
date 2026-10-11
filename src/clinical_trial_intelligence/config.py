from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnforcementMode(str, Enum):
    SOFT = "soft"
    HARD = "hard"


class EligibilityMode(str, Enum):
    REGEX = "regex"
    LLM = "llm"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mcp_server_url: str = Field(
        default="https://clinicaltrials.caseyjhand.com/mcp",
        description="MCP server URL",
    )
    mcp_transport: str = Field(default="streamable-http", description="MCP transport type")

    batch_size: int = Field(default=10, ge=1, le=100, description="NCTs per batch")
    parallel_workers: int = Field(default=3, ge=1, le=10, description="Concurrent workers per batch")
    request_timeout_ms: int = Field(default=30000, ge=5000, description="Request timeout in milliseconds")
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts")
    retry_backoff_base: float = Field(default=1.0, description="Base backoff in seconds")
    retry_backoff_max: float = Field(default=60.0, description="Max backoff in seconds")

    checkpoint_dir: Path = Field(default=Path("data/checkpoints"), description="Checkpoint directory")
    csv_output_dir: Path = Field(default=Path("output"), description="CSV output directory")
    episodic_log_dir: Path = Field(default=Path("data/episodic"), description="Episodic log directory")
    rag_dir: Path = Field(default=Path("data/rag"), description="RAG vector store directory")

    constitution_path: Path = Field(default=Path("constitution/constitution.yaml"), description="Constitution YAML path")
    enforcement_mode: EnforcementMode = Field(default=EnforcementMode.SOFT, description="Enforcement mode")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold for flags")

    eligibility_parser_mode: EligibilityMode = Field(default=EligibilityMode.REGEX, description="Eligibility parser mode")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key for LLM parsing")
    anthropic_model: str = Field(default="claude-3-haiku-20240307", description="Anthropic model")

    log_level: str = Field(default="INFO", description="Log level")
    log_json: bool = Field(default=True, description="JSON log format")
    log_file: Path = Field(default=Path("logs/pipeline.log"), description="Log file path")
    audit_log_file: Path = Field(default=Path("logs/constitution_audit.jsonl"), description="Audit log file path")

    otel_endpoint: str | None = Field(default="http://localhost:4317", description="OTEL collector endpoint")
    otel_service_name: str = Field(default="clinical-trial-intelligence", description="OTEL service name")
    enable_tracing: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")

    typesafe_api_key: str | None = Field(default=None, description="TypeSafe API key")


settings = Settings()