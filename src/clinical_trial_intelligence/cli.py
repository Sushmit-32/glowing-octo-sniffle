from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from clinical_trial_intelligence.config import PipelineConfig, EnforcementMode, EligibilityMode
from clinical_trial_intelligence.observability import setup_observability
from clinical_trial_intelligence.workflows.pipeline import run_pipeline

app = typer.Typer(name="clinical-trial-intelligence", help="Clinical Trial Intelligence Agent")
console = Console()


@app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Input CSV file with NCT codes"),
    output: Path = typer.Option(None, "--output", "-o", help="Output CSV file path"),
    batch_size: int = typer.Option(10, "--batch-size", "-b", help="Batch size"),
    parallel: int = typer.Option(3, "--parallel", "-p", help="Parallel workers"),
    checkpoint: Path = typer.Option(None, "--checkpoint", "-c", help="Checkpoint file to resume from"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from checkpoint"),
    config_file: Path = typer.Option(None, "--config", help="Config file path"),
    mcp_server: str = typer.Option("https://clinicaltrials.caseyjhand.com/mcp", "--mcp-server", help="MCP server URL"),
    enforcement: str = typer.Option("soft", "--enforcement", help="Enforcement mode: soft or hard"),
    eligibility: str = typer.Option("regex", "--eligibility", help="Eligibility parser mode: regex or llm"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level"),
) -> None:
    setup_observability()

    if not input.exists():
        console.print(f"[red]Input file not found: {input}[/red]")
        raise typer.Exit(1)

    nct_codes = []
    with open(input) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "nct_id" in row:
                nct_codes.append(row["nct_id"].strip())

    if not nct_codes:
        console.print("[red]No NCT codes found in input file[/red]")
        raise typer.Exit(1)

    if output is None:
        output = Path(f"trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    config = PipelineConfig(
        batch_size=batch_size,
        parallel_workers=parallel,
        mcp_server_url=mcp_server,
        checkpoint_dir=checkpoint.parent if checkpoint else Path("data/checkpoints"),
        csv_output_dir=output.parent,
        constitution_path=Path("constitution/constitution.yaml"),
        enforcement_mode=EnforcementMode(enforcement),
        eligibility_mode=EligibilityMode(eligibility),
        resume=resume,
        checkpoint_enabled=bool(checkpoint),
    )

    console.print(f"[green]Starting pipeline with {len(nct_codes)} NCT codes[/green]")
    console.print(f"Output: {output}")

    import asyncio
    result = asyncio.run(run_pipeline(nct_codes, config))

    console.print(f"[green]Pipeline completed![/green]")
    console.print(f"Total: {result.total_trials}, Successful: {result.successful_trials}, Failed: {result.failed_trials}")
    if result.output_path:
        console.print(f"Output saved to: {result.output_path}")


@app.command()
def review(
    input: Path = typer.Option(..., "--input", "-i", help="Input CSV file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output CSV for flagged records"),
    severity: str = typer.Option("warning", "--severity", "-s", help="Minimum severity"),
    principle: str = typer.Option(None, "--principle", help="Filter by principle"),
) -> None:
    if output is None:
        output = Path(f"flagged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    flagged = []
    with open(input) as f:
        reader = csv.DictReader(f)
        for row in reader:
            flags_str = row.get("constitutional_flags", "")
            if not flags_str:
                continue

            try:
                flags = json.loads(flags_str) if flags_str.startswith("[") else []
            except:
                flags = []

            for flag in flags:
                if severity and flag.get("severity", "") < severity:
                    continue
                if principle and flag.get("principle") != principle:
                    continue

                flagged.append({
                    "nct_id": row.get("nct_id", ""),
                    "principle": flag.get("principle", ""),
                    "flag": flag.get("flag", ""),
                    "field": flag.get("field", ""),
                    "severity": flag.get("severity", ""),
                    "evidence": flag.get("evidence", ""),
                    "suggestion": flag.get("suggestion", ""),
                })

    if flagged:
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flagged[0].keys())
            writer.writeheader()
            writer.writerows(flagged)
        console.print(f"[green]Found {len(flagged)} flagged records, saved to {output}[/green]")
    else:
        console.print("[yellow]No flagged records found[/yellow]")


@app.command()
def merge(
    original: Path = typer.Option(..., "--original", help="Original CSV file"),
    reviewed: Path = typer.Option(..., "--reviewed", help="Reviewed CSV file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output merged CSV file"),
) -> None:
    if output is None:
        output = Path(f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    reviewed_data = {}
    with open(reviewed) as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviewed_data[row["nct_id"]] = row

    with open(original) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        nct_id = row.get("nct_id", "")
        if nct_id in reviewed_data:
            row.update(reviewed_data[nct_id])

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"[green]Merged {len(reviewed_data)} reviewed records into {output}[/green]")


@app.command()
def audit(
    input: Path = typer.Option(..., "--input", "-i", help="Input CSV file"),
    constitution: Path = typer.Option(Path("constitution/constitution.yaml"), "--constitution", help="Constitution YAML"),
) -> None:
    console.print("[yellow]Audit functionality not yet implemented[/yellow]")


@app.command()
def validate(
    input: Path = typer.Option(..., "--input", "-i", help="Input CSV file with NCT codes"),
) -> None:
    import re
    nct_pattern = re.compile(r"^NCT\d{8}$", re.IGNORECASE)

    valid = []
    invalid = []

    with open(input) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nct = row.get("nct_id", "").strip().upper()
            if nct_pattern.match(nct):
                valid.append(nct)
            else:
                invalid.append(nct)

    console.print(f"[green]Valid: {len(valid)}[/green]")
    console.print(f"[red]Invalid: {len(invalid)}[/red]")
    if invalid:
        console.print("Invalid codes:", invalid)


@app.command()
def serve(
    transport: str = typer.Option("stdio", "--transport", help="Transport: stdio or http"),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP host"),
    port: int = typer.Option(8000, "--port", help="HTTP port"),
) -> None:
    from clinical_trial_intelligence.mcp_server import mcp
    console.print(f"[green]Starting MCP server on {transport}[/green]")
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    app()