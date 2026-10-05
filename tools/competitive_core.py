"""Data, measurement, validation, and decision policy for competitive benchmarks."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

GIB = 1024**3
RUN_RE = re.compile(r"run\s+\d+:\s+([0-9.]+)s")
IRIS_RE = re.compile(r"Done -> .+ \[([0-9.]+)s\]")
DT_RE = re.compile(
    r"Total generation time \(including model loading\):\s+([0-9.]+)\s+(ms|s)"
)
RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")
FOOTPRINT_RE = re.compile(r"(\d+)\s+peak memory footprint")


@dataclass(frozen=True)
class Case:
    model: str
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    tier: str
    settings: str
    runs: int
    warmups: int


@dataclass
class Inventory:
    engine: str
    installed: bool
    runnable: bool
    version: str = ""
    reason: str = ""


@dataclass
class ProcessResult:
    returncode: int
    wall_seconds: float
    stdout: str
    stderr: str
    peak_rss_bytes: int | None
    peak_footprint_bytes: int | None


@dataclass
class Result:
    engine: str
    model: str
    tier: str
    protocol: str
    status: str
    version: str = ""
    reason: str = ""
    seconds: list[float] = field(default_factory=list)
    median_seconds: float | None = None
    peak_rss_bytes: int | None = None
    peak_footprint_bytes: int | None = None
    memory_scope: str = "front-end process"
    output_path: str = ""
    content_status: str = "not-checked"
    content_detail: str = ""
    command: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    effective_steps: int | None = None


def timed_process(
    command: list[str],
    *,
    input_text: str | None,
    env: dict[str, str] | None,
    timeout: int,
    cwd: Path | None = None,
) -> ProcessResult:
    wrapped = ["/usr/bin/time", "-l", *command]
    started = time.perf_counter()
    proc = subprocess.run(
        wrapped,
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=timeout,
        check=False,
    )
    wall = time.perf_counter() - started
    rss = RSS_RE.search(proc.stderr)
    footprint = FOOTPRINT_RE.search(proc.stderr)
    return ProcessResult(
        proc.returncode,
        wall,
        proc.stdout,
        proc.stderr,
        int(rss.group(1)) if rss else None,
        int(footprint.group(1)) if footprint else None,
    )


def named_peak_footprint(name: str) -> int | None:
    proc = subprocess.run(
        ["footprint", "-p", name, "-f", "bytes", "--noCategories"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    values = re.findall(r"phys_footprint_peak:\s+(\d+) B", proc.stdout)
    return sum(int(value) for value in values) if values else None


def write_log(log_path: Path, command: list[str], processes: list[ProcessResult]) -> None:
    lines = ["command: " + json.dumps(command), ""]
    for index, proc in enumerate(processes, 1):
        lines.extend(
            (
                f"=== process {index}: exit={proc.returncode} wall={proc.wall_seconds:.6f}s ===",
                "--- stdout ---",
                proc.stdout,
                "--- stderr ---",
                proc.stderr,
            )
        )
    log_path.write_text("\n".join(lines), encoding="utf-8")


def peak(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def finish_result(
    result: Result,
    processes: list[ProcessResult],
    command: list[str],
    output: Path,
    log_path: Path,
) -> Result:
    write_log(log_path, command, processes)
    result.command = command
    result.output_path = str(output) if output.is_file() else ""
    result.peak_rss_bytes = peak([proc.peak_rss_bytes for proc in processes])
    result.peak_footprint_bytes = peak([proc.peak_footprint_bytes for proc in processes])
    if any(proc.returncode != 0 for proc in processes):
        result.status = "failed"
        result.reason = f"one or more processes failed; see {log_path}"
    if result.seconds:
        result.median_seconds = statistics.median(result.seconds)
    elif result.status == "ok":
        result.status = "failed"
        result.reason = f"no timing records parsed; see {log_path}"
    return result


def unavailable(engine: str, case: Case, item: Inventory) -> Result:
    return Result(
        engine=engine,
        model=case.model,
        tier=case.tier,
        protocol="none",
        status="unavailable",
        version=item.version,
        reason=item.reason,
    )


def check_content(result: Result, root: Path, python: str) -> None:
    if result.status != "ok" or not result.output_path:
        return
    checker = root / "tools" / "quality" / "content_check.py"
    proc = subprocess.run(
        [python, str(checker), result.output_path],
        capture_output=True,
        text=True,
        check=False,
    )
    result.content_status = "pass" if proc.returncode == 0 else "fail"
    result.content_detail = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        result.status = "invalid-output"
        result.reason = "content sanity check failed"


def ratio_gap(value: float, leader: float) -> float:
    return value / leader - 1.0


def verdict(
    results: list[Result],
    required: set[str],
    protocol: str,
) -> dict[str, object]:
    rows = [
        row
        for row in results
        if row.protocol == protocol
        and row.status == "ok"
        and row.content_status == "pass"
        and row.median_seconds is not None
    ]
    zdraw = next((row for row in rows if row.engine == "zdraw"), None)
    competitors = [row for row in rows if row.engine != "zdraw"]
    covered = {row.engine for row in competitors}
    missing = sorted(required - covered)
    if zdraw is None or not competitors:
        return {
            "protocol": protocol,
            "decision": "incomplete",
            "reason": "zdraw or a comparable competitor result is missing",
            "missing_required": missing,
        }

    speed_leader = min(competitors, key=lambda row: row.median_seconds or float("inf"))
    speed_gap = ratio_gap(zdraw.median_seconds or 0, speed_leader.median_seconds or 1)
    memory_rows = [row for row in competitors if row.peak_footprint_bytes is not None]
    memory_gap = None
    memory_leader = None
    if memory_rows and zdraw.peak_footprint_bytes is not None:
        memory_leader = min(memory_rows, key=lambda row: row.peak_footprint_bytes or sys.maxsize)
        memory_gap = ratio_gap(
            float(zdraw.peak_footprint_bytes),
            float(memory_leader.peak_footprint_bytes or 1),
        )

    worst_gap = max(speed_gap, memory_gap if memory_gap is not None else speed_gap)
    under_16 = (
        zdraw.peak_footprint_bytes <= 16 * GIB
        if zdraw.peak_footprint_bytes is not None
        else None
    )
    if missing:
        decision = "incomplete"
    elif speed_gap <= 0 and memory_gap is not None and memory_gap <= 0 and under_16:
        decision = "commercial-candidate"
    elif worst_gap <= 0.20:
        decision = "continue-engineering"
    elif worst_gap <= 0.50:
        decision = "conditional"
    else:
        decision = "open-source-only"
    return {
        "protocol": protocol,
        "decision": decision,
        "missing_required": missing,
        "speed_leader": speed_leader.engine,
        "speed_gap": speed_gap,
        "memory_leader": memory_leader.engine if memory_leader else None,
        "memory_gap": memory_gap,
        "zdraw_under_16_gib": under_16,
    }


def fmt_seconds(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "—"


def fmt_gib(value: int | None) -> str:
    return f"{value / GIB:.2f}" if value is not None else "—"


def write_reports(
    out_dir: Path,
    case: Case,
    inventory: dict[str, Inventory],
    results: list[Result],
    verdicts: list[dict[str, object]],
    host: dict[str, object],
) -> None:
    (out_dir / "inventory.json").write_text(
        json.dumps([asdict(item) for item in inventory.values()], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "case": asdict(case),
        "host": host,
        "results": [asdict(result) for result in results],
        "verdicts": verdicts,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fields = (
        "engine",
        "model",
        "tier",
        "protocol",
        "status",
        "version",
        "median_seconds",
        "peak_rss_bytes",
        "peak_footprint_bytes",
        "memory_scope",
        "output_path",
        "content_status",
        "reason",
    )
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({field: row[field] for field in fields})

    lines = [
        "# Competitive benchmark",
        "",
        f"- model: `{case.model}`",
        f"- tier: `{case.tier}`",
        f"- settings: `{case.settings}`",
        f"- workload: {case.width}×{case.height}, requested {case.steps} steps, seed {case.seed}",
        f"- runs: {case.runs}; warmups: {case.warmups}",
        "",
        "| engine | protocol | steps | status | median s | peak footprint GiB | memory scope | output |",
        "|---|---|---:|---|---:|---:|---|---|",
    ]
    for result in results:
        output = result.content_status if result.output_path else "—"
        lines.append(
            f"| {result.engine} | {result.protocol} | {result.effective_steps or '—'} | "
            f"{result.status} | "
            f"{fmt_seconds(result.median_seconds)} | "
            f"{fmt_gib(result.peak_footprint_bytes)} | {result.memory_scope} | {output} |"
        )
    lines.extend(("", "## Decision", ""))
    for item in verdicts:
        lines.append(
            f"- `{item['protocol']}`: **{str(item['decision']).upper()}**"
            + (
                f"; missing required: {', '.join(item['missing_required'])}"
                if item.get("missing_required")
                else ""
            )
        )
    lines.extend(
        (
            "",
            "Cold-process and warm-session rows are never pooled. Memory is the macOS",
            "`/usr/bin/time -l` peak footprint unless the row states a broader scope.",
            "External XPC or daemon memory is excluded from front-end-only rows and must",
            "be added before a public claim.",
            "An unavailable required competitor makes the decision incomplete.",
            "",
        )
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def host_info() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "loadavg": list(os.getloadavg()),
    }
