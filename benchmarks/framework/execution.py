"""Bounded isolated workers. Every timed invocation owns fresh scientific state."""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .comparison import evaluate
from .provenance import digest, hardware_info, source_info
from .specification import Bundle, Cell, Method, Observation, Participant, Row, System


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def worker(request_path: Path, result_path: Path) -> int:
    request = json.loads(request_path.read_text())
    row, system, participant = (Row.model_validate(request["row"]),
                                System.model_validate(request["system"]),
                                Participant.model_validate(request["participant"]))
    try:
        module, name = participant.entrypoint.split(":")
        with redirect_stdout(sys.stderr):
            adapter = getattr(importlib.import_module(module), name)
            # Import libraries/runtime before E2E; cold process wall includes them.
            run = adapter(row, system, participant, request["threads"])
            for _ in range(request["warmup"]):
                run()
            start = time.perf_counter_ns()
            result = run()
            elapsed = (time.perf_counter_ns() - start) / 1e9
        if not isinstance(result, dict) or set(result) != {"observables", "metadata"}:
            raise ValueError("adapter must return only observables and metadata; execution/timing fields belong to the framework")
        if not isinstance(result["observables"], dict) or not isinstance(result["metadata"], dict):
            raise ValueError("adapter observables and metadata must be objects")
        expected_binary = request.get("expected_binary_sha256")
        if expected_binary and result["metadata"].get("binary_sha256") != expected_binary:
            raise ValueError("loaded native binary does not match the build receipt")
        # Reject NaN/Inf before any result can be called completed.
        json.dumps(result, allow_nan=False)
        output = {"status": "completed", "seconds": elapsed, **result}
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            output.setdefault("metadata", {})["peak_process_rss_bytes"] = int(peak * (1 if sys.platform == "darwin" else 1024))
        except ImportError:
            output.setdefault("metadata", {})["peak_process_rss_bytes"] = None
    except (ModuleNotFoundError, ImportError) as error:
        output = {"status": "dependency_missing", "reason": str(error)}
    except Exception as error:
        traceback.print_exc()
        status = getattr(error, "benchmark_status", "error")
        output = {"status": status, "reason": f"{type(error).__name__}: {error}"}
    write_json(result_path, output)
    return 0 if output["status"] == "completed" else 1


def run_sample(row: Row, system: System, participant: Participant, *, phase: str,
               repeat: int, threads: int, warmup: int, timeout: float,
               out: Path, root: Path, expected_binary_sha256: str | None = None) -> Observation:
    stem = f"{row.id}--{system.id}--{participant.id}--{phase}-{repeat}"
    request_path, result_path = out / f"{stem}.input.json", out / f"{stem}.output.json"
    write_json(request_path, {"row": row.model_dump(), "system": system.model_dump(),
                             "participant": participant.model_dump(), "threads": threads,
                             "warmup": warmup, "expected_binary_sha256": expected_binary_sha256})
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    start = time.perf_counter()
    with (out / f"{stem}.log").open("w") as log:
        try:
            process = subprocess.Popen([sys.executable, "-m", "benchmarks", "worker",
                                        "--request", str(request_path), "--result", str(result_path)],
                                       cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
            try:
                data = json.loads(result_path.read_text()) if result_path.exists() else {
                    "status": "error", "reason": f"worker exited {process.returncode} without result"}
            except (ValueError, OSError) as error:
                data = {"status": "error", "reason": f"unreadable worker result: {error}"}
            if process.returncode and data.get("status") == "completed":
                data = {"status": "error", "reason": f"worker exit {process.returncode} after result"}
        except subprocess.TimeoutExpired:
            data = {"status": "timeout", "reason": f"worker exceeded {timeout:g} seconds"}
        except OSError as error:
            data = {"status": "error", "reason": f"worker launch/IO failed: {error}"}
    elapsed = time.perf_counter() - start
    log_artifact = str((out / f"{stem}.log").relative_to(out.parent))
    result_artifact = str(result_path.relative_to(out.parent))
    try:
        data.setdefault("metadata", {})["log_artifact"] = log_artifact
        json.dumps(data, allow_nan=False)
        observation = Observation(phase=phase, repeat=repeat, process_seconds=elapsed,
                                  artifact=result_artifact, **data)
    except (TypeError, ValueError, AttributeError) as error:
        if result_path.exists():
            result_path.replace(result_path.with_suffix(".invalid.json"))
        observation = Observation(phase=phase, repeat=repeat, status="error",
                                  reason=f"invalid worker result: {error}", process_seconds=elapsed,
                                  artifact=result_artifact, metadata={"log_artifact": log_artifact})
    # Controller-owned terminal artifact also exists after timeout/crash/schema failure.
    write_json(result_path, observation.model_dump())
    return observation


def run_campaign(methods: list[Method], *, tier: str, repeats: int, warmup: int,
                 threads: int, timeout: float, out: Path, root: Path,
                 require_cuda: bool = False, system_ids: list[str] | None = None,
                 catalog: dict | None = None, source_manifest: Path | None = None,
                 build_receipt: Path | None = None) -> Bundle:
    out = out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output must be a fresh or empty campaign directory")
    rows = []
    for method in methods:
        for original in method.rows:
            row = original.model_copy(deep=True)
            row.systems = [s for s in row.systems if tier in s.tiers and (not system_ids or s.id in system_ids)]
            if not row.systems:
                continue
            for p in row.participants:
                if require_cuda and p.id == "atv.cuda" and p.available:
                    p.required = True
            rows.append(row)
    if not rows:
        raise ValueError("selection contains no benchmark systems")
    if require_cuda and not any(p.id == "atv.cuda" and p.available for r in rows for p in r.participants):
        raise ValueError("selection has no implemented CUDA workload to require")
    ids = [r.id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate method row IDs")
    source = source_info(root, source_manifest, (out,))
    receipt = None
    if build_receipt is not None:
        from .build import verify_receipt

        receipt = verify_receipt(build_receipt, source)
        source["build_verified"] = True
    bundle = Bundle(campaign_id=uuid4().hex, started=datetime.now(UTC).isoformat(), tier=tier,
                    repeats=repeats, warmup=warmup, threads=threads, timeout_seconds=timeout,
                    profile=f"{sys.platform}-{os.uname().machine}-cpu{threads}-{tier}", source=source,
                    environment=hardware_info(), rows=rows,
                    cells=[Cell(row_id=r.id, system_id=s.id, participant_id=p.id, required=p.required)
                           for r in rows for s in r.systems for p in r.participants],
                    protocol={"timing_scope": "warm_e2e", "fresh_state": True,
                              "sample_schedule": "serial interleaved, reversed on alternate repeats",
                              "worker_policy": "fresh process; untimed warmup before each warm sample",
                              "selection_partial": bool(system_ids), "spec_sha256": digest([r.model_dump() for r in rows]),
                              "catalog": catalog or {},
                              "build_receipt": receipt,
                              "coverage_policy": {r.id: {"systems": [{"id": s.id, "tiers": s.tiers} for s in r.systems]}
                                                  for m in methods for r in m.rows},
                              "instrumented": False, "precision": "fp64"})
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_json(out / "results.json", bundle.model_dump())
    cells = {(c.row_id, c.system_id, c.participant_id): c for c in bundle.cells}
    try:
        for row in rows:
            for system in row.systems:
                for index in range(repeats + 1):
                    phase = "cold" if index == 0 else "warm"
                    participants = row.participants if index % 2 == 0 else list(reversed(row.participants))
                    for participant in participants:
                        cell = cells[row.id, system.id, participant.id]
                        if not participant.available:
                            cell.status = "not_implemented"
                            cell.reason = participant.unavailable_reason
                            continue
                        if cell.observations and cell.observations[-1].status != "completed":
                            continue
                        print(f"{row.id} / {system.id} / {participant.id} / {phase} {index}", flush=True)
                        observation = run_sample(row, system, participant, phase=phase, repeat=index,
                                                 threads=threads, warmup=0 if index == 0 else warmup,
                                                 timeout=timeout, out=artifacts, root=root,
                                                 expected_binary_sha256=receipt["binary_sha256"]
                                                 if receipt and participant.role == "atv" else None)
                        cell.observations.append(observation)
                        # Checkpoint every observation, including failures.
                        write_json(out / "results.json", bundle.model_dump())
    finally:
        try:
            final_source = source_info(root, source_manifest, (out,))
            bundle.source["changed_during_run"] = final_source["tree_sha256"] != source["tree_sha256"]
        except ValueError as error:
            bundle.source["changed_during_run"] = True
            bundle.source["source_error"] = str(error)
        bundle.finished = datetime.now(UTC).isoformat()
        evaluate(bundle)
        write_json(out / "results.json", bundle.model_dump())
    return bundle
