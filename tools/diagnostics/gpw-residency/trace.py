#!/usr/bin/env python3
"""Audit a real Nsight cudaProfilerApi force capture, not a source counter.

Only eigensolver scalars with the stated attribution are permitted. Geometry,
fixed tables and final outputs belong outside the captured force interval.
The capture command and source review establish the interval and scalar role;
this script establishes the actual directions, sizes and executed kernels.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("sqlite", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--required-kernel", action="append", default=[])
args = parser.parse_args()
db = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
db.row_factory = sqlite3.Row
tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
required = {"StringIds", "ENUM_CUDA_MEMCPY_OPER", "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME"}
errors = [f"missing Nsight table {name}" for name in sorted(required - tables)]
record = {"input": str(args.sqlite), "sha256": hashlib.sha256(args.sqlite.read_bytes()).hexdigest(),
          "scope": "cudaProfilerApi accepted physical W preparation and Gamma RKS single-grid force",
          "errors": errors, "formal_source_qualification": False,
          "formal_build_qualification": False, "external_engine_qualification": False}
if not errors:
    copies = [dict(row) for row in db.execute("""
        SELECT m.start, m.end, m.bytes, e.name AS direction, m.correlationId,
               s.value AS runtime_api, r.start AS runtime_start,
               r.end AS runtime_end, r.globalTid
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind=e.id
        LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON m.correlationId=r.correlationId
        LEFT JOIN StringIds s ON s.id=r.nameId ORDER BY m.start
    """)]
    kernels = [dict(row) for row in db.execute("""
        SELECT s.value AS name, count(*) AS launches, sum(k.end-k.start) AS duration_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
        GROUP BY s.value ORDER BY s.value
    """)]
    statuses = [copy for copy in copies if copy["direction"] == "CUDA_MEMCPY_KIND_DTOH"
                and copy["bytes"] == 4]
    vendor_scalars = []
    if "NVTX_EVENTS" in tables:
        for copy in copies:
            if copy["direction"] != "CUDA_MEMCPY_KIND_DTOH" or copy["bytes"] != 8:
                continue
            # CUDA 13 Dsyevd retrieves one scalar after its matrix-norm
            # reduction. Require actual vendor API nesting on the same host
            # thread and the executed norm producer before permitting it.
            ranges = list(db.execute("""
                SELECT n.start FROM NVTX_EVENTS n
                LEFT JOIN StringIds s ON s.id=n.textId
                WHERE coalesce(n.text,s.value)='cusolverDnXsyevd_legacy'
                  AND n.globalTid=? AND n.start<=? AND n.end>=?
            """, (copy["globalTid"], copy["runtime_start"], copy["runtime_end"])))
            if any(db.execute("""
                SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id=k.demangledName
                WHERE s.value LIKE '%lansy_M_stage2%' AND k.start>=? AND k.end<=?
            """, (row[0], copy["start"])).fetchone()[0] for row in ranges):
                vendor_scalars.append(copy)
    allowed = {"CUDA_MEMCPY_KIND_DTOD", "CUDA_MEMCPY_KIND_PTOP"}
    forbidden = [copy for copy in copies if copy["direction"] not in allowed
                 and copy not in statuses and copy not in vendor_scalars]
    if forbidden:
        errors.append("host/device field transfers occurred inside the force capture")
    if not kernels:
        errors.append("capture contains no device kernels")
    for name in args.required_kernel:
        if not any(name in kernel["name"] for kernel in kernels):
            errors.append(f"required force kernel did not execute: {name}")
    runtime_errors = [dict(row) for row in db.execute("""
        SELECT s.value AS name, r.returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id=r.nameId WHERE r.returnValue<>0
    """)]
    if runtime_errors:
        errors.append("capture contains nonzero CUDA runtime statuses")
    summary = collections.defaultdict(lambda: {"count": 0, "bytes": 0})
    for copy in copies:
        summary[copy["direction"]]["count"] += 1
        summary[copy["direction"]]["bytes"] += copy["bytes"]
    record.update(memcpy=copies, memcpy_summary=dict(summary), kernels=kernels,
                  permitted_solver_status_downloads=statuses,
                  permitted_vendor_norm_downloads=vendor_scalars,
                  forbidden_transfers=forbidden, runtime_errors=runtime_errors)
record["passed"] = not errors
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"passed": record["passed"], "errors": errors,
                  "memcpy_summary": record.get("memcpy_summary", {})}))
raise SystemExit(0 if record["passed"] else 1)
