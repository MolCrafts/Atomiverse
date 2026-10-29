#!/usr/bin/env python3
"""Audit an Nsight capture of steady CUDA MD steps for iron law 10.

The capture (cudaProfilerApi range) holds `--steps` steady steps followed by
the end-of-run retrieve. The stepping thread is the one issuing the launches;
everything it calls before its last launch belongs to the step loop. There it
may not synchronize, allocate, or copy device data to pageable memory. The
retrieve after the last launch may synchronize. Reported alongside: runtime
calls and host API time per step, launches per step, D2H traffic.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sqlite3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("sqlite", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--steps", type=int, required=True)
args = parser.parse_args()

FORBIDDEN = {
    "cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize",
    "cudaMemcpy", "cudaMemset", "cudaMemcpyToSymbol", "cudaMemcpyFromSymbol",
    "cudaMalloc", "cudaFree", "cudaMallocHost", "cudaFreeHost",
    "cudaHostAlloc", "cudaMallocManaged",
}
LAUNCHES = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cudaGraphLaunch"}


def base(name):
    return re.sub(r"_v\d+$", "", name or "")


db = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
db.row_factory = sqlite3.Row
tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
required = {"StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME"}
errors = [f"missing Nsight table {t}" for t in sorted(required - tables)]
record = {"input": str(args.sqlite),
          "sha256": hashlib.sha256(args.sqlite.read_bytes()).hexdigest(),
          "steps": args.steps, "errors": errors}

if not errors:
    calls = [dict(r) | {"api": base(r["name"])} for r in db.execute("""
        SELECT r.start, r.end, r.globalTid, r.correlationId, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        ORDER BY r.start""")]
    launches = collections.Counter(c["globalTid"] for c in calls if c["api"] in LAUNCHES)
    if not launches:
        errors.append("capture contains no launches")
    else:
        stepper = launches.most_common(1)[0][0]
        mine = [c for c in calls if c["globalTid"] == stepper]
        last = max(c["start"] for c in mine if c["api"] in LAUNCHES)
        loop = [c for c in mine if c["start"] <= last]
        violations = [c for c in loop if c["api"] in FORBIDDEN]
        per_api = collections.Counter(c["api"] for c in loop)
        host_ns = sum(c["end"] - c["start"] for c in loop)
        span_ns = last - loop[0]["start"] if loop else 0
        d2h = []
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            kinds = {}
            if "ENUM_CUDA_MEM_KIND" in tables:
                kinds = {r[0]: r[1] for r in db.execute("SELECT id, name FROM ENUM_CUDA_MEM_KIND")}
            loop_ids = {c["correlationId"]: c["api"] for c in loop}
            for m in db.execute("""SELECT bytes, copyKind, dstKind, correlationId
                                   FROM CUPTI_ACTIVITY_KIND_MEMCPY"""):
                if m["copyKind"] != 2 or m["correlationId"] not in loop_ids:
                    continue
                dst = kinds.get(m["dstKind"], str(m["dstKind"]))
                d2h.append({"bytes": m["bytes"], "dst": dst,
                            "api": loop_ids[m["correlationId"]]})
        unsafe = [c for c in d2h
                  if c["api"] != "cudaMemcpyAsync" or "PINNED" not in c["dst"].upper()]
        if violations:
            errors.append(f"{len(violations)} forbidden runtime calls in the step loop")
        if unsafe:
            errors.append(f"{len(unsafe)} D2H copies not async into pinned memory")
        reinstantiated = per_api.get("cudaGraphInstantiate", 0)
        record.update(
            stepping_thread=stepper,
            forbidden=[{"api": c["api"], "start": c["start"]} for c in violations],
            loop_runtime_calls=dict(sorted(per_api.items())),
            launches_per_step=sum(per_api[a] for a in LAUNCHES) / args.steps,
            runtime_calls_per_step=len(loop) / args.steps,
            host_api_us_per_step=host_ns / args.steps / 1e3,
            loop_wall_us_per_step=span_ns / args.steps / 1e3,
            d2h_in_loop={"count": len(d2h), "bytes": sum(c["bytes"] for c in d2h),
                         "unsafe": unsafe},
            graph_instantiations_in_loop=reinstantiated)
    # GPU idle gaps between consecutive device activities, attributed to the
    # (previous, next) activity pair: where a step waits on the host or on
    # a dependency rather than on work.
    # A fully graphed step records no kernel rows unless Nsight traces
    # graph nodes (--cuda-graph-trace=node).
    acts = [] if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables else [
        (r[0], r[1], r[2]) for r in db.execute("""
        SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id=k.shortName""")]
    for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                         ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        if table in tables:
            acts += [(r[0], r[1], f"{label}:{r[2]}B") for r in db.execute(
                f"SELECT start, end, bytes FROM {table}")]
    acts.sort()
    gaps = collections.defaultdict(lambda: [0, 0])
    busy = 0
    frontier = acts[0][1] if acts else 0
    for (s0, e0, n0), (s1, e1, n1) in zip(acts, acts[1:]):
        frontier = max(frontier, e0)
        if s1 > frontier:
            g = gaps[(n0[:48], n1[:48])]
            g[0] += s1 - frontier
            g[1] += 1
    for s0, e0, _ in acts:
        busy += e0 - s0
    if acts:
        record["gpu_span_us_per_step"] = (max(a[1] for a in acts) - acts[0][0]) / args.steps / 1e3
        record["gpu_idle_us_per_step"] = sum(g[0] for g in gaps.values()) / args.steps / 1e3
        record["top_gaps_us_per_step"] = [
            {"after": k[0], "before": k[1], "us_per_step": v[0] / args.steps / 1e3,
             "count": v[1]}
            for k, v in sorted(gaps.items(), key=lambda kv: -kv[1][0])[:8]]
    if not errors:
        api_time = collections.defaultdict(int)
        for c in loop:
            api_time[c["api"]] += c["end"] - c["start"]
        record["loop_api_us_per_step"] = {
            k: v / args.steps / 1e3 for k, v in sorted(api_time.items(), key=lambda kv: -kv[1])}
    kernels = [] if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables else [dict(r) for r in db.execute("""
        SELECT s.value AS name, count(*) AS launches, sum(k.end-k.start) AS ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        GROUP BY s.value ORDER BY ns DESC""")]
    record["kernels"] = kernels
    record["kernel_us_per_step"] = sum(k["ns"] for k in kernels) / args.steps / 1e3

record["passed"] = not errors
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: record.get(k) for k in (
    "passed", "errors", "launches_per_step", "runtime_calls_per_step",
    "host_api_us_per_step", "loop_wall_us_per_step", "kernel_us_per_step",
    "d2h_in_loop", "graph_instantiations_in_loop", "gpu_idle_us_per_step",
    "top_gaps_us_per_step", "loop_api_us_per_step")}, default=str))
raise SystemExit(0 if record["passed"] else 1)
