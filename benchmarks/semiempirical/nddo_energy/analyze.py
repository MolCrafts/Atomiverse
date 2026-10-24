"""Reproduce the bounded current 2026-10-24 report from the raw capture directory.

Usage: python analyze.py /path/to/sqm-force-closeout-results > summary.json
Only the Python standard library is used. SQLite files are opened read-only.
Environment metadata is deliberately selected, never copied wholesale.
"""

import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys


QUERIES = {
    "memcpy": """
SELECT e.label AS direction, m.bytes AS bytes_per_copy,
       COUNT(*) AS events, SUM(m.bytes) AS total_bytes,
       SUM(m.end-m.start) AS summed_duration_ns
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
GROUP BY e.label,m.bytes ORDER BY e.label,m.bytes
""",
    "runtime": """
SELECT s.value AS api, COUNT(*) AS calls,
       SUM(r.end-r.start) AS summed_duration_ns,
       COUNT(r.callchainId) AS calls_with_callchain,
       SUM(r.returnValue != 0) AS nonzero_return_values
FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
GROUP BY s.value ORDER BY s.value
""",
    "allocation_apis": """
SELECT s.value AS api, COUNT(*) AS calls,
       SUM(r.end-r.start) AS summed_duration_ns
FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
WHERE LOWER(s.value) LIKE '%alloc%' OR LOWER(s.value) LIKE '%free%'
GROUP BY s.value ORDER BY s.value
""",
    "runtime_bounds": """
SELECT MIN(start) AS first_start_ns, MAX(end) AS last_end_ns,
       MAX(end)-MIN(start) AS observed_api_span_ns
FROM CUPTI_ACTIVITY_KIND_RUNTIME
""",
    "kernels": """
SELECT COUNT(*) AS events, COUNT(DISTINCT streamId) AS distinct_streams
FROM CUPTI_ACTIVITY_KIND_KERNEL
""",
    "diagnostic_warnings_errors": """
SELECT d.severity, d.text FROM DIAGNOSTIC_EVENT d
JOIN ENUM_DIAGNOSTIC_SEVERITY_LEVEL e ON e.id=d.severity
WHERE e.label IN ('Warning','Error') ORDER BY d.timestamp
""",
    "capture_flags": """
SELECT name,value FROM META_DATA_CAPTURE
WHERE name IN ('DEVICE_DISPLAY_NAME','PROCESS_0:COLLECT_CUDA_TRACE',
'PROCESS_0:COLLECT_CPU_TRACE','PROCESS_0:COLLECT_GPU_MEMORY_USAGE',
'PROCESS_0:COLLECT_UVM_CPU_PAGE_FAULTS','PROCESS_0:COLLECT_UVM_GPU_PAGE_FAULTS',
'PROCESS_0:CUDA_SKIP_SOME_API_CALLS','SHOW_BACKTRACE') ORDER BY name
""",
    "export_metadata": """
SELECT name,value FROM META_DATA_EXPORT
WHERE name IN ('EXPORT_PRODUCT_VERSION','EXPORT_SCHEMA_VERSION',
'EXPORT_TIME_UTC','EXPORT_HOST') ORDER BY name
""",
}

BOUNDARY_QUERY = """
SELECT s.value AS api,r.start,r.end,r.correlationId,r.returnValue
FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
WHERE s.value LIKE 'cudaProfilerStart%' OR s.value LIKE 'cudaProfilerStop%'
ORDER BY r.start
"""
WINDOW_RUNTIME = """WITH scoped_runtime AS (
SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=:lo AND end<=:hi
) """
WINDOW_QUERIES = {
    name: WINDOW_RUNTIME + QUERIES[name].replace(
        "CUPTI_ACTIVITY_KIND_RUNTIME r", "scoped_runtime r")
    for name in ("runtime", "allocation_apis")
}
WINDOW_QUERIES["memcpy"] = """WITH scoped_memcpy AS (
SELECT g.* FROM CUPTI_ACTIVITY_KIND_MEMCPY g
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=g.correlationId
WHERE r.start>=:lo AND r.end<=:hi
) """ + QUERIES["memcpy"].replace("CUPTI_ACTIVITY_KIND_MEMCPY m", "scoped_memcpy m")
WINDOW_QUERIES["final_sync"] = """
SELECT r.start,r.end,r.returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON s.id=r.nameId
WHERE r.start>=:lo AND r.end<=:hi AND s.value LIKE 'cudaStreamSynchronize%'
ORDER BY r.end DESC LIMIT 1
"""
WINDOW_QUERIES["duplicate_runtime_correlations"] = """
SELECT COUNT(*) AS duplicate_ids FROM (
SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
GROUP BY correlationId HAVING COUNT(*)>1)
"""
for activity in ("KERNEL", "MEMCPY", "MEMSET", "SYNCHRONIZATION"):
    table = "CUPTI_ACTIVITY_KIND_" + activity
    WINDOW_QUERIES[activity.lower() + "_api_correlated"] = f"""
SELECT COUNT(*) AS events,MIN(g.start) AS first_start_ns,
       MAX(g.end) AS last_end_ns,COUNT(DISTINCT g.streamId) AS distinct_streams,
       SUM(g.start<:lo OR g.end>:hi) AS events_outside_window
FROM {table} g
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=g.correlationId
WHERE r.start>=:lo AND r.end<=:hi
"""
    WINDOW_QUERIES[activity.lower() + "_overlapping_other_api"] = f"""
SELECT COUNT(*) AS events FROM {table} g
LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=g.correlationId
WHERE g.start<:hi AND g.end>:lo
AND (r.start<:lo OR r.end>:hi OR r.correlationId IS NULL)
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_delta(a, b):
    assert len(a["charges_e"]) == len(b["charges_e"])
    return {
        "energy_ev": abs(a["energy_ev"] - b["energy_ev"]),
        "max_charge_e": max(abs(x-y) for x, y in zip(a["charges_e"], b["charges_e"])),
    }


def samples_summary(samples):
    if not samples:
        return None
    seconds = [x["seconds"] for x in samples]
    iterations = [x["iterations"] for x in samples]
    return {
        "calls": len(samples),
        "seconds_min": min(seconds),
        "seconds_median": statistics.median(seconds),
        "seconds_max": max(seconds),
        "iterations": iterations,
        "iterations_total": sum(iterations),
    }


def main(raw):
    energy = json.loads((raw / "energy.json").read_text())
    report = {
        **json.loads((raw / "provenance.json").read_text()),
        "metadata": {k: v for k, v in energy.items() if k != "series"},
        "file_sha256": {},
        "timings_unprofiled": [],
        "scientific_comparisons": [],
        "full_process_traces": {},
        "sql_queries": QUERIES,
        "full_process_boundary_query": BOUNDARY_QUERY,
        "full_process_window_queries": WINDOW_QUERIES,
        "limitations": [
            "One job, fixed geometry; no repeated/reversed revision experiment or speedup claim.",
            "Resident output vectors are checked only after the timed range; per-step vectors are unavailable.",
            "Only captured runtime API allocation/free names are counted; no allocator callchains, host malloc/new trace or separate driver API table.",
            "GPU memory usage and UVM page-fault collection were off; peak/live memory and implicit migrations cannot be determined.",
            "Absence of visible allocation calls cannot identify application versus library allocation ownership.",
        ],
    }
    lookup = {}
    for series in energy["series"]:
        key = (series["method"], series["atoms"], series["backend"], series["scope"])
        lookup[key] = series
        report["timings_unprofiled"].append({
            **{k: series[k] for k in ("method", "atoms", "backend", "scope")},
            "cold": samples_summary(series["cold"]),
            "warm": samples_summary(series["warm"]),
        })
    for method in ("am1", "rm1"):
        for atoms in (3, 12, 24, 48):
            cpu = lookup[method, atoms, "cpu", "host_frame_energy_charge"]
            for scope in ("host_frame_energy_charge", "resident_evaluate_energy_intent"):
                gpu = lookup[method, atoms, "cuda", scope]
                comparison = {"method": method, "atoms": atoms, "cuda_scope": scope,
                              "final_delta": result_delta(cpu["final_result"], gpu["final_result"])}
                if scope == "host_frame_energy_charge":
                    pairs = [result_delta(a["result"], b["result"])
                             for kind in ("cold", "warm")
                             for a, b in zip(cpu[kind], gpu[kind])]
                    comparison["all_25_paired_calls_max_delta"] = {
                        k: max(row[k] for row in pairs) for k in pairs[0]}
                for delta in ([comparison["final_delta"]]
                              + ([comparison["all_25_paired_calls_max_delta"]]
                                 if scope == "host_frame_energy_charge" else [])):
                    assert delta["energy_ev"] <= energy["compare_energy_tol_ev"]
                    assert delta["max_charge_e"] <= energy["compare_charge_tol_e"]
                report["scientific_comparisons"].append(comparison)
    for filename in ("energy.json", "provenance.json", "executable.sha256",
                     "CMakeCache.txt", "numerical-sources.sha256"):
        report["file_sha256"][filename] = sha256(raw / filename)
    for method in ("am1", "rm1"):
        for suffix in (".sqlite", ".nsys-rep", "-profile.json"):
            filename = f"{method}-48-full{suffix}"
            report["file_sha256"][filename] = sha256(raw / filename)
        database = raw / f"{method}-48-full.sqlite"
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        boundaries = [dict(row) for row in connection.execute(BOUNDARY_QUERY)]
        assert len(boundaries) == 2 and boundaries[0]["api"].startswith("cudaProfilerStart")
        assert boundaries[1]["api"].startswith("cudaProfilerStop")
        assert all(row["returnValue"] == 0 for row in boundaries)
        parameters = {"lo": boundaries[0]["end"], "hi": boundaries[1]["start"]}
        schema = {row[0]: row[1] for row in connection.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")}
        trace = {"schema": {name: schema[name] for name in (
                     "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_MEMCPY",
                     "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMSET",
                     "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")},
                 "profiler_api_boundaries": boundaries, "window_parameters_ns": parameters,
                 "window": {name: [dict(row) for row in connection.execute(query, parameters)]
                            for name, query in WINDOW_QUERIES.items()},
                 "whole_process": {name: [dict(row) for row in connection.execute(QUERIES[name])]
                                   for name in ("allocation_apis", "diagnostic_warnings_errors",
                                                "capture_flags", "export_metadata")}}
        connection.close()
        window = trace["window"]
        assert not trace["whole_process"]["diagnostic_warnings_errors"]
        assert window["duplicate_runtime_correlations"][0]["duplicate_ids"] == 0
        assert all(row["nonzero_return_values"] == 0 for row in window["runtime"])
        final_sync = window["final_sync"][0]
        assert final_sync["returnValue"] == 0
        for activity in ("kernel", "memcpy", "memset", "synchronization"):
            assert window[activity + "_api_correlated"][0]["events_outside_window"] == 0
            assert window[activity + "_overlapping_other_api"][0]["events"] == 0
            assert window[activity + "_api_correlated"][0]["last_end_ns"] <= final_sync["end"]
        profile = json.loads((raw / f"{method}-48-full-profile.json").read_text())
        resident = next(s for s in profile["series"]
                        if s["scope"] == "resident_evaluate_energy_intent")
        trace["profiled_warm_samples"] = samples_summary(resident["warm"])
        trace["final_delta_vs_unprofiled_resident"] = result_delta(
            lookup[method, 48, "cuda", "resident_evaluate_energy_intent"]["final_result"],
            resident["final_result"])
        trace["host_to_device_events"] = sum(
            row["events"] for row in window["memcpy"] if row["direction"] == "Host-to-Device")
        trace["visible_allocator_api_calls"] = sum(row["calls"] for row in window["allocation_apis"])
        report["full_process_traces"][method] = trace
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
