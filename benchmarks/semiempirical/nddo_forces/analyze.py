"""Analyze native four-method force timing and full-process Nsight captures.

Usage: python analyze.py RAW_DIRECTORY > summary.json
       python analyze.py --paired PAIRED_DIRECTORY > paired-summary.json
Required: forces.json, provenance.json, executable.sha256, CMakeCache.txt,
          numerical-sources.sha256; optional METHOD-ATOMS-MODE-full{.sqlite,
          .nsys-rep,-profile.json}. SQLite is opened read-only.
This deliberately leaves the historical energy-only analyzer unchanged.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys

# Reuse the existing trace-window queries; do not reinterpret its old outputs.
_spec = importlib.util.spec_from_file_location(
    "nddo_energy_analysis", Path(__file__).parents[1] / "nddo_energy" / "analyze.py")
energy_analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(energy_analysis)


def rows(db, query, parameters=()):
    return [dict(row) for row in db.execute(query, parameters)]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_delta(a, b):
    if len(a["charges_e"]) != len(b["charges_e"]) or len(a["forces_ev_a"]) != len(b["forces_ev_a"]):
        raise ValueError("scientific result vector sizes disagree")
    return {
        "energy_ev": abs(a["energy_ev"] - b["energy_ev"]),
        "max_charge_e": max(abs(x-y) for x, y in zip(a["charges_e"], b["charges_e"])),
        "max_force_ev_a": max(abs(x-y) for x, y in zip(a["forces_ev_a"], b["forces_ev_a"])),
    }


def check_delta(delta, metadata):
    limits = ("compare_energy_tol_ev", "compare_charge_tol_e", "compare_force_tol_ev_a")
    for key, limit in zip(delta, limits):
        if not delta[key] <= metadata[limit]:
            raise ValueError(f"{key}: got {delta[key]}, limit {metadata[limit]}")


def api_summary(db, table, parameters=None):
    clause = "WHERE r.start>=:lo AND r.end<=:hi" if parameters else ""
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    callchains = "COUNT(r.callchainId)" if "callchainId" in columns else "NULL"
    failures = "SUM(r.returnValue!=0)" if "returnValue" in columns else "NULL"
    return rows(db, f"""
SELECT s.value api, COUNT(*) calls, SUM(r.end-r.start) summed_duration_ns,
       {failures} nonzero_return_values,
       {callchains} calls_with_callchain
FROM {table} r JOIN StringIds s ON s.id=r.nameId {clause}
GROUP BY s.value ORDER BY summed_duration_ns DESC
""", parameters or {})


def memory_usage(db, schema, bounds):
    """Reconstruct observed live logical allocations, not reserved VRAM.

    Schema/enum labels come from the exported trace, not hard-coded enum IDs.
    See https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
    CUDA_GPU_MEMORY_USAGE_EVENTS. Unsupported schemas fail explicitly.
    """
    table = "CUDA_GPU_MEMORY_USAGE_EVENTS"
    if table not in schema:
        raise ValueError("missing CUDA memory events: capture with --cuda-memory-usage=true")
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    required = {"start", "globalPid", "deviceId", "contextId", "address", "bytes",
                "memKind", "memoryOperationType"}
    if not required <= columns:
        raise ValueError(f"unsupported memory event schema: {columns}")
    events = rows(db, f"""
SELECT m.*, o.label operation, o.name operation_symbol, k.label memory_kind
FROM {table} m
LEFT JOIN ENUM_CUDA_DEV_MEM_EVENT_OPER o ON o.id=m.memoryOperationType
LEFT JOIN ENUM_CUDA_MEM_KIND k ON k.id=m.memKind ORDER BY m.start,m.rowid
""")
    if not events:
        raise ValueError("memory collection produced no allocation events")
    if len(events) != db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]:
        raise ValueError("memory event enum join changed the raw event count")
    groups = {}
    window_events = {}
    for event in events:
        if event["operation"] is None or event["memory_kind"] is None:
            raise ValueError("memory event references an unknown enum value")
        if event["bytes"] < 0:
            raise ValueError("memory allocation/release size must be nonnegative")
        key = (event["globalPid"], event["deviceId"], event["memory_kind"])
        state = groups.setdefault(key, {
            "global_pid": key[0], "device_id": key[1], "memory_kind": key[2],
            "events": 0, "allocations": 0, "releases": 0,
            "observed_allocated_bytes": 0, "observed_released_bytes": 0,
            "observed_peak_live_bytes": 0, "observed_live_at_window_start_bytes": 0,
            "observed_peak_live_in_window_bytes": 0,
            "observed_live_at_window_end_bytes": 0,
            "first_event_ns": event["start"], "last_event_ns": event["start"],
            "issues": [], "live": {}, "bytes_live": 0,
        })
        time = event["start"]
        if bounds["lo"] <= time <= bounds["hi"]:
            event_key = (event["memory_kind"], event["operation"],
                         event.get("name"), event["bytes"])
            summary = window_events.setdefault(event_key, {
                "memory_kind": event["memory_kind"],
                "operation": event["operation"],
                "name": event.get("name"), "bytes_per_event": event["bytes"],
                "events": 0, "total_bytes": 0,
                "first_event_ns": time, "last_event_ns": time,
            })
            summary["events"] += 1
            summary["total_bytes"] += event["bytes"]
            summary["last_event_ns"] = time
        if time >= bounds["lo"]:
            state["observed_peak_live_in_window_bytes"] = max(
                state["observed_peak_live_in_window_bytes"], state["bytes_live"]
            ) if time <= bounds["hi"] else state["observed_peak_live_in_window_bytes"]
        address = (event["contextId"], event["address"])
        operation = (str(event["operation"]) + " " + str(event["operation_symbol"])).lower()
        size = event["bytes"]
        if "release" in operation or "free" in operation or "dealloc" in operation:
            state["releases"] += 1
            previous = state["live"].pop(address, None)
            if previous is None:
                state["issues"].append({"event_ns": time, "unmatched_release": event["address"], "bytes": size})
            else:
                if size not in (0, previous):
                    state["issues"].append({"event_ns": time, "release_size": size, "allocation_size": previous})
                state["bytes_live"] -= previous
                state["observed_released_bytes"] += previous
        elif "alloc" in operation:
            state["allocations"] += 1
            if address in state["live"]:
                state["issues"].append({"event_ns": time, "duplicate_live_allocation": event["address"]})
                state["bytes_live"] -= state["live"][address]
            state["live"][address] = size
            state["bytes_live"] += size
            state["observed_allocated_bytes"] += size
        else:
            raise ValueError(f"unsupported memory operation: {operation}")
        state["events"] += 1
        state["last_event_ns"] = time
        state["observed_peak_live_bytes"] = max(state["observed_peak_live_bytes"], state["bytes_live"])
        if time < bounds["lo"]:
            state["observed_live_at_window_start_bytes"] = state["bytes_live"]
            state["observed_peak_live_in_window_bytes"] = state["bytes_live"]
        if time <= bounds["hi"]:
            state["observed_live_at_window_end_bytes"] = state["bytes_live"]
            if time >= bounds["lo"]:
                state["observed_peak_live_in_window_bytes"] = max(
                    state["observed_peak_live_in_window_bytes"], state["bytes_live"])
    for state in groups.values():
        state["observed_live_at_capture_end_bytes"] = state.pop("bytes_live")
        state["live_allocations_at_capture_end"] = len(state.pop("live"))
        state["allocation_pairing_complete"] = not state["issues"]
    return {"schema": schema[table], "event_count": len(events),
            "resident_window_event_count": sum(row["events"] for row in window_events.values()),
            "resident_window_events_by_kind_operation_name_size": list(window_events.values()),
            "operation_labels": rows(db, "SELECT * FROM ENUM_CUDA_DEV_MEM_EVENT_OPER"),
            "groups": list(groups.values()),
            "scope": "Observed logical allocations per process/device/memory kind; not driver reservations, complete VRAM, or application ownership.",
            "capture_covers_allocations_before_resident_window": events[0]["start"] < bounds["lo"]}


def profile_report(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        schema = {row[0]: row[1] for row in db.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")}
        boundaries = rows(db, energy_analysis.BOUNDARY_QUERY)
        if len(boundaries) != 2 or not boundaries[0]["api"].startswith("cudaProfilerStart") or not boundaries[1]["api"].startswith("cudaProfilerStop"):
            raise ValueError("expected exactly one resident profiler interval")
        if any(row["returnValue"] for row in boundaries):
            raise ValueError("profiler marker API failed")
        bounds = {"lo": boundaries[0]["end"], "hi": boundaries[1]["start"]}
        whole, window = {}, {}
        for kind in ("RUNTIME", "DRIVER"):
            table = "CUPTI_ACTIVITY_KIND_" + kind
            if table in schema:
                whole[kind.lower() + "_apis"] = api_summary(db, table)
                window[kind.lower() + "_apis"] = api_summary(db, table, bounds)
                for target in (whole, window):
                    target[kind.lower() + "_allocation_apis"] = [row for row in target[kind.lower() + "_apis"]
                        if any(part in row["api"].lower() for part in ("alloc", "free", "memmap", "memunmap", "mempool"))]
            else:
                whole[kind.lower() + "_apis"] = None
                window[kind.lower() + "_apis"] = None
        for name, query in energy_analysis.WINDOW_QUERIES.items():
            window[name] = rows(db, query, bounds)
        kernel_query = """
SELECT s.value kernel, COUNT(*) calls, SUM(k.end-k.start) summed_duration_ns,
       MIN(k.end-k.start) min_duration_ns, MAX(k.end-k.start) max_duration_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
{clause} GROUP BY s.value ORDER BY summed_duration_ns DESC
"""
        whole["kernels"] = rows(db, kernel_query.format(clause=""))
        window["kernels"] = rows(db, kernel_query.format(clause="WHERE k.start>=:lo AND k.end<=:hi"), bounds)
        for name in ("diagnostic_warnings_errors", "capture_flags", "export_metadata", "runtime_bounds"):
            whole[name] = rows(db, energy_analysis.QUERIES[name])
        if whole["diagnostic_warnings_errors"]:
            raise ValueError(f"trace diagnostics require review: {whole['diagnostic_warnings_errors']}")
        if window["duplicate_runtime_correlations"][0]["duplicate_ids"]:
            raise ValueError("runtime correlations are not unique")
        sync = window["final_sync"]
        if not sync or sync[0]["returnValue"]:
            raise ValueError("resident interval lacks successful final stream synchronization")
        for activity in ("kernel", "memcpy", "memset", "synchronization"):
            event = window[activity + "_api_correlated"][0]
            if event["events_outside_window"] or window[activity + "_overlapping_other_api"][0]["events"]:
                raise ValueError(f"{activity} crosses resident capture boundary")
            if event["last_end_ns"] is not None and event["last_end_ns"] > sync[0]["end"]:
                raise ValueError(f"{activity} not completed by final synchronization")
        whole["capture_settings"] = rows(db, "SELECT name,value FROM META_DATA_CAPTURE WHERE UPPER(name) LIKE '%CUDA%' OR UPPER(name) LIKE '%MEMORY%' OR UPPER(name) LIKE '%UVM%' ORDER BY name")
        for kind in ("runtime", "driver"):
            if any(row["nonzero_return_values"] for row in (window[kind + "_apis"] or [])):
                raise ValueError(f"failed {kind} API inside resident window")
        memory = memory_usage(db, schema, bounds)
        faults = {}
        for table in schema:
            if "PAGE_FAULT" in table and not table.startswith("ENUM_"):
                cols = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
                fields = "COUNT(*) events"
                if "numberOfPageFaults" in cols:
                    fields += ",SUM(numberOfPageFaults) reported_page_faults"
                faults[table] = {"schema": schema[table], "whole_process": rows(db, f'SELECT {fields} FROM "{table}"')}
        return {"profiler_boundaries": boundaries, "window_parameters_ns": bounds,
                "whole_process": whole, "resident_window": window, "memory": memory,
                "page_fault_events": faults, "schema": {name: sql for name, sql in schema.items()
                    if "MEMORY" in name or "CALLCHAIN" in name or name.endswith(("RUNTIME", "DRIVER"))}}
    finally:
        db.close()


def main(raw):
    force = json.loads((raw / "forces.json").read_text())
    finite_json(force, "forces.json")
    if force["schema"] != 2 or force["profile_enabled"]:
        raise ValueError("forces.json must be the unprofiled native force run (schema 2)")
    report = {"provenance": json.loads((raw / "provenance.json").read_text()),
              "metadata": {k: v for k, v in force.items() if k != "series"},
              "file_sha256": {}, "timings_unprofiled": [], "scientific_comparisons": [],
              "full_process_traces": {}, "limitations": [
                  "Fixed and prescribed-moving warm solves reuse electronic density; the latter is not integrated BOMD throughput.",
                  "Profiled times include instrumentation overhead and are excluded from latency summaries.",
                  "cudaMemGetInfo is whole-device free memory at named checkpoints, not peak live allocation or method ownership.",
                  "GPU memory traces track observed logical allocations, not driver reservations or complete VRAM usage.",
                  "Runtime and driver allocation API counts are separate nested views and must not be summed.",
                  "Resident E/q/full-force vectors are validated only after the timed range; standalone results are validated every call.",
                  "No host malloc/new coverage is inferred from CUDA tracing; use the separate allocation harness/platform trace.",
                  "Single-revision measurements establish no causal optimization benefit or portable speedup."]}
    lookup = {(s["method"], s["atoms"], s["geometry_mode"], s["backend"], s["scope"]): s for s in force["series"]}
    for s in force["series"]:
        report["timings_unprofiled"].append({**{k: s[k] for k in ("method", "atoms", "geometry_mode", "backend", "scope")},
            "cold": energy_analysis.samples_summary(s["cold"]), "warm": energy_analysis.samples_summary(s["warm"]),
            "device_free_snapshots": {k: v for k, v in s.items() if k.startswith("device_free_")}})
    for method, atoms, mode in sorted({(s["method"], s["atoms"], s["geometry_mode"]) for s in force["series"]}):
        cpu = lookup[method, atoms, mode, "cpu", "host_frame_energy_charge_force"]
        for scope in ("host_frame_energy_charge_force", "resident_evaluate_force"):
            gpu = lookup[method, atoms, mode, "cuda", scope]
            deltas = [result_delta(cpu["final_result"], gpu["final_result"])]
            if scope.startswith("host_frame"):
                for phase in ("cold", "warm"):
                    if len(cpu[phase]) != len(gpu[phase]):
                        raise ValueError("paired timing call counts disagree")
                    deltas.extend(result_delta(a["result"], b["result"]) for a, b in zip(cpu[phase], gpu[phase]))
            for delta in deltas:
                check_delta(delta, force)
            report["scientific_comparisons"].append({"method": method, "atoms": atoms, "geometry_mode": mode, "cuda_scope": scope,
                "paired_results": len(deltas), "max_delta": {k: max(d[k] for d in deltas) for k in deltas[0]}})
    for path in sorted(raw.glob("*-full.sqlite")):
        profile_path = path.with_name(path.stem + "-profile.json")
        profile = json.loads(profile_path.read_text())
        finite_json(profile, profile_path.name)
        if not profile["profile_enabled"]:
            raise ValueError("profile JSON must include markers")
        resident = next(s for s in profile["series"] if s["scope"] == "resident_evaluate_force")
        trace = profile_report(path)
        trace["profiled_warm_samples"] = energy_analysis.samples_summary(resident["warm"])
        reference = lookup[resident["method"], resident["atoms"], resident["geometry_mode"], "cuda", "resident_evaluate_force"]
        delta = result_delta(reference["final_result"], resident["final_result"])
        check_delta(delta, force)
        trace["final_delta_vs_unprofiled"] = delta
        report["full_process_traces"][path.stem] = trace
        for artifact in (path, profile_path, path.with_suffix(".nsys-rep")):
            report["file_sha256"][artifact.name] = sha256(artifact)
    for name in ("forces.json", "provenance.json", "executable.sha256", "CMakeCache.txt", "numerical-sources.sha256"):
        report["file_sha256"][name] = sha256(raw / name)
    print(json.dumps(report, indent=2, allow_nan=False))


PAIRED_FILES = (
    ("before", "1-paired-before.json"), ("after", "2-after.json"),
    ("after", "3-after.json"), ("before", "4-paired-before.json"),
    ("before", "5-paired-before.json"), ("after", "6-after.json"),
)
SERIES_FIELDS = ("method", "atoms", "geometry_mode", "backend", "scope")
PAIRED_METADATA = (
    "schema", "precision", "boundary", "threads", "cold_calls",
    "standalone_warmup_calls", "nested_warmup_calls", "max_iter", "density_tol",
    "energy_tol_ev", "compare_energy_tol_ev", "compare_charge_tol_e",
    "compare_force_tol_ev_a", "profile_enabled", "machine", "hostname", "os",
    "kernel", "compiler", "gpu", "gpu_total_memory_bytes",
    "cuda_runtime_version", "cuda_driver_version",
)


def finite_json(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label}: nonfinite JSON value")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_json(child, f"{label}[{index}]")


def paired_samples(samples):
    summary = energy_analysis.samples_summary(samples)
    if summary is not None:
        iterations = summary["iterations"]
        summary.update(iterations_min=min(iterations),
                       iterations_median=statistics.median(iterations),
                       iterations_max=max(iterations))
    return summary


def paired_main(raw):
    rounds, reference_metadata = [], None
    for number, (revision, filename) in enumerate(PAIRED_FILES, 1):
        document = json.loads((raw / filename).read_text())
        finite_json(document, filename)
        metadata = {field: document[field] for field in PAIRED_METADATA}
        if metadata["schema"] != 2 or metadata["profile_enabled"]:
            raise ValueError(f"{filename}: paired timing requires unprofiled schema 2")
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            differences = [key for key in metadata if metadata[key] != reference_metadata[key]]
            raise ValueError(f"{filename}: paired metadata mismatch: {differences}")
        lookup = {}
        for series in document["series"]:
            key = tuple(series[field] for field in SERIES_FIELDS)
            if key in lookup:
                raise ValueError(f"{filename}: duplicate series {key}")
            if series["geometry_mode"] not in ("fixed", "moving"):
                raise ValueError(f"{filename}: unknown geometry mode {key}")
            if series["backend"] not in ("cpu", "cuda"):
                raise ValueError(f"{filename}: unknown backend {key}")
            if series["scope"] not in ("host_frame_energy_charge_force", "resident_evaluate_force"):
                raise ValueError(f"{filename}: unknown force timing boundary {key}")
            expected_cold = metadata["cold_calls"] if series["scope"].startswith("host_frame") else 0
            if len(series["cold"]) != expected_cold:
                raise ValueError(f"{filename}: cold call count contradicts metadata {key}")
            if not series["warm"]:
                raise ValueError(f"{filename}: empty warm timing sequence {key}")
            for phase in ("cold", "warm"):
                for index, sample in enumerate(series[phase]):
                    if sample["seconds"] <= 0 or type(sample["iterations"]) is not int or sample["iterations"] <= 0:
                        raise ValueError(f"{filename}: invalid timing/iterations {key}/{phase}/{index}")
            results = [series["final_result"]]
            if series["scope"] == "host_frame_energy_charge_force":
                results.extend(sample["result"] for phase in ("cold", "warm") for sample in series[phase])
            for result in results:
                if len(result["charges_e"]) != series["atoms"] or len(result["forces_ev_a"]) != 3 * series["atoms"]:
                    raise ValueError(f"{filename}: incomplete scientific vectors {key}")
            lookup[key] = series
        if not lookup:
            raise ValueError(f"{filename}: no timing series")
        if rounds and lookup.keys() != rounds[0]["lookup"].keys():
            raise ValueError(f"{filename}: method/size/mode/scope selections differ")
        rounds.append({"round": number, "revision": revision, "filename": filename,
                       "sha256": sha256(raw / filename), "lookup": lookup,
                       "metadata": {key: value for key, value in document.items() if key != "series"}})

    report = {"paired_order": [{key: value for key, value in record.items() if key != "lookup"}
                               for record in rounds],
              "matching_metadata": reference_metadata, "series": [],
              "limitations": [
                  "Ratios are after/before elapsed time on this allocation, not a portable speedup prediction.",
                  "CPU scientific outputs and iterations must equal the first before round exactly; CUDA iterations may differ.",
                  "Resident numerical validation covers final E/q/full force only; no per-call electronic vectors are downloaded.",
                  "Fixed/moving geometry identities are the named modes and identical selected sizes of the same final harness; source/binary provenance remains required.",
              ]}
    for key, reference in sorted(rounds[0]["lookup"].items()):
        item = dict(zip(SERIES_FIELDS, key))
        item["scientific_validation"] = []
        for record in rounds[1:]:
            candidate = record["lookup"][key]
            if candidate["orbitals"] != reference["orbitals"]:
                raise ValueError(f"{record['filename']}: orbital count mismatch {key}")
            pairs = [(reference["final_result"], candidate["final_result"])]
            for phase in ("cold", "warm"):
                if len(reference[phase]) != len(candidate[phase]):
                    raise ValueError(f"{record['filename']}: call count mismatch {key}/{phase}")
                if key[3] == "cpu" and [sample["iterations"] for sample in reference[phase]] != [sample["iterations"] for sample in candidate[phase]]:
                    raise ValueError(f"{record['filename']}: CPU iterations changed {key}/{phase}")
                if key[4] == "host_frame_energy_charge_force":
                    pairs.extend((a["result"], b["result"]) for a, b in zip(reference[phase], candidate[phase]))
            deltas = []
            for before, after in pairs:
                if key[3] == "cpu":
                    fields = ("energy_ev", "charges_e", "forces_ev_a", "hof_kcal_mol")
                    if any((field in before) != (field in after) or before.get(field) != after.get(field) for field in fields):
                        raise ValueError(f"{record['filename']}: CPU scientific output changed {key}")
                delta = result_delta(before, after)
                check_delta(delta, reference_metadata)
                deltas.append(delta)
            item["scientific_validation"].append({"round": record["round"], "compared_results": len(pairs),
                "max_delta_vs_first_before": {field: max(delta[field] for delta in deltas) for field in deltas[0]}})
        for phase in ("cold", "warm"):
            summaries, pooled = [], {"before": [], "after": []}
            for record in rounds:
                samples = record["lookup"][key][phase]
                summaries.append({"round": record["round"], "revision": record["revision"],
                                  "summary": paired_samples(samples)})
                pooled[record["revision"]].extend(samples)
            before = paired_samples(pooled["before"])
            after = paired_samples(pooled["after"])
            item[phase] = {"rounds": summaries, "pooled_before": before, "pooled_after": after,
                "pooled_median_after_over_before": after["seconds_median"] / before["seconds_median"] if before else None,
                "adjacent_pair_median_after_over_before": [
                    {"before_round": bi + 1, "after_round": ai + 1,
                     "ratio": summaries[ai]["summary"]["seconds_median"] / summaries[bi]["summary"]["seconds_median"]}
                    for bi, ai in ((0, 1), (3, 2), (4, 5))] if before else []}
        report["series"].append(item)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--paired":
        paired_main(Path(sys.argv[2]))
    elif len(sys.argv) == 2:
        main(Path(sys.argv[1]))
    else:
        raise SystemExit(__doc__)
