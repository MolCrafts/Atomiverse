"""Observable acceptance and eligible timing ratios; no scientific algorithms."""

from __future__ import annotations

import re
import statistics

import numpy as np

from .specification import Bundle, Cell


def compare_values(actual: object, reference: object, atol: float, rtol: float) -> dict:
    try:
        raw_a, raw_b = np.asarray(actual), np.asarray(reference)
        if raw_a.dtype.kind not in "fiu" or raw_b.dtype.kind not in "fiu":
            return {"pass": False, "reason": "observable must contain real numbers"}
        a, b = raw_a.astype(float), raw_b.astype(float)
    except (TypeError, ValueError, OverflowError):
        return {"pass": False, "reason": "observable is not a rectangular numeric array"}
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"pass": False, "reason": "shape mismatch, empty or nonfinite observable"}
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.abs(a - b)
        limit = atol + rtol * np.abs(b)
    if not np.isfinite(error).all() or not np.isfinite(limit).all():
        return {"pass": False, "reason": "error or tolerance arithmetic overflowed"}
    scale = float(error.max())
    rms = scale * float(np.sqrt(np.mean((error / scale) ** 2))) if scale else 0.0
    return {"pass": bool(np.all(error <= limit)), "max_abs_error": float(error.max()),
            "rms_error": rms, "atol": atol, "rtol": rtol,
            "max_limit": float(limit.max())}


def complete_samples(cell: Cell, repeats: int) -> bool:
    identities = [(o.phase, o.repeat) for o in cell.observations]
    expected = {("cold", 0)} | {("warm", index) for index in range(1, repeats + 1)}
    return len(identities) == len(expected) and set(identities) == expected


def evaluate(bundle: Bundle) -> Bundle:
    cells = {(c.row_id, c.system_id, c.participant_id): c for c in bundle.cells}
    for row in bundle.rows:
        for system in row.systems:
            ref = cells[row.id, system.id, row.reference]
            ref_good = complete_samples(ref, bundle.repeats) and all(o.status == "completed" for o in ref.observations)
            ref_obs = ref.observations[0].observables if ref_good else {}
            for participant in row.participants:
                cell: Cell = cells[row.id, system.id, participant.id]
                cell.numerical = "unverified"
                cell.reason = None
                cell.cuda_only = "unverified" if participant.backend == "cuda" and participant.role == "atv" else "not_applicable"
                cell.checks = []
                observations = cell.observations
                if not observations:
                    cell.status = "not_run" if participant.available else "not_implemented"
                    cell.reason = "no execution evidence" if participant.available else participant.unavailable_reason
                if observations and all(o.status == "completed" for o in observations):
                    cell.status = "completed"
                    if ref_good:
                        for index, observation in enumerate(observations):
                            for check in row.checks:
                                actual = observation.observables.get(check.observable)
                                target = ref_obs.get(check.observable)
                                result = (compare_values(actual, target, check.atol, check.rtol)
                                          if actual is not None and target is not None else
                                          {"pass": False, "reason": "missing required observable"})
                                cell.checks.append({"observable": check.observable, "unit": check.unit,
                                                    "observation": index, **result})
                        cell.numerical = "pass" if all(c["pass"] for c in cell.checks) else "fail"
                    else:
                        cell.numerical = "unverified"
                        cell.reason = "independent reference did not complete"
                    if not complete_samples(cell, bundle.repeats):
                        cell.status = "incomplete_samples"
                        cell.reason = "missing, duplicated or unexpected cold/warm sample identities"
                        if cell.numerical != "fail":
                            cell.numerical = "unverified"
                elif observations:
                    failed = next(o for o in observations if o.status != "completed")
                    cell.status, cell.reason = failed.status, failed.reason
                    cell.numerical = "fail" if cell.status in ("scientific_failure", "not_converged") else "unverified"
                warm = [o.seconds for o in observations if o.phase == "warm" and o.status == "completed" and o.seconds]
                cell.timing = {"scope": "warm_e2e", "samples_seconds": warm,
                               "median_seconds": statistics.median(warm) if warm else None,
                               "min_seconds": min(warm) if warm else None,
                               "max_seconds": max(warm) if warm else None,
                               "mad_seconds": statistics.median(abs(t - statistics.median(warm)) for t in warm) if warm else None,
                               "cold_process_seconds": next((o.process_seconds for o in observations if o.phase == "cold"), None)}
            for participant in row.participants:
                cell = cells[row.id, system.id, participant.id]
                cell.speedup = None
                reason = None
                if cell.numerical != "pass" or ref.numerical != "pass":
                    reason = "numerical acceptance incomplete or failed"
                elif len(cell.timing["samples_seconds"]) < 5 or len(ref.timing["samples_seconds"]) < 5:
                    reason = "at least five samples per participant required"
                elif any(o.metadata.get("resource_mismatch") for c in (cell, ref) for o in c.observations):
                    reason = "actual engine thread resources differ from the declared profile"
                elif bundle.protocol.get("timing_contaminated"):
                    reason = bundle.protocol.get("timing_contamination_reason") or "observed CPU contention invalidates timing ratios"
                elif any(len({str(o.metadata.get("binary_sha256")) for o in c.observations}) != 1 or
                         any(not re.fullmatch(r"[0-9a-f]{64}", str(o.metadata.get("binary_sha256")))
                             for o in c.observations) for c in (cell, ref)):
                    reason = "native binary fingerprint missing, invalid or changed between samples"
                elif bundle.source.get("changed_during_run"):
                    reason = "source changed during campaign"
                elif participant.role == "oracle" and participant.id != row.reference:
                    reason = "additional oracle; raw timing only"
                if reason is None:
                    cell.speedup = ref.timing["median_seconds"] / cell.timing["median_seconds"]
                cell.speedup_reason = reason
    required = [c for c in bundle.cells if c.required]
    bundle.summary = {"required": len(required), "passed": sum(c.numerical == "pass" for c in required),
                      "failed": sum(c.numerical == "fail" for c in required),
                      "incomplete": sum(c.numerical == "unverified" for c in required),
                      "optional_unavailable": sum(not c.required and c.status != "completed" for c in bundle.cells),
                      "complete": all(c.numerical == "pass" for c in required),
                      "performance_regressions": 0, "historical_status": "unbaselined"}
    return bundle
