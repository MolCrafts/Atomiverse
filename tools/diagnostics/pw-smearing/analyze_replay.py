"""Bound MP1 Fermi errors and spectral drift in an actual-QE SCF replay."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--replay", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
source = json.loads(args.replay.read_text())
groups = []
for point in source["spectra"]:
    if point["iter"] == 0:
        groups.append([])
    groups[-1].append(point)


def f(x):
    # MP1 identity independently checked against actual QE local pins.
    return 0.5 * math.erfc(x) - 0.5 * x * math.exp(-x * x) / math.sqrt(math.pi)


def delta(x):
    return (1.5 - x * x) * math.exp(-x * x) / math.sqrt(math.pi)


results = []
for solve, points in enumerate(groups):
    records = []
    for point in points:
        native_mu = point["mu_ha"]
        width = point["width_ha"]
        evals = point["evals_ha"]
        weights = point["weights"]
        mu = native_mu
        for _ in range(6):
            residual = (
                math.fsum(
                    w * f((e - mu) / width) for e, w in zip(evals, weights, strict=True)
                )
                - point["target"]
            )
            slope = math.fsum(
                w * delta((e - mu) / width) / width
                for e, w in zip(evals, weights, strict=True)
            )
            if abs(residual) <= 2e-15 or abs(slope) < 1e-10:
                break
            step = residual / slope
            if abs(step) > width * 1e-4:
                raise RuntimeError("Tighter-root step leaves the local branch")
            mu -= step
        occ_change = max(
            abs(f((e - mu) / width) - f((e - native_mu) / width)) for e in evals
        )
        records.append(
            {
                "iter": point["iter"],
                "tight_mu_ha": mu,
                "mu_shift_ha": mu - native_mu,
                "max_spinless_occupation_change": occ_change,
                "tight_number_residual": residual,
                "actual_qe_dN_dmu": point["qe_replay"]["dN_dmu_at_native_mu"],
                "actual_qe_number_residual": point["qe_replay"]["n_at_native_mu"]
                - point["target"],
            }
        )
    tail = points[40:] or points
    tail_records = records[40:] or records
    implicit_step_errors = []
    for previous, following in itertools.pairwise(tail):
        reduced = [
            (e - previous["mu_ha"]) / previous["width_ha"] for e in previous["evals_ha"]
        ]
        coefficients = [
            w * delta(x) for w, x in zip(previous["weights"], reduced, strict=True)
        ]
        predicted = math.fsum(
            c * (new - old)
            for c, old, new in zip(
                coefficients, previous["evals_ha"], following["evals_ha"], strict=True
            )
        ) / math.fsum(coefficients)
        implicit_step_errors.append(
            abs(following["mu_ha"] - previous["mu_ha"] - predicted)
        )
    results.append(
        {
            "solve_index": solve,
            "sample": points[0]["sample"],
            "iterations": len(points),
            "final_joint_residual": points[-1].get("residual"),
            "minimum_joint_residual": min(p.get("residual", math.inf) for p in points),
            "tail_start_iter": tail[0]["iter"],
            "tail_max_abs_mu_step_minus_implicit_spectral_prediction_ha": max(
                implicit_step_errors, default=0
            ),
            "tail_mu_range_ha": max(p["mu_ha"] for p in tail)
            - min(p["mu_ha"] for p in tail),
            "tail_min_abs_actual_qe_dN_dmu": min(
                abs(p["actual_qe_dN_dmu"]) for p in tail_records
            ),
            "tail_max_abs_actual_qe_number_residual": max(
                abs(p["actual_qe_number_residual"]) for p in tail_records
            ),
            "tail_max_abs_tight_mu_shift_ha": max(
                abs(p["mu_shift_ha"]) for p in tail_records
            ),
            "tail_max_spinless_occupation_change": max(
                p["max_spinless_occupation_change"] for p in tail_records
            ),
            "tail_eigenvalue_range_ha": [
                max(p["evals_ha"][i] for p in tail)
                - min(p["evals_ha"][i] for p in tail)
                for i in range(len(tail[0]["evals_ha"]))
            ],
            "local_tighter_root": records,
        }
    )
report = {
    "scope": "MP1 spinless spectrum; actual-QE residual/slope with independent local analytic Newton refinement; no full SCF rerun",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "replay_sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest(),
    "actual_qe_replay_summary": source["summary"],
    "solves": results,
}
args.out.write_text(json.dumps(report, indent=2) + "\n")
for result in results:
    if result["iterations"] >= 250:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key != "local_tighter_root"
                },
                indent=2,
            )
        )
