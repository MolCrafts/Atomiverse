"""Compile current and archived production smearing against compiled QE pins."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, tolerance):
    return actual is not None and abs(actual - expected) <= tolerance


def assess(data, oracle):
    failures = []
    errors = {
        key: []
        for key in (
            "point_f",
            "point_s",
            "fixed_mu_n",
            "fixed_mu_s",
            "root_n",
            "metallic_root_mu",
            "metallic_root_s",
        )
    }
    nonfinite = 0
    for i, (actual, reference) in enumerate(
        zip(data["points"], oracle["points"], strict=True)
    ):
        for native, qe in (("f", "occupation_spinless"), ("s", "entropy_spinless")):
            if actual[native] is None:
                nonfinite += 1
            else:
                errors["point_" + native].append(abs(actual[native] - reference[qe]))
            if not close(actual[native], reference[qe], 2e-13):
                failures.append(
                    {
                        "section": "point",
                        "index": i,
                        "field": native,
                        "actual": actual[native],
                        "reference": reference[qe],
                    }
                )
    for i, (actual, reference) in enumerate(
        zip(data["spectra"], oracle["spectra"], strict=True)
    ):
        for native, qe in (("n", "n_occ"), ("s", "entropy")):
            if actual["at_qe_mu"][native] is None:
                nonfinite += 1
            else:
                errors["fixed_mu_" + native].append(
                    abs(actual["at_qe_mu"][native] - reference[qe])
                )
            if not close(actual["at_qe_mu"][native], reference[qe], 2e-12):
                failures.append(
                    {
                        "section": "fixed_mu",
                        "index": i,
                        "field": native,
                        "actual": actual["at_qe_mu"][native],
                        "reference": reference[qe],
                    }
                )
        root = actual["native_root"]
        errors["root_n"].append(abs(root["n"] - reference["target"]))
        ok = close(root["n"], reference["target"], 1.5e-10)
        if reference["case"] == 1:
            # A chemical potential inside an insulating gap is not unique.
            ok &= reference["evals"][1] + 4 * reference["width"] < root["mu"]
            ok &= root["mu"] < reference["evals"][2] - 4 * reference["width"]
            ok &= close(root["s"], 0, 5e-8) and abs(reference["entropy"]) < 5e-8
        else:
            errors["metallic_root_mu"].append(abs(root["mu"] - reference["mu"]))
            errors["metallic_root_s"].append(abs(root["s"] - reference["entropy"]))
            ok &= close(root["mu"], reference["mu"], 1e-8)
            ok &= close(root["s"], reference["entropy"], 5e-8)
        if not ok:
            failures.append(
                {
                    "section": "fermi_root",
                    "index": i,
                    "actual": root,
                    "reference": reference,
                }
            )
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "point_count": len(data["points"]),
        "spectrum_count": len(data["spectra"]),
        "nonfinite_count": nonfinite,
        "max_abs_errors": {
            key: max(values, default=0) for key, values in errors.items()
        },
        "failures": failures,
    }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--qe-reference", type=Path, required=True)
parser.add_argument("--before-source", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--compiler", default="c++")
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
args.out.mkdir(parents=True, exist_ok=False)
oracle = json.loads(args.qe_reference.read_text())
compiler = shutil.which(args.compiler)
if not compiler:
    raise RuntimeError("C++ compiler unavailable")
probe = Path(__file__).with_name("probe.cpp")
sources = {
    "before": args.before_source.resolve(),
    "after": root / "src/cpu/electronic/smearing.cpp",
}
report = {
    "qe_reference_sha256": sha(args.qe_reference),
    "probe_sha256": sha(probe),
    "script_sha256": sha(__file__),
    "header_sha256": sha(root / "include/atomiverse/electronic/smearing.h"),
    "fixture_sha256": sha(root / "tests/cpu/planewave/smearing_qe_pins.h"),
    "compiler_sha256": sha(compiler),
    "compiler_version": subprocess.run(
        [compiler, "--version"], text=True, capture_output=True, check=True
    ).stdout,
    "scheduler": {
        key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST")
    },
    "results": {},
}
for label, source in sources.items():
    binary = args.out.resolve() / ("probe-" + label)
    command = [
        compiler,
        "-O2",
        "-std=c++20",
        "-I" + str(root / "include"),
        "-I" + str(root),
        str(probe),
        str(source),
        "-o",
        str(binary),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    (args.out / (label + "-build.log")).write_text(result.stdout + result.stderr)
    result.check_returncode()
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    (args.out / (label + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    report["results"][label] = {
        "source_sha256": sha(source),
        "binary_sha256": sha(binary),
        "compile_command": command,
        **assess(data, oracle),
    }
(args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            key: {k: v for k, v in value.items() if k in ("passed", "failure_count")}
            for key, value in report["results"].items()
        }
    )
)
if report["results"]["before"]["passed"] or not report["results"]["after"]["passed"]:
    raise SystemExit("Expected archived failure and repaired success")
