#!/usr/bin/env python3
"""Collect complete GPW Driver evidence; never execute scientific calculations."""

import argparse
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET


CPU_CASES = {
    "gamma-rks", "gamma-uks", "k-rks", "k-uks", "k-rks-u", "k-uks-u",
    "budgets", "nested-diis-rks", "nested-diis-uks", "nested-ot-rks",
    "nested-ot-uks",
}
CUDA_CASES = {"rks", "uks", "multigrid", "kpoints"}
UKS_PBE_CASES = {family + '-' + spin
                 for family in ('PBE', 'PBEsol', 'revPBE', 'RPBE')
                 for spin in ('h3', 'h2-emptybeta')}
PREFLIGHT_CASES = {f"{case}-{cutoff}" for case in CUDA_CASES for cutoff in (80, 100)}
CUDA_UNITS = (
    "test_cuda_gpw_stress", "test_cuda_gpw_forces", "test_cuda_gpw_ks_gga",
    "test_cuda_gpw_ks_chain", "test_cuda_gpw_ks_multigrid",
    "test_cuda_gpw_diis_eigen", "test_cuda_gpw_kpoint_diis_eigen",
    "test_cuda_gpw_kpoint_h_residency", "test_cuda_gpw_kpoint_stvnl",
)


def inventory(path):
    result = []
    suite = ""
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if re.fullmatch(r"[\w/]+\.", line):
            suite = line[:-1]
        elif suite and re.fullmatch(r"  [\w/]+", line):
            result.append(suite + "." + line.strip())
    return result


def collect(source, destination, profile, job, exit_code, uks_pbe=False):
    destination.mkdir(parents=True, exist_ok=True)
    failures, units, matrices = [], [], {}
    if uks_pbe and profile != "gh200":
        failures.append("UKS PBE Driver envelope requires native GH200 profile")
    if exit_code:
        failures.append(f"native build/run exit code: {exit_code}")
    for path in source.iterdir():
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
    for filename in ("source-before.json", "source-after.json", "binaries.sha256", "hardware.txt"):
        if not (source / filename).exists():
            failures.append("missing provenance: " + filename)
    try:
        before = json.loads((source / "source-before.json").read_text())
        after = json.loads((source / "source-after.json").read_text())
        if not before or before != after:
            failures.append("source changed during build/run or empty fingerprint")
    except (OSError, ValueError) as error:
        failures.append("invalid source fingerprint: " + str(error))
    if profile == "gh200":
        path = source / "hardware.txt"
        if not path.exists() or "GH200" not in path.read_text():
            failures.append("GH200 identification missing")

    names = CUDA_UNITS if profile == "gh200" else (
        ("cpu-units",) if profile == "cpu" else ("gpw-host", "force-host", "pw-host", "stress-host", "pw-force-host")
    )
    for name in names:
        try:
            root = ET.parse(source / (name + ".xml")).getroot()
            cases = list(root.iter("testcase"))
            actual = [c.get("classname", "") + "." + c.get("name", "") for c in cases]
            expected = inventory(source / (name + ".list"))
            if not expected or set(actual) != set(expected) or len(actual) != len(expected):
                failures.append("filtered/empty unit inventory: " + name)
            bad = [c.get("name") for c in cases if c.get("status") != "run"
                   or c.get("result") != "completed"
                   or any(c.find(tag) is not None for tag in ("failure", "error", "skipped"))]
            if bad:
                failures.append("failed/skipped units: " + name + " " + repr(bad))
            units.append({"name": name, "tests": len(cases), "failed_or_skipped": len(bad)})
        except (OSError, ET.ParseError) as error:
            failures.append("missing/invalid unit output: " + name + " " + str(error))

    def matrix(filename, prefix, expected):
        rows, found = [], []
        try:
            for line in (source / filename).read_text().splitlines():
                if line.startswith("FAIL"):
                    failures.append(filename + ": " + line)
                match = re.match(r"^PASS " + re.escape(prefix) + r" (\S+)(?: |$)", line)
                if match:
                    rows.append(line)
                    found.append(match.group(1))
        except OSError as error:
            failures.append("missing matrix: " + str(error))
        if set(found) != expected or len(found) != len(expected):
            failures.append("incomplete/duplicate matrix: " + filename)
        matrices[filename] = rows
        return rows

    if profile == "cpu":
        rows = matrix("cpu-driver.log", "GPW_CPU_DRIVER", CPU_CASES)
        mpi = matrix("cpu-driver-mpi2.log", "GPW_CPU_DRIVER", {c for c in CPU_CASES if c.startswith("k-")})
        def energies(lines):
            values = {}
            for line in lines:
                case = line.split()[2]
                values[case] = dict(re.findall(r"\b(E_Ha|shifted_E_Ha)=([-+\d.eE]+)", line))
            return values
        one, two = energies(rows), energies(mpi)
        for case in {c for c in CPU_CASES if c.startswith("k-")}:
            first, second = one.get(case, {}), two.get(case, {})
            if set(first) != {"E_Ha", "shifted_E_Ha"} or set(second) != set(first):
                failures.append("missing MPI energy fields: " + case)
            else:
                for field in first:
                    a, b = float(first[field]), float(second[field])
                    if not abs(a - b) <= 1e-8:
                        failures.append("MPI energy mismatch: " + case + " " + field)
        matrix("preflight.log", "GPW_PREFLIGHT", PREFLIGHT_CASES)
        matrix("oracles.log", "GPW_ORACLE", {"WATER_OT", "WATER_DIIS"})
        matrix("pbe-gpw.log", "GPW", {"PBE", "PBEsol", "revPBE", "RPBE"})
    elif profile == "gh200":
        matrix("cuda-driver.log", "GPW_CUDA_DRIVER", CUDA_CASES)
        matrix("pbe-gpw.log", "GPW", {"PBE", "PBEsol", "revPBE", "RPBE"})
        if uks_pbe:
            matrix("uks-driver.log", "GPW_CUDA_UKS_PBE_DRIVER", UKS_PBE_CASES)

    result = {
        "state": "passed" if not failures else "failed", "profile": profile,
        "slurm_job": job, "native_exit_code": exit_code,
        "formal_source_build_qualification": False,
        "device_execution_requested": profile == "gh200",
        "device_numerical_acceptance": profile == "gh200" and not failures,
        "unit_tests": sum(unit["tests"] for unit in units), "unit_executables": units,
        "public_cases": matrices, "failures": failures,
        "uks_pbe_driver_requested": uks_pbe,
        "scope": "Scoped GPW Driver/public method checks; original CP2K water pin on CPU; no new external-engine oracle",
    }
    (destination / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--profile", choices=("cpu", "cuda-build", "gh200"), required=True)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--uks-pbe", action="store_true",
                        help="Require all four families times H3/empty-beta nested cases")
    args = parser.parse_args()
    result = collect(args.source, args.destination, args.profile, args.job,
                     args.exit_code, args.uks_pbe)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["state"] == "passed" else 1)
