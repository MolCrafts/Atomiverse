#!/usr/bin/env python3
"""Collect this diagnostic's native outputs without running oracles or solves."""

import argparse
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET


def collect(source, destination, profile, job, exit_code, envelope="pbe-family"):
    expected_units = (
        "test_cuda_pw_xc_pbe",
        "test_cuda_pw_xc_pbe_spin",
        "test_cuda_pw_guards",
        "test_cuda_pw_density",
        "test_cuda_pw_force_local",
        "test_cuda_pw_cell_stress",
        "test_cuda_gpw_ks_gga",
        "test_cuda_gpw_stress",
    )
    destination.mkdir(parents=True, exist_ok=True)
    failures = []
    if exit_code:
        failures.append("native build/run exit code: %d" % exit_code)
    baselines = ("pw-cell-stress-baselines",) if envelope == "pw-driver" else ()
    units = []
    complete_methods = []
    for name in expected_units + baselines:
        path = source / (name + ".xml")
        if not path.exists():
            failures.append("missing native unit output: " + name)
            continue
        shutil.copyfile(path, destination / path.name)
        try:
            root = ET.parse(path).getroot()
            counts = {
                key: sum(int(suite.get(key, "0")) for suite in root)
                for key in ("tests", "failures", "errors", "skipped", "disabled")
            }
            group = complete_methods if name in baselines else units
            group.append(dict(name=name, **counts))
            if counts["tests"] < 1 or any(
                counts[key] for key in ("failures", "errors", "skipped", "disabled")
            ):
                failures.append("unit tests failed, skipped or empty: " + name)
            actual = [
                case.get("classname", "") + "." + case.get("name", "")
                for case in root.iter("testcase")
            ]
            if name in baselines and set(actual) != {
                "CudaPWCellStress.H2ClosedShellCpuTwin",
                "CudaPWCellStress.HSmearRestrictedFinite",
                "CudaPWCellStress.H2ClosedShellPbeCpuTwin",
            }:
                failures.append("original complete-SCF cases missing: " + name)
            manifest = source / (name + ".list")
            if not manifest.exists():
                failures.append("full test inventory missing: " + name)
                continue
            shutil.copyfile(manifest, destination / manifest.name)
            expected = []
            suite_name = ""
            for line in manifest.read_text().splitlines():
                line = line.split("#", 1)[0].rstrip()
                if re.fullmatch(r"[\w/]+\.", line):
                    suite_name = line[:-1]
                elif suite_name and re.fullmatch(r"  [\w/]+", line):
                    expected.append(suite_name + "." + line.strip())
            if (
                not expected
                or set(actual) != set(expected)
                or len(actual) != len(expected)
                or len(actual) != counts["tests"]
            ):
                failures.append("filtered or incomplete test inventory: " + name)
        except (ET.ParseError, ValueError) as error:
            failures.append("invalid unit output %s: %s" % (name, error))

    functionals = {"PBE", "PBEsol", "revPBE", "RPBE"}
    matrices = {}
    expected_matrices = [
        (
            "pw",
            "pw-public.log",
            r"^PASS (PBE|PBEsol|revPBE|RPBE) (RKS|UKS) .*twin_energy_error=",
            {(name, spin) for name in functionals for spin in ("RKS", "UKS")},
        ),
        (
            "gpw",
            "gpw-public.log",
            r"^PASS GPW (PBE|PBEsol|revPBE|RPBE) CPU_CUDA ",
            {(name,) for name in functionals},
        ),
    ]
    if envelope == "pw-driver":
        expected_matrices.extend([
            (
                "pw_fd",
                "pw-fd-public.log",
                r"^PASS (PBE|PBEsol|revPBE|RPBE) (RKS|UKS) .*twin_energy_error=",
                {(name, spin) for name in functionals for spin in ("RKS", "UKS")},
            ),
            (
                "pw_driver",
                "pw-driver.log",
                r"^PASS PW_DRIVER (fixed|fd) (RKS|UKS) ",
                {(mode, spin) for mode in ("fixed", "fd") for spin in ("RKS", "UKS")},
            ),
            (
                "pw_existing_qe",
                "pw-existing-public.log",
                r"^PASS (PBE H (?:RKS|UKS) (?:smear|fixed)(?: truncated/reuse metadata)?|PBE Si nonzero forces|actual QE7\.5 Si2 CUDA Mermin)(?= iterations=|$| max\|F\|=| E=)",
                {(f"PBE H {spin} {mode}{suffix}",)
                 for spin in ("RKS", "UKS") for mode in ("fixed", "smear")
                 for suffix in ("", " truncated/reuse metadata")}
                | {("PBE Si nonzero forces",), ("actual QE7.5 Si2 CUDA Mermin",)},
            ),
        ])
    for kind, filename, pattern, expected in expected_matrices:
        path = source / filename
        rows = []
        found = []
        if path.exists():
            shutil.copyfile(path, destination / filename)
            for line in path.read_text().splitlines():
                match = re.search(pattern, line)
                if match:
                    found.append(match.groups())
                    rows.append(line)
                if line.startswith("FAIL "):
                    failures.append(line)
        if set(found) != expected or len(found) != len(expected):
            failures.append("incomplete or duplicate %s CPU/CUDA matrix" % kind)
        matrices[kind] = rows

    hardware = source / "hardware.txt"
    if hardware.exists():
        shutil.copyfile(hardware, destination / hardware.name)
        hardware_text = hardware.read_text()
    else:
        hardware_text = ""
    if profile == "gh200" and "GH200" not in hardware_text:
        failures.append("GH200 hardware identification missing")
    for name in ("source-before.json", "source-after.json", "binaries.sha256"):
        path = source / name
        if path.exists():
            shutil.copyfile(path, destination / name)
        else:
            failures.append("missing provenance output: " + name)
    for path in source.glob("*.log"):
        shutil.copyfile(path, destination / path.name)
    for name in ("submission.sh", "CMakeCache.txt"):
        path = source / name
        if path.exists():
            shutil.copyfile(path, destination / name)
    before, after = source / "source-before.json", source / "source-after.json"
    if not before.exists() or not after.exists():
        failures.append("build/run source fingerprints missing")
    else:
        try:
            first, last = json.loads(before.read_text()), json.loads(after.read_text())
            if not isinstance(first, dict) or not first or first != last:
                failures.append("diagnostic source changed or fingerprint is empty")
        except (OSError, ValueError) as error:
            failures.append("invalid source fingerprint: " + str(error))

    summary = {
        "state": "passed" if not failures else "failed",
        "profile": profile,
        "envelope": envelope,
        "slurm_job": job,
        "native_exit_code": exit_code,
        "formal_source_build_qualification": False,
        "scope": "native units and CPU/CUDA consistency; no new external oracle",
        "unit_tests": sum(unit["tests"] for unit in units),
        "unit_executables": units,
        "complete_method_tests": sum(item["tests"] for item in complete_methods),
        "complete_method_executables": complete_methods,
        "public_cases": matrices,
        "failures": failures,
    }
    (destination / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--profile", choices=("gh200",), required=True)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--envelope", choices=("pbe-family", "pw-driver"),
                        default="pbe-family")
    args = parser.parse_args()
    result = collect(
        args.source, args.destination, args.profile, args.job, args.exit_code,
        args.envelope,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["state"] == "passed" else 1)
