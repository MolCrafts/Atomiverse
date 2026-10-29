#!/usr/bin/env python3
"""Collect narrow GPW/actual-CP2K evidence, including complete failed cases."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
IDS = tuple("gpw-h2-rks-" + model for model in ("pbesol", "revpbe", "rpbe"))
BOUNDS = {"energy_ha": 5e-6, "each_force_ha_bohr": 5e-5,
          "each_supported_diagonal_stress_ha_bohr3": 5e-5}
# Unit conversion only, identical to core/units.h and the public consumer.
GPA_PER_HA_BOHR3 = 160.2176634 * 27.211386245988 * 1.8897261245650618 ** 3
TOKEN = (r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|[A-Za-z_]\w*|'
         r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|\S')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpp_tokens(text):
    return [token for token in re.findall(TOKEN, text, re.S)
            if not token.startswith(("//", "/*"))]


def inventory(path):
    names, suite = [], ""
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if re.fullmatch(r"[\w/]+\.", line):
            suite = line[:-1]
        elif suite and re.fullmatch(r"  [\w/]+", line):
            names.append(suite + "." + line.strip())
    return names


def external_reference(path, errors, expected_ids=IDS, atom_count=2,
                       export_name="export_cp2k_pins.py", pins_name="cp2k_pins.h",
                       require_spin=False):
    data = json.loads(path.read_text())
    cases = data.get("cases", [])
    if (data.get("state") != "captured" or len(cases) != len(expected_ids)
            or {c.get("id") for c in cases} != set(expected_ids)):
        errors.append("external reference must contain exactly three accepted cases" if expected_ids == IDS
                      else "external reference case set mismatch")
    if data.get("predeclared_bounds") != BOUNDS:
        errors.append("external reference changed the original physical gates")
    if not re.fullmatch(r"[0-9a-f]{64}", data.get("binary_sha256", "")) or sha(Path(data["binary"])) != data["binary_sha256"]:
        errors.append("actual external CP2K binary hash mismatch")
    for key in ("basis_data", "potential_data"):
        if sha(path.parent / data[key + "_snapshot"]) != data[key + "_sha256"]:
            errors.append("external data snapshot hash mismatch: " + key)
    if sha(path.parent / data["exact_capture_source"]) != data["script_sha256"]:
        errors.append("actual external launch-source hash mismatch")
    spec = importlib.util.spec_from_file_location("actual_cp2k_capture", HERE / "capture_cp2k.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    for case in cases:
        identity = case["id"]
        for key in ("input", "output"):
            if sha(path.parent / case[key]) != case[key + "_sha256"]:
                errors.append(identity + " external " + key + " hash mismatch")
        raw = (path.parent / case["output"]).read_text()
        parsed = parser.parse(raw)
        if atom_count != 2:
            parsed["forces_ha_bohr"] = [[float(x) for x in row] for row in re.findall(
                r"^\s*FORCES\|\s*\d+\s+([-+\d.EeDd]+)\s+([-+\d.EeDd]+)\s+([-+\d.EeDd]+)\s+[-+\d.EeDd]+", raw, re.M)][-atom_count:]
            values = ([parsed["energy_ha"]] if parsed["energy_ha"] is not None else []) + [v for row in parsed["forces_ha_bohr"] for v in row] + ([v for row in parsed["stress_tensor_gpa"] for v in row] if parsed["stress_tensor_gpa"] else [])
            parsed["complete_printed_quantities"] = len(parsed["forces_ha_bohr"]) == atom_count and len(values) == 1 + 3*atom_count + 9 and all(math.isfinite(v) for v in values)
        if require_spin:
            counts = [int(x) for x in re.findall(r"Number of electrons:\s+(\d+)", raw)]
            grid = [int(x) for x in re.findall(r"PW_GRID\|\s+Bounds\s+[123]\s+[-\d]+\s+[-\d]+\s+Points:\s+(\d+)", raw)[:3]]
            if (counts != [2, 1]*20 or case.get("all_printed_spin_electron_counts") != counts
                    or case.get("n_alpha") != 2 or case.get("n_beta") != 1
                    or grid != [36, 32, 30] or case.get("initial_grid_points") != grid
                    or "Spin unrestricted (spin-polarized) Kohn-Sham calculation" not in raw):
                errors.append(identity + " actual UKS spin/grid contract mismatch")
        for key in ("energy_ha", "forces_ha_bohr", "stress_tensor_gpa", "engine", "converged"):
            if parsed[key] != case[key]:
                errors.append(identity + " raw-output quantity mismatch: " + key)
        scfs = len(re.findall(r"SCF run converged in\s+\d+\s+steps", raw))
        if (not case.get("accepted_external_output") or case.get("exit_code") != 0
                or not parsed["converged"] or not parsed["complete_printed_quantities"]
                or scfs != 20 or case.get("scf_converged_runs") != 20):
            errors.append(identity + " requires a completed CP2K numerical-stress run with 20 converged SCFs")
        if ("STRESS| Numerical stress tensor [bar]" not in raw
                or not all("FUNCTIONAL| " + xc + ":" in parsed["printed_xc"]
                           for xc in case["xc_sections"])):
            errors.append(identity + " actual XC/numerical-stress label mismatch")
    with tempfile.TemporaryDirectory() as tmp:
        header = Path(tmp) / "cp2k_pins.h"
        run = subprocess.run([sys.executable, str(HERE / export_name),
                              str(path), str(header)], capture_output=True, text=True)
        if (run.returncode or not header.is_file()
                or cpp_tokens(header.read_text()) != cpp_tokens((HERE / pins_name).read_text())):
            errors.append("hash-verified external pins differ at the C++ token level")
    return {case["id"]: case for case in cases}


def parse_native(log, backend, reference, errors, expected_ids=IDS, atom_count=2,
                 protocol="PBE_PERIODIC_CP2K", force_protocol="GPW_CP2K_FORCE",
                 stress_protocol="GPW_CP2K_DIAG_STRESS", require_spin=False):
    summaries, forces, stresses = {}, {}, {}
    for line in log.splitlines():
        backend_pattern = "CPU|CUDA|cpu|cuda" if require_spin else "CPU|CUDA"
        match = re.fullmatch(rf"(PASS|FAIL) {re.escape(protocol)} (\S+) ({backend_pattern}) (.*)", line)
        kind = "summary"
        if not match:
            match = re.fullmatch(rf"({re.escape(force_protocol)}|{re.escape(stress_protocol)}) (\S+) (CPU|CUDA) (.*)", line)
            if match:
                kind = "FORCE" if match[1] == force_protocol else "DIAG_STRESS"
        if not match:
            if (protocol in line or force_protocol in line or stress_protocol in line
                    or (protocol == "PBE_PERIODIC_CP2K" and "GPW_CP2K_" in line)):
                errors.append("unparsed native case line: " + line)
            continue
        first, identity, actual_backend, payload = match.groups()
        if require_spin:
            actual_backend = actual_backend.upper()
        if identity not in expected_ids or actual_backend != backend:
            errors.append("unexpected case/backend: " + identity + " " + actual_backend)
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", payload))
        if kind == "summary":
            target, key = summaries, identity
            fields["reported_status"] = first
            if require_spin and first == "FAIL" and len(fields) == 1:
                fields["native_error_message"] = payload
        elif kind == "FORCE":
            if fields.get("atom") not in tuple(str(i) for i in range(atom_count)):
                errors.append("unexpected atom index: " + identity + " " + str(fields.get("atom")))
                continue
            target, key = forces, (identity, fields.get("atom"))
        else:
            target, key = stresses, identity
        if key in target:
            errors.append("duplicate native " + kind + " " + str(key))
        target[key] = fields
    records = []
    for identity in expected_ids:
        local = []
        case, summary = reference.get(identity), summaries.get(identity, {})
        record = {"id": identity, "backend": backend,
                  "reported_status": summary.get("reported_status", "MISSING"),
                  "summary": summary, "forces_ha_bohr": [], "diagonal_stress_ha_bohr3": []}
        if not summary:
            local.append("missing case summary")
        if summary.get("reported_status") != "PASS":
            local.append("native case did not report PASS")

        def numeric(fields, key):
            try:
                value = float(fields[key])
                if not math.isfinite(value):
                    raise ValueError("nonfinite")
                return value
            except (KeyError, ValueError, TypeError):
                local.append("missing/nonfinite numeric field " + key)
                return None

        energy = numeric(summary, "E_Ha")
        record["energy_ha"] = energy
        for key in ("dE_Ha", "dF_Ha_Bohr", "dDiagStress_Ha_Bohr3",
                    "failed_force_lanes", "failed_stress_lanes", "iterations"):
            value = numeric(summary, key)
            if value is not None and value < 0:
                local.append("negative summary field " + key)
        for key, bound in (("dE_Ha", BOUNDS["energy_ha"]),
                           ("dF_Ha_Bohr", BOUNDS["each_force_ha_bohr"]),
                           ("dDiagStress_Ha_Bohr3", BOUNDS["each_supported_diagonal_stress_ha_bohr3"])):
            value = numeric(summary, key)
            if value is not None and value > bound:
                local.append("reported original physical gate failed: " + key)
        for key in ("failed_force_lanes", "failed_stress_lanes", "iterations"):
            value = numeric(summary, key)
            if value is not None and not value.is_integer():
                local.append("noninteger count: " + key)
        for atom in range(atom_count):
            fields = forces.get((identity, str(atom)), {})
            record["forces_ha_bohr"].append([numeric(fields, "F" + axis + "_Ha_Bohr")
                                                    for axis in "xyz"])
        fields = stresses.get(identity, {})
        record["diagonal_stress_ha_bohr3"] = [numeric(fields, axis + "_Ha_Bohr3")
                                             for axis in ("xx", "yy", "zz")]
        if case:
            record["reference_energy_ha"] = case["energy_ha"]
            record["reference_forces_ha_bohr"] = case["forces_ha_bohr"]
            expected_stress = [case["stress_tensor_gpa"][i][i] / GPA_PER_HA_BOHR3 for i in range(3)]
            record["reference_diagonal_stress_ha_bohr3"] = expected_stress
            de = abs(energy - case["energy_ha"]) if energy is not None else None
            df = [[abs(value - case["forces_ha_bohr"][atom][axis]) if value is not None else None
                   for axis, value in enumerate(row)] for atom, row in enumerate(record["forces_ha_bohr"])]
            ds = [abs(value - expected_stress[i]) if value is not None else None
                  for i, value in enumerate(record["diagonal_stress_ha_bohr3"])]
            record.update(energy_error_ha=de, each_force_error_ha_bohr=df,
                          each_diagonal_stress_error_ha_bohr3=ds)
            comparisons = [("dE_Ha", de, BOUNDS["energy_ha"]),
                           ("dF_Ha_Bohr", max(v for row in df for v in row) if all(v is not None for row in df for v in row) else None, BOUNDS["each_force_ha_bohr"]),
                           ("dDiagStress_Ha_Bohr3", max(ds) if all(v is not None for v in ds) else None, BOUNDS["each_supported_diagonal_stress_ha_bohr3"])]
            for key, actual, bound in comparisons:
                if actual is not None:
                    if actual > bound:
                        local.append("original physical gate failed: " + key)
                    reported = numeric(summary, key)
                    if reported is not None and abs(actual - reported) > 1e-12:
                        local.append("summary disagrees with full native quantities: " + key)
            for key, lanes, bound in (("failed_force_lanes", [v for row in df for v in row], BOUNDS["each_force_ha_bohr"]),
                                      ("failed_stress_lanes", ds, BOUNDS["each_supported_diagonal_stress_ha_bohr3"])):
                if all(v is not None for v in lanes):
                    actual = sum(v > bound for v in lanes)
                    if numeric(summary, key) != actual:
                        local.append("summary failed-lane count mismatch: " + key)
        else:
            local.append("missing independent reference case")
        if require_spin:
            alpha, beta, spin, entropy = [numeric(summary, key) for key in
                                          ("n_alpha", "n_beta", "spin", "e_entropy_Ha")]
            record.update(alpha_overlap_trace=alpha, beta_overlap_trace=beta,
                          total_spin=spin, entropy_ha=entropy)
            if (alpha is None or beta is None or spin is None or entropy is None
                    or abs(alpha-2.) > 1e-8 or abs(beta-1.) > 1e-8
                    or abs(spin-1.) > 1e-8 or abs(spin-(alpha-beta)) > 1e-8 or entropy != 0):
                local.append("integer UKS overlap traces/spin/entropy contract failed")
        if not numeric(summary, "iterations"):
            local.append("native converged SCF requires positive iteration count")
        record.update(passed=not local, errors=sorted(set(local)))
        records.append(record)
        errors.extend(identity + ": " + message for message in record["errors"])
    return records


def provenance(native, backend, xmls, lists, errors,
               consumer_binary="pbe-periodic-cp2k-oracles",
               consumer_sources=("cp2k.cpp", "cp2k_pins.h", "export_cp2k_pins.py", "capture_cp2k.py", "collect_cp2k.py")):
    result = {"files": {}, "unit_suites": []}
    for name in ("source-before.json", "source-after.json", "binaries.sha256", "hardware.txt"):
        path = native / name
        if path.is_file():
            result["files"][name] = sha(path)
        else:
            errors.append("missing native provenance: " + name)
    try:
        before = json.loads((native / "source-before.json").read_text())
        after = json.loads((native / "source-after.json").read_text())
        if not before or before != after:
            errors.append("empty or changing native source fingerprint")
        required = ["include/atomiverse/electronic/gaussian_core.h",
                    "include/atomiverse/cpu/gpw/internal/core_overlap.h",
                    "src/cpu/gpw/gpw_solver.cpp", "src/cpu/gpw/internal/gpw_forces.cpp",
                    "src/cpu/gpw/internal/gpw_stress.cpp"]
        required += [str((HERE / name).relative_to(REPO)) for name in consumer_sources]
        if backend == "CUDA":
            required += ["src/cuda/gpw/gpw_solver.cu", "src/cuda/gpw/internal/gpw_forces.cu"]
        for name in required:
            if name not in before or before[name] != sha(REPO / name):
                errors.append("required source hash missing/mismatched: " + name)
        result["source_entry_count"] = len(before)
    except (OSError, ValueError, TypeError) as exc:
        errors.append("invalid source provenance: " + str(exc))
    try:
        entries = []
        for line in (native / "binaries.sha256").read_text().splitlines():
            match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
            if not match:
                errors.append("invalid binary fingerprint line: " + line)
                continue
            digest, filename = match.groups()
            path = Path(filename)
            if not path.is_absolute():
                path = REPO / path
            entries.append({"path": filename, "sha256": digest})
            if not path.is_file() or sha(path) != digest:
                errors.append("native binary no longer matches recorded hash: " + filename)
        if not any(Path(entry["path"]).name == consumer_binary for entry in entries):
            errors.append("CP2K consumer binary fingerprint missing")
        if not any(Path(entry["path"]).name == "libatomiverse.a" for entry in entries):
            errors.append("native library fingerprint missing")
        result["binaries"] = entries
        hardware = (native / "hardware.txt").read_text().strip()
        if not hardware or (backend == "CUDA" and ("GH200" not in hardware or "aarch64" not in hardware)):
            errors.append("missing hardware identity or required GH200/aarch64")
        result["hardware"] = hardware
    except OSError as exc:
        errors.append("invalid binary/hardware provenance: " + str(exc))
    if not xmls:
        xmls = sorted(native.glob("*.xml"))
    if lists and len(lists) != len(xmls):
        errors.append("unit XML/inventory argument count mismatch")
    if not xmls:
        errors.append("native unit XML provenance missing")
    for i, xml in enumerate(xmls):
        try:
            listing = lists[i] if lists and i < len(lists) else xml.with_suffix(".list")
            expected = inventory(listing)
            document = ET.parse(xml).getroot()
            cases = list(document.iter("testcase"))
            if any(int(node.get(key, "0")) for node in document.iter()
                   for key in ("failures", "errors", "disabled")):
                errors.append("native unit XML declares failed/error/disabled tests: " + str(xml))
            actual = [case.get("classname", "") + "." + case.get("name", "") for case in cases]
            failures = [actual[n] for n, case in enumerate(cases)
                        if case.find("failure") is not None or case.find("error") is not None or case.find("skipped") is not None
                        or case.get("status") == "notrun"]
            if not expected or len(expected) != len(set(expected)) or sorted(actual) != sorted(expected) or failures:
                errors.append("native unit XML/inventory mismatch or failed/skipped tests: " + str(xml))
            result["unit_suites"].append({"xml": str(xml), "xml_sha256": sha(xml),
                                           "inventory": str(listing), "inventory_sha256": sha(listing),
                                           "test_count": len(cases), "cases": actual, "failures": failures})
        except (OSError, ValueError, ET.ParseError) as exc:
            errors.append("invalid native unit evidence: " + str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", required=True, choices=("CPU", "CUDA"))
    parser.add_argument("--job", required=True, type=int)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--log", default="main.log")
    parser.add_argument("--unit-xml", type=Path, action="append", default=[])
    parser.add_argument("--unit-list", type=Path, action="append", default=[])
    args = parser.parse_args()
    errors, reference = [], HERE / "cp2k-2026-10-27/reference.json"
    cases = {}
    try:
        cases = external_reference(reference, errors)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        errors.append("invalid independent reference: " + str(exc))
    log_path = args.native / args.log
    try:
        log = log_path.read_text()
    except OSError as exc:
        errors.append("missing native log: " + str(exc))
        log = ""
    records = parse_native(log, args.backend, cases, errors)
    if args.exit_code:
        errors.append("native command exited " + str(args.exit_code))
    native_provenance = provenance(args.native, args.backend, args.unit_xml, args.unit_list, errors)
    result = {"passed": not errors, "errors": errors, "backend": args.backend,
              "job_id": args.job, "native_directory": str(args.native),
              "native_log": str(log_path), "native_log_sha256": sha(log_path) if log_path.is_file() else None,
              "independent_engine": "CP2K 2026.2", "external_reference_sha256": sha(reference),
              "external_stress_method": "CP2K NUMERICAL; 20 converged SCFs per model",
              "scope": "Periodic orthorhombic H2 Gamma RKS; one 80Ry grid; DZV-GTH-PADE/GTH-PBE-q1; PBEsol/revPBE/RPBE; absolute E, both atoms' Cartesian forces, three diagonal physical stresses",
              "original_physical_gates": BOUNDS, "records": records,
              "native_provenance": native_provenance,
              "formal_source_qualification": False, "formal_build_qualification": False,
              "general_functional_qualification": False,
              "limits": ["No GPW shear qualification: only xx/yy/zz supported and checked",
                         "CP2K stress is full-SCF numerical strain differentiation; no analytical-stress oracle claim",
                         "No UKS, multigrid, k-mesh, smearing or different-potential qualification",
                         "Actual CP2K analytical CORE_PPL GRID stress abort retained separately"]}
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.native.iterdir():
        if path.is_file() and path.suffix in (".log", ".json", ".sha256", ".txt", ".sh", ".xml", ".list"):
            destination = args.output / path.name
            if path.resolve() != destination.resolve():
                shutil.copy2(path, destination)
    unit_paths = [Path(suite[key]) for suite in native_provenance["unit_suites"] for key in ("xml", "inventory")]
    for path in unit_paths:
        if path.is_file() and path.resolve() != (args.output / path.name).resolve():
            shutil.copy2(path, args.output / path.name)
    (args.output / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "cases": len(records), "errors": errors}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
