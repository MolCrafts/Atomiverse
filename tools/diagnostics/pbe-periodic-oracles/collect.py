#!/usr/bin/env python3
"""Collect exact public PW comparisons against hash-verified real QE pins."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("native", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--backend", required=True, choices=["CPU", "CUDA"])
p.add_argument("--job", required=True, type=int)
p.add_argument("--exit-code", required=True, type=int)
a = p.parse_args()
root = Path(__file__).resolve().parent
errors = []
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def cpp_tokens(text):
    # Formatting generated pins must not change their provenance. Compare
    # every C++ token, preserving string/numeric spellings, and ignore comments.
    pattern = (r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|[A-Za-z_]\w*|'
               r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|\S')
    return [token for token in re.findall(pattern, text, flags=re.S)
            if not token.startswith(("//", "/*"))]
reference = root / "qe-2026-10-27/reference.json"
data = json.loads(reference.read_text())
expected = {system + "-" + model for system in ("si2-rks", "h3-uks")
            for model in ("pbesol", "revpbe", "rpbe")}
with tempfile.TemporaryDirectory() as tmp:
    generated = Path(tmp) / "pins.h"
    export = subprocess.run([sys.executable, str(root / "export_pins.py"),
                             str(reference), str(generated)], capture_output=True, text=True)
    if export.returncode or cpp_tokens(generated.read_text()) != cpp_tokens((root / "pins.h").read_text()):
        errors.append("actual QE inputs/outputs/potentials or exported pins failed integrity checks")
if a.exit_code:
    errors.append(f"native command exited {a.exit_code}")
log = (a.native / "main.log").read_text()
pattern = re.compile(r"^PASS PBE_PERIODIC_QE (\S+) (CPU|CUDA) E_Ha=(\S+) dE_Ha=(\S+) "
                     r"dF_Ha_Bohr=(\S+) dStress_eV_A3=(\S+) iterations=(\d+)$", re.M)
records = []
for match in pattern.finditer(log):
    identity, backend, energy, de, df, ds, iterations = match.groups()
    record = dict(id=identity, backend=backend, energy_ha=float(energy),
                  energy_error_ha=float(de), force_error_ha_bohr=float(df),
                  stress_error_ev_ang3=float(ds), iterations=int(iterations))
    records.append(record)
    if not all(math.isfinite(record[key]) for key in
               ("energy_ha", "energy_error_ha", "force_error_ha_bohr", "stress_error_ev_ang3")):
        errors.append(f"nonfinite result {identity}")
    bounds = data["predeclared_bounds"]
    if (record["energy_error_ha"] > bounds["energy_ha"]
            or record["force_error_ha_bohr"] > bounds["each_force_ha_bohr"]
            or record["stress_error_ev_ang3"] > bounds["each_stress_ev_ang3"]):
        errors.append(f"original physical gate failed {identity}")
if (len(records) != 6 or {r["id"] for r in records} != expected
        or any(r["backend"] != a.backend for r in records)):
    errors.append("missing, duplicate, unexpected or wrong-backend case")
if "FAIL PBE_PERIODIC_QE" in log:
    errors.append("native comparison reported failure")
for filename in ("source-before.json", "source-after.json", "binaries.sha256", "hardware.txt"):
    if not (a.native / filename).is_file(): errors.append(f"missing provenance {filename}")
if not errors:
    if json.loads((a.native / "source-before.json").read_text()) != json.loads((a.native / "source-after.json").read_text()):
        errors.append("source changed during native acceptance")
    hardware = (a.native / "hardware.txt").read_text()
    if a.backend == "CUDA" and ("aarch64" not in hardware or "GH200" not in hardware):
        errors.append("native CUDA acceptance requires the recorded GH200")
result = dict(passed=not errors, errors=errors, backend=a.backend, job_id=a.job,
              native_directory=str(a.native), independent_engine="Quantum ESPRESSO 7.6",
              scope="Gamma FD-smearing PBEsol/revPBE/RPBE: Si2 RKS and skew H3 UKS, absolute energy, every force and six-component numerical strain-FD stress",
              external_reference_sha256=sha(reference), records=records,
              formal_source_qualification=False, formal_build_qualification=False,
              general_functional_qualification=False,
              limits=["No closed-form stress claim", "No GPW or molecular CUDA qualification", "No k-mesh, Gaussian/MP or other-potential qualification"])
a.output.mkdir(parents=True, exist_ok=True)
for path in a.native.iterdir():
    if path.is_file() and path.suffix in (".log", ".json", ".sha256", ".txt", ".sh"):
        shutil.copy2(path, a.output / path.name)
(a.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"passed": result["passed"], "errors": errors, "cases": len(records)}))
raise SystemExit(0 if result["passed"] else 1)
