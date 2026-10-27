"""Compile unmodified QE 7.6 smearing routines and retain independent scalar pins."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

FILES = {
    "wgauss.f90": "Modules",
    "w0gauss.f90": "Modules",
    "w1gauss.f90": "Modules",
    "sumkg.f90": "PW/src",
    "efermig.f90": "PW/src",
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-dir", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--header", type=Path, required=True)
parser.add_argument("--compiler", default="gfortran")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
source = args.out / "sources"
source.mkdir()
provenance = {}
for name, folder in FILES.items():
    url = f"https://raw.githubusercontent.com/QEF/q-e/qe-7.6/{folder}/{name}"
    if args.source_dir:
        data = (args.source_dir / name).read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
    (source / name).write_bytes(data)
    provenance[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
local = Path(__file__).resolve().parent
compiler = shutil.which(args.compiler)
if compiler is None:
    raise RuntimeError("Fortran compiler unavailable")
binary = args.out.resolve() / "qe-smearing-oracle"
command = [
    compiler,
    "-O2",
    "-cpp",
    "-ffree-line-length-none",
    str(local / "qe_support.f90"),
    *[str((source / name).resolve()) for name in FILES],
    str(local / "qe_driver.f90"),
    "-o",
    str(binary),
]
result = subprocess.run(
    command, cwd=args.out, text=True, capture_output=True, check=False
)
(args.out / "build.log").write_text(result.stdout + result.stderr)
result.check_returncode()
result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
(args.out / "qe.out").write_text(result.stdout + result.stderr)
points, spectra = [], []
inputs = {
    1: ([-0.5, -0.3, 0.2, 0.4, 0.8], 4.0, 0.01, 2.0),
    2: ([-0.2, -0.08, 0.01, 0.11, 0.24], 3.1, 0.05, 2.0),
    3: ([-0.02, 0.02, 0.8, 1.0, 2.0], 2.0, 0.01, 2.0),
    4: ([-0.2, -0.08, 0.01, 0.11, 0.24], 2.7, 0.07, 1.0),
}


def mode(order):
    return (1, 1) if order == -99 else (3, 1) if order == 0 else (2, order)


for line in result.stdout.splitlines():
    words = line.split()
    if words and words[0] == "POINT":
        order = int(words[1])
        kind, mp = mode(order)
        x, f, entropy = map(float, words[2:])
        points.append(
            {
                "kind": kind,
                "mp_order": mp,
                "qe_order": order,
                "x": x,
                "occupation_spinless": f,
                "entropy_spinless": entropy,
            }
        )
    elif words and words[0] == "SPECTRUM":
        case, order = map(int, words[1:3])
        kind, mp = mode(order)
        mu, number, entropy, free_correction = map(float, words[3:])
        evals, target, width, degen = inputs[case]
        spectra.append(
            {
                "case": case,
                "kind": kind,
                "mp_order": mp,
                "qe_order": order,
                "evals": evals,
                "weights": [1.0] * 5,
                "target": target,
                "width": width,
                "spin_degen": degen,
                "mu": mu,
                "n_occ": number,
                "entropy": entropy,
                "free_energy_correction": free_correction,
            }
        )
if len(points) != 65 or len(spectra) != 20:
    raise RuntimeError("incomplete QE reference output")
evidence = {
    "source": "Actual compiled QE 7.6 wgauss/w1gauss/efermig/sumkg routines, no formula reimplementation",
    "scope": "one serial pool; support modules define kinds/constants and identity serial reductions only",
    "compile_command": command,
    "compiler_version": subprocess.run(
        [compiler, "--version"], text=True, capture_output=True, check=True
    ).stdout,
    "compiler_sha256": hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "source_files": provenance,
    "support_sha256": hashlib.sha256(
        (local / "qe_support.f90").read_bytes()
    ).hexdigest(),
    "driver_sha256": hashlib.sha256((local / "qe_driver.f90").read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scheduler": {
        key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST")
    },
    "points": points,
    "spectra": spectra,
}
(args.out / "reference.json").write_text(json.dumps(evidence, indent=2) + "\n")
header = [
    "// Actual QE 7.6 routine output; tools/diagnostics/pw-smearing/capture_qe.py.",
    "#pragma once",
    "#include <array>",
    "namespace qe_smearing_pin {",
    "struct Point { int kind, order; double x,f,s; };",
    f"inline constexpr std::array<Point,{len(points)}> points = {{{{",
]
for p in points:
    header.append(
        "    {"
        + f"{p['kind']},{p['mp_order']},"
        + ",".join(
            format(p[k], ".17g")
            for k in ("x", "occupation_spinless", "entropy_spinless")
        )
        + "},"
    )
header.extend(
    [
        "}};",
        "struct Spectrum { int id,kind,order; std::array<double,5> evals; double target,width,degen,mu,n,s; };",
        f"inline constexpr std::array<Spectrum,{len(spectra)}> spectra = {{{{",
    ]
)
for p in spectra:
    header.append(
        "    {"
        + f"{p['case']},{p['kind']},{p['mp_order']},"
        + "{{"
        + ",".join(format(v, ".17g") for v in p["evals"])
        + "}},"
        + ",".join(
            format(p[k], ".17g")
            for k in ("target", "width", "spin_degen", "mu", "n_occ", "entropy")
        )
        + "},"
    )
header.extend(["}};", "} // namespace qe_smearing_pin", ""])
args.header.write_text("\n".join(header))
print(
    "Captured",
    len(points),
    "local points and",
    len(spectra),
    "spectra from actual QE routines",
)
