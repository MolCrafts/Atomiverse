"""Replay diagnostic-only CPU PW SCF spectra through actual QE efermig."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--qe-source-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(exist_ok=False, parents=True)
local = Path(__file__).resolve().parent
spectra = []
sample = "unlabelled"
solve_index = -1
for line in args.trace.read_text().splitlines():
    if line.startswith("STRAIN_SAMPLE"):
        sample = line
    elif line.startswith("TRACE_SPECTRUM "):
        point = json.loads(line.split(" ", 1)[1])
        if point["iter"] == 0:
            solve_index += 1
        spectra.append({"sample": sample, "solve_index": solve_index, **point})
    elif line.startswith("TRACE_RESIDUAL ") and spectra:
        spectra[-1].update(json.loads(line.split(" ", 1)[1]))
if not spectra:
    raise RuntimeError("No trace spectra")
inputs = [str(len(spectra))]
for i, point in enumerate(spectra):
    inputs.extend(
        [
            f"{i} {len(point['evals_ha'])} {point['target']} {point['width_ha']:.17g} 1 1 {point['mu_ha']:.17g}",
            " ".join(format(x, ".17g") for x in point["evals_ha"]),
            " ".join(format(x, ".17g") for x in point["weights"]),
        ]
    )
input_data = "\n".join(inputs) + "\n"
(args.out / "input.txt").write_text(input_data)
compiler = shutil.which("gfortran")
if not compiler:
    raise RuntimeError("Fortran compiler unavailable")
sources = [
    args.qe_source_dir.resolve() / name
    for name in ("wgauss.f90", "w0gauss.f90", "w1gauss.f90", "sumkg.f90", "efermig.f90")
]
binary = args.out.resolve() / "qe-replay"
command = [
    compiler,
    "-O2",
    "-cpp",
    "-ffree-line-length-none",
    str(local / "qe_support.f90"),
    *map(str, sources),
    str(local / "replay_qe.f90"),
    "-o",
    str(binary),
]
build = subprocess.run(
    command, cwd=args.out, text=True, capture_output=True, check=False
)
(args.out / "build.log").write_text(build.stdout + build.stderr)
build.check_returncode()
run = subprocess.run(
    [str(binary)], input=input_data, text=True, capture_output=True, check=True
)
(args.out / "qe.out").write_text(run.stdout + run.stderr)
replayed = 0
for line in run.stdout.splitlines():
    if not line.startswith("REPLAY "):
        continue
    fields = line.split()
    point = spectra[int(fields[1])]
    point["qe_replay"] = dict(
        zip(
            (
                "mu",
                "gaussian_mu",
                "n_at_native_mu",
                "entropy_at_native_mu",
                "dN_dmu_at_native_mu",
                "native_mu",
            ),
            map(float, fields[2:]),
            strict=True,
        )
    )
    replayed += 1
if replayed != len(spectra):
    raise RuntimeError("Incomplete QE replay")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


summary = {
    "spectra_count": len(spectra),
    "max_abs_mu_native_vs_actual_qe": max(
        abs(p["mu_ha"] - p["qe_replay"]["mu"]) for p in spectra
    ),
    "max_abs_number_residual_at_native_mu": max(
        abs(p["target"] - p["qe_replay"]["n_at_native_mu"]) for p in spectra
    ),
    "min_abs_dN_dmu_at_native_mu": min(
        abs(p["qe_replay"]["dN_dmu_at_native_mu"]) for p in spectra
    ),
    "negative_slope_count": sum(
        p["qe_replay"]["dN_dmu_at_native_mu"] < 0 for p in spectra
    ),
    "scheduler": {
        key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST")
    },
    "compile_command": command,
    "files_sha256": {
        str(p): sha(p)
        for p in [
            args.trace,
            Path(compiler),
            binary,
            Path(__file__),
            local / "replay_qe.f90",
            local / "qe_support.f90",
            *sources,
        ]
    },
}
(args.out / "replay.json").write_text(
    json.dumps({"summary": summary, "spectra": spectra}, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
