"""Run real QE and retain H3 spin/smearing energy, forces and stress.

This file only launches the independent engine and parses its printed output.
The three widths probe a nonzero-spin, nonzero-entropy test envelope; a
collapsed magnetization remains recorded and cannot qualify that envelope.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def last_number(pattern, text):
    values = re.findall(pattern, text, flags=re.MULTILINE)
    return float(values[-1].replace("D", "E")) if values else None


def parse(text):
    stress_blocks = re.findall(
        r"total\s+stress\s+\(Ry/bohr\*\*3\).*\n([^\n]+)\n([^\n]+)\n([^\n]+)",
        text,
    )
    tensor = None
    if stress_blocks:
        tensor = [[float(v) for v in row.split()[:3]] for row in stress_blocks[-1]]
    force_blocks = re.findall(
        r"Forces acting on atoms[^\n]*\n(.*?)(?:\n\s*The |Total force|Computing stress)",
        text,
        flags=re.DOTALL,
    )
    forces = []
    if force_blocks:
        forces = [
            [float(v) for v in row]
            for row in re.findall(
                r"atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)",
                force_blocks[-1],
            )
        ]
    result = {
        "engine": next(
            (line.strip() for line in text.splitlines() if "Program PWSCF" in line),
            None,
        ),
        "converged": "convergence has been achieved" in text and "JOB DONE." in text,
        "energy_ry": last_number(r"^!\s+total energy\s*=\s*([-+\d.Ee]+)", text),
        "magnetization_e": last_number(r"total magnetization\s*=\s*([-+\d.Ee]+)", text),
        "entropy_energy_ry": last_number(
            r"smearing contrib\.\s*\(-TS\)\s*=\s*([-+\d.Ee]+)", text
        ),
        "stress_tensor_ry_bohr3": tensor,
        "forces_ry_bohr": forces,
    }
    if tensor:
        conversion = 13.605693122994 * 1.8897261245650618**3
        pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        result["stress_voigt_ev_ang3"] = [tensor[i][j] * conversion for i, j in pairs]
    result["spin_nontrivial"] = (
        result["magnetization_e"] is not None and abs(result["magnetization_e"]) > 1e-3
    )
    result["entropy_nontrivial"] = (
        result["entropy_energy_ry"] is not None
        and abs(result["entropy_energy_ry"]) > 1e-5
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--parse-existing", action="store_true")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Capture LSDA/PBE and all three declared smearing kinds",
    )
    parser.add_argument(
        "--mp-probe",
        action="store_true",
        help="Probe MP1 at 0.04/0.12 Ry with both functionals",
    )
    args = parser.parse_args()
    if args.matrix and args.mp_probe:
        parser.error("--matrix and --mp-probe are separate captures")
    if args.parse_existing:
        original = args.out / "reference.json"
        report = json.loads(original.read_text())
        report["original_capture_reference_sha256"] = sha(original)
        report["parser_sha256"] = sha(__file__)
        report["reparsed_utc"] = datetime.now(timezone.utc).isoformat()
        for item in report["cases"]:
            log = args.out / item["output"]
            if sha(log) != item["output_sha256"]:
                raise ValueError("Original oracle output changed before reparsing")
            item.update(parse(log.read_text()))
            item["accepted_external_output"] = (
                item["exit_code"] == 0
                and item["converged"]
                and item["energy_ry"] is not None
                and item["stress_tensor_ry_bohr3"] is not None
                and len(item["forces_ry_bohr"]) == 3
                and all(
                    math.isfinite(v)
                    for row in item["stress_tensor_ry_bohr3"]
                    for v in row
                )
                and all(math.isfinite(v) for row in item["forces_ry_bohr"] for v in row)
            )
        (args.out / "reference-parsed.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        return int(not all(c["accepted_external_output"] for c in report["cases"]))
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    binary = args.binary.resolve()
    template = (source / "h3.in").read_text()
    published = Path(
        "/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.2-gmpflf-2026.03/share/cp2k/data/GTH_POTENTIALS"
    )
    report = {
        "scope": "Actual QE spin-smearing energy/full-stress oracle capture; no native qualification implied",
        "utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "binary": str(binary),
        "binary_sha256": sha(binary),
        "script_sha256": sha(__file__),
        "input_template_sha256": sha(source / "h3.in"),
        "potential_sha256": sha(source / "H_GTH_PBE_q1.gth"),
        "published_potential_source": str(published),
        "published_potential_source_sha256": sha(published)
        if published.exists()
        else None,
        "environment": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
            ]
        },
        "bounds_fixed_before_native_comparison": {
            "energy_ha": 1e-5,
            "each_stress_ev_ang3": 2e-3,
        },
        "stress_convention": "compressive positive; physical Cartesian tensor; Voigt xx,yy,zz,xy,xz,yz",
        "cases": [],
    }
    failed = False
    cases = (
        [
            (xc, name, smear, width)
            for xc in ["LSDA", "PBE"]
            for name, smear, width in [
                ("fd", "fermi-dirac", 0.02),
                ("gaussian", "gaussian", 0.04),
                ("mp1", "methfessel-paxton", 0.02),
            ]
        ]
        if args.matrix
        else [("PBE", "fd", "fermi-dirac", w) for w in [0.12, 0.04, 0.02]]
    )
    if args.mp_probe:
        cases = [
            (xc, "mp1", "methfessel-paxton", w)
            for xc in ["LSDA", "PBE"]
            for w in [0.04, 0.12]
        ]
    for xc, name, smear, width in cases:
        case_id = (
            f"{xc.lower()}-{name}-{width:.2f}"
            if args.matrix or args.mp_probe
            else f"fd-{width:.2f}"
        )
        work = args.out / case_id
        work.mkdir()
        inp = work / "h3.in"
        functional = "SLA+VWN+NOGX+NOGC" if xc == "LSDA" else "PBE"
        inp.write_text(
            template.replace("degauss=0.12", f"degauss={width:.2f}")
            .replace("input_dft='PBE'", f"input_dft='{functional}'")
            .replace("smearing='fermi-dirac'", f"smearing='{smear}'")
        )
        shutil.copyfile(source / "H_GTH_PBE_q1.gth", work / "H_GTH_PBE_q1.gth")
        command = [str(binary), "-in", "h3.in"]
        log = work / "qe.out"
        try:
            with log.open("w") as stream:
                proc = subprocess.run(
                    command,
                    cwd=work,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
            data = parse(log.read_text())
            data.update(
                xc=xc,
                input_dft=functional,
                smearing=name,
                degauss_ry=width,
                command=command,
                exit_code=proc.returncode,
                input_sha256=sha(inp),
                output_sha256=sha(log),
                output=str(log.relative_to(args.out)),
            )
            data["accepted_external_output"] = (
                proc.returncode == 0
                and data["converged"]
                and data["energy_ry"] is not None
                and data["stress_tensor_ry_bohr3"] is not None
                and len(data["forces_ry_bohr"]) == 3
                and all(
                    math.isfinite(v)
                    for row in data["stress_tensor_ry_bohr3"]
                    for v in row
                )
                and all(math.isfinite(v) for row in data["forces_ry_bohr"] for v in row)
            )
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
            data = {
                "xc": xc,
                "smearing": name,
                "case_id": case_id,
                "degauss_ry": width,
                "accepted_external_output": False,
                "error": str(exc),
            }
        report["cases"].append(data)
        failed |= not data["accepted_external_output"]
        (args.out / "reference.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(data, allow_nan=False), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
