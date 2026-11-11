"""Run a fresh public ATV solve or live independent oracle in each sample."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.framework.comparison import compare_values
from benchmarks.framework.provenance import file_hash
from benchmarks.framework.specification import Participant, Row, System
from benchmarks.periodic.cases import ROOT
from benchmarks.scf.hf import (
    BOHR_IN_ANGSTROM,
    HARTREE_IN_EV,
    BindingMissing,
    NotConverged,
    ScientificFailure,
)

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
TRIPLE = rf"({NUMBER})\s+({NUMBER})\s+({NUMBER})"
STRESS_AU_IN_EV_ANG3 = HARTREE_IN_EV / BOHR_IN_ANGSTROM**3


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _required(pattern: str, output: str, label: str) -> re.Match:
    match = re.search(pattern, output, re.MULTILINE)
    if match is None:
        raise ScientificFailure(f"Missing {label} in engine output")
    return match


def _observables(energy: float, forces: list, stress: list, n: int, lanes: int) -> dict:
    if (
        len(forces) != n
        or any(len(f) != 3 for f in forces)
        or len(stress) != lanes
        or not all(
            math.isfinite(x) for x in [energy, *stress, *(x for f in forces for x in f)]
        )
    ):
        raise ScientificFailure(
            "Incomplete or nonfinite energy/full-force/stress output"
        )
    return {
        "energy_hartree": energy,
        "forces_hartree_per_bohr": forces,
        "stress": stress,
    }


def qe_observables(
    output: str, system: System, *, fermi_dirac: bool
) -> tuple[str, dict]:
    version = _required(r"Program PWSCF\s+v\.\s*(\S+)", output, "QE version")[1]
    if "convergence has been achieved" not in output or "JOB DONE." not in output:
        raise NotConverged("QE did not finish a converged SCF")
    energy = _number(
        _required(
            rf"^!\s+total energy\s*=\s*({NUMBER})\s+Ry", output, "QE total energy"
        )[1]
    )
    if fermi_dirac:
        entropy = _number(
            _required(
                rf"smearing contrib\.\s*\(-TS\)\s*=\s*({NUMBER})",
                output,
                "QE entropy term",
            )[1]
        )
        if entropy >= -1e-5:
            raise ScientificFailure("QE reference has no nontrivial smearing energy")
    if system.spin:
        moment = _number(
            _required(
                rf"total magnetization\s*=\s*({NUMBER})", output, "QE magnetization"
            )[1]
        )
        if abs(moment) <= 0.1:
            raise ScientificFailure("QE reference collapsed to a different spin state")
    force_header = _required(
        r"Forces acting on atoms \(cartesian axes, Ry/au\):",
        output,
        "QE Ry/Bohr forces",
    )
    force_table = re.split(
        r"\n[ \t]*(?:\n|The |Total force|Computing stress)",
        output[force_header.end() :].lstrip(),
        maxsplit=1,
    )[0]
    rows = re.findall(
        rf"^[ \t]*atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*{TRIPLE}\s*$",
        force_table,
        re.MULTILINE,
    )
    n = len(system.atomic_numbers)
    if [int(row[0]) for row in rows] != list(range(1, n + 1)):
        raise ScientificFailure("QE force rows do not match all ordered atoms")
    forces = [[_number(v) / 2 for v in row[1:]] for row in rows]
    if max(abs(v) for row in forces for v in row) <= 1e-3:
        raise ScientificFailure("QE force reference is trivial")
    stress_header = _required(
        r"total\s+stress\s+\(Ry/bohr\*\*3\)[^\n]*\n", output, "QE stress tensor"
    )
    tensor = [
        [_number(value) / 2 for value in line.split()[:3]]
        for line in output[stress_header.end() :].splitlines()[:3]
    ]
    if len(tensor) != 3 or any(len(row) != 3 for row in tensor):
        raise ScientificFailure("QE did not print all three stress rows")
    stress = [
        tensor[i][j] * STRESS_AU_IN_EV_ANG3
        for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    ]
    return version, _observables(energy / 2, forces, stress, n, 6)


def cp2k_observables(
    output: str, system: System, *, stress_kind: str
) -> tuple[str, dict]:
    version = _required(
        r"CP2K\| version string:\s+CP2K version\s+(\S+)", output, "CP2K version"
    )[1]
    if (
        "SCF run converged" not in output
        or "SCF run NOT converged" in output
        or "PROGRAM ENDED AT" not in output
    ):
        raise NotConverged("CP2K did not finish all reference SCFs")
    # CP2K prints energies for the strained cells before the final reference.
    final = output
    if stress_kind == "Numerical":
        end = _required(r"NUMERICAL STRESS END", output, "end of CP2K strain work")
        final = output[end.end() :]
    energy = _number(
        _required(
            rf"ENERGY\| Total FORCE_EVAL \( QS \) energy \[hartree\]\s+({NUMBER})",
            final,
            "CP2K unstrained total energy",
        )[1]
    )
    if system.spin:
        counts = re.findall(
            r"Number of electrons:[ \t]+(\d+)[ \t]*$", output, re.MULTILINE
        )
        if not counts or counts != ["2", "1"] * (len(counts) // 2):
            raise ScientificFailure(
                "CP2K UKS reference did not retain 2 alpha / 1 beta"
            )
    force_header = _required(
        r"FORCES\| Atomic forces \[hartree/bohr\]", final, "CP2K Ha/Bohr forces"
    )
    force_table = final[force_header.end() :].split("FORCES| Sum", 1)[0]
    rows = re.findall(
        rf"^[ \t]*FORCES\|[ \t]+(\d+)[ \t]+{TRIPLE}[ \t]+{NUMBER}[ \t]*$",
        force_table,
        re.MULTILINE,
    )
    n = len(system.atomic_numbers)
    if [int(row[0]) for row in rows] != list(range(1, n + 1)):
        raise ScientificFailure("CP2K force rows do not match all ordered atoms")
    forces = [[_number(v) for v in row[1:]] for row in rows]
    stress_header = _required(
        rf"STRESS\|\s+{stress_kind} stress tensor \[bar\]", final, "CP2K stress"
    )
    stress_table = final[stress_header.end() :].split("STRESS| 1/3 Trace", 1)[0]
    rows = re.findall(
        rf"^[ \t]*STRESS\|\s+([xyz])\s+{TRIPLE}\s*$", stress_table, re.MULTILINE
    )
    if [row[0] for row in rows] != list("xyz"):
        raise ScientificFailure(
            "CP2K did not print the ordered stress tensor"
        )
    stress = [
        _number(row[i + 1]) * 1e-4 / 160.2176634 / STRESS_AU_IN_EV_ANG3
        for i, row in enumerate(rows)
    ]
    return version, _observables(energy, forces, stress, n, 3)


def _binary(variable: str, default: str) -> Path:
    requested = os.environ.get(variable, default)
    resolved = shutil.which(requested)
    if resolved is None:
        raise BindingMissing(
            f"Set {variable} to the executable (could not find {requested})"
        )
    return Path(resolved).resolve()


def _run(
    binary: Path, *, cwd: Path, args: tuple[str, ...] = (), stdin: str | None = None
) -> str:
    run = subprocess.run(
        [str(binary), *args],
        check=False,
        cwd=cwd,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sys.stderr.write(run.stdout)
    if run.returncode:
        raise ScientificFailure(
            f"{binary.name} exited {run.returncode}; see engine log"
        )
    return run.stdout


def atv_observables(output: str, row: Row, system: System) -> dict:
    n = len(system.atomic_numbers)
    lanes = len(row.model["stress_components"])

    def parse(record: str) -> dict:
        values = [_number(x) for x in _required(
            rf"^{record} ([^\n]+)$", output, record
        )[1].split()]
        if len(values) != 1 + 3 * n + lanes:
            raise ScientificFailure("ATV result has the wrong number of observables")
        forces = [values[1 + 3 * i : 4 + 3 * i] for i in range(n)]
        return _observables(values[0], forces, values[1 + 3 * n :], n, lanes)

    observables = parse("ATV_PERIODIC_RESULT")
    if any(row.model.get("first_atom_image", ())):
        translated = parse("ATV_PERIODIC_TRANSLATED")
        for check in row.checks:
            key = check.observable
            if key in translated and not compare_values(
                translated[key], observables[key], check.atol, check.rtol
            )["pass"]:
                raise ScientificFailure(f"Periodic translation changed {key}")
        observables.update({f"translated_{key}": value for key, value in translated.items()})
    if row.model.get("first_atom_seed_shift_bohr"):
        initial = parse("ATV_PERIODIC_INITIAL")
        for check in row.checks:
            key = check.observable
            if compare_values(
                observables[key], initial[key], check.atol, check.rtol
            )["pass"]:
                raise ScientificFailure(f"Displaced geometry did not change {key}")
    return observables


def atv_adapter(row: Row, system: System, participant: Participant, threads: int):
    cuda = participant.backend == "cuda"
    binary = (
        _binary("ATV_BENCH_PERIODIC_CUDA", "atv_periodic_dft_cuda")
        if cuda
        else _binary("ATV_BENCH_PERIODIC", str(ROOT / "build/bin/atv_periodic_dft"))
    )
    identity = file_hash(binary)
    model = row.model
    pw = model["method"] == "pw"

    def run():
        n = len(system.atomic_numbers)
        header = [
            model[key]
            for key in (
                "method",
                "xc",
                "cutoff_ry",
                "tol",
                "max_iter",
            )
        ]
        if cuda:
            header[0] += ".cuda"
        controls = (
            ("expected_electrons", "degauss_ry", "extra_bands", "mix_beta")
            if pw
            else (
                "pseudopotential",
                "basis",
                "eps_default",
                "diis_depth",
                "strain_step",
            )
        )
        tokens = [
            *header,
            system.spin,
            n,
            *(x for vector in model["cell_vectors_angstrom"] for x in vector),
            *(
                value
                for z, mass, r in zip(
                    system.atomic_numbers,
                    model["atomic_masses_amu"],
                    system.positions_angstrom,
                    strict=True,
                )
                for value in (z, mass, *r)
            ),
            *(model[key] for key in controls),
        ]
        with TemporaryDirectory(prefix="atv-periodic-") as directory:
            scratch = Path(directory)
            if pw:
                potentials = model["nc_potentials"]
                tokens.extend((int(model["fermi_dirac"]), len(potentials)))
                for z, kind, source in potentials:
                    name = source
                    if kind == "upf":
                        name = Path(source).name
                        staged = scratch / name
                        shutil.copyfile(ROOT / source, staged)
                        if file_hash(staged) != model["oracle_assets"][source]:
                            raise ScientificFailure(f"NC asset changed: {name}")
                    tokens.extend((z, kind, name))
            else:
                tokens.extend((model["ngrids"], model["rel_cutoff"]))
                tokens.extend(model["k_mesh"])
                tokens.append(model["expected_electrons"])
                tokens.extend(model["first_atom_image"])
                tokens.append(model["first_atom_seed_shift_bohr"])
            text = " ".join(map(str, tokens)) + "\n"
            output = _run(binary, cwd=scratch, stdin=text)
        return {
            "observables": atv_observables(output, row, system),
            "metadata": {
                "binary_path": str(binary),
                "binary_sha256": identity,
                "requested_threads": threads,
                "engine": participant.engine,
                "native_input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            },
        }

    return run


def oracle_adapter(row: Row, system: System, participant: Participant, threads: int):
    qe = participant.engine == "qe"
    if qe:
        binary = _binary("ATV_BENCH_PW_X", "pw.x")
    else:
        default = shutil.which("cp2k.psmp") or "cp2k"
        binary = _binary("ATV_BENCH_CP2K", default)
    identity = file_hash(binary)
    model = row.model
    source = ROOT / model["oracle_input"]
    if file_hash(source) != model["oracle_input_sha256"]:
        raise ScientificFailure("Archived oracle input changed after declaration")
    deck = source.read_text()
    assets = {ROOT / name: sha for name, sha in model["oracle_assets"].items()}
    if not qe:
        data = Path(
            os.environ.get("CP2K_DATA_DIR", binary.parent.parent / "share/cp2k/data")
        )
        for key, name in (
            ("BASIS_SET_FILE_NAME", model["basis_set_file"]),
            ("POTENTIAL_FILE_NAME", "GTH_POTENTIALS"),
        ):
            path = data / name
            if not path.is_file():
                raise BindingMissing(
                    f"Set CP2K_DATA_DIR to the directory containing {name}"
                )
            deck, count = re.subn(
                rf"^([ \t]*{key})[ \t]+\S+[ \t]*$",
                rf"\1 {name}",
                deck,
                flags=re.MULTILINE,
            )
            if count != 1:
                raise ScientificFailure(f"Expected exactly one CP2K {key} directive")
            assets[path] = file_hash(path)

    def run():
        with TemporaryDirectory(prefix=f"atv-{participant.engine}-") as directory:
            scratch = Path(directory)
            for path, expected in assets.items():
                staged = scratch / path.name
                shutil.copyfile(path, staged)
                if file_hash(staged) != expected:
                    raise ScientificFailure(
                        f"Oracle asset changed during execution: {path.name}"
                    )
            (scratch / source.name).write_text(deck)
            output = _run(
                binary, cwd=scratch, args=("-in" if qe else "-i", source.name)
            )
        if qe:
            version, observables = qe_observables(
                output, system, fermi_dirac=model["fermi_dirac"]
            )
        else:
            version, observables = cp2k_observables(
                output, system, stress_kind=model["stress_kind"]
            )
            nx, ny, nz = model["k_mesh"]
            if nx * ny * nz > 1:
                count = _required(
                    r"BRILLOUIN\| List of Kpoints \[2 Pi/Bohr\][ \t]+(\d+)",
                    output, "CP2K k-point count"
                )
                if int(count[1]) != nx * ny * nz:
                    raise ScientificFailure("CP2K did not use the full k-point mesh")
            if model["ngrids"] > 1:
                active_levels = {
                    int(level)
                    for level, count in re.findall(
                        r"^\s*count for grid\s+(\d+):\s+(\d+)", output, re.MULTILINE
                    )
                    if int(count) > 0
                }
                if len(active_levels) < 2:
                    raise ScientificFailure("CP2K multigrid used fewer than two levels")
            if any(model["first_atom_image"]):
                # Both native coordinate representations describe this CP2K cell.
                observables |= {
                    f"translated_{key}": value for key, value in observables.items()
                }
        if version != model["oracle_version"]:
            raise ScientificFailure(
                f"Expected {participant.engine} {model['oracle_version']}, got {version}"
            )
        return {
            "observables": observables,
            "metadata": {
                "binary_path": str(binary),
                "binary_sha256": identity,
                "engine": participant.engine,
                "engine_version": version,
                "requested_threads": threads,
                "archived_input_sha256": model["oracle_input_sha256"],
                "executed_input_sha256": hashlib.sha256(deck.encode()).hexdigest(),
                "asset_sha256": {path.name: sha for path, sha in assets.items()},
            },
        }

    return run
