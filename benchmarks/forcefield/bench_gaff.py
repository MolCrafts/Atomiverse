"""Fixed-geometry GAFF/GAFF2 with AM1-BCC/ABCG2, versus AmberTools.

The task includes fresh graph/input construction, an AM1 SCF, equivalencing and
BCC, GAFF typing/parameter assignment, and result materialization. AmberTools
also assembles its parameters with tleap. No geometry generation, optimization,
or cached scientific intermediate is inside this task. See README.md for the
acceptance limits and the limits of the parameter-level comparison.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from benchmarks.framework.provenance import digest, file_hash
from benchmarks.framework.specification import Check, Method, Participant, Row, System

HERE = Path(__file__).resolve().parent
CANONICAL = json.loads((HERE / "systems.json").read_text(encoding="utf-8"))
AMBER_DEFAULT = "/software/sse2/el9_epyc9005/easybuild/prefix/software/AmberTools/26.1-gmpich-2026.03"


class ScientificFailure(RuntimeError):
    benchmark_status = "scientific_failure"


def method_row(atomtype: str, correction: str) -> Row:
    systems = []
    for name, case in CANONICAL["systems"].items():
        smoke = name in {"ethanol", "acetate", "trimethylphosphate"}
        systems.append(System(
            id=name, label=f"{name} ({len(case['atoms'])} atoms)",
            atomic_numbers=[a["atomic_number"] for a in case["atoms"]],
            positions_angstrom=[[a[c] for c in ("x", "y", "z")] for a in case["atoms"]],
            charge=case["charge"],
            tiers=(["smoke"] if smoke else []) + ["acceptance", "performance"],
        ))
    return Row(
        id=f"forcefield.{atomtype}.{correction}.parameterize",
        label=f"{atomtype.upper()} + {correction.upper()}, fixed geometry",
        capability="forcefield.gaff.parameterization", task="parameterization",
        model={
            "atomtype": atomtype, "correction": correction,
            "comparison_class": "matched_accuracy", "geometry_optimization": False,
            "am1_density_tolerance": 1e-10, "am1_energy_tolerance_ev": 1e-10,
            "sqm_scfconv": 1e-10, "sqm_maxcyc": 0, "equivalencing": 1,
            "canonical_sha256": file_hash(HERE / "systems.json"),
            "chemistry": CANONICAL["systems"],
            "scope": "full fixed-geometry molecular parameterization; no dynamics",
        },
        systems=systems,
        participants=[
            Participant(id="atv.cpu", engine="atomiverse+molpy", backend="cpu", role="atv",
                        required=True, entrypoint="benchmarks.forcefield.bench_gaff:atv_adapter"),
            Participant(id="ambertools.cpu", engine="AmberTools", backend="cpu", role="oracle",
                        required=True, entrypoint="benchmarks.forcefield.bench_gaff:amber_adapter"),
        ], reference="ambertools.cpu",
        checks=[
            Check(observable="atomtype_codes", unit="integer label", atol=0,
                  justification="Exact byte encoding of each GAFF atom type in canonical atom order."),
            Check(observable="charges_e", unit="e", atol=1e-3,
                  justification="Fixed-geometry AM1-BCC; sqm exposes base Mulliken charges at 1e-3 e precision "
                  "(up to 5e-4 e rounding), plus 5e-4 e independent AM1 implementation allowance. "
                  "Declared before this benchmark; not an optimized-geometry equivalence claim."),
            Check(observable="bond_parameters", unit="indices; kcal/mol/A^2; A", atol=1e-6, rtol=1e-7,
                  justification="Same explicit bond instances and Amber unhalved K, after prmtop serialization."),
            Check(observable="angle_parameters", unit="indices; kcal/mol/rad^2; degree", atol=1e-4, rtol=1e-7,
                  justification="Same explicit angle instances; prmtop equilibrium radians converted to degrees. "
                  "Amber's frozen radian/degree constant causes about 5e-5 degree at ordinary bond angles."),
            Check(observable="dihedral_parameters", unit="indices; kcal/mol; integer; degree", atol=1e-5, rtol=1e-7,
                  justification="Canonical proper quartets and nonzero Fourier components; zero terms omitted."),
            Check(observable="improper_parameters", unit="indices; kcal/mol; integer; degree", atol=1e-5, rtol=1e-7,
                  justification="Improper slot ordering retained up to full quartet reversal at phase 0/pi; "
                  "tleap reverses a quartet to avoid an unencodable negative-zero fourth pointer. "
                  "No arbitrary outer-atom permutation is accepted."),
            Check(observable="atom_lj_parameters", unit="kcal/mol; A", atol=1e-6, rtol=1e-7,
                  justification="Per-atom self Lennard-Jones epsilon/sigma; prmtop A/B conversion precision."),
            Check(observable="atom_masses", unit="amu", atol=1e-6, rtol=1e-8,
                  justification="Per-atom masses from the assigned force-field types and serialized prmtop."),
            Check(observable="special_bonds", unit="dimensionless", atol=1e-12,
                  justification="Amber LJ (0,0,1/2) and Coulomb (0,0,1/1.2) graph-distance weights."),
        ],
    )


SPEC = Method(id="gaff", rows=[method_row(a, c) for a in ("gaff", "gaff2") for c in ("bcc", "abcg2")])


def _graph(mp, case: dict):
    mol = mp.Atomistic()
    atoms = [mol.add_atom(**props) for props in case["atoms"]]
    for bond in case["bonds"]:
        mol.add_bond(atoms[bond["atomi"]], atoms[bond["atomj"]],
                     bond_type=bond["bond_type"], bond_number=bond["bond_number"])
    return mol


def _parameters(ff, frame) -> tuple[dict, dict]:
    """Materialize canonical parameter records; no force-field algorithm lives here."""
    numeric = {}
    labels = [str(t) for t in frame["atoms"]["type"]]
    numeric["atomtype_codes"] = [int.from_bytes(t.encode("ascii"), "big") for t in labels]
    numeric["charges_e"] = [float(q) for q in frame["atoms"]["charge"]]
    atom_params = {t.name: t.params for t in ff.get_types("atom")}
    numeric["atom_masses"] = [float(atom_params[t]["mass"]) for t in labels]
    pairs = {t.name: t.params for t in ff.get_types("pair") if "epsilon" in t.params}
    numeric["atom_lj_parameters"] = [[float(pairs[t]["epsilon"]), float(pairs[t]["sigma"])] for t in labels]
    numeric["special_bonds"] = [list(map(float, weights)) for weights in ff.special_bonds]
    counts = {}
    for category, block, width, columns in (
        ("bond", "bonds", 2, ("k", "r0")),
        ("angle", "angles", 3, ("k", "theta0")),
        ("dihedral", "dihedrals", 4, ()),
        ("improper", "impropers", 4, ("k", "periodicity", "phase")),
    ):
        rows = []
        type_params = {t.name: t.params for t in ff.get_types(category)}
        counts[block] = 0
        if block in frame:
            data = frame[block]
            counts[block] = data.n_rows
            for index, label in enumerate(data["type"]):
                ends = tuple(int(data[f"atom{letter}"][index]) for letter in "ijkl"[:width])
                if category != "improper":
                    ends = min(ends, ends[::-1])
                params = type_params[str(label)]
                if category == "dihedral":
                    term = 1
                    while f"k{term}" in params:
                        k = float(params[f"k{term}"])
                        if k != 0:
                            phase = float(params[f"phase{term}"]) % 360.0
                            rows.append([*ends, k, float(params[f"periodicity{term}"]), phase])
                        term += 1
                elif category == "improper":
                    phase = float(params["phase"]) % 360.0
                    # Only the full quartet reversal is equivalent here, and
                    # this benchmark limits that normalization to 0/pi phases.
                    # In particular we never sort the three outer atoms.
                    if phase in (0.0, 180.0):
                        ends = min(ends, ends[::-1])
                    rows.append([*ends, float(params["k"]), float(params["periodicity"]), phase])
                else:
                    rows.append([*ends, *(float(params[col]) for col in columns)])
        # A nonempty sentinel distinguishes an absent relation category and lets
        # the generic numeric checker compare empty physical term sets strictly.
        numeric[f"{category}_parameters"] = sorted(rows) if rows else [[-1.0]]
    estimates = []
    for category in ("bond", "angle", "dihedral", "improper"):
        for row in ff.get_types(category):
            p = row.params
            if p.get("estimated", False):
                estimates.append({"category": category, "type": row.name, "params": p})
    return numeric, {"atom_types": labels, "topology_counts": counts, "estimated_types": estimates}


def _job_directory(system: System, participant: Participant) -> Path:
    root = os.environ.get("ATV_GAFF_ARTIFACT_ROOT")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{system.id}-{participant.id}-", dir=root))


def atv_adapter(row: Row, system: System, participant: Participant, threads: int):
    import atomiverse as atv
    import molpy as mp
    import numpy as np
    import threadpoolctl

    native = importlib.import_module(atv.AM1BCCCharges.__module__)
    core = importlib.import_module("molpy._core")
    limits = threadpoolctl.threadpool_limits(limits=threads)
    paths = {str(Path(module.__file__).resolve()): file_hash(module.__file__) for module in (native, core)}
    molpy_root = Path(__file__).resolve().parents[3] / "molpy"
    meta = {
        "engine": "atomiverse+molpy", "binary_path": str(Path(native.__file__).resolve()),
        "binary_sha256": file_hash(native.__file__), "binaries_sha256": paths,
        "molpy_version": mp.__version__, "requested_threads": threads,
        "threadpools": threadpoolctl.threadpool_info(),
        "canonical_sha256": row.model["canonical_sha256"],
        "actual_engine_config": {k: v for k, v in row.model.items() if k != "chemistry"},
        "scope_limits": "Parameter-level oracle comparison; no independent force/energy evaluator or dynamics.",
    }
    if molpy_root.is_dir():
        receipt_path = os.environ.get("GAFF_MOLPY_SOURCE_RECEIPT")
        if receipt_path:
            receipt_file = Path(receipt_path)
            receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
            sibling = receipt["source"]
            if digest(sibling["source_files"]) != sibling["tree_sha256"]:
                raise ScientificFailure("molpy source receipt has an invalid source-tree digest")
            if receipt["binaries_sha256"] != paths:
                raise ScientificFailure("loaded native binaries do not match the source receipt")
            meta["molpy_source_receipt_sha256"] = file_hash(receipt_file)
        else:
            from benchmarks.framework.provenance import source_info

            sibling = source_info(molpy_root)
            sibling["repository"] = "MolCrafts/molpy"
        meta["molpy_source"] = {k: v for k, v in sibling.items() if k != "source_files"}
    case = row.model["chemistry"][system.id]

    def run():
        # Keeping limits alive pins native BLAS/OpenMP for the whole worker.
        _ = limits
        mol = _graph(mp, case)
        model = atv.AM1BCCCharges(
            method=atv.AM1(charge=system.charge, density_tol=row.model["am1_density_tolerance"],
                           energy_tol=row.model["am1_energy_tolerance_ev"]),
            correction=row.model["correction"],
        )
        start = time.perf_counter()
        charges = np.asarray(model.assign(mol))
        charge_seconds = time.perf_counter() - start
        am1_result = model.last_am1_result()
        if not am1_result.converged:
            raise ScientificFailure("native AM1 SCF did not converge")
        if charges.dtype != np.float64 or charges.shape != (len(case["atoms"]),):
            raise ScientificFailure("charge model did not return an atom-ordered float64 array")
        if not np.isfinite(charges).all() or abs(float(charges.sum()) - system.charge) > 1e-6:
            raise ScientificFailure("nonfinite or nonconserved native AM1-BCC charge")
        for atom, charge in zip(mol.atoms, charges):
            atom["charge"] = float(charge)
        start = time.perf_counter()
        typifier = mp.typifier.GaffTypifier(atomtype=row.model["atomtype"])
        typed = typifier.typify(mol)
        ff = typifier.get_forcefield()
        typing_seconds = time.perf_counter() - start
        observables, detail = _parameters(ff, typed.to_frame())
        detail.update(meta)
        detail["phases_seconds"] = {"am1_bcc": charge_seconds, "gaff_typing": typing_seconds}
        detail["native_net_charge_error_e"] = float(charges.sum()) - system.charge
        detail["am1_iterations"] = am1_result.iterations
        detail["am1_converged"] = am1_result.converged
        detail["resource_mismatch"] = any(p.get("num_threads") != threads
                                             for p in threadpoolctl.threadpool_info()
                                             if p.get("num_threads") is not None)
        return {"observables": observables, "metadata": detail}

    return run


def _invoke(command: list[str], directory: Path, name: str, env: dict) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, cwd=directory, env=env, text=True, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    elapsed = time.perf_counter() - start
    (directory / f"{name}.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise ScientificFailure(f"{name} returned {result.returncode}; see {directory / (name + '.log')}")
    return elapsed


def amber_adapter(row: Row, system: System, participant: Participant, threads: int):
    import molpy as mp

    prefix = Path(os.environ.get("GAFF_AMBERHOME", AMBER_DEFAULT)).resolve()
    executables = {name: prefix / "bin" / name for name in ("antechamber", "parmchk2", "tleap")}
    real = {name: prefix / "bin/wrapped_progs" / name for name in ("antechamber", "parmchk2")}
    real.update({"tleap": prefix / "bin/teLeap", "sqm": prefix / "bin/sqm"})
    identities = {str(path): file_hash(path) for path in real.values()}
    # Parameter tables are scientific inputs, not inferred from a version label.
    table_paths = [prefix / "dat/leap/parm" / f"{row.model['atomtype']}.dat",
                   prefix / "dat/antechamber" / ("BCCPARM.DAT" if row.model["correction"] == "bcc"
                                                else "BCCPARM_ABCG2.DAT"),
                   prefix / "dat/antechamber/PARMCHK.DAT"]
    tables = {str(path): file_hash(path) for path in table_paths}
    env = dict(os.environ)
    env["AMBERHOME"] = str(prefix)
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(threads)
    case = row.model["chemistry"][system.id]
    meta = {
        "engine": "AmberTools", "install_label": prefix.name, "install_prefix": str(prefix),
        "reference_kind": "external_native", "binary_path": str(real["antechamber"]),
        "binary_sha256": digest(identities), "binaries_sha256": identities, "tables_sha256": tables,
        "canonical_sha256": row.model["canonical_sha256"], "requested_threads": threads,
        "actual_engine_config": {k: v for k, v in row.model.items() if k != "chemistry"},
        "resource_policy": "Fresh subprocesses inherit identical OMP/BLAS thread limits; raw sqm output retained.",
        "scope_limits": "Parameter-level oracle comparison; no independent force/energy evaluator or dynamics.",
    }

    def run():
        directory = _job_directory(system, participant)
        source = directory / "input.mol2"
        source.write_text(case["input_mol2"], encoding="utf-8")
        phases = {}
        phases["antechamber"] = _invoke([
            str(executables["antechamber"]), "-i", str(source), "-fi", "mol2", "-o", "typed.mol2",
            "-fo", "mol2", "-at", row.model["atomtype"], "-c", row.model["correction"],
            "-nc", str(system.charge), "-eq", "1", "-pf", "y", "-ek",
            "qm_theory='AM1', maxcyc=0, scfconv=1.d-10",
        ], directory, "antechamber", env)
        sqm = (directory / "sqm.out").read_text(encoding="utf-8")
        if "Calculation Completed" not in sqm:
            raise ScientificFailure(f"sqm did not report completion; see {directory / 'sqm.out'}")
        phases["parmchk2"] = _invoke([
            str(executables["parmchk2"]), "-i", "typed.mol2", "-f", "mol2", "-o", "ligand.frcmod",
            "-s", row.model["atomtype"],
        ], directory, "parmchk2", env)
        text = (directory / "ligand.frcmod").read_text(encoding="utf-8")
        if "need revision" in text.lower() or "needs revision" in text.lower():
            raise ScientificFailure(f"parmchk2 left unresolved parameters; see {directory / 'ligand.frcmod'}")
        leap = directory / "build.leap"
        leap.write_text(
            f"source leaprc.{row.model['atomtype']}\n"
            "loadAmberParams ligand.frcmod\nLIG = loadMol2 typed.mol2\n"
            "saveAmberParm LIG ligand.prmtop ligand.inpcrd\nquit\n", encoding="utf-8",
        )
        phases["tleap"] = _invoke([str(executables["tleap"]), "-f", "build.leap"], directory, "tleap", env)
        ff, frame = mp.io.amber.AmberPrmtopForcefieldReader().read_system(directory / "ligand.prmtop")
        observables, detail = _parameters(ff, frame)
        # The MOL2 names guard atom order independently of prmtop type grouping.
        typed = mp.io.read_mol2(directory / "typed.mol2")["atoms"]
        expected = [f"{mp.Element.by_number(a['atomic_number']).symbol}{i}" for i, a in enumerate(case["atoms"], 1)]
        if list(typed["name"]) != expected:
            raise ScientificFailure("AmberTools reordered or renamed canonical atoms")
        if list(frame["atoms"]["name"]) != expected:
            raise ScientificFailure("tleap changed canonical atom order; parameter indices are not comparable")
        detail.update(meta)
        detail["raw_directory"] = str(directory)
        detail["phases_seconds"] = phases
        detail["oracle_net_charge_error_e"] = sum(observables["charges_e"]) - system.charge
        detail["sqm_version_line"] = next((line.strip() for line in sqm.splitlines()
                                            if "SQM VERSION" in line), None)
        detail["frcmod_text"] = text
        return {"observables": observables, "metadata": detail}

    return run


if __name__ == "__main__":
    from benchmarks.__main__ import method_main

    raise SystemExit(method_main(SPEC))
