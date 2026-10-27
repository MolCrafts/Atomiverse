"""Molecular KS benchmarks through public APIs and live independent PySCF.

Adapter factories import and fingerprint runtimes only. Each measured call
constructs its own frame, molecule, integration grid and solver, then releases
them. Gaussian CUDA KS is explicitly unavailable until a device method exists.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from benchmarks.framework.provenance import file_hash
from benchmarks.framework.specification import Check, Participant, Row, System
from benchmarks.scf.hf import (
    BOHR_IN_ANGSTROM,
    HARTREE_IN_EV,
    BindingMissing,
    CapabilityUnavailable,
    NotConverged,
    ScientificFailure,
    _metadata,
    _shared_libraries,
    _thread_mismatch,
)
from benchmarks.scf.hf import (
    systems as hf_systems,
)

# Explicit ingredients avoid PySCF alias changes and distinguish VWN5 LSDA
# from the VWN-RPA correlation used by ATV's B3LYP implementation.
PYSCF_XC = {
    "LSDA": "LDA_X,LDA_C_VWN",
    "PBE": "GGA_X_PBE,GGA_C_PBE",
    "PBEsol": "GGA_X_PBE_SOL,GGA_C_PBE_SOL",
    "RevPBE": "GGA_X_PBE_R,GGA_C_PBE",
    "RPBE": "GGA_X_RPBE,GGA_C_PBE",
    "PBE0": "0.25*HF + 0.75*GGA_X_PBE,GGA_C_PBE",
    "BLYP": "GGA_X_B88,GGA_C_LYP",
    "B3LYP": "0.20*HF + 0.08*LDA_X + 0.72*GGA_X_B88,0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP",
    # Standard libxc CAM includes attenuated B88 exchange and VWN5.
    # A custom B3LYP + long-range-HF composition is a different model.
    "CAMB3LYP": "HYB_GGA_XC_CAM_B3LYP",
}


def systems(*names: str) -> list[System]:
    """Read DFT-specific geometries and share the existing HF fixtures."""
    canonical = json.loads(
        (Path(__file__).resolve().parents[1] / "systems/dft.json").read_text()
    )
    return [
        System.model_validate(canonical[name])
        if name in canonical
        else hf_systems(name)[0]
        for name in names
    ]


def row(
    method: str, xc: str, task: str, names: list[str], *, basis: str = "STO-3G"
) -> Row:
    if (
        method not in {"RKS", "UKS"}
        or xc not in PYSCF_XC
        or task not in {"energy", "forces"}
    ):
        raise ValueError("unsupported molecular KS benchmark declaration")
    model = {
        "method": method,
        "xc": xc,
        "pyscf_xc": PYSCF_XC[xc],
        "basis": basis,
        "cartesian": basis in {"STO-3G", "6-31G", "6-31G*"},
        "comparison_class": "matched_accuracy",
        "grid": "MatchPySCF",
        "pyscf_grid_level": 3,
        "grid_response": True,
        "small_rho_cutoff": 0.0,
        "init_guess": "minao",
        "conv_tol": 1e-12,
        "conv_tol_grad": 1e-8,
        "max_cycle": 200,
        "damping": 0.0,
        "diis_space": 8,
        "diis_start_cycle": 1,
    }
    checks = [
        Check(
            observable="energy_hartree",
            unit="Ha",
            atol=1e-8,
            justification="Live PySCF KS with explicit XC ingredients and matched level-3 quadrature.",
        )
    ]
    if task == "forces":
        checks.append(
            Check(
                observable="forces_hartree_per_bohr",
                unit="Ha/Bohr",
                atol=1e-6,
                justification="Every analytic force component against live PySCF with full moving-grid response.",
            )
        )
    selected = systems(*names)
    if basis != "STO-3G":
        for system in selected:
            system.tiers = [tier for tier in system.tiers if tier != "smoke"]
    return Row(
        id=f"{method.lower()}.{xc.lower()}.{basis.lower().replace('-', '').replace('*', 's')}.{task}",
        label=f"{method} {xc}/{basis} {task}",
        capability=f"dft.{method.lower()}.{'gradient' if task == 'forces' else 'energy'}",
        task=task,
        model=model,
        systems=selected,
        reference="pyscf.cpu",
        participants=[
            Participant(
                id="atv.cpu",
                engine="atomiverse",
                backend="cpu",
                role="atv",
                required=True,
                entrypoint="benchmarks.scf.dft:atv_adapter",
                options={"max_memory_mb": 4000.0},
            ),
            Participant(
                id="atv.cuda",
                engine="atomiverse",
                backend="cuda",
                role="atv",
                available=False,
                unavailable_reason=f"Gaussian CUDA {method} is not implemented; CUDA PW/GPW are separate representations",
                entrypoint="benchmarks.scf.dft:atv_adapter",
            ),
            Participant(
                id="pyscf.cpu",
                engine="pyscf",
                backend="cpu",
                role="oracle",
                required=True,
                entrypoint="benchmarks.scf.dft:pyscf_adapter",
            ),
        ],
        checks=checks,
    )


def atv_adapter(row: Row, system: System, participant: Participant, threads: int):
    if not participant.available or participant.backend != "cpu":
        raise CapabilityUnavailable(
            participant.unavailable_reason or "Gaussian CUDA KS is not implemented"
        )
    np = importlib.import_module("numpy")
    molpy = importlib.import_module("molpy")
    scf = importlib.import_module("atomiverse.scf")
    threadpoolctl = importlib.import_module("threadpoolctl")
    cls = getattr(scf, row.model["method"], None)
    if cls is None or not hasattr(cls, "grid"):
        raise BindingMissing(
            "Public KS method with an explicit integration-grid control is required"
        )
    limits = threadpoolctl.threadpool_limits(limits=threads)
    backend = scf.backend_info()
    meta = _metadata(
        "atomiverse",
        importlib.import_module(cls.__module__),
        threadpoolctl,
        threads,
        backend="cpu",
        backend_info=backend,
    )
    meta.update(
        requested_model=row.model,
        comparison_class="matched_accuracy",
        resource_mismatch=_thread_mismatch(meta["threadpools"], threads)
        or backend.get("openmp_threads") != threads,
    )

    def run():
        _ = limits
        xyz = np.asarray(system.positions_angstrom, dtype=np.float64)
        frame = molpy.Frame()
        atoms = molpy.Block()
        for axis, name in enumerate(["x", "y", "z"]):
            atoms.insert(name, xyz[:, axis].copy())
        atoms.insert(
            "atomic_number", np.asarray(system.atomic_numbers, dtype=np.uint32)
        )
        frame["atoms"] = atoms
        frame.box = molpy.Box.cube(max(40.0, float(np.ptp(xyz, axis=0).max()) + 24.0))
        kwargs = {
            "basis": row.model["basis"],
            "xc": row.model["xc"],
            "charge": system.charge,
            "tol": row.model["conv_tol"],
            "max_iter": row.model["max_cycle"],
        }
        if row.model["method"] == "UKS":
            electrons = sum(system.atomic_numbers) - system.charge
            kwargs.update(
                n_alpha=(electrons + system.spin) // 2,
                n_beta=(electrons - system.spin) // 2,
            )
        solver = cls(**kwargs)
        try:
            solver.grid = row.model["grid"]
            solver.max_memory_mb = participant.options["max_memory_mb"]
            for field in [
                "init_guess",
                "damping",
                "conv_tol_grad",
                "diis_space",
                "diis_start_cycle",
            ]:
                setattr(solver, field, row.model[field])
            if solver.grid != row.model["grid"] or solver.xc != row.model["xc"]:
                raise ScientificFailure(
                    "Native KS grid or XC differs from the requested model"
                )
            forces = None
            if row.task == "forces":
                forces = np.asarray(solver.forces(frame), dtype=np.float64).copy()
                columns = np.column_stack(
                    [np.asarray(frame["atoms"][key]) for key in ["fx", "fy", "fz"]]
                )
                if columns.shape != forces.shape or not np.allclose(
                    forces, columns, rtol=0, atol=1e-8
                ):
                    raise ScientificFailure(
                        "Frame forces disagree with the public KS force result"
                    )
            else:
                solver.run(frame)
            energy = float(solver.energy())
            if not solver.converged() or not np.isfinite(energy):
                raise NotConverged("Native KS did not return a finite converged energy")
            frame_energy = float(frame.meta["energy"]) / HARTREE_IN_EV
            if not np.isfinite(frame_energy) or abs(frame_energy - energy) > 1e-10:
                raise ScientificFailure("Frame energy and KS Hartree total disagree")
            observables = {"energy_hartree": energy}
            if forces is not None:
                forces *= BOHR_IN_ANGSTROM / HARTREE_IN_EV
                if (
                    forces.shape != (len(system.atomic_numbers), 3)
                    or not np.isfinite(forces).all()
                ):
                    raise ScientificFailure("Invalid native KS force array")
                if np.linalg.norm(forces) < 1e-8:
                    raise ScientificFailure(
                        "Nonstationary KS workload returned zero forces"
                    )
                if system.id == "water_asymmetric" and np.any(np.abs(forces) < 1e-8):
                    raise ScientificFailure(
                        "Asymmetric water requires nine nonzero force components"
                    )
                observables["forces_hartree_per_bohr"] = forces.tolist()
            resolved = {
                field: getattr(solver, field)
                for field in [
                    "basis",
                    "charge",
                    "xc",
                    "grid",
                    "tol",
                    "max_iter",
                    "max_memory_mb",
                    "init_guess",
                    "damping",
                    "conv_tol_grad",
                    "diis_space",
                    "diis_start_cycle",
                ]
            }
            resolved.update(
                converged=True,
                iterations=int(solver.iterations()),
                solver_type=type(solver).__name__,
                final_residuals="not_exposed",
            )
            return {
                "observables": observables,
                "metadata": {**meta, "resolved_settings": resolved},
            }
        finally:
            del solver, frame, atoms

    return run


def pyscf_adapter(row: Row, system: System, participant: Participant, threads: int):
    if participant.backend != "cpu":
        raise CapabilityUnavailable(
            "This molecular KS reference adapter requires PySCF CPU"
        )
    np = importlib.import_module("numpy")
    pyscf = importlib.import_module("pyscf")
    gto = importlib.import_module("pyscf.gto")
    dft = importlib.import_module("pyscf.dft")
    gen_grid = importlib.import_module("pyscf.dft.gen_grid")
    radi = importlib.import_module("pyscf.dft.radi")
    threadpoolctl = importlib.import_module("threadpoolctl")
    limits = threadpoolctl.threadpool_limits(limits=threads)
    pyscf.lib.num_threads(threads)
    meta = _metadata(
        "pyscf",
        dft,
        threadpoolctl,
        threads,
        backend="cpu",
        pyscf_version=pyscf.__version__,
        pyscf_num_threads=pyscf.lib.num_threads(),
    )
    for file in _shared_libraries(
        Path(pyscf.__file__).resolve().parent / "lib", recursive=True
    ):
        meta["binaries_sha256"][str(file)] = file_hash(file)
    primary = next(
        (path for path in meta["binaries_sha256"] if "libcgto" in path), None
    )
    meta.update(
        binary_path=primary,
        binary_sha256=meta["binaries_sha256"].get(primary),
        requested_model=row.model,
        comparison_class="matched_accuracy",
        resource_mismatch=_thread_mismatch(meta["threadpools"], threads)
        or pyscf.lib.num_threads() != threads,
    )

    def run():
        _ = limits
        mol = gto.M(
            atom=list(zip(system.atomic_numbers, system.positions_angstrom)),
            basis=row.model["basis"],
            charge=system.charge,
            spin=system.spin,
            unit="Angstrom",
            cart=row.model["cartesian"],
            verbose=0,
        )
        solver = getattr(dft, row.model["method"])(mol)
        try:
            solver.xc = row.model["pyscf_xc"]
            solver.grids.level = row.model["pyscf_grid_level"]
            solver.grids.prune = gen_grid.nwchem_prune
            solver.grids.radi_method = radi.treutler_ahlrichs
            solver.grids.radii_adjust = radi.treutler_atomic_radii_adjust
            solver.grids.becke_scheme = gen_grid.original_becke
            solver.small_rho_cutoff = row.model["small_rho_cutoff"]
            solver.conv_tol = row.model["conv_tol"]
            solver.conv_tol_grad = row.model["conv_tol_grad"]
            solver.max_cycle = row.model["max_cycle"]
            solver.diis_space = row.model["diis_space"]
            solver.diis_start_cycle = row.model["diis_start_cycle"]
            solver.damp = row.model["damping"]
            solver.init_guess = row.model["init_guess"]
            energy = float(solver.kernel())
            if not solver.converged or not np.isfinite(energy):
                raise NotConverged("Independent PySCF KS did not converge")
            observables = {"energy_hartree": energy}
            if row.task == "forces":
                gradient = solver.nuc_grad_method()
                gradient.grid_response = row.model["grid_response"]
                forces = -np.asarray(gradient.kernel(), dtype=np.float64)
                del gradient
                if (
                    forces.shape != (len(system.atomic_numbers), 3)
                    or not np.isfinite(forces).all()
                ):
                    raise ScientificFailure("Invalid independent KS analytic forces")
                if np.linalg.norm(forces) < 1e-8:
                    raise ScientificFailure(
                        "Independent nonstationary KS workload returned zero forces"
                    )
                if system.id == "water_asymmetric" and np.any(np.abs(forces) < 1e-8):
                    raise ScientificFailure(
                        "Independent asymmetric water has a zero force component"
                    )
                observables["forces_hartree_per_bohr"] = forces.tolist()
            resolved = {
                "xc": solver.xc,
                "grid_level": solver.grids.level,
                "grid_points": len(solver.grids.weights),
                "radial": "treutler_ahlrichs",
                "pruning": "nwchem_prune",
                "radii_adjust": "treutler_atomic_radii_adjust",
                "partition": "original_becke",
                "small_rho_cutoff": float(solver.small_rho_cutoff),
                "grid_response": row.model["grid_response"]
                if row.task == "forces"
                else "not_requested",
                "cartesian": bool(mol.cart),
                "nao": int(mol.nao_nr()),
                "converged": True,
                "iterations": getattr(solver, "cycles", None),
                "init_guess": solver.init_guess,
                "conv_tol": float(solver.conv_tol),
                "conv_tol_grad": float(solver.conv_tol_grad),
                "max_cycle": int(solver.max_cycle),
                "damping": float(solver.damp),
                "diis_space": int(solver.diis_space),
                "diis_start_cycle": int(solver.diis_start_cycle),
                "solver_type": type(solver).__module__ + "." + type(solver).__name__,
            }
            return {
                "observables": observables,
                "metadata": {**meta, "resolved_settings": resolved},
            }
        finally:
            del solver, mol

    return run
