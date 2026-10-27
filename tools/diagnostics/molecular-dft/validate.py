#!/usr/bin/env python3
"""Compare public Gaussian KS energies and complete analytic forces to live PySCF.

Run in a compute allocation with a freshly built HF-enabled Python extension.
This is a scientific diagnostic, outside the CTest unit tier. Imports and both
engines must succeed; unconverged, missing and nonfinite results never pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--xc", nargs="+", default=["PBE", "PBE0", "BLYP", "B3LYP"])
    parser.add_argument(
        "--system",
        nargs="+",
        choices=["water", "h2plus", "o2"],
        default=["water", "h2plus", "o2"],
    )
    parser.add_argument("--basis", nargs="+", default=["STO-3G", "def2-SVP"])
    parser.add_argument(
        "--native-fd",
        action="store_true",
        help="Add own-energy finite differences as a consistency check; never replaces the external oracle",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = str(args.threads)
    import atomiverse.scf as av
    import molpy
    import numpy as np
    import pyscf
    from atomiverse._binding import _cpp
    from pyscf import dft, gto, lib
    from threadpoolctl import threadpool_info, threadpool_limits

    limits = threadpool_limits(limits=args.threads)
    lib.num_threads(args.threads)
    # The asymmetry exercises all nine water force components, including x.
    systems = {
        "water": (
            [8, 1, 1],
            [(0.0, 0.0, 0.1173), (0.13, 0.7572, -0.4692), (-0.07, -0.7572, -0.4692)],
            0,
            0,
        ),
        "h2plus": ([1, 1], [(-0.1, 0.07, -0.2), (0.66, 0.22, 0.15)], 1, 1),
        "o2": ([8, 8], [(-0.1, 0.07, -0.2), (0.02, -0.03, 1.0015)], 0, 2),
    }
    source_root = Path(__file__).resolve().parents[3]
    sources = [
        Path(__file__).resolve(),
        source_root / "src/cpu/scf/internal/gradient.cpp",
        source_root / "src/cpu/scf/internal/integral_derivatives.cpp",
        source_root / "src/cpu/scf/rys_eri.cpp",
        source_root / "src/cpu/xc/xc_functional.cpp",
        source_root / "src/cpu/xc/xc_gga.cpp",
        source_root / "include/atomiverse/cpu/xc/xc_internal.h",
        source_root / "src/cpu/scf/internal/hamiltonian_builder.cpp",
    ]
    evidence = {
        "command": sys.argv,
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "hardware": platform.uname()._asdict(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "scheduler": {
            key: os.environ.get(key)
            for key in ["SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST"]
        },
        "native_binary": str(Path(_cpp.__file__).resolve()),
        "native_binary_sha256": digest(_cpp.__file__),
        "source_sha256": {str(p.relative_to(source_root)): digest(p) for p in sources},
        "build_verified": False,
        "pyscf": pyscf.__version__,
        "oracle_binaries": {
            str(p.resolve()): digest(p)
            for p in sorted(Path(pyscf.__file__).resolve().parent.rglob("*.so"))
        },
        "numpy": np.__version__,
        "requested_threads": args.threads,
        "threadpools": threadpool_info(),
        "controls": {
            "native_grid": "MatchPySCF",
            "pyscf_grid_level": 3,
            "analytic_grid_response": True,
            "tolerance_ha": 1e-12,
            "residual_tolerance": 1e-8,
            "max_iter": 200,
            "initial_guess": "minao",
            "damping": 0.0,
            "diis_space": 8,
            "diis_start_cycle": 1,
        },
        "gates": {
            "energy_ha": 1e-8,
            "full_force_component_ha_bohr": 1e-6,
            "both_engines_converged": True,
            "finite_observables": True,
        },
        "records": [],
    }
    output = args.out / "results.json"
    failed = False
    for system in args.system:
        zs, xyz, charge, spin = systems[system]
        method = "RKS" if spin == 0 else "UKS"
        for basis in args.basis:
            for xc in args.xc:
                label = f"{method}-{system}-{basis}-{xc}"
                record = {
                    "case": label,
                    "method": method,
                    "system": system,
                    "basis": basis,
                    "xc": xc,
                    "atomic_numbers": zs,
                    "positions_angstrom": xyz,
                    "charge": charge,
                    "spin": spin,
                    "cart": basis == "STO-3G",
                }
                try:

                    def make_frame(positions, atomic_numbers=zs):
                        frame = molpy.Frame()
                        atoms = molpy.Block()
                        for axis, key in enumerate(["x", "y", "z"]):
                            atoms.insert(
                                key,
                                np.array(
                                    [p[axis] for p in positions], dtype=np.float64
                                ),
                            )
                        atoms.insert("mass", np.array(atomic_numbers, dtype=np.float64))
                        atoms.insert(
                            "atomic_number", np.array(atomic_numbers, dtype=np.uint32)
                        )
                        frame["atoms"] = atoms
                        frame.box = molpy.Box.cube(20.0)
                        return frame

                    frame = make_frame(xyz)
                    electrons = sum(zs) - charge
                    occupation = (
                        {}
                        if spin == 0
                        else {
                            "n_alpha": (electrons + spin) // 2,
                            "n_beta": (electrons - spin) // 2,
                        }
                    )
                    native = getattr(av, method)(
                        basis=basis,
                        xc=xc,
                        charge=charge,
                        tol=1e-12,
                        max_iter=200,
                        **occupation,
                    )
                    native.grid = "MatchPySCF"
                    native.init_guess = "minao"
                    native.damping = 0.0
                    native.conv_tol_grad = 1e-8
                    native.diis_space = 8
                    native.diis_start_cycle = 1
                    # Public forces are eV/Angstrom; match the native CODATA
                    # conversion rather than a separately rounded Bohr value.
                    native_force = np.asarray(native.forces(frame)) / 51.42206747467393
                    native_energy = float(native.energy())
                    mol = gto.M(
                        atom=list(zip(zs, xyz)),
                        unit="Angstrom",
                        basis=basis,
                        spin=spin,
                        charge=charge,
                        cart=record["cart"],
                        verbose=0,
                    )
                    oracle = getattr(dft, method)(mol)
                    oracle.xc = "cam-b3lyp" if xc == "CAMB3LYP" else xc
                    oracle.grids.level = 3
                    oracle.init_guess, oracle.damp = "minao", 0.0
                    oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = (
                        1e-12,
                        1e-8,
                        200,
                    )
                    oracle.diis_space, oracle.diis_start_cycle = 8, 1
                    oracle_energy = float(oracle.kernel())
                    gradient = oracle.nuc_grad_method()
                    gradient.grid_response = True
                    oracle_force = -gradient.kernel()
                    energy_error = abs(native_energy - oracle_energy)
                    force_error = float(np.max(np.abs(native_force - oracle_force)))
                    finite = bool(
                        np.isfinite(native_energy)
                        and np.isfinite(oracle_energy)
                        and np.isfinite(native_force).all()
                        and np.isfinite(oracle_force).all()
                    )
                    converged = bool(native.converged() and oracle.converged)
                    passed = (
                        finite
                        and converged
                        and energy_error <= 1e-8
                        and force_error <= 1e-6
                    )
                    if args.native_fd:
                        h_bohr = 1e-4
                        h_angstrom = h_bohr / 1.8897261245650618
                        fd_force = np.zeros((len(zs), 3))
                        for atom in range(len(zs)):
                            for axis in range(3):
                                energies = []
                                for sign in (1, -1):
                                    displaced = np.array(xyz, dtype=float)
                                    displaced[atom, axis] += sign * h_angstrom
                                    native.run(make_frame(displaced))
                                    if not native.converged():
                                        raise RuntimeError(
                                            "native displaced SCF did not converge"
                                        )
                                    energies.append(native.energy())
                                fd_force[atom, axis] = -(energies[0] - energies[1]) / (
                                    2 * h_bohr
                                )
                        fd_error = float(np.max(np.abs(native_force - fd_force)))
                        record["own_energy_fd_consistency"] = {
                            "step_bohr": h_bohr,
                            "force_ha_bohr": fd_force.tolist(),
                            "max_force_error_ha_bohr": fd_error,
                            "passed": bool(np.isfinite(fd_error) and fd_error <= 1e-6),
                            "external_qualification": False,
                        }
                        passed = (
                            passed and record["own_energy_fd_consistency"]["passed"]
                        )
                    record.update(
                        native_energy_ha=native_energy,
                        oracle_energy_ha=oracle_energy,
                        native_force_ha_bohr=native_force.tolist(),
                        oracle_force_ha_bohr=oracle_force.tolist(),
                        native_converged=bool(native.converged()),
                        oracle_converged=bool(oracle.converged),
                        energy_error_ha=energy_error,
                        max_force_error_ha_bohr=force_error,
                        finite=finite,
                        passed=passed,
                    )
                except Exception as error:  # noqa: BLE001 -- retain a failing scientific cell and continue the matrix
                    record.update(
                        passed=False,
                        error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                    )
                evidence["records"].append(record)
                failed |= not record["passed"]
                evidence["passed"] = not failed
                output.write_text(
                    json.dumps(evidence, indent=2, allow_nan=False) + "\n"
                )
                print(
                    label,
                    "PASS" if record["passed"] else "FAIL",
                    record.get("energy_error_ha"),
                    record.get("max_force_error_ha_bohr"),
                    record.get("error", ""),
                    flush=True,
                )
    # Keep the threadpool controller alive through every scientific call.
    del limits
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
