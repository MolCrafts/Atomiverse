#!/usr/bin/env python3
"""L-BFGS geometry optimization of an argon dimer to the analytic LJ minimum.

Demonstrates the always-available geometry-optimization surface:

* the ``Minimize`` driver assembled from components — ``set_optimizer``
  (singular: L-BFGS here), ``add_criterion`` (plural: the optimizer never
  judges itself), ``set_neighbor`` and ``set_evaluator``;
* the loop-control metadata a run returns (``iterations``, ``converged``,
  ``stop_reason``), as opposed to the physics, which is written back into
  the ``molpy.Frame``;
* a result with a closed-form answer: the 12-6 Lennard-Jones pair
  potential has its minimum at r* = 2^(1/6)·σ, so the optimized bond
  length can be checked against arithmetic instead of a stored number.

Recipe and gates are those of the C++ end-to-end case
``regressions/minimizer/lj_dimer``: start at 1.2 r*, L-BFGS(10, 0.05, 20),
200 iterations, |r − r*| < 0.01 Å.

MolCrafts stack only (molpy + numpy) — no ASE, no matplotlib. Headless,
single process, well under a second. It self-checks its own result and
**exits nonzero** when a check fails, so it can be smoke-run.

Run — development tree, extension built into ``build-py``::

    cd /path/to/Atomiverse
    PYTHONPATH="$PWD/build-py:$PWD/python/src" python python/examples/minimize_lj_dimer.py

Run — installed wheel (pure-Python phase 1, then the native phase 2)::

    pip install atomiverse
    python -m atomiverse config minimize --src /path/to/Atomiverse
    python -m atomiverse compile -j 8
    python -m atomiverse install
    python minimize_lj_dimer.py

Units: Å, eV, eV/Å.
"""

from __future__ import annotations

import sys

import molpy
import numpy as np

import atomiverse as av

# ── Lennard-Jones argon, the parameters of regressions/minimizer/lj_dimer ──
_EPS: float = 0.0104  # well depth [eV]
_SIGMA: float = 3.4  # collision diameter [Å]
_CUTOFF: float = 2.5 * _SIGMA  # [Å]
_SKIN: float = 0.3  # neighbor skin [Å]
_MASS_AR: float = 39.948  # [amu]
_Z_AR: int = 18  # atomic number

#: Analytic minimum of the 12-6 Lennard-Jones pair potential — arithmetic,
#: not a stored measurement (regressions/minimizer/lj_dimer/REFERENCE.md).
_R_STAR: float = 2.0 ** (1.0 / 6.0) * _SIGMA  # 3.816371 Å

#: Isolation cell: at 40 Å the dimer never sees a periodic image of itself.
_BOX: float = 40.0  # [Å]

#: Displaced start and iteration budget of the C++ case.
_R_START: float = 1.2 * _R_STAR  # [Å]
_MAX_ITER: int = 200

#: Geometry gate of reg_lj_dimer.cpp (`std::fabs(r - r_star) < 0.01`).
_R_TOL: float = 0.01  # [Å]


def build_dimer_frame(separation: float) -> "molpy.Frame":
    """Two argon atoms *separation* Å apart along x, as a ``molpy.Frame``.

    Geometry optimization needs no velocities: the driver evaluates the
    forces itself and walks downhill on them.

    @param separation: Initial interatomic distance [Å].
    @return A two-atom frame in a cubic cell of edge ``_BOX``.
    """
    atoms = molpy.Block()
    atoms.insert("x", np.array([0.0, separation], dtype=np.float64))
    atoms.insert("y", np.zeros(2, dtype=np.float64))
    atoms.insert("z", np.zeros(2, dtype=np.float64))
    atoms.insert("mass", np.full(2, _MASS_AR, dtype=np.float64))
    atoms.insert("atomic_number", np.full(2, _Z_AR, dtype=np.uint32))

    frame = molpy.Frame()
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(_BOX)
    return frame


def separation(frame: "molpy.Frame") -> float:
    """Distance between the two atoms of a dimer frame [Å].

    Plain Euclidean, no minimum image: the dimer sits at ~4 Å inside a
    40 Å cell, so the direct distance is already the nearest image.

    @param frame: Two-atom molpy.Frame.
    @return Interatomic distance [Å].
    """
    atoms = frame["atoms"]
    delta = np.array(
        [
            np.diff(np.asarray(atoms[key], dtype=np.float64))[0]
            for key in ("x", "y", "z")
        ],
        dtype=np.float64,
    )
    return float(np.linalg.norm(delta))


def main() -> int:
    frame = build_dimer_frame(_R_START)
    print(f"=== L-BFGS on an Ar2 Lennard-Jones dimer, start r = {_R_START:.6f} Å ===")

    opt = av.Minimize()
    opt.set_optimizer(av.LBFGS(memory=10, alpha0=0.05, max_ls=20))
    # Tolerances belong to the criterion, never to the optimizer: all five
    # must hold at once (max/RMS force [eV/Å], max/RMS step [Å], ΔE [eV]).
    opt.add_criterion(
        av.ForceConverged(
            ftol=0.01, rms_ftol=0.005, max_dtol=0.05, rms_dtol=0.02, etol=1e-6
        )
    )
    opt.set_neighbor(av.NeighborList(_CUTOFF, _SKIN))
    opt.set_evaluator(av.PairLJCut(_EPS, _SIGMA, _CUTOFF))

    result = opt.run(frame, _MAX_ITER)

    r = separation(frame)
    energy = frame.meta["energy"]  # [eV]
    forces = np.stack(
        [
            np.asarray(frame["atoms"][key], dtype=np.float64)
            for key in ("fx", "fy", "fz")
        ],
        axis=1,
    )
    max_force = float(np.max(np.linalg.norm(forces, axis=1)))  # [eV/Å]

    print(f"r  (optimized)   : {r:12.6f} Å")
    print(f"r* = 2^(1/6)·σ   : {_R_STAR:12.6f} Å")
    print(f"|r - r*|         : {abs(r - _R_STAR):12.3e} Å   (gate {_R_TOL} Å)")
    # PairLJCut is energy-shifted at the cutoff, so the well bottom sits a
    # little above -epsilon rather than exactly on it.
    print(f"energy           : {energy:12.8f} eV  (-ε = {-_EPS:.8f} eV)")
    print(f"max |F|          : {max_force:12.3e} eV/Å")
    print(
        f"iterations       : {result.iterations}  "
        f"converged={result.converged}  reason={result.stop_reason}"
    )

    failures: list[str] = []
    if not abs(r - _R_STAR) < _R_TOL:
        failures.append(
            f"optimized bond length {r!r} Å is not the analytic minimum "
            f"{_R_STAR!r} Å within {_R_TOL} Å"
        )
    if not result.converged:
        failures.append(
            f"the criterion never fired: stopped after {result.iterations} "
            f"iterations with reason {result.stop_reason!r}"
        )

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
