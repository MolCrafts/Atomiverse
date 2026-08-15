#!/usr/bin/env python3
"""AM1 semiempirical single point on methane: energy, ΔHf and Mulliken charges.

Demonstrates the always-available electronic-structure surface that needs
no optional build flag (AM1 ships on every wheel):

* ``single_point(frame, evaluator)`` — one solve at one fixed geometry,
  the counterpart of C++ ``cpu::single_point``;
* the two result channels: the public ``molrs.Frame`` (total energy [eV]
  in ``frame.meta``, per-atom Mulliken charges [e] written back into the
  ``charge`` column) and the solver-internal detail on
  ``evaluator.last_result()`` (Hartree / kcal·mol⁻¹, iteration count);
* what AM1 is for — a fitted NDDO model of the valence electrons, cheap
  enough to run interactively yet giving real partial charges.

The geometry is the tetrahedral CH4 the MOPAC AM1 pin was taken at (H at
±0.629118 Å per axis, i.e. C-H = 1.0897 Å); ``python/tests/test_semiempirical.py``
grades these same charges against that pinned MOPAC reference. This script
only checks the physics that must hold for any neutral closed-shell
molecule, so it stays a demonstration and not a second copy of the pin.

MolCrafts stack only (molrs + numpy) — no ASE, no matplotlib. Headless,
single process, well under a second. It self-checks its own result and
**exits nonzero** when a check fails, so it can be smoke-run.

Run — development tree, extension built into ``build-py``::

    cd /path/to/Atomiverse
    PYTHONPATH="$PWD/build-py:$PWD/python/src" python python/examples/am1_ch4.py

Run — installed wheel (pure-Python phase 1, then the native phase 2)::

    pip install atomiverse
    python -m atomiverse config am1 --src /path/to/Atomiverse
    python -m atomiverse compile -j 8
    python -m atomiverse install
    python am1_ch4.py

Units: Å, eV, Hartree, kcal/mol, e.
"""

from __future__ import annotations

import sys

import molrs
import numpy as np

import atomiverse as av

#: Tetrahedral CH4: carbon at the origin, the four hydrogens along the
#: body diagonals at ±_H per axis, giving C-H = sqrt(3)·_H = 1.0897 Å.
_H: float = 0.629118  # [Å]

#: Element symbols and masses of the five sites, C first.
_SYMBOLS: tuple[str, ...] = ("C", "H", "H", "H", "H")
_ATOMIC_NUMBERS: tuple[int, ...] = (6, 1, 1, 1, 1)
_MASSES: tuple[float, ...] = (12.011, 1.008, 1.008, 1.008, 1.008)  # [amu]

#: Isolation cell — AM1 is a molecular method, so the box only keeps the
#: frame well formed; nothing periodic happens in it.
_BOX: float = 20.0  # [Å]

#: Mulliken charges of a neutral molecule sum to zero by construction; the
#: only deviation is floating-point round-off in the population trace.
_TOL_NEUTRALITY: float = 1e-6  # [e]


def build_ch4_frame() -> "molrs.Frame":
    """Tetrahedral methane as a ``molrs.Frame``.

    The ``atomic_number`` column [Z] is how the evaluator knows which
    element's AM1 parameters to apply; ``mass`` is carried so the same
    frame could be handed to MD (Born-Oppenheimer dynamics) unchanged.

    @return A five-atom frame in a cubic cell of edge ``_BOX``.
    """
    atoms = molrs.Block()
    atoms.insert("x", np.array([0.0, _H, -_H, -_H, _H], dtype=np.float64))
    atoms.insert("y", np.array([0.0, _H, -_H, _H, -_H], dtype=np.float64))
    atoms.insert("z", np.array([0.0, _H, _H, -_H, -_H], dtype=np.float64))
    atoms.insert("mass", np.array(_MASSES, dtype=np.float64))
    atoms.insert("atomic_number", np.array(_ATOMIC_NUMBERS, dtype=np.uint32))

    frame = molrs.Frame()
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(_BOX)
    return frame


def main() -> int:
    frame = build_ch4_frame()
    print("=== AM1 single point on CH4 (C-H = 1.0897 Å) ===")

    evaluator = av.AM1Evaluator(charge=0)
    av.single_point(frame, evaluator)
    result = evaluator.last_result()

    # Two views of the same solve: the frame carries the public currency
    # (energy [eV], charges [e]); last_result() carries solver detail.
    energy_ev = frame.meta["energy"]
    frame_charges = np.asarray(frame["atoms"]["charge"], dtype=np.float64)
    # The checks below read last_result(): those charges are always double,
    # while the frame column is StorageReal and is single precision on an
    # ATV_REAL_STORAGE=float wheel.
    charges = np.asarray(result.mulliken_charges, dtype=np.float64)

    print(f"converged        : {result.converged}  in {result.iterations} iterations")
    print(f"total energy     : {energy_ev:14.6f} eV")
    print(f"electronic energy: {result.electronic_energy_hartree:14.6f} Ha")
    print(f"heat of formation: {result.heat_of_formation_kcal_mol:14.6f} kcal/mol")
    print("Mulliken charges [e] (frame charge column):")
    for symbol, charge in zip(_SYMBOLS, frame_charges):
        print(f"  {symbol:<2}  {charge:+.6f}")
    print(f"  sum {charges.sum():+.3e}")

    failures: list[str] = []
    if not result.converged:
        failures.append(f"SCF did not converge in {result.iterations} iterations")
    if not np.isfinite(energy_ev):
        failures.append(f"total energy is not finite: {energy_ev!r} eV")
    if not abs(charges.sum()) < _TOL_NEUTRALITY:
        failures.append(
            f"Mulliken charges of neutral CH4 sum to {charges.sum()!r} e, "
            f"not 0 within {_TOL_NEUTRALITY} e"
        )
    if not charges[0] < 0.0:
        # Carbon is the more electronegative partner in a C-H bond, so it
        # takes the negative end of the four bond dipoles.
        failures.append(f"carbon carries a non-negative charge: {charges[0]!r} e")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
