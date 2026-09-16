"""Gaussian-basis SCF methods (mirrors ``cpu/scf/``).

SCF is short for *self-consistent field*. These are **mean-field** methods:
each electron is treated as moving in the averaged field of all the others,
and because that field is built from the very orbitals it determines, the
equations are solved by iterating — build the field, diagonalize, rebuild —
until the total energy and the residual stop changing. The orbitals are
expanded in a fixed set of Gaussian functions, the *basis set*.

The surface is the **concrete method** — ``RHF``, ``UHF``, ``RKS``, ``UKS``.
There is no ``SCF`` facade, no ``SCFConfig`` bag and no factory: every
parameter is a constructor argument, and the only I/O currency is a
``molpy.Frame``.

Two letters name each method. ``R`` (restricted) means the spin-up (alpha)
and spin-down (beta) electrons share one set of spatial orbitals, i.e. a
closed shell with every occupied orbital doubly filled; ``U``
(unrestricted) gives the two spin channels their own orbitals, which is
what an open-shell system with unpaired electrons needs. ``HF``
(Hartree-Fock) treats electron exchange exactly and leaves out electron
correlation; ``KS`` (Kohn-Sham density-functional theory) replaces both by
an approximate exchange-correlation *functional* — a recipe mapping the
electron density to an energy — chosen by the ``functional`` argument.

Example::

    import atomiverse as av

    rhf = av.RHF(basis="STO-3G", charge=0, tol=1e-9, max_iter=100)
    rhf.run(frame)                     # frame.meta["energy"] is eV
    print(rhf.energy(), rhf.converged(), rhf.iterations())   # Hartree

    forces = rhf.forces(frame)         # (N, 3) numpy array [eV/A]

Constructors
------------

``RHF(basis="STO-3G", charge=0, tol=1e-9, max_iter=100)``
    Closed-shell Hartree-Fock.

``UHF(basis="STO-3G", charge=0, multiplicity=1, tol=1e-8, max_iter=100)``
    Open-shell Hartree-Fock. ``multiplicity`` is M = 2S + 1 with S the total
    spin — 1 singlet (all electrons paired), 2 doublet (one unpaired), 3
    triplet (two unpaired). The alpha / beta split is resolved at ``run`` /
    ``forces`` time, once the frame fixes the electron count
    N_e = sum(Z) - charge: n_alpha = (N_e + M - 1)/2 and
    n_beta = (N_e - M + 1)/2. Those are integers only for an even N_e with an
    odd M or an odd N_e with an even M; any other pairing, or M > N_e + 1,
    raises ``ValueError`` there rather than at construction.

``RKS(functional="PBE", basis="STO-3G", charge=0, tol=1e-9, max_iter=100)``
    Closed-shell Kohn-Sham DFT.

``UKS(functional="LSDA", basis="STO-3G", charge=0, multiplicity=1,
tol=1e-8, max_iter=100)``
    Open-shell Kohn-Sham DFT; ``multiplicity`` as for ``UHF``.

``basis`` is one of ``STO-3G``, ``6-31G``, ``6-31G*``, ``6-31+G*``,
``def2-SVP``, ``def2-TZVP``, ``aug-cc-pVDZ``; ``functional`` is one of
``LSDA``, ``PBE``, ``PBEsol``, ``RevPBE``, ``RPBE``, ``BLYP``,
``OptB88VdW``, ``B3LYP``, ``PBE0``, ``B2PLYP``. An unsupported name raises
``ValueError`` listing the supported ones — there is no fallback basis or
functional.

Convergence
-----------

``tol`` is the threshold on |dE|, the total-energy change between two
successive iterations [Hartree]. ``converged()`` is True only when that
*and* a second measure are met: the Frobenius norm of the DIIS error matrix
below sqrt(``tol``). DIIS (*direct inversion of the iterative subspace*) is
the standard SCF accelerator; its error matrix measures how far the current
electron density is from self-consistency, so it is the residual of the
loop. ``max_iter`` caps the iteration count — hitting it leaves
``converged()`` False rather than raising, so check it.

Units
-----

``charge`` [e], ``tol`` [Hartree]; ``energy()`` returns Hartree (the atomic
energy unit the solver works in, 1 Ha = 27.211386 eV) while the frame's
``meta["energy"]`` is the same number in eV (the public Frame currency), and
``forces()`` is eV/A. ``forces()`` also writes ``fx`` / ``fy`` / ``fz``
back to the frame, the same write-back contract as MD; ``run()`` does not —
it computes no forces, and a frame's ``fx``/``fy``/``fz`` come back at
whatever they were (zeros for a fresh frame).

Interop
-------

These methods drive their own SCF loop, so they are used directly rather
than through ``single_point`` — but ``single_point(frame, rhf)`` accepts
them for symmetry with the AM1 evaluator and simply calls ``run``.

Capability gate: the methods exist only on an extension built with
``ATV_BUILD_HF=ON``. This module imports on every build and exports what
was compiled, so ``__all__`` reflects the real capability instead of
pretending an unavailable method exists.

Pinned numbers behind this surface: ``python/tests/test_scf.py`` grades
RHF/STO-3G water against the -74.961754 Ha value of
``regressions/scf/h2o_rhf_sto3g/REFERENCE.md`` (1e-3 Ha window), UHF/6-31G*
N2+ against ``regressions/scf/n2_ionization``, B3LYP/def2-SVP water against
``regressions/scf/b3lyp-fused-jk-h2o``, and the oxygen force against
``regressions/scf/h2o_rhf_sto3g_grad``.
"""

from __future__ import annotations

from atomiverse._binding import _cpp

if hasattr(_cpp, "RHF"):
    RHF = _cpp.RHF
    UHF = _cpp.UHF
    RKS = _cpp.RKS
    UKS = _cpp.UKS
    __all__ = ["RHF", "UHF", "RKS", "UKS"]
else:
    __all__: list[str] = []
