"""Gaussian-basis SCF methods (mirrors ``cpu/scf/``).

SCF is short for *self-consistent field*. These are **mean-field** methods:
each electron is treated as moving in the averaged field of all the others,
and because that field is built from the very orbitals it determines, the
equations are solved by iterating — build the field, diagonalize, rebuild —
until the total energy and the residual stop changing. The orbitals are
expanded in a fixed set of Gaussian functions, the *basis set*.

The surface is the **concrete method** — ``RHF``, ``UHF``, ``ROHF``, ``GHF``,
``RKS``, ``UKS``; CUDA builds additionally carry ``CudaRHF`` and ``CudaUHF``.
There is no ``SCF`` facade, no ``SCFConfig`` bag and no factory: every
identity parameter is a constructor argument — the identity
(``basis``, ``charge``, ``n_alpha`` / ``n_beta``,
``xc``) is read-only afterwards, and the only I/O currency is a
``molpy.Frame``.

CPU HF uses one automatic exact J/K strategy with eightfold integral caching
within ``max_memory_mb`` (4000 MB by default). OpenMP uses the runtime thread
count by default; ``set_threads(n)`` overrides it. Other controls include
``init_guess`` (``"hcore"`` or ``"minao"``),
``damping``, ``level_shift``, ``conv_tol_grad``, ``diis_space`` and
``diis_start_cycle``. They forward native controls; unsupported method/control
combinations raise in the native solver. CUDA controls retain their native
constructor settings rather than claiming unsupported CPU controls.

Two letters name each method. ``R`` (restricted) means the spin-up (alpha)
and spin-down (beta) electrons share one set of spatial orbitals, i.e. a
closed shell with every occupied orbital doubly filled; ``U``
(unrestricted) gives the two spin channels their own orbitals, which is
what an open-shell system with unpaired electrons needs. ``HF``
(Hartree-Fock) treats electron exchange exactly and leaves out electron
correlation; ``KS`` (Kohn-Sham density-functional theory) replaces both by
an approximate exchange-correlation *functional* — a recipe mapping the
electron density to an energy — chosen by the ``xc`` argument.

Example::

    import atomiverse as av

    rhf = av.RHF(basis="STO-3G", charge=0, tol=1e-9, max_iter=100)
    rhf.run(frame)                     # frame.meta["energy"] is eV
    print(rhf.energy(), rhf.converged(), rhf.iterations())   # Hartree

    forces = rhf.forces(frame)         # (N, 3) numpy array [eV/A]

Constructors
------------

``RHF(basis, charge=0, tol=1e-9, max_iter=100)``
    Closed-shell Hartree-Fock.

``UHF(basis, n_alpha, n_beta, charge=0, tol=1e-8, max_iter=100)``
    Open-shell Hartree-Fock with ``n_alpha`` spin-up and ``n_beta``
    spin-down electrons. For spin multiplicity M = 2S + 1 (S the total spin —
    1 singlet, 2 doublet, 3 triplet) and electron count
    N_e = sum(Z) - charge: n_alpha = (N_e + M - 1)/2 and
    n_beta = (N_e - M + 1)/2. ``n_alpha + n_beta`` must equal the frame's
    N_e; a mismatch raises ``ValueError`` at ``run`` / ``forces``, once the
    frame fixes N_e.

``RKS(basis, xc, charge=0, tol=1e-9, max_iter=100)``
    Closed-shell Kohn-Sham DFT.

``UKS(basis, n_alpha, n_beta, xc="LSDA", charge=0, tol=1e-8, max_iter=100)``
    Open-shell Kohn-Sham DFT; ``n_alpha`` / ``n_beta`` as for ``UHF``.

``ROHF(basis, n_alpha, n_beta, charge=0, tol=1e-10, max_iter=200)``
    Restricted open-shell HF with shared spatial orbitals and split occupation.

``GHF(basis, n_electrons, tol=1e-10, max_iter=200, spin_rotation=0.2)``
    Real two-component spinor HF. ``spin_rotation`` forwards the native per-AO
    initial moment rotation [radians]. After convergence,
    ``spin_density_block("alpha", "beta")`` returns an owned AO density block.

``CudaRHF`` / ``CudaUHF`` take the same constructor arguments as ``RHF`` /
``UHF``. They forward the existing public CUDA methods; unsupported basis,
element or force envelopes raise without a CPU fallback. ``run`` computes
energy only, ``forces`` explicitly computes analytic forces, and both complete
``cudaDeviceSynchronize`` before returning. CUDA nonconvergence raises and
does not publish the failed candidate to the supplied Frame.

``backend_info()`` reports the actual CPU compiler, OpenMP maximum team size
and the team size observed by a parallel region in the calling process.
``cuda_backend_info()`` reports GPU identities, CUDA driver/runtime versions,
current-device free/total memory snapshots, and completed synchronization.
``cuda_synchronize()`` completes work on the current device, for timing or
explicit cleanup boundaries. The CUDA functions exist only on CUDA HF builds;
memory snapshots are not peak-memory measurements.

``basis`` is one of ``STO-3G``, ``6-31G``, ``6-31G*``, ``6-31+G*``,
``def2-SVP``, ``def2-TZVP``, ``aug-cc-pVDZ``, ``cc-pVDZ``; ``xc`` is one of
``LSDA``, ``PBE``, ``PBEsol``, ``RevPBE``, ``RPBE``, ``BLYP``,
``OptB88VdW``, ``B3LYP``, ``PBE0``, ``B2PLYP``, ``CAMB3LYP``. An unsupported name raises
``ValueError`` listing the supported ones — there is no fallback basis or
functional.

Convergence
-----------

For CPU methods, ``tol`` is the threshold on |dE|, the total-energy change between two
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

Capability gate: the methods exist only on an extension built with the
``hf`` module (``ATV_BUILD_HF=ON``). On one built without it, importing this
module raises ``MissingDomainError`` (an ``ImportError``) naming the flag.

Pinned numbers behind this surface: ``python/tests/test_scf.py`` grades
RHF/STO-3G water against the -74.961754 Ha value of
``regressions/scf/rhf/REFERENCE.md`` (1e-3 Ha window), UHF/6-31G*
N2+ against ``regressions/scf/uhf``, B3LYP/def2-SVP water against
``regressions/scf/rks``, and the oxygen force against
``regressions/scf/rhf``.
"""

from __future__ import annotations

from atomiverse._binding import _cpp, require_domain

require_domain(__name__, "hf")

RHF = _cpp.RHF
UHF = _cpp.UHF
RKS = _cpp.RKS
UKS = _cpp.UKS
ROHF = _cpp.ROHF
GHF = _cpp.GHF
backend_info = _cpp.hf_backend_info
__all__ = ["RHF", "UHF", "ROHF", "GHF", "RKS", "UKS", "backend_info"]

if hasattr(_cpp, "CudaRHF"):
    CudaRHF = _cpp.CudaRHF
    CudaUHF = _cpp.CudaUHF
    cuda_backend_info = _cpp.cuda_hf_backend_info
    cuda_synchronize = _cpp.cuda_hf_synchronize
    __all__ += ["CudaRHF", "CudaUHF", "cuda_backend_info", "cuda_synchronize"]
