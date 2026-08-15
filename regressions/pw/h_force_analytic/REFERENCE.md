# Reference — plane-wave analytic ionic forces (GTH-PBE-q1)

## Method

Hellmann–Feynman local + Ewald ionic forces after one plane-wave SCF
(`pw_forces_analytic`). Plane waves are cell-fixed. See Martin, *Electronic
Structure* (CUP), and Quantum ESPRESSO `force_lc.f90` / `force_ew.f90` for the
production form.

**Coverage limit — read before trusting this case.** The nonlocal KB term is
zero here, because `GTH-PBE-q1` for hydrogen is purely local: it carries a
local channel only (`kPot_H_GTH_PBE_q1` declares zero projector channels in
`include/atomiverse/cpu/gpw/basis_data/gpw_tables.h`). So this case exercises
the local + Ewald force terms and **cannot** detect a dropped or wrong KB
projector force. That coverage lives in the Si/GTH gate in
`tests/cpu/planewave/test_pw_forces_fd.cpp`, which must be kept green
alongside this one.

FD of the total energy (`pw_forces_fd`) remains the numerical oracle; this
regression pins the **symmetry** result that both paths must satisfy.

## System

One hydrogen atom (Z=1) at the origin of a cubic cell \(L=6\) Å. Settings:
\(E_\mathrm{cut}=5\) Ry, Γ-only k-mesh, `max_iter=8`, `gth_name = "GTH-PBE-q1"`
(analytic Goedecker–Teter–Hutter pseudopotential). The bare-Coulomb "toy" local
potential this case originally used has been deleted from the tree.

**Open shell, not closed shell.** One valence electron is a doublet
(\(S=1/2\)), so the run is collinear spin-polarized with `spin_polarized = true`
and `multiplicity = 2` (\(N_\uparrow=1\), \(N_\downarrow=0\)). The restricted,
fixed-occupation path rejects an odd electron count outright —
`internal::require_closed_shell` (`src/cpu/planewave/internal/pw_setup.cpp`),
the one gate `PW::run`, `pw_forces_analytic` and `pw_stress` all call. The
default `multiplicity = 1` fails a *different* guard, the collinear path's
parity check: \(N_e + (M-1)\) must be even, which 1 + 0 is not.

## Pin

By lattice symmetry the ionic force on a single atom at a high-symmetry site is
**exactly zero**. This is a symmetry identity and does *not* depend on the SCF
reaching `cfg.tol` — the 8-iteration run deliberately stops short — so the
bound is a machine-precision one:

\[
|F| < 10^{-9}~\mathrm{eV/Å}.
\]

Measured 2026-08-15 with GTH-PBE-q1, spin-polarized doublet:
\(|F| = 6.58\times10^{-16}\) on both the analytic and the evaluator path
(\(E = -12.0277\) eV; SCF converged in 5 iterations, \(m = 1\)).

Public `PWEvaluator::with_gradient(true)` must write a finite energy and a force
vector satisfying the same bound (analytic path).
