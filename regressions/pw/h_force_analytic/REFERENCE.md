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

## Pin

By lattice symmetry the ionic force on a single atom at a high-symmetry site is
**exactly zero**. This is a symmetry identity and does *not* depend on the SCF
reaching `cfg.tol` — the 8-iteration run deliberately stops short — so the
bound is a machine-precision one:

\[
|F| < 10^{-9}~\mathrm{eV/Å}.
\]

Measured 2026-08-09 with GTH-PBE-q1: \(|F| = 3.9\times10^{-15}\) on both the
analytic and the evaluator path.

Public `PWEvaluator::with_gradient(true)` must write a finite energy and a force
vector satisfying the same bound (analytic path).
