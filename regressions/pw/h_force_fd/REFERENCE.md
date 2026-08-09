# Reference — plane-wave FD ionic forces (GTH-PBE-q1)

## Method

Central finite differences of the plane-wave total energy:

\[
F_{a\alpha} = -\frac{E(R_{a\alpha}+h)-E(R_{a\alpha}-h)}{2h}
\]

with \(E\) converted to eV and \(h\) in Å (`pw_forces_fd`). Same oracle form as
GPW's `gpw_forces_fd` before analytic forces. See Martin, *Electronic Structure*
(CUP), Hellmann–Feynman chapter for the physical force; FD is the standard
numerical check.

## System

One hydrogen atom (Z=1) at the origin of a cubic cell \(L=6\) Å. Settings:
\(E_\mathrm{cut}=5\) Ry, Γ-only k-mesh, `max_iter=8`, `gth_name = "GTH-PBE-q1"`
(analytic Goedecker–Teter–Hutter pseudopotential). The bare-Coulomb "toy"
local potential this case originally used has been deleted from the tree.

## Pin

By lattice symmetry the ionic force on a single atom at a high-symmetry site is
**exactly zero**. This is a symmetry identity, so it does *not* depend on the
SCF reaching `cfg.tol` — the 8-iteration run deliberately stops short. Only
numerical noise survives, and the two paths have different floors:

\[
|F|_\mathrm{FD} < 10^{-4}~\mathrm{eV/Å},
\qquad
|F|_\mathrm{evaluator} < 10^{-9}~\mathrm{eV/Å}.
\]

The FD bound is set by the \(h=5\times10^{-3}\) Å step and the 1e-4 SCF
tolerance, not by machine epsilon; the evaluator returns the analytic gradient.
Measured 2026-08-09 with GTH-PBE-q1: \(|F|_\mathrm{FD}=8.3\times10^{-7}\),
\(|F|_\mathrm{evaluator}=3.9\times10^{-15}\).

Public `PWEvaluator::with_gradient(true)` must write a finite energy and a force
vector satisfying the evaluator bound.
