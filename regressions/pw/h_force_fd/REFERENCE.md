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

**Open shell, not closed shell.** One valence electron is a doublet
(\(S=1/2\)), so the run is collinear spin-polarized with `spin_polarized = true`
and `multiplicity = 2` (\(N_\uparrow=1\), \(N_\downarrow=0\)). The restricted,
fixed-occupation path rejects an odd electron count outright —
`internal::require_closed_shell` (`src/cpu/planewave/internal/pw_setup.cpp`),
called by `PW::run` (which is what the \(6N\) FD solves go through) and by
`pw_forces_analytic` (the evaluator half of this case) alike. The default
`multiplicity = 1` fails a *different* guard, the collinear path's parity
check: \(N_e + (M-1)\) must be even, which 1 + 0 is not.

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
Measured 2026-08-15 with GTH-PBE-q1, spin-polarized doublet:
\(|F|_\mathrm{FD}=1.4\times10^{-11}\),
\(|F|_\mathrm{evaluator}=6.58\times10^{-16}\) (\(E=-12.0277\) eV).

Public `PWEvaluator::with_gradient(true)` must write a finite energy and a force
vector satisfying the evaluator bound.
