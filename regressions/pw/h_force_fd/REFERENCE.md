# Reference — plane-wave FD ionic forces (toy PP)

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
\(E_\mathrm{cut}=5\) Ry, Γ-only k-mesh, `max_iter=8`, empty `upf_path` (toy
local potential).

## Pin

By lattice symmetry the ionic force on a single atom at a high-symmetry site is
**exactly zero**. Numerical FD + short SCF leave only residual noise:

\[
|F| < 0.5~\mathrm{eV/Å}.
\]

Public `PWEvaluator::with_gradient(true)` must write a finite energy and a force
vector satisfying the same bound.
