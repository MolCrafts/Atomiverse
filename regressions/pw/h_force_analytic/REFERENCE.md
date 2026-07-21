# Reference — plane-wave analytic ionic forces (toy PP)

## Method

Hellmann–Feynman local + Ewald ionic forces after one plane-wave SCF
(`pw_forces_analytic`). Plane waves are cell-fixed; nonlocal KB terms are zero
on the toy (empty UPF) path. See Martin, *Electronic Structure* (CUP), and
Quantum ESPRESSO `force_lc.f90` / `force_ew.f90` for the production form.

FD of the total energy (`pw_forces_fd`) remains the numerical oracle; this
regression pins the **symmetry** result that both paths must satisfy.

## System

One hydrogen atom (Z=1) at the origin of a cubic cell \(L=6\) Å. Settings:
\(E_\mathrm{cut}=5\) Ry, Γ-only k-mesh, `max_iter=8`, empty `upf_path` (toy
local potential).

## Pin

By lattice symmetry the ionic force on a single atom at a high-symmetry site is
**exactly zero**. Residual noise from a short SCF is bounded by

\[
|F| < 0.5~\mathrm{eV/Å}.
\]

Public `PWEvaluator::with_gradient(true)` must write a finite energy and a force
vector satisfying the same bound (analytic path).
