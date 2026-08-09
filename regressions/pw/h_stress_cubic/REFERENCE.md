# Reference — plane-wave cell stress (cubic single H)

## Method

Cell stress is obtained from central finite differences of the plane-wave SCF
energy under symmetric strain at fixed fractional coordinates (`pw_stress`).
This is the same oracle class used for early ionic FD forces. Nielsen–Martin
analytic decomposition is a follow-on.

## System

One H at the origin of a cubic cell \(L=6\) Å. `gth_name = "GTH-PBE-q1"` (analytic Goedecker–Teter–Hutter
pseudopotential; the bare-Coulomb "toy" potential this case originally used has
been deleted from the tree), \(E_\mathrm{cut}=5\) Ry,
Γ-only, short SCF.

## Pin

By cubic symmetry, off-diagonal Voigt components (xy, xz, yz) vanish within
noise:

\[
|\sigma_{xy}|,|\sigma_{xz}|,|\sigma_{yz}| < 10^{-9}~\mathrm{eV/Å^3}.
\]

Diagonal components are finite and mutually equal within

\[
\max_{i\neq j}|\sigma_{ii}-\sigma_{jj}| < 10^{-6}~\mathrm{eV/Å^3}.
\]

Both are symmetry identities and do *not* depend on the SCF reaching
`cfg.tol`, but they have different numerical floors and so carry different
bounds. The off-diagonals are structurally zero (machine precision), whereas
each *diagonal* component comes from an independent central-difference
evaluation inside `pw_stress`, so its floor is set by the FD step and the
1e-4 / 8-iteration SCF.

Measured 2026-08-09 with GTH-PBE-q1: off-diagonals \(\le 1.1\times10^{-14}\);
\(\sigma_{xx}=-0.011384554\), \(\sigma_{yy}=-0.011384539\),
\(\sigma_{zz}=-0.011384546\), i.e. spread \(1.5\times10^{-8}\) against a
signal of \(0.01138\) (\(1.3\times10^{-6}\) relative). Bit-identical at 1, 4
and 8 OpenMP threads.
