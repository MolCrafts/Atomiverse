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

**Open shell, not closed shell.** One valence electron is a doublet
(\(S=1/2\)), so the run is collinear spin-polarized with `spin_polarized = true`
and `multiplicity = 2` (\(N_\uparrow=1\), \(N_\downarrow=0\)). The restricted,
fixed-occupation path rejects an odd electron count outright —
`internal::require_closed_shell` (`src/cpu/planewave/internal/pw_setup.cpp`),
which `pw_stress` calls exactly as `PW::run` and `pw_forces_analytic` do. The
default `multiplicity = 1` fails a *different* guard, the collinear path's
parity check: \(N_e + (M-1)\) must be even, which 1 + 0 is not.

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

Measured 2026-08-15 with GTH-PBE-q1, spin-polarized doublet: off-diagonals
\(\le 1.049\times10^{-14}\); \(\sigma_{xx}=-0.0050847238\),
\(\sigma_{yy}=-0.005084722912\), \(\sigma_{zz}=-0.005084722795\), i.e. spread
\(1.006\times10^{-9}\) against a signal of \(0.005085\)
(\(2.0\times10^{-7}\) relative). Identical to the printed 10 significant
digits at 1, 4 and 8 OpenMP threads.
