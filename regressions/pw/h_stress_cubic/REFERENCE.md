# Reference — plane-wave cell stress (cubic single H)

## Method

Cell stress is obtained from central finite differences of the plane-wave SCF
energy under symmetric strain at fixed fractional coordinates (`pw_stress`).
This is the same oracle class used for early ionic FD forces. Nielsen–Martin
analytic decomposition is a follow-on.

## System

One H at the origin of a cubic cell \(L=6\) Å. Toy PP, \(E_\mathrm{cut}=5\) Ry,
Γ-only, short SCF.

## Pin

By cubic symmetry, off-diagonal Voigt components (xy, xz, yz) vanish within
noise:

\[
|\sigma_{xy}|,|\sigma_{xz}|,|\sigma_{yz}| < 0.5~\mathrm{eV/Å^3}.
\]

Diagonal components are finite and mutually equal within \(0.5~\mathrm{eV/Å^3}\).
