# multiscale/oniom

One case; every section below is run by `atv_regressions --case multiscale/oniom` and keeps its own oracle record.


## qmmm_oniom_h2_ar

**Method:** Two-layer subtractive ONIOM (Svensson et al., *J. Phys. Chem.*
**100**, 19357 (1996)):

\[
E = E_{\mathrm{high}}(\mathrm{H_2}) + E_{\mathrm{low}}(\mathrm{H_2+Ar})
  - E_{\mathrm{low}}(\mathrm{H_2})
\]

- high: DFTB2 vacuum on QM mask (H₂)
- low: shifted 12-6 LJ with H and Ar diagonals (Lorentz–Berthelot H–Ar)

**Pin:** algebraic identity \(E = E_h + E_{\ell,r} - E_{\ell,m}\) to \(10^{-10}\)
eV (same binary, same step) — not a third-party total-energy oracle. Nonzero
\(E_{\mathrm{low}}(\mathrm{model})\) witnesses that the model subtraction is
active (H–H LJ diagonal present).

## Engine lifecycle and NDDO mechanical coupling

The existing `qmmm_oniom_h2_ar` section also checks repeated DFTB2 preparation
and borrowing across parent Evaluate lifetimes. AM1/RM1 mechanical coupling
uses the distorted water geometry in the executed OpenMOPAC 23.2.5 fixtures,
plus one Ar MM site. The QM contribution is compared with the public standalone
method; the three QM–Ar Lennard-Jones pairs use a closed-form shifted-energy
and force reference with Lorentz–Berthelot mixing. A separate potential adds
0.75 eV and 0.125 eV/Å to verify accumulation.

Initial-force and trial evaluations do not create or advance an accepted
electronic guess, including with the explicit cold-start policy enabled for
probes. Four subsequent normal geometries match an unprobed method in SCF
iteration count, energy (1e-10 eV) and every force component (1e-9 eV/Å).
A deliberately exhausted SCF leaves accepted history usable for retry. The
same borrowed engine is exercised through a second parent Runtime and after
that parent's destruction. These are composition checks, not additional
external oracle values or broader method/element coverage.
