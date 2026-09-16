# qmmm_oniom_h2_ar

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
