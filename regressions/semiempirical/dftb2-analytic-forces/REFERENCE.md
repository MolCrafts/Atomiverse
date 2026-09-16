# Reference — DFTB2 analytic nuclear forces (H2O)

oracle: literature
oracle_version: Hourahine et al., J. Chem. Phys. 152, 124101 (2020); Elstner
  et al., Phys. Rev. B 58, 7260 (1998) — closed-form SCC-DFTB2 nuclear gradient
family_oracle: dftb+

No DFTB+ run produced an absolute energy or force for this geometry. The
gate is analytic \(F\) vs re-SCC FD of the same ATV energy. Do not read
`oracle: dftb+` into that identity.

## Method

SCC-DFTB2 (Elstner et al., Phys. Rev. B **58**, 7260 (1998)):

\[
E = \sum_i n_i\varepsilon_i - \tfrac12\sum_{AB}\Delta q_A\gamma_{AB}\Delta q_B + E_{\mathrm{rep}}
\]

Nuclear forces are the **closed-form** gradient at the SCC density (Hellmann–Feynman
+ Pulay energy-weighted density + Hamiltonian shift on \(\partial S\) + fixed-\(\Delta q\)
\(-\partial E_{\mathrm{SCC}}\) + repulsive) — **not** a re-SCC finite difference.
This is the DFTB+ force module boundary.

## Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB2`
- SK / Hubbard / repulsive: static `basis_data/dftb_mio_tables.h` (mio-**style**
  sketch for H/C/N/O). **Not** bit-identical to DFTB.org mio-1-1.

## Geometry (Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

Isolated molecule, cubic box 20 Å. `scc_tol = 1e-10`.

## Pins (hard-coded gates)

1. **Energy:** finite, \(|E| > 10^{-3}\) eV.
2. **Forces:** all components finite; \(|F|^2 > 10^{-8}\) at off-eq geometry.
3. **Translation:** \(|\sum_i F_i| \le 10^{-6}\) eV/Å per Cartesian.
4. **FD identity:** analytic \(F\) vs central **re-SCC** FD of the same \(E\)
   (\(h=10^{-4}\) Å) on \(F_{O,x}\) and \(F_{H1,y}\) within **1×10⁻⁵ eV/Å**
   (measured 2.5e-07 / 4.9e-09 eV/Å). A converged SCC-DFTB2 gradient is
   stationary, so this is FD-exact to the central-difference truncation error.

Offline pin note: `tools/tb_oracle/inputs/dftbplus/dftb2_analytic_forces_note.md`.

## Closed — the polar-gradient defect (found 2026-07-28, fixed 2026-07-28)

The gate used to read **1.0 eV/Å** and pass at \(|d| = 0.23\) / \(0.75\)
eV/Å, i.e. ~4× looser than the error it was meant to catch, with the standing
explanation "Mulliken charge response beyond fixed-\(\Delta q\) HF". That
explanation was wrong — a converged SCC-DFTB2 gradient is stationary and needs
no coupled-perturbed correction. Three real defects were stacked on top of each
other, which is why the earlier A/B on the \(\gamma\) term alone
(shipped 0.228, term dropped 1.020, sign flipped 2.268 eV/Å) concluded the
shipped form was "best of three" and closed the wrong door:

1. **The SCC never converged.** The undamped Picard iteration was a period-2
   limit cycle, so both the analytic force and the FD reference were sampled
   off different branches of a cycle. Fixed by Anderson charge mixing
   (`internal::ChargeMixer`, public `mix_beta`, default 0.3); a budget-exhausted
   SCC now throws.
2. **The reported energy was not the functional the gradient differentiates.**
   `E = E_band − E_dc` carried a spurious \(\sum_A V_A q^0_A\) (**7.85 eV**
   on this geometry) because \(E_{\mathrm{band}}\) contracts \(V_A\) with the
   **gross** Mulliken population. Now
   \(E = \mathrm{Tr}(\rho H^0) + E^{(2)} + E_{\mathrm{rep}}\).
3. **The \(\gamma\) gradient term had the wrong sign.** With
   \(E^{(2)} = \tfrac12\sum_{AB}\gamma_{AB}\Delta q_A\Delta q_B\) and
   bond \(= R_B - R_A\), the ordered double sum gives the pair term
   \(\gamma_{AB}\Delta q_A\Delta q_B\) and \(\partial r/\partial\) bond
   \(= +\hat r\), so \(\partial E^{(2)}/\partial\) bond
   \(= +\Delta q_A\Delta q_B\,d\gamma/dr\). The code carried a minus.
   Present in all three assemblies (`analytic_forces_restricted`,
   `analytic_forces_unrestricted`, `analytic_forces_3ob`).

Residual after the fix, measured on this case:

| case | \(\Delta q\) | analytic vs re-SCC FD (before → after) |
| ---- | ------------ | -------------------------------------- |
| O₂ triplet (`reg_dftb2-spin-analytic-forces`) | 0 | 3.0e-07 → 3.0e-07 eV/Å |
| H₂O MioSketch (this case) | ≠ 0 | 0.23 / 0.75 → **2.5e-07 / 4.9e-09** eV/Å |
| H₂O 3ob (`reg_dftb2-analytic-forces-threeob`) | ≠ 0 | 0.84 / 0.24 → **4.4e-07 / 2.0e-09** eV/Å |
| H₂O stress (`reg_dftb2-analytic-stress`) | ≠ 0 | 1.3e-04 → **1.1e-10** eV/Å³ |

The energy of this case moved from −44.60971806 to **−91.70909683 eV** as a
consequence of (1) and (2): the old number was a limit-cycle snapshot over a
double-counting route that was wrong at second order. It is an ATV twin
snapshot, not an oracle value — no DFTB+ identity is claimed for it.

**Closed 2026-07-29:** the owed open-shell \(\Delta q \ne 0\) gate landed as
the **NO doublet** in `regressions/semiempirical/dftb2-spin-analytic-stress`,
and `analytic_forces_unrestricted` / `analytic_forces_restricted` were folded
into one spin-resolved `analytic_forces_stress`, so a sign defect in the
\(\gamma\) or Pulay term can no longer survive in one copy while being fixed in
another.

## Perf

`perf_exempt: true` — no DFTB+ absolute-force twin was run for this
geometry (FD identity only). Analytic path replaces \(6N\) full re-SCC
force evaluations with one post-SCC contraction.
