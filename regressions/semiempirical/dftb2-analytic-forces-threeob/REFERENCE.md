# Reference — DFTB2 ThreeOb analytic nuclear forces (H2O)

oracle: literature
oracle_version: Hourahine et al., J. Chem. Phys. 152, 124101 (2020); Gaus
  et al., J. Chem. Theory Comput. 9, 338 (2013) 3ob-3-1 — closed-form
  ThreeOb nuclear gradient
family_oracle: dftb+

No DFTB+ run produced an absolute energy or force for this off-eq H2O
geometry. The gate is analytic \(F\) vs re-SCC FD of the same ATV 3ob
energy. The CH4 3ob energy pin (`dftb2-3ob-organic`) is a different
system and is not this case's oracle. Do not read `oracle: dftb+` into
the FD identity.

## Method

SCC-DFTB2 on static **3ob-3-1** organic SK (`DftbSkSet::ThreeOb`):

\[
E = \mathrm{Tr}(P H_0) + \tfrac12\sum_{AB}\Delta q_A\gamma_{AB}\Delta q_B + E_{\mathrm{rep}}
\]

Nuclear forces are the **closed-form** gradient at the SCC density
(Hellmann–Feynman + Pulay energy-weighted density + Hamiltonian shift on
\(\partial S\) + fixed-\(\Delta q\) \(\gamma\) geometry terms + repulsive
spline derivative) — **not** a re-SCC finite difference.
This is the DFTB+ force module boundary applied to production 3ob tables.

## Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB2`
- SK / Hubbard / repulsive: static `basis_data/dftb_3ob_tables.h` (DFTB.org
  3ob-3-1 H/C/N/O; Gaus et al. 2013). Same tables as `dftb2-3ob-organic`.

## Geometry (Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

Isolated molecule, cubic box 20 Å. `scc_tol = 1e-10`, `max_scc = 80`.

## Pins (hard-coded gates)

1. **Energy:** finite, \(|E| > 10^{-3}\) eV.
2. **Forces:** all components finite; \(|F|^2 > 10^{-8}\) at off-eq geometry.
3. **Translation:** \(|\sum_i F_i| \le 10^{-6}\) eV/Å per Cartesian.
4. **FD identity:** analytic \(F\) vs central **re-SCC** FD of the same \(E\)
   (\(h=10^{-4}\) Å) on \(F_{O,x}\) and \(F_{H1,y}\) within **1×10⁻⁵ eV/Å**.

Offline pin note:
`tools/tb_oracle/inputs/dftbplus/dftb2_analytic_forces_threeob_note.md`.

Measured after the gradient fix (2026-07-28):

```text
E = -110.03663082 eV                      (unchanged — see below)
F_O_x  analytic 2.36539434  re-SCC FD 2.36539390  |d| 4.374e-07 eV/A
F_H1_y analytic 0.03167127  re-SCC FD 0.03167127  |d| 1.968e-09 eV/A
```

## Closed — the polar-gradient defect (fixed 2026-07-28)

The gate used to read **1.0 eV/Å** and pass at 0.84 / 0.24 eV/Å — an **8×
relative** disagreement on \(F_{H1,y}\) (0.267 vs 0.032). Cause: the SCC pair
gradient

```text
dE/dbond |_gamma  =  -dq_A dq_B  d(gamma_AB)/dr        // WRONG SIGN
```

carried a minus in all three assemblies (`analytic_forces_restricted`,
`analytic_forces_unrestricted`, `analytic_forces_3ob`). With
\(E^{(2)} = \tfrac12\sum_{AB}\gamma_{AB}\Delta q_A\Delta q_B\) and
bond \(= R_B - R_A\), the ordered double sum contributes
\(\gamma_{AB}\Delta q_A\Delta q_B\) to the A–B pair and
\(\partial r/\partial\) bond \(= +\hat r\), so the correct term is
\(+\Delta q_A\Delta q_B\,d\gamma/dr\). Three copies of one wrong formula,
which is why every \(\Delta q = 0\) case (O₂ 3.0e-07 eV/Å) was exact while
every polar case was off.

An earlier A/B that flipped the sign measured 2.268 eV/Å and concluded the
shipped sign was right; that A/B was run on top of two other defects (a
non-convergent Picard SCC and a double-counting route that carried a spurious
\(\sum_A V_A q^0_A\)) and was therefore not measuring the gradient at all.

| case | Δq | analytic vs re-SCC FD (before → after) |
| ---- | -- | -------------------------------------- |
| O₂ triplet (`reg_dftb2-spin-analytic-forces`) | 0 | 3.0e-07 → 3.0e-07 eV/Å |
| H₂O MioSketch (`reg_dftb2-analytic-forces`) | ≠ 0 | 0.23 / 0.75 → 2.5e-07 / 4.9e-09 eV/Å |
| H₂O 3ob (this case) | ≠ 0 | **0.84 / 0.24 → 4.4e-07 / 2.0e-09** eV/Å |

**The 3ob energy did not move** (−110.03663082 eV before and after), and
neither did the DFTB+ 24.1 pin of `reg_dftb2-3ob-organic` (CH₄,
−87.84300546 eV vs pin −87.86856831 eV). That is the control for the
double-counting change: the 3ob path already assembled
\(E = \mathrm{Tr}(\rho H^0) + E^{(2)} + E_{\mathrm{rep}}\), so only the
MioSketch path's energies moved. Its SCC also already converged
(same value with and without mixing).

**Still owed:** ThreeOb has an analytic SK derivative path
(`fill_sk_block_deriv_3ob`, `rep_3ob_dr`), so its `with_stress` fallback to
multi-SCC affine strain-FD is no longer necessary — a closed-form 3ob pair
virial is derivable and not implemented. An open-shell \(\Delta q \ne 0\)
gate is also owed (O₂ is homonuclear, so it cannot see this class of bug).

## Perf

`perf_exempt: true` — no DFTB+ absolute-force twin was run for this
geometry (FD identity only). Analytic path replaces \(6N\) full re-SCC
force evaluations with one post-SCC contraction.
