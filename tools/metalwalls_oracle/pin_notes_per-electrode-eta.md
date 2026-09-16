# Pin notes — per-electrode-eta

**Date:** 2026-07-29 (re-pinned against the real MetalWalls binary;
supersedes the 2026-07-27 literature-only note)
**Slug:** `per-electrode-eta`
**oracle:** metalwalls
**oracle_version:** MetalWalls git `c8ad175d` (release branch, 2026-06-18),
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
**verdict:** **confirmed** — pinned numbers unchanged, tolerance unchanged.

The closed form the pin was originally taken from is unchanged and its
citations still stand (Reed–Lanning–Madden JCP **126**, 084704 (2007);
Scalfi et al. JCP **153**, 174704 (2020); MetalWalls JOSS **5**(53), 2373
(2020)); it is now *verified* rather than *assumed*.

## What was checked

MetalWalls' own sources implement exactly the elastance ATV implements:

| Element | MW source | ATV source |
| --- | --- | --- |
| \(\eta_{ij}=\eta_i\eta_j/\sqrt{\eta_i^2+\eta_j^2}\) | `src/coulomb_sr_2DPBC.inc:56-58` | `const_pot_bo.cpp:294-296` |
| \(A_{ii}=\sqrt{2/\pi}\,\eta_i\) (a.u.) via `selffactor = (2α−√2η)/√π` | `src/coulomb_self.f90:63` | `const_pot_bo.cpp:322-323` |
| RHS \(b_i=\Psi_i-V_i^{\rm elyt}\) | `src/electrode_charge.F90:91` | `const_pot_bo.cpp` `build_rhs_` |
| per-electrode \(\eta\) is per **species** | `src/species.f90:317` `charge gaussian η q` | `add_electrode(type, psi, eta)` |

MW *does* express per-electrode η; `mw_expressible = true`.

## Twin construction and its two corrections

MW is unconditionally periodic (Ewald); ATV case 1 (`xy_image_range = 0`) is
the isolated pair. Two effects had to be removed before comparing:

1. **MW `num_pbc 2` is z-periodic at spacing \(L_z\).** The 2D \(k_z\)
   integral is a Riemann sum of step \(2\pi/L_z\) (`src/ewald.f90:203-217`).
   Measured: \(L_z\) 40 → 160 Å at \(L_{xy}=40\) Å moves \(Q\) by −9.2e-5
   relative (≈8 % of the whole image term). Production runs use
   \(L_z = 20\,L_{xy}\), which leaves ≈1e-8 relative.
2. **The xy image lattice**, removed analytically. For the electroneutral
   two-site solve \(S = A_{11}+A_{22}-2A_{12} = \Delta\Psi/Q\) is
   background-free and
   \(\Delta S(L)=2k\sum_{\mathbf n\neq0}[R_{\mathbf n}^{-1}-(R_{\mathbf n}^2+d^2)^{-1/2}]\)
   converges absolutely (all erf factors are 1 to <1e-300 here).

Evidence the model is right, not fitted: \(S_{\rm MW}(L)-\Delta S(L)\) is flat
to **3e-11 relative** across \(L_{xy}=40\ldots320\) Å while \(\Delta S\) itself
falls by 510×. Tightening `coulomb_rtol`/`coulomb_ktol` from 1e-15 to 1e-20
changes \(Q\) by <1e-12 relative, so MW's own Ewald is converged.

## Numbers

MW `charges.out` is in atomic units, whose charge unit **is** the elementary
charge (file header: `1 au = 1.602176565e-19 C = 1 e`), so \(Q\) needs no
conversion; the inputs do:
η × 0.529177210903 Å/bohr, d ÷ 0.529177210903, Ψ ÷ 27.211386245988 V per Ha/e.

| \((\eta_1,\eta_2)\) [Å⁻¹] | MW raw \(Q_{\rm top}\), \(L=320\) Å | MW image-corrected | closed form, exact \(k\) | old/kept pin (ATV \(k\)) | MW vs pin |
| --- | --- | --- | --- | --- | --- |
| (1.979, 0.955) | 0.037722009201931 | 0.037722099576710 | 0.037722099576397 | **0.037722100829787** | −3.322e-08 |
| (1.979, 1.979) | 0.026126911632594 | 0.026126954987021 | 0.026126954986871 | 0.026126955854989 | −3.322e-08 |
| (0.955, 0.955) | 0.067816480237705 | 0.067816772335944 | 0.067816772334931 | 0.067816774588275 | −3.321e-08 |

The residual is **not** physics. \(Q\propto1/k\), and

\[
\frac{k_{\rm exact}-k_{\rm ATV}}{k_{\rm exact}}
= \frac{14.399645478456184-14.399645}{14.399645478456184} = 3.3227\times10^{-8},
\]

i.e. the 8-digit truncation of `units::coulomb`
(`include/atomiverse/core/units.h:140`) against
27.211386245988 × 0.529177210903. Using the identical constant, MW and the
pin agree to **6e-12 … 1.5e-11** — under the `matrix_inversion 1.0e-14` solver
noise. Pin kept: it is what ATV's own \(k\) produces, and the 1e-9 regression
tolerance is a code-vs-closed-form check, not an oracle tolerance.

Note (no action taken, no debt hidden): `units::coulomb` is the only constant
in `core/units.h` carried to 8 digits — `kEvInHartree` and the Å↔bohr factor
carry 12–17. Tightening it would shift **every** electrostatic regression pin
in the tree by ~3e-8 and is out of scope for a re-pin; recorded here so the
next campaign can decide deliberately rather than rediscover it.

## Case 2 — and the MW trap that nearly produced a false discrepancy

Case 2 is 2×2 walls in a 4×4 Å cell at a 5 Å gap. The naive MW twin (that
same 4 Å cell) gives \(Q_{\rm top}=1.9633237859942\times10^{-2}\) e and looked
like a **0.87 % ATV error**. It is not: MW applies the minimum-image
convention, so `coulomb_rcut ≤ L/2 = 2` Å, while the electrode–electrode
short-range term carries \(\mathrm{erfc}(\eta_{ij}r)\) with
\(\eta_{ij}=0.860\) Å⁻¹ = 0.455 bohr⁻¹ — still **1.5e-2** at \(r=2\) Å. MW
does not warn; it runs and prints a clean, tolerance-converged, wrong number.
Replicating into a physically identical \(n\times n\) supercell lets `rcut`
grow and the answer converges:

| supercell | sites/wall | `rcut` [Å] | \(Q_{\rm top}\) per 4×4 Å cell [e] |
| --- | --- | --- | --- |
| 1×1 | 4 | 2.00 | 1.963323785994192e-02 ← cutoff-limited |
| 2×2 | 16 | 3.99 | 1.980427008706078e-02 |
| 3×3 | 36 | 5.99 | 1.980450775193917e-02 |
| 4×4 | 64 | 7.98 | 1.980450776415409e-02 |
| 6×6 | 144 | 11.98 | 1.980450776415245e-02 |

**Rule for this oracle: check `erfc(eta_ij * rcut)`, not just
`erfc(alpha * rcut)`, before trusting any MW electrode run.** `coulomb_rtol`
only controls the former.

ATV's side converges too, but slowly. `ConstPotBO`'s truncated real-space
image sum is absolutely convergent — square and circular truncation Richardson-
extrapolate to the same limit (1.98045081e-02 / 1.98045065e-02 e) — and that
limit matches the converged MW value to **+1.7e-8 / −6.5e-8**, i.e. within the
extrapolation's own uncertainty and of the same order as the `units::coulomb`
truncation. So ATV's **periodic-image path is verified as well**, not only the
isolated case-1 path.

The regression still does not hard-code the case-2 charge: it runs
`xy_image_range = 100`, and that sum approaches the 2D-Ewald limit only as
\(O(1/R)\) (measured: +0.631 % at R=100, +0.315 % at 200, +0.0788 % at 800,
+0.0197 % at 3200, fitting \(0 + 63.7\%/R\) to four digits). Pinning \(Q\) at
R=100 would pin a numerical knob's truncation error, not physics. The case-2
assertions (neutrality, sign, shared-η bracket) are truncation-independent and
MW reproduces the bracket from the shared-η table.

Consequence for the code (fixed in this branch): the doc comment on
`ConstPotBO::xy_image_range` claimed the sum "matches LAMMPS ELECTRODE
madelung / plate_cap when large (e.g. 200+)". At R = 200 the residual for this
geometry is 1.2 %; the comment now states the measured \(O(1/R)\) law instead.

## Provenance correction

The 2026-07-27 note called \(\eta_2 = 0.955\) Å⁻¹ "the Merlet/Salanne graphite
width used in MW examples". **Wrong** — MW input is atomic units and MW's
graphite width is `charge gaussian 0.955234657` **bohr⁻¹**
(`example/nanoporous-graphene/runtime.inpt:55`, whose own comment reads
`# gaussian_width (bohr^-1) charge`) = **1.805131886 Å⁻¹**. The numeral was
lifted from an atomic-unit input and relabelled Å⁻¹. The pin is unaffected —
0.955 Å⁻¹ is simply a dissimilar second width, and MW was run at exactly that
value — but the citation was never evidence for it. \(\eta_1=1.979\) Å⁻¹ *is*
genuine: MW `tests/charge_neutrality` uses `charge gaussian 1.04724169991593`
bohr⁻¹ = 1.979 Å⁻¹.

## Inputs

`tools/metalwalls_oracle/inputs/metalwalls/per-electrode-eta/`:
`case1_L320/{runtime,data}.inpt`, `case2_super3x3/{runtime,data}.inpt`, generators
`gen_case1.py` (box sweep / η variants) and `gen_case2.py`, comparison driver
`analyze.py`. Run with
`mpirun -n 1 /nobackup/.../opt/metalwalls/mw` after
`module load buildenv-gcc/2026.03-mpich` in a **login shell**.

MW gotcha worth keeping: an electrode-only cell has zero mobile DOF, so MW's
per-step temperature is NaN and it aborts in `print_step`
(`src/system.F90:1382`). Both twins carry one neutral, non-interacting
spectator atom at rest purely to make \(T = 0\) K well defined; it contributes
nothing to the electrostatics.

## perf

`perf_exempt = false`. Matched twin = case 2 (identical 4×4×24 Å cell, same
2×2 walls, same gap, same η, one BO solve), MW `mpirun -n 1` vs ATV single
rank, `OMP_NUM_THREADS=1`:

- MetalWalls: 0.32–0.76 s (median 0.44), 1 BO solve.
- `reg_per-electrode-eta`: 0.04 s (7/7 runs), **5** BO solves.

\(\text{ratio}_{\rm ATV/MW} \approx 0.09\), conservative (ATV charged for 5
solves vs MW's 1). Not a kernel benchmark: different algorithms (real-space
image lattice vs Ewald) and both dominated by process start-up at this size.
