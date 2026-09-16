# Reference — per-electrode Gaussian width η_e (ConstPotBO multi-wall)

oracle: metalwalls
oracle_version: MetalWalls git `c8ad175d` (release branch, 2026-06-18), built
with GCC 14.3 + MPICH + OpenBLAS at
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
(upstream self-check `tests/maze/electrodes/LiCl-Al` reproduces its shipped
`energies.ref` Hamiltonian to 1.2e-12 relative)

**Re-pinned 2026-07-29 against the real MetalWalls binary.** Waves 1–5 pinned
this slug from the *literature closed form* because no MW binary existed on
the host. The binary now exists; the pinned numbers were re-derived from MW
directly and are **confirmed** (see the discrepancy budget below). The old
literature citations remain valid as the origin of the closed form:
Reed–Lanning–Madden JCP 126, 084704 (2007) DOI 10.1063/1.2464084;
Scalfi et al. JCP 153, 174704 (2020) DOI 10.1063/5.0028232;
MetalWalls JOSS 5(53), 2373 (2020) DOI 10.21105/joss.02373.

MW twin inputs: `tools/metalwalls_oracle/inputs/metalwalls/per-electrode-eta/`
(`case1_L320/`, `case2_super3x3/`, generators `gen_case1.py` / `gen_case2.py`,
comparison driver `analyze.py`). Full derivation:
`tools/metalwalls_oracle/pin_notes_per-electrode-eta.md`.
**ctest never launches MetalWalls** — the numbers below are hard-coded.

## Physics pinned

MetalWalls-style per-electrode Gaussian charge widths on the
constant-potential BO solve (public surface:
`ConstPotBO::add_electrode(type, Psi, eta)`), with the published elastance
elements (units: eV, Å, e; \(k = 14.399645\) eV·Å/e² = `units::coulomb`):

- \(A_{ii} = k\sqrt{2/\pi}\,\eta_i\)
- \(A_{ij} = k\,\mathrm{erf}(\eta_{ij} r_{ij})/r_{ij}\),
  \(\eta_{ij}=\eta_i\eta_j/\sqrt{\eta_i^2+\eta_j^2}\)

MetalWalls uses **exactly** this convention. Verified in the MW sources:
`src/coulomb_sr_2DPBC.inc:56-58` builds
`etaij = (etai*etaj)/sqrt(etai*etai + etaj*etaj)` from the per-electrode
`electrodes(itype)%eta`, and `src/coulomb_self.f90:63` subtracts
`selffactor = (2α − √2·η_i)/√π` from the Ewald diagonal, whose net effect is
\(A_{ii} = \sqrt{2/\pi}\,\eta_i\) in atomic units. `src/electrode_charge.F90:91`
forms the same RHS `b_i = Ψ_i − V_i^{elyt}` that `ConstPotBO::build_rhs_` does.

## Units: MW (atomic) → ATV (eV, Å, e, V)

| Quantity | ATV value | MW input value (a.u.) | Conversion |
| --- | --- | --- | --- |
| \(\eta_{\mathrm{bot}}\) | 1.979 Å⁻¹ | 1.047241700377037 bohr⁻¹ | ×0.529177210903 Å/bohr |
| \(\eta_{\mathrm{top}}\) | 0.955 Å⁻¹ | 0.505364236412365 bohr⁻¹ | ×0.529177210903 |
| \(d\) (case 1) | 4 Å | 7.558904498503081 bohr | ÷0.529177210903 |
| \(\Delta\Psi\) | 1 V | 3.674932217565499e-02 Ha/e | ÷27.211386245988 V per Ha/e |
| \(Q\) | e | e | MW `charges.out` header: 1 a.u. = 1 e — **no conversion** |
| \(k=e^2/4\pi\varepsilon_0\) | 14.399645 eV·Å/e² | 1 Ha·bohr/e² | 27.211386245988 × 0.529177210903 = **14.399645478456184** |

Dimensional cross-check (not a number match): MW returns a charge, whose atomic
unit *is* the elementary charge, so the only conversion that can hide an error
is the elastance scale \(k\). \(Q=\Delta\Psi/S\) with \(S\propto k\), so
\(Q\propto \Delta\Psi/k\): converting ΔΨ from V to Ha/e (÷27.2114) and \(k\)
from Ha·bohr to eV·Å (×27.2114×0.52918) leaves \(Q\) invariant up to
0.529177210903, i.e. the Bohr radius must appear exactly once — it does.

## Case 1 — two-site vacuum capacitor, dissimilar η

\(\eta_1 = 1.979\) Å⁻¹ (LAMMPS ELECTRODE default, Ahrens-Iwers et al. JCP 157,
084801 (2022); identical to MW `tests/charge_neutrality` `charge gaussian
1.04724169991593`), \(\eta_2 = 0.955\) Å⁻¹ (a deliberately dissimilar second
width — see the provenance correction below), \(d = 4\) Å,
\(\Delta\Psi = 1\) V, electroneutral:

\[
Q = \frac{\Delta\Psi}{A_{11}+A_{22}-2A_{12}} = \mathbf{0.037722100829787}\ e
\]

(\(A_{11}=22.737234510126967\), \(A_{22}=10.972237977347778\),
\(A_{12}=3.599907138217925\), \(\eta_{12}=0.860091344088193\) Å⁻¹).

### Why the MW twin needs an extrapolation

ATV case 1 is the **image-free** two-site capacitor: `ConstPotBO` with the
default `xy_image_range = 0` sums a single minimum image, i.e. exactly one pair
at \(r=4\) Å. MetalWalls is an Ewald code and is **unconditionally periodic** —
there is no aperiodic mode. The MW twin therefore carries the 2D image lattice,
and the pin is its \(L\to\infty\) limit. For the electroneutral two-site solve
the elastance combination \(S = A_{11}+A_{22}-2A_{12} = \Delta\Psi/Q\) is
background-free, and its periodic value exceeds the isolated one by an
*absolutely convergent* dipole-lattice sum (every erf factor is 1 to <1e-300 at
these separations):

\[
\Delta S(L) = 2k\!\!\sum_{\mathbf n\neq 0}\!\Big[\frac{1}{R_{\mathbf n}}
 - \frac{1}{\sqrt{R_{\mathbf n}^2+d^2}}\Big],\qquad R_{\mathbf n}=L|\mathbf n| .
\]

A second MW artefact must be removed first: `num_pbc 2` approximates the
\(k_z\) integral by a Riemann sum of step \(2\pi/L_z\)
(`src/ewald.f90:203-217`), which is numerically identical to keeping
**z-periodic images at spacing \(L_z\)**. At \(L_z=L_{xy}\) those contribute
≈8 % of the total image term (measured: \(L_z\) 40 Å → 160 Å moves \(Q\) by
−9.2e-5 relative). All production runs below use \(L_z = 20\,L_{xy}\).

### Raw MetalWalls output — `charges.out` (atomic units = e)

`mpirun -n 1 mw` on `inputs/metalwalls/per-electrode-eta/case1_L320` and its
box-size siblings (\(L_z = 20L_{xy}\)). MW prints \(Q_{\mathrm{bot}}\) first;
\(Q_{\mathrm{top}} = -Q_{\mathrm{bot}}\) exactly.

| \(L_{xy}\) [Å] | MW \(Q_{\mathrm{top}}\) [e] (raw) | \(S_{\rm MW}=\Delta\Psi/Q\) | \(\Delta S(L)\) | \(S_{\rm MW}-\Delta S\) | \(Q_{\rm iso}\) [e] |
| --- | --- | --- | --- | --- | --- |
| 40 | 0.037676075064003 | 26.542042882684 | 3.2384e-02 | 26.509658979107 | 0.037722099736859 |
| 60 | 0.037708419130699 | 26.519276677549 | 9.6176e-03 | 26.509659058459 | 0.037722099623945 |
| 80 | 0.037716322200046 | 26.513719834507 | 4.0608e-03 | 26.509659077777 | 0.037722099596457 |
| 120 | 0.037720386573408 | 26.510862979994 | 1.2039e-03 | 26.509659087696 | 0.037722099582342 |
| 160 | 0.037721376735941 | 26.510167086431 | 5.0800e-04 | 26.509659090111 | 0.037722099578905 |
| 240 | 0.037721885367143 | 26.509809630858 | 1.5054e-04 | 26.509659091352 | 0.037722099577140 |
| **320** | **0.037722009201931** | 26.509722603769 | 6.3512e-05 | 26.509659091653 | **0.037722099576710** |

\(S_{\rm MW}-\Delta S\) is **flat to 3e-11 relative over a factor-8 box sweep**
while \(\Delta S\) itself shrinks by 510×; that flatness is the evidence that
the image model is right and that MW's isolated-limit elastance is
26.5096590917 eV/e².

### Verdict: confirmed — residual is the truncated Coulomb constant

| Quantity | Value |
| --- | --- |
| Pinned \(Q\) (ATV, \(k=14.399645\)) | 0.037722100829787 e |
| MW \(Q_{\rm iso}\) (\(L=320\) Å, image-corrected) | 0.037722099576710 e |
| Closed form with **exact** \(k=14.399645478456184\) | 0.037722099576397 e |
| MW vs exact-\(k\) closed form | **+8.3e-12** |
| MW vs pin | **−3.322e-08** |
| \((k_{\rm exact}-k_{\rm ATV})/k_{\rm exact}\) | **3.3227e-08** |

\(Q\propto 1/k\), so the −3.322e-08 offset is *entirely* the 8-digit truncation
of `units::coulomb = 14.399645` (`include/atomiverse/core/units.h:140`) against
27.211386245988 × 0.529177210903 = 14.399645478456184 eV·Å/e². Cancel that one
constant and MetalWalls reproduces the pinned physics to **8e-12** — below the
`matrix_inversion 1.0e-14` solver tolerance's own noise floor. The pin is kept
unchanged: it is what ATV's own \(k\) produces, and the regression's 1e-9
tolerance is a code-vs-its-own-closed-form check, not an oracle tolerance.

The same constant explains the offset for both shared-η twins, so the
discrepancy budget is closed with no free parameter (MW at \(L=320\) Å,
\(L_z=6400\) Å, image-corrected):

| \((\eta_1,\eta_2)\) [Å⁻¹] | MW \(Q_{\rm iso}\) [e] | closed form, exact \(k\) | MW vs exact \(k\) | closed form, ATV \(k\) | MW vs ATV \(k\) |
| --- | --- | --- | --- | --- | --- |
| (1.979, 0.955) | 0.037722099576710 | 0.037722099576397 | +8.3e-12 | 0.037722100829787 | −3.322e-08 |
| (1.979, 1.979) | 0.026126954987021 | 0.026126954986871 | +5.8e-12 | 0.026126955854989 | −3.322e-08 |
| (0.955, 0.955) | 0.067816772335944 | 0.067816772334931 | +1.5e-11 | 0.067816774588275 | −3.321e-08 |

The two shared-η rows pin \(A_{11}(\eta)\) and \(A_{22}(\eta)\) **separately**
(there \(A_{11}=A_{22}\)), so the mixed row is over-determined: MW confirms the
per-electrode \(\eta_{ij}\) mixing rule, not just one lumped combination.

| Check | Tol |
| --- | --- |
| \(Q_{\mathrm{top}}\) vs 0.037722100829787 e | rel ≤ 1e-9 |
| \(\lvert\sum Q\rvert\) | ≤ 1e-12 e |

### Provenance correction (unit-convention slip in the pre-2026-07-29 note)

The superseded REFERENCE.md called \(\eta_2 = 0.955\) Å⁻¹ "the Merlet/Salanne
graphite width used in MW examples". That is **wrong**: MW inputs are atomic
units, and MW's graphite width is `charge gaussian 0.955234657` **bohr⁻¹**
(MW's own comment in `example/nanoporous-graphene/runtime.inpt:55` reads
`# gaussian_width (bohr^-1) charge`), i.e. **1.805131886 Å⁻¹** — the numeral
0.955 was lifted from an atomic-unit input and relabelled Å⁻¹. The pinned
*number* is unaffected (the case is defined by whichever two widths are chosen,
and \(\eta_2=0.955\) Å⁻¹ is a legitimately dissimilar second width, now
verified against MW at exactly that value), but the citation was not evidence
for it. \(\eta_1 = 1.979\) Å⁻¹ *is* genuine: MW's own
`tests/charge_neutrality` uses `charge gaussian 1.04724169991593` bohr⁻¹
= 1.979 Å⁻¹.

## Case 2 — multi-atom two-wall, mixed η, 2D xy Gaussian images

2×2-atom square walls (a = 2 Å, \(L_x=L_y=4\) Å), wall separation 5 Å,
\(\eta_{\mathrm{bot}}=1.979\), \(\eta_{\mathrm{top}}=0.955\) Å⁻¹,
\(\Delta\Psi=1\) V, `xy_image_range = 100`.

MW twin `inputs/metalwalls/per-electrode-eta/case2_super3x3` (same 2 Å site lattice,
same 5 Å gap, `num_pbc 2`, `matrix_inversion 1.0e-14`). Two MW-side pitfalls,
both real, both caught here:

1. **The primitive 4 Å cell is unusable in MW.** MW applies the minimum-image
   convention, so `coulomb_rcut ≤ L/2 = 2` Å, while the electrode–electrode
   short-range term carries \(\mathrm{erfc}(\eta_{ij}r)\) with
   \(\eta_{ij}=0.860\) Å⁻¹ — still 1.5e-2 at \(r=2\) Å. That truncation is
   *silent*: MW runs and prints a clean, converged, wrong answer
   (\(Q_{\rm top}=1.9633237859942\times10^{-2}\) e, 0.87 % low). The fix is a
   physically identical \(n\times n\) **supercell**, which lets `rcut` grow.
2. z-images are harmless here (unlike case 1): dense walls act as uniform
   dipole layers, so `charges.out` is invariant to 1.5e-12 over
   \(L_z = 24\ldots960\) Å.

| MW supercell | sites/wall | `rcut` [Å] | \(Q_{\mathrm{top}}\) per 4×4 Å cell [e] |
| --- | --- | --- | --- |
| 1×1 (4 Å) | 4 | 2.00 | 1.963323785994192e-02 ← **cutoff-limited, do not use** |
| 2×2 (8 Å) | 16 | 3.99 | 1.980427008706078e-02 |
| 3×3 (12 Å) | 36 | 5.99 | **1.980450775193917e-02** |
| 4×4 (16 Å) | 64 | 7.98 | 1.980450776415409e-02 |
| 6×6 (24 Å) | 144 | 11.98 | 1.980450776415245e-02 |

Converged MW value: \(Q_{\mathrm{top}} = 1.980450776415\times10^{-2}\) e,
uniform over sites to 5e-15, exactly electroneutral.

**ATV agrees with it.** `ConstPotBO`'s truncated real-space image sum is
absolutely convergent (square and circular truncation Richardson-extrapolate
to the same limit, 1.98045081e-02 / 1.98045065e-02 e), and that limit matches
MW to **+1.7e-8 / −6.5e-8** — i.e. to the extrapolation's own uncertainty and
the same order as the `units::coulomb` truncation. The periodic-image path is
therefore verified, not merely the isolated one.

The case-2 charge is nevertheless **not** hard-coded, because the regression
runs `xy_image_range = 100` and that sum converges only as \(O(1/R)\):

| `xy_image_range` | \(Q_{\mathrm{top}}\) [e] | vs converged MW |
| --- | --- | --- |
| 100 (what the regression runs) | 1.992946777240788e-02 | +0.631 % |
| 200 | 1.986694868399804e-02 | +0.315 % |
| 800 | 1.982011096692778e-02 | +0.0788 % |
| 3200 | 1.980840858393783e-02 | +0.0197 % |
| →∞ (Richardson) | 1.98045081e-02 | +1.7e-8 |

Pinning \(Q\) at \(R=100\) would pin the truncation error of a numerical knob,
not the physics, so the asserted invariants below stay truncation-independent.
MW independently reproduces the ordering this case asserts —
\(Q(\eta{=}1.979\ \text{shared}) < Q(\text{mixed}) < Q(\eta{=}0.955\
\text{shared})\) — from the shared-η table above.

| Check | Tol |
| --- | --- |
| \(\lvert\sum Q\rvert\) | ≤ 1e-10 e |
| \(Q_{\mathrm{top}} > 0\) (with \(\Psi_{\mathrm{top}}>\Psi_{\mathrm{bot}}\)) | sign |
| \(Q^{\mathrm{shared}\ 1.979} < Q^{\mathrm{mixed}} < Q^{\mathrm{shared}\ 0.955}\) | strict |
| 3-arg `add_electrode` at equal η ≡ 2-arg shared path | ≤ 1e-14 e/site |

## Wall clock (matched twin)

Same box (4×4×24 Å), same 2×2 walls, same η, same ΔΨ, one BO solve, single
rank, `OMP_NUM_THREADS=1`, shared 48-core node:

| | wall clock | work done |
| --- | --- | --- |
| MetalWalls `mpirun -n 1` | 0.32–0.76 s (median 0.44) | 1 BO solve + 2D Ewald setup + file I/O |
| `reg_per-electrode-eta`, single rank | 0.04 s (7/7 runs) | **5** BO solves (case 1 + 4× case 2 at R = 100) |

\(\text{ratio}_{\rm ATV/MW} \approx 0.09\) — and that is conservative: ATV is
charged for five solves against MetalWalls' one. The two are not doing the
same arithmetic (real-space image lattice vs Ewald), and at this size both are
dominated by process start-up, so read the ratio as "same order, ATV not
slower", not as a kernel benchmark.

## Reproducing

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cd tools/metalwalls_oracle/inputs/metalwalls/per-electrode-eta/case1_L320
  mpirun -n 1 /nobackup/.../work/opt/metalwalls/mw > run.stdout 2>&1
  sed -n 5,6p charges.out
'
```
