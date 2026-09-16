# Reference — AM1 water absolute energy / ΔHf vs OpenMOPAC

```text
oracle: mopac
oracle_version: MOPAC 7.00 (public domain), archived transcript
```

First **absolute OpenMOPAC pin** on the Atomiverse NDDO surface. Everything
that shipped before this gated AM1/RM1 against *itself* (analytic force vs its
own re-SCF finite difference, ΔHf inside a ±8 kcal/mol window, RM1-vs-AM1
discrimination) — nothing would have caught a systematic parameter or integral
defect that shifted the absolute energy.

## Method

AM1 (Austin Model 1), NDDO/MNDO functional form with Gaussian core-core
corrections — Dewar, Zoebisch, Healy & Stewart, *J. Am. Chem. Soc.* **107**,
3902–3909 (1985). Closed-shell RHF, `cpu::semiempirical::AM1`.

Oracle family lock: NDDO → **OpenMOPAC only**. DFTB+ is a forbidden pin source
for this row (`docs/developer/tb-method-loop.md`).

## Offline oracle run

`mopac` is **not installed** on the Atomiverse build hosts, and no regression
may shell out to an oracle binary. The pin is therefore a *published* MOPAC
transcript, transcribed verbatim to
`tools/tb_oracle/inputs/mopac/am1-water/mopac_am1_water.out.txt` with the
reconstructed deck beside it.

Source: MOPAC 7.00 (public domain) run of `AM1 PRECISE EF T=48H` on water,
published at <https://www.cup.uni-muenchen.de/ch/compchem/energy/MOPAC_output.html>
(H. Zipse, LMU München computational-chemistry course archive, page dated
2004-11-29). The transcript prints its own parameter provenance —
`H: (AM1): M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 107 3902-3909 (1985)`, ditto
O — so the numbers and the citation are the same artifact.

## Geometry — and why *this* geometry

**A published AM1 ΔHf belongs to the AM1-optimized structure.** It cannot be
pinned at an arbitrary geometry, and the SE-only build has
`ATV_BUILD_MINIMIZER=OFF`, so the case cannot re-optimize either. The
convention chosen here is: **pin at the optimized internal coordinates printed
by the same run that printed the energy** — geometry and energy from one
self-consistent transcript, no cross-source splicing.

C2v, O at origin, H1 on +x, H2 in the xy plane (MOPAC's own frame):

| Structure | r(O–H) [Å] | ∠HOH [°] | r(H–H) [Å] | Role |
| --------- | ---------- | -------- | ---------- | ---- |
| **AM1 optimum** | 0.961267 | 103.53103 | 1.510121 | main pin |
| input | 1.000000 | 105.00000 | 1.586707 | off-stationary pin |

Cartesian coordinates of the optimum, as the case constructs them:

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.0000000 | 0.0000000 | 0 |
| H | 0.9612670 | 0.0000000 | 0 |
| H | −0.2249095 | 0.9345854 | 0 |

(MOPAC printed −0.2249 / 0.9346 to 4 decimals — same structure.)
Cubic 30 Å box, isolation only. Charge 0, singlet.

Two consequences make the optimized-geometry convention a **sharper** gate, not
a weaker one:

1. **Stationarity is itself testable.** MOPAC stopped at
   `RMS GRADIENT = 0.00141` kcal/mol/Å ≈ 6×10⁻⁵ eV/Å. If ATV's AM1 is the same
   functional, that structure must be a minimum of *ATV's* energy too. The case
   asserts it (max|F| ≤ 5×10⁻³ eV/Å) — a gradient-level agreement check
   independent of the energy pin, which none of the existing AM1 regressions
   make. Measured: **8.5×10⁻⁴ eV/Å**, against 1.65 eV/Å at the un-optimized
   input geometry (a 330× separation, asserted so the gate cannot go vacuous).
2. **Coordinate truncation cannot masquerade as an error.** At a minimum the
   energy is quadratic, so the ≤5×10⁻⁷ Å / ≤10⁻⁵° rounding of the printed
   coordinates perturbs E at second order. Measured curvature along r is
   ≈ 91 eV/Å²; a parabolic fit puts ATV's own minimum 3.8×10⁻⁵ Å from the pinned
   r, worth 7×10⁻⁸ eV. Six orders of magnitude below the tolerance.

The **input-geometry pin** (MOPAC `CYCLE 1 HEAT`, a non-stationary point)
removes the remaining objection that agreement might depend on the gradient
vanishing. It does not: the residual is the same size at both geometries.

## Pins

| Quantity | Geometry | MOPAC | ATV | Δ | Tol |
| -------- | -------- | ----- | --- | - | --- |
| `TOTAL ENERGY` | optimum | **−348.56254** eV | −348.56316283 | −6.2×10⁻⁴ eV | 2×10⁻³ eV |
| `FINAL HEAT OF FORMATION` | optimum | **−59.24079** kcal/mol | −59.255061 | −1.43×10⁻² | 0.05 kcal/mol |
| `NET ATOMIC CHARGE` O | optimum | **−0.3827** e | −0.38271 | −1×10⁻⁵ e | 2×10⁻⁴ e |
| `NET ATOMIC CHARGE` H | optimum | **+0.1914** e | +0.19135 | −5×10⁻⁵ e | 2×10⁻⁴ e |
| max‖F‖ (stationarity) | optimum | RMS grad 0.00141 kcal/mol/Å | 8.51×10⁻⁴ eV/Å | — | ≤ 5×10⁻³ eV/Å |
| `CYCLE 1 HEAT` | input | **−57.64871** kcal/mol | −57.664384 | −1.57×10⁻² | 0.05 kcal/mol |
| max‖F‖ (non-vacuity) | input | — | 1.65 eV/Å | — | > 0.5 eV/Å |

ATV values measured on the CPU reference host (x86_64 epyc, GCC 14.3,
OpenBLAS), **bit-identical at `OMP_NUM_THREADS=1` and `8`**.

`ΔHf = (E_tot − Σ_A EISOL_A)·23.061 + Σ_A ΔHf_atom(A)` [kcal/mol]
(<https://openmopac.net/Manual/calc_of_hof.html>), so the ΔHf pin gates the SCF
energy **and** the thermochemistry constants; pinning the total energy
separately says which of the two moved. The constants are checked directly by
`tests/cpu/semiempirical/test_am1_thermochemistry.cpp`
(`EISOL(H) = −11.396427` eV, `EISOL(O) = −316.09952` eV,
`ΔHf_atom(H) = 52.102`, `ΔHf_atom(O) = 59.559` kcal/mol) — all equal to MOPAC's
to every printed digit, which is why the ΔHf residual (1.43×10⁻²) is exactly
the energy residual times 23.061 (6.2×10⁻⁴ × 23.061 = 1.43×10⁻²) and nothing
more.

## Tolerance rationale

The residual is **6.2×10⁻⁴ eV** (0.014 kcal/mol), which is 120× the
transcript's 5-decimal print precision. That is not noise, so it was traced
rather than absorbed.

**Cause: the physical-constant convention.** ATV's NDDO integrals use CODATA-2018
(`nddo_hcore.cpp:21-22`, `nddo_two_electron.cpp:18-19`,
`nddo_overlap.cpp:17` — `a₀ = 0.529177210903` Å, 1 Ha = `27.211386245988` eV,
so e²·k = 14.3996455 eV·Å). MOPAC froze `a₀ = 0.529167` Å, 1 Ha = `27.21` eV
and `e²·k = fpc_9 = 14.399` eV·Å — a 4.5×10⁻⁵ relative difference, and one
MOPAC does not even apply self-consistently (0.529167 × 27.21 = 14.398634,
not 14.399).

Measured, not assumed: rebuilding ATV with **MOPAC's frozen constants**
substituted for CODATA moves this energy from −348.56316 to **−348.56243** eV,
i.e. the residual collapses from 6.2×10⁻⁴ to 1.1×10⁻⁴ eV and ΔHf from
−59.25506 to **−59.23816** (residual 2.6×10⁻³ kcal/mol). **82 % of the gap is
the constant convention**; the ~10⁻⁴ eV remainder is oracle-side (MOPAC's own
internally inconsistent constants, its numerically re-derived AD/AQ/AM
multipole additives, and the 0.00141 kcal/mol/Å residual gradient of the
published structure).

ATV keeps CODATA — it is the project-wide convention (`core/units.h`) and is
physically correct; MOPAC's constants are a 1980s freeze. So the tolerance must
admit the offset while staying far below any defect class this gate exists to
catch:

| | Scale | vs `2×10⁻³` eV tolerance |
| --- | ----- | --- |
| Observed residual (constant convention) | 6.2×10⁻⁴ eV | 3.2× headroom |
| Thread-count / BLAS variation | 0 (bit-identical 1 vs 8 threads) | — |
| Transcript print precision | 5×10⁻⁶ eV | 400× tighter than the tolerance |
| **Wrong NDDO parameter set** (RM1 at the same geometry) | 3.005 eV | **1500× the tolerance** |
| **Wrong parameter set**, in ΔHf (RM1: −38.80 kcal/mol) | 20.4 kcal/mol | **410× the 0.05 tolerance** |

RM1 is the sharpest available in-tree probe of the gate: it is the *same*
functional form with reparameterized values, i.e. exactly the defect class a
parameter-transcription error would produce, and it lands 1500 tolerances away.

Charge tolerance `2×10⁻⁴` e is set by the transcript, which prints charges to 4
decimals (half-ULP 5×10⁻⁵, which is precisely the observed H residual). The
stationarity tolerance `5×10⁻³` eV/Å is 6× the measured residual and 330× below
the off-optimum gradient the case also asserts — it cannot pass by accident.

**No tolerance here was widened to make a check pass.** The energy/ΔHf
tolerances were set from the constant-convention offset *after* it was measured
and explained; the charge and stationarity tolerances were set from the
transcript's own precision.

## Perf

MOPAC is not installed on the build host, so a fair wall
T_atv / T_oracle is not measurable → **perf_exempt**. ATV wall for the whole
case (4 SCF solves + 2 force evaluations, 1 thread): ≈ 0.02 s. The archived
MOPAC run reports `COMPUTATION TIME = .000 SECONDS` on 2003 hardware — both are
below the resolution of a meaningful comparison for a 3-atom molecule.

## Gaps

- **`ELECTRONIC ENERGY` / `CORE-CORE REPULSION` are not pinned separately**
  (−493.26959 / +144.70704 eV in the transcript). `AM1Result` exposes only the
  total; the decomposition is reachable only through `AM1::state()`, which
  returns the internal `NDDOScfState`
  (`include/atomiverse/cpu/semiempirical/am1_solver.h:9,60` includes and
  re-exports `internal/nddo_solver.h`). CLAUDE.md lists that internal-type leak
  as scheduled-for-deletion debt, so this case does **not** consume it. Pinning
  the decomposition — which would localize a defect to the one- vs two-electron
  side — needs a public accessor on `AM1Result` first.
- **Koopmans IP not pinned** (12.46417 eV) — `AM1Result` exposes no orbital
  energies. Same fix, same row.
- **Dipole not pinned** (1.860 D total, point-charge 1.094 + hybrid 0.767) —
  no AM1 dipole on the public surface.
- **Constant convention.** ATV is CODATA, MOPAC 7.00 is frozen 1980s constants;
  every NDDO pin against a MOPAC transcript will carry a ~5×10⁻⁵ relative floor.
  Not a defect, but it bounds how tight *any* future AM1/RM1/PM* mopac pin can
  be, and it should be stated once in the roadmap rather than rediscovered per
  row.
- **Modern OpenMOPAC not cross-checked.** The transcript is MOPAC 7.00. A run
  on a current OpenMOPAC (which has its own constant and optimiser updates)
  would tell us whether the 1.1×10⁻⁴ eV constant-aligned remainder is
  ATV-side or a 7.00 artifact. Needs a host with `mopac`.
- **Geometry taken from the transcript, not re-optimized.** `ATV_BUILD_MINIMIZER`
  is OFF in the SE-only build, so the case cannot verify ATV's *own* AM1
  optimum coincides with MOPAC's beyond the stationarity + curvature argument
  above.
