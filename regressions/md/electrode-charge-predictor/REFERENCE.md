# Reference — electrode charge predictor (ASPC)

oracle: metalwalls
oracle_version: MetalWalls release 20.05, git **c8ad175d** (release branch,
2026-06-18), binary at
`/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`
oracle_kind: real binary + source constants (re-pinned 2026-07-29; supersedes the
wave-2 literature-formula pin)

## Physics

Constant-potential electrode charges \(Q(t)\) evolve smoothly as the electrolyte
moves. MetalWalls warm-starts the iterative (CG) charge solve from a short
history of prior \(Q\) using **Kolafa's Always-Stable Predictor-Corrector**
(J. Comput. Chem. **25**, 335 (2004)). LAMMPS ELECTRODE has no ASPC twin, so
MetalWalls is the only oracle for this feature.

## The pin — MetalWalls' predictor coefficients

MetalWalls hard-codes the predictor in `src/electrode_charge.F90:572-577`
(verbatim from the tree the pinned binary was built from):

```fortran
    real(wp), parameter :: B1 =  22.0_wp /  7.0_wp
    real(wp), parameter :: B2 = -55.0_wp / 14.0_wp
    real(wp), parameter :: B3 =  55.0_wp / 21.0_wp
    real(wp), parameter :: B4 = -22.0_wp / 21.0_wp
    real(wp), parameter :: B5 =   5.0_wp / 21.0_wp
    real(wp), parameter :: B6 =  -1.0_wp / 42.0_wp
...
          q_atoms(i, q_now) = B1 * q_atoms(i, q_m1) + B2 * q_atoms(i, q_m2) + ...
```

Six stored charge vectors (`QATOMS_NUM_STEPS = 7`, `src/system.F90:107`);
the predictor switches on from step 7 (`src/system.F90:1430-1435`,
`src/electrode_charge.F90:499`). There is **no user switch** and no other
order — \(K=5\) is MetalWalls' only choice.

Those six constants are exactly the \(K=5\) row of Kolafa's closed form
(Eq. 10):

\[
B_j = (-1)^{j+1}\, j\, \frac{\binom{2K+2}{K+1-j}}{\binom{2K}{K}},
\qquad j = 1 \ldots K+1
\]

| \(j\) | closed form | MetalWalls | decimal |
| --- | --- | --- | --- |
| 1 | \(\binom{12}{5}/\binom{10}{5} = 792/252\) | `22/7` | 3.142857142857143 |
| 2 | \(-2\binom{12}{4}/252 = -990/252\) | `-55/14` | −3.9285714285714284 |
| 3 | \(+3\binom{12}{3}/252 = 660/252\) | `55/21` | 2.619047619047619 |
| 4 | \(-4\binom{12}{2}/252 = -264/252\) | `-22/21` | −1.0476190476190477 |
| 5 | \(+5\binom{12}{1}/252 = 60/252\) | `5/21` | 0.23809523809523808 |
| 6 | \(-6\binom{12}{0}/252 = -6/252\) | `-1/42` | −0.023809523809523808 |

Invariants: \(\sum_j B_j = 1\) (consistency) and \(\sum_j j B_j = 0\)
(exactness through degree 1). \(\sum_j j^2 B_j \ne 0\) — ASPC is **deliberately
not** exact for quadratics; that damping is what buys unconditional stability.

**No unit conversion is involved in this pin.** The \(B_j\) are dimensionless
linear-combination weights on a charge history; the only quantity in the
identity \(Q^{\rm pred} = \sum_j B_j Q_{n+1-j}\) is \(Q\) itself, in \(e\) on
both sides (MetalWalls' atomic-unit charge *is* \(e\), see the `charges.out`
header "1 au = 1.602176565e-19 C = 1 e"). A conversion slip cannot manufacture
agreement here.

## Verdict — atv_bug, found and fixed

| | old pin (waves 1–5) | real MetalWalls c8ad175d |
| --- | --- | --- |
| order | \(K=2\) (3 history vectors) | \(K=5\) (6 history vectors, fixed) |
| coefficients (newest → oldest) | \(\{3, -3, 1\}\) | \(\{22/7, -55/14, 55/21, -22/21, 5/21, -1/42\}\) |
| same family at \(K=2\)? | — | \(\{5/2, -2, 1/2\}\) |
| exactness | degree ≤ 2 | degree ≤ 1 (damped) |

The old pin \(\{3,-3,1\}\) is **Lagrange (Newton-forward) polynomial
extrapolation**, not ASPC at any order. Relative discrepancy against the same
\(K=2\) row of the real family: \(3/2.5 - 1 = +20\%\) on \(B_1\),
\(-3/-2 - 1 = +50\%\) on \(B_2\), \(1/0.5 - 1 = +100\%\) on \(B_3\) — i.e. the
pin was a different algorithm, not a mis-tolerance.

This was an Atomiverse implementation error, not merely a mislabelled pin:
`src/cpu/electrodes/charge_aspc.cpp` computed Lagrange coefficients while its
`correct()` used the genuine Kolafa corrector weights
\(\frac{K+2}{2K+3}, \frac{K+1}{2K+3}\). Those corrector weights are derived
*assuming* the ASPC predictor; pairing them with Lagrange extrapolation is the
one combination that has neither the accuracy of Lagrange nor the
time-reversibility of ASPC — and plain polynomial extrapolation in a
self-consistent loop is precisely the scheme Kolafa 2004 exists to replace.

**Fix (this branch):** `compute_coefficients_()` now evaluates Kolafa's
closed form with exact integer binomials and a single division, so the \(K=5\)
row is bit-identical to MetalWalls' constants (792/252 and 22/7 round to the
same double). No tolerance was touched; no assertion was weakened.

**Known identical defect, NOT fixed here:** the SCF/CPMD twin
`src/cpu/electronic/aspc_predictor.cpp:24-39` (`ASPCPredictor::compute_coefficients`,
and its `ASPCPredictorCplx` sibling) has the same Lagrange-labelled-as-Kolafa
predictor next to a correct Kolafa corrector. It is owned by the in-flight
`esmm-cpmd-*` work and is not touched from this branch; it needs its own fix.

## Geometry

### CG vs dense pin (2 electrode sites)

Madelung-like 4-atom primary cell, `xy_image_range = 0`:

| Site | Type | role | \(z\) [Å] |
| --- | --- | --- | --- |
| 0 | 1 | bot electrode | 8 |
| 1 | 2 | top electrode | 12 |
| 2 | 3 | electrolyte + | 9 |
| 3 | 3 | electrolyte − | 11 |

Box \(L_x=L_y=1\), \(L_z=20\) Å. \(\eta=2\) Å⁻¹, \(\Psi=(0,1)\) V.

### CG iteration comparison (multi-site)

3×3 atoms per wall (18 electrode sites) + 4 electrolyte charges in
\(L_{xy}=6\) Å, \(L_z=20\) Å. Electrolyte \(z\) drifts each launch so
\(b=\Phi_{\mathrm{elec}}\) changes. Primary cell only.

## The MetalWalls twin that was actually run

`tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/`
(`runtime.inpt` + `data.inpt`), run as

```
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cd /tmp/mw_oracle_electrode-charge-predictor
mpirun -n 1 .../opt/metalwalls/mw > run.stdout 2>&1
```

Same 18 electrode sites + 4 ions, same \(\eta\), same box, same \(\Psi\), with
every quantity converted into MetalWalls' atomic units:

| ATV (public units) | factor | MetalWalls (a.u.) |
| --- | --- | --- |
| \(L_{xy} = 6\) Å | ÷ 0.529177210903 Å/Bohr | 11.33835674775462 Bohr |
| \(L_z = 20\) Å | ÷ 0.529177210903 | 37.79452249251541 Bohr |
| \(\eta = 2.0\) Å⁻¹ | × 0.529177210903 | 1.058354421806 Bohr⁻¹ |
| \(\Psi_{\rm top} = 1\) V | ÷ 27.211386245988 V/(Ha/e) | 0.036749322175655 Ha/e |
| \(\Delta t = 10\) fs | × 41.341 a.u./fs | 413.41 a.u. |

Dimensional check on \(\eta\): \(\eta\) is an inverse length, so Å⁻¹ → Bohr⁻¹
**multiplies** by 0.529177 (fewer inverse-Bohr than inverse-Å) — the opposite
sense from the box lengths, as it must be. The kernel convention matches:
MetalWalls' same-\(\eta\) pair interaction is \({\rm erf}(\eta r/\sqrt2)/r\)
(self factor \(\eta/\sqrt{2\pi}\), `src/coulomb_self.f90:63`) and Atomiverse
uses `erf((eta/√2) r)/r` (`src/cpu/dynamics/const_pot_bo.cpp:368`).

Raw MetalWalls output at step 0 (`charges.out`, atomic units = \(e\)):

```
# step        0
-3.716936940768154E-002
-3.862179629716499E-002
...
Q(E1, bottom, Psi=0 V) = -3.566190297414107E-01 e
Q(E2, top,    Psi=1 V) = +3.566190297414106E-01 e
sum                    = -6.9e-18 e
```

`run.out` step 0: `|step| Global charge on electrode 1: -3.56619E-01`.

**These absolute charges are deliberately not pinned by this regression**, and
they must not be compared numerically with the ATV multi-site case: the ATV
regression runs `xy_image_range = 0` — a **non-periodic primary-cell direct
sum** — whereas MetalWalls always applies full 2D-PBC Ewald (`num_pbc 2`), which
it has no option to disable. Those are different electrostatic models, so a
charge mismatch here would say nothing about the predictor. Absolute
const-potential charges are pinned by the sibling slugs
(`const-pot-bo-parallel-plate`, `gaussian-electrode-charge`).

## What Atomiverse asserts

Public API:

- `cpu::electrodes::ChargeASPC` (vector ASPC)
- `cpu::md::ConstPotBO` with `use_cg`, `predictor_order`, `last_iters()`

| Check | Tol / gate |
| --- | --- |
| ASPC \(K=5\) coeffs == MetalWalls `B1..B6` | rel ≤ \(10^{-15}\) each; \(\sum B_j = 1\) to \(10^{-14}\) |
| ASPC \(K=2\) coeffs \(\{1/2,-2,5/2\}\) | exact |
| linear history \(Q_k=k\) → \(Q_{n+1}=n+1\) | ≤ \(10^{-12}\) (degree-1 exactness) |
| quadratic history \(Q_k=k^2\) → 8, **not** the Lagrange 9 | ≤ \(10^{-12}\) (ASPC-vs-Lagrange discriminator) |
| CG \(Q\) vs dense Cholesky on fixed RHS | abs ≤ \(10^{-8}\) e |
| Mean CG iters with `predictor_order=2` **and** `=5` **<** cold (`=0`) after history fill | strict inequality (measured on this branch: cold 6.000, \(K=2\) 5.000, \(K=5\) 5.000 over 10 post-fill steps) |
| \(\lvert\sum Q_i\rvert\) | ≤ \(10^{-8}\) e |

Measured regression output after the re-pin:

```
ASPC K=5 == MetalWalls B1..B6 {22/7,-55/14,55/21,-22/21,5/21,-1/42}, sum=1.0000000000000002
ASPC K=2: omega={0.5,-2,2.5}, linear exact, quadratic damped to 8
CG vs dense: Q_top dense=2.71753556e-01 cg=2.71753556e-01 Δ=0.00e+00
Multi-site: n_atoms=22 n_electrode=18
Mean CG iters (post-fill n=10): cold=6.000 ASPC K=2 5.000  ASPC K=5 (MetalWalls) 5.000
PASS
```

Sibling regressions that consume `predictor_order` were re-run against the
fixed coefficients and are unchanged: `reg_const-pot-il-edl-box` reproduces its
pinned charges to its own 1e-6 tolerance, and `reg_maze-symplectic-charge-dynamics`
still shows the loose-CG ASPC twin an order of magnitude less time-reversible
than MaZe (`ret_pos` 7.5e-07 Å vs 1.4e-14 Å).

## What MetalWalls could NOT confirm

The "warm start reduces CG iteration count" claim is **not** A/B-testable
against this binary, for two specific reasons:

1. MetalWalls has no switch to disable the predictor — it turns on
   unconditionally at step 7 (`src/system.F90:1430`), and there is no runtime
   keyword for it. The only available contrast is steps 1–6 vs steps 7+ of the
   same trajectory, which are different configurations.
2. MetalWalls' non-predictor branch is not a cold start: it initialises from the
   previous step's charges (`src/electrode_charge.F90:503-508`). Atomiverse's
   `predictor_order = 0` starts CG from the **zero** vector
   (`src/cpu/dynamics/const_pot_bo.cpp:670`), so "cold" means different things
   in the two codes.

Measured anyway on the twin: at `cg 1.0e-10` MetalWalls needs **4** iterations
at every one of the 30 steps; tightening to `cg 1.0e-14` gives 5–6 iterations
with no visible step-7 discontinuity. With only 18 unknowns the CG count
saturates and the predictor's benefit is below resolution in either code. The
Atomiverse inequality assertion therefore remains an internal-consistency check
of its own warm-start path, not an oracle-confirmed number — and it is labelled
as such here rather than dressed up as a MetalWalls result.

## Perf

```
perf_exempt: true
perf_exempt_reason: >
  A real MetalWalls binary now exists and was run (30 steps, 0.9265 s wall,
  0.0309 s/step, mpirun -n 1), but the two codes do not solve the same linear
  system for this case. MetalWalls is matrix-free: every CG iteration re-runs a
  full 2D-PBC Ewald apply_A over 18 unknowns plus LJ and Verlet integration.
  The ATV regression builds the dense 18x18 elastance ONCE
  (rebuild_matrix_every_step = false) over a non-periodic primary-cell sum, with
  no Ewald, no pair potential and no integrator. A wall-clock ratio would
  measure the electrostatics model and the matrix-free-vs-dense choice, not the
  charge predictor this slug pins. Pair-plate charge timing belongs to
  const-pot-bo-parallel-plate, where both codes run the same 2D-Ewald model.
ratio_atv_over_oracle: n/a
within_10x: exempt
```

## Repro of pin

1. Coefficients: `sed -n '556,596p' src/electrode_charge.F90` in the MetalWalls
   tree at c8ad175d, or evaluate the closed form
   `B_j = (-1)^(j+1) j C(2K+2,K+1-j)/C(2K,K)` at `K=5`.
   The binary exposes no diagnostic that prints the guess (CG errors out at
   `max_iterations = 0` rather than reporting the predictor output,
   `src/cg.f90:457-463`), so the pin is a source-level read of the exact tree
   the pinned binary was built from — not a black-box fit.
2. Twin run: the inputs under
   `tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/`
   reproduce `Q(E2) = +3.566190297414106E-01 e` at step 0.
