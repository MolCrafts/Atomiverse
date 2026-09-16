# Reference — XL-BOMD on a Gaussian-basis SCF (H₂O NVE, RHF/STO-3G forces)

## Method

**Extended-Lagrangian Born–Oppenheimer MD (XL-BOMD)** in its dissipative
(Niklasson) form, used as the *initial-guess propagator* of a Gaussian-basis
mean-field method:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::scf::RHF` bound at `md::Phase::force` (`add_ref`), with
  **`xl_bomd = true`, `xl_bomd_k = 5`**
- The method propagates an auxiliary AO density matrix `P` alongside the nuclei
  and hands it to the SCF as the initial guess each step:

  \[
  P_{n+1} = 2P_n - P_{n-1} + \kappa\,(D_n - P_n)
            + \alpha \sum_{k=0}^{K} c_k P_{n-k}
  \]

  with \(D_n\) the SCF-converged density at step \(n\) and \((\kappa,\alpha,c_k)\)
  the \(K=5\) row of Table I in Niklasson et al. (2009):
  \(\kappa = 1.82\), \(\alpha = 0.018\), \(c = \{-6, 14, -8, -3, 4, -1\}\).
- The SCF is still converged to `tol` every step (SCF-converged XL-BOMD; the
  linearized / zero-SCF shadow variant is out of scope and stays GPW-only).
- **Guess ladder.** `MeanFieldMethod` resolves one ordered chain —
  *explicit guess* > *XL `aux()`* > *ASPC predict* > *last-P reuse* — and
  `xl_bomd` is mutually exclusive with `predictor(...)` and with `cpmd`
  (two owners of "what is the density now" is the bug class those gates
  prevent). The auxiliary density advances **once per converged nested run**;
  a standalone `run()`/`forces()` leaves the history untouched.
- Units on the public surface: energy eV, length Å, time fs, force eV/Å.
  The SCF reports Hartree internally and writes eV to `frame.meta[kEnergy]`.

**Literature:**

- A. M. N. Niklasson, P. Steneteg, A. Odell, N. Bock, M. Challacombe,
  C. J. Tymczak, E. Holmström, G. Zheng, V. Weber, *J. Chem. Phys.* **130**,
  214109 (2009) — **dissipative XL-BOMD**; Table I is the coefficient source,
  asserted bitwise at unit level in `tests/cpu/electronic/test_xl_bomd.cpp`.
  The claim this case exercises is that of §II–III: the auxiliary density is a
  *better and time-reversible* SCF guess than reusing the last converged
  density, so warm steps converge in fewer iterations without the one-sided
  error that pumps energy into the trajectory.
- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008) — the
  underlying time-reversible scheme.
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC, the *extrapolation*
  scheme XL-BOMD contrasts with (`ASPC ≠ XL`; the two are mutually exclusive
  here).
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP 2009) §2.4–2.5 — BOMD
  energy conservation as the short-trajectory diagnostic; Frenkel & Smit
  §4.3.4, Allen & Tildesley §3.2 — symplectic Velocity-Verlet envelope.

The GPW twin of this case is `regressions/dynamics/xl-bomd-water` (same
propagator, plane-wave-assisted forces); the AO Car–Parrinello siblings on the
same nuclei are `dynamics/cpmd-rhf-h2`, `cpmd-rks-water`, `cpmd-uhf-o2`.

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated molecule |
| Geometry | CP2K getting-started, **centred in the cell** (same nuclei as `cpmd-rks-water`) |
| Masses | O 15.999 amu; H 1.008 amu |
| Cell | cubic \(L = 10\) Å (SCF ignores PBC; the box only frames the molecule) |
| Method | RHF (restricted Hartree–Fock), analytic gradient |
| Basis | STO-3G (7 AOs, 10 electrons) |
| SCF | `tol = 1e-6` Ha (\|ΔE\| exit), `max_iter = 200` |
| XL-BOMD | `xl_bomd = true`, `K = 5` (κ = 1.82, α = 0.018) |
| Ensemble | NVE |
| `dt` | **0.25 fs** (see *Timestep* below) |
| Initial velocity | COM-preserving H stretch \(v_x(\mathrm{H1})=+0.0015\), \(v_x(\mathrm{H2})=-0.0015\) Å/fs |
| Steps | 20 (step 1 is the cold seed; the means use steps 9–20) |

| Atom | x | y | z (Å) |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

### Timestep — why 0.25 fs and not the family's 0.5 fs

The auxiliary density is a harmonic oscillator with \(\kappa = \omega^2\Delta t^2\)
(Table I fixes κ = 1.82, i.e. \(\omega\Delta t = 1.35\)), so it tracks the true
density only while the *nuclear* motion is slow compared with that oscillator.
The tracking error scales as \((\omega_\mathrm{nuc}\Delta t)^2/\kappa\), against
\(\omega_\mathrm{nuc}\Delta t\) for plain reuse — the advantage ratio is
\(\kappa/(\omega_\mathrm{nuc}\Delta t)\) and *collapses as Δt grows*. Water's
O–H stretch has a ≈ 9 fs period, so 0.25 fs is ≈ 36 steps/period (the regime
XL-BOMD is designed for) while 1.0 fs is ≈ 9. This is measured, not asserted —
see the guess-error table below. **A future editor raising `dt` here will
watch the `vs reuse` pin fail; that is the pin working, not flaking.**

## Measured landscape (probed 2026-07-27, `dev`, before freezing)

All numbers below are from this tree on the Arrhenius x86_64 login node
(GCC 14.3, OpenBLAS 0.3.32). Every iteration sequence was **bit-identical at
`OMP_NUM_THREADS=1` and `8`**.

### 1. Guess quality (the physical signal)

The SCF exits on \(|\Delta E| < \mathrm{tol}\), which makes the iteration
counter *coarse and bimodal* — it cannot resolve how good a guess is once the
guess is "good enough". Guess quality was therefore measured directly: warm the
trajectory, then take one step at `tol = 1e6` (a one-shot: the loop stops after
2 iterations) and compare that energy with the fully converged energy at the
same geometry. The HF energy is stationary at the converged density, so
\(E_\text{one-shot} - E_\text{ref} \sim O(|\delta P|^2) \ge 0\) is a monotone
guess-error meter [Ha]:

| Δt (fs) | warm steps | err(XL) | err(reuse) | reuse/XL |
| --- | --- | --- | --- | --- |
| 0.25 | 4 | 4.92e-08 | 6.64e-06 | **135×** |
| 0.25 | 8 | 9.01e-08 | 1.15e-05 | **128×** |
| 0.25 | 12 | 3.06e-08 | 8.17e-06 | **267×** |
| 0.25 | 16 | 7.07e-09 | 3.59e-06 | **507×** |
| 0.5 | 4 | 5.98e-06 | 4.72e-05 | 7.9× |
| 0.5 | 8 | 2.49e-06 | 1.42e-05 | 5.7× |
| 0.5 | 12 | 4.75e-07 | 1.85e-06 | 3.9× |
| 0.5 | 16 | 2.07e-06 | 2.41e-07 | 0.12× |
| 1.0 | 4 | 1.27e-04 | 5.32e-05 | 0.42× |
| 1.0 | 8 | 4.18e-05 | 1.19e-07 | 0.00× |
| 1.0 | 12 | 3.17e-05 | 5.58e-05 | 1.8× |
| 1.0 | 16 | 1.44e-04 | 5.50e-08 | 0.00× |

Two conclusions were frozen from this table: the propagator **is** wired
correctly (a 500× better guess is not something a mis-seeded history produces),
and its advantage is a strong function of Δt exactly as κ = ω²Δt² predicts.

### 2. Warm-step SCF iterations (what the case pins)

Mean over steps 9–20 (the dissipative history is K+1 = 6 deep, so by step 9 the
stationary seed is fully flushed). "cold" = a brand-new `RHF` at each geometry
the XL leg visited; "reuse" = the same trajectory with `xl_bomd = false`;
"ASPC(3)" is context only (not run by the case):

| Δt (fs) | tol (Ha) | steps | XL | reuse | cold | ASPC(3) | gain vs cold | gain vs reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0.25** | **1e-6** | **20** | **5.667** | **6.583** | **8.250** | 6.000 | **2.583** | **0.917** |
| 0.25 | 1e-6 | 24 | 5.750 | 6.750 | 8.375 | — | 2.625 | 1.000 |
| 0.25 | 1e-6 | 40 | 5.406 | 6.500 | 8.688 | — | 3.281 | 1.094 |
| 0.25 | 1e-9 | 20 | 7.917 | 8.833 | 9.750 | 7.500 | 1.833 | 0.917 |
| 0.25 | 1e-9 | 40 | 7.969 | 8.313 | 9.906 | — | 1.937 | 0.344 |
| 0.5 | 1e-6 | 20 | 6.500 | 6.917 | 8.750 | — | 2.250 | 0.417 |
| 0.5 | 1e-9 | 20 | 8.000 | 8.167 | 10.000 | — | 2.000 | 0.167 |
| 0.5 | 1e-11 | 20 | 9.000 | 9.000 | 10.000 | — | 1.000 | 0.000 |
| 1.0 | 1e-9 | 20 | 8.833 | 8.667 | 9.833 | — | 1.000 | **−0.167** |

Read this table before touching the configuration. The **cold** margin is large
and stable everywhere (1.0–3.3 iterations); the **reuse** margin is the
sensitive one, because it is the difference of two integer-valued counts that
are already close to the tol-imposed floor. The frozen row is the first one.

The per-step counts of the frozen row (step 1 is the cold seed in every leg):

```
XL   : 9 5 6 6 6 6 6 6 6 6 6 5 6 6 5 5 6 5 6 6
reuse: 9 5 5 7 6 7 6 7 7 7 7 7 7 7 6 6 6 6 6 7
cold : 9 9 9 9 9 9 9 9 9 9 9 9 8 8 8 8 8 8 8 7
```

### 3. Trajectory and conservation

| Quantity | Measured |
| --- | --- |
| \(E_0 = E_\mathrm{pot}+E_\mathrm{kin}\) after step 1 | −2039.3776633187 eV |
| \(E\) after step 20 | −2039.3792117890 eV |
| worst relative \(|E-E_0|/\max(1,|E_0|)\) (XL) | 1.955e-06 |
| worst relative drift (plain-BOMD twin) | 1.954e-06 |
| max \|Δx\| between the XL and reuse legs | 7.6e-09 Å |
| final O–H | 1.0654 / 1.0654 Å |
| wall (login node, OMP=1) | 0.41 s |

The energy excursion is a smooth O–H-stretch oscillation of ≈ 4 meV amplitude
with no secular ramp; it is dominated by the Velocity-Verlet discretization,
not by the SCF — the plain-BOMD twin lands within 1e-9 of the same number. A
20-step run cannot demonstrate the *absence* of long-time drift (XL-BOMD's
headline property); this pin is an **envelope guard** that a broken recursion
(sign error, non-density aux, shape leak, stale history) breaks immediately.

## Observables and tolerances asserted

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV] with \(E_0\) after
step 1, and \(\bar n_X\) the mean SCF iteration count over steps 9–20 of leg
\(X\).

| # | Assertion | Frozen | Measured | Headroom |
| - | --------- | ------ | -------- | -------- |
| 1 | \(\bar n_\mathrm{cold} - \bar n_\mathrm{XL} \ge\) | 1.50 | 2.583 | 1.7× |
| 2 | \(\bar n_\mathrm{reuse} - \bar n_\mathrm{XL} \ge\) | 0.40 | 0.917 | 2.3× |
| 3 | \(\max_t |E(t)-E_0|/\max(1,|E_0|) \le\) | 2.0e-05 | 1.955e-06 | 10× |
| 4 | max \|Δx\| between the two MD legs ≤ | 1e-06 Å | 7.6e-09 | 130× |
| 5 | `xl.xl_advances()` == | 20 | 20 | exact |
| 6 | `reuse.xl_advances()` == | 0 | 0 | exact |
| 7 | cold replay of step 1 == XL leg's own seed count | identity | 9 == 9 | exact |
| 8 | SCF converged at **every** step of both MD legs | — | yes | — |
| 9 | final PE inside the bound-H₂O window (−2200, −1900) eV | — | −2039.62 | — |
| 10 | both O–H in (0.70, 1.30) Å; all coordinates finite | — | 1.0654 | — |

Assertions 5–7 are the *structural* pins that keep 1–2 honest: 5 proves the
propagator ran on the nuclear clock (one advance per converged nested step, not
per entry), 6 proves the twin really is un-propagated, and 7 proves the cold
baseline was measured at the geometries the XL leg actually solved at rather
than at a replay that silently diverged.

This is **not** a golden snapshot of an Atomiverse total energy: no number
above is asserted as a value. What is asserted is the *iteration-count
reduction* Niklasson (2009) claims for the scheme, plus the conservation
envelope and the structural witnesses.

## Offline oracle

**Status:** none required, and none used. The pinned quantity is a *relative*
property of two runs of this code (warm vs cold vs reuse SCF iteration counts
on one trajectory), which no external engine can supply — SCF iteration counts
are not comparable across programs (different convergence measure, DIIS
history, integral screening and initial guess). The *coefficients* of the
recursion are the external fact, and they are hard-coded from Table I of
Niklasson et al. (2009) and asserted bitwise in
`tests/cpu/electronic/test_xl_bomd.cpp` (every row satisfies
\(\sum_k c_k = 0\) and \(\sum_k k\,c_k = 0\)). Absolute RHF/STO-3G energetics
for this method/basis are pinned against PySCF elsewhere in `regressions/scf/`.

No third-party package is imported, executed or read at build or test time.

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | The observable is an internal iteration-count ratio; there is no cross-engine wall twin to compare against (and no CP2K/PySCF binary on the worker host, 2026-07-27 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

ATV wall, Arrhenius login node, `OMP_NUM_THREADS=1`, 2026-07-27: **0.41 s**
(60 s budget). The three legs cost 0.17 s (XL) + 0.17 s (reuse) + 0.06 s (20
cold single points).

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | `reg_xl-bomd-rhf-water` asserts ac-004 of `.claude/specs/esmm-cpmd-05-scf-xlbomd/acceptance.md`; ac-001…ac-003 are pinned by `tests/cpu/scf/test_scf_xl_bomd.cpp`, the Table-I coefficients by `tests/cpu/electronic/test_xl_bomd.cpp` |
| Speed | `perf_exempt=true` (no cross-engine wall twin for an iteration-count observable) |
