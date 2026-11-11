# dynamics/xl-bomd

One case; every section below is run by `atv_regressions --case dynamics/xl-bomd` and keeps its own oracle record.


## Reference — XL-BOMD short trajectory (H₂O NVE, GPW forces)

### Method

**Extended-Lagrangian Born–Oppenheimer MD (XL-BOMD)** in its dissipative
(Niklasson) form, on Gaussian-and-plane-waves (GPW / Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true` and
  **`xl_bomd = true`, `xl_bomd_k = 5`**
- The evaluator propagates an auxiliary AO density matrix `P` alongside the
  nuclei and uses it as the SCF initial guess each step:

  \[
  P_{n+1} = 2P_n - P_{n-1} + \kappa\,(D_n - P_n)
            + \alpha \sum_{k=0}^{K} c_k P_{n-k}
  \]

  with \(D_n\) the SCF density at step \(n\) and \((\kappa,\alpha,c_k)\) the
  \(K=5\) row of Table I in Niklasson et al. (2009):
  \(\kappa = 1.82\), \(\alpha = 0.018\), \(c = \{-6, 14, -8, -3, 4, -1\}\).
- SCF is still converged to `tol` each step (SCF-converged XL-BOMD; the
  linearized-functional zero-SCF variant is out of scope for this foothold).
- Fixed cubic cell (GPW rejects box changes after `build()`)
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature (the pinned observables and coefficients):**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008) —
  time-reversible extended-Lagrangian BOMD.
- A. M. N. Niklasson, P. Steneteg, A. Odell, N. Bock, M. Challacombe,
  C. J. Tymczak, E. Holmström, G. Zheng, V. Weber, *J. Chem. Phys.* **130**,
  214109 (2009) — dissipative XL-BOMD; **Table I** is the exact coefficient
  pin asserted bitwise in `tests/cpu/gpw/test_gpw_xl_bomd.cpp` (every row
  satisfies \(\sum_k c_k = 0\) and \(\sum_k k\,c_k = 0\)).
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC (the
  *extrapolation* scheme XL-BOMD supersedes here; `ASPC ≠ XL`).
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — BOMD energy
  conservation as the short-trajectory diagnostic; Frenkel & Smit §4.3.4,
  Allen & Tildesley §3.2 — symplectic Velocity Verlet envelope.
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW
  Quickstep energy/force framework (CP2K method family).

**Force quality** for this geometry/method family is offline-pinned vs
**CP2K 2026.1** in `regressions/gpw/rks` (max |ΔF| ≈ 3.4×10⁻⁴
Ha/Bohr); plain-BOMD MD binding + NVE conservation is pinned in
`regressions/dynamics/bomd`. This slug pins the **XL auxiliary-density
propagation** (exact Table-I coefficients at unit level) and the **NVE
conservation envelope with the XL guess engaged** on the product
`GPWEvaluator` path.

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative drift that a working XL-BOMD guess must obey —
a wrong recursion (sign error, non-density aux, shape leak) destroys SCF
convergence or the drift envelope immediately.

### System

Identical twin of `gpw-bomd-water` (only `xl_bomd` differs):

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT |
| Geometry | CP2K getting-started (same as `gpw-forces-water`) |
| Masses | O 15.999 amu; H 1.008 amu |
| Cell | cubic \(L = 10\) Å; `PERIODIC NONE` / `nonperiodic = true` |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 200 Ry (envelope run; forces pin uses 400 Ry) |
| SCF | `tol = 1e-5`, `max_iter = 60`, DIIS diagonalization path |
| XL-BOMD | `K = 5` (κ = 1.82, α = 0.018) |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | COM-preserving H stretch \(v_x(\mathrm{H1})=+0.0015\), \(v_x(\mathrm{H2})=-0.0015\) Å/fs |
| Steps | 1 warm-up (seeds the XL history) + 4 production (chunked) |

| Atom | x | y | z (Å) |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

### Offline oracle (CP2K)

**Status:** no `cp2k` / `cp2k.psmp` / `cp2k.ssmp` / `cp2k.sopt` binary and no
CP2K Lmod module on the worker host used for this pin (Arrhenius login-class
node, re-checked **2026-07-25**) — same inventory as the `gpw-bomd-water`
pin. A CP2K twin (short NVE Quickstep MD on the same H₂O cell; ASPC
extrapolation is CP2K's default guess propagation for MD — record
`EXTRAPOLATION` settings when pinning) is sketched in
`tools/cp2k_oracle/inputs/xl_bomd_water_nve.inp`; when a binary is
available, re-run offline via `tools/cp2k_oracle/` and replace this section
with the version string + measured drift envelope + fair wall.

Until then the regression asserts the **literature XL-BOMD/NVE conservation
envelope** plus finite/bounded trajectory diagnostics and per-step SCF
convergence with the XL guess; the exact Niklasson Table-I coefficients are
asserted at unit level (`test_cpu_gpw_xl_bomd`).

### Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) taken
after a single warm-up step (energies populated; XL history seeded).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.05
\]

Additional gates:

- SCF **converged at every sampled step** with the XL auxiliary-density
  guess (`last_result().converged`, `iters < max_iter`)
- PE finite and bound (\(< -100\) eV after warm-up for PBE water)
- KE finite and \(\ge 0\), non-trivial after warm-up
- All coordinates finite
- Soft O–H bond ceiling: both \(r_{\mathrm{OH}} < 1.8\) Å after production

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step AIMD (2026-07-25 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

Note: single-point force wall vs CP2K measured in `gpw-forces-water`
(ratio **3.56**, within_10x). In XL mode the force-pass SCF additionally
warm-starts from the just-converged MO coefficients (same geometry), so the
XL trajectory is not slower than the plain-BOMD twin.

### Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_xl-bomd-water` asserts A7 of `.claude/specs/xl-bomd-water/acceptance.md`; unit pin A1–A6 in `test_cpu_gpw_xl_bomd` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |

**ATV wall (login-class, OMP=4, 2026-07-25):** measured at gate time — see
final summary in the worker report. Measured relative drift and final SCF
iteration count are printed by the regression binary.


## Reference — XL-BOMD on a Gaussian-basis SCF (H₂O NVE, RHF/STO-3G forces)

### Method

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

The GPW twin of this case is `regressions/dynamics/xl-bomd` (same
propagator, plane-wave-assisted forces); the AO Car–Parrinello siblings on the
same nuclei are `dynamics/cpmd-rhf-h2`, `cpmd-rks-water`, `cpmd-uhf-o2`.

### System

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

#### Timestep — why 0.25 fs and not the family's 0.5 fs

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

### Measured landscape (probed 2026-07-27, `dev`, before freezing)

All numbers below are from this tree on the Arrhenius x86_64 login node
(GCC 14.3, OpenBLAS 0.3.32). Every iteration sequence was **bit-identical at
`OMP_NUM_THREADS=1` and `8`**.

#### 1. Guess quality (the physical signal)

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

#### 2. Warm-step SCF iterations (what the case pins)

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

#### 3. Trajectory and conservation

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

### Observables and tolerances asserted

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

### Offline oracle

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

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | The observable is an internal iteration-count ratio; there is no cross-engine wall twin to compare against (and no CP2K/PySCF binary on the worker host, 2026-07-27 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

ATV wall, Arrhenius login node, `OMP_NUM_THREADS=1`, 2026-07-27: **0.41 s**
(60 s budget). The three legs cost 0.17 s (XL) + 0.17 s (reuse) + 0.06 s (20
cold single points).

### Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | `reg_xl-bomd-rhf-water` asserts ac-004 of `.claude/specs/esmm-cpmd-05-scf-xlbomd/acceptance.md`; ac-001…ac-003 are pinned by `tests/cpu/scf/test_scf_xl_bomd.cpp`, the Table-I coefficients by `tests/cpu/electronic/test_xl_bomd.cpp` |
| Speed | `perf_exempt=true` (no cross-engine wall twin for an iteration-count observable) |


## Reference — XL-BOMD short trajectory (O₂ triplet NVE, Γ-UKS GPW)

### Method

**Extended-Lagrangian Born–Oppenheimer MD (XL-BOMD)** in its dissipative
(Niklasson) form, on **open-shell Γ-UKS** Gaussian-and-plane-waves (GPW /
Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 7`, `n_beta = 5`**, **`xl_bomd = true`, `xl_bomd_k = 5`**
- Independent auxiliary AO densities \(P^\alpha, P^\beta\) are propagated
  with the same Table-I recursion and used as SCF initial spin densities:

  \[
  P^\sigma_{n+1} = 2P^\sigma_n - P^\sigma_{n-1} + \kappa\,(D^\sigma_n - P^\sigma_n)
            + \alpha \sum_{k=0}^{K} c_k P^\sigma_{n-k}
  \]

  with \(D^\sigma_n\) the SCF spin density at step \(n\) and
  \((\kappa,\alpha,c_k)\) the \(K=5\) row of Table I in Niklasson et al.
  (2009): \(\kappa = 1.82\), \(\alpha = 0.018\),
  \(c = \{-6, 14, -8, -3, 4, -1\}\).
- SCF is still converged to `tol` each step (SCF-converged XL-BOMD).
- Fixed cubic cell; MT / nonperiodic
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) —
  Table I coefficients (unit-pinned in `test_cpu_gpw_xl_bomd`)
- Closed-shell product path: `regressions/dynamics/xl-bomd`
- UKS O₂ energy / forces offline CP2K pins:
  `regressions/gpw/uks`, `gpw-uks-forces`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative NVE drift with dual-spin XL guesses engaged —
a wrong recursion (α-only, swapped channels, non-density aux) destroys SCF
convergence or the drift envelope immediately.

### System

| Quantity | Value |
| -------- | ----- |
| Species | O₂ (2 × O), isolated / MT |
| Geometry | \(R = 1.2075\) Å, centered in cubic cell |
| Masses | 15.999 amu each |
| Cell | cubic \(L = 12\) Å; `PERIODIC NONE` / `nonperiodic = true` |
| Spin | UKS triplet, \(n_\alpha=7\), \(n_\beta=5\) (GTH-q6, 12 valence e⁻) |
| XC | PADE (LSD) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 |
| CUTOFF | 200 Ry (envelope run) |
| SCF | `tol = 1e-5`, `max_iter = 80`, DIIS diagonalization |
| XL-BOMD | `K = 5`; dual α/β propagators |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_z = \pm 0.0008\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

### Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`xl-bomd-water`. A CP2K twin sketch (short NVE Quickstep UKS MD; ASPC
extrapolation is CP2K's MD default density-guess propagation) lives in
`tools/cp2k_oracle/inputs/xl-bomd-uks/`. When a binary is available, re-run
offline and replace this section with version + measured drift + fair wall.

Until then the regression asserts the **literature XL-BOMD/NVE conservation
envelope** plus finite/bounded trajectory diagnostics, triplet ⟨S²⟩ window,
and per-step SCF convergence with dual XL spin guesses; Table-I coefficients
remain unit-pinned (`test_cpu_gpw_xl_bomd`).

### Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) after
one warm-up step (XL histories seeded).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.05
\]

Additional gates:

- SCF **converged at every sampled step** with dual XL spin density guesses
- PE finite and bound (\(< -100\) eV after warm-up)
- KE finite and \(\ge 0\), non-trivial after warm-up
- ⟨S²⟩ in \([1.5, 2.5]\) (triplet window) after warm-up and production
- All coordinates finite
- Soft O–O window: \(0.9 < r_{\mathrm{OO}} < 1.6\) Å after production

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step UKS AIMD (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

### Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_xl-bomd-uks` asserts A7; unit A1–A6 in `test_cpu_gpw_xl_bomd` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |


## Reference — multi-k XL-BOMD short trajectory (H₂ NVE, GPW forces)

### Method

**Extended-Lagrangian Born–Oppenheimer MD (XL-BOMD)** in its dissipative
(Niklasson) form, on **multi-k closed-shell RKS** Gaussian-and-plane-waves
(GPW / Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  **`xl_bomd = true`, `xl_bomd_k = 5`**
- One auxiliary complex density \(P_k\) per sampled k is propagated with
  the Table-I recursion and used as the multi-k SCF initial density:

  \[
  P^{k}_{n+1} = 2P^{k}_n - P^{k}_{n-1} + \kappa\,(D^{k}_n - P^{k}_n)
            + \alpha \sum_{m=0}^{K} c_m P^{k}_{n-m}
  \]

  with \(D^{k}_n\) the SCF density at step \(n\) and
  \((\kappa,\alpha,c_m)\) the \(K=5\) row of Table I in Niklasson et al.
  (2009): \(\kappa = 1.82\), \(\alpha = 0.018\),
  \(c = \{-6, 14, -8, -3, 4, -1\}\).
- SCF is still converged to `tol` each step (SCF-converged XL-BOMD).
- Periodic cubic cell; PADE XC
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) —
  Table I coefficients (unit-pinned in `test_cpu_gpw_xl_bomd`)
- Γ product paths: `regressions/dynamics/xl-bomd`, `xl-bomd-uks`
- Multi-k analytic-force cases: `regressions/gpw/mbd`, `gpw-vdw-kpoint-forces`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative NVE drift with multi-k XL \(P_k\) guesses
engaged — a wrong recursion (count leak, real Γ guess on k path, shape
error) destroys SCF convergence or the drift envelope immediately.

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | closed-shell RKS, \(n_\mathrm{occ}=1\) |
| XC | PADE (LDA) |
| Basis | DZV-GTH-PADE |
| PP | GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF | `tol = 1e-5`, `max_iter = 60`, DIIS diagonalization |
| XL-BOMD | `K = 5`; one `XlBomdCplx` per k |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

### Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`xl-bomd-water` / `xl-bomd-uks`. A CP2K twin sketch (short NVE Quickstep MD
with ASPC extrapolation + `&KPOINTS`) lives in
`tools/cp2k_oracle/inputs/xl-bomd-kpoints/`. When a binary is available,
re-run offline and replace this section with version + measured drift +
fair wall.

Until then the regression asserts the **literature XL-BOMD/NVE conservation
envelope** plus multi-k payload diagnostics (`n_kpts >= 2`, `P_k` size),
per-step SCF convergence with XL \(P_k\) guesses, and soft H–H window.
Table-I coefficients remain unit-pinned (`test_cpu_gpw_xl_bomd`).

### Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) after
one warm-up step (XL histories seeded).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.05
\]

Additional gates:

- SCF **converged at every sampled step** with multi-k XL density guesses
- `n_kpts >= 2` and `P_k.size() == n_kpts` after warm-up
- PE finite and bound (\(< -5\) eV after warm-up)
- KE finite and \(\ge 0\), non-trivial after warm-up
- All coordinates finite
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step multi-k AIMD (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |


## Reference — zero-SCF linearized XL-BOMD forces (H₂O NVE, GPW)

### Method

**Zero-SCF linearized-functional XL-BOMD** (Niklasson shadow Born–Oppenheimer
potential) on Gaussian-and-plane-waves (GPW) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE)
- Force/energy: `cpu::gpw::GPWEvaluator` with
  **`xl_bomd = true`, `xl_bomd_k = 5`, `xl_bomd_linearized = true`**,
  `compute_forces = true`
- Cold seed (step 0): full SCF → seed XL history with \(D_{\mathrm{SCF}}\)
- Warm steps: auxiliary \(n = P_{\mathrm{aux}}\); one \(H[n]\) construction
  (fixed-density, `max_iter = 0`); one diagonalization \(D = q[n]\); **no
  iterative SCF** (`last_result().iters == 0`)
- Shadow energy
  \(\mathcal{E} = \mathrm{Tr}(H[n]\,D) - \mathrm{Tr}(H[n]\,n) + E_{\mathrm{KS}}[n]\)
  (recovers \(E_{\mathrm{KS}}\) at \(D = n\))
- Forces: Hellmann–Feynman of \(\mathcal{U}(R,n)\) via
  `internal::gpw_forces_at_density(D, C_{\mathrm{occ}}, H[n])`
- XL advance(\(D\)) with Table-I \(K=5\) dissipation

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
  (unit-pinned in `test_cpu_gpw_xl_bomd`)
- A. M. N. Niklasson & M. J. Cawkwell, *Phys. Rev. B* **86**, 174308 (2012);
  Niklasson, *J. Chem. Phys.* **147**, 054103 (2017) — shadow / linearized
  free-energy functional
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW

### System

Identical twin of `xl-bomd-water` (only `xl_bomd_linearized` differs):

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT |
| Geometry | CP2K getting-started |
| Cell | cubic \(L = 10\) Å; nonperiodic |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 200 Ry |
| XL-BOMD | \(K = 5\); linearized forces on |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Steps | 1 cold seed + 4 production |

| Atom | x | y | z (Å) |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

### Offline oracle (CP2K)

**Status:** no `cp2k` binary / Lmod module on the worker host used for this
pin (Arrhenius login-class, 2026-07-26 inventory) — same as prior XL/BOMD
AIMD slugs. Twin sketch kept at
`tools/cp2k_oracle/inputs/xl-bomd-linearized-forces/h2o_xl_linearized_nve.inp`
(Quickstep MD with low `MAX_SCF` / density-matrix propagation notes). When a
binary is available, re-run offline via `tools/cp2k_oracle/` and replace this
section with version + measured drift + fair wall.

Until then the regression asserts the **literature shadow-XL / NVE envelope**
plus **zero warm-step SCF iterations** and trajectory diagnostics. Table-I
coefficients remain unit-pinned (`test_cpu_gpw_xl_bomd`).

### Observable and tolerance

Let \(E(t) = E_{\mathrm{pot}}(t) + E_{\mathrm{kin}}(t)\) [eV], with \(E_0\) after
the cold seed.

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Warm steps: `last_result().iters == 0` and `converged == true`
- Cold seed: `iters > 0` and converged
- PE finite and bound (\(< -100\) eV after seed)
- KE finite, \(\ge 0\), non-trivial after seed
- Soft O–H: both \(r_{\mathrm{OH}} < 1.8\) Å after production

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary for fair AIMD wall twin (2026-07-26) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

Note: warm steps are cheaper than full-SCF XL (one \(H\) build + one
diagonalization vs iterative DIIS).

### Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | A1–A4 unit; A5 `reg_xl-bomd-linearized-forces` |
| Speed | `perf_exempt=true` |


## Reference — zero-SCF linearized XL-BOMD shadow-density stress (periodic H₂ NVE)

### Method

**Zero-SCF linearized-functional XL-BOMD** (Niklasson shadow Born–Oppenheimer
potential) with **analytic fixed-\(P\) Hellmann–Feynman cell stress** at the
shadow payload:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE)
- Force/energy/stress: `cpu::gpw::GPWEvaluator` with
  `compute_forces = true`, **`compute_stress = true`**,
  `stress_use_fd = false`, wrapped by
  `cpu::electronic::XlBomdForce{gpw, 5, XlBomdLinearized{}}`
- Cold seed (step 0): full SCF → analytic fixed-\(P\) stress at that SCF's
  state (same path as `GpwDft::stress`) → seeds XL histories
- Warm steps: zero-SCF shadow forces (as in `xl-bomd-linearized-forces`) **plus**
  the evaluator's shadow-state stress (strain step / axes from
  `method().stress_strain_h` / `.stress_axes`) — fixed-\(P\) Lagrangian at
  \(P = D = q[n]\), \(W\) from the \(H[n]\) eigenpairs; **no reference SCF**;
  read back through `last_stress_ha_bohr3()` and the Frame's `kStress`
- `stress_use_fd` + linearized remains **refused** at `build()` (multi-SCF
  strain-FD re-converges the SCF under strain — Born–Oppenheimer stress, not
  the shadow functional)

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008); *J. Chem. Phys.*
  **130**, 214109 (2009) Table I; *Phys. Rev. B* **86**, 174308 (2012);
  *J. Chem. Phys.* **147**, 054103 (2017) — shadow / linearized functional
- O. H. Nielsen & R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — fixed-\(P\)
  cell theorem
- Sibling force product: `regressions/dynamics/xl-bomd`
- BO stress product: `regressions/gpw/rks`, `gpw-kpoint-stress`

### System (twin of the CP2K input)

Multi-k compressed-H₂ GPW system:

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic; bond \(R = 1.0\) Bohr along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| XC | PADE (LDA) |
| CUTOFF | 300 Ry |
| k mesh (legs A/B) | MONKHORST-PACK 1×1×2, `k_time_reversal=false` ⇔ CP2K `FULL_GRID .TRUE.` |
| k mesh (legs C/D) | Γ only |
| SCF (cold seed) | `tol = 1e-7`, `max_iter = 60`, `mix = 0.3` |
| XL-BOMD | \(K = 5\); linearized shadow forces + shadow stress |
| Ensemble / dt / steps | NVE, 0.5 fs, 1 cold seed + 4 production |
| Initial velocity | antisymmetric stretch \(v_x = \mp 0.001\) Å/fs |

Four legs in one binary, all `compute_stress = true`:

- **A (anchor, k):** converged BOMD multi-k — per-step \(\sigma_{xx}\) gated vs CP2K
- **B (product, k):** linearized multi-k — warm `iters == 0` + finite \(\sigma\)
- **C (anchor, Γ):** converged BOMD Γ
- **D (product, Γ):** linearized Γ

### Offline oracle (CP2K)

CP2K 2026.1 has **no zero-SCF shadow-functional XL product**, so the shadow
stress itself cannot be twinned. What CP2K **does** pin is the converged-BOMD
multi-k stress surface the shadow must track:

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`~/.local/bin/cp2k` → Spack `cp2k.psmp`) |
| Host | Arrhenius epyc login (offline; **never** at ctest) |
| Input | `tools/cp2k_oracle/inputs/xl-bomd-linearized-stress/h2_xl_linearized_stress_nve.inp` |
| Launch | `OMP_NUM_THREADS=8 cp2k -i h2_xl_linearized_stress_nve.inp` |
| Date | 2026-07-31 |
| Stress mode | **`STRESS_TENSOR ANALYTICAL`** (accepted with `KPOINTS`; not refused) |
| Wall | **2.13 s** (`/usr/bin/time`) |

**Pinned `.ener` Pot:**

| Step | Kin [Ha] | Pot [Ha] | Cons [Ha] |
| ---- | -------- | -------- | --------- |
| 0 | 0.000003839 | **−1.074050184** | −1.074046345 |
| 1 | 0.020127024 | **−1.097682104** | −1.077555080 |
| 2 | 0.046817650 | **−1.125544577** | −1.078726927 |
| 3 | 0.052842084 | **−1.130531455** | −1.077689371 |
| 4 | 0.044574339 | **−1.121507812** | −1.076933473 |

**Pinned `.stress` diagonals** (`PROJECT-1.stress` [bar] → Ha/Bohr³ with
\(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\)):

| Step | \(\sigma_{xx}\) [Ha/Bohr³] | \(\sigma_{yy}\) | \(\sigma_{zz}\) |
| ---- | -------------------------- | --------------- | --------------- |
| 0 | **1.998440567215e-04** | −7.010935192196e-08 | −2.386241961179e-09 |
| 1 | **1.701541305617e-04** | −7.810856540460e-08 | −2.134981164927e-09 |
| 2 | **1.087027341964e-04** | −9.651444883875e-08 | −1.350517576735e-09 |
| 3 | **4.492363435060e-05** | −1.194770631848e-07 | 2.382815718937e-11 |
| 4 | **−8.093268435310e-06** | −1.406572107551e-07 | 1.879278928819e-09 |

Off-diagonal components are \(\sim 10^{-12}\) bar (numerical noise). The hard
CP2K gate in the regression is on \(\sigma_{xx}\) (dominant compressed-bond
component); yy/zz are near-zero noise on this orthorhombic stretch.

### Observable, pins, and tolerances

`reg_xl-bomd-linearized-stress.cpp` asserts (measured 2026-07-31, OMP=8):

| Gate | Pin / band | Measured |
| ---- | ---------- | -------- |
| B/D warm steps | `iters == 0`, `converged`, finite off-diag-0 \(\sigma\), `kStress` written | ✓ |
| Dual gate (seed k mesh) | `GpwDft::stress` on two independent builds agree ≤ **1e-10**; vs `gpw_stress_fd` ≤ **1e-6** Ha/Bohr³ (library returns eV/Å³; divided by `units::kHartreePerBohr3InEvPerAngstrom3` before gating) | **0** / **9.86e-11** |
| A \(\sigma_{xx}\) vs CP2K | ≤ **1e-4** Ha/Bohr³ | worst **6.026e-05** (seed residual ~3.4e-6; trajectory residual grows with engine-model force drift) |
| A/B diag stress | ≤ **1e-5** Ha/Bohr³ | worst **1.970e-06** |
| C/D diag stress | ≤ **1e-5** Ha/Bohr³ | worst **1.969e-06** |
| NVE drift (B, D) | ≤ 0.05 | B **4.056e-3**, D **4.057e-3** |
| Geometry | soft H–H 0.4–1.5 Å | r ≈ 0.907–0.908 Å |

Measured per-step \(\sigma_{xx}\) [Ha/Bohr³] (2026-07-31):

| Step | CP2K | A (BOMD-k) | B (lin-k) | C (BOMD-Γ) | D (lin-Γ) |
| ---- | ---- | ---------- | --------- | ---------- | --------- |
| 0 | 1.998441e-04 | 1.964528e-04 | 1.964528e-04 | 1.964574e-04 | 1.964574e-04 |
| 1 | 1.701541e-04 | 1.464863e-04 | 1.471052e-04 | 1.464903e-04 | 1.471092e-04 |
| 2 | 1.087027e-04 | 5.558410e-05 | 5.642855e-05 | 5.558798e-05 | 5.643209e-05 |
| 3 | 4.492363e-05 | −1.533995e-05 | −1.625187e-05 | −1.533570e-05 | −1.624754e-05 |
| 4 | −8.093268e-06 | −5.905601e-05 | −6.102614e-05 | −5.905180e-05 | −6.102122e-05 |

The A/B (and C/D) envelope is the decisive shadow gate: wrong payload,
missing \(W\), or silent stress drop moves the linearized trace away from the
BOMD anchor by orders more than \(2\times 10^{-6}\). Absolute CP2K residual is
dominated by the known periodic-GPW lattice-sum/PP-tail class (seed) plus
trajectory geometry drift (same class as the sibling energy band).

### Perf gate (fair wall, same host / OMP=8 / single rank)

| Metric | Value |
| ------ | ----- |
| \(T_{\mathrm{cp2k}}\) (4-step converged-BOMD k + analytical stress) | **2.13 s** |
| \(T_{\mathrm{atv}}\) (leg B: 1 seed + 4 linearized steps with shadow stress) | **5.12 s** |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **2.4** |
| within_10x | **true** |
| perf_exempt | false |
| Regression wall (dual gate + four legs) | **36.8 s** (≤ 60 s budget) |

Shadow stress adds 6 fixed-\(P\) energy evals per warm step (no extra SCF);
the linearized path remains faster than full BOMD-k with stress (A 6.04 s vs
B 5.12 s on this window).

### Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | unit `test_cpu_gpw_stress` (fixed-\(P\) stress + unconverged throw) + unit `test_cpu_gpw_xl_bomd` (build accepts stress; refuses `stress_use_fd`) + e2e this regression |
| Speed | ratio 2.4 ≤ 10 (measured) |

### Provenance

- Spec: `.claude/specs/xl-bomd-linearized-stress/` (delete on close)
- Twin input: `tools/cp2k_oracle/inputs/xl-bomd-linearized-stress/h2_xl_linearized_stress_nve.inp`
- Queue: `docs/developer/cp2k-method-queue.md` row `xl-bomd-linearized-stress`
