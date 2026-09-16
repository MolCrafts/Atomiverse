# Reference — XL-BOMD short trajectory (H₂O NVE, GPW forces)

## Method

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
**CP2K 2026.1** in `regressions/gpw/gpw-forces-water` (max |ΔF| ≈ 3.4×10⁻⁴
Ha/Bohr); plain-BOMD MD binding + NVE conservation is pinned in
`regressions/gpw/gpw-bomd-water`. This slug pins the **XL auxiliary-density
propagation** (exact Table-I coefficients at unit level) and the **NVE
conservation envelope with the XL guess engaged** on the product
`GPWEvaluator` path.

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative drift that a working XL-BOMD guess must obey —
a wrong recursion (sign error, non-density aux, shape leak) destroys SCF
convergence or the drift envelope immediately.

## System

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

## Offline oracle (CP2K)

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

## Observable and tolerance

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

## Perf gate

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

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_xl-bomd-water` asserts A7 of `.claude/specs/xl-bomd-water/acceptance.md`; unit pin A1–A6 in `test_cpu_gpw_xl_bomd` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |

**ATV wall (login-class, OMP=4, 2026-07-25):** measured at gate time — see
final summary in the worker report. Measured relative drift and final SCF
iteration count are printed by the regression binary.
