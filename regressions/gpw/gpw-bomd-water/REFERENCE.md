# Reference — GPW BOMD short trajectory (H₂O NVE)

## Method

Born–Oppenheimer molecular dynamics (BOMD) with Gaussian-and-plane-waves
(GPW / Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`
  (analytic Hellmann–Feynman + Pulay + grid/core; product path)
- Fixed cubic cell (GPW rejects box changes after `build()`)
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

Literature basis for the **observable** (total-energy conservation on NVE when
\(\mathbf{F} \approx -\nabla E\)):

- Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.3.4
  (Velocity Verlet is symplectic; energy error bounded/oscillatory).
- Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §3.2.
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — BOMD chapter:
  energy conservation is the primary short-trajectory diagnostic when the
  electronic structure is solved self-consistently each step.
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW
  Quickstep energy/force framework (same method family as CP2K).

**Force quality** for this geometry/method is already offline-pinned vs
**CP2K 2026.1** in `regressions/gpw/gpw-forces-water` (max |ΔF| ≈ 3.4×10⁻⁴
Ha/Bohr). This slug pins **MD binding + NVE conservation** on the product
`GPWEvaluator` path, not a second absolute force table.

This is **not** a golden snapshot of Atomiverse total energy. The pin is the
**envelope** on relative drift that a working force↔energy binding must obey.

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT |
| Geometry | CP2K getting-started (same as `gpw-forces-water`) |
| Masses | O 15.999 amu; H 1.008 amu |
| Cell | cubic \(L = 10\) Å; `PERIODIC NONE` / `nonperiodic = true` |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | **200 Ry** (BOMD envelope; forces pin uses 400 Ry) |
| SCF | `tol = 1e-5`, `max_iter = 60`, `mix = 0.3` |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | COM-preserving H stretch \(v_x(\mathrm{H1})=+0.0015\), \(v_x(\mathrm{H2})=-0.0015\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

| Atom | x | y | z (Å) |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

**Why cheaper cutoff than the forces pin?** One ENERGY_FORCE at 400 Ry is
~40 s wall on login-class ATV (see `gpw-forces-water` REFERENCE). A multi-step
BOMD at that cost exceeds the ≤60 s regression gate. 200 Ry keeps the same
method family (PBE / MOLOPT / GTH / MT / analytic forces) while staying inside
the time budget; absolute E/F parity remains the forces-pin’s job.

## Offline oracle (CP2K)

**Status:** no `cp2k` / `cp2k.psmp` / `cp2k.sopt` binary and no CP2K Lmod
module on the worker host used for this pin (Arrhenius login-class node,
2026-07-23 inventory). A CP2K twin would be short NVE AIMD / MD with
Quickstep GPW on the same H₂O cell (see suggested input under
`tools/cp2k_oracle/inputs/` when added); when a binary is available, re-run
offline via `tools/cp2k_oracle/` and replace this section with version string
+ measured drift envelope + fair wall.

Until then the regression asserts the **literature NVE/BOMD conservation
envelope** and finite/bounded trajectory diagnostics, not a CP2K absolute
trajectory energy series. Force twin remains `gpw-forces-water` (CP2K 2026.1).

## Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) taken
after a single warm-up step (energies populated).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.05
\]

Additional gates:

- PE finite and bound (\(< -100\) eV after warm-up for PBE water)
- KE finite and \(\ge 0\), non-trivial after warm-up
- All coordinates finite
- Soft O–H bond ceiling: both \(r_{\mathrm{OH}} < 1.8\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step AIMD |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

When a CP2K twin becomes available, record wall times for the same step
count / comparable settings and require \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}
\le 10\) or document a blocked-speed status after one optimize pass.

Note: single-point force wall vs CP2K already measured in
`gpw-forces-water` (ratio **3.56**, within_10x).

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_gpw-bomd-water` asserts A4–A6 of
  `.claude/specs/gpw-bomd-water/spec.md` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |

**ATV wall (login-class, OMP=4, 2026-07-23):** ≈ **35 s** for 1 warm-up + 4
production steps at CUTOFF 200 Ry — within A10 (≤ 60 s). Measured relative
drift \(\delta \approx 1.2\times 10^{-5}\) ≪ 0.05.
