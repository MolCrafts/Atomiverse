# Reference — plane-wave BOMD foothold (H atom NVE)

## Method

Born–Oppenheimer molecular dynamics (BOMD) with plane-wave DFT forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::pw::PWEvaluator` with `with_gradient(true)`
  (analytic local Hellmann–Feynman + Ewald). The pseudopotential is the
  published analytic **GTH-PBE-q1** parameterization for hydrogen, evaluated
  in-tree from `cpu/gpw/basis_data/gpw_tables.h` — no external file. That
  entry carries no nonlocal projectors, so the KB term is legitimately zero
  here (it is not "switched off").
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

Literature basis for the **observable** (total-energy conservation on NVE when
\(\mathbf{F} \approx -\nabla E\)):

- Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.3.4
  (Velocity Verlet is symplectic; energy error bounded/oscillatory).
- Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §3.2.
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — BOMD chapter:
  energy conservation is the primary short-trajectory diagnostic when the
  electronic structure is solved self-consistently each step.

**Symmetry pin** for residual force (same as `regressions/pw/h_force_analytic`):
a single ion at a cubic high-symmetry site has \(\mathbf{F}=\mathbf{0}\) by
lattice symmetry — a statement about the lattice, independent of which
pseudopotential is used. Residual from short SCF is bounded by
\(|F| < 0.5\) eV/Å.

This is **not** a golden snapshot of Atomiverse total energy. The pin is the
**envelope** on relative drift that a working force↔energy binding must obey.

## System

| Quantity | Value |
| -------- | ----- |
| Species | 1 × H (Z = 1) at the origin |
| Mass | 1.008 amu |
| Cell | cubic \(L = 6\) Å |
| Pseudopotential | `gth_name = "GTH-PBE-q1"` (analytic GTH; same foothold as `h_force_*`) |
| \(E_\mathrm{cut}\) | 5 Ry |
| k-mesh | Γ-only (`k_mesh = 1`) |
| SCF | `tol = 1e-4`, `max_iter = 12`, `mix_beta = 0.5` |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | \(v_x = 0.002\) Å/fs (hand-set; `InitVelocity` COM-zeros N=1) |
| Steps | 1 warm-up + 10 production (chunked) |

**Why not H₂?** At the 5 Ry cutoff this slug uses, the H 1s state is far
from represented, so a two-proton cell has no meaningful bonding curve and a
short NVE traj is not a useful energy-drift pin. The product force math is
already covered by
`h_force_analytic` / FD unit tests on dimers; this slug pins **MD binding +
NVE conservation** on the symmetry-stable single-H foothold. A production
Si/UPF BOMD traj is a follow-on once `pw-lda-si` re-pins bulk energy.

## Offline oracle (CP2K)

**Status:** no `cp2k` / `cp2k.psmp` / `cp2k.sopt` binary and no CP2K Lmod
module on the worker host used for this pin (Arrhenius login-class node,
2026-07-23 inventory). A CP2K twin would be a short NVE AIMD for a single H
(or H₂ with a real PP) in a comparable cell; when a binary is available,
re-run offline via `tools/cp2k_oracle/` and replace this section with version
string + measured drift envelope.

Until then the regression asserts the **literature NVE/BOMD conservation
envelope** and the **symmetry force residual**, not a CP2K absolute energy.

## Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) taken
after a single warm-up step (energies populated).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.05
\]

Additional gates:

- PE finite; KE finite and \(\ge 0\)
- All coordinates finite
- \(|F| < 0.5\) eV/Å after warm-up (symmetry residual)

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K (or QE) binary on host for a fair wall twin |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

When a CP2K twin becomes available, record wall times for the same step
count / comparable settings and require \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}
\le 10\) or document a blocked-speed status after one optimize pass.

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_pw-bomd-h` asserts A4–A6 of
  `.claude/specs/pw-bomd-h/spec.md` |
| Speed | `perf_exempt=true` (no CP2K binary) |
