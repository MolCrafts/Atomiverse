# Plain multi-k Born–Oppenheimer MD on GPW — periodic H₂ NVE (DZV-GTH-PADE)

*(Technical-report page; narrative house style TBD — rewrite later. CP2K
capability loop, 2026-07-31.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-bomd-kpoints` |
| Regression | `regressions/gpw/gpw-bomd-kpoints/` |
| Public API | `cpu::md::MD` + `cpu::md::VelocityVerlet` + `cpu::gpw::GPWEvaluator{kinds, cutoff_ry, …, compute_forces=true}` with `k_mesh = {1,1,2}`, `k_time_reversal = false`; step-0 forces seeded by `cpu::single_point(frame, gpw)` |
| Oracle | **CP2K 2026.1** Quickstep AIMD (`RUN_TYPE MD` + `&KPOINTS`), offline via `tools/cp2k_oracle/inputs/gpw-bomd-kpoints/` |

## Summary

Plain Born–Oppenheimer MD on a Monkhorst–Pack k-mesh: the nuclei follow
NVE velocity-Verlet, and at **every** step the multi-k closed-shell RKS
GPW problem is re-converged to full SCF tolerance, with analytic
k-summed forces driving the integrator. No auxiliary electronic
propagator is involved — contrast `xl-bomd-kpoints` (extended-Lagrangian
density), `cpmd-kpoints` (fictitious orbital dynamics) and
`second-gen-cpmd-kpoints`. The implementation delta of this slug is the
**previous-step \(P_k\) warm start** on the plain multi-k MD path:
after a converged step, the complex densities \(\{P_k\}\) seed the next
SCF (the k-path analogue of the Γ `guess_C`/`guess_P` block; CP2K's
`EXTRAPOLATION USE_PREV_P`). Previously every plain multi-k BOMD step
was a cold SCF.

## Method

At each MD step \(n\) the electronic problem is solved self-consistently
on the sampled k set \(\{k\}\):

$$ F_k[\,\{P_{k'}\}\,]\,C_k = S_k C_k \varepsilon_k, \qquad
   P = \sum_k w_k\, \mathrm{Re}\, P_k , $$

then the nuclei advance by velocity Verlet on the k-summed analytic
force \(F_I = -\sum_k w_k\, \partial E_k / \partial R_I\). Spin
treatment here is restricted (RKS); the UKS multi-k warm start is
covered by the unit gate (the solver rejects a partial one-spin guess —
both \(P_k^\sigma\) are seeded or none). The mesh is MP 1×1×2
**unreduced** (`k_time_reversal = false`, 2 k points, \(w = 1/2\)),
matching CP2K `FULL_GRID .TRUE.`. XC is PADE (LDA); no dispersion, no
smearing, no thermostat (NVE). Not included: time-reversal-reduced
meshes in this twin, XL/CPMD propagators, k-point stress.

## Implementation

Public surface only — Frame in, Frame out; parameters are constructor
arguments / members of the concrete method (no `*Config` bags).

```cpp
// minimal call sequence — match regressions/gpw/gpw-bomd-kpoints/
cpu::Frame frame(2);                      // periodic H2, cubic 12 Bohr cell
// … fill kAtom_x/y/z, kAtom_type (Z=1), velocities; frame.box = …

cpu::gpw::GPWEvaluator gpw{kinds, /*cutoff_ry=*/300.0, /*…*/};
gpw.compute_forces    = true;
gpw.k_mesh            = {1, 1, 2};
gpw.k_time_reversal   = false;            // 2 k, w = 1/2 (CP2K FULL_GRID)
gpw.tol               = 1e-7;

cpu::single_point(frame, gpw);            // VV bootstrap: step-0 forces

cpu::md::MD md;
md.set_integrator(cpu::md::VelocityVerlet{0.5});   // fs
md.set_evaluator_ref(gpw);
md.build();
md.run(frame, 4);                         // 4 NVE steps
```

```bash
cmake --build build --target reg_gpw-bomd-kpoints
./build/regressions/reg_gpw-bomd-kpoints
```

Sources: `src/cpu/gpw/` (evaluator MD path + \(P_k\) warm start),
`include/atomiverse/cpu/gpw/`. Unit gate: `test_cpu_gpw_bomd`
(same-geometry relaunch converges in strictly fewer iterations to the
same fixed point, RKS and UKS multi-k).

## Validation (vs CP2K)

Source: `regressions/gpw/gpw-bomd-kpoints/REFERENCE.md` — offline pins,
never a live oracle at ctest time. Twin: periodic H₂ (bond 1.0 Bohr,
cubic 12 Bohr cell), DZV-GTH-PADE / GTH-PADE-q1 / PADE, CUTOFF 300 Ry
NGRIDS 1, MP 1×1×2 unreduced, NVE dt 0.5 fs × 4 steps, ATV `tol = 1e-7`
⇔ CP2K `EPS_SCF 1e-7`. CP2K 2026.1 (`cp2k.psmp`), single rank, OMP=8,
2026-07-31.

| Quantity | Atomiverse | CP2K pin | Tol | Notes |
| -------- | ---------: | -------: | --- | ----- |
| Step-0 E_pot [Ha] | −1.074456793 | −1.074050184 | 1e-3 | Δ = 4.07e-4 — Γ-point periodic-GPW model band (see below) |
| Step-0 \(F_x\) atom 0 [Ha/Bohr] | −0.339676 | −0.345535381 | 8e-3 | Δ = 5.86e-3; analytic ≡ FD to 6e-6 (`gpw-kpoint-forces`) |
| E_pot step 1 [Ha] | trace | −1.097682104 | 2e-3 | worst Δ over steps 1–4 = 2.33e-4 |
| E_pot step 2 [Ha] | trace | −1.125544577 | 2e-3 | |
| E_pot step 3 [Ha] | trace | −1.130531455 | 2e-3 | |
| E_pot step 4 [Ha] | trace | −1.121507812 | 2e-3 | |
| NVE drift (rel.) | 4.25e-3 | 4.4e-3 | ≤ 0.05 | dt = 0.5 fs is deliberately aggressive for this compressed bond |
| Multi-k payload | `n_kpts == 2`, converged every step | — | exact | full \(P_k\), warm start never worse (8 ≤ 8 iters; CP2K 14 cold → 8–9 warm) |

The Γ control (600 Ry) shows *identical* energy/force deltas
(4.07e-4 / 5.85e-3), so the residual is the pre-existing Γ-point
periodic-GPW model band on this compressed periodic H₂, not a k-point
defect — the multi-k machinery contributes < 1e-5 on top.

## Performance

Source: `regressions/gpw/gpw-bomd-kpoints/REFERENCE.md` (fair wall,
same host, single rank, `OMP_NUM_THREADS=8`).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| H₂ NVE, 1 single point + 4 MD steps, CUTOFF 300, 2 k | 33.4 | 2.14 | **0.064** | ✓ | ATV DIIS vs CP2K direct-P mixing; same SCF tol |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\); no exemption.

## Limitations

- Converged-limit energy/force parity vs CP2K sits on the Γ-point
  periodic-GPW model band (4e-4 Ha / 6e-3 Ha/Bohr on this system);
  deeper Γ force parity is a queue follow-on, not a k-point item.
- Warm start is \(P_k\) reuse only — no multi-step density
  extrapolation (ASPC on the multi-k path is a follow-on).
- Twin uses an unreduced mesh; time-reversal-reduced BOMD is untested
  against CP2K here.

## References

- Offline pin: `regressions/gpw/gpw-bomd-kpoints/REFERENCE.md`
- Twin input: `tools/cp2k_oracle/inputs/gpw-bomd-kpoints/h2_bomd_kpoints_nve.inp`
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP), BOMD chapter
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) (Quickstep GPW)
- Monkhorst & Pack, *Phys. Rev. B* **13**, 5188 (1976)
- Siblings: `gpw-bomd-water` (Γ), `gpw-kpoint-forces`, `xl-bomd-kpoints`, `cpmd-kpoints`, `second-gen-cpmd-kpoints`
