# Reference — multi-k UKS Car–Parrinello MD short trajectory (periodic H₂ NVE, GPW)

## Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on **multi-k collinear UKS** Gaussian-and-plane-waves
(GPW / Quickstep) forces — the dual-spin Bloch orbital set \(C_k^\sigma\)
propagated per spin **and** per sampled k (the `cpmd-kpoints` ×
`cpmd-uks` crossing):

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 1`**, **`n_beta = 1`** (closed-shell fillings on the UKS
  path — both spin channels propagate and H₂ stays bound; triplet H₂ is
  repulsive and cannot pin NVE, the `xl-bomd-uks-kpoints` precedent),
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  wrapped in `cpu::electronic::CarParrinello{gpw, μ}`; the electronic dt is
  pushed by `cpu::md::MD` from the `VelocityVerlet` integrator
  (`TimestepProvider` / `TimestepConsumer`) at build
- One `cpu::electronic::CpmdCplx` per **(k, spin)** propagates occupied
  Bloch coefficients with electronic RATTLE
  (\(C_k^{\sigma\dagger} S_k C_k^\sigma = I\)); production steps use the
  dual-spin fixed-density multi-k KS eval (`iters == 0`,
  `H_ks_k` + `H_ks_k_beta`), spin-diagonal EOM occupancy 1:
  \(\mu \ddot C_k^\sigma = -(I - S_k C_k^\sigma C_k^{\sigma\dagger})
  (2 H_k^\sigma C_k^\sigma) + \text{RATTLE}\)
- Conserved diagnostic:
  \(E_{\mathrm{cons}} = T_n + T_e + E_{\mathrm{KS}}\) with
  \(T_e = \sum_k w_k \sum_\sigma \tfrac12\mu\lVert\dot C_k^\sigma\rVert_F^2\)
- Periodic cubic cell; PADE (LSD) XC
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:**

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — electronic RATTLE
- Parent products: `regressions/dynamics/cpmd-kpoints` (multi-k RKS Bloch CP),
  `regressions/dynamics/cpmd-uks` (Γ dual-spin CP)
- Multi-k UKS BO surface pins: `regressions/gpw/gpw-kpoints-uks`,
  `gpw-kpoint-uks-forces` (CP2K-anchored energies/forces)
- Multi-k UKS AIMD family twin: `regressions/dynamics/xl-bomd-uks-kpoints`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative \(E_{\mathrm{cons}}\) drift with dual-spin
multi-k Bloch CP engaged, plus a **UKS-vs-RKS closed-shell consistency
identity** — a wrong \(C_k^\sigma\) seed, a missing β channel, a real-only
force path, or SCF on production steps destroys one of them immediately.

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | collinear UKS, \(n_\alpha = n_\beta = 1\) (closed-shell fillings) |
| XC | PADE (LSD) |
| Basis | DZV-GTH-PADE |
| PP | GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run; grid series in `gpw-bomd-kpoints`) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF seed | `tol = 1e-5`, `max_iter = 60`, DIIS |
| CPMD | \(\mu = 400\) a.u.; production fixed-density dual-spin |
| Ensemble | NVE |
| `dt` | 0.1 fs (nuclear = electronic) |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 cold SCF seed + 1 CP baseline + 4 production |

## Offline oracle

**Family policy (`cpmd-kpoints` / `cpmd-uks`, queue-assigned): literature
NVE conservation envelope.** The gate is a *method-family lock*, not a
missing binary: CP2K 2026.1 builds and runs on this host (it pinned
`gpw-bomd-kpoints` on 2026-07-31), but CP2K has **no fictitious-mass
Car–Parrinello propagator** — its `RUN_TYPE MD` is Born–Oppenheimer and its
CP-like method is second-generation CPMD (ASPC + corrector). The CP
conserved quantity \(T_n + T_e + E_{\mathrm{KS}}\) with the fictitious
\(T_e[\dot C_k^\sigma]\) is not a number CP2K can print for any input.

The **BO surface underneath is CP2K-anchored** by the shipped multi-k UKS
pins (`gpw-kpoints-uks` energies, `gpw-kpoint-uks-forces` forces, both vs
offline CP2K Quickstep LSD + KPOINTS), and this regression additionally
asserts the closed-shell identity below against the multi-k **RKS** path
that `gpw-bomd-kpoints` pinned against CP2K 2026.1 directly.

## Observable and tolerance

Let \(E(t) = E_{\mathrm{KS}}(t) + T_n(t) + T_e(t)\) [eV], with \(E_0\) after
the first CP production step (\(T_e\) engaged).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- **UKS-vs-RKS closed-shell consistency:** dedicated multi-k UKS
  (`n_alpha = n_beta = 1`) and multi-k RKS (`n_occ = 1`) single points
  (`cpu::single_point`, plain `GPWEvaluator`, same geometry/mesh/cutoff,
  `tol = 1e-7`) agree within **3×10⁻⁴ eV** ≈ 1.1×10⁻⁵ Ha (the shipped
  `gpw-kpoints-uks` A2 identity holds to 5×10⁻⁶ Ha at this tol; at the MD
  envelope `tol = 1e-5` the residual is convergence noise)
- Production SCF **iters == 0** (dual-spin fixed-density multi-k eval)
- `n_kpts >= 2` after seed
- \(T_e > 0\) once the electrons are propagated (dual-spin sum)
- PE finite and bound (\(< -5\) eV after warm-up); KE finite, \(\ge 0\),
  non-trivial after warm-up
- All coordinates finite
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | Oracle family lock: no engine on host implements fictitious-mass multi-k UKS CPMD (CP2K 2026.1 has no CP propagator — BOMD/SGCP only), so no fair same-method wall twin exists |
| `ratio_atv_over_oracle` | n/a |
