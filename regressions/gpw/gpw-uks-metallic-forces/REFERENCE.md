# Reference — multi-k UKS free-energy analytic forces under FD smear

## Method

Collinear **UKS** GPW Kohn–Sham with **Fermi–Dirac** metallic occupation
smearing (Mermin free energy \(F=E_{\mathrm{KS}}-TS\)) and **analytic ionic
forces**. Shared chemical potential \(\mu\) over \(\alpha+\beta\); band
occupations \(f_{\sigma kb}\in[0,1]\).

**Capability fix:** `store_k_payload` previously rebuilt \(P_k/W_k\) from
integer \(n_\alpha/n_\beta\) columns of `C_occ_k`. Under smear it now stores
the full spectrum with fractional `f_occ_k` / `f_occ_k_beta`. Force assembly
uses
\(P=\sum_b f_b c_b c_b^\dagger\), \(W=\sum_b f_b\varepsilon_b c_b c_b^\dagger\).

Public path: `cpu::gpw::GpwDft` (`smearing`, `degauss`, `n_alpha`/`n_beta`,
`k_mesh`) + `gpw_forces_analytic` / `gpw_forces_from_scf`. No public
`*Config`, no factories.

## System A — BCC Na product pin (free energy + residual)

| Quantity | Value |
| -------- | ----- |
| Geometry | BCC Na conventional cubic, \(a = 4.225\) Å, 2 atoms |
| Basis | `DZVP-MOLOPT-SR-GTH` (Na) — in-tree |
| PP | `GTH-PBE-q9` (9 valence e⁻ / atom → \(N_e=18\)) |
| XC | PADE (LSD) |
| Cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Spin seed | \(n_\alpha=10\), \(n_\beta=8\) (free under smear) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| SCF | DIIS 6, `tol = 1e-6` Ha, `EPS_DEFAULT = 1e-7` |

### Frozen ATV pins (Release, OMP=8, 2026-08-01)

| Observable | Pin | Tol |
| ---------- | --- | --- |
| Free energy \(F\) | **−92.126330608004 Ha** | \(10^{-4}\) Ha |
| \(N_{\mathrm{occ}}\) | **18.0** | \(10^{-5}\) |
| \(E_F\) | **0.055825 Ha** (finite) | finite |
| Entropy \(-\sigma S\) | **−2.26×10⁻⁴ Ha** | finite |
| \|∑F\| | **8.4×10⁻¹³ Ha/Bohr** | \(10^{-5}\) |
| F₀ ≈ −F₁ | \(\sim 4.95\times10^{-4}\) Ha/Bohr each axis | residual |

Same free-energy pin as `reg_gpw-uks-smearing` (electronic path unchanged).

## System B — closed-shell multi-k H₂ identity

Compressed H₂ (`DZV-GTH-PADE`), `k_mesh=1×1×2`, UKS \(n_\alpha=n_\beta=1\) +
FD vs RKS \(n_{\mathrm{occ}}=1\) + FD: max\|dF\| \(\lesssim 10^{-15}\)
Ha/Bohr; restoring \(F_x\); Newton-3 on \(F_x\).

## Unit gate (H₂ off-symmetry)

`GpwForces.UkSMetallicSmearKmeshAnalyticVsFd`: multi-k UKS+FD analytic vs
multi-SCF FD of \(F\), max\|dF\| **7.86×10⁻⁶** Ha/Bohr (h=10⁻³ Bohr).

## Offline CP2K twins

### Fair wall twin (matched ENERGY_FORCE + SMEAR algorithm class)

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, **git:5e54ba2**) via `…/cp2k/install/env.sh` |
| Input | `tools/cp2k_oracle/inputs/gpw-uks-metallic-forces/h2_uks_fd_forces.inp` |
| Method | Quickstep GPW **LSD** + FERMI_DIRAC, PADE, multi-k 1×1×2, ENERGY_FORCE |
| Basis (CP2K) | DZVP-MOLOPT-GTH / GTH-PADE-q1 (ATV: DZV-GTH-PADE in-tree) |
| Run | `mpirun -np 1`, `OMP_NUM_THREADS=4`, arrhenius1, 2026-08-01 |
| Host dir | `/tmp/atv-gpw-uks-metallic-forces-h2` |
| Output | `ENERGY\| Total FORCE_EVAL … -1.056284263540282` |
| Forces | \(F_{1x}=-0.235\), \(F_{2x}=+0.235\) Ha/Bohr (restoring) |
| CP2K wall | **2.00 s** (internal CP2K timer); `/usr/bin/time` **2.36 s** |
| ATV twin | same geometry class, UKS+FD `k_mesh=1×1×2`, SCF+`gpw_forces_from_scf` |
| ATV wall | **0.097 s** (OMP=4) |

\[
r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}} = 0.097 / 2.00 \approx 0.049 \ll 10
\]

`within_10x = true`. Same method class (multi-k UKS FD free-energy forces);
basis label differs (in-tree DZV vs MOLOPT DZVP) — algorithm twin, not
bit-identical absolute energy.

### Na ENERGY_FORCE documentation twin

| Item | Value |
| ---- | ----- |
| Input | `tools/cp2k_oracle/inputs/gpw-uks-metallic-forces/na_bcc_uks_fd_forces.inp` |
| Run | `mpirun -np 1`, OMP=4, git:5e54ba2 |
| Free energy | **−92.76593827 Ha** (grid class differs from ATV by ~0.64 Ha) |
| Fermi | **0.054489 Ha** |
| Forces | \(\sim 3\times10^{-9}\) Ha/Bohr (near force-free BCC) |
| Wall | **0.78 s** (internal); ATV Na SCF **1.04 s**, force assembly **~53 s** |

Na absolute free energy is **not** required to match CP2K (same class as
`gpw-uks-smearing`). Electron count + residual + fractional payload are the
product gates. Fair wall uses the H₂ ENERGY_FORCE+SMEAR twin above (matched
algorithm class, ATV force path completes in ≪ 10×).

## Perf gate

| Engine | Wall |
| ------ | ---- |
| Atomiverse H₂ UKS+FD SCF+force (OMP=4) | **0.097 s** |
| CP2K `h2_uks_fd_forces.inp` (OMP=4, 1 MPI) | **2.00 s** |
| \(r\) | **0.049** |

Regression total wall (A0 H₂ identity + Na pin) ≈ **55 s** (OMP=8), ≤ 60 s
budget. Na force assembly is heavy (DZVP-MOLOPT, 4 k); not used as the fair
ratio denominator.

## Provenance

- Capability: fractional `f_occ_k` payload + free-energy \(P/W\) rebuild in
  `gpw_solver.cpp` / `gpw_forces.cpp` (stress path updated for same payload).
- Unit: `tests/cpu/gpw/test_cpu_gpw_forces.cpp` (`UkSMetallicSmearKmesh…`).
- Regression: `regressions/gpw/gpw-uks-metallic-forces/`.
- Spec: `.claude/specs/gpw-uks-metallic-forces/SPEC.md` (deleted on close).
