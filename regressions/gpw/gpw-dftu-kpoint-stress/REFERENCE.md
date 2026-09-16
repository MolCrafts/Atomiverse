# Reference — multi-k UKS GPW + Dudarev DFT+U analytic cell stress

## Method

Unrestricted Kohn–Sham GPW with **Monkhorst–Pack k-point sampling**,
**Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken), and **Hellmann–Feynman
fixed-\(\{P_k^\sigma,W_k^\sigma\}\)** analytic cell stress:

1. One multi-k UKS+DFT+U SCF → occupied Bloch MOs and band energies
   (eigenvalues include \(V_U\)) → rebuild \(P_k^\sigma\), \(W_k^\sigma\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[\{P_k^\sigma\},\{W_k^\sigma\};\varepsilon]\) under orthorhombic
   diagonal strain at fixed fractional coordinates, with BZ-averaged
   \(E_U[n^\sigma(\{P_k^\sigma\},\{S_k\})]\) at strained \(S_k\).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; Ha/Bohr³).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) on the same method.
Γ-limit identity: complex \(k=\Gamma\) + U stress ≡ real Γ + U stress.

Public API: `cpu::gpw::gpw_stress(atoms, method, h)` with `k_mesh`,
`n_alpha`/`n_beta`, `plus_u`. No public `*Config`, no factories.

## System

| Item | Value |
|------|--------|
| Molecule | H₂ high-spin (M = 3), compressed bond along *x* |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic **12 Bohr**, periodic XYZ |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| XC | PADE LSD |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| Cutoff | 100 Ry, `local_pp_on_grid` |
| k mesh | MONKHORST-PACK 1×1×2 (TR-folded, nk=1) |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) |
| Stress step | \(h=2\times 10^{-3}\) |

## Offline CP2K twin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`).
- **Input:** `tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/h2_uks_dftu_kpoint_stress.inp`
  (NUMERICAL stress + PLUS_U + KPOINTS; 8 Bohr cell variant for wall estimate).
- **Date:** 2026-07-26
- **Result:** same class of pathology as `gpw-dftu-kpoints` — **unphysical
  Mulliken charges**, large negative DFT+U, SCF fails to converge within
  MAX_SCF=100 (abort). **Not** used as absolute stress/energy twin.
- **Decisive gates:** internal dual-gate + Γ-limit identity + U activation.

```
  DFT+U energy: ~ −175 Ha (unphysical)
  Total energy: ~ −174 Ha
  *** SCF run NOT converged ***
```

## Atomiverse first-green (2026-07-26)

| Quantity | Value |
|----------|--------|
| **E (k-UKS+U 1×1×2, 100 Ry)** | **−0.546250995860 Ha** (76 iters, nk=1) |
| e_dftu | **−0.243601 Ha** (multi-function s shell; can be <0) |
| σ_xx analytic | −3.00982070e-04 Ha/Bohr³ |
| σ_yy analytic | −6.59874084e-05 Ha/Bohr³ |
| σ_zz analytic | −6.58998481e-05 Ha/Bohr³ |
| max\|σ_an−σ_fd\| | **9.33×10⁻⁴** Ha/Bohr³ |
| max\|σ_kΓ − σ_Γ\| | **1.3×10⁻⁸** Ha/Bohr³ |
| \|E(U)−E(0)\| | **1.79×10⁻²** Ha |
| Wall (OMP=8, Release) | **~2.4 s** |

## Tolerances

| Check | Tol |
| ----- | --- |
| max\|σ_an − σ_fd\| diag | **1×10⁻³** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| max\|σ_kΓ − σ_Γ\| | **5×10⁻⁴** Ha/Bohr³ |
| \|E − pin\| soft | **1×10⁻³** Ha |
| \|E(U)−E(0)\| floor | **1×10⁻⁴** Ha |
| max\|σ_diag\| floor | > **1×10⁻⁸** Ha/Bohr³ |

Bond-axis residual (xx) dominates the dual-gate; transverse axes agree at
~1e-6. Nonlinear Mulliken \(n(S)\) + large \|e_dftu\| on multi-function s
shells make the multi-SCF FD noisier than bare multi-k UKS (~1e-10). Γ-limit
identity proves the multi-k \(E_U\) Lagrangian path matches the shipped Γ
DFT+U stress.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No fair absolute CP2K NUMERICAL stress twin (unphysical Mulliken + SCF non-convergence on H₂+U multi-k toy; same class as `gpw-dftu-kpoints`). Dual-gate wall ~2.4 s is product+FD, not a fair oracle ratio. |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |

## Provenance

- Spec: `.claude/specs/gpw-dftu-kpoint-stress/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/h2_uks_dftu_kpoint_stress.inp`
- Deps: `gpw-dftu-kpoints` ✓, `gpw-dftu-stress` ✓, `gpw-kpoint-uks-stress` ✓
- Citations: Dudarev *et al.*, PRB **57**, 1505 (1998); Nielsen & Martin, PRB **32**, 3780 (1985).

## Re-pin 2026-07-31 — variational +U scatter

`gpw-dftu-kpoint-forces` fixed the Dudarev Fock contribution to the
variational form ∂E_U/∂P = ½(V̂S+SV̂) (the bare-V̂ scatter left the SCF fixed
point non-stationary for the reported E_U[n(P,S)]; CP2K uses the S-weighted
form). All +U fixed points moved; ATV-self absolute pins below were
re-measured on the new functional (identity/dual gates unchanged and tighter).

- E(k-UKS+U 1×1×2, 100 Ry): −0.546250995860 → **−0.734377348082 Ha**; max|σ_an−σ_fd| improved 9.3e-4 → **3.0e-9 Ha/Bohr³** (gate tightened to 1e-4).
