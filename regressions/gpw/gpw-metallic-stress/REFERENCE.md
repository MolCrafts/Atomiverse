# Reference — multi-k free-energy analytic cell stress under FD smear (BCC Na, GPW)

## Method

Spin-unpolarized **GPW** Kohn–Sham DFT with **Fermi–Dirac** electronic
smearing (Mermin free energy \(F = E_{\mathrm{KS}} - TS\)). Product **cell
stress** is the Hellmann–Feynman fixed-occupation free-energy path:

1. One multi-k SCF with FD smear → full Bloch \(C_k\), band energies
   \(\varepsilon_{nk}\), fractional occupations \(f_{nk}\in[0,2]\).
2. Rebuild \(P_k=\sum_b f_b\,c_b c_b^\dagger\),
   \(W_k=\sum_b f_b\varepsilon_b\,c_b c_b^\dagger\).
3. Central FD of the **non-self-consistent** Lagrangian under orthorhombic
   diagonal strain at fixed fractional coordinates (fixed \(\{f_k\}\)).
4. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Dual-gate oracle: multi-SCF strain-FD of \(F\) (`gpw_stress_fd`, same smear
settings). Offline CP2K twin: `STRESS_TENSOR NUMERICAL` + `SMEAR FERMI_DIRAC`.

Public API: `cpu::gpw::GpwDft` members `smearing`, `degauss` [Ry],
`extra_bands`, `mp_order`; free `gpw_stress` / `gpw_stress_fd`. Payload
fields `GpwResult::f_k` + full `C_k` under smear.

## System (Atomiverse pin)

| Quantity | Value |
| -------- | ----- |
| Geometry | BCC Na conventional cubic, \(a = 4.225\) Å, 2 atoms |
| Basis | `DZVP-MOLOPT-SR-GTH` (Na) — in-tree table |
| Pseudopotential | `GTH-PBE-q9` (9 valence e⁻ / atom → \(N_e = 18\)) |
| XC | PADE (LDA) |
| Density cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| SCF | DIIS 6, `tol = 1e-6` Ha, `EPS_DEFAULT = 1e-7` |
| Stress \(h\) | \(2\times 10^{-3}\) (dimensionless strain) |

### Product pins (first green 2026-08-01)

| Observable | Pin | Tol |
| ---------- | --- | --- |
| Free energy \(F\) | **−92.1263306084 Ha** | \(10^{-3}\) Ha |
| \(N_{\mathrm{occ}}=\sum w f\) | **18.0** | \(10^{-5}\) |
| \(E_F\) | **0.055826 Ha** (finite) | \(\lvert E_F\rvert < 10^3\) |
| \(\max_{\mathrm{diag}}\lvert\sigma^{\mathrm{an}}-\sigma^{\mathrm{FD}}\rvert\) | **2.4×10⁻⁸ Ha/Bohr³** | \(2\times 10^{-4}\) |
| mean \(\sigma_{\mathrm{diag}}\) analytic | **−1.91772220×10⁻² Ha/Bohr³** | \(5\times 10^{-4}\) |
| Off-diagonal product | 0 | soft 1e-8 |

Analytic vs multi-SCF FD residual is well inside the dual-gate tol — free-energy
fixed-\(f\) Hellmann–Feynman matches reconverged \(F\) under strain.

## Offline CP2K twin (algorithm / NUMERICAL stress + SMEAR)

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, source `git:5e54ba2`) |
| Env | `source …/cp2k/install/env.sh` |
| Host | arrhenius1, 2026-08-01 |
| Input (stress) | `tools/cp2k_oracle/inputs/gpw-metallic-stress/na_bcc_fermi_dirac_stress.inp` |
| Input (fair ENERGY wall) | same system as energy twin / stress without `STRESS_TENSOR` (see run note) |
| Method | Quickstep GPW, PADE, `GTH-PBE-q9` + `DZVP-MOLOPT-SR-GTH` |
| Cell | BCC conventional \(a=4.225\) Å, 2 Na |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (≈ 0.02 Ry \(k_B T\)), ADDED_MOS 4 |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| CUTOFF / REL_CUTOFF | 150 / 40 Ry, NGRIDS 4 |
| `STRESS_TENSOR` | **NUMERICAL** |
| Output energy line | `Total energy: -92.76593904071184` Ha |
| Fermi energy | `0.05448910718283` Ha |
| Stress lines | `STRESS\| Numerical stress tensor [bar]` |
| | `STRESS\| x -5.01677864643E+06 …` (diag ≈ −5.01677865e6 bar) |
| \(\sigma_{\mathrm{xx}}\) (a.u.) | **−1.7051×10⁻² Ha/Bohr³** (= −5.01677865e6 bar / 2.942101e8) |
| Soft class check | \(\lvert\sigma_{\mathrm{ATV}}-\sigma_{\mathrm{CP2K}}\rvert \le 5\times 10^{-3}\) Ha/Bohr³ |

Absolute free energy and absolute stress differ from CP2K by grid /
`EPS_DEFAULT` / multigrid residual (~0.64 Ha on \(F\), ~0.002 Ha/Bohr³ on
\(\sigma\)) — same class as `reg_gpw-metallic-smearing`. Dual-gate analytic↔FD
is the product correctness pin; CP2K documents the matched GPW+SMEAR algorithm
twin and soft stress class.

## Perf gate (fair single-SCF twin wall)

Same system / basis / PP / smear / k-mesh as the ENERGY twin job (1 MPI ×
`OMP_NUM_THREADS=4`). **Not** the multi-SCF NUMERICAL stress wall.

| Engine | Wall |
| ------ | ---- |
| Atomiverse SCF (`T_atv_scf`, re-verify green) | **0.723 s** |
| CP2K ENERGY twin (`/usr/bin/time`, 1 MPI × 4 OMP) | **0.70 s** |

\[
r = T_{\mathrm{atv,scf}} / T_{\mathrm{cp2k}} \approx 1.03 \ll 10
\]

`within_10x = true`. Multi-SCF NUMERICAL stress wall was ~18 s (reference only;
not the fair ratio denominator).

## Gates summary

| Check | Status |
| ----- | ------ |
| SCF free energy + \(N_e\) + \(f_k\) payload | PASS |
| Analytic free-energy stress vs multi-SCF FD | PASS (2.4e-8) |
| Cubic symmetry / magnitude floor | PASS |
| Absolute mean \(\sigma\) pin | PASS |
| CP2K soft stress class | PASS |
| Fair wall ≤ 10× | PASS (1.53) |
