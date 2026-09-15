# DFPT phonons at Γ, joint collinear spin × pure GGA — diamond Si (PW PBE, NC-UPF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-gga-spin-si` |
| Regression | `regressions/response/dfpt-gga-spin-si/` |
| Public API | `cpu::pw::DfptPhonon` (members) + `cpu::pw::DfptPhononResult`; `PW.xc = XCType::PBE` + `PW.spin_polarized = true` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — **joint GGA × collinear spin** ✓ (`reg_dfpt-gga-spin-si`) |
| Dep | `dfpt-gga-si` ✓ (restricted PBE kernel); `dfpt-spin-si` ✓ (collinear LSDA kernel) |

## Summary

The two parent products pin each axis of the DFPT XC kernel alone: a **true
pure-GGA kernel** on the restricted path (`dfpt-gga-si`) and a **collinear
spin** kernel for LSDA (`dfpt-spin-si`). This product pins the **joint**
path: spin-resolved Sternheimer response where
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) is the second functional derivative of
the **spin-GGA** operator, obtained by central finite difference of
`XCBuilder::step_spin` (spectral \(\nabla\rho_\sigma\) + GGA flux) around the
frozen spin ground state. Product system: diamond silicon, 2-atom fcc
primitive cell, minimal twin (10 Ry, \(2\times2\times2\) MP, in-tree NC UPF),
`spin_polarized = true`, multiplicity 1 (\(M=0\)). Public surface: concrete
`cpu::pw::PW` members + `cpu::pw::DfptPhonon` members — no `*Config`, no
factories. Dual gate: the **offline same-Hamiltonian spin-PBE energy second
difference** (sharp) plus a **CP2K 2026.1 UKS-PBE k-point
`VIBRATIONAL_ANALYSIS`** physics anchor (first DFPT-family slug with a real
CP2K twin and a fair wall). Cite: Baroni et al., *Rev. Mod. Phys.* **73**,
515 (2001); Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

## Method

Per Cartesian displacement, the Sternheimer equation is solved per spin
channel with spinor occupations \(f\in[0,1]\), and the induced spin densities
are screened jointly:

\[
\Delta V_{\mathrm{Hxc},\sigma}[\Delta n_\uparrow,\Delta n_\downarrow]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + \sum_{\sigma'} K_{\mathrm{xc}}^{\sigma\sigma'}[\Delta n_{\sigma'}],
\]

with Hartree from the total \(\Delta n\). Pure LSDA keeps the local
\(f_{\mathrm{xc}}^{\sigma\sigma'}\) table; **pure GGA (this slug)** applies
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) by central FD of the ground-state
spin-GGA \(V_{\mathrm{xc},\sigma}[n_\uparrow,n_\downarrow]\) along the
induced spin-density direction (`apply_xc_kernel_spin_fd`,
`src/cpu/planewave/internal/dfpt.cpp`). Hybrids are rejected. At \(M=0\) the
joint kernel must reproduce the restricted-GGA parent — a consistency limit
the regression checks against both parents' pins.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Spin | collinear, `spin_polarized = true`, multiplicity 1 (\(N_\uparrow=N_\downarrow=4\)) |
| XC | **PBE** pure spin-GGA kernel (`XCType::PBE`) |
| Structure | diamond Si, 2-atom fcc primitive, \(a = 5.43\) Å |
| PP | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| Cutoff | **10 Ry**; k-mesh **2×2×2** MP |
| SCF | density L2 tol \(10^{-8}\), linear mix on spin channels |
| Response | `response_tol` \(10^{-5}\), `mix_beta` 0.7, CG tol \(10^{-9}\) |
| \(\mathbf{q}\) | Γ (`q_frac = {0,0,0}`) |

**Key exclusions:** finite-q GGA is shipped as `dfpt-gga-finite-q-si`; the
joint finite-q × spin × GGA product is `dfpt-gga-spin-finite-q-si`. Hybrids,
\(M\neq0\) cells, CUDA, live CP2K in ctest.

## Implementation

```cpp
PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry  = 10.0;
method.k_mesh   = 2;
method.xc       = cpu::scf::XCType::PBE;   // pure GGA kernel
method.spin_polarized = true;              // joint spin × GGA response
method.multiplicity   = 1;
method.tol      = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.response_tol = 1e-5;
dfpt.cg_tol       = 1e-9;
dfpt.mix_beta     = 0.7;
const DfptPhononResult r = dfpt.compute(method, frame);
```

Build and run (scoped):

```bash
cmake --build build -j"$(nproc)" --target atomiverse reg_dfpt-gga-spin-si
cmake --build build --target run_reg_dfpt-gga-spin-si
```

## Validation (vs offline pins)

Source: `regressions/response/dfpt-gga-spin-si/REFERENCE.md`. Measured
2026-07-31 (Release, OMP=1):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Energy-FD \(C[(0x),(0x)]\) [eV/Å²] (spin-PBE SCF) | **17.3454** (hard-coded) | oracle for DFPT \(C_{00}\) | \(h=0.03\) Å; \(E_0=-7.820593875040\) Ha |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **17.3711** | \(\|C_{00}-17.3454\|\le0.30\) (~0.15 %) | = restricted parent 17.3711 |
| Acoustic \(\omega\) [cm⁻¹] | 12.09 ×3 | \(\|\omega\|\le30\) | no ASR imposition |
| Optical triplet [cm⁻¹] | **579.82, 579.82, 579.96** | each \(\in[450,650]\); spread \(\le10\) | = restricted parent |
| CP2K 2026.1 anchor [cm⁻¹] | 463.44, 552.19, 552.19 | inside the same window | UKS-PBE k-point FD vib, GTH/MOLOPT twin |
| Max row sum [eV/Å²] | 0.0151 | \(\le0.05\) | translational invariance |
| Pre-symmetrization asymmetry [eV/Å²] | \(9.6\times10^{-10}\) | \(\le10^{-5}\) | Hermitian diagnostic |
| Wall | **44.7 s** | ≤ 60 s | 6 responses × 2 spins, 8 k |

Cross-parent consistency at \(M=0\): joint DFPT C00 and optical triplet are
4-decimal identical to `dfpt-gga-si`; the spin-FD energy pin 17.3454 sits
2e-4 from the restricted-FD 17.3456 — the joint kernel adds no spurious spin
response while exercising the full spin-resolved `step_spin` FD path.

## Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| Γ spin×PBE phonons Si (CP2K FD vib, `EPS_SCF 1e-8`, serial) | 221.8 | 44.7 | **0.20** | yes |

First DFPT-family slug with a real CP2K wall twin (binary
`~/.local/bin/cp2k`, CP2K 2026.1 rev `85b8a9b49c`) — the parents' stale
"no CP2K binary" `perf_exempt` reason no longer holds on this host.

## References

- Offline pins: `regressions/response/dfpt-gga-spin-si/REFERENCE.md`
- Regression: `regressions/response/dfpt-gga-spin-si/reg_dfpt-gga-spin-si.cpp`
- Oracle note + CP2K input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-si.note`,
  `tools/cp2k_oracle/inputs/dfpt-gga-spin-si/si2_uks_pbe_vib.inp`
- Parents: `docs/tutorial/methods/dfpt-gga-si.md`,
  `docs/tutorial/methods/dfpt-spin-si.md`
- Literature: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001); Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**,
  3865 (1996)
