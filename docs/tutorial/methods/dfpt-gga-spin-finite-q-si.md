# DFPT phonons at finite q, joint collinear spin × pure GGA — diamond Si at X (PW PBE, NC-UPF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-gga-spin-finite-q-si` |
| Regression | `regressions/response/dfpt-gga-spin-finite-q-si/` |
| Public API | `cpu::pw::DfptPhonon` (members, `q_frac`) + `cpu::pw::DfptPhononResult`; `PW.xc = XCType::PBE` + `PW.spin_polarized = true` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — **finite-q × spin × pure GGA** ✓ (`reg_dfpt-gga-spin-finite-q-si`) |
| Queue | `docs/developer/cp2k-method-queue.md` → **done** (shift wave 3, 2026-07-31); fair ratio **0.19** |
| Dep | `dfpt-gga-spin-si` ✓ (Γ, spin × PBE); `dfpt-gga-finite-q-si` ✓ (X supercell, restricted PBE) |

## Summary

The two parent products pin each joint axis alone: **collinear spin × pure
GGA** at Γ (`dfpt-gga-spin-si`) and **finite-\(\mathbf{q}=X\)** with a
restricted GGA kernel (`dfpt-gga-finite-q-si`). This product pins the
**triple** path: commensurate X → 2×1×1 supercell Γ-DFPT + IFC Fourier where
every supercell Sternheimer response is spin-resolved and screened with
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) from central FD of the spin-GGA
`XCBuilder::step_spin` operator. Product system: diamond silicon, 2-atom fcc
primitive cell, minimal twin (10 Ry, Γ-only supercell mesh, in-tree NC UPF),
`spin_polarized = true`, multiplicity 1 (\(M=0\)), `XCType::PBE`. Public
surface: concrete `cpu::pw::PW` members + `cpu::pw::DfptPhonon` members — no
`*Config`, no factories. Dual gate: the **offline same-Hamiltonian supercell
energy second difference** (sharp; at \(M=0\) the restricted supercell pin
21.2006 eV/Å²) plus a **CP2K 2026.1 supercell UKS-PBE
`VIBRATIONAL_ANALYSIS`** physics anchor whose eigenvector phases identify
the folded X branch. Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515
(2001); Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991); Perdew, Burke,
Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

## Method

A phonon at rational \(\mathbf{q}=(h/n_a,k/n_b,l/n_c)\) is exactly the Γ
phonon of the \(n_a\times n_b\times n_c\) supercell folded back by
\(e^{i\mathbf{q}\cdot\mathbf{R}}\); for \(X=(1/2,0,0)\) the product is
spin-resolved Γ-DFPT on the 2×1×1 supercell (12 Sternheimer responses × 2
spins) followed by the IFC Fourier. Per Cartesian displacement the
Sternheimer equation is solved per spin channel and the induced spin
densities are screened jointly:

\[
\Delta V_{\mathrm{Hxc},\sigma}[\Delta n_\uparrow,\Delta n_\downarrow]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + \sum_{\sigma'} K_{\mathrm{xc}}^{\sigma\sigma'}[\Delta n_{\sigma'}],
\]

with Hartree from the total \(\Delta n\) and
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) from central FD of the ground-state
spin-GGA operator (`apply_xc_kernel_spin_fd` /
`dfpt_gamma_spin_impl` on the supercell; the direct q-path
`dfpt_q_spin_impl` carries the same kernel for incommensurate \(q\)). At
\(M=0\) the joint kernel must reproduce the restricted finite-q GGA parent
— a consistency limit the regression checks.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → commensurate **2×1×1** supercell (4 atoms) |
| Spin | collinear, `spin_polarized = true`, multiplicity 1 |
| XC | **PBE** pure spin-GGA kernel (`XCType::PBE`) |
| Structure | diamond Si, 2-atom fcc primitive, \(a = 5.43\) Å |
| PP | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| Cutoff | **10 Ry**; k-mesh **1×1×1** on the supercell (Γ+X folded) |
| SCF | density L2 tol \(10^{-8}\) (DFPT-internal spin SCF caps mix ≤ 0.4) |
| Response | `response_tol` \(10^{-4}\), `mix_beta` 0.25, CG tol \(10^{-8}\), max_iter 120 |

**Key exclusions:** incommensurate-q joint spin-GGA pin (shared code; separate
census decision), spin × GGA × finite-q × metallic smearing, hybrids, CUDA,
live CP2K in ctest.

## Implementation

```cpp
PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry  = 10.0;
method.k_mesh   = 1;
method.xc       = cpu::scf::XCType::PBE;   // pure GGA kernel
method.spin_polarized = true;              // joint spin × GGA response
method.multiplicity   = 1;
method.tol      = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0};             // X → 2×1×1 supercell Γ-DFPT
dfpt.response_tol = 1e-4;
dfpt.response_max_iter = 120;
dfpt.cg_tol = 1e-8;
dfpt.mix_beta = 0.25;
const DfptPhononResult r = dfpt.compute(method, frame);
```

Build and run (scoped):

```bash
cmake --build build -j"$(nproc)" --target atomiverse reg_dfpt-gga-spin-finite-q-si
cmake --build build --target run_reg_dfpt-gga-spin-finite-q-si
```

## Validation (vs offline pins)

Source: `regressions/response/dfpt-gga-spin-finite-q-si/REFERENCE.md`.
Measured 2026-07-31 (Release; OMP=1):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Energy-FD \(C[(0x),(0x)]\) [eV/Å²] (supercell, M=0) | **21.2006** (hard-coded) | oracle for DFPT \(C_{00}\) | same-Ham. pin; \(M=0\) ≡ restricted supercell parent |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **21.1911** | \(\|C_{00}-21.2006\|\le0.30\) (~0.04 %); vs restricted parent 21.1974 \(\le0.05\) | supercell diagonal |
| \(\omega(X)\) [cm⁻¹] | −108.83, −108.32, 239.12, 451.33, 668.84, 668.90 | \(\|\omega\|\) low 3 ∈ [50, 400]; high 3 ∈ [300, 850] | soft TA pair ≤ 2 imaginary (minimal twin) |
| CP2K 2026.1 X anchor [cm⁻¹] | 97.98, 131.27, 370.58, 428.24, 457.48, 458.68 | inside the same windows | supercell UKS-PBE FD vib; X by eigenvector phase; digit-identical to restricted twin at \(M=0\) |
| Pre-symmetrization asymmetry [eV/Å²] | \(9.5\times10^{-6}\) | \(\le10^{-4}\) | supercell Hermitian diagnostic |
| Wall | **107.9 s** (serial) | fair ratio ≤ 10 | 12 supercell responses × 2 spins; family 60 s aspirational exceeded by spin doubling |

## Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| X phonons Si 2×1×1 supercell UKS-PBE (CP2K FD vib, `EPS_SCF 1e-8`, serial) | 577.8 | 107.9 | **0.19** | yes |

Third DFPT-family slug with a real CP2K wall twin (binary
`~/.local/bin/cp2k`, CP2K 2026.1 rev `85b8a9b49c`); closes the joint
spin × GGA × finite-q product path.

## References

- Offline pins: `regressions/response/dfpt-gga-spin-finite-q-si/REFERENCE.md`
- Regression: `regressions/response/dfpt-gga-spin-finite-q-si/reg_dfpt-gga-spin-finite-q-si.cpp`
- Oracle note + CP2K input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-finite-q-si.note`,
  `tools/cp2k_oracle/inputs/dfpt-gga-spin-finite-q-si/si4_uks_pbe_vib.inp`
- Parents: `docs/tutorial/methods/dfpt-gga-spin-si.md`,
  `docs/tutorial/methods/dfpt-gga-finite-q-si.md`
- Literature: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001); Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991);
  Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
