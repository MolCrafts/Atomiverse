# DFPT phonons at finite q, joint collinear spin × pure GGA × metallic smear — diamond Si at X (PW PBE, NC-UPF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-gga-spin-finite-q-smearing` |
| Regression | `regressions/response/dfpt-gga-spin-finite-q-smearing/` |
| Public API | `cpu::pw::DfptPhonon` (members, `q_frac`) + `cpu::pw::DfptPhononResult`; `PW.xc = XCType::PBE` + `PW.spin_polarized = true` + `PW.smearing = fermi_dirac` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — **finite-q × spin × pure GGA × smear** ✓ (`reg_dfpt-gga-spin-finite-q-smearing`) |
| Queue | `docs/developer/cp2k-method-queue.md` → **done** (shift wave 5, 2026-08-01); fair ratio **0.25** |
| Dep | `dfpt-gga-spin-smearing` ✓; `dfpt-gga-spin-finite-q-si` ✓; `dfpt-finite-q-spin-smearing` ✓ |

## Summary

Parents pin the pairwise and triple axes alone: **collinear spin × pure GGA ×
smear** at Γ (`dfpt-gga-spin-smearing`), **finite-\(\mathbf{q}=X\)** with joint
spin × GGA fixed occ (`dfpt-gga-spin-finite-q-si`), and **finite-q spin +
smear** on the LSDA direct path (`dfpt-finite-q-spin-smearing`). This product
pins the **four-way** path: commensurate X → 2×1×1 supercell Γ-DFPT + IFC
Fourier where every supercell Sternheimer response is spin-resolved, screened
with \(K_{\mathrm{xc}}^{\sigma\sigma'}\) from central FD of the spin-GGA
`XCBuilder::step_spin` operator, **and** weighted by Fermi–Dirac occupations
(shared \(\mu\)). Product system: diamond silicon, 2-atom fcc primitive cell,
minimal twin (10 Ry, Γ-only supercell mesh, in-tree NC UPF),
`spin_polarized = true`, multiplicity 1 (\(M=0\)), `XCType::PBE`,
`smearing = fermi_dirac`, `degauss = 0.02` Ry. Public surface: concrete
`cpu::pw::PW` members + `cpu::pw::DfptPhonon` members — no `*Config`, no
factories. Dual gate: the **offline same-Hamiltonian supercell free-energy
second difference** (sharp; at \(M=0\) gapped fill the fixed-occ supercell
pin 21.2006 eV/Å²) plus a **CP2K 2026.1 supercell UKS-PBE+SMEAR
`VIBRATIONAL_ANALYSIS`** physics anchor whose eigenvector phases identify
the folded X branch. Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515
(2001); de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995); Perdew, Burke,
Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

## Method

A phonon at rational \(\mathbf{q}=(h/n_a,k/n_b,l/n_c)\) is exactly the Γ
phonon of the \(n_a\times n_b\times n_c\) supercell folded back by
\(e^{i\mathbf{q}\cdot\mathbf{R}}\); for \(X=(1/2,0,0)\) the product is
spin-resolved Γ-DFPT on the 2×1×1 supercell (12 Sternheimer responses × 2
spins, FD occupation weights) followed by the IFC Fourier. Per Cartesian
displacement the Sternheimer equation is solved per spin channel and the
induced spin densities are screened jointly:

\[
\Delta V_{\mathrm{Hxc},\sigma}[\Delta n_\uparrow,\Delta n_\downarrow]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + \sum_{\sigma'} K_{\mathrm{xc}}^{\sigma\sigma'}[\Delta n_{\sigma'}],
\]

with Hartree from the total \(\Delta n\) and
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) from central FD of the ground-state
spin-GGA operator (`dfpt_gamma_spin_impl` on the supercell; the direct q-path
`dfpt_q_spin_impl` carries the same kernel for incommensurate \(q\)). Gapped
fill keeps near-integer FD occupations so free energy \(F\approx E\); at
\(M=0\) the joint kernel must reproduce the fixed-occ finite-q spin×GGA
parent — a consistency limit the regression checks.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → commensurate **2×1×1** supercell (4 atoms) |
| Spin | collinear, `spin_polarized = true`, multiplicity 1 |
| XC | **PBE** pure spin-GGA kernel (`XCType::PBE`) |
| Smear | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| Structure | diamond Si, 2-atom fcc primitive, \(a = 5.43\) Å |
| PP | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| Cutoff | **10 Ry**; k-mesh **1×1×1** on the supercell (Γ+X folded) |
| SCF | density L2 tol \(10^{-8}\) (DFPT-internal spin SCF caps mix ≤ 0.4) |
| Response | `response_tol` \(10^{-4}\), `mix_beta` 0.25, CG tol \(10^{-8}\), max_iter 120 |

**Key exclusions:** incommensurate-q joint spin-GGA×smear pin (shared code;
separate census decision), hybrids, CUDA, live CP2K in ctest.

## Implementation

```cpp
PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry  = 10.0;
method.k_mesh   = 1;
method.xc       = cpu::scf::XCType::PBE;
method.spin_polarized = true;
method.multiplicity   = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss  = 0.02;
method.extra_bands = 0;
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
cmake --build build -j"$(nproc)" --target atomiverse reg_dfpt-gga-spin-finite-q-smearing
cmake --build build --target run_reg_dfpt-gga-spin-finite-q-smearing
```

## Validation (vs offline pins)

Source: `regressions/response/dfpt-gga-spin-finite-q-smearing/REFERENCE.md`.
Measured 2026-08-01 (Release; OMP=1):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Free-energy FD \(C[(0x),(0x)]\) [eV/Å²] (supercell, M=0 + FD) | **21.2006** (hard-coded) | oracle for DFPT \(C_{00}\) | same-Ham. pin; gapped fill ⇒ F=E |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **21.1911** | \(\|C_{00}-21.2006\|\le0.30\) (~0.04 %); vs fixed-occ parent 21.1911 \(\le0.05\) | supercell diagonal |
| \(\omega(X)\) [cm⁻¹] | −108.83, −108.32, 239.12, 451.33, 668.84, 668.90 | \(\|\omega\|\) low 3 ∈ [50, 400]; high 3 ∈ [300, 850] | soft TA pair ≤ 2 imaginary (minimal twin) |
| CP2K 2026.1 X anchor [cm⁻¹] | 98.22, 130.53, 364.05, 427.18, 452.08, 458.19 | inside the same windows | supercell UKS-PBE+SMEAR FD vib; X by eigenvector phase |
| Pre-symmetrization asymmetry [eV/Å²] | \(9.5\times10^{-6}\) | \(\le10^{-4}\) | supercell Hermitian diagnostic |
| Wall | **148.1 s** (serial) | fair ratio ≤ 10 | 12 supercell responses × 2 spins + smear weights |

## Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| X phonons Si 2×1×1 supercell UKS-PBE+SMEAR (CP2K FD vib, `EPS_SCF 1e-8`, serial) | 585.9 | 148.1 | **0.25** | yes |

Real CP2K wall twin (binary `…/cp2k/install/bin/cp2k.psmp`, CP2K 2026.1 rev
`git:5e54ba2`); closes the joint finite-q × spin × GGA × metallic-smear
product path.

## References

- Offline pins: `regressions/response/dfpt-gga-spin-finite-q-smearing/REFERENCE.md`
- Regression: `regressions/response/dfpt-gga-spin-finite-q-smearing/reg_dfpt-gga-spin-finite-q-smearing.cpp`
- Oracle note + CP2K input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-finite-q-smearing.note`,
  `tools/cp2k_oracle/inputs/dfpt-gga-spin-finite-q-smearing/si4_uks_pbe_smear_vib.inp`
- Parents: `docs/tutorial/methods/dfpt-gga-spin-smearing.md`,
  `docs/tutorial/methods/dfpt-gga-spin-finite-q-si.md`,
  `docs/tutorial/methods/dfpt-finite-q-spin-smearing.md`
- Literature: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001); de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995);
  Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
