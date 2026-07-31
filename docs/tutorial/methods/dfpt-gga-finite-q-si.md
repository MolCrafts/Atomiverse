# DFPT phonons at finite q with a pure GGA kernel — diamond Si at X (PW PBE, NC-UPF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-gga-finite-q-si` |
| Regression | `regressions/response/dfpt-gga-finite-q-si/` |
| Public API | `cpu::pw::DfptPhonon` (members, `q_frac`) + `cpu::pw::DfptPhononResult`; `PW.xc = XCType::PBE` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — **finite-q × pure GGA** ✓ (`reg_dfpt-gga-finite-q-si`) |
| Queue | `docs/developer/cp2k-method-queue.md` → **done** (shift wave 2, 2026-07-31); fair ratio **0.17** |
| Dep | `dfpt-finite-q-si` ✓ (commensurate X path, LSDA); `dfpt-gga-si` ✓ (Γ, PBE kernel) |

## Summary

The two parent products pin each axis alone: the **commensurate finite-q
path** (X → 2×1×1 supercell Γ-DFPT + IFC Fourier) with an LSDA kernel
(`dfpt-finite-q-si`), and the **true pure-GGA (PBE) kernel** at Γ
(`dfpt-gga-si`). This product pins the joint path: finite
\(\mathbf{q}=X=(1/2,0,0)\) where every supercell Sternheimer response is
screened with \(K_{\mathrm{xc}}\) from central finite difference of the same
GGA `XCBuilder` operator the ground-state SCF converged (spectral
\(\nabla\rho\) + GGA flux). Product system: diamond silicon, 2-atom fcc
primitive cell, minimal twin (10 Ry, Γ-only supercell mesh, in-tree NC UPF),
`XCType::PBE`. Public surface: concrete `cpu::pw::PW` members +
`cpu::pw::DfptPhonon` members — no `*Config`, no factories. Dual gate: the
**offline same-Hamiltonian supercell PBE energy second difference** (sharp)
plus a **CP2K 2026.1 supercell `VIBRATIONAL_ANALYSIS`** physics anchor whose
eigenvector phases identify the folded X branch. Cite: Baroni et al., *Rev.
Mod. Phys.* **73**, 515 (2001); Giannozzi et al., *Phys. Rev. B* **43**,
7231 (1991); Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865
(1996).

## Method

A phonon at rational \(\mathbf{q}=(h/n_a,k/n_b,l/n_c)\) is exactly the Γ
phonon of the \(n_a\times n_b\times n_c\) supercell folded back by
\(e^{i\mathbf{q}\cdot\mathbf{R}}\); for \(X=(1/2,0,0)\) the product is
Γ-DFPT on the 2×1×1 supercell (12 Sternheimer responses) followed by the IFC
Fourier (`PhononDispersion::frequencies_at_q`). The response screening per
iteration is

\[
\Delta V_{\mathrm{Hxc}}[\Delta n]
  = \Delta V_{\mathrm{H}}[\Delta n] + K_{\mathrm{xc}}[\Delta n],
\]

where pure LDA keeps the local \(f_{\mathrm{xc}}\) table and **pure GGA
(this slug)** applies \(K_{\mathrm{xc}}\) by central FD of the ground-state
GGA \(V_{\mathrm{xc}}[n]\) along the induced density direction
(`add_xc_kernel_gamma`, `src/cpu/planewave/internal/dfpt.cpp`) — identical
machinery to the Γ parent, exercised here on the supercell whose Hessian
carries the X modes. Hybrids are rejected.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → commensurate **2×1×1** supercell (4 atoms) |
| XC | **PBE** pure GGA kernel (`XCType::PBE`), restricted |
| Structure | diamond Si, 2-atom fcc primitive, \(a = 5.43\) Å |
| PP | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| Cutoff | **10 Ry**; k-mesh **1×1×1** on the supercell (Γ+X folded) |
| SCF | density L2 tol \(10^{-8}\) |
| Response | `response_tol` \(10^{-4}\), `mix_beta` 0.25, CG tol \(10^{-8}\), max_iter 120 |

**Key exclusions:** joint spin × GGA × finite-q is shipped as
`dfpt-gga-spin-finite-q-si`; dedicated incommensurate-q GGA pin (shared code;
separate census decision), hybrids, CUDA, live CP2K in ctest.

## Implementation

```cpp
PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry  = 10.0;
method.k_mesh   = 1;
method.xc       = cpu::scf::XCType::PBE;   // pure GGA kernel
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
cmake --build build -j"$(nproc)" --target atomiverse reg_dfpt-gga-finite-q-si
cmake --build build --target run_reg_dfpt-gga-finite-q-si
```

## Validation (vs offline pins)

Source: `regressions/response/dfpt-gga-finite-q-si/REFERENCE.md`. Measured
2026-07-31 (Release; identical numbers at OMP=1 and OMP=8):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Energy-FD \(C[(0x),(0x)]\) [eV/Å²] (supercell PBE SCF) | **21.2006** (hard-coded) | oracle for DFPT \(C_{00}\) | \(h=0.03\) Å; \(E_0=-15.121055415608\) Ha |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **21.1974** | \(\|C_{00}-21.2006\|\le0.30\) (~0.015 %) | supercell diagonal |
| \(\omega(X)\) [cm⁻¹] | −108.49, −107.98, 239.26, 451.36, 668.88, 668.94 | \(\|\omega\|\) low 3 ∈ [50, 400]; high 3 ∈ [300, 850] | soft TA pair ≤ 2 imaginary (minimal twin, same as LSDA parent) |
| CP2K 2026.1 X anchor [cm⁻¹] | 97.98, 131.27, 370.58, 428.24, 457.48, 458.68 | inside the same windows | supercell FD vib; X branch by eigenvector phase |
| LSDA parent \(\lvert\omega\rvert\) [cm⁻¹] | 130, 130, 230, 448, 673, 673 | cross-parent sanity | XC-kernel change only |
| Pre-symmetrization asymmetry [eV/Å²] | \(7.3\times10^{-6}\) | \(\le10^{-4}\) | supercell Hermitian diagnostic |
| Wall | **49.8 s** (serial) | ≤ 60 s | 12 supercell responses, Γ-only mesh |

## Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| X phonons Si 2×1×1 supercell (CP2K FD vib, `EPS_SCF 1e-8`, serial) | 288.7 | 49.8 | **0.17** | yes |

Second DFPT-family slug with a real CP2K wall twin (binary
`~/.local/bin/cp2k`, CP2K 2026.1 rev `85b8a9b49c`); the LSDA finite-q
parent's stale "no CP2K binary" `perf_exempt` reason no longer holds on this
host.

## References

- Offline pins: `regressions/response/dfpt-gga-finite-q-si/REFERENCE.md`
- Regression: `regressions/response/dfpt-gga-finite-q-si/reg_dfpt-gga-finite-q-si.cpp`
- Oracle note + CP2K input: `tools/cp2k_oracle/inputs/dfpt-gga-finite-q-si.note`,
  `tools/cp2k_oracle/inputs/dfpt-gga-finite-q-si/si4_pbe_vib.inp`
- Parents: `docs/tutorial/methods/dfpt-finite-q-si.md`,
  `docs/tutorial/methods/dfpt-gga-si.md`
- Literature: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001); Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991);
  Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
