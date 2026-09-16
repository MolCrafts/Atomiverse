# DFPT phonons at Γ with pure GGA kernel — diamond Si (PW PBE, NC-UPF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-gga-si` |
| Regression | `regressions/response/dfpt-gga-si/` |
| Public API | `cpu::pw::DfptPhonon` (members) + `cpu::pw::DfptPhononResult`; `PW.xc = XCType::PBE` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — **GGA kernel** ✓ (`reg_dfpt-gga-si`) |
| Dep | `dfpt-phonon-si` ✓ (LSDA Γ product); ground-state twin `pw-pbe-gga-si` ✓ |

## Summary

**Density-functional perturbation theory** builds the dynamical matrix from
self-consistent first-order response (Sternheimer) rather than `6N`
finite-displacement SCF solves. This product extends the Γ-point PW path
beyond the LSDA kernel (`dfpt-phonon-si`) to a **true pure-GGA XC kernel**
(PBE): \(K_{\mathrm{xc}}\) is the second functional derivative of the same
`XCBuilder` operator used in the ground-state SCF (central finite difference
of \(V_{\mathrm{xc}}[n]\) with spectral \(\nabla\rho\) and GGA flux). Product
system: diamond silicon, 2-atom fcc primitive cell, minimal twin (10 Ry,
\(2\times2\times2\) MP, in-tree NC UPF). Public surface: concrete
`cpu::pw::PW` members + `cpu::pw::DfptPhonon` members — no `*Config`, no
factories. Sharp correctness is the **offline same-Hamiltonian energy second
difference** (not live CP2K); fair wall vs CP2K is **`perf_exempt`** (no CP2K
binary on the pin host). Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515
(2001); Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

## Method

**DFPT at \(\mathbf{q}=\mathbf{0}\).** For each Cartesian ionic displacement
direction, solve the Sternheimer equation in the conduction space

\[
P_c\,(H - \varepsilon)\,P_c\,|\Delta\psi\rangle
=
-P_c\,\Delta V_{\mathrm{ext+Hxc}}\,|\psi\rangle,
\]

screen the induced density with Hartree + XC kernel

\[
\Delta V_{\mathrm{Hxc}}[\Delta n]
=
\Delta V_{\mathrm{H}}[\Delta n]
+
K_{\mathrm{xc}}[\Delta n],
\]

and assemble Cartesian force constants \(C_{I\alpha,J\beta}\) together with
analytic Ewald second derivatives and KB nonlocal first/second derivative
terms. Mass-weighted diagonalization yields \(\omega\) [cm⁻¹]. The acoustic
sum rule is **not** imposed; it is a validation observable.

**GGA kernel (this slug).** Pure LDA keeps a local \(f_{\mathrm{xc}}\) table.
For pure GGA (`XCType::PBE` and siblings with
`hybrid_exchange_fraction == 0`), \(K_{\mathrm{xc}}\) is obtained by central
finite difference of the ground-state \(V_{\mathrm{xc}}[n]\) from the same
spectral-gradient `XCBuilder` path used in SCF (true GGA, not
\(\sigma=0\)). Hybrids are rejected (no exact-exchange response).

**Ground state.** Restricted fixed occupations, UPF norm-conserving
pseudopotential, closed-shell insulator. Same Hamiltonian family as
`pw-pbe-gga-si` (true PBE on the FFT grid).

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Spin | Restricted (closed shell) |
| XC | **PBE** pure GGA kernel (`XCType::PBE`) |
| Structure | diamond Si, 2-atom fcc primitive, \(a = 5.43\) Å |
| Basis positions [Å] | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| PP | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **2×2×2** MP (unshifted) |
| SCF | density L2 tol \(10^{-8}\), Pulay mixing |
| Response | `response_tol` \(10^{-5}\), `mix_beta` 0.7, CG tol \(10^{-9}\) |
| \(\mathbf{q}\) | Γ (`q_frac = {0,0,0}`) |

**Key exclusions (non-goals of this slug)**

- Live CP2K / Quantum ESPRESSO inside ctest (offline pins only; preferred
  oracle note at `tools/cp2k_oracle/inputs/dfpt-gga-si.note`).
- Hybrid / meta-GGA response kernels.
- Metallic smearing / Fermi-surface terms; collinear UKS DFPT.
- LO–TO splitting / Born effective charges (nonpolar Si does not need them
  for the product gate).
- CUDA DFPT (CPU product only).
- Absolute \(\omega(\Gamma)\) digit match to experimental Raman or a live
  CP2K process at pin time (minimal twin + no CP2K binary).

## Implementation

Public surface only (`cpu::Frame` + concrete method members; no `*Config`,
no `make_*` / `build_*`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// Diamond Si FCC primitive: a = 5.43 Å; basis (0,0,0), (a/4,a/4,a/4)
cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_mass)[0] = 28.0855;
frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[1] = 5.43 / 4.0;
frame.span(kAtom_y)[1] = 5.43 / 4.0;
frame.span(kAtom_z)[1] = 5.43 / 4.0;
// FCC cell matrix as in reg_dfpt-gga-si.cpp
// frame.box.set_cell(...);

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry  = 10.0;                         // [Ry]
method.k_mesh   = 2;                            // 2×2×2 MP
method.xc       = cpu::scf::XCType::PBE;        // pure GGA kernel
method.tol      = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.response_tol = 1e-5;
dfpt.cg_tol       = 1e-9;
dfpt.mix_beta     = 0.7;
// q_frac default {0,0,0} = Γ
const DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm⁻¹] ascending, length 3N
// r.force_constants [eV/Å²], shape [3N, 3N]
```

`compute` throws `std::invalid_argument` unless `method.xc` is pure LSDA or
pure GGA (`hybrid_exchange_fraction == 0`), `upf_path` is set, smearing is
off, and spin is restricted — hybrids and metals are not silent fallbacks.

Headers / sources:

- Public: `include/atomiverse/cpu/planewave/dfpt_phonon.h`
- Private assembly: `include/atomiverse/cpu/planewave/internal/dfpt.h`,
  `src/cpu/planewave/internal/dfpt.cpp`, `src/cpu/planewave/dfpt_phonon.cpp`

Build and run the regression (scoped; never bare full-tree build for this
slug alone):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j"$(nproc)" --target atomiverse reg_dfpt-gga-si
  cmake --build build --target run_reg_dfpt-gga-si
'
```

## Validation (vs offline pin / preferred CP2K)

Source: `regressions/response/dfpt-gga-si/REFERENCE.md` (offline pin; never
live CP2K in ctest). Preferred external oracle: CP2K Quickstep DFPT /
`VIBRATIONAL_ANALYSIS` on the same cell (PBE) — see
`tools/cp2k_oracle/inputs/dfpt-gga-si.note`. **Host status 2026-07-26:** no
CP2K binary/module; sharp gate is same-Hamiltonian energy FD.

Diamond Si primitive (a = 5.43 Å, 10 Ry, 2×2×2 MP, PBE, NC UPF). Measured
pins (Release, OMP=8, 2026-07-26):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Energy-FD \(C[(0x),(0x)]\) [eV/Å²] | **17.3456** (hard-coded) | oracle for DFPT \(C_{00}\) | \(h=0.03\) Å; \(E_0=-15.6411877416\) Ry; \(E_\pm=-15.6406136403/-15.6406144498\) Ry |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **17.3711** | \(\|C_{00}-17.3456\| \le 0.30\) (~0.15 %) | true discrete Hessian twin |
| Acoustic \(\omega\) [cm⁻¹] | 12.08, 12.08, 12.09 | \(\|\omega\| \le 30\) | no ASR imposition |
| Optical triplet [cm⁻¹] | **579.82, 579.82, 579.96** | each \(\in [450,650]\); spread \(\le 10\) | lit. Raman ~517 cm⁻¹; minimal-settings bias |
| Max force-constant row sum [eV/Å²] | 0.0151 | \(\le 0.05\) | translational invariance |
| Pre-symmetrization \(\max\|C-C^T\|/2\) [eV/Å²] | \(9.6\times10^{-10}\) | \(\le 10^{-5}\) | Hermitian diagnostic |
| Wall | ≈ 15 s | ≤ 60 s e2e budget | 2 atoms, 6 responses, 8 k |

Unit tests (`test_cpu_pw_dfpt`): analytic Ewald dynamical matrix vs FD of
Ewald forces, Sternheimer CG on free-electron Hamiltonian, pure-GGA
acceptance / hybrid reject guards. No full SCF e2e under `tests/`.

## Performance

Source: the case `REFERENCE.md` + REFERENCE.md.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Γ DFPT Si PBE (2-atom, 10 Ry, 2×2×2) | n/a | ≈ 15 | n/a | n/a | **`perf_exempt`** — no CP2K binary on pin host |

When a CP2K binary is available: record \(\omega(\Gamma)\) [cm⁻¹], version,
and fair ratio \(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require \(r\le 10\)).

## Limitations

- **Minimal twin:** 10 Ry / 2×2×2 / experimental lattice / LDA-generated NC
  UPF used as fixed valence PP — optical \(\omega\) sits ~12 % above the
  experimental Raman TO(Γ); that offset is settings bias, not the
  correctness gate (energy-FD \(C_{00}\) is).
- **Γ product only for this slug** — finite-q / incommensurate DFPT and
  phonon dispersions are sibling products (`dfpt-finite-q-si`,
  `dfpt-incommensurate-si`, `phonon-dispersion-si` family).
- **No external absolute energy twin** until CP2K/QE is re-run offline and
  pins land in REFERENCE.md.
- CPU only; CUDA PW DFPT out of this loop’s scope.

## References

- Offline pin: `regressions/response/dfpt-gga-si/REFERENCE.md`
- Regression: `regressions/response/dfpt-gga-si/reg_dfpt-gga-si.cpp`
- Oracle note: `tools/cp2k_oracle/inputs/dfpt-gga-si.note`
- Sibling LSDA product: `docs/tutorial/methods/dfpt-phonon-si.md`
- Ground-state PBE twin: `docs/tutorial/methods/pw-pbe-gga-si.md`
- Literature: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001); Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**,
  3865 (1996); Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991)
  (Raman / DFPT-LDA optical anchor ~517 cm⁻¹)
