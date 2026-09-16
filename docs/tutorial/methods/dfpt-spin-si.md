# DFPT phonons at Γ — collinear spin, diamond Si (PW LSDA)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-spin-si` |
| Regression | `regressions/response/dfpt-spin-si/` |
| Public API | `cpu::pw::DfptPhonon` (members) + `cpu::pw::DfptPhononResult`; `PW::spin_polarized` / `multiplicity` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — collinear-spin product ✓ |
| Dep | `dfpt-phonon-si` ✓ (restricted LSDA Γ); sibling `dfpt-gga-si` ✓ (GGA kernel) |

## Summary

**Collinear-spin density-functional perturbation theory** at \(\mathbf{q}=0\)
extends the plane-wave DFPT foothold (`dfpt-phonon-si`) to
`method.spin_polarized = true`. Each Sternheimer response is spin-resolved
(\(f=1\) per spin-orbital); Hartree screening uses total \(\Delta n\), and the
XC kernel is the local collinear \(f_{\mathrm{xc}}^{\sigma\sigma'}\) table for
pure LSDA. This product pins closed-shell collinear (\(M=0\)) diamond Si on
the same NC-UPF / 10 Ry / 2×2×2 twin as the restricted LSDA gate, so the
spin kernel can be checked against a same-Hamiltonian energy second
difference without changing the physical system. Product gate:
`regressions/response/dfpt-spin-si` (`reg_dfpt-spin-si`). Preferred offline
oracle: CP2K Quickstep DFPT / `VIBRATIONAL_ANALYSIS` with LSD/UKS on the same
cell — **not available on the pin host**; sharp correctness is the
energy-FD pin. Performance: **`perf_exempt`**.

## Method

**Level.** Γ-point DFPT dynamical matrix for concrete method `cpu::pw::PW`
with collinear spin, pure LSDA XC, UPF norm-conserving pseudopotential, and
fixed occupations (no metallic smearing).

**Response equations.** For perturbation direction \(u_{I\alpha}\) at
\(\mathbf{q}=0\), the first-order density response on each spin channel
satisfies a self-consistent Sternheimer problem. The induced Hartree + XC
potential is

\[
\Delta V_{\mathrm{Hxc},\sigma}[\Delta n_\uparrow,\Delta n_\downarrow]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + \sum_{\sigma'} K_{\mathrm{xc}}^{\sigma\sigma'}[\Delta n_{\sigma'}],
\]

with \(\Delta n = \Delta n_\uparrow + \Delta n_\downarrow\), occupation
\(f=1\) per spin-orbital, and \(K_{\mathrm{xc}}\) the collinear LSDA kernel
(local \(f_{\mathrm{xc}}^{\sigma\sigma'}\) table). Pure GGA kernels use a
finite-difference of `XCBuilder::step_spin` (covered by `dfpt-gga-si` on the
restricted path; this slug’s e2e twin is LSDA + spin). After \(3N\)
responses, the Cartesian force-constant matrix \(C\) [eV/Å²] is assembled
(analytic Ewald second derivatives + KB nonlocal terms + electronic
response), Hermitian-symmetrized, mass-weighted, and diagonalized to
wavenumbers \(\omega\) [cm⁻¹].

**Closed-shell collinear.** The regression sets `multiplicity = 1` so
\(N_\uparrow = N_\downarrow = 4\) for diamond Si (8 valence electrons). The
collinear framework is fully exercised (two spin channels, spin XC kernel,
spin density mixing) while remaining a non-magnetic insulator — optical
frequencies therefore land in the same LDA window as restricted LSDA.

**Observables (on result, never Frame).**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\omega_k\) | `DfptPhononResult::frequencies` (length \(3N\), ascending) | cm⁻¹ |
| \(C\) | `DfptPhononResult::force_constants` | eV/Å² |
| \(\max|C-C^\top|/2\) | `max_asymmetry` (pre-symmetrization) | eV/Å² |

**Key exclusions.**

- Incommensurate-\(\mathbf{q}\) + collinear spin (throws; restricted-only
  finite-\(q\) path — see `dfpt-incommensurate-si`).
- Hybrids, meta-GGA, metallic smearing, empty `upf_path` (toy \(V_\mathrm{loc}\)).
- LO–TO / Born charges (nonpolar Si; not required).
- Open-shell magnetic phonons as a separate physics pin (this slug is the
  \(M=0\) collinear framework gate).
- Public `*Config` bags, `make_*` / `build_*` factories, CUDA DFPT.
- Live CP2K at ctest time (offline pins only).

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001). Literature LDA optical Γ ~ 517 cm⁻¹: Giannozzi et al.,
*Phys. Rev. B* **43**, 7231 (1991).

## Implementation

Public surface only (`cpu::Frame` + method / DFPT members; no `*Config`, no
factories). Header: `include/atomiverse/cpu/planewave/dfpt_phonon.h`.

Minimal sequence (matches `reg_dfpt-spin-si.cpp`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// diamond Si 2-atom fcc primitive, a = 5.43 Å
cpu::Frame frame(2);
// … types Z=14, mass, basis atom at (a/4,a/4,a/4), set fcc cell …

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = 10.0;
method.k_mesh = 2;
method.xc = cpu::scf::XCType::LSDA;
method.spin_polarized = true;   // collinear DFPT path
method.multiplicity = 1;        // M = 0
method.tol = 1e-8;

DfptPhonon dfpt;
dfpt.response_tol = 1e-5;
dfpt.cg_tol = 1e-9;
dfpt.mix_beta = 0.7;
DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm^-1]; r.force_constants [eV/Å²]
```

Build and run the regression (`ATV_BUILD_REGRESSIONS=ON`, PW ON):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_dfpt-spin-si
  ./build/regressions/reg_dfpt-spin-si
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/planewave/dfpt_phonon.h` | `DfptPhonon` / `DfptPhononResult` |
| `src/cpu/planewave/dfpt_phonon.cpp` | Γ / commensurate / incommensurate dispatch |
| `include/atomiverse/cpu/planewave/internal/dfpt.h` | private Sternheimer assembly |
| `regressions/response/dfpt-spin-si/` | e2e + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/dfpt-spin-si.note` | offline CP2K twin note |

## Validation (vs CP2K / same-Hamiltonian FD)

Source:
[`regressions/response/dfpt-spin-si/REFERENCE.md`](../../../regressions/response/dfpt-spin-si/REFERENCE.md).
Oracle priority: offline **CP2K DFPT / `VIBRATIONAL_ANALYSIS`** (LSD/UKS) when
a binary is available; **this pin host had no CP2K module** — sharp pin is the
central second difference of the **collinear-spin** PW SCF total energy at
identical ecut/k/XC/UPF/spin. **Never** live CP2K in ctest.

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, \(a = \mathbf{5.43}\) Å |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| XC | `XCType::LSDA` (collinear LSDA kernel) |
| Spin | collinear, `spin_polarized = true`, multiplicity = 1 (\(N_\uparrow=N_\downarrow=4\)) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **2×2×2** MP (unshifted) |
| SCF | tol \(10^{-8}\) (density L2) |
| Response | `response_tol` \(10^{-5}\), `mix_beta` 0.7, CG tol \(10^{-9}\) |

### Accuracy table

| Quantity | Atomiverse / pin | External / rule | Tol / band | Notes |
| -------- | ---------------: | --------------- | ---------- | ----- |
| Energy-FD \(C[(0x),(0x)]\) | **17.3390 eV/Å²** | same-Ham. \((E(+h)-2E_0+E(-h))/h^2\), \(h=0.03\) Å, spin SCF | pin definition | \(E_0=-7.805522314842\) Ha |
| DFPT \(C[(0x),(0x)]\) | **17.3415 eV/Å²** | energy-FD pin | \(\lvert\Delta\rvert\le\mathbf{0.30}\) (~0.015 %) | sharp correctness gate |
| Acoustic \(\omega\) (no ASR) | 1.33, 1.34, 1.34 cm⁻¹ | Γ acoustic sum rule | \(\lvert\omega\rvert\le\mathbf{30}\) cm⁻¹ | ASR never imposed |
| Optical triplet | **579.45, 579.45, 579.59 cm⁻¹** | LDA / Raman window (~517 lit.) | **[500, 650]** cm⁻¹ | +~12 % minimal-settings bias |
| Optical spread | 0.14 cm⁻¹ | degeneracy | \(\le\mathbf{10}\) cm⁻¹ | |
| Max row sum (ASR) | 0.0002 eV/Å² | translational invariance | \(\le\mathbf{0.05}\) | |
| Pre-sym \(\lvert C-C^\top\rvert/2\) | \(7.3\times10^{-10}\) eV/Å² | Hermitian \(C\) | \(\le\mathbf{10^{-5}}\) | |

Measured twin: REFERENCE host note, **2026-07-26**, Release, `OMP_NUM_THREADS=8`.
Sibling restricted LSDA product: `regressions/response/dfpt-phonon-si`
(energy-FD pin 17.2393 eV/Å² class; optical ~576–578 cm⁻¹). Closed-shell
collinear \(M=0\) matches that optical window within the documented
minimal-settings bias.

**CP2K.** When `cp2k.psmp` is available offline: run DFPT /
`VIBRATIONAL_ANALYSIS` on the same cell with LSD/UKS and a comparable NC/GTH
setup; record \(\omega(\Gamma)\) [cm⁻¹], version, and wall. Twin note:
`tools/cp2k_oracle/inputs/dfpt-spin-si.note`.

## Performance

Source: `regressions/response/dfpt-spin-si/REFERENCE.md`. Queue / handoff
wave 7: **`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si Γ collinear DFPT (10 Ry, 2×2×2) | n/a | ≈ **34** (informative) | n/a | n/a (**exempt**) | No CP2K DFPT binary on pin host |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K module/binary on the build host for a fair DFPT wall twin this shift |
| ATV wall (informative) | ≈ 34 s (2-atom Si, 6 responses × 2 spins, 8 k-points) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a CP2K binary is available offline: time a comparable DFPT twin, append
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) to REFERENCE.md, require \(r\le 10\).

## Limitations

- **Γ only** for the collinear product pin; incommensurate \(q\) + spin is a
  follow-on (throws today).
- Twin is **closed-shell collinear** (\(M=0\)), not a magnetic open-shell
  phonon spectrum vs experiment.
- Minimal ecut/k (10 Ry, 2×2×2) → optical \(\omega\) ~12 % above literature
  DFPT-LDA 517 cm⁻¹; compare within the documented bias window, not as a
  production Raman prediction.
- LSDA e2e pin only (GGA kernel product is `dfpt-gga-si`).
- No LO–TO splitting / Born charges; nonpolar Si only for this gate.
- CPU only; CUDA DFPT out of scope for this loop.
- Fair CP2K wall ratio deferred until a binary is on the pin host
  (`perf_exempt`).

## References

- Offline pin: [`regressions/response/dfpt-spin-si/REFERENCE.md`](../../../regressions/response/dfpt-spin-si/REFERENCE.md)
- Regression: [`regressions/response/dfpt-spin-si/reg_dfpt-spin-si.cpp`](../../../regressions/response/dfpt-spin-si/reg_dfpt-spin-si.cpp)
- Oracle note: [`tools/cp2k_oracle/inputs/dfpt-spin-si.note`](../../../tools/cp2k_oracle/inputs/dfpt-spin-si.note)
- Sibling tutorials: [dfpt-phonon-si](dfpt-phonon-si.md),
  [fd-phonon-supercell](fd-phonon-supercell.md)
- Roadmap: [dft.md](../../roadmap/dft.md) §11 DFPT
- S. Baroni, S. de Gironcoli, A. Dal Corso, P. Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001)
- P. Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991) (Si optical Γ LDA)
