# Metallic occupation DOS — bulk silicon (PW + Fermi–Dirac)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dos-metallic-si` |
| Regression | `regressions/pw/dos-metallic-si/` |
| Public API | `cpu::pw::PW` (`smearing`, `degauss`, `extra_bands`) + `cpu::pw::Dos` / `DosResult` |
| Roadmap | [dft.md](../../roadmap/dft.md) § bands/DOS / metallic smear|
## Summary

Plane-wave **Gaussian-broadened density of states** after a spin-unpolarized
LDA SCF with **Fermi–Dirac metallic occupation smearing**. Geometry is bulk
silicon (diamond, 2-atom fcc primitive) with the same ONCVPSP Si UPF and
tutorial ecut/k as `bands-dos-si`, but SCF occupations are fractional and the
DOS post-processor recomputes the chemical potential \(\mu\) on a denser
Monkhorst–Pack mesh. The occupied integral is then to \(E_F\), not to VBM:

\[
n_{\mathrm{elec}}(E_F)=\int_{-\infty}^{E_F} D(E)\,dE \approx n_e.
\]

Public path: members on `cpu::pw::PW` and `cpu::pw::Dos` only — **no** public
`*Config`, **no** `make_*` / `build_*` factories. Product regression:
`regressions/pw/dos-metallic-si` (`reg_dos-metallic-si`). Oracle = offline
CP2K Quickstep product counterpart (`&SMEAR METHOD FERMI_DIRAC`); absolute
\(D(E)\) is **not** a GTH+Gaussian ↔ NC-UPF twin. Performance: **`perf_exempt`**
(no fair CP2K wall on the pin host).

## Method

**Level.** Periodic plane-wave Kohn–Sham DFT. Self-consistent density and
effective potential from the PW stack (Hartree in \(G\)-space, LDA XC on the
FFT grid, Davidson diagonalization, NC UPF) with **metallic occupations**.
After SCF, **DOS is non-self-consistent**: \(V_\mathrm{eff}\) is frozen; only
k-dependent kinetic/projectors change while eigenvalues are recomputed on a
denser mesh.

**Occupations (closed-shell convention).** Spinless \(f(x)\in[0,1]\) is
doubled to band occupation \(f_b\in[0,2]\). With reduced energy
\(x = (\varepsilon-\mu)/\sigma\) and \(\sigma =\) `degauss` [Ry]:

| Kind | Spinless \(f(x)\) | Public enum |
| ---- | ----------------- | ----------- |
| Fermi–Dirac | \(1/(1+e^{x})\) | `Smearing::fermi_dirac` |
| Gaussian | \(\tfrac12\mathrm{erfc}(x)\) | `Smearing::gaussian` |
| Methfessel–Paxton order \(N\) | MP series of order `mp_order` | `Smearing::methfessel_paxton` |

This twin pins **Fermi–Dirac** with `degauss = 0.02` Ry and `extra_bands = 4`
(empty states near \(E_F\) for Davidson). Silicon has \(n_e=8\) valence
electrons/cell.

**Free energy (SCF).** With entropy weight \(s\) for the chosen smear (FD:
\(s = -2[f\ln f+(1-f)\ln(1-f)]\) in band form),

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -TS = -\sigma\sum_{k,b} w_k\, s_{kb}.
\]

`DosResult::scf_energy_ry` reports this free energy when smearing is on.

**DOS mesh and broadening.** Uniform MP mesh of size `Dos::k_mesh`³; Gaussian
broadening on the NSCF spectrum

\[
D(E_i)
  = \sum_{n\mathbf{k}} w_{\mathbf{k}}\,
    \frac{2}{\sqrt{2\pi}\,\sigma}\,
    \exp\!\Bigl(-\frac{(E_i-\varepsilon_{n\mathbf{k}})^2}{2\sigma^2}\Bigr)
\]

with spinless closed-shell weight 2 absorbed into the kernel
(`gaussian_dos` in `bands_dos.h`). Units: \(D(E)\) in states / Ry / cell.

**Chemical potential on the DOS mesh.** When `method.smearing != none`,
\(\mu\) is recomputed from the denser-mesh eigenvalues so that
\(\sum w f = n_e\); the occupied integral uses \(E_\ast=E_F\)
(`DosResult::e_fermi_ry`). Fixed occupations (`Smearing::none`) integrate to
VBM instead — that path is `bands-dos-si`.

**Gapped Si + FD (this twin).** At the tutorial ecut/k, diamond Si remains an
open-gap semiconductor: FD occupations stay near-integer, `gap_ry > 0` on the
DOS mesh, and \(E_F\) sits in the gap (within a few `degauss` of
\([E_\mathrm{vbm},E_\mathrm{cbm}]\)). The scientific gate is the **metallic
DOS product path** (no throw) plus electron count from \(\int^{E_F} D\), not a
metal-like vanishing gap. True metallic energy pins live on
`pw-smearing-al` (FCC Al).

**XC / spin.** `XCType::LSDA` (Slater + PW92) on the PW grid; restricted (no
collinear spin on this path). `Bands::compute` still rejects metallic
smearing — band-path + FD is out of scope here.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(D(E)\) | `DosResult::dos` | 1/Ry/cell |
| \(E_i\) | `DosResult::energy_ry` bin centres | Ry |
| \(E_F\) | `DosResult::e_fermi_ry` (DOS-mesh \(\mu\)) | Ry |
| \(n_\mathrm{elec}(E_F)\) | `DosResult::n_elec_to_vbm` (name kept; cut = \(E_F\) under smear) | electrons/cell |
| \(E_\mathrm{vbm}\), \(E_\mathrm{cbm}\), \(E_g\) | spectrum extrema / gap on DOS mesh | Ry |
| \(F_\mathrm{SCF}\) | `DosResult::scf_energy_ry` | Ry |

**Key exclusions (this slug).**

- Fixed-occupation insulator bands + DOS → `bands-dos-si`.
- Spin-polarized / spin-resolved DOS; metallic **band path**
  (`Bands` + smear still throws).
- Metallic SCF energy pin on a true metal → `pw-smearing-al`.
- Metallic DFPT → `dfpt-metallic-si`.
- GPW `&SMEAR` productization (separate GPW metallic rows).
- Absolute \(\varepsilon_n\) / \(D(E)\) parity with CP2K Quickstep GTH+DZVP.
- Production-converged \(E_\mathrm{cut}\) / dense k when they break ≤ 60 s e2e.
- Live CP2K at ctest time (offline pins only).
- Public `*Config` bags; `make_*` / `build_*` factories.
- CUDA PW surface as a CPU compatibility API for this pin.

## Implementation

Public surface only (`cpu::Frame` + concrete method / post-processor members).
Headers: `include/atomiverse/cpu/planewave/bands_dos.h`, `pw_solver.h`.
Implementation: `src/cpu/planewave/bands_dos.cpp` (smear-restricted SCF →
freeze \(V_\mathrm{eff}\) → NSCF Davidson on DOS mesh → \(\mu\) + Gaussian
DOS). Occupations: `src/cpu/planewave/internal/smearing.*` (not a user type).

Minimal sequence (matches `reg_dos-metallic-si.cpp`):

```cpp
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/bands_dos.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;       // Å
constexpr double kEcutRy = 15.0;  // tutorial incomplete basis
constexpr int kMeshScf = 2;       // 2×2×2 MP SCF
constexpr int kNBands = 8;
constexpr double kDegaussRy = 0.02;
constexpr int kExtraBands = 4;

cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_x)[1] = kA / 4.0;
frame.span(kAtom_y)[1] = kA / 4.0;
frame.span(kAtom_z)[1] = kA / 4.0;

const double h = kA / 2.0;
Mat3 cell;
// fcc primitive: (0,h,h), (h,0,h), (h,h,0)
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method;
method.upf_path = "…/tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = kEcutRy;
method.k_mesh = kMeshScf;
method.tol = 1e-6;
method.max_iter = 80;
method.mix_beta = 0.5;
method.xc = cpu::scf::XCType::LSDA;
method.smearing = Smearing::fermi_dirac;
method.degauss = kDegaussRy;       // Ry
method.extra_bands = kExtraBands;

Dos dos;
dos.k_mesh = 4;
dos.n_bands = kNBands;
dos.sigma_ry = 0.05;
dos.n_bins = 200;
const DosResult dr = dos.compute(method, frame);
// dr.e_fermi_ry, dr.n_elec_to_vbm (integral to E_F), dr.gap_ry, dr.dos
```

Build and run the regression (`ATV_BUILD_PW=ON`, `ATV_BUILD_REGRESSIONS=ON`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_dos-metallic-si -j$(nproc)
  ./build/regressions/reg_dos-metallic-si
'
```

Unit (no full SCF e2e): `test_cpu_pw_bands_dos` — Gaussian integral identity,
`integrate_dos_to`, spectrum helpers. Occupation closed forms:
`test_cpu_pw_smearing`.

## Validation (vs CP2K note / literature)

Source: `regressions/pw/dos-metallic-si/REFERENCE.md` (offline pin; never live
third party in ctest). Closed acceptance: feature merged on `dev` as queue row
`dos-metallic-si` (spec deleted at done; history in git).

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Same family as | `regressions/pw/bands-dos-si`, `pw-lda-si` |

### Twin geometry and settings (regression)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive |
| \(a\) | 5.43 Å |
| Basis atoms | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| SCF k-mesh | \(2\times 2\times 2\) (`PW::k_mesh = 2`) |
| XC | `XCType::LSDA` |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 4` |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 80 |
| `mix_beta` | 0.5 |
| DOS | MP `k_mesh=4`, `n_bands=8`, \(\sigma=0.05\) Ry, 200 bins |

### Accuracy table

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| SCF | `scf_converged == true` | — | — | FD free-energy SCF |
| \(E_F\) | finite | finite | \(\lvert E_F\rvert < 10^3\) Ry | DOS-mesh \(\mu\) |
| \(n_e(E_F)\) | \(\approx 8\) | \(n_e=8\) | **±1.0** | `n_elec_to_vbm` under smear = \(\int^{E_F} D\) |
| Measured \(n_e(E_F)\) (informative) | **≈ 8.01** | 8 | inside tol | REFERENCE host 2026-07-26, Release |
| Measured \(E_F\) (informative) | **≈ 0.465 Ry** | — | — | sits in gap for this twin |
| Measured \(F\) (informative) | **≈ −15.686 Ry** | — | not a CP2K twin | SCF free energy, 6 iters |
| Gap on DOS mesh | `gap_ry > 0` | open insulator | strict | gapped Si + FD, not metal collapse |
| \(E_F\) placement | in \([E_\mathrm{vbm},E_\mathrm{cbm}]\) | textbook Si | **±4×degauss** | \(\mathrm{degauss}=0.02\) Ry |
| DOS grid | non-empty, `energy_ry.size()==dos.size()` | — | — | 200 bins default |

**Incomplete-basis rationale.** Twin ecut and SCF k match the `bands-dos-si`
tutorial class. The product gate is occupation conservation under the metallic
DOS path, plus an open gap and a physically placed \(E_F\) — not absolute
eigenvalue or \(D(E)\) freeze against CP2K.

**CP2K.** Quickstep `BAND_STRUCTURE` / DOS use **GTH-PADE + DZVP-GTH**, not
ONCV plane waves. Absolute \(\varepsilon_n\) / \(D(E)\) are **not** comparable.
Product counterpart: `&SMEAR METHOD FERMI_DIRAC` with
\(k_B T \approx 0.02\) Ry (\(\approx 3158\) K), `ADDED_MOS ≥ 4`, denser k-mesh
DOS print. Offline note:

`tools/cp2k_oracle/inputs/dos-metallic-si.note`

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + SMEAR FERMI_DIRAC on diamond Si |
| CP2K binary this host | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | occupation integral \(\approx n_e\); metallic DOS path enabled |

## Performance

Source: `regressions/pw/dos-metallic-si/REFERENCE.md`. Queue / handoff row:
**`perf_exempt`** (no fair GTH twin wall on pin host).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si metallic DOS ecut=15 / scf k=2 + FD + DOS k=4 | n/a | ≈ 23.5 (informative) | n/a | n/a (**exempt**) | No CP2K module on pin host; absolute \(D(E)\) not a CP2K twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~23.5 s (Release, 8 OMP, ecut 15 / k=2 + DOS k=4 + FD) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a CP2K binary is available offline: time the same geometry with the
skeleton in `tools/cp2k_oracle/inputs/dos-metallic-si.note`, append
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to REFERENCE.md, require \(r\le 10\).
Wall comparison is same-geometry product counterpart only, not absolute
DOS parity (GTH+Gaussian vs NC-UPF PW).

## Limitations

- Tutorial ecut / SCF k only — not production-converged metallic DOS.
- Geometry is **gapped Si** under FD: validates the metallic occupation **path**
  and \(\int^{E_F} D \approx n_e\), not a vanishing-gap metal spectrum
  (use `pw-smearing-al` for free-energy metal pins).
- `Bands` still rejects metallic smearing; high-symmetry-path bands under FD
  are a follow-on, not this slug.
- No spin-resolved DOS; collinear spin + smear DOS unsupported on `Dos`.
- Absolute \(\varepsilon_n\) / \(D(E)\) not comparable to CP2K Quickstep GTH decks.
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); this
  regression always uses the in-tree ONCV Si UPF.
- CUDA PW surface is not a CPU compatibility API for this pin.
- No MPI multi-rank scaling gate (`mpi-pw-scaling-4rank` separate).

## References

- Offline pin: [`regressions/pw/dos-metallic-si/REFERENCE.md`](../../../regressions/pw/dos-metallic-si/REFERENCE.md)
- Regression: [`regressions/pw/dos-metallic-si/reg_dos-metallic-si.cpp`](../../../regressions/pw/dos-metallic-si/reg_dos-metallic-si.cpp)
- Public API: [`include/atomiverse/cpu/planewave/bands_dos.h`](../../../include/atomiverse/cpu/planewave/bands_dos.h)
- Offline CP2K twin note: `tools/cp2k_oracle/inputs/dos-metallic-si.note`
- Oracle rules: `tools/cp2k_oracle/README.md` (never live in ctest)
- Related fixed-occ bands+DOS: [bands-dos-si](bands-dos-si.md)
- Related metallic SCF (Al free energy): [pw-smearing-al](pw-smearing-al.md)
- Related metallic DFPT: [dfpt-metallic-si](dfpt-metallic-si.md) (if present)
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
- M. Methfessel and A. T. Paxton, *Phys. Rev. B* **40**, 3616 (1989)
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965)
