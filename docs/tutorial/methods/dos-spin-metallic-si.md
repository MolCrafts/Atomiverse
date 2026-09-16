# Collinear-spin + metallic FD plane-wave DOS — bulk silicon

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dos-spin-metallic-si` |
| Regression | `regressions/pw/dos-spin-metallic-si/` |
| Public API | `cpu::pw::PW` (`spin_polarized`, `multiplicity`, `smearing`, `degauss` [Ry], `extra_bands`) + `cpu::pw::Dos` + `DosResult` |
| Roadmap | [dft.md](../../roadmap/dft.md) § bands/DOS (collinear spin + metallic smear DOS)|
## Summary

**Collinear spin-polarized** plane-wave Kohn–Sham DFT with **Fermi–Dirac metallic
occupation smearing** on the SCF density, followed by a **Gaussian-broadened
density of states** for bulk silicon (diamond, 2-atom fcc primitive cell).
Self-consistent α/β densities come from a spin-polarized LDA SCF
(`spin_polarized = true`, `multiplicity = 1` seeds \(M = 0\); magnetization is
free under smear) plus `Smearing::fermi_dirac` / `degauss` / `extra_bands` on a
norm-conserving ONCVPSP UPF. Post-SCF, each spin channel is diagonalized
**non-self-consistently** on a denser Monkhorst–Pack mesh with frozen
\(V_{\mathrm{eff},\sigma}\); \(\mu\) is recomputed on the DOS mesh; the product
reports total \(D = D_\alpha + D_\beta\) (spin-orbital weight 1), per-channel
curves, the occupied integral \(n_\mathrm{elec}(E_F)\) (field
`n_elec_to_vbm`), \(E_F\), and an open gap on the DOS mesh. Scientific gates:
SCF converges, \(nspin=2\), \(\lvert M\rvert\le 0.1\),
\(\lvert n_\mathrm{elec}(E_F)-8\rvert\le 1\), \(gap\_ry > 0\), \(E_F\) near the
gap window \(\pm 4\sigma\), and \(D \approx D_\alpha + D_\beta\). Absolute
\(D(E)\) is **not** pinned to CP2K Quickstep (GTH+Gaussian vs ONCV-PW). Product
regression: `regressions/pw/dos-spin-metallic-si` (`reg_dos-spin-metallic-si`).
Performance: **`perf_exempt`** (no fair CP2K PW twin wall on the pin host).

## Method

**Level.** Periodic plane-wave Kohn–Sham DFT, **collinear spin**
(\(nspin=2\)) **and** metallic occupations. Self-consistent spin densities and
spin-dependent effective potentials from the same PW stack as
`pw-uks-fe-smearing` / DFPT spin+smear (Hartree in \(G\)-space, LSDA XC on the
FFT grid, Davidson per spin, NC UPF), with **fractional occupations** driven by

\[
f(x)=\frac{1}{1+e^{x}},\qquad
x=\frac{\varepsilon-\mu}{\sigma},\qquad
\sigma=\texttt{degauss}\ \mathrm{[Ry]}.
\]

Collinear spin-orbitals use weight 1 per spin channel (not the closed-shell
doubling of restricted metallic SCF). Shared chemical potential \(\mu\) is
fixed by electron-count bisection so \(n_e = Z_\mathrm{val}N_\mathrm{atoms}=8\)
for the two-Si cell. Multiplicity \(2S+1 = 1\) **seeds**
\(N_\uparrow = N_\downarrow = 4\); under smear \(M = N_\uparrow - N_\downarrow\)
is free and remains \(\approx 0\) on gapped Si. Extra empty bands for Davidson
near \(E_F\) come from `extra_bands` (here 4). SCF total is the **Mermin free
energy** \(F = E_\mathrm{KS} - TS\) when smearing is on
(`DosResult::scf_energy_ry`).

**DOS (collinear + metallic).** After SCF, freeze \(V_{\mathrm{eff},\sigma}\)
from the mixed spin densities; recompute eigenvalues on a denser MP mesh
(`Dos::k_mesh`³); form a Gaussian-broadened spectrum. Uniform MP mesh; per-spin
Gaussian kernel with spin-orbital weight 1:

\[
D_\sigma(E_i)
  = \sum_{n\mathbf{k}} w_{\mathbf{k}}\,
    \frac{1}{\sqrt{2\pi}\,\sigma}\,
    \exp\!\Bigl(-\frac{(E_i-\varepsilon_{n\mathbf{k}}^\sigma)^2}{2\sigma^2}\Bigr),
\qquad
D(E) = D_\alpha(E) + D_\beta(E).
\]

Units: \(D(E)\) in states / Ry / cell. Chemical potential is **recomputed on the
DOS mesh**; the occupied integral is to \(E_F\) (not a fixed-occupation VBM):

\[
n_\mathrm{elec}(E_F)
  = \int_{-\infty}^{E_F} D(E)\,dE
  \approx n_e = 8.
\]

The product field remains `DosResult::n_elec_to_vbm` (historical name); for this
path it is the integral to \(E_F\). On \(M\approx 0\), each channel holds
\(\approx 4\) electrons to the same \(E_F\).

**Gap and \(E_F\) placement (gapped twin).** `DosResult::gap_ry` =
\(E_\mathrm{cbm} - E_\mathrm{vbm}\) over the DOS mesh eigenvalues (both spins).
Diamond Si remains an open-gap insulator on the tutorial mesh; near-integer FD
occupations leave \(E_F\) in the gap. The regression requires \(gap\_ry > 0\) and

\[
E_\mathrm{vbm}-4\sigma \le E_F \le E_\mathrm{cbm}+4\sigma,
\qquad \sigma=\texttt{degauss}.
\]

The twin is a **spin + metallic DOS code path** on an insulator, not a claim
that Si is a metal or that the gap vanishes.

**XC / spin.** `XCType::LSDA` (Slater + PW92) on the PW grid; collinear LSDA
(not noncollinear, not spinor) **with** Fermi–Dirac smear.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(D(E)\) | `DosResult::dos` (total) | 1/Ry/cell |
| \(D_\alpha, D_\beta\) | `dos_alpha` / `dos_beta` | 1/Ry/cell |
| \(nspin\) | 2 when collinear | — |
| \(M\) | `magnetization` \(= N_\uparrow - N_\downarrow\) | electrons |
| \(n_\mathrm{elec}(E_F)\) | `n_elec_to_vbm` (integral to \(E_F\)) | electrons/cell |
| \(E_F\) | `e_fermi_ry` (DOS-mesh \(\mu\)) | Ry |
| \(E_\mathrm{vbm}, E_\mathrm{cbm}\) | `e_vbm_ry` / `e_cbm_ry` | Ry |
| \(E_g\) | `gap_ry` | Ry |
| \(F\) | `scf_energy_ry` (Mermin free energy) | Ry |

Conversion: `cpu/planewave/qe_units.h` (`rydberg_to_ev = 13.605693`). Internal
Ry/Ha are fine when cited; public single-point energy on Frame remains eV
elsewhere in the product surface — this post-processor reports Ry on
`DosResult`.

**Key exclusions (this slug).**

- Restricted (closed-shell) bands+DOS → `bands-dos-si`.
- Collinear fixed-occ DOS → `dos-spin-si`.
- Metallic restricted DOS → `dos-metallic-si`.
- Collinear spin + metallic **bands** (k-path) → `bands-spin-metallic-si`.
- Metallic restricted bands → `bands-metallic-si`.
- True metal energy pin (FCC Al / Fe) → `pw-smearing-al` / `pw-uks-fe-smearing`.
- GPW / Gaussian PDOS → CP2K character note only, not absolute \(D(E)\).
- Absolute \(\varepsilon_n\) / \(D(E)\) parity with CP2K Quickstep GTH+DZVP.
- Production-converged \(E_\mathrm{cut}\) (~60 Ry) / dense k when they break
  the ≤ 60 s e2e budget.
- Live CP2K at ctest time (offline pins only).
- Public `*Config` bags; `make_*` / `build_*` factories.
- Noncollinear / spinor KS; CUDA PW spin+smear DOS (CUDA PW remains
  LDA / spin-unpolarized).

## Implementation

Public surface only (`cpu::Frame` + concrete method / post-processor
members; no `*Config`, no factories). Headers:
`include/atomiverse/cpu/planewave/bands_dos.h`,
`pw_solver.h`, `qe_units.h`. Implementation:
`src/cpu/planewave/bands_dos.cpp` (collinear FD SCF → freeze
\(V_{\mathrm{eff},\sigma}\) → NSCF Davidson on DOS mesh → recompute \(\mu\) →
Gaussian \(D_\alpha, D_\beta, D\)).

Minimal sequence (matches `reg_dos-spin-metallic-si.cpp`):

```cpp
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/bands_dos.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;        // Å
constexpr double kEcutRy = 15.0;   // tutorial incomplete basis
constexpr int kMeshScf = 2;        // 2×2×2 MP SCF
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
method.spin_polarized = true;
method.multiplicity = 1;                 // seed M = 0; N↑ = N↓ = 4
method.smearing = Smearing::fermi_dirac;
method.degauss = kDegaussRy;
method.extra_bands = kExtraBands;

Dos dos;
dos.k_mesh = 4;
dos.n_bands = kNBands;
dos.sigma_ry = 0.05;
dos.n_bins = 200;
const DosResult dr = dos.compute(method, frame);
// dr.nspin == 2; dr.dos / dos_alpha / dos_beta;
// dr.n_elec_to_vbm ≈ 8 (integral to E_F); dr.e_fermi_ry;
// dr.gap_ry > 0; |dr.magnetization| ≤ 0.1
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
  cmake --build build --target reg_dos-spin-metallic-si -j$(nproc)
  ./build/regressions/reg_dos-spin-metallic-si
'
```

Unit (no full SCF e2e): `test_cpu_pw_bands_dos` — Gaussian integral identity,
spin-degeneracy kernel, synthetic gap finder (shared with restricted /
metallic / spin DOS units). Full collinear spin+smear SCF+DOS is the
regression only.

## Validation (vs literature / CP2K note)

Source: `regressions/pw/dos-spin-metallic-si/REFERENCE.md` (offline pin; never
live third party in ctest). Closed acceptance: queue/handoff
`dos-spin-metallic-si` done 2026-07-26 (`perf_exempt`; harness wave 12).

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Same family as | `regressions/pw/dos-spin-si`, `dos-metallic-si`, `bands-dos-si`, `pw-lda-si` |

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
| Spin | collinear, `multiplicity = 1` (seed \(N_\uparrow=N_\downarrow=4\)) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 4` |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 80 |
| `mix_beta` | 0.5 |
| DOS | MP `k_mesh=4`, `n_bands=8`, \(\sigma=0.05\) Ry, 200 bins |

### Accuracy table

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| SCF | `scf_converged == true` | — | — | collinear FD free-energy SCF |
| \(nspin\) | **2** | collinear | exact | `DosResult::nspin` |
| \(M = N_\uparrow-N_\downarrow\) | **0** (host) | singlet Si | **\(\lvert M\rvert\le 0.1\)** | free under smear; seed \(M=0\) |
| \(n_\mathrm{elec}(E_F)\) | **≈ 8.01** (host) | \(n_e=8\) | **±1.0** | integral to \(E_F\) (`n_elec_to_vbm`) |
| Per-spin occupied | \(n_\alpha, n_\beta \approx 4\) | \(M\approx 0\) | **±1.0** each | trap integral to \(E_F\) |
| \(D\) vs \(D_\alpha+D_\beta\) | \(\lvert\int(D_\alpha+D_\beta)-\int D\rvert\) to \(E_F\) | identity | **≤ 0.05** | shared energy grid |
| Gap on DOS mesh | **> 0** (`gap ≈ 0.035` Ry host) | open insulator | strict `gap_ry > 0` | incomplete ecut/k |
| \(E_F\) finite | \(\lvert E_F\rvert < 10^3\) Ry | DOS-mesh \(\mu\) | strict | `e_fermi_ry` |
| \(E_F\) placement | **≈ 0.473** Ry in \([E_\mathrm{vbm},E_\mathrm{cbm}]\pm 4\sigma\) | gapped FD | \(\sigma=0.02\) Ry | near-integer fill |
| \(F\) (informative) | **≈ −15.672** Ry | — | host note | not a CP2K energy twin |
| Channel lengths | `\|dos_alpha\|=\|dos_beta\|=\|dos\|` | — | exact | 200 bins |

**Incomplete-basis rationale.** Twin ecut and SCF k match the tutorial class of
`dos-spin-si` / `dos-metallic-si` / `bands-metallic-si`. The scientific gate is
**character** (collinear + metallic path on, \(M\approx 0\), electron count from
\(\int^{E_F} D\), open gap, \(E_F\) in gap window, channel consistency), not
freeze of Atomiverse-only \(D(E)\) digits or absolute eigenvalue parity with
GTH Quickstep.

**CP2K.** Quickstep `LSD` + `&SMEAR METHOD FERMI_DIRAC` + `PRINT%PDOS` uses
**GTH-PADE + DZVP-GTH**, not ONCV plane waves. Absolute \(D(E)\) /
\(\varepsilon_n\) are **not** comparable. Character pins (\(M\approx 0\),
occupied integral ≈ 8, open gap, path enabled) are the scientific twin.
Offline note:

`tools/cp2k_oracle/inputs/dos-spin-metallic-si.note`

Suggested twin: PADE, GTH-PADE-q4 + DZVP-GTH-q4, MP \(2\times 2\times 2\),
`LSD`, `MULTIPLICITY 1`, \(k_B T \approx 0.02\) Ry
(`ELECTRONIC_TEMPERATURE [K] 3157.75`), `ADDED_MOS ≥ 4`, `PDOS` with
`NLUMO 4`.

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + LSD + SMEAR FERMI_DIRAC + PDOS |
| CP2K binary (pin host) | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | \(\int^{E_F} D \approx n_e\); collinear + metallic DOS path enabled |

## Performance

Source: `regressions/pw/dos-spin-metallic-si/REFERENCE.md`. Queue / handoff
row: **`perf_exempt`** (character pins; no fair GTH twin wall on pin host).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si collinear+FD DOS ecut=15 / scf k=2 + DOS k=4 | n/a | ≈ 36.6 (informative) | n/a | n/a (**exempt**) | No CP2K module on pin host; absolute \(D(E)\) not a CP2K twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~36.6 s (Release, 8 OMP, ecut 15 / k=2 + DOS k=4 + collinear + FD, 2026-07-26) |
| Host measured (2026-07-26) | \(n_\mathrm{elec}(E_F)\approx 8.01\), \(E_F\approx 0.473\) Ry, gap ≈ 0.035 Ry, SCF 13 iters, \(F\approx-15.672\) Ry, \(mag=0\) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a CP2K binary is available offline: time the same geometry with
`tools/cp2k_oracle/inputs/dos-spin-metallic-si.note`, append
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to REFERENCE.md, require \(r\le 10\).
Wall comparison is same-geometry only, not absolute-\(D(E)\) parity
(GTH+Gaussian vs NC-UPF PW).

## Limitations

- Tutorial ecut / SCF k only — not production-converged dense spin+metallic DOS.
- Twin is **gapped Si** under FD smear (near-integer occupations); it does
  **not** pin a true metal vanishing gap (use `pw-smearing-al` /
  `pw-uks-fe-smearing` for metallic free energy).
- Integral field is still named `n_elec_to_vbm` but means \(\int^{E_F} D\) on
  this path.
- No k-path bands on this slug — use `bands-spin-metallic-si` for spin-resolved
  metallic-occupation \(\varepsilon_n(\mathbf{k})\).
- Absolute \(D(E)\) not comparable to CP2K Quickstep GTH decks.
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); this
  regression always uses the in-tree ONCV Si UPF.
- CUDA PW surface is not a CPU compatibility API for this pin (spin-unpolarized).
- No MPI multi-rank scaling gate on this regression.
- Gap value on a sparse tutorial DOS mesh is a **positivity** gate, not a
  Yin–Cohen LDA gap window (that lives on `bands-dos-si` for the Γ→X
  restricted path).

## References

- Offline pin: [`regressions/pw/dos-spin-metallic-si/REFERENCE.md`](../../../regressions/pw/dos-spin-metallic-si/REFERENCE.md)
- Regression: [`regressions/pw/dos-spin-metallic-si/reg_dos-spin-metallic-si.cpp`](../../../regressions/pw/dos-spin-metallic-si/reg_dos-spin-metallic-si.cpp)
- Public API: [`include/atomiverse/cpu/planewave/bands_dos.h`](../../../include/atomiverse/cpu/planewave/bands_dos.h)
- Optional CP2K character / wall note: `tools/cp2k_oracle/inputs/dos-spin-metallic-si.note`
- Sibling collinear fixed-occ DOS: [dos-spin-si](dos-spin-si.md)
- Sibling metallic restricted DOS: [dos-metallic-si](dos-metallic-si.md) (if present)
- Sibling metallic restricted bands: [bands-metallic-si](bands-metallic-si.md)
- Sibling restricted DOS: [bands-dos-si](bands-dos-si.md)
- Sibling collinear energy/spin SCF: [pw-spin-polarized-o2](pw-spin-polarized-o2.md)
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
- Closed acceptance (historical): queue row `dos-spin-metallic-si` 2026-07-26
  (`perf_exempt`; harness wave 12 handoff)
