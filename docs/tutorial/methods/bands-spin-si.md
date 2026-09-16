# Plane-wave collinear spin band structure — bulk silicon

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `bands-spin-si` |
| Regression | `regressions/pw/bands-spin-si/` |
| Public API | `cpu::pw::PW` (`spin_polarized`, `multiplicity`) + `cpu::pw::Bands` → `BandsResult` (`nspin`, `eigenvalues_ry` / `eigenvalues_beta_ry`, `magnetization`) |
| Roadmap | [dft.md](../../roadmap/dft.md) § bands/DOS + collinear spin bands|

## Summary

**Collinear spin-polarized** non-self-consistent **plane-wave band structure**
for bulk silicon (diamond, 2-atom fcc primitive cell) after a spin-polarized
LDA SCF on a norm-conserving ONCVPSP UPF. The product freezes each spin
channel’s \(V_{\mathrm{eff},\sigma}\) from `cpu::pw::PW` members
(`spin_polarized = true`, `multiplicity = 1` ⇒ \(M=0\)), then diagonalizes
along a high-symmetry **k-path** (`cpu::pw::Bands`). Observables are
\(\varepsilon_n^\sigma(\mathbf{k})\) [Ry] for \(\alpha/\beta\), the textbook
**Γ→X** gap [eV] on the α channel, collinear magnetization \(M\), and α≈β
parity at the VBM/CBM for closed-shell \(M=0\). Scientific gate is a literature
incomplete-basis window around the Yin–Cohen LDA indirect gap (~0.45 eV), plus
spin-channel consistency — not absolute eigenvalue parity with CP2K Quickstep
(GTH+Gaussian). Product regression: `regressions/pw/bands-spin-si`
(`reg_bands-spin-si`). Performance: **`perf_exempt`** (no fair CP2K PW twin wall
on the pin host). Sibling closed-shell path: `bands-dos-si`.

## Method

**Level.** Periodic plane-wave Kohn–Sham DFT with **collinear** spin (nspin=2).
Self-consistent spin densities and spin-dependent effective potentials from the
same PW stack as `pw-spin-polarized-o2` / spin-capable `PW` (Hartree from total
\(\rho=\rho_\uparrow+\rho_\downarrow\), LDA XC on the FFT grid with separate
\(V_{\mathrm{xc}}^\sigma\), dual Davidson diagonalizations, NC UPF). After SCF,
**bands** are **non-self-consistent**: each channel’s final
\(V_{\mathrm{eff},\sigma}\) is frozen; only the k-dependent kinetic/projectors
change while eigenvalues are recomputed per spin.

**Spin / occupations.** Fixed fillings from multiplicity \(2S+1\):

\[
N_\uparrow - N_\downarrow = 2S = m - 1,\qquad
N_\uparrow + N_\downarrow = N_{\mathrm{val}}.
\]

Silicon NC-PP \(Z_{\mathrm{val}}=4\) × 2 atoms → \(N_{\mathrm{val}}=8\);
`multiplicity = 1` → \(M=0\), \(N_\uparrow=N_\downarrow=4\). With fixed
closed-shell collinear occupations the α and β spectra are degenerate
(within numerical noise).

**Bands (k-path).** Fractional path \(\Gamma\)–\(X\)–\(W\)–\(L\)–\(\Gamma\)–\(K\)
via `Bands::set_diamond_path(n_seg)`. For each path \(\mathbf{k}\) and spin
\(\sigma\),

\[
H[V_{\mathrm{eff},\sigma}]\, \psi_{n\mathbf{k}}^\sigma
  = \varepsilon_n^\sigma(\mathbf{k})\, \psi_{n\mathbf{k}}^\sigma,
\qquad
\tfrac12|\mathbf{k}+\mathbf{G}|^2 \le E_\mathrm{cut},
\]

with `n_bands` states **per spin** (valence + some conduction). Silicon has
\(n_{\mathrm{occ},\sigma}=4\) occupied bands per channel when \(M=0\).

**Gap definitions.** The regression pins the **textbook Si transition** on the
**α** channel (β is checked for parity, not for a separate gap window):

\[
E_{\Gamma\to X}
  = \bigl(\varepsilon_{c}^\alpha(X) - \varepsilon_{v}^\alpha(\Gamma)\bigr)
    \times 13.605693~\mathrm{eV},
\qquad
\varepsilon_{v}^\alpha(\Gamma)=\varepsilon_{n_\mathrm{occ}-1}^\alpha(\Gamma),
\quad
\varepsilon_{c}^\alpha(\mathbf{k})=\varepsilon_{n_\mathrm{occ}}^\alpha(\mathbf{k}).
\]

Local (same-\(\mathbf{k}\)) gaps for **both** spins
\(\varepsilon_{n_\mathrm{occ}}^\sigma(\mathbf{k})-\varepsilon_{n_\mathrm{occ}-1}^\sigma(\mathbf{k})\)
must stay positive. Indirect character requires
\(\varepsilon_c^\alpha(X)<\varepsilon_c^\alpha(\Gamma)\).

**α≈β parity (\(M=0\)).** At Γ VBM and X CBM,

\[
\bigl|\varepsilon^\alpha - \varepsilon^\beta\bigr| \le 5\times 10^{-4}\,\mathrm{Ry}.
\]

**XC / spin.** `XCType::LSDA` (Slater + PW92) on the PW grid; collinear fixed
occupations (`Smearing::none`). No metallic smearing on this twin; no
spin-resolved DOS (see Limitations).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_n^\alpha(\mathbf{k})\) | `BandsResult::eigenvalues_ry` | Ry |
| \(\varepsilon_n^\beta(\mathbf{k})\) | `BandsResult::eigenvalues_beta_ry` | Ry |
| \(n_{\mathrm{spin}}\) | `BandsResult::nspin` (want 2) | — |
| \(M\) | \(N_\uparrow-N_\downarrow\) (`BandsResult::magnetization`) | e⁻ |
| \(E_{\Gamma\to X}\) | \((\varepsilon_c^\alpha(X)-\varepsilon_v^\alpha(\Gamma))\times 13.605693\) | eV |
| \(E_\mathrm{SCF}\) | `BandsResult::energy_ry` | Ry |

Conversion: `cpu/planewave/qe_units.h` (`rydberg_to_ev = 13.605693`). Public
Frame energy remains eV when written through Evaluate/meta; this post-processor
reports SCF energy in Ry on `BandsResult` (internal, cited).

**Key exclusions (this slug).**

- Closed-shell bands + DOS → `bands-dos-si`.
- Spin-resolved / metallic DOS → residual / later queue (Dos rejects collinear
  today; `dos-metallic-si` is the metallic restricted path).
- Molecular PW UKS energy foothold → `pw-spin-polarized-o2`; k-mesh UKS metal →
  `pw-uks-kmesh-fe`.
- Berry phase with spin → `berry-spin-si`; DFPT spin → `dfpt-spin-si` and kin.
- Absolute \(\varepsilon_n\) parity with CP2K Quickstep GTH+DZVP (different PP
  and basis).
- Production-converged \(E_\mathrm{cut}\) (~60 Ry) / dense path when they break
  the ≤ 60 s e2e budget.
- Live CP2K / QE / ABINIT at ctest time (offline pins only).
- Public `*Config` bags; `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + concrete method / post-processor members;
no `*Config`, no factories). Headers:
`include/atomiverse/cpu/planewave/bands_dos.h`,
`pw_solver.h`, `qe_units.h`. Implementation:
`src/cpu/planewave/bands_dos.cpp` (collinear SCF → freeze
\(V_{\mathrm{eff},\sigma}\) → NSCF Davidson on path per spin).

Minimal sequence (matches `reg_bands-spin-si.cpp`):

```cpp
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/bands_dos.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;       // Å
constexpr double kEcutRy = 15.0;  // tutorial incomplete basis
constexpr int kMeshScf = 2;       // 2×2×2 MP SCF
constexpr int kNBands = 8;
constexpr int kNOcc = 4;          // Si: n_e = 8, M = 0 → N↑ = N↓ = 4

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
method.xc = cpu::scf::XCType::LSDA;
method.spin_polarized = true;
method.multiplicity = 1;              // M = 0 closed-shell collinear

Bands bands;
bands.n_bands = kNBands;
bands.set_diamond_path(4);            // Γ–X–W–L–Γ–K
const BandsResult br = bands.compute(method, frame);
// br.nspin == 2; br.eigenvalues_ry (α); br.eigenvalues_beta_ry (β)
// br.magnetization ≈ 0; br.scf_converged
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
  cmake --build build --target reg_bands-spin-si -j$(nproc)
  ./build/regressions/reg_bands-spin-si
'
```

Unit coverage for the shared path machinery (restricted spectrum helpers,
diamond path endpoints) lives under `test_cpu_pw_bands_dos`; the collinear e2e
gate is **only** the regression above (no full SCF e2e under `tests/`).

## Validation (vs literature / CP2K note)

Source: `regressions/pw/bands-spin-si/REFERENCE.md` (offline pin; never live
third party in ctest). Queue row closed 2026-07-26 (``).

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Same family as | `regressions/pw/bands-dos-si`, `regressions/pw/pw-lda-si` |

### Literature pin (indirect gap)

Bulk Si is an **indirect** gap semiconductor: VBM at \(\Gamma\), CBM near
\(X\). With \(M=0\) collinear occupations the α and β channels are degenerate;
the scientific pins are the same textbook transition plus α≈β parity.

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Experimental \(E_g\) | ~1.12 eV | textbook |
| Converged LDA \(E_g\) (Γ\(_v\) → near-\(X_c\)) | **~0.45 eV** | Yin & Cohen, *Phys. Rev. B* **26**, 5668 (1982) |

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
| Spin | collinear, `multiplicity = 1` (\(N_\uparrow=N_\downarrow=4\)) |
| SCF tol | \(10^{-6}\) (density L2) |
| `max_iter` | 80 |
| Bands path | Γ–X–W–L–Γ–K, `n_seg=4`, `n_bands=8` **per spin** |

### Accuracy table

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| LDA \(E_g\) (converged lit.) | — | **~0.45 eV** | literature | Yin–Cohen PRB 26, 5668 (1982) |
| Exp. \(E_g\) | — | ~1.12 eV | context only | LDA underestimates |
| \(E_{\Gamma\to X}\) (α, twin) | band gate | Yin–Cohen window | **[0.05, 2.50] eV** | incomplete ecut/k; rejects metallic collapse / multi-eV disasters |
| \(n_{\mathrm{spin}}\) | **2** | collinear | exact | `BandsResult::nspin` |
| \(\lvert M\rvert\) | \(\le 10^{-6}\) | \(M=0\) fixed occ | **1e-6** | closed-shell collinear |
| Local same-\(k\) gap (α and β) | \(> 0\) | open insulator | strict | every path k, both spins |
| Indirect character | \(\varepsilon_c^\alpha(X)<\varepsilon_c^\alpha(\Gamma)\) | Si textbook | boolean | path samples include Γ and X |
| α≈β at Γ VBM, X CBM | \(\lvert\Delta\varepsilon\rvert\le 5\times 10^{-4}\) Ry | \(M=0\) degeneracy | **5e-4 Ry** | hard-coded in `reg_bands-spin-si` |
| SCF | `scf_converged == true` | — | — | preceding collinear SCF |
| Eigenvalue order | sorted ascending per (spin, k) | — | — | 8 bands each channel |

**Incomplete-basis rationale.** Twin ecut and SCF k are deliberately light
(same class as `bands-dos-si` / `pw-lda-si`). The gate is a **literature-anchored
window** on the α-channel Γ→X transition plus spin-channel consistency, not a
freeze of Atomiverse-only digits. The wide band allows residual path scatter
without accepting a metallic or multi-eV disaster.

**CP2K.** Quickstep `BAND_STRUCTURE` with `LSD` / UKS uses **GTH-PADE +
DZVP-GTH**, not ONCV plane waves. Absolute \(\varepsilon_n\) are **not**
comparable. Character (indirect Si, \(\varepsilon_c(X)<\varepsilon_c(\Gamma)\)),
\(M=0\), and the Yin–Cohen LDA Γ→X window are the scientific pins. Optional
offline wall / character skeleton:
`tools/cp2k_oracle/inputs/bands-spin-si.note` (same geometry, \(2\times 2\times 2\)
MP, `LSD` + `MULTIPLICITY 1`, path Γ–X–W–L–Γ–K with `NPOINTS 4`).

## Performance

Source: `regressions/pw/bands-spin-si/REFERENCE.md`. Queue / handoff row:
**`perf_exempt`** (literature Γ→X window + α≈β; no fair GTH twin wall on pin
host).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si collinear bands ecut=15 / scf k=2 + path | n/a | ≈ 16 (informative) | n/a | n/a (**exempt**) | No CP2K module on pin host; absolute \(\varepsilon_n\) not a CP2K twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~16 s (Release, OMP=8, ecut 15 / k=2 + collinear path, 2026-07-26) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a CP2K binary is available offline: time the same geometry with
`tools/cp2k_oracle/inputs/bands-spin-si.note`, append
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to REFERENCE.md, require \(r\le 10\).
Wall comparison is same-geometry only, not absolute-eigenvalue parity
(GTH+Gaussian vs NC-UPF PW).

## Limitations

- Tutorial ecut / SCF k only — not production-converged band gaps.
- Global \(\min\varepsilon_c-\max\varepsilon_v\) over a **sparse** path can
  spuriously go slightly negative at incomplete \(E_\mathrm{cut}\) from
  path-point scatter; the product gate therefore uses **Γ→X (α)** and local
  same-\(k\) positivity on both spins (see Method / Validation).
- No spin-resolved DOS, metallic smearing, or GPW k-path on this slug.
- No Berry / Wannier / DFPT response here — those are separate queue rows
  (`berry-spin-si`, `dfpt-spin-si`, …).
- Absolute \(\varepsilon_n\) not comparable to CP2K Quickstep GTH decks.
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); this
  regression always uses the in-tree ONCV Si UPF.
- CUDA PW surface is not a CPU compatibility API for this pin.
- No MPI multi-rank scaling gate (`mpi-pw-scaling-4rank` is a separate slug).

## References

- Offline pin: [`regressions/pw/bands-spin-si/REFERENCE.md`](../../../regressions/pw/bands-spin-si/REFERENCE.md)
- Regression: [`regressions/pw/bands-spin-si/reg_bands-spin-si.cpp`](../../../regressions/pw/bands-spin-si/reg_bands-spin-si.cpp)
- Public API: [`include/atomiverse/cpu/planewave/bands_dos.h`](../../../include/atomiverse/cpu/planewave/bands_dos.h)
- Optional CP2K wall / character skeleton: `tools/cp2k_oracle/inputs/bands-spin-si.note`
- Closed-shell sibling: [bands-dos-si](bands-dos-si.md)
- Collinear energy footholds: [pw-spin-polarized-o2](pw-spin-polarized-o2.md),
  [pw-uks-kmesh-fe](pw-uks-kmesh-fe.md)
- Related energy foothold: [pw-lda-si](pw-lda-si.md)
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)
- M. T. Yin and M. L. Cohen, *Phys. Rev. B* **26**, 5668 (1982)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
