# Collinear-spin metallic-occupation band structure — diamond Si / plane-wave

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `bands-spin-metallic-si` |
| Regression | `regressions/pw/bands-spin-metallic-si/` |
| Public API | `cpu::pw::PW` members `spin_polarized`, `multiplicity`, `smearing`, `degauss` [Ry], `extra_bands` + `cpu::pw::Bands` / `BandsResult::{nspin,eigenvalues_ry,eigenvalues_beta_ry,e_fermi_ry,gap_ry,e_vbm_ry,e_cbm_ry,magnetization,energy_ry}` |
| Roadmap | [dft.md](../../roadmap/dft.md) § bands + collinear spin + metallic occupations|
## Summary

Non-self-consistent **plane-wave band structure** for bulk silicon after a
**collinear spin-polarized** LDA SCF with **Fermi–Dirac metallic occupation
smearing**. The product freezes \(V_{\mathrm{eff},\sigma}\) from `cpu::pw::PW`
(\(nspin=2\), fractional occupations, Mermin free energy \(F\)), then
diagonalizes each spin channel on the diamond high-symmetry **k-path** via
`cpu::pw::Bands`. Twin geometry matches `bands-spin-si` / `bands-metallic-si`
/ `bands-dos-si` (diamond Si, \(a=5.43\) Å, NC UPF, ecut 15 Ry, SCF
\(2\times2\times2\)). Multiplicity seed \(M=0\) keeps α≈β and near-zero
magnetization; Si stays **gapped** at these settings so near-integer FD
occupations leave \(E_F\) in the gap. The scientific gate is the
**spin+smear bands path** (no throw) + open integer-fill spectrum + finite
\(E_F\) near \([E_\mathrm{vbm},E_\mathrm{cbm}]\) + Yin–Cohen-window Γ→X (α)
+ α≈β parity — not a vanishing metal-like gap. Product regression:
`regressions/pw/bands-spin-metallic-si` (`reg_bands-spin-metallic-si`).
Performance: **`perf_exempt`** (no fair CP2K PW twin wall on the pin host).

## Method

**Level.** Periodic plane-wave Kohn–Sham DFT with **collinear spin**
(\(nspin=2\)). Self-consistent spin densities and spin-dependent effective
potentials from the same PW stack as `bands-spin-si` / `pw-uks-fe-smearing`
(Hartree in \(G\)-space, LSDA XC on the FFT grid, Davidson per spin, NC UPF),
with **fractional occupations** driven by

\[
f(x)=\frac{1}{1+e^{x}},\qquad
x=\frac{\varepsilon-\mu}{\sigma},\qquad
\sigma=\texttt{degauss}\ \mathrm{[Ry]}.
\]

Per spin channel, \(f_\sigma\in[0,1]\) (spin-orbital weight 1). Fermi level
\(\mu\) is fixed by electron-count bisection so
\(n_e=Z_\mathrm{val}N_\mathrm{atoms}=8\) for the two-Si cell. With
`multiplicity = 1` the seed is \(M=N_\uparrow-N_\downarrow=0\); under FD smear
\(M\) is free and must relax to \(\lvert M\rvert\le 10^{-3}\) on this twin.
Extra empty bands for Davidson near \(E_F\) come from `extra_bands` (here 4).
SCF total is the **Mermin free energy** \(F=E_\mathrm{KS}-TS\) when smearing
is on (`BandsResult::energy_ry`).

**Bands (k-path, NSCF).** After SCF, each \(V_{\mathrm{eff},\sigma}\) is frozen.
Fractional path \(\Gamma\)–\(X\)–\(W\)–\(L\)–\(\Gamma\)–\(K\) via
`Bands::set_diamond_path(n_seg)`. For each path \(\mathbf{k}\) and spin
\(\sigma\in\{\alpha,\beta\}\),

\[
H[V_{\mathrm{eff},\sigma}]\, \psi_{n\mathbf{k}}^\sigma
  = \varepsilon_n^\sigma(\mathbf{k})\, \psi_{n\mathbf{k}}^\sigma,
\qquad
\tfrac12|\mathbf{k}+\mathbf{G}|^2 \le E_\mathrm{cut},
\]

with `n_bands` states per spin (valence + conduction). Path k-points are
**not** a BZ mesh — \(\mu\) reported on `BandsResult::e_fermi_ry` is the
**SCF** value, not re-binned from path weights. α eigenvalues land in
`eigenvalues_ry`; β in `eigenvalues_beta_ry`; `nspin == 2`.

**Gap definitions (gapped twin).** Silicon has \(n_e=8\). With \(M\approx 0\)
seed the product uses the **integer-fill** cut \(n_\mathrm{occ}=4\) per spin
even though occupations were fractional during SCF:

\[
E_g = E_\mathrm{cbm}-E_\mathrm{vbm}
    \quad [\mathrm{Ry}],
\]

and local same-\(\mathbf{k}\) gaps
\(\varepsilon_{n_\mathrm{occ}}^\sigma(\mathbf{k})
-\varepsilon_{n_\mathrm{occ}-1}^\sigma(\mathbf{k})>0\) on every path sample
and both spins (no inverted spectrum). \(E_F\) must sit near the gap window:

\[
E_\mathrm{vbm}-4\sigma \le E_F \le E_\mathrm{cbm}+4\sigma,
\qquad \sigma=\texttt{degauss}.
\]

Textbook **Γ→X** transition on the α channel (same literature window as
`bands-dos-si` / `bands-spin-si`):

\[
E_{\Gamma\to X}^\alpha
  = \bigl(\varepsilon_{c}^\alpha(X) - \varepsilon_{v}^\alpha(\Gamma)\bigr)
    \times 13.605693~\mathrm{eV},
\qquad
0.05 \le E_{\Gamma\to X}^\alpha \le 2.50~\mathrm{eV},
\]

with indirect character \(\varepsilon_c^\alpha(X)<\varepsilon_c^\alpha(\Gamma)\).
α≈β parity at Γ VBM and X CBM:
\(\lvert\varepsilon^\alpha-\varepsilon^\beta\rvert \le 10^{-3}\) Ry.

Gapped Si at tutorial ecut/k keeps near-integer FD occupations; the twin is a
**collinear spin + metallic-occupation code path** on an insulator, not a
claim that Si is a metal or a ferromagnet.

**XC / spin.** `XCType::LSDA` (Slater + PW92) on the PW grid; collinear
(`spin_polarized = true`) **and** metallic smear (`Smearing::fermi_dirac`).
Restricted-only metallic bands → `bands-metallic-si`. Fixed-occupation
collinear bands → `bands-spin-si`. Spin+smear DOS → `dos-spin-metallic-si`.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_n^\alpha(\mathbf{k})\) | `BandsResult::eigenvalues_ry` | Ry |
| \(\varepsilon_n^\beta(\mathbf{k})\) | `BandsResult::eigenvalues_beta_ry` | Ry |
| \(E_F\) | `BandsResult::e_fermi_ry` (SCF μ) | Ry |
| \(E_\mathrm{vbm}\), \(E_\mathrm{cbm}\) | path extrema at integer fill | Ry |
| \(E_g\) | `BandsResult::gap_ry` | Ry |
| \(M\) | `BandsResult::magnetization` | electrons |
| \(E_{\Gamma\to X}^\alpha\) | \((\varepsilon_c^\alpha(X)-\varepsilon_v^\alpha(\Gamma))\times 13.605693\) | eV |
| \(F\) | `BandsResult::energy_ry` (Mermin free energy) | Ry |

Conversion: `cpu/planewave/qe_units.h` (`rydberg_to_ev = 13.605693`).

**Key exclusions (this slug).**

- Fixed-occupation insulator bands + DOS → `bands-dos-si`.
- Collinear spin bands (fixed occ) → `bands-spin-si`.
- Restricted metallic bands → `bands-metallic-si`.
- Spin+smear DOS → `dos-spin-metallic-si`.
- Collinear fixed-occ DOS → `dos-spin-si`.
- True metal energy pin (FCC Al) → `pw-smearing-al`.
- UKS + FD SCF without k-path → `pw-uks-fe-smearing`.
- GPW / Gaussian k-path → `gpw-kpoints-si`.
- Absolute \(\varepsilon_n\) parity with CP2K Quickstep GTH+DZVP (different PP
  and basis).
- Production-converged \(E_\mathrm{cut}\) (~60 Ry) / dense path when they break
  the ≤ 60 s e2e budget.
- Live CP2K / QE / ABINIT at ctest time (offline pins only).
- Public `*Config` bags; `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + concrete method / post-processor
members; no `*Config`, no factories). Headers:
`include/atomiverse/cpu/planewave/bands_dos.h`,
`pw_solver.h`, `qe_units.h`. Implementation: `src/cpu/planewave/bands_dos.cpp`
(collinear metallic SCF → freeze \(V_{\mathrm{eff},\sigma}\) → NSCF Davidson on
path per spin).

Minimal sequence (matches `reg_bands-spin-metallic-si.cpp`):

```cpp
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/bands_dos.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
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
method.multiplicity = 1;              // seed M=0; free under smear
method.smearing = Smearing::fermi_dirac;
method.degauss = kDegaussRy;
method.extra_bands = kExtraBands;

Bands bands;
bands.n_bands = kNBands;
bands.set_diamond_path(4);              // Γ–X–W–L–Γ–K
const BandsResult br = bands.compute(method, frame);
// br.nspin == 2; br.eigenvalues_ry (α), br.eigenvalues_beta_ry (β)
// br.e_fermi_ry; br.gap_ry; br.magnetization; br.energy_ry = Mermin F
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
  cmake --build build --target reg_bands-spin-metallic-si -j$(nproc)
  ./build/regressions/reg_bands-spin-metallic-si
'
```

Unit (no full SCF e2e): spin+smear bands path is covered via the
`bands_dos` product surface; full collinear metallic SCF+path is the
regression only.

## Validation (vs literature / CP2K note)

Source: `regressions/pw/bands-spin-metallic-si/REFERENCE.md` (offline pin;
never live third party in ctest). Queue / handoff: done 2026-07-26,
`perf_exempt`.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Same family as | `regressions/pw/pw-lda-si`, `bands-dos-si`, `bands-spin-si`, `bands-metallic-si` |

### Literature pin (indirect Γ→X gap)

Bulk Si is an **indirect** gap semiconductor: VBM at \(\Gamma\), CBM near
\(X\).

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Experimental \(E_g\) | ~1.12 eV | textbook |
| Converged LDA \(E_g\) (Γ\(_v\) → near-\(X_c\)) | **~0.45 eV** | Yin & Cohen, *Phys. Rev. B* **26**, 5668 (1982) |

With \(M=0\) seed and mild FD smear, α and β stay nearly degenerate; \(E_F\)
sits in the open Si gap (near-integer occupations).

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
| Spin | collinear, `multiplicity = 1` (seed \(M=0\)) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 4` |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 80 |
| Bands path | Γ–X–W–L–Γ–K, `n_seg=4`, `n_bands=8` per spin |

### Accuracy table

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| SCF | `scf_converged == true` | — | — | collinear metallic free-energy SCF |
| `nspin` | 2 | collinear | strict | α + β channels |
| \(\lvert M\rvert\) | \(\le 10^{-3}\) | seed \(M=0\) | strict | free under smear |
| Path endpoints | includes Γ and X | diamond path | boolean | `set_diamond_path` |
| Eigenvalue order | sorted ascending per (k, spin) | — | — | 8 bands/spin |
| Local same-\(k\) gap | \(\varepsilon_4^\sigma-\varepsilon_3^\sigma > 0\) | open insulator | strict | every path k, both spins |
| Integer-fill \(E_g\) | `gap_ry > 0` | open Si | strict | not a metal-collapse gate |
| \(E_F\) finite | \(\lvert E_F\rvert < 10^3\) Ry | SCF μ | strict | `e_fermi_ry` |
| \(E_F\) placement | in \([E_\mathrm{vbm},E_\mathrm{cbm}]\pm 4\sigma\) | gapped FD | \(\sigma=0.02\) Ry | near-integer fill |
| Γ→X (α) | band gate | Yin–Cohen window | **[0.05, 2.50] eV** | incomplete ecut/k |
| Indirect character | \(\varepsilon_c^\alpha(X)<\varepsilon_c^\alpha(\Gamma)\) | Si textbook | boolean | path samples Γ and X |
| α≈β | \(\lvert\Delta\varepsilon\rvert\le 10^{-3}\) Ry | \(M\approx 0\) | at Γ VBM, X CBM | spin parity |
| Measured twin (informative, 2026-07-26) | \(E_F\approx 0.465\) Ry, \(E_g\approx 0.043\) Ry, mag=0, Γ→X(α)≈0.58 eV, \(F\approx -15.686\) Ry | SCF | host note | REFERENCE Release |

**Incomplete-basis rationale.** Twin ecut and SCF k are deliberately light
(same class as `pw-lda-si` / `bands-spin-si` / `bands-metallic-si`). The gate
is the **collinear spin + metallic bands code path** plus open spectrum /
\(E_F\) placement / Γ→X window / α≈β on gapped Si — not a freeze of
Atomiverse-only free-energy digits as a literature metal property, and not
absolute eigenvalue parity with CP2K GTH+Gaussian.

**CP2K.** Quickstep `BAND_STRUCTURE` with `LSD` + `&SMEAR METHOD FERMI_DIRAC`
uses **GTH-PADE + DZVP-GTH**, not ONCV plane waves. Absolute \(\varepsilon_n\)
are **not** comparable. Product counterpart (character: collinear spin +
metallic occupations enabled; open Si gap; \(E_F\) in gap; α≈β) is documented
offline:

`tools/cp2k_oracle/inputs/bands-spin-metallic-si.note`

Suggested twin: PADE, GTH-PADE-q4 + DZVP-GTH-q4, `LSD` / `MULTIPLICITY 1`,
MP \(2\times2\times2\), \(k_B T \approx 0.02\) Ry
(`ELECTRONIC_TEMPERATURE [K] 3157.75`), `ADDED_MOS ≥ 4`, path Γ–X–W–L–Γ–K.

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + LSD + SMEAR FERMI_DIRAC + BAND_STRUCTURE |
| CP2K binary (pin host) | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | spin+smear bands path; open Si gap; \(E_F\) in gap; α≈β |

## Performance

Source: `regressions/pw/bands-spin-metallic-si/REFERENCE.md`. Queue / handoff
row: **`perf_exempt`** (no fair GTH twin wall on pin host).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si spin+metallic bands ecut=15 / scf k=2 + path n_seg=4 + collinear FD | n/a | ≈ 45 (informative) | n/a | n/a (**exempt**) | No CP2K module on pin host; absolute \(\varepsilon_n\) not a CP2K twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~45 s (Release, ecut 15 / k=2 + path + collinear FD; measured 2026-07-26) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a CP2K binary is available offline: time the same geometry with
`tools/cp2k_oracle/inputs/bands-spin-metallic-si.note`, append
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to REFERENCE.md, require \(r\le 10\).
Wall comparison is same-geometry only, not absolute-eigenvalue parity
(GTH+Gaussian vs NC-UPF PW).

## Limitations

- Tutorial ecut / SCF k only — not production-converged spin+metallic band
  structure or dense path productization.
- Twin is **gapped Si** under FD smear with \(M\approx 0\) (near-integer
  occupations, α≈β); it does **not** pin a true metal vanishing gap or a
  finite-magnetization metal (use `pw-smearing-al` / `pw-uks-fe-smearing` for
  metallic SCF energy character).
- Path \(\mu\) is the SCF value; do not treat path-k weights as a BZ mesh
  for re-solving \(E_F\).
- No DOS on this slug — see `dos-spin-metallic-si` for spin+smear DOS.
- Absolute \(\varepsilon_n\) not comparable to CP2K Quickstep GTH decks.
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); this
  regression always uses the in-tree ONCV Si UPF.
- CUDA PW surface is not a CPU compatibility API for this pin.
- No MPI multi-rank scaling gate (`mpi-pw-scaling-4rank` later).

## References

- Offline pin: [`regressions/pw/bands-spin-metallic-si/REFERENCE.md`](../../../regressions/pw/bands-spin-metallic-si/REFERENCE.md)
- Regression: [`regressions/pw/bands-spin-metallic-si/reg_bands-spin-metallic-si.cpp`](../../../regressions/pw/bands-spin-metallic-si/reg_bands-spin-metallic-si.cpp)
- Public API: [`include/atomiverse/cpu/planewave/bands_dos.h`](../../../include/atomiverse/cpu/planewave/bands_dos.h)
- Optional CP2K wall / character skeleton: `tools/cp2k_oracle/inputs/bands-spin-metallic-si.note`
- Sibling fixed-occ spin bands: regression `regressions/pw/bands-spin-si`
- Sibling restricted metallic bands: [bands-metallic-si](bands-metallic-si.md)
- Sibling insulator bands+DOS: [bands-dos-si](bands-dos-si.md)
- Sibling spin DOS (fixed occ): [dos-spin-si](dos-spin-si.md)
- Related energy foothold: [pw-lda-si](pw-lda-si.md)
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)
- M. T. Yin and M. L. Cohen, *Phys. Rev. B* **26**, 5668 (1982)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
