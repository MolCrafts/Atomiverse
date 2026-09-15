# Metallic-occupation band structure — diamond Si / plane-wave

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `bands-metallic-si` |
| Regression | `regressions/pw/bands-metallic-si/` |
| Public API | `cpu::pw::PW` members `smearing`, `degauss` [Ry], `extra_bands` + `cpu::pw::Bands` / `BandsResult::{eigenvalues_ry,e_fermi_ry,gap_ry,e_vbm_ry,e_cbm_ry,energy_ry}` |
| Roadmap | [dft.md](../../roadmap/dft.md) § bands + metallic occupations|
## Summary

Non-self-consistent **plane-wave band structure** for bulk silicon after a
spin-unpolarized LDA SCF with **Fermi–Dirac metallic occupation smearing**.
The product freezes \(V_\mathrm{eff}\) from `cpu::pw::PW` (fractional
occupations, Mermin free energy \(F\)), then diagonalizes on the diamond
high-symmetry **k-path** via `cpu::pw::Bands`. Twin geometry matches
`bands-dos-si` / `dos-metallic-si` (diamond Si, \(a=5.43\) Å, NC UPF,
ecut 15 Ry, SCF \(2\times2\times2\)). Si stays **gapped** at these settings;
near-integer FD occupations leave \(E_F\) in the gap. The scientific gate is
the **metallic bands path** (no throw) + open integer-fill spectrum + finite
\(E_F\) near \([E_\mathrm{vbm},E_\mathrm{cbm}]\), not a vanishing metal-like
gap. Product regression: `regressions/pw/bands-metallic-si`
(`reg_bands-metallic-si`). Performance: **`perf_exempt`** (no fair CP2K PW
twin wall on the pin host).

## Method

**Level.** Periodic plane-wave Kohn–Sham DFT. Self-consistent density and
effective potential from the same PW stack as `pw-lda-si` / `pw-smearing-al`
(Hartree in \(G\)-space, LDA XC on the FFT grid, Davidson diagonalization,
NC UPF), with **fractional occupations** driven by

\[
f(x)=\frac{1}{1+e^{x}},\qquad
x=\frac{\varepsilon-\mu}{\sigma},\qquad
\sigma=\texttt{degauss}\ \mathrm{[Ry]}.
\]

Closed-shell convention doubles the spinless \(f\in[0,1]\) to band occupation
\(f_b\in[0,2]\). Fermi level \(\mu\) is fixed by electron-count bisection so
\(n_e=Z_\mathrm{val}N_\mathrm{atoms}=8\) for the two-Si cell. Extra empty
bands for Davidson near \(E_F\) come from `extra_bands` (here 4). SCF total is
the **Mermin free energy** \(F=E_\mathrm{KS}-TS\) when smearing is on
(`BandsResult::energy_ry`).

**Bands (k-path, NSCF).** After SCF, \(V_\mathrm{eff}\) is frozen. Fractional
path \(\Gamma\)–\(X\)–\(W\)–\(L\)–\(\Gamma\)–\(K\) via
`Bands::set_diamond_path(n_seg)`. For each path \(\mathbf{k}\),

\[
H[V_\mathrm{eff}]\, \psi_{n\mathbf{k}}
  = \varepsilon_n(\mathbf{k})\, \psi_{n\mathbf{k}},
\qquad
\tfrac12|\mathbf{k}+\mathbf{G}|^2 \le E_\mathrm{cut},
\]

with `n_bands` states (valence + conduction). Path k-points are **not** a BZ
mesh — \(\mu\) reported on `BandsResult::e_fermi_ry` is the **SCF** value, not
re-binned from path weights.

**Gap definitions (gapped twin).** Silicon has \(n_e=8\) →
\(n_\mathrm{occ}=4\) integer valence bands. The product gate uses the
**integer-fill** valence/conduction cut even though occupations were fractional
during SCF:

\[
E_g = E_\mathrm{cbm}-E_\mathrm{vbm}
    = \min_{n\ge n_\mathrm{occ},\mathbf{k}}
      \varepsilon_n(\mathbf{k})
      - \max_{n<n_\mathrm{occ},\mathbf{k}}
      \varepsilon_n(\mathbf{k})
    \quad [\mathrm{Ry}],
\]

and local same-\(\mathbf{k}\) gaps
\(\varepsilon_{n_\mathrm{occ}}(\mathbf{k})-\varepsilon_{n_\mathrm{occ}-1}(\mathbf{k})>0\)
on every path sample (no inverted spectrum). \(E_F\) must sit near the gap
window:

\[
E_\mathrm{vbm}-4\sigma \le E_F \le E_\mathrm{cbm}+4\sigma,
\qquad \sigma=\texttt{degauss}.
\]

Gapped Si at tutorial ecut/k keeps near-integer FD occupations; the twin is a
**metallic-occupation code path** on an insulator, not a claim that Si is a
metal.

**XC / spin.** `XCType::LSDA` (Slater + PW92) on the PW grid; **restricted**
metallic smear only. Collinear spin + metallic smear on bands is rejected
(follow-on; see `bands-spin-si` for fixed-occ spin bands).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_n(\mathbf{k})\) | `BandsResult::eigenvalues_ry` | Ry |
| \(E_F\) | `BandsResult::e_fermi_ry` (SCF μ) | Ry |
| \(E_\mathrm{vbm}\), \(E_\mathrm{cbm}\) | path extrema at integer fill | Ry |
| \(E_g\) | `BandsResult::gap_ry` | Ry |
| \(F\) | `BandsResult::energy_ry` (Mermin free energy) | Ry |

Conversion: `cpu/planewave/qe_units.h` (`rydberg_to_ev = 13.605693`).

**Key exclusions (this slug).**

- Fixed-occupation insulator bands + DOS → `bands-dos-si`.
- Metallic DOS (integral to \(E_F\)) → `dos-metallic-si`.
- Collinear spin bands (fixed occ) → `bands-spin-si`.
- True metal energy pin (FCC Al) → `pw-smearing-al`.
- Spin + metallic smear on bands / DOS (product reject; residual).
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
`pw_solver.h`. Implementation: `src/cpu/planewave/bands_dos.cpp`
(restricted metallic SCF → freeze \(V_\mathrm{eff}\) → NSCF Davidson on
path).

Minimal sequence (matches `reg_bands-metallic-si.cpp`):

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
method.smearing = Smearing::fermi_dirac;
method.degauss = kDegaussRy;
method.extra_bands = kExtraBands;

Bands bands;
bands.n_bands = kNBands;
bands.set_diamond_path(4);              // Γ–X–W–L–Γ–K
const BandsResult br = bands.compute(method, frame);
// br.eigenvalues_ry[ik][ib] [Ry]; br.e_fermi_ry; br.gap_ry;
// br.energy_ry = Mermin F; br.scf_converged
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
  cmake --build build --target reg_bands-metallic-si -j$(nproc)
  ./build/regressions/reg_bands-metallic-si
'
```

Unit (no full SCF e2e): metallic bands path is covered via the
`bands_dos` product surface; full metallic SCF+path is the regression only.

## Validation (vs literature / CP2K note)

Source: `regressions/pw/bands-metallic-si/REFERENCE.md` (offline pin; never
live third party in ctest). Queue / handoff: done 2026-07-26,
`perf_exempt`.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Same family as | `regressions/pw/pw-lda-si`, `bands-dos-si`, `dos-metallic-si` |

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
| Bands path | Γ–X–W–L–Γ–K, `n_seg=4`, `n_bands=8` |

### Accuracy table

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| SCF | `scf_converged == true` | — | — | metallic free-energy SCF |
| `nspin` | 1 | restricted | strict | collinear+smear rejected |
| Path endpoints | includes Γ and X | diamond path | boolean | `set_diamond_path` |
| Eigenvalue order | sorted ascending per k | — | — | 8 bands |
| Local same-\(k\) gap | \(\varepsilon_4-\varepsilon_3 > 0\) | open insulator | strict | every path k |
| Integer-fill \(E_g\) | `gap_ry > 0` | open Si | strict | not a metal-collapse gate |
| \(E_F\) finite | \(\lvert E_F\rvert < 10^3\) Ry | SCF μ | strict | `e_fermi_ry` |
| \(E_F\) placement | in \([E_\mathrm{vbm},E_\mathrm{cbm}]\pm 4\sigma\) | gapped FD | \(\sigma=0.02\) Ry | near-integer fill |
| Measured twin (informative, 2026-07-26) | \(E_F\approx 0.465\) Ry, \(E_g\approx 0.043\) Ry, \(F\approx -15.686\) Ry | SCF 6 iters | host note | REFERENCE Release OMP=8 |

**Incomplete-basis rationale.** Twin ecut and SCF k are deliberately light
(same class as `pw-lda-si` / `bands-dos-si`). The gate is the **metallic
bands code path** plus open spectrum / \(E_F\) placement on gapped Si — not a
freeze of Atomiverse-only free-energy digits as a literature metal property,
and not absolute eigenvalue parity with CP2K GTH+Gaussian.

**CP2K.** Quickstep `BAND_STRUCTURE` + `&SMEAR METHOD FERMI_DIRAC` use
**GTH-PADE + DZVP-GTH**, not ONCV plane waves. Absolute \(\varepsilon_n\)
are **not** comparable. Product counterpart (character: metallic occupations
enabled; open Si gap; \(E_F\) in gap) is documented offline:

`tools/cp2k_oracle/inputs/bands-metallic-si.note`

Suggested twin: PADE, GTH-PADE-q4 + DZVP-GTH-q4, MP \(2\times2\times2\),
\(k_B T \approx 0.02\) Ry (`ELECTRONIC_TEMPERATURE [K] 3157.75`),
`ADDED_MOS ≥ 4`, path Γ–X–W–L–Γ–K.

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K 2026.1+ Quickstep PADE + SMEAR FERMI_DIRAC + BAND_STRUCTURE |
| CP2K binary (pin host) | **not available** (`module avail cp2k` empty at pin time) |
| Scientific pin | metallic bands path; open Si gap; \(E_F\) in gap window |

## Performance

Source: `regressions/pw/bands-metallic-si/REFERENCE.md`. Queue / handoff row:
**`perf_exempt`** (no fair GTH twin wall on pin host).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si metallic bands ecut=15 / scf k=2 + path n_seg=4 + FD | n/a | ≈ 14.3 (informative) | n/a | n/a (**exempt**) | No CP2K module on pin host; absolute \(\varepsilon_n\) not a CP2K twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~14.3 s (Release, 8 OMP, ecut 15 / k=2 + path + FD) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a CP2K binary is available offline: time the same geometry with
`tools/cp2k_oracle/inputs/bands-metallic-si.note`, append
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to REFERENCE.md, require \(r\le 10\).
Wall comparison is same-geometry only, not absolute-eigenvalue parity
(GTH+Gaussian vs NC-UPF PW).

## Limitations

- Tutorial ecut / SCF k only — not production-converged metallic band
  structure or dense path productization.
- Twin is **gapped Si** under FD smear (near-integer occupations); it does
  **not** pin a true metal vanishing gap (use `pw-smearing-al` for FCC Al
  free energy).
- Path \(\mu\) is the SCF value; do not treat path-k weights as a BZ mesh
  for re-solving \(E_F\).
- No collinear spin + metallic smear on bands (product reject).
- No DOS on this slug — see `dos-metallic-si`.
- Absolute \(\varepsilon_n\) not comparable to CP2K Quickstep GTH decks.
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); this
  regression always uses the in-tree ONCV Si UPF.
- CUDA PW surface is not a CPU compatibility API for this pin.
- No MPI multi-rank scaling gate (`mpi-pw-scaling-4rank` later).

## References

- Offline pin: [`regressions/pw/bands-metallic-si/REFERENCE.md`](../../../regressions/pw/bands-metallic-si/REFERENCE.md)
- Regression: [`regressions/pw/bands-metallic-si/reg_bands-metallic-si.cpp`](../../../regressions/pw/bands-metallic-si/reg_bands-metallic-si.cpp)
- Public API: [`include/atomiverse/cpu/planewave/bands_dos.h`](../../../include/atomiverse/cpu/planewave/bands_dos.h)
- Optional CP2K wall / character skeleton: `tools/cp2k_oracle/inputs/bands-metallic-si.note`
- Sibling insulator bands: [bands-dos-si](bands-dos-si.md)
- Sibling metallic SCF energy: [pw-smearing-al](pw-smearing-al.md)
- Related energy foothold: [pw-lda-si](pw-lda-si.md)
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
