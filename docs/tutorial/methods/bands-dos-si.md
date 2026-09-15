# Plane-wave band structure + DOS — bulk silicon

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `bands-dos-si` |
| Regression | `regressions/pw/bands-dos-si/` |
| Public API | `cpu::pw::PW` (SCF members) + `cpu::pw::Bands` / `cpu::pw::Dos` + `BandsResult` / `DosResult` |
| Roadmap | [dft.md](../../roadmap/dft.md) § bands/DOS|

## Summary

Non-self-consistent **plane-wave band structure** and **Gaussian-broadened
density of states** for bulk silicon (diamond, 2-atom fcc primitive cell)
after a spin-unpolarized LDA SCF on a norm-conserving ONCVPSP UPF. The
product path freezes \(V_\mathrm{eff}\) from `cpu::pw::PW` members, then
diagonalizes on a high-symmetry **k-path** (`cpu::pw::Bands`) and a denser
Monkhorst–Pack mesh (`cpu::pw::Dos`). Observables are \(\varepsilon_n(\mathbf{k})\)
[Ry], the textbook **Γ→X** gap [eV], and the occupied DOS integral
\(n_\mathrm{elec\_to\_vbm}\). Scientific gate is a literature incomplete-basis
window around the Yin–Cohen LDA indirect gap (~0.45 eV), not absolute
eigenvalue parity with CP2K Quickstep (GTH+Gaussian). Product regression:
`regressions/pw/bands-dos-si` (`reg_bands-dos-si`). Performance:
**`perf_exempt`** (no fair CP2K PW twin wall on the pin host).

## Method

**Level.** Periodic plane-wave Kohn–Sham DFT. Self-consistent density and
effective potential from the same PW stack as `pw-lda-si` (Hartree in
\(G\)-space, LDA XC on the FFT grid, Davidson diagonalization, NC UPF). After
SCF, **bands** and **DOS** are **non-self-consistent**: the mixed density’s
final \(V_\mathrm{eff}\) is frozen; only the k-dependent kinetic/projectors
change while eigenvalues are recomputed.

**Bands (k-path).** Fractional path \(\Gamma\)–\(X\)–\(W\)–\(L\)–\(\Gamma\)–\(K\)
via `Bands::set_diamond_path(n_seg)`. For each path \(\mathbf{k}\),

\[
H[V_\mathrm{eff}]\, \psi_{n\mathbf{k}}
  = \varepsilon_n(\mathbf{k})\, \psi_{n\mathbf{k}},
\qquad
\tfrac12|\mathbf{k}+\mathbf{G}|^2 \le E_\mathrm{cut},
\]

with `n_bands` states (valence + some conduction). Silicon has
\(n_e=8\) → \(n_\mathrm{occ}=4\) occupied bands in the closed-shell convention.

**Gap definitions.** The regression pins the **textbook Si transition**, not
necessarily the global max/min over a sparse path:

\[
E_{\Gamma\to X}
  = \bigl(\varepsilon_{c}(X) - \varepsilon_{v}(\Gamma)\bigr)
    \times 13.605693~\mathrm{eV},
\qquad
\varepsilon_{v}(\Gamma)=\varepsilon_{n_\mathrm{occ}-1}(\Gamma),
\quad
\varepsilon_{c}(\mathbf{k})=\varepsilon_{n_\mathrm{occ}}(\mathbf{k}).
\]

Local (same-\(\mathbf{k}\)) gaps
\(\varepsilon_{n_\mathrm{occ}}(\mathbf{k})-\varepsilon_{n_\mathrm{occ}-1}(\mathbf{k})\)
must stay positive (no inverted spectrum at a sampled point). Indirect
character requires \(\varepsilon_c(X)<\varepsilon_c(\Gamma)\); the direct
\(\Gamma\) gap is larger than \(E_{\Gamma\to X}\) within a small tolerance.

**DOS.** Uniform MP mesh of size `Dos::k_mesh`³; Gaussian broadening

\[
D(E_i)
  = \sum_{n\mathbf{k}} w_{\mathbf{k}}\,
    \frac{2}{\sqrt{2\pi}\,\sigma}\,
    \exp\!\Bigl(-\frac{(E_i-\varepsilon_{n\mathbf{k}})^2}{2\sigma^2}\Bigr)
\]

with spinless closed-shell weight 2 absorbed into the kernel
(`gaussian_dos` in `bands_dos.h`). Units: \(D(E)\) in states / Ry / cell;
occupied integral \(n_\mathrm{elec\_to\_vbm}=\int_{-\infty}^{E_\mathrm{vbm}} D(E)\,dE\)
targets \(n_e=8\).

**XC / spin.** `XCType::LSDA` (Slater + PW92) on the PW grid; restricted fixed
occupations (`Smearing::none`). No metallic smearing, no collinear spin on
this twin.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_n(\mathbf{k})\) | `BandsResult::eigenvalues_ry` | Ry |
| \(E_{\Gamma\to X}\) | \((\varepsilon_c(X)-\varepsilon_v(\Gamma))\times 13.605693\) | eV |
| \(E_g^\mathrm{direct}(\Gamma)\) | same-band indices at \(\Gamma\) | eV |
| \(D(E)\) | `DosResult::dos` | 1/Ry/cell |
| \(n_\mathrm{elec\_to\_vbm}\) | occupied DOS integral | electrons/cell |
| \(E_\mathrm{SCF}\) | `BandsResult::energy_ry` | Ry |

Conversion: `cpu/planewave/qe_units.h` (`rydberg_to_ev = 13.605693`).

**Key exclusions (this slug).**

- GPW / Gaussian k-path → `gpw-kpoints-si`.
- Spin-polarized bands / spin-resolved DOS.
- Berry phase, Wannier centers, DFPT / FD phonons → later queue rows
  (`berry-polarization`, `wannier-centers`, `dfpt-phonon-si`, …).
- Absolute \(\varepsilon_n\) parity with CP2K Quickstep GTH+DZVP (different
  PP and basis).
- Production-converged \(E_\mathrm{cut}\) (~60 Ry) / dense k when they break
  the ≤ 60 s e2e budget.
- Live CP2K / QE / ABINIT at ctest time (offline pins only).
- Public `*Config` bags; `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + concrete method / post-processor
members; no `*Config`, no factories). Headers:
`include/atomiverse/cpu/planewave/bands_dos.h`,
`pw_solver.h`, `qe_units.h`. Implementation:
`src/cpu/planewave/bands_dos.cpp` (restricted SCF → freeze \(V_\mathrm{eff}\)
→ NSCF Davidson on path / DOS mesh).

Minimal sequence (matches `reg_bands-dos-si.cpp`):

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
constexpr int kNOcc = 4;          // Si: n_e = 8

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
method.max_iter = 50;
method.xc = cpu::scf::XCType::LSDA;

Bands bands;
bands.n_bands = kNBands;
bands.set_diamond_path(4);              // Γ–X–W–L–Γ–K
const BandsResult br = bands.compute(method, frame);
// br.eigenvalues_ry[ik][ib] [Ry]; br.kfrac; br.scf_converged

Dos dos;
dos.k_mesh = 4;
dos.n_bands = kNBands;
dos.sigma_ry = 0.05;
dos.n_bins = 200;
const DosResult dr = dos.compute(method, frame);
// dr.dos, dr.n_elec_to_vbm, dr.energy_ry bins
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
  cmake --build build --target reg_bands-dos-si -j$(nproc)
  ./build/regressions/reg_bands-dos-si
'
```

Unit (no full SCF e2e): `test_cpu_pw_bands_dos` — path endpoints, Gaussian
integral identity, synthetic gap finder.

## Validation (vs literature / CP2K note)

Source: `regressions/pw/bands-dos-si/REFERENCE.md` (offline pin; never live
third party in ctest). Closed acceptance: git history
`.claude/specs/bands-dos-si/` (spec `034fed2e`, feature merged on `dev`).

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Same family as | `regressions/pw/pw-lda-si` |

### Literature pin (indirect gap)

Bulk Si is an **indirect** gap semiconductor: VBM at \(\Gamma\), CBM near
\(X\).

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
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 50 |
| Bands path | Γ–X–W–L–Γ–K, `n_seg=4`, `n_bands=8` |
| DOS | MP `k_mesh=4`, `n_bands=8`, \(\sigma=0.05\) Ry, 200 bins |

### Accuracy table

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| LDA \(E_g\) (converged lit.) | — | **~0.45 eV** | literature | Yin–Cohen PRB 26, 5668 (1982) |
| Exp. \(E_g\) | — | ~1.12 eV | context only | LDA underestimates |
| \(E_{\Gamma\to X}\) (twin) | band gate | Yin–Cohen window | **[0.05, 2.50] eV** | incomplete ecut/k; rejects metallic collapse / multi-eV disasters |
| Measured twin (informative) | **≈ 0.48 eV** | ~0.45 eV | inside window | REFERENCE host note 2026-07-24, Release |
| Local same-\(k\) gap | \(> 0\) | open insulator | strict | every path k |
| Indirect character | \(\varepsilon_c(X)<\varepsilon_c(\Gamma)\) | Si textbook | boolean | path samples include Γ and X |
| Direct \(\Gamma\) vs Γ→X | \(E_g^\mathrm{dir}(\Gamma) \ge E_{\Gamma\to X}-0.05\) eV | Si | −0.05 eV slack | direct larger than indirect |
| \(n_\mathrm{elec\_to\_vbm}\) | \(\approx 8\) | \(n_e=8\) | **±1.0** | Gaussian DOS on tutorial mesh |
| SCF | `scf_converged == true` | — | — | both `Bands` and `Dos` |
| Eigenvalue order | sorted ascending per k | — | — | 8 bands |

**Incomplete-basis rationale.** Twin ecut and SCF k are deliberately light
(same class as `pw-lda-si`). The gate is a **literature-anchored window** on
the Γ→X transition, not a freeze of Atomiverse-only digits. On the pin host
the measured Γ→X gap sits near the Yin–Cohen LDA value despite the tutorial
settings; the wide band still allows residual path scatter without accepting
a metallic or multi-eV disaster.

**CP2K.** Quickstep `BAND_STRUCTURE` / DOS use **GTH-PADE + DZVP-GTH**, not
ONCV plane waves. Absolute \(\varepsilon_n\) are **not** comparable.
Character (indirect Si, \(\varepsilon_c(X)<\varepsilon_c(\Gamma)\)) and the
Yin–Cohen LDA Γ→X window are the scientific pins. Optional offline wall /
character input: `tools/cp2k_oracle/inputs/si_bands_dos_note.inp` (same
geometry, \(2\times 2\times 2\) MP, path Γ–X–W–L–Γ–K with `NPOINTS 4`).

## Performance

Source: `regressions/pw/bands-dos-si/REFERENCE.md`. Queue / handoff row:
**`perf_exempt`** (literature Γ→X window; no fair GTH twin wall on pin host).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si bands+DOS ecut=15 / scf k=2 + path + DOS k=4 | n/a | ≈ 22 (informative) | n/a | n/a (**exempt**) | No CP2K module on pin host; absolute \(\varepsilon_n\) not a CP2K twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~22 s (Release, 8 OMP, ecut 15 / k=2 + path + DOS k=4) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a CP2K binary is available offline: time the same geometry with
`tools/cp2k_oracle/inputs/si_bands_dos_note.inp`, append
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to REFERENCE.md, require \(r\le 10\).
Wall comparison is same-geometry only, not absolute-eigenvalue parity
(GTH+Gaussian vs NC-UPF PW).

## Limitations

- Tutorial ecut / SCF k only — not production-converged band gaps or dense
  DOS productization.
- Global \(\min\varepsilon_c-\max\varepsilon_v\) over a **sparse** path can
  spuriously go slightly negative at incomplete \(E_\mathrm{cut}\) from
  path-point scatter; the product gate therefore uses **Γ→X** and local
  same-\(k\) positivity (see Method / Validation).
- No spin-polarized bands, metallic smearing, or GPW k-path on this slug.
- No Berry / Wannier / DFPT response here — those queue rows depend on this
  foothold (`bands-dos-si` ✓ in handoff).
- Absolute \(\varepsilon_n\) not comparable to CP2K Quickstep GTH decks.
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); this
  regression always uses the in-tree ONCV Si UPF.
- CUDA PW surface is not a CPU compatibility API for this pin.
- No MPI multi-rank scaling gate (`mpi-pw-scaling-4rank` later).

## References

- Offline pin: [`regressions/pw/bands-dos-si/REFERENCE.md`](../../../regressions/pw/bands-dos-si/REFERENCE.md)
- Regression: [`regressions/pw/bands-dos-si/reg_bands-dos-si.cpp`](../../../regressions/pw/bands-dos-si/reg_bands-dos-si.cpp)
- Public API: [`include/atomiverse/cpu/planewave/bands_dos.h`](../../../include/atomiverse/cpu/planewave/bands_dos.h)
- Optional CP2K wall / character skeleton: `tools/cp2k_oracle/inputs/si_bands_dos_note.inp`
- Related energy foothold: [pw-lda-si](pw-lda-si.md)
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)
- M. T. Yin and M. L. Cohen, *Phys. Rev. B* **26**, 5668 (1982)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
- Closed acceptance (historical): `.claude/specs/bands-dos-si/` in git
  (`034fed2e`)
