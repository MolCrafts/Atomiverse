# Phonon dispersion ω(q) — diamond Si (SW IFC Fourier)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `phonon-dispersion-si` |
| Regression | `regressions/response/phonon-dispersion-si/` |
| Public API | `cpu::vib::PhononDispersion` (members) + `cpu::vib::PhononDispersionResult` + IFCs from `cpu::vib::Phonon` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 — phonon dispersions ω(q) ✓ (finite-q DFPT Sternheimer ○) |
| Dep | `fd-phonon-supercell` ✓ (supercell FD Φ); `dfpt-phonon-si` ✓ is orthogonal (Γ DFPT, not this Fourier path) |

## Summary

**Phonon band structure** \(\omega(\mathbf{q})\) for a crystal is obtained by
Fourier-transforming supercell interatomic force constants (IFCs) onto the
**unit-cell** dynamical matrix at arbitrary fractional wavevector \(\mathbf{q}\),
then mass-weighting and diagonalizing. Product surface:
`cpu::vib::PhononDispersion` — phonopy-style IFC Fourier, not a second force
loop and not finite-\(q\) DFPT. IFCs come from the existing crystal FD preset
`cpu::vib::Phonon` (or any equivalent Cartesian Hessian in the same atom
order). Product gate: `regressions/response/phonon-dispersion-si`
(`reg_phonon-dispersion-si`) — diamond Si **Stillinger–Weber**, conventional
cubic 8-atom unit cell, 2×2×2 supercell Φ, high-symmetry path **Γ → X → L**.
Performance: **`perf_exempt`** (no offline CP2K `VIBRATIONAL_ANALYSIS` wall
on the pin host).

## Method

**Level.** Given Cartesian force constants \(\Phi\) of an
\(n_a\times n_b\times n_c\) supercell (atom order =
`cpu::vib::expand_supercell`), build the mass-weighted dynamical matrix of the
**unit cell** at fractional \(\mathbf{q}\):

\[
D_{\alpha\beta}(\kappa\kappa';\mathbf{q})
  = \frac{1}{\sqrt{m_\kappa m_{\kappa'}}}
    \sum_{\ell'}
    \Phi_{\alpha\beta}(0\kappa,\,\ell'\kappa')
    \,e^{i\mathbf{q}\cdot\mathbf{R}_{\ell'}}
  \qquad[\mathrm{a.u.}].
\]

\(\Phi\) is converted eV/Å² → Hartree/Bohr² with the same NIST factors as
`MassWeighting` / `PhononDiagonalizer`. Eigenvalues of \(D(\mathbf{q})\) become
wavenumbers [cm⁻¹] via `nist::au2wavenumber()` (CODATA-2014 path shared with
molecular vib and supercell-Γ phonons). Imaginary modes (\(\lambda < 0\)) are
reported as **negative** \(\omega\). Branches at each \(\mathbf{q}\) are sorted
**ascending**.

**IFC assembly (upstream).** Central differences of forces on the supercell
with \(h = 10^{-3}\) Å (`kHessianStepAngstrom`), same product as
`fd-phonon-supercell`:

\[
\Phi_{i\alpha,j\beta}
  = -\frac{F_{j\beta}(+\mathbf{e}_{i\alpha}h) - F_{j\beta}(-\mathbf{e}_{i\alpha}h)}{2h}
  \quad[\mathrm{eV}/\mathrm{\AA}^2],
\]

then \(\Phi \leftarrow (\Phi+\Phi^\top)/2\). Force engines are user-bound on
`vib::Phase::force` (classical `Potential`, PW, GPW, …). This slug pins SW.

**q-path.** Fractional crystal coordinates of the **unit cell**. The regression
uses a short conventional-cubic path:

| Label | Fractional \(\mathbf{q}\) |
| ----- | ------------------------: |
| Γ | \((0,0,0)\) |
| mid Γ–X | \((1/4,0,0)\) |
| X | \((1/2,0,0)\) |
| mid X–L | \((1/2,1/4,1/4)\) |
| L | \((1/2,1/2,1/2)\) |

API also provides `PhononDispersion::set_diamond_path(n_seg)` (FCC/diamond
Γ–X–W–L–Γ–K, same waypoints as `cpu::pw::Bands::set_diamond_path`). Empty
`qfrac` at `compute` time fills `set_diamond_path(4)`.

**Why conventional cubic unit cell (not 2-atom primitive).** The classical
neighbor cell list uses `Box::length(d) = cell(d,d)`, which is **0** for the
diamond primitive triclinic cell and can hang orthorhombic min-image helpers.
The Fourier API itself is cell-agnostic; the conventional 8-atom cubic cell
is the proven SW geometry from `fd-phonon-supercell`. Triclinic neighbor fix
is out of scope for this slug.

**Relation to other phonon products.**

| Product | What it computes | Surface |
| ------- | ---------------- | ------- |
| `cpu::vib::Hessian` | free-molecule FD Hessian + TR project + thermochem | molecular |
| `cpu::vib::Phonon` | supercell FD Φ → \(\omega\) at **supercell Γ** | crystal Γ (folded) |
| `cpu::vib::PhononDispersion` | Fourier of supercell Φ → \(\omega(\mathbf{q})\) on unit cell | crystal **q-path** |
| `cpu::pw::DfptPhonon` | analytic DFPT dynamical matrix at **q = 0** (PW LSDA) | Γ DFPT foothold |

At \(\mathbf{q}=0\), the Fourier spectrum of the conventional cell recovers the
same acoustic / optical physics as the supercell-Γ FD spectrum of the 64-atom
cell (zone-folded branches sit among the \(3N_\mathrm{unit}=24\) modes).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\omega_b(\mathbf{q})\) | `PhononDispersionResult::frequencies_cm[iq][ibranch]` | cm⁻¹ |
| \(\mathbf{q}_\mathrm{frac}\) | `PhononDispersionResult::qfrac` | crystal |
| \(N\) | `n_atoms_unit` (unit cell); branches \(= 3N\) | — |
| \(n_a,n_b,n_c\) | supercell factors that built Φ | — |
| \(\Phi\) | upstream `PhononResult::hessian` | eV/Å² |

Frequencies live on **`PhononDispersionResult`**, never on Frame (Frame
purity). Upstream forces on Frame remain eV/Å during the FD loop.

**Key exclusions (binding).**

- Finite-\(\mathbf{q}\) **DFPT** Sternheimer (analytic response at \(q\neq 0\))
  — follow-on; IFC Fourier is the productized dispersion surface.
- Acoustic sum-rule **projection** as a required product step (full \(3N\)
  branches; near-zero acoustics at Γ are an acceptance check, not imposed).
- LO–TO splitting / Born effective charges (nonpolar Si does not need them;
  polar materials are future work).
- Production-converged PW/GPW FD IFCs vs CP2K Quickstep frequencies (composition
  path is public; pin is classical SW for ≤ 60 s e2e).
- Molecular thermochemistry / free-molecule TR projection (`cpu::vib::Hessian`).
- Public `*Config` bags, `make_*` / `build_*` factories, CUDA phonon path.
- Live CP2K at ctest time (offline pins only).

## Implementation

Public surface only (`cpu::Frame` + members / ctor args; no `PhononDispersionConfig`,
no factories). Headers:
`include/atomiverse/cpu/vibration/phonon_dispersion.h`,
`phonon.h`, `phonon_result.h`, `supercell.h`. Implementation:
`src/cpu/vibration/phonon_dispersion.cpp`.

Minimal sequence (matches `reg_phonon-dispersion-si.cpp`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/sw.h"
#include "atomiverse/cpu/vibration/phonon.h"
#include "atomiverse/cpu/vibration/phonon_dispersion.h"
#include "atomiverse/cpu/vibration/supercell.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// conventional cubic diamond Si, a = 5.431 Å, 8 atoms (unit)
cpu::Frame unit = /* diamond_conventional(5.431, 1); see regression */;
const int na = 2, nb = 2, nc = 2; // IFC supercell → 64 atoms

cpu::SWPotential pot(molrs::Element::Si, "stillinger-weber");
const double cutoff = pot.cutoff(); // a*σ ≈ 3.7712 Å

cpu::Frame sc = cpu::vib::expand_supercell(unit, na, nb, nc);

cpu::vib::Phonon phonon(/*na=*/1, /*nb=*/1, /*nc=*/1); // Γ of already-expanded SC
phonon.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, false});
phonon.on(ATV::vib::Phase::force).add(cpu::StillingerWeber{pot});
phonon.build();
const cpu::vib::PhononResult pr = phonon.run(sc);
// pr.hessian [eV/Å²], shape [3*N_sc, 3*N_sc]

cpu::vib::PhononDispersion path;
path.qfrac = {
    {0.0, 0.0, 0.0},   // Γ
    {0.25, 0.0, 0.0},
    {0.5, 0.0, 0.0},   // X
    {0.5, 0.25, 0.25},
    {0.5, 0.5, 0.5},   // L
};
// or: path.set_diamond_path(/*n_seg=*/4);

const cpu::vib::PhononDispersionResult r =
    path.compute(pr.hessian, unit, na, nb, nc);
// r.frequencies_cm[iq][ibranch] [cm^-1]; r.n_branches == 24
```

Build and run the regression (`ATV_BUILD_REGRESSIONS=ON`; SW twin needs MD
potentials + vib):

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
  cmake --build build --target reg_phonon-dispersion-si -j$(nproc)
  ./build/regressions/reg_phonon-dispersion-si
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/vibration/phonon_dispersion.h` | `PhononDispersion` / `PhononDispersionResult` |
| `src/cpu/vibration/phonon_dispersion.cpp` | Fourier \(D(\mathbf{q})\) + eigensolve |
| `include/atomiverse/cpu/vibration/phonon.h` | upstream FD IFCs |
| `include/atomiverse/cpu/vibration/supercell.h` | `expand_supercell` |
| `regressions/response/phonon-dispersion-si/` | e2e + `REFERENCE.md` |

## Validation (vs CP2K / literature)

Source:
[`regressions/response/phonon-dispersion-si/REFERENCE.md`](../../../regressions/response/phonon-dispersion-si/REFERENCE.md).
Oracle priority: offline **CP2K `VIBRATIONAL_ANALYSIS`** (+ band structure) when
a binary is available; **this pin host had no CP2K module** — surrogate is the
acoustic sum rule plus the published Stillinger–Weber optical / zone-boundary
band windows already used by `fd-phonon-supercell`. **Never** live CP2K in
ctest.

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, **conventional cubic 8-atom** cell |
| Lattice constant \(a\) | **5.431 Å** (SW fit target) |
| Force engine | Stillinger–Weber Si (`cpu::StillingerWeber`) |
| SW parameters | Stillinger & Weber, *Phys. Rev. B* **31**, 5262 (1985) |
| IFC supercell | **2×2×2** conventional → **64 atoms**, \(L = 10.862\) Å |
| Why 2×2×2 | SW cutoff \(a\sigma = 3.7712\) Å requires \(L > 2\times\) cutoff |
| FD step \(h\) | \(10^{-3}\) Å (`kHessianStepAngstrom`) |
| Neighbor | always-rebuild (`check=false`), cutoff \(a\sigma\) |
| Masses | IUPAC-2013 via `cpu::vib::mass_amu(Z)` (regression mass 28.0855 amu) |
| Branches | \(3N = 24\) (conventional unit cell) |
| q-path | Γ → mid → X → mid → L (5 points; fractional of cubic) |

### Why SW not PW/GPW for the first pin

A full ab initio FD Hessian is \(6N_\mathrm{sc}\) SCF+force evaluations. The
**machinery** under test is IFC Fourier + public `PhononDispersion` API +
Γ/X/L branch sanity. SW supplies periodic many-body forces with a literature
potential and finishes in well under the ≤ 60 s e2e budget (same twin family
as `reg_fd-phonon-supercell`). Binding `PWEvaluator::with_gradient(true)` (or
GPW) on `vib::Phase::force` is the same public composition path for a later
CP2K frequency twin.

### Accuracy table

Hard-coded offline pins (REFERENCE, 2026-07-25, Release, `OMP_NUM_THREADS=4`):

| Quantity | Atomiverse / pin | External / rule | Tol / band | Notes |
| -------- | ---------------: | --------------- | ---------- | ----- |
| Branches / path length | **24 / 5** | \(3N\), \(N=8\); 5 q | exact | hard-coded |
| Γ \(\lvert\omega\rvert_\mathrm{ac,max}\) (3 lowest) | **0.00** cm⁻¹ | Γ acoustic sum rule | **&lt; 30** cm⁻¹ | no ASR projection |
| Γ \(\omega_\mathrm{max}\) (optical) | **594.82** cm⁻¹ | SW optical scale | **[500, 900]** cm⁻¹ | same window as FD Γ twin |
| X \(\omega_\mathrm{max}\) | **562.56** cm⁻¹ | SW zone-boundary band | **[300, 900]** cm⁻¹ | finite optical/acoustic |
| L \(\omega_\mathrm{max}\) | **559.27** cm⁻¹ | SW zone-boundary band | **[300, 900]** cm⁻¹ | finite optical/acoustic |
| Sorted / finite | yes | every path q | exact | NaN / disorder fail |
| Frame pollution | none | result-only | — | \(\omega\) on `PhononDispersionResult` |

Experimental Si LO(Γ) ≈ 520 cm⁻¹ is **not** the pin — SW is classical and
systematically stiffer; compare the SW optical window used by
`fd-phonon-supercell`, not Raman experiment.

**CP2K.** When `cp2k.psmp` is available offline: run a fixed
`VIBRATIONAL_ANALYSIS` / phonon band input on diamond Si (same \(a\),
comparable force model or Quickstep), record \(\omega(\Gamma/X/L)\) [cm⁻¹]
and wall time in REFERENCE.md with CP2K version. Skeleton notes may live under
`tools/cp2k_oracle/inputs/`.

## Performance

Source: `regressions/response/phonon-dispersion-si/REFERENCE.md`. Queue /
handoff: **`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| SW 64-atom FD IFC + Fourier Γ/X/L | n/a | ≈ **0.07** (informative) | n/a | n/a (**exempt**) | No CP2K binary on pin host |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K module/binary on the build host for a fair wall twin this shift |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS; twin ≪ 1 s) |
| Cost class | same FD cost as `fd-phonon-supercell` (64-atom SW, \(6N\) displacements) + cheap Fourier path |

When a CP2K binary is available offline: time a comparable vibrational /
phonon-band twin, append \(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to
REFERENCE.md, require \(r \le 10\).

## Limitations

- First product pin is **classical SW**, not PW/GPW FD frequencies vs CP2K
  Quickstep (composition path for ab initio forces is public; pin is future
  work).
- **IFC Fourier**, not finite-\(\mathbf{q}\) DFPT; analytic \(q\neq 0\)
  Sternheimer remains open on the roadmap.
- Conventional cubic unit cell for the SW neighbor path; triclinic primitive
  diamond hangs classical min-image (infra debt, out of slug scope).
- No ASR post-projection as a product feature; acoustic near-zero at Γ is a
  check.
- No LO–TO / Born charges; no polar materials on this twin.
- No CUDA phonon dispersion; no MPI multi-rank FD scaling gate here.
- SW optical \(\omega\) is systematically stiffer than experimental Si LO(Γ)
  (~520 cm⁻¹); compare SW literature, not Raman experiment, for this twin.

## References

- Offline pin: [`regressions/response/phonon-dispersion-si/REFERENCE.md`](../../../regressions/response/phonon-dispersion-si/REFERENCE.md)
- Regression: [`regressions/response/phonon-dispersion-si/reg_phonon-dispersion-si.cpp`](../../../regressions/response/phonon-dispersion-si/reg_phonon-dispersion-si.cpp)
- Upstream Γ FD twin: [fd-phonon-supercell](fd-phonon-supercell.md),
  [dfpt-phonon-si](dfpt-phonon-si.md) (orthogonal PW Γ DFPT)
- Roadmap: [dft.md](../../roadmap/dft.md) §11 phonons / DFPT
- Oracle notes: `tools/cp2k_oracle/README.md`
- Narrative style refs: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md),
  [Chapter 14 — Vibrations](../ch14-vibrations.md)
- F. H. Stillinger and T. A. Weber, *Phys. Rev. B* **31**, 5262 (1985)
- S. Baroni, S. de Gironcoli, A. Dal Corso, P. Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001) (DFPT context; finite-\(q\) DFPT is the follow-on)
- Phonopy / Togo–Tanaka — supercell IFC Fourier construction
