# Finite-displacement supercell phonons — diamond Si (Γ, SW twin)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `fd-phonon-supercell` |
| Regression | `regressions/pw/fd-phonon-supercell/` |
| Public API | `cpu::vib::Phonon` (ctor `na,nb,nc` / members) + `cpu::vib::PhononResult` + `cpu::vib::expand_supercell` |
| Roadmap | [dft.md](../../roadmap/dft.md) § Finite-displacement phonons ✓; enables `dfpt-phonon-si` |

## Summary

**Finite-displacement (FD) crystal phonons** assemble Cartesian force constants
on a periodic supercell by central differences of forces, mass-weight the
dynamical matrix at the **supercell Γ point**, and return the full \(3N\)
spectrum in **cm⁻¹**. Product surface is the preset `cpu::vib::Phonon` over
`Driver<vib::Phase>` — not molecular `cpu::vib::Hessian` (which projects free-
molecule translations/rotations and has no supercell). Force engines are
**user-bound** on `vib::Phase::force` (classical `Potential`, PW, GPW, …), same
composition rule as AIMD / molecular vib. Product gate:
`regressions/pw/fd-phonon-supercell` (`reg_fd-phonon-supercell`) — diamond Si
**Stillinger–Weber** 64-atom twin (acoustic sum rule + optical band).
Performance: **`perf_exempt`** (no offline CP2K `VIBRATIONAL_ANALYSIS` wall on
the pin host).

## Method

**Level.** Γ-point phonons of an \(n_a\times n_b\times n_c\) supercell of a
periodic unit cell. Lattice vectors are the rows of `frame.box.cell()` [Å]:

\[
\mathbf{a}' = n_a\mathbf{a},\quad
\mathbf{b}' = n_b\mathbf{b},\quad
\mathbf{c}' = n_c\mathbf{c}.
\]

Image \((i,j,k)\) places every unit-cell atom at
\(\mathbf{R} + i\mathbf{a} + j\mathbf{b} + k\mathbf{c}\)
(\(i\in[0,n_a)\), etc.). Types and masses replicate; tags renumber uniquely.
`expand_supercell` does **not** copy topology (bonds/angles) — force engines
that need connectivity rebuild it.

**Force constants.** Central difference with \(h = 10^{-3}\) Å
(`kHessianStepAngstrom`), same step as molecular Hessian:

\[
\Phi_{i\alpha,j\beta}
  = -\frac{F_{j\beta}(+\mathbf{e}_{i\alpha}h) - F_{j\beta}(-\mathbf{e}_{i\alpha}h)}{2h}
  \quad[\mathrm{eV}/\mathrm{\AA}^2].
\]

Symmetrize \(\Phi \leftarrow (\Phi+\Phi^\top)/2\).

**Dynamical matrix at Γ.** Mass-weight via the shared `MassWeighting` unit path
(Hartree/Bohr²/amu), then

\[
D_{i\alpha,j\beta}
  = \frac{\Phi_{i\alpha,j\beta}}{\sqrt{m_i m_j}}.
\]

Diagonalize \(D\); convert eigenvalues to wavenumbers with
`nist::au2wavenumber()` (CODATA-2014 path shared with molecular vib). Report
**all \(3N\)** modes ascending; imaginary eigenvalues as **negative** \(\omega\).
At a stress-free crystal minimum the three acoustic branches at \(\mathbf{q}=0\)
vanish; residual \(|\omega|\) is FD / force noise.

**Pipeline (structural components, fixed budget).** Same phase enum as
`Hessian` (`setup` → `displace` → `force` → `accumulate` → … → `diagonalize`);
termination is \(6N_\mathrm{sc}\) displacements + one finalize pass. Crystal
path uses `PhononDiagonalizer` (full \(3N\)) instead of the molecular TR
projector + thermochemistry.

| | Molecular `cpu::vib::Hessian` | Crystal `cpu::vib::Phonon` |
| - | --- | --- |
| System | free molecule | crystal (PBC forces) |
| Cell | ignored for modes | supercell expansion |
| Diagonalizer | TR project \(3N-6/5\) | full \(3N\) spectrum |
| Output | vib + thermochem | \(\omega\) [cm⁻¹] (+ Hessian / modes on result) |

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\omega_k\) | `PhononResult::frequencies` (length \(3N\), ascending) | cm⁻¹ |
| \(\Phi\) | `PhononResult::hessian` | eV/Å² |
| \(N\) | `n_atoms` (supercell); `n_unit` before expansion | — |
| \(n_a,n_b,n_c\) | supercell factors on the result | — |

Frequencies and force constants live on **`PhononResult`**, not on Frame
(Frame purity). Forces on Frame remain eV/Å during the FD loop.

**Key exclusions (binding from closed spec).**

- DFPT / analytic dynamical matrices → `dfpt-phonon-si` (depends on this foothold ✓).
- Finite-\(q\) band structure beyond the supercell fold (phonopy-style path).
- Acoustic sum-rule **projection** as a required product step (full \(3N\) is the
  surface; near-zero acoustics are an acceptance check).
- Molecular thermochemistry / IR / free-molecule TR projection (stays on
  `cpu::vib::Hessian`).
- Public `*Config` bags, `make_*` / `build_*` factories, CUDA phonon path.
- Live CP2K `VIBRATIONAL_ANALYSIS` at ctest time (offline pins only).

## Implementation

Public surface only (`cpu::Frame` + ctor args / public members; no `PhononConfig`,
no factories). Headers:
`include/atomiverse/cpu/vibration/phonon.h`,
`phonon_result.h`, `supercell.h`, `phonon_diagonalizer.h`.

Minimal sequence (matches `reg_fd-phonon-supercell.cpp` — Frame already is the
2×2×2 conventional supercell; `Phonon(1,1,1)` takes Γ of that cell):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/sw.h"
#include "atomiverse/cpu/vibration/phonon.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// diamond Si conventional 2×2×2, a = 5.431 Å → N = 64, L = 10.862 Å
cpu::Frame sc = /* build diamond_conventional(5.431, 2); see regression */;

cpu::SWPotential pot(molrs::Element::Si, "stillinger-weber");
const double cutoff = pot.cutoff();  // a*σ ≈ 3.7712 Å; need L > 2×cutoff

cpu::vib::Phonon phonon(/*na=*/1, /*nb=*/1, /*nc=*/1);
// optional expansion of a true unit cell:
// phonon.na = 2; phonon.nb = 2; phonon.nc = 2;
// or: cpu::Frame sc = cpu::vib::expand_supercell(unit, 2, 2, 2);

phonon.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, false});
phonon.on(ATV::vib::Phase::force).add(cpu::StillingerWeber{pot});
// PW/GPW later: .add(cpu::pw::PWEvaluator{…}.with_gradient(true));

phonon.build();
cpu::vib::PhononResult r = phonon.run(sc);
// r.frequencies [cm^-1], length 3*N; r.hessian [eV/Å²]; r.n_atoms
```

Build and run the regression (`ATV_BUILD_REGRESSIONS=ON`; SW twin needs MD
potentials + vib; PW/GPW flags match the CP2K-loop default configure):

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
  cmake --build build --target reg_fd-phonon-supercell -j$(nproc)
  ./build/regressions/reg_fd-phonon-supercell
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/vibration/phonon.h` | `cpu::vib::Phonon` preset |
| `include/atomiverse/cpu/vibration/phonon_result.h` | `PhononResult` |
| `include/atomiverse/cpu/vibration/supercell.h` | `expand_supercell` |
| `include/atomiverse/cpu/vibration/phonon_diagonalizer.h` | full \(3N\) eigensolve |
| `src/cpu/vibration/phonon.cpp` | `run()` / expand + FD loop |
| `tests/cpu/vibration/test_phonon_supercell.cpp` | unit: expand, acoustics, missing-force reject |
| `regressions/pw/fd-phonon-supercell/` | e2e + `REFERENCE.md` |

## Validation (vs CP2K / literature)

Source: [`regressions/pw/fd-phonon-supercell/REFERENCE.md`](../../../regressions/pw/fd-phonon-supercell/REFERENCE.md).
Oracle priority: offline **CP2K `VIBRATIONAL_ANALYSIS`** when a binary is
available; **this pin host had no CP2K module** — surrogate is the acoustic
sum rule plus the published Stillinger–Weber optical scale. **Never** live
CP2K in ctest.

### Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, **2×2×2 conventional** cubic supercell, **64 atoms** |
| Lattice constant \(a\) | **5.431 Å** (SW fit target); box \(L = 2a = 10.862\) Å |
| Why 2×2×2 | SW cutoff \(a\sigma = 3.7712\) Å requires \(L > 2\times\) cutoff (same gate as `reg_sw_si`) |
| `Phonon` factors | \(n_a=n_b=n_c=1\) on the already-built supercell (Γ of that cell) |
| Force engine | Stillinger–Weber Si (`cpu::StillingerWeber`) |
| SW parameters | Stillinger & Weber, *Phys. Rev. B* **31**, 5262 (1985) |
| FD step \(h\) | \(10^{-3}\) Å (`kHessianStepAngstrom`) |
| Neighbor | always-rebuild (`check=false`), cutoff \(a\sigma\) |

Fractional diamond sites tiled over \(2^3\) conventional 8-atom cells (see
`reg_fd-phonon-supercell.cpp`). `expand_supercell` is unit-tested separately
from this e2e twin.

### Why SW not PW for the first pin

A full PW/GPW FD Hessian is \(6N\) SCF+force evaluations. The **machinery**
under test is supercell expansion + crystal \(3N\) diagonalization + public
`Phonon` API. SW supplies periodic many-body forces with a literature potential
and finishes in well under the ≤ 60 s e2e budget. Binding
`PWEvaluator::with_gradient(true)` (or GPW) on `vib::Phase::force` is the same
public composition path for a later CP2K energy/frequency twin.

### Accuracy table

| Quantity | Atomiverse / pin | External / rule | Tol / band | Notes |
| -------- | ---------------: | --------------- | ---------- | ----- |
| Mode count | **192** | \(3N\), \(N=64\) | exact | hard-coded in regression |
| Acoustic \(\lvert\omega\rvert_\max\) (3 lowest) | **0.000 cm⁻¹** (measured) | Γ acoustic sum rule | **&lt; 30 cm⁻¹** | residual = FD / force noise |
| Highest optical \(\omega\) | **594.822 cm⁻¹** (measured) | SW optical scale | **[500, 900] cm⁻¹** | exp. Si TO(Γ) ≈ 520 cm⁻¹; SW overshoots |
| Frame pollution | none | result-only | — | \(\omega\), \(\Phi\) on `PhononResult` |

Measured twin (REFERENCE host note, 2026-07-23, Release, `OMP_NUM_THREADS=8`).
The optical band rejects unit disasters (Hz vs cm⁻¹, free-atom zero spectrum,
NaN) without freezing Atomiverse’s own first-run digits as a golden snapshot.

**CP2K.** When `cp2k.psmp` is available offline: run a fixed
`VIBRATIONAL_ANALYSIS` input on diamond Si (same \(a\), comparable force model
or Quickstep), record \(\omega\) [cm⁻¹] and wall time in REFERENCE.md with CP2K
version. Skeleton inputs may live under `tools/cp2k_oracle/inputs/`.

## Performance

Source: `regressions/pw/fd-phonon-supercell/REFERENCE.md`. Queue / handoff:
**`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| SW 64-atom Γ FD | n/a | ≈ **0.06** (informative) | n/a | n/a (**exempt**) | No CP2K `VIBRATIONAL_ANALYSIS` binary on pin host |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K module/binary on the build host for a fair wall twin this shift |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS; twin ≪ 1 s) |

When a CP2K binary is available offline: time a comparable
`VIBRATIONAL_ANALYSIS` twin, append \(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to
REFERENCE.md, require \(r \le 10\).

## Limitations

- First product pin is **classical SW**, not PW/GPW FD frequencies vs CP2K
  Quickstep (composition path for ab initio forces is public; pin is future work).
- Γ of the supercell only — no finite-\(q\) dispersion path beyond folding.
- No ASR post-projection as a product feature; acoustic near-zero is a check.
- No DFPT (`dfpt-phonon-si`), no molecular IR/thermochem on this preset.
- No CUDA phonon driver; no MPI multi-rank FD scaling gate here.
- SW optical \(\omega\) is systematically stiffer than experimental Si TO(Γ)
  (~520 cm⁻¹); compare SW literature, not Raman experiment, for this twin.

## References

- Offline pin: [`regressions/pw/fd-phonon-supercell/REFERENCE.md`](../../../regressions/pw/fd-phonon-supercell/REFERENCE.md)
- Regression: [`regressions/pw/fd-phonon-supercell/reg_fd-phonon-supercell.cpp`](../../../regressions/pw/fd-phonon-supercell/reg_fd-phonon-supercell.cpp)
- Closed acceptance (git): `wf/fd-phonon-supercell` / `.claude/specs/fd-phonon-supercell/`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Narrative style refs: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md),
  [Chapter 9 — GPW](../ch09-gpw.md)
- F. H. Stillinger and T. A. Weber, *Phys. Rev. B* **31**, 5262 (1985)
- Standard FD phonon machinery (phonopy-style finite displacements; CP2K
  `VIBRATIONAL_ANALYSIS` as intended Quickstep twin)
