# Slab (quasi-2D) Ewald — parallel-plate foothold (Yeh–Berkowitz)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `slab-2d-ewald-foothold` |
| Regression | `regressions/md/slab-2d-ewald-foothold/` |
| Public API | `cpu::CoulLong` + `cpu::KSpaceEwald` (`slab_factor` member) on `cpu::Evaluate` / `cpu::md::MD` |
| Roadmap / queue | `docs/roadmap/force-fields.md` § Direct Ewald slab; queue `docs/developer/metalwalls-method-queue.md` `slab-2d-ewald-foothold` (done, wave 1) |

## Summary

**Quasi-2D electrostatics** for planar (slab) geometries: 3D Ewald on a
*z*-extended cell plus the Yeh–Berkowitz \(k=0\) dipole correction, matching
LAMMPS `kspace_style ewald` with `kspace_modify slab f`. Public control is the
member `cpu::KSpaceEwald::slab_factor` (no `*Config` bag). The product gate is a
parallel-plate toy (4×\(+e\) / 4×\((-e)\) planes in a \(12\times12\times20\) Å
box, \(f=3\)) with hard-coded energy and mean \(F_z\) pins from offline LAMMPS
`run 0`. This is MetalWalls-map **E2** foothold physics for electrochemical /
electrode MD; **oracle is LAMMPS** (feature also exists there — not MetalWalls).
Regression: `reg_slab-2d-ewald-foothold`. Fair wall ratio
\(T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}} \approx 6.49\) (within 10×).

## Method

**3D Ewald split.** Point-charge Coulomb is split at Ewald parameter \(\alpha\)
into a short-range real-space erfc piece (`cpu::CoulLong`) and a reciprocal
piece (`cpu::KSpaceEwald` over `longrange::EwaldSum`):

\[
E_{\mathrm{real}}
  = \frac{k_e}{2}\sum_{i\neq j}
    q_i q_j \frac{\mathrm{erfc}(\alpha r_{ij})}{r_{ij}},
\quad
E_{\mathrm{recip}}
  = k_e\,\frac{2\pi}{V}\sum_{\mathbf{k}\neq 0}
    \frac{e^{-k^2/4\alpha^2}}{k^2}\,|S(\mathbf{k})|^2,
\]

plus self and net-charge terms (metal units: \(k_e =\) `units::coulomb`;
energies in eV). The reciprocal sum is implemented over the positive
hemisphere with prefactor \(4\pi/V\) (LAMMPS-equivalent).

**Slab (Yeh–Berkowitz).** When `slab_factor` \(f > 1\), the reciprocal sum uses
the extended period \(L_z' = f L_z\) and volume \(V' = L_x L_y L_z'\), and the
\(k=0\) slab correction of Yeh & Berkowitz is applied (LAMMPS
`Ewald::slabcorr`):

\[
E_{\mathrm{slab}}
  = k_e\,\frac{2\pi}{V'}
    \Bigl(
      M_z^2 - Q\sum_i q_i z_i^2 - Q^2 L_z'^2/12
    \Bigr),
\quad
M_z = \sum_i q_i z_i,\quad Q = \sum_i q_i.
\]

Neutral systems drop the \(Q\) terms. Forces: \(F_{z,i}\) receives a term
\(\propto q_i(M_z - Q z_i)\). This approximates non-periodic boundaries in \(z\)
while remaining periodic in \(xy\).

**Pin geometry (this slug).**

| Quantity | Value |
| -------- | ----- |
| Box | \(12\times 12\times 20\) Å orthorhombic |
| Boundary (LAMMPS twin) | `p p f` (ATV uses 3D min-image; vacuum + short cutoff make them equivalent here) |
| Charges | 4× \(+1\,e\) at \(z=5\) Å; 4× \(-1\,e\) at \(z=15\) Å on a 6 Å \(xy\) grid |
| Pair | `coul/long` cutoff **5.9 Å** (no same-plane pairs → real-space \(E=0\)) |
| \(\alpha\) (`gewald`) | **0.5** Å⁻¹ fixed |
| KSpace accuracy | \(10^{-6}\) |
| Slab factor | \(f = 3.0\) → \(L_z'=60\) Å, \(V'=8640\) Å³ |
| Dipole | \(M_z = -40\) e·Å |

**Key exclusions (non-goals of this slug)**

- True 2D-only Ewald (no 3D+slab vacuum) — future.
- PME / PPPM slab (`kspace_modify slab` on FFT styles) — not this pin.
- Constant-potential electrode charge solve (BO / CG) — other electrochem
  slugs (`const-pot-bo-parallel-plate`, …); charge never solved inside
  `KSpace*`.
- Live LAMMPS (or MetalWalls) inside ctest — offline pins only.
- Production slab MD equilibration / multi-rank product bench of this twin.

## Implementation

Public surface only (`cpu::Frame` + component constructors / members; no
`*Config`, no `make_*` / `build_*` factories). Headers:

- `include/atomiverse/cpu/potentials/coul_long.h` — real-space long Coulomb
- `include/atomiverse/cpu/longrange/ewald.h` — `KSpaceEwald` + `slab_factor`
- `include/atomiverse/cpu/longrange/slab_corr.h` — Yeh–Berkowitz helpers
- `include/atomiverse/cpu/evaluate/evaluate.h` — single-point Evaluate preset
- `include/atomiverse/cpu/neighbor/neighbor_list.h`

Minimal sequence (matches `reg_slab-2d-ewald-foothold.cpp`):

```cpp
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

using namespace ATV;

// frame: 8 atoms, charges ±1 e, box 12×12×20 Å (see regression)
const double cutoff = 5.9;
const double alpha = 0.5;
const double accuracy = 1e-6;
const double slab_factor = 3.0;

cpu::Evaluate eval;
eval.set_evaluator(cpu::CoulLong{alpha, cutoff});
{
    cpu::KSpaceEwald recip(alpha, accuracy);
    recip.slab_factor = slab_factor;   // LAMMPS kspace_modify slab 3.0
    eval.set_evaluator(std::move(recip));
}
eval.set_neighbor(cpu::md::NeighborList<>{cutoff, /*skin=*/0.5});
eval.build();
eval.run(frame);
// frame.meta.get(kEnergy) [eV]; forces on kAtom_fx/fy/fz [eV/Å]
```

On classical MD the same components bind via `cpu::md::MD::set_evaluator` /
default force-phase bindings (`CoulLong` + `KSpaceEwald` as the matched
long-range pair). `slab_factor ≤ 1` (default `0`) keeps pure 3D Ewald.

```bash
# MD-only configure (electrochem / classical loop)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_slab-2d-ewald-foothold -j$(nproc)
'
./build/regressions/reg_slab-2d-ewald-foothold
```

Oracle input (offline only):  
`tools/metalwalls_oracle/inputs/lammps/slab-2d-ewald-foothold/in.lammps`.

## Validation (oracle: LAMMPS)

Source: `regressions/md/slab-2d-ewald-foothold/REFERENCE.md` (offline pin;
**never** live LAMMPS in ctest).

| Field | Value |
| ----- | ----- |
| Oracle | **LAMMPS** (`lmp`) |
| `oracle_version` | develop / patch_30Mar2026-1074-g7f680de296 |
| Twin | `run 0`, `units metal`, `pair_style coul/long 5.9`, `kspace_style ewald 1e-6`, `kspace_modify gewald 0.5` + `slab 3.0` |
| Why not MetalWalls | Slab Ewald exists in LAMMPS; loop rule: LAMMPS is the binding oracle when the feature overlaps |

| Quantity | Atomiverse gate | LAMMPS pin | Tol | Notes |
| -------- | --------------: | ---------: | --- | ----- |
| \(E\) (slab) | hard-coded match | **63.0848970842656 eV** | rel \(10^{-4}\) or abs \(5\times10^{-3}\) eV | \(E_{\mathrm{LONG}}=E_{\mathrm{TOTAL}}\); \(E_{\mathrm{COUL}}=0\) |
| mean \(F_z\) (+ plane) | hard-coded match | **+2.51351 eV/Å** | \(10^{-3}\) eV/Å | four \(+e\) atoms at \(z=5\) |
| mean \(F_z\) (− plane) | hard-coded match | **−2.51351 eV/Å** | \(10^{-3}\) eV/Å | four \(-e\) atoms at \(z=15\) |
| \(E(f=3)-E(f=0)\) | \(|\Delta E| > 1\) eV | — | guard | no-op slab implementation fails |

Physics citation: Yeh & Berkowitz, *J. Chem. Phys.* **111**, 3155 (1999);
LAMMPS `Ewald::slabcorr`. Unit tests for `slab_factor` live under
`tests/cpu/longrange/test_slab_ewald.cpp` (not e2e).

## Performance

Source: same `REFERENCE.md` fair wall (not ctest). Geometry as above;
`OMP_NUM_THREADS=1`; **2000** force evaluations after warm-up. LAMMPS:
`run 2000` with `fix setforce 0` (reported loop time). ATV: hot-path
`CoulLong` + `KSpaceEwald` only (build once; no Evaluate re-init).

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | -----------------------------------------: | ---- | ------ |
| 8-atom slab, 2000 force evals | **0.205141** | **1.330859** | **6.49** | ✓ | half-sphere Ewald; tiny system (not production slab MD) |

Gate: \(r \le 10\) (`within_10x`: true). One applied optimize noted in
REFERENCE: reciprocal sum over the positive hemisphere with prefactor
\(4\pi/V\) (was full sphere at \(2\pi/V\)).

## Limitations

- **Foothold only.** Energy/forces on a minimal parallel-plate toy; not a full
  electrode electrolyte MD product case.
- **Direct Ewald only.** PME/PPPM slab path not covered by this slug.
- **Boundary model.** ATV does not emulate LAMMPS `boundary p p f` as a
  separate box type; vacuum + cutoff makes the pin equivalent. Users with
  dense slabs must ensure adequate vacuum / \(f\) as in the LAMMPS docs.
- **Speed debt.** \(r\approx 6.5\) on this micro twin is inside the 10× gate but
  is not a claim of production Ewald throughput; half-sphere reciprocal is
  already applied.
- **Charge solve is orthogonal.** Constant-potential electrode charges are
  Components on `cpu::md::MD`, never folded into `KSpaceEwald`.
- **No true 2D Ewald** (sum only in-plane) in-tree yet.

## References

- Offline pin + fair wall: `regressions/md/slab-2d-ewald-foothold/REFERENCE.md`
- Regression consumer: `regressions/md/slab-2d-ewald-foothold/reg_slab-2d-ewald-foothold.cpp`
- LAMMPS twin input: `tools/metalwalls_oracle/inputs/lammps/slab-2d-ewald-foothold/in.lammps`
- Oracle notes: `tools/metalwalls_oracle/README.md`
- Queue / handoff: `docs/developer/metalwalls-method-queue.md`,
  `docs/developer/metalwalls-method-handoff.md`
- Design: `docs/developer/metalwalls-method-loop.md`
- Roadmap: `docs/roadmap/force-fields.md` (Direct Ewald + slab)
- I.-C. Yeh, M. L. Berkowitz, *J. Chem. Phys.* **111**, 3155 (1999)
- V. Ballenegger et al., *J. Chem. Phys.* **131**, 094107 (2009) (non-neutral extension)
- P. P. Ewald, *Ann. Phys.* **369**, 253 (1921)
