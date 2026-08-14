# Γ-point periodic SCC-DFTB2 — 1-D carbon chain

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dftb2-periodic-gamma` |
| Regression | `regressions/semiempirical/dftb2-periodic-gamma/` |
| Public API | `cpu::semiempirical::DFTB2` (members) + `cpu::Evaluate` |
| Roadmap | [`semiempirical.md`](../../roadmap/semiempirical.md) — Periodic DFTB (Γ-point) ✓ |
| Queue | TB method loop (`docs/developer/tb-method-queue.md`) `dftb2-periodic-gamma` (done) |
| Family oracle | **DFTB+** (absolute pin deferred to bit-identical SK; see Validation) |

## Summary

**Self-consistent-charge DFTB2** (Elstner *et al.*) at the **Γ point** for a
periodic cell: lattice-summed two-center \(H^0\) and \(S\), screened
DFTB+ Elstner \(\gamma\) (short-range expGamma), and pair repulsive within static SK cutoffs.
The product gate is a minimal **1-D carbon dimer crystal** (two C per
orthorhombic cell) under `cpu::Evaluate` + `cpu::semiempirical::DFTB2`
(default `DftbSkSet::MioSketch`). Energy is written to
`frame.meta.get(kEnergy)` in **eV**. Molecular limit is recovered when all
cell edges \(\gg r_\mathrm{cut}\); the regression requires a finite gap
between the dense-chain energy and the same dimer in a 20 Å cubic box.

This page is the TB-loop foothold for **Γ-point PBC energy**, not k-mesh,
stress, or DFTB3. Family numerical oracle remains **DFTB+**; the absolute
energy pin on this twin is the **ATV mio-style static Hamiltonian** (same
hygiene as `h2o_dftb2_mio`) because host builds may lack a DFTB+ binary and
MioSketch C–C is not a full DFTB.org mio-1-1 `.skf` dump. Bit-identical
DFTB+ energy identity ships separately for H–H mio-1-1
(`dftb2-dftbplus-energy-pin`) and 3ob organics (`dftb2-3ob-organic`).

## Method

**Level.** Restricted closed-shell SCC-DFTB2. Electronic Hamiltonian in a
minimal valence AO basis (s on H; s/p on C/N/O) with two-center
Slater–Koster interpolation of \(H_{\mu\nu}^0(R)\) and \(S_{\mu\nu}(R)\),
plus a second-order charge-fluctuation (SCC) term and a short-range
pair-repulsive potential. Citation: Elstner *et al.*, *Phys. Rev. B*
**58**, 7260 (1998), DOI
[10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260).

**Energy (closed shell).** With band occupations \(n_i\) and eigenvalues
\(\varepsilon_i\) of the SCC Fock matrix in the nonorthogonal basis,

\[
E = \sum_i n_i\,\varepsilon_i
  - \tfrac12\sum_{AB}\Delta q_A\,\gamma_{AB}\,\Delta q_B
  + E_{\mathrm{rep}},
\]

where \(\Delta q_A\) are Mulliken charge fluctuations, \(\gamma_{AB}\) is the
screened Coulomb kernel (Hubbard \(U\) at short range, \(1/R\) tail), and
\(E_{\mathrm{rep}}\) is the tabulated/exponential pair repulsive. Double
counting of the SCC piece follows the DFTB+ convention (electronic part
as \(\mathrm{Tr}(P H^0)+E_{\mathrm{scc}}\)).

**Γ-point PBC.** There is **no k-mesh**. Matrices and real-space sums are
taken over lattice translations \(\mathbf{T}=n_0\mathbf{a}_1+n_1\mathbf{a}_2+n_2\mathbf{a}_3\)
with \(|\mathbf{T}|\) up to the maximum pair cutoff (element `r_cut`,
extended for mio-1-1 H–H grids when present):

\[
H^0_{\mu\nu}
  = \sum_{\mathbf{T}}
    H^{0,\mathrm{SK}}_{\mu\nu}\big(|\mathbf{R}_\nu+\mathbf{T}-\mathbf{R}_\mu|\big),
\quad
S_{\mu\nu}
  = \sum_{\mathbf{T}}
    S^{\mathrm{SK}}_{\mu\nu}\big(|\mathbf{R}_\nu+\mathbf{T}-\mathbf{R}_\mu|\big).
\]

Screened \(\gamma_{AB}\) and \(E_{\mathrm{rep}}\) use the same lattice shell
enumeration, truncated at the static SK cutoff — **not** Ewald long-range
Coulomb. When all \(L_d \gg r_\mathrm{cut}\), only \(\mathbf{T}=\mathbf{0}\)
contributes and the molecular (isolated) limit is recovered.

**Parameters (this twin).** `sk_set = MioSketch` (default): multi-element
H/C/N/O from in-tree `basis_data/dftb_mio_tables.h` (exponential sketches for
most pairs) plus mio-1-1 H–H radial grids in `dftb_mio_hh_sk.h`. **Not** a
runtime `.skf` reader; **not** bit-identical DFTB.org mio-1-1 for C–C.

**SCC settings (regression).** `scc_tol = 1e-10`, `max_scc = 50` (default),
`with_forces = false`, `charge = 0`, `multiplicity = 1`.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E\) | Total electronic + repulsive energy on `kEnergy` | eV |
| \(\Delta q_A\) | Mulliken fluctuations (`last_mulliken_dq()`) | e |

**Key exclusions (this slug).**

- **k-point** sampling beyond Γ → queue `dftb2-kpoints` (○).
- Analytic **stress** / variable-cell DFTB → roadmap ○.
- **DFTB3** (third-order γ) → separate method / queue.
- Collinear **spin** product → `spin-dftb2-o2` (forces still FD for open shell).
- Analytic nuclear **forces** under PBC (min-image molecular analytic path
  only; this regression is energy-only).
- **Dispersion** (D3/D4) on DFTB → `dftb2-dispersion-d3` (○).
- Live **DFTB+** at ctest time (offline pins only; see `tools/tb_oracle/`).
- OpenMOPAC as reference (**forbidden** for the DFTB family).
- GFN-xTB as a “DFTB3 successor” (**forbidden** framing; xtb oracle is a
  separate family).

## Implementation

Public surface only (`cpu::Frame` + concrete method members; **no** `*Config`
bag, **no** `make_*` / `build_*` factories). Headers:
`include/atomiverse/cpu/semiempirical/dftb2.h`,
`include/atomiverse/cpu/evaluate/evaluate.h`,
`include/atomiverse/cpu/frame/frame.h`. Implementation:
`src/cpu/semiempirical/dftb2.cpp` (lattice enumeration + Γ H⁰/S/γ/rep).
Static tables: `include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h`.

Minimal sequence (matches `reg_dftb2-periodic-gamma.cpp`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

// 1-D C dimer crystal: r(C–C)=1.4 Å, Lx=2.8 Å → second-neighbor images in range.
// Ly=Lz=12 Å isolate the chain from its periodic images in y/z.
cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 6;
frame.span(kAtom_type)[1] = 6;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_x)[1] = 1.4;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_mass)[i] = 12.011;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::orthorhombic(2.8, 12.0, 12.0);

cpu::Evaluate eval;
cpu::semiempirical::DFTB2 dftb;  // sk_set = MioSketch (default)
dftb.with_forces = false;
dftb.scc_tol = 1e-10;
eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
eval.build();
eval.run(frame);

const double E_eV = frame.meta.get(kEnergy);
```

Build / run (SE-only configure; scoped targets — never bare full-tree build
for iteration):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON \
        -DATV_BUILD_MD=OFF -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF \
        -DATV_BUILD_VIB=OFF -DATV_BUILD_PATH=OFF \
        -DATV_BUILD_POTENTIALS=OFF -DATV_BUILD_NEIGHBOR=OFF \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_KSPACE=OFF -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j"$(nproc)" --target atomiverse reg_dftb2-periodic-gamma
  cmake --build build --target run_reg_dftb2-periodic-gamma
'
```

Unit footholds (finite energy; PBC vs molecular separation):
`tests/cpu/semiempirical/test_dftb2.cpp`
(`PeriodicGammaChainFinite`, `PeriodicGammaDiffersFromIsolatedDimer`).

## Validation (vs offline pin)

Source: `regressions/semiempirical/dftb2-periodic-gamma/REFERENCE.md`
(offline pin; **never** live DFTB+ in ctest).

```text
oracle: literature
oracle_version: ATV-mio-style-sketch (Hamiltonian twin; not DFTB.org mio-1-1)
family_oracle: dftb+
```

**Geometry.** Two C atoms, orthorhombic cell \(L_x=2.8\) Å, \(L_y=L_z=12.0\) Å.

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----: | ----: | ----: |
| C | 0.00 | 0.00 | 0.00 |
| C | 1.40 | 0.00 | 0.00 |

With \(r(\mathrm{C–C})=1.4\) Å and \(L_x=2.8\) Å, each carbon has
nearest-neighbor contacts along \(\pm x\) from the partner **and** its
periodic images; lattice-sum \(H/S\) differs from a single min-image
molecular dimer.

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Total energy \(E\) [eV] | **−113.5976598145** | literature (ATV mio-style twin) | ±0.05 eV | Hard pin in `reg_dftb2-periodic-gamma` |
| \(\lvert E-E_{\mathrm{mol}}\rvert\) [eV] | \(> 0.05\) | sanity (same dimer, 20 Å cubic) | — | Lattice images must matter |

Related absolute **DFTB+ 24.1** energy identity (different system: H₂
mio-1-1): `regressions/semiempirical/dftb2-dftbplus-energy-pin/` —
\(E=-18.36635052\) eV, tol \(10^{-3}\) eV. Use that path when claiming
oracle-version parity; do **not** treat this C-chain pin as a DFTB+ digit
match until bit-identical C–C SK tables ship.

## Performance

Source: `REFERENCE.md`.

```text
perf_exempt: true
perf_exempt_reason: no offline DFTB+ binary on host; fair wall deferred
```

No fair wall twin \(T_{\mathrm{ATV}}/T_{\mathrm{DFTB+}}\) is asserted for this
slug. A related molecular energy pin
(`dftb2-dftbplus-energy-pin`) reports informal walls ATV ~0.06 s vs DFTB+
24.1 ~0.15 s (\(r\approx 0.4\), ≤10×) — not this periodic twin, not ctest.

## Limitations

- Absolute energy is an **ATV Hamiltonian twin**, not DFTB+ absolute digits
  for C–C MioSketch. Family oracle remains DFTB+ once bit-identical SK
  coverage lands.
- Real-space truncation of \(\gamma\) (no Ewald) is intentional for this
  SK-cutoff model; long-range electrostatics in dense 3-D ionic cells are
  outside the pin.
- Forces on this regression are off; PBC analytic forces / stress are not
  product-gated here.
- k-mesh, DFTB3, D3, TD-DFTB, and noncollinear spin remain roadmap ○.
- Elements limited to H/C/N/O static tables; unsupported \(Z\) throws.

## References

- Offline pin: [`regressions/semiempirical/dftb2-periodic-gamma/REFERENCE.md`](../../../regressions/semiempirical/dftb2-periodic-gamma/REFERENCE.md)
- Regression: [`reg_dftb2-periodic-gamma.cpp`](../../../regressions/semiempirical/dftb2-periodic-gamma/reg_dftb2-periodic-gamma.cpp)
- Public API: [`include/atomiverse/cpu/semiempirical/dftb2.h`](../../../include/atomiverse/cpu/semiempirical/dftb2.h)
- Roadmap: [`docs/roadmap/semiempirical.md`](../../roadmap/semiempirical.md)
- Queue / handoff: [`docs/developer/tb-method-queue.md`](../../developer/tb-method-queue.md),
  [`docs/developer/tb-method-handoff.md`](../../developer/tb-method-handoff.md)
- Oracle tooling: [`tools/tb_oracle/README.md`](../../../tools/tb_oracle/README.md)
- Literature: Elstner *et al.*, *Phys. Rev. B* **58**, 7260 (1998)
- Related: H₂ DFTB+ energy pin `dftb2-dftbplus-energy-pin`; analytic
  forces `dftb2-analytic-forces`; spin `spin-dftb2-o2`
