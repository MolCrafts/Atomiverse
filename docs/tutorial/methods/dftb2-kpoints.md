# Multi-k SCC-DFTB2 — 1-D carbon chain (4×1×1 Monkhorst–Pack)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dftb2-kpoints` |
| Regression | `regressions/semiempirical/dftb2-kpoints/` |
| Public API | `cpu::semiempirical::DFTB2` (`k_mesh`, `k_time_reversal` members) + `cpu::Evaluate` |
| Roadmap | [`semiempirical.md`](../../roadmap/semiempirical.md) — multi-k energy ✓; multi-k F/stress ○ |
| Queue | TB method loop (`docs/developer/tb-method-queue.md`) `dftb2-kpoints` (done, harness wave 3) |
| Family oracle | **DFTB+** only (never OpenMOPAC / xtb) |

## Summary

**Self-consistent-charge DFTB2** (Elstner *et al.*) with **Monkhorst–Pack
k-mesh beyond Γ**: Bloch-phased lattice sums of two-center \(H^0(\mathbf{k})\)
and \(S(\mathbf{k})\), k-weighted band energy and Mulliken charges, and
real-space (Γ-style) screened \(\gamma\) + pair repulsive. The product gate is
the same minimal **1-D carbon dimer crystal** used by
`dftb2-periodic-gamma`, now sampled with `k_mesh = {4, 1, 1}` under
`cpu::Evaluate` + `cpu::semiempirical::DFTB2` (default
`DftbSkSet::MioSketch`). Energy is written to `frame.meta.get(kEnergy)` in
**eV**.

This page is the TB-loop foothold for **multi-k energy**, not multi-k
forces/stress, DFTB3, or open-shell k-mesh. Family numerical oracle remains
**DFTB+**; absolute energy on this twin is the **ATV mio-style static
Hamiltonian** (same hygiene as `dftb2-periodic-gamma`) because MioSketch C–C
is not a full DFTB.org mio-1-1 `.skf` dump and host builds may lack a DFTB+
binary. Bit-identical DFTB+ energy identity ships for H–H mio-1-1
(`dftb2-dftbplus-energy-pin`) and 3ob organics (`dftb2-3ob-organic`).

## Method

**Level.** Restricted closed-shell SCC-DFTB2. Electronic Hamiltonian in a
minimal valence AO basis (s on H; s/p on C/N/O) with two-center
Slater–Koster interpolation of \(H_{\mu\nu}^0(R)\) and \(S_{\mu\nu}(R)\),
plus a second-order charge-fluctuation (SCC) term and a short-range
pair-repulsive potential. Citation: Elstner *et al.*, *Phys. Rev. B*
**58**, 7260 (1998), DOI
[10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260).

### Energy (closed shell, multi-k)

With k-point weights \(w_k\) (normalized \(\sum_k w_k = 1\)), complex Bloch
matrices \(H^{0k}\), \(S^k\), and density \(P^k\),

\[
E = \sum_k w_k\,\mathrm{Re}\,\mathrm{Tr}\bigl(P^k H^{0k}\bigr)
  + E^{(2)}[\{\Delta q\}]
  + E_{\mathrm{rep}},
\]

where the SCC second-order piece is k-independent,

\[
E^{(2)} = \tfrac12\sum_{AB}\Delta q_A\,\gamma_{AB}\,\Delta q_B,
\]

and \(\Delta q_A\) is the **k-weighted** Mulliken fluctuation. The SCC shift
on each atom,

\[
\varepsilon_A = \sum_B \gamma_{AB}\,\Delta q_B,
\]

is real and k-independent; it is added as an onsite contribution to the
generalized eigenproblem at every \(\mathbf{k}\).

### Bloch lattice sums

For fractional wavevector \(\boldsymbol{\kappa}\) (crystal coordinates) and
lattice vectors \(\mathbf{T}=n_0\mathbf{a}_1+n_1\mathbf{a}_2+n_2\mathbf{a}_3\),

\[
H^{0k}_{\mu\nu}
  = \sum_{\mathbf{T}}
    e^{2\pi i\,\boldsymbol{\kappa}\cdot\mathbf{n}}\,
    H^{0,\mathrm{SK}}_{\mu\nu}\big(|\mathbf{R}_\nu+\mathbf{T}-\mathbf{R}_\mu|\big),
\quad
S^{k}_{\mu\nu}
  = \sum_{\mathbf{T}}
    e^{2\pi i\,\boldsymbol{\kappa}\cdot\mathbf{n}}\,
    S^{\mathrm{SK}}_{\mu\nu}\big(|\mathbf{R}_\nu+\mathbf{T}-\mathbf{R}_\mu|\big).
\]

Shell enumeration matches the Γ-path cutoff (element `r_cut`, extended for
mio-1-1 H–H grids when present). Screened \(\gamma_{AB}\) and \(E_{\mathrm{rep}}\)
use the **same real-space lattice shell** as Γ — **not** Ewald long-range
Coulomb.

### Monkhorst–Pack mesh (CP2K / DFTB+ convention)

Subdivisions \(N_d\) per reciprocal axis (`k_mesh = {N_0, N_1, N_2}`):

\[
\kappa_d = \frac{2r - N_d - 1}{2 N_d},\qquad r = 1,\ldots,N_d.
\]

Uniform weight \(w_0 = 1/(N_0 N_1 N_2)\) before folding. With
`k_time_reversal = true` (default), \(\pm\mathbf{k}\) pairs are folded and
weights doubled — exact for real-Hamiltonian closed-shell densities. Default
`k_mesh = {1,1,1}` keeps the **real Γ solver** (pin-safe path for
`dftb2-periodic-gamma` and molecular cases).

DFTB+ twin uses the equivalent SupercellFolding:

```text
KPointsAndWeights = SupercellFolding {
  4 0 0
  0 1 0
  0 0 1
  0.0 0.0 0.0
}
```

### Parameters (this twin)

| Setting | Value |
| ------- | ----- |
| `sk_set` | `MioSketch` (default) |
| `k_mesh` | `{4, 1, 1}` (along chain axis) |
| `k_time_reversal` | `true` |
| `scc_tol` | \(1\times10^{-10}\) |
| `max_scc` | 50 (default) |
| `with_forces` | **false** (multi-k forces throw) |
| `with_stress` | false |
| `charge` / `multiplicity` | 0 / 1 (closed shell required for multi-k) |

Static tables: `basis_data/dftb_mio_tables.h` (H/C/N/O exponential sketches +
mio-1-1 H–H). **Not** a runtime `.skf` reader; **not** bit-identical DFTB.org
mio-1-1 for C–C.

### Observables

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E\) | Total electronic + repulsive on `kEnergy` | eV |
| \(E_\Gamma\) | Same cell at `k_mesh={1,1,1}` (control) | eV |
| \(\Delta q_A\) | k-weighted Mulliken fluctuations (`last_mulliken_dq()`) | e |

### Key exclusions (this slug)

- Multi-k **analytic forces / stress** → still ○ (any `k_mesh[d] > 1` with
  `with_forces` / electronic `with_stress` throws).
- **Open shell** or **ThreeOb** multi-k → not supported (throw).
- **DFTB3** third-order γ, **TD-DFTB**, noncollinear spin.
- **Dispersion** D3 under multi-k electronic path (D3 is geometry-only when
  enabled on molecular paths; this reg does not set `d3`).
- Live **DFTB+** at ctest time (offline pins only; `tools/tb_oracle/`).
- OpenMOPAC as reference (**forbidden** for the DFTB family).
- GFN-xTB as a “DFTB3 successor” (**forbidden** framing).

## Implementation

Public surface only (`cpu::Frame` + concrete method members; **no** `*Config`
bag, **no** `make_*` / `build_*` factories). Headers:
`include/atomiverse/cpu/semiempirical/dftb2.h`,
`include/atomiverse/cpu/evaluate/evaluate.h`,
`include/atomiverse/cpu/frame/frame.h`. Implementation:
`src/cpu/semiempirical/dftb2.cpp` (`monkhorst_pack_mesh`, Bloch-phased
`fill_sk_block_phased`, k-weighted SCC). Static tables:
`include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h`.

Minimal sequence (matches `reg_dftb2-kpoints.cpp`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

// 1-D C dimer crystal: r(C–C)=1.4 Å, Lx=2.8 Å (same as dftb2-periodic-gamma).
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
dftb.with_forces = false;       // multi-k is energy-only
dftb.scc_tol = 1e-10;
dftb.k_mesh = {4, 1, 1};        // Monkhorst–Pack along chain
dftb.k_time_reversal = true;
eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
eval.build();
eval.run(frame);

const double E_eV = frame.meta.get(kEnergy);
// gate: |E − (−98.9944775364)| ≤ 0.05 eV
// and |E − E_Γ| > 0.01 eV with E_Γ ≈ −113.5976598145 eV
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
  cmake --build build -j"$(nproc)" --target atomiverse reg_dftb2-kpoints
  cmake --build build --target run_reg_dftb2-kpoints
'
```

Related unit footholds (when present): `tests/cpu/semiempirical/test_dftb2.cpp`
for Γ chain finiteness; multi-k product gate is the regression only.

## Validation (vs offline pin)

Source: `regressions/semiempirical/dftb2-kpoints/REFERENCE.md`
(offline pin; **never** live DFTB+ in ctest).

```text
oracle: dftb+
oracle_version: DFTB+ family (offline twin; ATV mio-style SK Hamiltonian pin)
family_oracle: dftb+
```

**Geometry.** Two C atoms, orthorhombic cell \(L_x=2.8\) Å, \(L_y=L_z=12.0\) Å
(identical to `dftb2-periodic-gamma`).

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----: | ----: | ----: |
| C | 0.00 | 0.00 | 0.00 |
| C | 1.40 | 0.00 | 0.00 |

**k-mesh.** \(4\times1\times1\) Monkhorst–Pack along \(x\); time-reversal on.
Offline twin: `tools/tb_oracle/inputs/dftbplus/dftb2-kpoints/`.

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Total energy \(E\) (4×1×1) [eV] | **−98.9944775364** | literature (ATV mio-style twin; family dftb+) | ±0.05 eV | Hard pin in `reg_dftb2-kpoints` |
| \(\lvert E - E_\Gamma\rvert\) [eV] | \(> 0.01\) | sanity | — | Multi-k must separate from Γ |
| \(E_\Gamma\) (1×1×1 control) [eV] | **−113.5976598145** | matches `reg_dftb2-periodic-gamma` | ±0.05 eV | Shared cell / Hamiltonian |

Measured host residual vs the multi-k pin is far tighter than the ±0.05 eV
hygiene window (pin is the ATV Hamiltonian twin, not DFTB.org C–C digits).
Related absolute **DFTB+ 24.1** energy identity (different system: H₂
mio-1-1): `dftb2-dftbplus-energy-pin` — \(E=-18.36635052\) eV, tol
\(10^{-3}\) eV. Do **not** claim DFTB+ absolute digit parity for this C-chain
mesh until bit-identical C–C SK tables ship.

## Performance

Source: `REFERENCE.md`.

```text
perf_exempt: true
perf_exempt_reason: no offline DFTB+ binary on host; fair wall deferred
```

No fair wall twin \(T_{\mathrm{ATV}}/T_{\mathrm{DFTB+}}\) is asserted for this
slug. A related molecular energy pin (`dftb2-dftbplus-energy-pin`) reports
informal walls ATV ~0.06 s vs DFTB+ 24.1 ~0.15 s (\(r\approx 0.4\), ≤10×) —
not this multi-k twin, not ctest. Gate convention (TB loop):
\(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) unless `perf_exempt`.

## Limitations

| Item | Status |
| ---- | ------ |
| Multi-k SCC-DFTB2 energy (restricted MioSketch) | **closed** (this slug) |
| Γ control identity with `dftb2-periodic-gamma` | **closed** (regression control) |
| Absolute DFTB+ digits for C–C MioSketch | deferred (ATV Hamiltonian twin) |
| Multi-k analytic forces / electronic stress | **○** (throws if requested) |
| Open-shell / ThreeOb multi-k | **○** (throws) |
| Real-space \(\gamma\) (no Ewald) | intentional SK-cutoff model |
| DFTB3 / TD-DFTB / noncollinear spin | roadmap / queue |
| Live oracle in CI | **forbidden** — offline pins only |

Foothold ≠ capability closed: multi-k **energy** is product-gated; multi-k
gradients, metallic smearing on DFTB, and bit-identical multi-element SK
remain open ladder tips.

## References

- Offline pin: [`regressions/semiempirical/dftb2-kpoints/REFERENCE.md`](../../../regressions/semiempirical/dftb2-kpoints/REFERENCE.md)
- Regression: [`reg_dftb2-kpoints.cpp`](../../../regressions/semiempirical/dftb2-kpoints/reg_dftb2-kpoints.cpp)
- Public API: [`include/atomiverse/cpu/semiempirical/dftb2.h`](../../../include/atomiverse/cpu/semiempirical/dftb2.h)
- Offline twin: [`tools/tb_oracle/inputs/dftbplus/dftb2-kpoints/`](../../../tools/tb_oracle/inputs/dftbplus/dftb2-kpoints/)
- Family rules: [`tools/tb_oracle/README.md`](../../../tools/tb_oracle/README.md)
- Roadmap: [`docs/roadmap/semiempirical.md`](../../roadmap/semiempirical.md)
- Queue / handoff: [`docs/developer/tb-method-queue.md`](../../developer/tb-method-queue.md),
  [`docs/developer/tb-method-handoff.md`](../../developer/tb-method-handoff.md)
- Sibling Γ-point page: [`dftb2-periodic-gamma.md`](dftb2-periodic-gamma.md)
- Sibling absolute DFTB+ pin: [`dftb2-dftbplus-energy-pin.md`](dftb2-dftbplus-energy-pin.md)
- Literature: Elstner *et al.*, *Phys. Rev. B* **58**, 7260 (1998)
