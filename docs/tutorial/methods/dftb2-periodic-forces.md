# Γ-PBC SCC-DFTB2 analytic nuclear forces — off-eq 1-D C chain

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dftb2-periodic-forces` |
| Regression | `regressions/semiempirical/dftb2-periodic-forces/` |
| Public API | `cpu::semiempirical::DFTB2` (`with_forces = true`, restricted MioSketch) + `cpu::Evaluate` |
| Roadmap | [`semiempirical.md`](../../roadmap/semiempirical.md) — lattice-image-consistent Γ-PBC nuclear \(F\) ✓ |
| Queue | TB method loop wave 3 (`docs/developer/tb-method-queue.md`) `dftb2-periodic-forces` (done) |
| Family oracle | **DFTB+** (formula / force-module boundary; absolute SK digits deferred) |

## Summary

**Closed-form nuclear forces** for **Γ-point periodic** self-consistent-charge
DFTB2 (Elstner *et al.*) under the **same lattice-image walk** used for the
energy (\(H^0\), \(S\), screened \(\gamma\), repulsive). This deepens the
energy-only periodic foothold (`dftb2-periodic-gamma`) and the molecular
analytic-force path (`dftb2-analytic-forces`): forces are no longer a
min-image-only contraction of a lattice-summed energy.

The product gate is an **off-equilibrium 1-D carbon dimer crystal** (two C per
orthorhombic cell, \(r=1.20\) Å, \(L_x=2.8\) Å) so left/right lattice shells
differ and \(\mathbf{F}\ne 0\). Forces land on `frame.span(kAtom_f*)` in
**eV/Å**; energy on `frame.meta.get(kEnergy)` in **eV**. Public surface:
concrete `DFTB2` with members — no public `*Config`, no `make_*` / `build_*`.
Oracle family: **DFTB+** only (never OpenMOPAC for DFTB). Absolute \(E\)/\(F\)
vs DFTB.org mio-1-1 is **not** claimed (MioSketch SK); the hard pin is
**analytic \(F=-\nabla E\)** vs re-SCC central FD of the ATV lattice-summed
energy. `perf_exempt: true` (no DFTB+ binary / fair wall on pin host).

## Method

**Level.** Restricted closed-shell SCC-DFTB2 at the **Γ point**. Citation:
Elstner *et al.*, *Phys. Rev. B* **58**, 7260 (1998), DOI
[10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260). Force
module boundary: Hourahine *et al.*, *J. Chem. Phys.* **152**, 124101 (2020)
(DFTB+).

**Energy (closed shell).** With occupations \(n_i\) and eigenvalues
\(\varepsilon_i\) of the SCC Fock matrix in the nonorthogonal basis,

\[
E = \sum_i n_i\,\varepsilon_i
  - \tfrac12\sum_{AB}\Delta q_A\,\gamma_{AB}\,\Delta q_B
  + E_{\mathrm{rep}},
\]

where \(\Delta q_A\) are Mulliken charge fluctuations, \(\gamma_{AB}\) is the
screened Elstner Coulomb kernel (Hubbard \(U\) at short range,
\(1/R\) tail), and \(E_{\mathrm{rep}}\) is the short-range pair repulsive.
Double counting of the SCC piece follows the DFTB+ convention.

**Γ-point lattice sums.** There is **no k-mesh** on this page (multi-k energy
is `dftb2-kpoints`; multi-k forces remain ○). Matrices and real-space sums
walk lattice translations \(\mathbf{T}=n_0\mathbf{a}_1+n_1\mathbf{a}_2+n_2\mathbf{a}_3\)
with \(|\mathbf{T}|\) up to the maximum pair cutoff (element `r_cut`):

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

**Ionic forces (lattice-image-consistent, closed-form).** Implementation:
`analytic_forces_restricted` in `src/cpu/semiempirical/dftb2.cpp`. After one
SCC solve at the Γ density,

\[
\mathbf{F}_A = -\frac{\partial E}{\partial \mathbf{R}_A}
=
\mathbf{F}_A^{H_0}
+ \mathbf{F}_A^{\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{SCC}\,\gamma}
+ \mathbf{F}_A^{\mathrm{rep}},
\]

with **pairwise accumulation over every lattice image** of \(B\) within
\(r_\mathrm{cut}\) (and the symmetric \(A\leftarrow B+\mathbf{T}\) bookkeeping),
matching `build_h0_s` / `gamma_pair` / `repulsive_energy`:

| Term | Contraction (schematic, per image bond) |
| ---- | --------------------------------------- |
| Electronic \(H_0\) | \(\mathrm{Tr}\!\bigl(P\,\partial H_0/\partial R\bigr)\) |
| Pulay + shift on \(S\) | \(-\mathrm{Tr}\!\bigl(W\,\partial S/\partial R\bigr) + \mathrm{Tr}\!\bigl(P\,(\tfrac12(V_A+V_B)\,\partial S/\partial R)\bigr)\) with \(V_A=\sum_B\gamma_{AB}\Delta q_B\), \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| SCC \(\gamma\) (fixed \(\Delta q\)) | \(-\Delta q_A\Delta q_B\,(\partial\gamma_{AB}/\partial R)\) per image |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) from static pair form |

Self-images (\(A=A+\mathbf{T}\), \(\mathbf{T}\ne\mathbf{0}\)) contribute
**zero** nuclear force (geometry-independent \(|\mathbf{T}|\)).

**Why not min-image only.** Wave-2 status was: energy lattice-summed, but
`analytic_forces_restricted` contracted only min-image pairs — so
\(F\neq-\nabla E\) under engaged PBC. Wave-3 product walks the **same**
image list as the energy; the regression FD protocol recomputes a full re-SCC
lattice-summed \(E\) at \(\mathbf{R}\pm h\hat{e}\).

**Parameters (this twin).** `sk_set = MioSketch` (default): multi-element
H/C/N/O from in-tree `basis_data/dftb_mio_tables.h` (exponential sketches for
most pairs) plus mio-1-1 H–H radial grids in `dftb_mio_hh_sk.h`. **Not** a
runtime `.skf` reader; **not** bit-identical DFTB.org mio-1-1 for C–C.

**SCC / force settings (regression).**

| Knob | Value |
| ---- | ----- |
| System | Off-eq 1-D C dimer crystal (two C / cell) |
| Cell | Orthorhombic \(L_x=2.8\), \(L_y=L_z=12.0\) Å |
| Bond | \(r(\mathrm{C–C})=1.20\) Å \(\ne L_x/2\) so \(F_x\neq 0\) |
| Method | SCC-DFTB2, closed shell (`multiplicity = 1`, `charge = 0`) |
| SK set | `DftbSkSet::MioSketch` (default) |
| SCC | `scc_tol = 1\times 10^{-10}`, `max_scc = 50` |
| Forces | `with_forces = true` → closed-form lattice-image path |
| Dispersion | `d3 = false` |
| Units | \(E\) [eV], \(\mathbf{R}\) [Å], \(\mathbf{F}\) [eV/Å] |

**Geometry [Å].**

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| C | 6 | 0.00 | 0.00 | 0.00 |
| C | 6 | 1.20 | 0.00 | 0.00 |

With \(r=1.20\) Å and \(L_x=2.8\) Å, left/right image contacts sit at
1.20 Å and 1.60 Å (plus farther shells inside \(r_\mathrm{cut}\approx 4.5\) Å).
The equilibrium twin \(r=1.4\) of `dftb2-periodic-gamma` has vanishing force
by symmetry and is **not** used here.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E\) | Total electronic + repulsive energy on `kEnergy` | eV |
| \(\mathbf{F}_A\) | Analytic nuclear force on `kAtom_fx/y/z` | eV/Å |
| \(\sum_A\mathbf{F}_A\) | Net force (translation invariance) | eV/Å |

**Key exclusions (this slug).**

- Absolute \(E\)/\(F\) vs DFTB+ + DFTB.org **mio-1-1** — sketch SK; deepen via
  bit-identical C–C tables later (family oracle remains DFTB+).
- **k-mesh** forces beyond Γ → multi-k energy is `dftb2-kpoints`; multi-k
  analytic \(F\) remains ○.
- Analytic **cell stress** under PBC → sibling `dftb2-analytic-stress`
  (pair virial of the same image terms; separate product gate).
- **DFTB3** / third-order \(\gamma\) forces → separate method / queue.
- Collinear **open-shell** PBC forces — open-shell analytic path exists
  (`dftb2-spin-analytic-forces`) but this reg is restricted C-chain only.
- **Dispersion** (D3) under PBC → `dftb2-dispersion-d3` (molecular D3 product;
  periodic D3 lattice sum still ○).
- Live **DFTB+** / OpenMOPAC / xtb from ctest — **forbidden**; offline pins only.
- Using OpenMOPAC as a DFTB force reference — family lock forbids it.
- Treating GFN-xTB as a “DFTB3 successor” — forbidden framing.

## Implementation

Public surface only (`cpu::Frame` + concrete method members; **no** `*Config`
bag, **no** `make_*` / `build_*` factories). Headers:
`include/atomiverse/cpu/semiempirical/dftb2.h`,
`include/atomiverse/cpu/evaluate/evaluate.h`,
`include/atomiverse/cpu/frame/frame.h`. Implementation:
`src/cpu/semiempirical/dftb2.cpp` (`enumerate_lattice` /
`lattice_images_for` + `analytic_forces_restricted` image walk). Static tables:
`include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h`.

Minimal sequence (matches `reg_dftb2-periodic-forces.cpp`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

// Off-eq 1-D C dimer: r=1.20 Å, Lx=2.8 Å → images at 1.20 and 1.60 Å (etc.).
cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 6;
frame.span(kAtom_type)[1] = 6;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_x)[1] = 1.20;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_mass)[i] = 12.011;
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
        frame.span(kAtom_fz)[i] = 0;
}
frame.box = cpu::Box::orthorhombic(2.8, 12.0, 12.0);
frame.meta.get(kEnergy) = 0;

cpu::Evaluate eval;
cpu::semiempirical::DFTB2 dftb;  // sk_set = MioSketch (default)
dftb.with_forces = true;
dftb.scc_tol = 1e-10;
eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
eval.build();
eval.run(frame);

// Energy: frame.meta.get(kEnergy) [eV]
// Forces: kAtom_fx / fy / fz [eV/Å]
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
  cmake --build build -j"$(nproc)" --target atomiverse \
        test_cpu_dftb2 reg_dftb2-periodic-forces
  ctest --test-dir build -R "^(test_cpu_dftb2)$" --output-on-failure
  cmake --build build --target run_reg_dftb2-periodic-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/dftb2.h` | Public `DFTB2` / `DftbSkSet` |
| `include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h` | Static mio-style SK / Hubbard / rep |
| `src/cpu/semiempirical/dftb2.cpp` | Lattice enum + Γ H⁰/S + `analytic_forces_restricted` |
| `regressions/semiempirical/dftb2-periodic-forces/reg_dftb2-periodic-forces.cpp` | Product gate |
| `regressions/semiempirical/dftb2-periodic-forces/REFERENCE.md` | Offline gates + oracle line |
| `tools/tb_oracle/inputs/dftbplus/dftb2-periodic-forces/` | Offline DFTB+ geometry twin |
| `tests/cpu/semiempirical/test_dftb2.cpp` | Unit footholds for PBC / analytic F |

## Validation (DFTB+ / literature)

Source:
[`regressions/semiempirical/dftb2-periodic-forces/REFERENCE.md`](../../../regressions/semiempirical/dftb2-periodic-forces/REFERENCE.md).  
Oracle lines:

```text
oracle: literature
oracle_version: formula-level (Elstner 1998; Hourahine et al. JCP 152, 124101 (2020); ATV mio-style sketch)
family_oracle: dftb+
```

Hard-coded gates in the regression binary — **never** live DFTB+ from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (identity) | ATV SCC-DFTB2 MioSketch, Γ-PBC, closed shell |
| Oracle family | DFTB+ force structure (not OpenMOPAC, not xtb) |
| Offline twin | `tools/tb_oracle/inputs/dftbplus/dftb2-periodic-forces/` |
| Atomiverse path | `DFTB2` + `Evaluate`, `with_forces=true`, `scc_tol=1e-10` |
| FD protocol | Central re-SCC energy-only, \(h=10^{-4}\) Å, lattice-summed \(E\) |

**Hard-coded gates (regression):**

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E\) [eV] | finite, \(\|E\| > 10^{-3}\) | hard | Sketch SK — not absolute mio-1-1 DFTB+ \(E\) |
| \(\|F\|^2\) [eV²/Å²] | \(> 10^{-8}\) | hard | Off-eq geometry must produce force |
| \(\|\sum_i F_{i,d}\|\) [eV/Å] | \(\le 10^{-6}\) | hard | Translation invariance, each Cartesian |
| \(F_{0,x}\) vs re-SCC FD [eV/Å] | identity | **\(1\times 10^{-5}\)** | Lattice-image \(F=-\nabla E\) |
| \(F_{1,x}\) vs re-SCC FD [eV/Å] | identity | **\(1\times 10^{-5}\)** | Partner atom |
| \(\lvert E_{\mathrm{PBC}}-E_{\mathrm{mol}}\rvert\) [eV] | \(> 0.05\) | hard | Same dimer, 20 Å cubic must separate |

**Tolerance rationale.** Absolute DFTB+ + mio-1-1 forces are **not** the gate
(SK sketch for C–C). The product claim is: (1) closed-form forces walk the
**same** lattice images as \(E\); (2) net force vanishes; (3) selected
\(F_x\) components match re-SCC central FD of ATV’s own lattice-summed \(E\)
to \(10^{-5}\) eV/Å (tighter than the multi-element molecular H₂O residual
class in `dftb2-analytic-forces`, because fixed-\(\Delta q\) vs re-SCC charge
response is small on this monoelement chain). Offline DFTB+ input geometry
matches the twin for future absolute pins when bit-identical SK land.

Related absolute **DFTB+ 24.1** energy identity (different system: H₂
mio-1-1): `dftb2-dftbplus-energy-pin` — \(E=-18.36635052\) eV. Related
Γ-energy twin: `dftb2-periodic-gamma` (\(r=1.4\) Å equilibrium chain).

## Performance

Source: `REFERENCE.md`.

```text
perf_exempt: true
perf_exempt_reason: no offline DFTB+ binary on host; fair wall deferred
```

| Twin | \(T_{\mathrm{DFTB+}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| Γ-PBC C-chain ENERGY_FORCE | — | — | — | exempt | No DFTB+ Lmod module / fair bench; formula pin only |

Gate: fair \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a twin
exists. Analytic path replaces \(6N\) full re-SCC force evaluations with
**one** post-SCC contraction over lattice image bonds. A related molecular
energy pin (`dftb2-dftbplus-energy-pin`) reports informal walls ATV ~0.06 s
vs DFTB+ 24.1 ~0.15 s (\(r\approx 0.4\), ≤10×) — not this periodic force twin,
not ctest.

## Limitations

- Absolute \(E\)/\(F\) is an **ATV Hamiltonian twin**, not DFTB+ absolute digits
  for C–C MioSketch. Family oracle remains DFTB+ once bit-identical SK
  coverage lands.
- Real-space truncation of \(\gamma\) (no Ewald) is intentional for this
  SK-cutoff model; long-range electrostatics in dense 3-D ionic cells are
  outside the pin.
- **Fixed-\(\Delta q\)** force identity vs full re-SCC FD: monoelement chain
  residual is \(\sim 10^{-5}\) eV/Å; multi-element molecular residual remains
  larger (see `dftb2-analytic-forces` H₂O tol 1 eV/Å).
- Multi-k analytic forces / stress, DFTB3 analytic forces, periodic D3 lattice
  sum, and TD-DFTB remain roadmap ○.
- Elements limited to H/C/N/O static tables; unsupported \(Z\) throws.
- **SE-only cmake** for this loop — do not enable MD/MINIMIZER/HF/PW/GPW/CUDA
  for the TB product gate; do not build `reg_am1_water_bomd` /
  `reg_am1_water_minimize` under this configure.
- Family lock: **DFTB+ only** for DFTB rows — never OpenMOPAC as DFTB oracle;
  never treat GFN-xTB as DFTB3 successor.

## References

- Offline pin: [`regressions/semiempirical/dftb2-periodic-forces/REFERENCE.md`](../../../regressions/semiempirical/dftb2-periodic-forces/REFERENCE.md)
- Regression: [`reg_dftb2-periodic-forces.cpp`](../../../regressions/semiempirical/dftb2-periodic-forces/reg_dftb2-periodic-forces.cpp)
- Public API: [`include/atomiverse/cpu/semiempirical/dftb2.h`](../../../include/atomiverse/cpu/semiempirical/dftb2.h)
- Offline twin: [`tools/tb_oracle/inputs/dftbplus/dftb2-periodic-forces/`](../../../tools/tb_oracle/inputs/dftbplus/dftb2-periodic-forces/)
- Oracle tooling: [`tools/tb_oracle/README.md`](../../../tools/tb_oracle/README.md)
- Roadmap: [`docs/roadmap/semiempirical.md`](../../roadmap/semiempirical.md)
- Queue / handoff / loop: [`docs/developer/tb-method-queue.md`](../../developer/tb-method-queue.md),
  [`docs/developer/tb-method-handoff.md`](../../developer/tb-method-handoff.md),
  [`docs/developer/tb-method-loop.md`](../../developer/tb-method-loop.md),
  [`docs/developer/oracle-capability-loop.md`](../../developer/oracle-capability-loop.md)
- Spec history (closed, deleted at done): git `wf/dftb2-periodic-forces` /
  wave-3 harness (`.claude/specs/dftb2-periodic-forces/` removed at done)
- Literature: M. Elstner *et al.*, *Phys. Rev. B* **58**, 7260 (1998);
  B. Hourahine *et al.*, *J. Chem. Phys.* **152**, 124101 (2020) (DFTB+)
- Related: Γ energy `dftb2-periodic-gamma`; molecular analytic F
  `dftb2-analytic-forces`; analytic stress `dftb2-analytic-stress`; multi-k
  energy `dftb2-kpoints`; H₂ DFTB+ energy pin `dftb2-dftbplus-energy-pin`
