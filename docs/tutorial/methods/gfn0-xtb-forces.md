# GFN0-xTB analytic nuclear forces — off-eq H₂O

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn0-xtb-forces` |
| Regression | `regressions/semiempirical/gfn0-xtb-forces/` |
| Public API | `cpu::semiempirical::GFN0` (`with_forces = true`); `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; GFN-xTB is a sibling family) |
| Queue | TB method loop — wave 8 done (`e870f86d`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 GFN0-xTB energy ✓; family analytic gradient still ○ (GFN1/2 `with_forces` throw) |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Closed-form nuclear gradient of production **GFN0-xTB** (Pracht, Caldeweyher,
Ehlert, Grimme, ChemRxiv 2019) on off-equilibrium water, using Atomiverse’s
static H/C/N/O tables transcribed from pprcht/gfn0 `gfn0_param.f90`. After one
EEQ solve and one extended-Hückel diagonalization of the CN- and
charge-shifted Hamiltonian, forces are \(\mathbf{F}=-\nabla E\) at that
density — Hellmann–Feynman + Pulay on \(\partial S\), EEQ/CN chains through
the self-energy, pair-repulsion, two-body D4(BJ), and the polar-bond
short-range correction (SRB is identically zero on H₂O) — written to
`frame.span(kAtom_f*)` in eV/Å. This is the **xtb `--gfn 0 --grad` module
boundary** (pprcht `build_dSH0`, `chargeEquilibration`, `gfn0_repulsion`,
`gfn0_shortranged`), **not** a \(6N\) energy finite-difference and **not**
SCC-DFTB. Public surface: concrete method `cpu::semiempirical::GFN0` with
members (`charge`, `with_forces`); no public `*Config`, no `make_*` /
`build_*`. Product gate: `reg_gfn0-xtb-forces`. Energy identity already ships
on methanol (`reg_gfn0-beyond-toy`). Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.092\le 10\).

## Method

**GFN0-xTB energy** (Pracht et al., ChemRxiv 2019,
DOI 10.26434/chemrxiv.8326202.v1) — non-SCC, one EHT diagonalization:

\[
E = E_{\mathrm{H0}} + E_{\mathrm{rep}} + E_{\mathrm{IES}} + E_{\mathrm{D4}} + E_{\mathrm{SRB}}.
\]

| Term | Role |
| ---- | ---- |
| \(E_{\mathrm{H0}}\) | Occupied eigenvalues of a CN- and \(q\)-shifted extended-Hückel \(H\), \(S\) from STO-\(n\)G |
| \(E_{\mathrm{rep}}\) | Classical pair repulsion \(Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B\,e^{-\alpha r^{3/2}}/r\) |
| \(E_{\mathrm{IES}}\) | Electronegativity-equilibration (EEQ) electrostatics with Gaussian-smeared Coulomb |
| \(E_{\mathrm{D4}}\) | Two-body DFT-D4(BJ); \(s_6=1\), \(s_8=2.85\), \(a_1=0.80\), \(a_2=4.60\), \(s_9=0\) |
| \(E_{\mathrm{SRB}}\) | Polar heteronuclear short-range bond correction; **both** atoms must have \(Z\in[5,9]\) |

Parameter path: static tables → erf coordination numbers → EEQ charges →
shifted atomic levels \(\varepsilon_{A\ell}\) → \(H/S\) → generalized eigen →
band energy + classical terms. In ATV this is
`Gfn0Element → CN → EEQ → self-energy → EHT → repulsion/D4/SRB → E/F`
with parameters as members of `GFN0` and tables in
`basis_data/gfn0_tables.h`. Unsupported \(Z\) throws.

**Coordination number** (Pyykkö radii, \(k_{\mathrm{CN}}=7.5\), cut at
\(\mathrm{CN}_{\max}=8\)):

\[
\mathrm{CN}_A^{\mathrm{raw}}
  = \sum_{B}\tfrac12\bigl(1+\mathrm{erf}[-k_{\mathrm{CN}}(r_{AB}-r_0)/r_0]\bigr),\quad
r_0=\tfrac43(R_A^{\mathrm{cov}}+R_B^{\mathrm{cov}}),
\]

then \(\mathrm{CN}_A=\ln(1+e^{\mathrm{CN}_{\max}})-\ln(1+e^{\mathrm{CN}_{\max}-\mathrm{CN}^{\mathrm{raw}}_A})\).

**Shell self-energy** (feeds the EHT diagonal):

\[
\varepsilon_{A\ell}
  = \varepsilon^0_{A\ell}
  - k_{\mathrm{CN},\ell}\,\mathrm{CN}_A
  - k_{q,\ell}\,q_A
  - k_{q,\mathrm{atom}}\,q_A^2.
\]

**Ionic forces (closed-form)** — `energy_gfn0(..., want_grad=true)` in
`src/cpu/semiempirical/gfn0.cpp`. xtb prints \(\nabla E\) [Eh/\(a_0\)]; ATV
stores \(\mathbf{F}=-\nabla E\) [eV/Å]:

\[
\mathbf{F}_A = -\frac{\partial E}{\partial\mathbf{R}_A}
=
\mathbf{F}_A^{\mathrm{H0}}
+ \mathbf{F}_A^{\mathrm{IES}}
+ \mathbf{F}_A^{\mathrm{rep}}
+ \mathbf{F}_A^{\mathrm{D4}}
+ \mathbf{F}_A^{\mathrm{SRB}},
\]

with pairwise accumulation (antisymmetrized bond contributions) plus CN/\(q\)
chains:

| Term | Contraction (schematic) |
| ---- | ----------------------- |
| Electronic \(H_0\) + Pulay | \(2\,\mathrm{Tr}\!\bigl((P\odot H_0-W)\,\partial S/\partial R\bigr)\) plus shell-polynomial \(\partial\Pi/\partial R\); \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| Self-energy chain | \(\sum_A (\partial E_{\mathrm{H0}}/\partial\varepsilon_A)\,(\partial\varepsilon_A/\partial\mathrm{CN}_A\,\partial\mathrm{CN}_A/\partial R + \partial\varepsilon_A/\partial q_A\,\partial q_A/\partial R)\) |
| EEQ / IES | \(\partial(q^\top J q/2 - q^\top x)/\partial R\) from \(\partial J/\partial R\) and \(\partial\chi/\partial\mathrm{CN}\) |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) of the \(e^{-\alpha r^{3/2}}/r\) pair |
| D4 | Analytic two-body D4(BJ) forces (`cpu::scf::dftd4`) |
| SRB | Pair \(e^{-\mathrm{pre}\,(r-r_0(\mathrm{CN}))^2}\) plus \(\partial r_0/\partial\mathrm{CN}\) |

The \(6N\) central-FD product path is **deleted**. The regression still
compares two selected components against a **central energy-only** FD of ATV’s
own \(E\) (\(h=10^{-4}\) Å) as an identity gate, not as the production force.

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, **off-equilibrium** so \(\|F\|\neq 0\) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | GFN0-xTB, closed shell, `charge = 0` (even valence-electron count) |
| Tables | Static H/C/N/O from pprcht/gfn0 `gfn0_param.f90` (xtb 6.7.1) |
| Forces | `with_forces = true` → closed-form path (not re-EHT FD) |
| Accuracy | xtb `--acc 1.0` (default GFN0) |
| Units | \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å] |
| Conversion | \(a_0=0.52917726\) Å (GFN0 table); \(1\,\mathrm{Ha}=27.211386245988\) eV |
| Observable | Finite \(E\); net \(\sum_A\mathbf{F}_A=0\); analytic vs xtb \(\nabla E\); analytic vs energy FD |

**Geometry [Å]** (`tools/tb_oracle/inputs/xtb/h2o_gfn0_offeq_grad/mol.xyz`):

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H1 | 1 | 0.96 | 0.00 | 0.00 |
| H2 | 1 | −0.24 | 0.93 | 0.00 |

**Key exclusions (this page / acceptance):**

- Methanol **energy** vs xtb `--gfn 0` — sibling `gfn0-beyond-toy` (that pin
  covers C–O SRB; water has no SRB because H is outside \(Z\in[5,9]\)).
- **GFN1 / GFN2** forces — siblings `gfn1-xtb-forces`, `gfn2-xtb-forces`.
  Family analytic-gradient row stays ○ only for remaining GFN deepen
  (AES-in-Fock, solvent, PBC).
- GFN0 **FD Hessian** — sibling `gfn0-xtb-analytic-hessian`. Family GFN1/2
  Hessian is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md).
  IPEA energy is [`ipea-xtb-small`](ipea-xtb-small.md). PBC, spin, sTDA,
  ALPB/GBSA, CPCM-X — other queue rows. GFN-FF is classical (scope-out of
  this loop).
- Beyond-CHNO tables — unsupported \(Z\) throws; deepen via
  `gfn2-beyond-chno`.
- Live xtb / DFTB+ / OpenMOPAC from ctest — **forbidden**; offline pins only.
- Using DFTB+ or OpenMOPAC as a GFN0 force reference — family lock forbids it.
- Treating GFN0 as a DFTB3 successor — **forbidden**.

## Implementation

Public surface only (`cpu::Frame`, `cpu::semiempirical::GFN0` members; no
public `*Config`, no `make_*` / `build_*`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn0.h"

using namespace ATV;

cpu::Frame frame(3);
// Off-eq water (Å) so |F| ≠ 0 — same xyz as the xtb twin
frame.span(kAtom_type)[0] = 8;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_type)[1] = 1;
frame.span(kAtom_x)[1] = 0.96;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.0;
frame.span(kAtom_type)[2] = 1;
frame.span(kAtom_x)[2] = -0.24;
frame.span(kAtom_y)[2] = 0.93;
frame.span(kAtom_z)[2] = 0.0;
for (int i = 0; i < 3; ++i) {
    frame.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
        frame.span(kAtom_fz)[i] = 0;
}
frame.box = cpu::Box::cubic(20.0);
frame.meta.get(kEnergy) = 0;

cpu::semiempirical::GFN0 gfn0;
gfn0.charge = 0;
gfn0.with_forces = true;

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add(std::move(gfn0));
eval.build();
eval.run(frame);

// Energy: frame.meta.get(kEnergy) [eV]
// Forces: kAtom_fx / fy / fz [eV/Å]
```

SE-only build and regression (binding for this loop — never bare
`cmake --build build`):

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
        -DATV_BUILD_VIB=OFF -DATV_BUILD_PATH=OFF -DATV_BUILD_POTENTIALS=OFF \
        -DATV_BUILD_NEIGHBOR=OFF -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF \
        -DATV_BUILD_GPW=OFF -DATV_BUILD_KSPACE=OFF -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse test_cpu_xtb_gfn0 \
        reg_gfn0-xtb-forces
  ctest --test-dir build -R "^(test_cpu_xtb_gfn0)$" --output-on-failure
  cmake --build build --target run_reg_gfn0-xtb-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/gfn0.h` | Public `GFN0` (`charge`, `with_forces`) |
| `include/atomiverse/cpu/semiempirical/basis_data/gfn0_tables.h` | Static H/C/N/O GFN0 rows + global scales |
| `src/cpu/semiempirical/gfn0.cpp` | EEQ + EHT + analytic \(\nabla E\) (HF+Pulay, CN/\(q\), rep, D4, SRB) |
| `regressions/semiempirical/gfn0-xtb-forces/reg_gfn0-xtb-forces.cpp` | Product gate |
| `regressions/semiempirical/gfn0-xtb-forces/REFERENCE.md` | Offline pins + oracle line |
| `tools/tb_oracle/inputs/xtb/h2o_gfn0_offeq_grad/` | xtb 6.7.1 twin (`mol.xyz`, `xtb.out.txt`, `gradient.txt`) |
| `tests/cpu/semiempirical/test_xtb_gfn0.cpp` | Unit analytic-vs-FD + net-force invariance |

## Validation (xtb)

Source:
[`regressions/semiempirical/gfn0-xtb-forces/REFERENCE.md`](../../../regressions/semiempirical/gfn0-xtb-forces/REFERENCE.md).  
Oracle: **`xtb`** `--gfn 0 --grad` (Pracht et al., ChemRxiv 2019; Bannwarth et
al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021)).  
`oracle_version`: **6.7.1**.  
Hard-coded gates in the regression binary — **never** live xtb from ctest.

```text
oracle: xtb
oracle_version: 6.7.1
```

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (identity) | ATV GFN0-xTB closed shell, H/C/N/O tables |
| Oracle family | xtb (not OpenMOPAC, not DFTB+) |
| Offline twin | `tools/tb_oracle/inputs/xtb/h2o_gfn0_offeq_grad/` |
| Command | `xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0 --grad` (`OMP_NUM_THREADS=1`) |
| Atomiverse path | `GFN0` + `Evaluate`, `with_forces=true` |
| FD protocol | Central energy-only, \(h=10^{-4}\) Å (identity, not the product force) |
| Conversion | \(F=-\nabla E\times(27.211386245988\,\mathrm{eV/Eh})/0.52917726\,\mathrm{Å}\) |

**xtb \(\nabla E\) [Eh/\(a_0\)]** (`gradient.txt`):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O | −3.4755123224283×10⁻⁴ | −9.4449534348479×10⁻⁴ | ~0 |
| H1 | −3.8500960780473×10⁻³ | +3.4420757341719×10⁻³ | ~0 |
| H2 | +4.1976473102901×10⁻³ | −2.4975803906871×10⁻³ | ~0 |

xtb also printed TOTAL ENERGY **−4.366816412151 Eh** and
\(\|\nabla E\|=0.007179285426\) Eh/\(a_0\).

**Hard-coded gates (regression):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−4.366816412151 Eh** | \(1\times10^{-3}\) eV | family TB energy floor |
| \(F_{O,x}\) | `kAtom_fx[0]` | **+0.017871801258 eV/Å** | \(1\times10^{-4}\) eV/Å | \(-\nabla E\) converted |
| \(F_{H1,y}\) | `kAtom_fy[1]` | **−0.176998634239 eV/Å** | \(1\times10^{-4}\) eV/Å | dual-component product gate |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-10}\) eV/Å | translation invariance, each Cartesian |
| \(F_{O,x}\), \(F_{H1,y}\) vs energy FD | identity | — | \(3\times10^{-5}\) eV/Å | \(h=10^{-4}\) Å; unit test same |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=3.4\times10^{-8}\) eV,
\(\lvert F_{O,x}-F_{\mathrm{pin}}\rvert=2.6\times10^{-6}\) eV/Å,
analytic vs energy FD \(6.5\times10^{-7}\) eV/Å.

**Tolerance rationale.** Energy floor is the family \(10^{-3}\) eV TB pin.
Force tol \(10^{-4}\) eV/Å sits well above xtb’s printed \(10^{-16}\) Eh/\(a_0\)
noise on \(F_z\). Water has no SRB term; methanol SRB + C–O is covered by
`reg_gfn0-beyond-toy` plus the unit FD identity on this geometry
(EHT + EEQ + repulsion + D4).

## Performance

Source: `REFERENCE.md`. Fair wall asserted in `reg_gfn0-xtb-forces`
(\(r\le 10\)). Analytic path is one EEQ + one diagonalization (replaces 18
energy solves of a \(6N\) FD).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN0 `--gfn 0 --grad` | **0.007** (1 thread) | **6.45×10⁻⁴** | **≈ 0.092** | ✓ | xtb cold wall from archived `xtb.out.txt`; ATV tables are static |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a twin exists.

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** Odd / non-integer valence count throws.
- **Water has no SRB.** Polar-bond SRB is gated on methanol energy
  (`gfn0-beyond-toy`) plus the FD identity here; no absolute xtb SRB force
  pin on this geometry.
- **GFN1 / GFN2 forces still throw.** Family “GFN-xTB analytic gradient”
  stays ○ until those slugs land. This page closes GFN0 \(F=-\nabla E\) only.
- **No Hessian / PBC / spin / solvent / IPEA** on this page — see Phase C
  queue. GFN-FF is classical (never pending on this loop).
- **SE-only cmake** for this loop — do not enable MD/MINIMIZER/HF/PW/GPW/CUDA
  for the TB product gate; do not build `reg_am1_water_bomd` /
  `reg_am1_water_minimize` under this configure.
- Family lock: **xtb only** for GFN rows — never DFTB+ or OpenMOPAC as a
  GFN0 oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: GFN0 energy + GFN0 \(G\) is not the GFN product closed.

## References

- Offline pin: [`regressions/semiempirical/gfn0-xtb-forces/REFERENCE.md`](../../../regressions/semiempirical/gfn0-xtb-forces/REFERENCE.md)
- Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn0_offeq_grad/` (`mol.xyz`,
  `xtb.out.txt`, `gradient.txt`)
- Energy sibling: `regressions/semiempirical/gfn0-beyond-toy/REFERENCE.md`
- Oracle README: `tools/tb_oracle/README.md`
- Queue / handoff / loop: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`,
  `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Spec history (closed, deleted at done): git `wf/gfn0-xtb-forces`
- P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme, ChemRxiv (2019),
  DOI 10.26434/chemrxiv.8326202.v1
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
