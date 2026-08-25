# GFN0-xTB FD Hessian — off-eq H₂O vs xtb `--hess`

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn0-xtb-analytic-hessian` |
| Regression | `regressions/semiempirical/gfn0-xtb-analytic-hessian/` |
| Public API | `cpu::semiempirical::GFN0` (`with_forces = true`) on `cpu::vib::Hessian` at `vib::Phase::force` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface); [Ch. 14 Vibrations](../ch14-vibrations.md) (`cpu::vib::Hessian` FD driver) |
| Queue | TB method loop — wave 9 done (`d4582d31` / `30a6be4c`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 GFN0 energy + analytic \(G\) + **FD Hessian** ✓; family GFN1/2 is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md) |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Nuclear Hessian of production **GFN0-xTB** (Pracht, Caldeweyher, Ehlert,
Grimme, ChemRxiv 2019) on off-equilibrium water, assembled as a **\(6N\)
central finite difference of the already-shipped analytic gradient**
(\(h=10^{-3}\) Å) through `cpu::vib::Hessian` with
`cpu::semiempirical::GFN0` (`with_forces = true`) on `vib::Phase::force`.
This is the **xtb `--gfn 0 --hess` module boundary** (analytic \(G\),
numerical \(H\); xtb step \(0.005\,a_0\)), **not** a closed-form
\(\partial^2 E/\partial R_i\partial R_j\) kernel and **not** SCC-DFTB.
After mass-weighting (IUPAC-2013 from \(Z\)) and TR projection the three
vibrational wavenumbers are written on `HessianResult::frequencies` [cm⁻¹];
the Cartesian matrix is `HessianResult::hessian` [eV/Å²]. Public surface:
concrete method `GFN0` plus the vibration Driver preset; no public
`*Config`, no `make_*` / `build_*`. Product gate:
`reg_gfn0-xtb-analytic-hessian`. Energy + analytic \(G\) already ship
(`reg_gfn0-beyond-toy`, `reg_gfn0-xtb-forces`). Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.06\le 10\).

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
`basis_data/gfn0_tables.h`. Unsupported \(Z\) throws. Water has **no SRB**
(H is outside \(Z\in[5,9]\)).

**Analytic nuclear forces** already ship (`gfn0-xtb-forces`):
\(\mathbf{F}=-\nabla E\) at the EEQ + EHT density — Hellmann–Feynman + Pulay
on \(\partial S\), EEQ/CN chains through the self-energy, pair-repulsion,
two-body D4(BJ), and SRB. Written to `frame.span(kAtom_f*)` in eV/Å.

**Hessian (this page).** Central FD of those analytic forces, then
symmetrized. With \(g=-F\) and \(h=10^{-3}\) Å
(`cpu::vib::kHessianStepAngstrom`):

\[
H_{ij}
  \approx
  \frac{g_i(+h\,e_j)-g_i(-h\,e_j)}{2h},
  \qquad
  H\leftarrow\tfrac12(H+H^\top).
\]

For \(N=3\) this is a fixed **\(6N=18\)** force budget plus one finalize
pass (restore \(R_0\), mass-weight, TR-project, diagonalize). The driver
is `cpu::vib::Hessian` (`Driver<vib::Phase>`); stages
`setup → displace → force → accumulate → diagonalize → thermochem →
end_of_step`. Masses for \(\omega\) come from IUPAC-2013 `mass_amu(Z)`
(`kAtom_type`), **not** `kAtom_mass`. xtb `--hess` is the same algorithm
class (analytic \(G\), numerical \(H\)) at step \(0.005\,a_0\approx 0.00265\) Å.

**Not included.** Closed-form \(\partial^2 E/\partial R_i\partial R_j\)
(CPHF / GFN0 second-derivative kernel). GFN1/GFN2 Hessian
(`gfn-xtb-analytic-hessian`). IR intensities from an APT (FD driver does
not solve CPHF). PBC / spin / solvent / IPEA. GFN-FF (classical).

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, **off-equilibrium** (same xyz as `reg_gfn0-xtb-forces`) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | GFN0-xTB, closed shell, `charge = 0` |
| Tables | Static H/C/N/O from pprcht/gfn0 `gfn0_param.f90` (xtb 6.7.1) |
| Forces | `with_forces = true` → closed-form \(G\) at each displacement |
| Hessian | `cpu::vib::Hessian`, \(h=10^{-3}\) Å central FD of \(G\) |
| Accuracy | xtb `--acc 1.0` (default GFN0); xtb Hessian step \(0.005\,a_0\) |
| Units | \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å]; \(H\) [eV/Å²]; \(\omega\) [cm⁻¹] |
| Conversion | \(a_0=0.52917726\) Å (GFN0 table); \(1\,\mathrm{Ha}=27.211386245988\) eV |
| Thermo ctor | \(T=298.15\) K, \(P=101325\) Pa, multiplicity 1, \(\sigma=1\); empty recorder paths |

**Geometry [Å]** (`tools/tb_oracle/inputs/xtb/h2o_gfn0_hess/mol.xyz`):

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H1 | 1 | 0.96 | 0.00 | 0.00 |
| H2 | 1 | −0.24 | 0.93 | 0.00 |

xtb reports “incompletely optimized geometry” (RMS \(\|\nabla E\|=0.00718\)
Eh/\(a_0\)); all three vibrational eigenvalues are real.

**Key exclusions (this page / acceptance):**

- Closed-form GFN0 Hessian kernel — not shipped; this page is FD of analytic
  \(G\), matching xtb `--hess`.
- **GFN1 / GFN2 Hessian** — sibling [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md).
- GFN0 **energy** (methanol) — sibling `gfn0-beyond-toy`.
- GFN0 **analytic \(G\)** (this geometry) — sibling `gfn0-xtb-forces`.
- IPEA, PBC, spin, sTDA, ALPB/GBSA, CPCM-X — other queue rows. GFN-FF is
  classical (scope-out of this loop).
- Beyond-CHNO tables — unsupported \(Z\) throws; deepen via
  `gfn2-beyond-chno`.
- Live xtb / DFTB+ / OpenMOPAC from ctest — **forbidden**; offline pins only.
- Using DFTB+ or OpenMOPAC as a GFN0 Hessian reference — family lock forbids
  it.
- Treating GFN0 as a DFTB3 successor — **forbidden**.

## Implementation

Public surface only (`cpu::Frame`, `cpu::semiempirical::GFN0` members,
`cpu::vib::Hessian` ctor args; no public `*Config`, no `make_*` /
`build_*`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn0.h"
#include "atomiverse/cpu/vibration/hessian.h"

using namespace ATV;

cpu::Frame frame(3);
// Off-eq water (Å) — same xyz as the xtb `--hess` twin
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

// Empty recorder paths: do not write frequencies.tsv / mode_*.xyz.
cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
hess.on(vib::Phase::force).add(std::move(gfn0));
hess.build();
const cpu::vib::HessianResult res = hess.run(frame);
// res.hessian: [9,9] dense, eV/Å²; res.frequencies: 3 modes, cm⁻¹
// Frame geometry is restored to R0; leftover kEnergy is the last displacement.
```

SE-only build **plus the VIB exception** for this slug (never bare
`cmake --build build`; never HF/MD/CUDA):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON \
        -DATV_BUILD_VIB=ON \
        -DATV_BUILD_MD=OFF -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF \
        -DATV_BUILD_PATH=OFF -DATV_BUILD_POTENTIALS=OFF \
        -DATV_BUILD_NEIGHBOR=OFF -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF \
        -DATV_BUILD_GPW=OFF -DATV_BUILD_KSPACE=OFF -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse test_cpu_xtb_gfn0_hessian \
        reg_gfn0-xtb-analytic-hessian
  ctest --test-dir build -R "^(test_cpu_xtb_gfn0_hessian)$" --output-on-failure
  cmake --build build --target run_reg_gfn0-xtb-analytic-hessian
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/gfn0.h` | Public `GFN0` (`charge`, `with_forces`) |
| `include/atomiverse/cpu/semiempirical/basis_data/gfn0_tables.h` | Static H/C/N/O GFN0 rows + \(a_0=0.52917726\) Å |
| `src/cpu/semiempirical/gfn0.cpp` | EEQ + EHT + analytic \(\nabla E\) |
| `include/atomiverse/cpu/vibration/hessian.h` | `cpu::vib::Hessian` FD preset |
| `include/atomiverse/cpu/vibration/hessian_result.h` | `HessianResult` (\(H\), \(\omega\), \(n_{\mathrm{imag}}\)) |
| `include/atomiverse/cpu/vibration/hessian_workspace.h` | \(h=10^{-3}\) Å (`kHessianStepAngstrom`) |
| `regressions/semiempirical/gfn0-xtb-analytic-hessian/reg_gfn0-xtb-analytic-hessian.cpp` | Product gate |
| `regressions/semiempirical/gfn0-xtb-analytic-hessian/REFERENCE.md` | Offline pins + oracle line |
| `tools/tb_oracle/inputs/xtb/h2o_gfn0_hess/` | xtb 6.7.1 twin (`mol.xyz`, `xtb.out.txt`, `hessian.txt`, `vibspectrum.txt`) |
| `tests/cpu/semiempirical/test_xtb_gfn0_hessian.cpp` | Unit: `Hessian::build` accepts GFN0 force kind (no `run`) |

## Validation (xtb)

Source:
[`regressions/semiempirical/gfn0-xtb-analytic-hessian/REFERENCE.md`](../../../regressions/semiempirical/gfn0-xtb-analytic-hessian/REFERENCE.md).  
Oracle: **`xtb`** `--gfn 0 --hess` (Pracht et al., ChemRxiv 2019; Bannwarth et
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
| Engine (identity) | ATV GFN0-xTB closed shell, H/C/N/O tables + `cpu::vib::Hessian` |
| Oracle family | xtb (not OpenMOPAC, not DFTB+) |
| Offline twin | `tools/tb_oracle/inputs/xtb/h2o_gfn0_hess/` |
| Command | `xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0 --hess` (`OMP_NUM_THREADS=1`) |
| Atomiverse path | `GFN0` on `Hessian` at `vib::Phase::force`, `with_forces=true` |
| FD protocol | Central of analytic \(G\), \(h=10^{-3}\) Å (xtb: \(0.005\,a_0\)) |
| Conversion | \(H[\mathrm{eV}/\mathrm{Å}^2]=H[\mathrm{Eh}/a_0^2]\times 27.211386245988/(0.52917726)^2\) |

**xtb projected \(\omega\) [cm⁻¹]** (`vibspectrum.txt` modes 7–9; six TR
zeros projected out):

| Mode | Assignment | xtb 6.7.1 |
| ---- | ---------- | --------: |
| \(\omega_1\) | bend | **1563.14** |
| \(\omega_2\) | symmetric stretch | **3497.87** |
| \(\omega_3\) | antisymmetric stretch | **3505.03** |

xtb also printed TOTAL ENERGY **−4.366816412151 Eh** and Cartesian
\(H_{O_x,O_x}=0.4793058695\) Eh/\(a_0^2\) (`hessian.txt`). IR intensities
in the twin (17.01 / 18.60 / 16.02 km mol⁻¹) are **not** a product gate
(ATV FD Hessian has no APT).

**Hard-coded gates (regression):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) | `kEnergy` (fresh Evaluate after restore) | **−4.366816412151 Eh** | \(1\times10^{-3}\) eV | family TB energy floor |
| \(\omega_1\) (bend) | `res.frequencies(0)` | **1563.14 cm⁻¹** | 10 cm⁻¹ | projected, ascending |
| \(\omega_2\) (sym stretch) | `res.frequencies(1)` | **3497.87 cm⁻¹** | 10 cm⁻¹ | |
| \(\omega_3\) (asym stretch) | `res.frequencies(2)` | **3505.03 cm⁻¹** | 10 cm⁻¹ | |
| \(H_{O_x,O_x}\) | `res.hessian(0,0)` | **46.57588 eV/Å²** | 0.2 eV/Å² | from \(0.4793058695\) Eh/\(a_0^2\) |
| Hessian shape | identity | \(9\times 9\) | hard | finite, \(\lvert H-H^\top\rvert\le 10^{-8}\) eV/Å² |
| Rotor | identity | nonlinear | hard | `res.is_linear == false`; 3 modes |
| \(n_{\mathrm{imag}}\) | identity | 0 | hard | all three \(\omega>100\) cm⁻¹ |
| Geometry restore | identity | \(R_0\) | \(10^{-12}\) Å | `Hessian::run` restores displacements |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=3.4\times10^{-8}\) eV,
\(\Delta\omega = 0.05/2.73/1.87\) cm⁻¹,
\(\lvert H_{O_x,O_x}-H_{\mathrm{pin}}\rvert=3.8\times10^{-2}\) eV/Å².

**Tolerance rationale.** Energy floor is the family \(10^{-3}\) eV TB pin
(same geometry as `reg_gfn0-xtb-forces`). Frequency / Cartesian-\(H\) tols
absorb the FD-step mismatch (xtb \(0.005\,a_0 \approx 0.00265\) Å vs ATV
\(10^{-3}\) Å) plus IUPAC-2013 vs xtb isotope masses
(\(\sim 0.3\,\mathrm{cm}^{-1}\) on H stretches). The 10 cm⁻¹ / 0.2 eV/Å²
gates sit well above the measured residual and still catch a sign, unit, or
mass-weighting error on the stiff stretches (\(\sim 3500\) cm⁻¹).

## Performance

Source: `REFERENCE.md`. Fair wall asserted in
`reg_gfn0-xtb-analytic-hessian` (\(r\le 10\)). Same algorithm class: analytic
\(G\), numerical \(H\) (18 force evaluations + finalize vs xtb `--hess`).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN0 `--gfn 0 --hess` | **0.02** (1 thread, `/usr/bin/time`) | **\(\sim 1.2\times10^{-3}\)** | **≈ 0.06** | ✓ | xtb printed wall 0.007 s; regression pins the process wall 0.02 s; ATV tables are static |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a twin exists.

## Limitations

- **FD of analytic \(G\), not closed-form \(H\).** Cost is \(O(N_{\mathrm{atoms}})\)
  GFN0 force evaluations. A GFN0 second-derivative kernel is not on this page.
- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** Odd / non-integer valence count throws.
- **Off-eq geometry.** xtb warns “incompletely optimized”; frequencies are
  still all real. Do not cite these \(\omega\) as a GFN0 minimum of water.
- **Water has no SRB.** Polar-bond SRB is gated on methanol energy
  (`gfn0-beyond-toy`); this Hessian does not exercise \(\partial^2 E_{\mathrm{SRB}}\).
- **GFN1 / GFN2 Hessian.** Sibling [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md).
  This page closes GFN0 FD \(H\) only.
- **No PBC / spin / solvent / IR** on this page — see Phase C queue.
  IPEA energy is [`ipea-xtb-small`](ipea-xtb-small.md). GFN-FF is classical
  (never pending on this loop).
- **VIB exception.** This slug (and `se-analytic-hessian` /
  `gfn-xtb-analytic-hessian`) may set `-DATV_BUILD_VIB=ON`. Do not enable
  MD/MINIMIZER/HF/PW/GPW/CUDA for the TB product gate; do not build
  `reg_am1_water_bomd` / `reg_am1_water_minimize` under this configure.
- Family lock: **xtb only** for GFN rows — never DFTB+ or OpenMOPAC as a
  GFN0 oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: GFN0 energy + \(G\) + FD \(H\) is not the GFN product
  closed.

## References

- Offline pin: [`regressions/semiempirical/gfn0-xtb-analytic-hessian/REFERENCE.md`](../../../regressions/semiempirical/gfn0-xtb-analytic-hessian/REFERENCE.md)
- Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn0_hess/` (`mol.xyz`,
  `xtb.out.txt`, `hessian.txt`, `vibspectrum.txt`)
- Force sibling: [`gfn0-xtb-forces.md`](gfn0-xtb-forces.md) /
  `regressions/semiempirical/gfn0-xtb-forces/REFERENCE.md`
- Energy sibling: `regressions/semiempirical/gfn0-beyond-toy/REFERENCE.md`
- SE Hessian foothold (DFTB2/H₂, structural): `regressions/semiempirical/se-analytic-hessian/REFERENCE.md`
- Oracle README: `tools/tb_oracle/README.md`
- Queue / handoff / loop: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`,
  `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Spec history (closed, deleted at done): git `wf/gfn0-xtb-analytic-hessian`
- P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme, ChemRxiv (2019),
  DOI 10.26434/chemrxiv.8326202.v1
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
