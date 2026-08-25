# GFN1/GFN2-xTB FD Hessian — off-eq H₂O vs xtb `--hess`

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn-xtb-analytic-hessian` |
| Regression | `regressions/semiempirical/gfn-xtb-analytic-hessian/` |
| Public API | `cpu::semiempirical::GFN1` / `GFN2` (`with_forces = true`) on `cpu::vib::Hessian` at `vib::Phase::force` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md); [Ch. 14 Vibrations](../ch14-vibrations.md) |
| Queue | TB method loop — wave 10 done (`46c26afa` / `55e8c762`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 GFN-xTB analytic Hessian ✓ (GFN0/1/2 FD \(H\)); AES-in-Fock landed on [`gfn2-aes-fock`](gfn2-aes-fock.md) (angular \(F\) / GFN2 \(\omega\) is CAMM-vs-\(R\)) |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Nuclear Hessian of production **GFN1-xTB** (primary) and **GFN2-xTB**
(family close) on off-equilibrium water, assembled as a **\(6N\)
central finite difference of the already-shipped analytic gradient**
(\(h=10^{-3}\) Å) through `cpu::vib::Hessian` with
`cpu::semiempirical::GFN1` / `GFN2` (`with_forces = true`) on
`vib::Phase::force`. This is the **xtb `--gfn 1 --hess` / `--gfn 2 --hess`
module boundary**. xtb 6.7.1 prints an **analytical** Hessian (log:
“analytical hessian”); ATV stays FD of analytic \(G\) (same shape as
[`gfn0-xtb-analytic-hessian`](gfn0-xtb-analytic-hessian.md) /
`se-analytic-hessian`), **not** a closed-form
\(\partial^2 E/\partial R_i\partial R_j\) kernel and **not** `libxtb`.
GFN0 FD Hessian already ships. Sibling [`gfn2-aes-fock`](gfn2-aes-fock.md)
puts GFN2 AES in the SCC Fock; the GFN2 \(\omega\) residual is the
**CAMM-vs-\(R\)** angular hole (not a missing Fock term).
After mass-weighting (IUPAC-2013 from \(Z\)) and TR projection the three
vibrational wavenumbers are written on `HessianResult::frequencies`
[cm⁻¹]; the Cartesian matrix is `HessianResult::hessian` [eV/Å²]. Public
surface: concrete methods plus the vibration Driver preset; no public
`*Config`, no `make_*` / `build_*`. Product gate:
`reg_gfn-xtb-analytic-hessian`. Energy + analytic \(G\) already ship
(`reg_gfn1-xtb-organic`, `reg_gfn1-xtb-forces`, `reg_gfn2-xtb-water`,
`reg_gfn2-xtb-forces`). Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.11\le 10\) (GFN1 primary).

## Method

**GFN1-xTB energy** (Grimme, Bannwarth, Shushkov, *J. Chem. Theory Comput.*
**13**, 1989 (2017), DOI
[10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)) —
SCC EHT already shipped by `reg_gfn1-xtb-organic` /
[`gfn1-xtb-forces`](gfn1-xtb-forces.md):

\[
E = \mathrm{Tr}(P H_0)
  + \tfrac12 q\,\gamma\,q
  + \tfrac13 \Gamma q^3
  + E_{\mathrm{rep}}
  + E_{\mathrm{D3}}.
\]

| Term | Role |
| ---- | ---- |
| \(E_{H^0}=\mathrm{Tr}(P H_0)\) | Occupied density on a CN-shifted EHT Hamiltonian; \(S\) from Stewart STO-\(n\)G |
| \(E_\gamma=\tfrac12 q\,\gamma\,q\) | Shell-resolved Klopman–Ohno SCC (\(g_{\exp}=2\), harmonic-mean hardness) |
| \(E_\Gamma=\tfrac13\sum_A\Gamma_A q_A^3\) | On-site third-order; frozen-\(q\) \(\partial E_\Gamma/\partial R=0\) |
| \(E_{\mathrm{rep}}\) | Classical pair \(Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B\,e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) |
| \(E_{\mathrm{D3}}\) | Molecular D3(BJ); GFN1 set \(s_6=1\), \(s_8=2.4\), \(a_1=0.63\), \(a_2=5.0\) (xtb \(s_9=0\)) |

Hydrogen is double-ζ (1s + orthogonalized 2s polarization). Analytic
\(\mathbf{F}=-\nabla E\) already ships (`gfn1-xtb-forces`): Hellmann–Feynman
+ Pulay, CN/shell-poly chains, \(\partial\gamma/\partial R\), pair
repulsion, D3(BJ). Parameters as members of `GFN1`; tables in
`basis_data/gfn1_tables.h` (tblite `gfn1.f90` / xtb `param_gfn1-xtb.txt`).
Unsupported \(Z\) throws.

**GFN2-xTB energy** (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.*
**15**, 1652 (2019), DOI
[10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)) —
already shipped by `reg_gfn2-xtb-water` /
[`gfn2-xtb-forces`](gfn2-xtb-forces.md):

\[
E = \operatorname{Tr}(P H^{0})
  + E_{\mathrm{IES}}
  + E^{(3)}
  + E_{\mathrm{rep}}
  + E_{\mathrm{D4}}
  + E_{\mathrm{AES}}.
\]

| Term | Role |
| ---- | ---- |
| \(\operatorname{Tr}(P H^{0})\) | Occupied eigenvalues of a CN-shifted extended-Hückel \(H\), \(S\) from STO-\(n\)G |
| \(E_{\mathrm{IES}}\) | Shell-resolved isotropic electrostatics, \(\gamma_{AB}\) with \(g_{\exp}=2\) |
| \(E^{(3)}\) | On-site cubic \(\frac13\sum_i \Gamma_i q_i^3\) |
| \(E_{\mathrm{rep}}\) | Screened \(Z_{\mathrm{eff}}\) pair repulsion |
| \(E_{\mathrm{D4}}\) | Grimme D4; GFN2 damping \(s_6=1\), \(s_8=2.7\), \(a_1=0.52\), \(a_2=5\), \(s_9=5\) |
| \(E_{\mathrm{AES}}\) | Anisotropic electrostatics + dipole/quadrupole AXC; in the SCC Fock as of [`gfn2-aes-fock`](gfn2-aes-fock.md) |

Analytic \(\mathbf{F}=-\nabla E\) already ships (`gfn2-xtb-forces`) at frozen
CAMM. AES-in-Fock is [`gfn2-aes-fock`](gfn2-aes-fock.md); the remaining
angular \(F\) / GFN2 \(\omega\) hole is CAMM-vs-\(R\). Parameters as
members of `GFN2`; tables in `basis_data/gfn2_xtb_tables.h`. Unsupported
\(Z\) throws.

GFN-xTB is **not** DFTB3++ — different Hamiltonian, tables, and oracle
(**xtb** only). **Not** `libxtb`.

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
(`kAtom_type`), **not** `kAtom_mass`. xtb 6.7.1 `--hess` on GFN1/GFN2 is
an **analytical** Hessian (different algorithm class from ATV’s FD of
\(G\)); the GFN0 twin is analytic \(G\), numerical \(H\) at step
\(0.005\,a_0\). The product gate is still \(\omega\) and one Cartesian
\(H\) element vs the xtb printout, with tols that absorb the path
mismatch.

**Not included.** Closed-form \(\partial^2 E/\partial R_i\partial R_j\)
(CPHF / GFN second-derivative kernel). IR intensities from an APT (FD
driver does not solve CPHF). PBC / spin / solvent. GFN-FF (classical).
IPEA energy / \(G\) / FD Hessian are [`ipea-xtb-small`](ipea-xtb-small.md) /
[`ipea-xtb-forces`](ipea-xtb-forces.md) /
[`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md). AES-in-Fock is
[`gfn2-aes-fock`](gfn2-aes-fock.md).

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, **off-equilibrium** (same xyz as `reg_gfn1-xtb-forces` / `reg_gfn0-xtb-analytic-hessian`) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| GFN1 | `--gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess` |
| GFN2 | `--gfn 2 --chrg 0 --uhf 0 --acc 0.01 --hess` (same 0.96 Å xyz; **not** the 1.10 Å stretch of `reg_gfn2-xtb-forces`) |
| Tables | Static H/C/N/O from tblite `gfn1.f90` / `gfn2.f90` (xtb 6.7.1) |
| Forces | `with_forces = true` → closed-form \(G\) at each displacement |
| Hessian | `cpu::vib::Hessian`, \(h=10^{-3}\) Å central FD of \(G\) |
| xtb Hessian path | **analytical** (log: “analytical hessian”) |
| Units | \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å]; \(H\) [eV/Å²]; \(\omega\) [cm⁻¹] |
| Conversion | \(a_0=0.529177210903\) Å (CODATA `units::kBohrInAngstrom`); \(1\,\mathrm{Ha}=27.211386245988\) eV |
| Thermo ctor | \(T=298.15\) K, \(P=101325\) Pa, multiplicity 1, \(\sigma=1\); empty recorder paths |

**Geometry [Å]** (`tools/tb_oracle/inputs/xtb/h2o_gfn1_hess/mol.xyz` and
`h2o_gfn2_hess/mol.xyz`):

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H1 | 1 | 0.96 | 0.00 | 0.00 |
| H2 | 1 | −0.24 | 0.93 | 0.00 |

xtb reports “incompletely optimized geometry” on both twins; all three
vibrational eigenvalues are real.

**Key exclusions (this page / acceptance):**

- Closed-form GFN1/GFN2 Hessian kernel — not shipped; this page is FD of
  analytic \(G\). xtb 6.7.1 `--hess` is analytical; the pin is still
  \(\omega\) / \(H_{O_x,O_x}\).
- **GFN0 Hessian** — sibling [`gfn0-xtb-analytic-hessian`](gfn0-xtb-analytic-hessian.md).
- GFN1 **energy** (methane) — sibling `gfn1-xtb-organic`.
- GFN1 **analytic \(G\)** (this geometry) — sibling `gfn1-xtb-forces`.
- GFN2 **energy** / quadrupole AES — siblings `gfn2-xtb-water`,
  `gfn2-quadrupole-aes`.
- GFN2 **analytic \(G\)** (1.10 Å stretch) — sibling `gfn2-xtb-forces`.
- AES-in-Fock — sibling [`gfn2-aes-fock`](gfn2-aes-fock.md). GFN2
  \(\omega\) tols on this page **absorb** the remaining CAMM-vs-\(R\)
  hole; they are not a close of that slope.
- IPEA energy / \(G\) / FD Hessian — [`ipea-xtb-small`](ipea-xtb-small.md) /
  [`ipea-xtb-forces`](ipea-xtb-forces.md) /
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md). PBC, spin,
  sTDA, ALPB/GBSA, CPCM-X — other queue rows. GFN-FF is classical
  (scope-out of this loop).
- Beyond-CHNO tables — unsupported \(Z\) throws; deepen via
  `gfn2-beyond-chno`.
- Live xtb / DFTB+ / OpenMOPAC from ctest — **forbidden**; offline pins only.
- Using DFTB+ or OpenMOPAC as a GFN Hessian reference — family lock forbids
  it.
- Treating GFN-xTB as a DFTB3 successor — **forbidden**.

## Implementation

Public surface only (`cpu::Frame`, `cpu::semiempirical::GFN1` / `GFN2`
members, `cpu::vib::Hessian` ctor args; no public `*Config`, no `make_*` /
`build_*`). Primary pin is GFN1:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn1.h"
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

cpu::semiempirical::GFN1 gfn1;
gfn1.charge = 0;
gfn1.with_forces = true;

// Empty recorder paths: do not write frequencies.tsv / mode_*.xyz.
cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
hess.on(vib::Phase::force).add(std::move(gfn1));
hess.build();
const cpu::vib::HessianResult res = hess.run(frame);
// res.hessian: [9,9] dense, eV/Å²; res.frequencies: 3 modes, cm⁻¹
// Frame geometry is restored to R0; leftover kEnergy is the last displacement.
```

GFN2 is the same shape with `cpu::semiempirical::GFN2` in place of `GFN1`.
The regression also re-evaluates energy after restore via `cpu::Evaluate`
(fresh method; leftover `kEnergy` from the last displacement is not the
pin).

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
        -DATV_BUILD_VIB=ON -DATV_BUILD_CLI=OFF \
        -DATV_BUILD_MD=OFF -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF \
        -DATV_BUILD_PATH=OFF -DATV_BUILD_POTENTIALS=OFF \
        -DATV_BUILD_NEIGHBOR=OFF -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF \
        -DATV_BUILD_GPW=OFF -DATV_BUILD_KSPACE=OFF -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse \
        test_cpu_xtb_gfn_hessian reg_gfn-xtb-analytic-hessian
  ctest --test-dir build -R "^test_cpu_xtb_gfn_hessian$" --output-on-failure
  cmake --build build --target run_reg_gfn-xtb-analytic-hessian
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/gfn1.h` | Public `GFN1` (`charge`, `scc_tol`, `with_forces`) |
| `include/atomiverse/cpu/semiempirical/gfn2.h` | Public `GFN2` (`charge`, `scc_tol`, `with_forces`) |
| `include/atomiverse/cpu/semiempirical/basis_data/gfn1_tables.h` | Static H/C/N/O GFN1 rows |
| `include/atomiverse/cpu/semiempirical/basis_data/gfn2_xtb_tables.h` | Static H/C/N/O GFN2 rows |
| `src/cpu/semiempirical/gfn1.cpp` | SCC + analytic \(\nabla E\) |
| `src/cpu/semiempirical/gfn2.cpp` | SCC + AES (post-SCC) + analytic \(\nabla E\) |
| `include/atomiverse/cpu/vibration/hessian.h` | `cpu::vib::Hessian` FD preset |
| `include/atomiverse/cpu/vibration/hessian_result.h` | `HessianResult` (\(H\), \(\omega\), \(n_{\mathrm{imag}}\)) |
| `include/atomiverse/cpu/vibration/hessian_workspace.h` | \(h=10^{-3}\) Å (`kHessianStepAngstrom`) |
| `regressions/semiempirical/gfn-xtb-analytic-hessian/reg_gfn-xtb-analytic-hessian.cpp` | Product gate |
| `regressions/semiempirical/gfn-xtb-analytic-hessian/REFERENCE.md` | Offline pins + oracle line |
| `tools/tb_oracle/inputs/xtb/h2o_gfn1_hess/` | xtb 6.7.1 GFN1 twin (`mol.xyz`, `xtb.out.txt`, `hessian.txt`, `vibspectrum.txt`) |
| `tools/tb_oracle/inputs/xtb/h2o_gfn2_hess/` | xtb 6.7.1 GFN2 twin (same files) |
| `tests/cpu/semiempirical/test_xtb_gfn_hessian.cpp` | Unit: `Hessian::build` accepts GFN1 and GFN2 force kinds (no `run`) |

## Validation (xtb)

Source:
[`regressions/semiempirical/gfn-xtb-analytic-hessian/REFERENCE.md`](../../../regressions/semiempirical/gfn-xtb-analytic-hessian/REFERENCE.md).  
Oracle: **`xtb`** `--gfn 1 --hess` / `--gfn 2 --hess` (Grimme et al., *J.
Chem. Theory Comput.* **13**, 1989 (2017); Bannwarth et al., *J. Chem.
Theory Comput.* **15**, 1652 (2019); Bannwarth et al., *WIREs Comput. Mol.
Sci.* **11**, e01493 (2021)).  
`oracle_version`: **6.7.1**.  
Hard-coded gates in the regression binary — **never** live xtb from ctest.

```text
oracle: xtb
oracle_version: 6.7.1
```

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (identity) | ATV GFN1/GFN2-xTB closed shell, H/C/N/O tables + `cpu::vib::Hessian` |
| Oracle family | xtb (not OpenMOPAC, not DFTB+) |
| Offline twins | `tools/tb_oracle/inputs/xtb/h2o_gfn1_hess/`, `h2o_gfn2_hess/` |
| Command | `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess` (`OMP_NUM_THREADS=1`); GFN2 replaces `--gfn 1` |
| Atomiverse path | `GFN1` / `GFN2` on `Hessian` at `vib::Phase::force`, `with_forces=true` |
| FD protocol | Central of analytic \(G\), \(h=10^{-3}\) Å (xtb: **analytical** Hessian) |
| Conversion | \(H[\mathrm{eV}/\mathrm{Å}^2]=H[\mathrm{Eh}/a_0^2]\times 27.211386245988/(0.529177210903)^2\) |

**xtb projected \(\omega\) [cm⁻¹]** (`vibspectrum.txt` modes 7–9; six TR
zeros projected out):

| Mode | Assignment | GFN1 xtb 6.7.1 | GFN2 xtb 6.7.1 |
| ---- | ---------- | -------------: | -------------: |
| \(\omega_1\) | bend | **1474.96** | **1565.10** |
| \(\omega_2\) | symmetric stretch | **3657.78** | **3634.95** |
| \(\omega_3\) | antisymmetric stretch | **3761.56** | **3643.62** |

xtb also printed TOTAL ENERGY **−5.768641651774 Eh** (GFN1) /
**−5.070383095219 Eh** (GFN2) and Cartesian \(H_{O_x,O_x}=0.5466593694\)
/ \(0.5210291876\) Eh/\(a_0^2\) (`hessian.txt`). IR intensities in the
twins (GFN1 233.62 / 45.82 / 45.96 km mol⁻¹; GFN2 131.85 / 12.89 / 7.38
km mol⁻¹) are **not** a product gate (ATV FD Hessian has no APT).

**Hard-coded gates (regression):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| GFN1 \(E\) | `kEnergy` (fresh Evaluate after restore) | **−5.768641651774 Eh** | \(5\times10^{-3}\) eV | family TB energy floor (`reg_gfn1-xtb-forces`) |
| GFN1 \(\omega_1\) (bend) | `res.frequencies(0)` | **1474.96 cm⁻¹** | 10 cm⁻¹ | projected, ascending; **primary pin** |
| GFN1 \(\omega_2\) (sym stretch) | `res.frequencies(1)` | **3657.78 cm⁻¹** | 10 cm⁻¹ | |
| GFN1 \(\omega_3\) (asym stretch) | `res.frequencies(2)` | **3761.56 cm⁻¹** | 10 cm⁻¹ | |
| GFN1 \(H_{O_x,O_x}\) | `res.hessian(0,0)` | **53.12087 eV/Å²** | 0.2 eV/Å² | from \(0.5466593694\) Eh/\(a_0^2\) |
| GFN2 \(E\) | `kEnergy` (fresh Evaluate after restore) | **−5.070383095219 Eh** | \(1\times10^{-3}\) eV | family TB energy floor (`reg_gfn2-xtb-water`) |
| GFN2 \(\omega_1\) (bend) | `res.frequencies(0)` | **1565.10 cm⁻¹** | 150 cm⁻¹ | CAMM-vs-\(R\) hole |
| GFN2 \(\omega_2\) (sym stretch) | `res.frequencies(1)` | **3634.95 cm⁻¹** | 150 cm⁻¹ | |
| GFN2 \(\omega_3\) (asym stretch) | `res.frequencies(2)` | **3643.62 cm⁻¹** | 150 cm⁻¹ | |
| GFN2 \(H_{O_x,O_x}\) | `res.hessian(0,0)` | **50.63029 eV/Å²** | 1.0 eV/Å² | from \(0.5210291876\) Eh/\(a_0^2\); AES hole |
| Hessian shape | identity | \(9\times 9\) | hard | finite, \(\lvert H-H^\top\rvert\le 10^{-8}\) eV/Å² |
| Rotor | identity | nonlinear | hard | `res.is_linear == false`; 3 modes |
| \(n_{\mathrm{imag}}\) | identity | 0 | hard | all three \(\omega>100\) cm⁻¹ |
| Geometry restore | identity | \(R_0\) | \(10^{-12}\) Å | `Hessian::run` restores displacements |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**

| | GFN1 | GFN2 |
| - | ---- | ---- |
| \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert\) | \(2.767\times10^{-3}\) eV | \(2.64\times10^{-5}\) eV |
| \(\Delta\omega\) | 0.38 / 0.40 / 5.48 cm⁻¹ | 121 / 89 / 19 cm⁻¹ |
| \(\lvert H_{O_x,O_x}-H_{\mathrm{pin}}\rvert\) | \(0.104\) eV/Å² | \(0.751\) eV/Å² |

**Tolerance rationale.** Energy floors match the family TB pins
(`reg_gfn1-xtb-forces` 5 meV; `reg_gfn2-xtb-water` 1 meV). GFN1 frequency
/ Cartesian-\(H\) tols absorb FD-of-\(G\) vs xtb **analytical** Hessian
plus IUPAC-2013 vs xtb isotope masses (the GFN1 angular-\(F\) kernel hole
does **not** move these \(\omega\); measured residual sits well under
10 cm⁻¹ / 0.2 eV/Å²). GFN2 \(\omega\) / \(H\) tols are the stated
**CAMM-vs-\(R\) hole** (AES-in-Fock is [`gfn2-aes-fock`](gfn2-aes-fock.md);
the water bend is the angular mode, \(\Delta\omega_1=121\) cm⁻¹).
Structural gates: 9×9 finite symmetric Hessian, nonlinear rotor, 3 real
modes, \(n_{\mathrm{imag}}=0\).

## Performance

Source: `REFERENCE.md`. Fair wall asserted in
`reg_gfn-xtb-analytic-hessian` (\(r\le 10\)). Same *product* (projected
\(\omega\) of H₂O); **not** the same Hessian algorithm class (ATV:
analytic \(G\), numerical \(H\), 18 force evaluations + finalize; xtb
6.7.1: analytical Hessian). Primary ratio is GFN1.

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN1 `--gfn 1 --hess` | **0.03** (1 thread, `/usr/bin/time`) | **\(\sim 3.2\times10^{-3}\)** | **≈ 0.11** | ✓ | xtb printed wall 0.020 s; regression pins the process wall 0.03 s; ATV tables are static; xtb Hessian is analytical |
| H₂O GFN2 `--gfn 2 --hess` | **0.02** (1 thread, `/usr/bin/time`) | **\(\sim 1.4\times10^{-3}\)** | **≈ 0.07** | ✓ | xtb printed wall 0.012 s; GFN2 \(\omega\) carries the CAMM-vs-\(R\) hole |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a twin exists.

## Limitations

- **FD of analytic \(G\), not closed-form \(H\).** Cost is
  \(O(N_{\mathrm{atoms}})\) GFN1/GFN2 force evaluations. A GFN
  second-derivative kernel is not on this page. xtb 6.7.1 `--hess` is
  analytical; \(\omega\) tols absorb that path difference.
- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** Odd / non-integer valence count throws.
- **Off-eq geometry.** xtb warns “incompletely optimized”; frequencies are
  still all real. Do not cite these \(\omega\) as a GFN1/GFN2 minimum of
  water.
- **GFN2 CAMM-vs-\(R\) hole.** AES-in-Fock is
  [`gfn2-aes-fock`](gfn2-aes-fock.md); GFN2 \(\Delta\omega_1=121\) cm⁻¹
  on the water bend is the remaining angular slope (xtb `mmompop`
  two-centre \(\Theta\) on H), not a defect of the FD driver. The
  150 cm⁻¹ / 1.0 eV/Å² GFN2 tols are **not** a close of that hole.
- **GFN1 energy kernel hole.** GFN1 \(\lvert\Delta E\rvert=2.767\) meV on
  this water (CH₄ energy residual 10.2 meV on `reg_gfn1-xtb-organic`) is
  kernel/D3/intcut, not a missing Hessian term. It does not move GFN1
  \(\omega\) past 10 cm⁻¹.
- **GFN0 Hessian.** Sibling [`gfn0-xtb-analytic-hessian`](gfn0-xtb-analytic-hessian.md).
  This page closes GFN1 (primary) + GFN2 (family, with stated hole) FD \(H\).
- **No PBC / spin / solvent / IR** on this page — see Phase C queue.
  IPEA energy / \(G\) / FD Hessian are [`ipea-xtb-small`](ipea-xtb-small.md) /
  [`ipea-xtb-forces`](ipea-xtb-forces.md) /
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md). GFN-FF is
  classical (never pending on this loop).
- **VIB exception.** This slug (and `se-analytic-hessian` /
  `gfn0-xtb-analytic-hessian` / `ipea-xtb-analytic-hessian`) may set
  `-DATV_BUILD_VIB=ON`. Do not enable
  MD/MINIMIZER/HF/PW/GPW/CUDA for the TB product gate; do not build
  `reg_am1_water_bomd` / `reg_am1_water_minimize` under this configure.
- Family lock: **xtb only** for GFN rows — never DFTB+ or OpenMOPAC as a
  GFN oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: GFN CHNO energy + analytic \(G\) + family FD \(H\) +
  AES-in-Fock + IPEA E+\(G\)+\(H\) is not the GFN product closed
  (CAMM-vs-\(R\), ALPB, PBC, spin, sTDA, `--vip` remain).

## References

- Offline pin: [`regressions/semiempirical/gfn-xtb-analytic-hessian/REFERENCE.md`](../../../regressions/semiempirical/gfn-xtb-analytic-hessian/REFERENCE.md)
- Twins: `tools/tb_oracle/inputs/xtb/h2o_gfn1_hess/` and
  `h2o_gfn2_hess/` (`mol.xyz`, `xtb.out.txt`, `hessian.txt`,
  `vibspectrum.txt`)
- Force siblings: [`gfn1-xtb-forces.md`](gfn1-xtb-forces.md) /
  `regressions/semiempirical/gfn1-xtb-forces/REFERENCE.md`;
  [`gfn2-xtb-forces.md`](gfn2-xtb-forces.md) /
  `regressions/semiempirical/gfn2-xtb-forces/REFERENCE.md`
- Energy siblings: `regressions/semiempirical/gfn1-xtb-organic/REFERENCE.md`;
  `regressions/semiempirical/gfn2-xtb-water/REFERENCE.md`
- GFN0 Hessian sibling: [`gfn0-xtb-analytic-hessian.md`](gfn0-xtb-analytic-hessian.md)
- IPEA Hessian sibling: [`ipea-xtb-analytic-hessian.md`](ipea-xtb-analytic-hessian.md)
- SE Hessian foothold (DFTB2/H₂, structural): `regressions/semiempirical/se-analytic-hessian/REFERENCE.md`
- Oracle README: `tools/tb_oracle/README.md`
- Queue / handoff / loop: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`,
  `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Spec history (closed, deleted at done): git `wf/gfn-xtb-analytic-hessian`
- S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory Comput.* **13**,
  1989 (2017), DOI 10.1021/acs.jctc.7b00118 (GFN1)
- C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory Comput.* **15**,
  1652 (2019), DOI 10.1021/acs.jctc.8b01176 (GFN2)
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
