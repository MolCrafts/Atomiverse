# GFN2-xTB analytic nuclear forces — off-eq H₂O

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn2-xtb-forces` |
| Regression | `regressions/semiempirical/gfn2-xtb-forces/` |
| Public API | `cpu::semiempirical::GFN2` (`with_forces = true`); `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; GFN-xTB is a sibling family) |
| Queue | TB method loop — wave 9 done (`b8e86743`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 GFN2-xTB energy + analytic G ✓ (CHNO); AES-in-Fock landed on [`gfn2-aes-fock`](gfn2-aes-fock.md) (angular \(F_{H1,y}\) is CAMM-vs-\(R\)) |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Closed-form nuclear gradient of production **GFN2-xTB** (Bannwarth, Ehlert,
Grimme, *J. Chem. Theory Comput.* **15**, 1652 (2019)) on off-equilibrium
water, using Atomiverse’s static H/C/N/O tables transcribed from tblite
`gfn2.f90` / xtb `param_gfn2-xtb.txt`. After one shell-resolved SCC, forces
are \(\mathbf{F}=-\nabla E\) at that density — Hellmann–Feynman + Pulay on
\(\partial S\), CN chain through the CN-shifted \(H^0\), IES
\(\partial\gamma/\partial R\), pair repulsion, Grimme D4, and the AES kernel
at **frozen CAMM** — written to `frame.span(kAtom_f*)` in eV/Å. This is the
**xtb `--gfn 2 --grad` / tblite `get_hamiltonian_gradient` +
`get_multipole_gradient_0d` module boundary**, **not** a \(6N\) re-SCC
finite-difference and **not** SCC-DFTB. Public surface: concrete method
`cpu::semiempirical::GFN2` with members (`charge`, `scc_tol`, `with_forces`);
no public `*Config`, no `make_*` / `build_*`. Product gate:
`reg_gfn2-xtb-forces`. Energy identity already ships on the same molecule
(`reg_gfn2-xtb-water`, `reg_gfn2-quadrupole-aes`). Fair wall asserted
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\le 10\) (xtb archived wall 0.018 s).

This slug kept AES as a **post-SCC energy**. Sibling
[`gfn2-aes-fock`](gfn2-aes-fock.md) puts charge–dipole/quadrupole + AXC
in the SCC Fock. Stretch \(F_{O,x}\) matches xtb; the angular
\(F_{H1,y}\) hole is **CAMM-vs-\(R\)** (xtb `mmompop` two-centre \(\Theta\)
on H), not a missing Fock term and not a defect of the \(H^0\)/IES/rep/D4
closed-form path.

## Method

**GFN2-xTB energy** (Bannwarth et al., *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176) — closed-shell molecular SCC:

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
| \(E_{\mathrm{AES}}\) | Anisotropic electrostatics (charge–dipole, dipole–dipole, charge–quadrupole) + dipole/quadrupole AXC; **post-SCC energy**, not in the Fock |

Parameter path: static tables → D3-logistic CN → CN-shifted \(H^0/S\) →
generalized eigen → Mulliken shell \(q\) → SCC mix of \(\gamma\) shifts →
band + IES + \(E^{(3)}\) + repulsion + D4 + AES. In ATV this is
`Gfn2ElementRow → CN → H/S → eigen → P → SCC(Hubbard, γ) → AES(CAMM) → E/F`
with parameters as members of `GFN2` and tables in
`basis_data/gfn2_xtb_tables.h`. Unsupported \(Z\) throws.

\(H^{0}\) is the CN-shifted STO-\(n\)G valence Hamiltonian (tblite
`get_hamiltonian` scales \(k_{s}=1.85\), \(k_{p}=2.23\),
\(w_{\mathrm{exp}}=0.5\)). Occupation is restricted integer filling (8 valence
electrons on this H₂O). GFN-xTB is **not** DFTB3++ — different Hamiltonian,
tables, and oracle (**xtb** only).

**Ionic forces (closed-form)** — `solve_gfn2(..., want_grad=true)` in
`src/cpu/semiempirical/gfn2.cpp`. xtb prints \(\nabla E\) [Eh/\(a_0\)]; ATV
stores \(\mathbf{F}=-\nabla E\) [eV/Å]:

\[
\mathbf{F}_A = -\frac{\partial E}{\partial\mathbf{R}_A}
=
\mathbf{F}_A^{H^{0}}
+ \mathbf{F}_A^{\mathrm{IES}}
+ \mathbf{F}_A^{\mathrm{rep}}
+ \mathbf{F}_A^{\mathrm{D4}}
+ \mathbf{F}_A^{\mathrm{AES}},
\]

with pairwise accumulation (antisymmetrized bond contributions) plus CN
chains:

| Term | Contraction (schematic) |
| ---- | ----------------------- |
| Electronic \(H^0\) + Pulay | \(2\,\mathrm{Tr}\!\bigl((P\odot H^0-W)\,\partial S/\partial R\bigr)\) plus shell-polynomial \(\partial\Pi/\partial R\); \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| Self-energy CN chain | \(\sum_A (\partial E_{H^0}/\partial\varepsilon_A)\,(\partial\varepsilon_A/\partial\mathrm{CN}_A)\,(\partial\mathrm{CN}_A/\partial R)\) |
| IES \(\gamma\) (fixed \(q\)) | \(\sum_{ij} q_i q_j\,(\partial\gamma_{ij}/\partial R)\) |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) of the GFN2 \(Z_{\mathrm{eff}} e^{-\alpha r^{k_{\exp}}}/r\) pair (\(k_{\exp}=1.5\), light–light \(1.0\)) |
| D4 | Analytic two-body + ATM D4 forces (`cpu::scf::dftd4`) |
| AES kernel (frozen CAMM) | tblite `get_multipole_gradient_0d`: charge–dipole, dipole–dipole, charge–quadrupole + CN\(\to R_{\mathrm{mp}}\) damping chain; CAMM dipoles chain through \(\partial E_{\mathrm{AES}}/\partial\boldsymbol{\mu}\) at frozen \(P\) (incl. dipole AXC) |

The \(6N\) re-SCC product path is **not** the production force. The regression
still compares two selected components against a **central energy-only** FD of
ATV’s own \(E\) (\(h=10^{-4}\) Å) as an identity gate.

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, **off-equilibrium** (O–H stretched 0.96 → **1.10** Å) so \(\|F\|\neq 0\) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | GFN2-xTB, closed shell, `charge = 0` (even valence-electron count) |
| Tables | Static H/C/N/O from tblite `gfn2.f90` / xtb `param_gfn2-xtb.txt` |
| Forces | `with_forces = true` → closed-form path (not re-SCC FD) |
| Accuracy | xtb `--acc 1.0` (default GFN2) |
| Units | \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å] |
| Conversion | \(a_0=1/1.8897261245650618\) Å (CODATA); \(1\,\mathrm{Ha}=27.211386245988\) eV |
| Observable | Finite \(E\); net \(\sum_A\mathbf{F}_A=0\); analytic vs xtb \(\nabla E\) on \(F_{O,x}\); analytic vs energy FD |

**Geometry [Å]** (`tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/mol.xyz`):

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H1 | 1 | **1.10** | 0.00 | 0.00 |
| H2 | 1 | −0.24 | 0.93 | 0.00 |

This is **not** the 0.96 Å energy geometry of `gfn2-xtb-water` /
`gfn2-quadrupole-aes`. xtb 6.7.1 reports 6 AO, 4 shells, 8 electrons;
HOMO–LUMO 10.606 eV; Mulliken \(q(\mathrm{O})=-0.549\),
\(q(\mathrm{H})=0.271/0.277\).

**Key exclusions (this page / acceptance):**

- Near-eq **energy** vs xtb `--gfn 2` — siblings `gfn2-xtb-water` (isotropic +
  dipole AES) and `gfn2-quadrupole-aes` (charge–quadrupole + \(q_{\mathrm{kernel}}\)).
- **AES as an SCC Fock potential.** This slug’s kernel is post-energy;
  sibling [`gfn2-aes-fock`](gfn2-aes-fock.md) lands the Fock term. The
  angular \(F_{H1,y}\) vs xtb hole is CAMM-vs-\(R\), not this slug.
- Dipole–quadrupole / quadrupole–quadrupole pair AES ranks; quadrupole CAMM
  \(\partial Q/\partial R\).
- **GFN0 / GFN1** forces — siblings `gfn0-xtb-forces`, `gfn1-xtb-forces`.
- GFN-xTB **Hessian** is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md);
  IPEA energy / \(G\) / FD Hessian / VIP/VEA are [`ipea-xtb-small`](ipea-xtb-small.md) /
  [`ipea-xtb-forces`](ipea-xtb-forces.md) /
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md) /
  [`ipea-xtb-vip`](ipea-xtb-vip.md). PBC, spin,
  sTDA, ALPB/GBSA, CPCM-X — other queue rows. GFN-FF is classical
  (scope-out of this loop).
- Beyond-CHNO tables — unsupported \(Z\) throws; deepen via
  `gfn2-beyond-chno`.
- Live xtb / DFTB+ / OpenMOPAC from ctest — **forbidden**; offline pins only.
- Using DFTB+ or OpenMOPAC as a GFN2 force reference — family lock forbids it.
- Treating GFN2 as a DFTB3 successor — **forbidden**.

## Implementation

Public surface only (`cpu::Frame`, `cpu::semiempirical::GFN2` members; no
public `*Config`, no `make_*` / `build_*`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn2.h"

using namespace ATV;

cpu::Frame frame(3);
// Off-eq water (Å): O–H stretched 0.96 → 1.10 so |F| ≠ 0
frame.span(kAtom_type)[0] = 8;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_type)[1] = 1;
frame.span(kAtom_x)[1] = 1.10;
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

cpu::semiempirical::GFN2 gfn2;
gfn2.with_forces = true;

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(gfn2);
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
  cmake --build build -j$(nproc) --target atomiverse test_cpu_gfn2_params \
        reg_gfn2-xtb-forces
  ctest --test-dir build -R "^(test_cpu_gfn2_params)$" --output-on-failure
  cmake --build build --target run_reg_gfn2-xtb-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/gfn2.h` | Public `GFN2` (`scc_tol`, `charge`, `with_forces`) |
| `include/atomiverse/cpu/semiempirical/basis_data/gfn2_xtb_tables.h` | Static H/C/N/O GFN2 rows + AES/D4 globals |
| `src/cpu/semiempirical/gfn2.cpp` | SCC + AES energy + analytic \(\nabla E\) (HF+Pulay, CN, IES, rep, D4, frozen-CAMM AES) |
| `regressions/semiempirical/gfn2-xtb-forces/reg_gfn2-xtb-forces.cpp` | Product gate |
| `regressions/semiempirical/gfn2-xtb-forces/REFERENCE.md` | Offline pins + oracle line |
| `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/` | xtb 6.7.1 twin (`mol.xyz`, `xtb.out.txt`, `gradient.txt`) |
| `tests/cpu/semiempirical/test_gfn2_params.cpp` | Unit analytic-vs-FD + net-force invariance |

## Validation (xtb)

Source:
[`regressions/semiempirical/gfn2-xtb-forces/REFERENCE.md`](../../../regressions/semiempirical/gfn2-xtb-forces/REFERENCE.md).  
Oracle: **`xtb`** `--gfn 2 --grad` (Bannwarth et al., *J. Chem. Theory Comput.*
**15**, 1652 (2019); Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493
(2021)).  
`oracle_version`: **6.7.1**.  
Hard-coded gates in the regression binary — **never** live xtb from ctest.

```text
oracle: xtb
oracle_version: 6.7.1
```

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (identity) | ATV GFN2-xTB closed shell, H/C/N/O tables |
| Oracle family | xtb (not OpenMOPAC, not DFTB+) |
| Offline twin | `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/` |
| Command | `xtb mol.xyz --gfn 2 --chrg 0 --uhf 0 --acc 1.0 --grad` (`OMP_NUM_THREADS=1`) |
| Atomiverse path | `GFN2` + `Evaluate`, `with_forces=true` |
| FD protocol | Central energy-only, \(h=10^{-4}\) Å (identity, not the product force) |
| Conversion | \(F=-\nabla E\times\) `units::kHartreePerBohrInEvPerAngstrom` |

**xtb \(\nabla E\) [Eh/\(a_0\)]** (`gradient.txt`):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O | −7.5243103995119×10⁻² | −1.7310256877367×10⁻² | ~0 |
| H1 | +7.5511039446358×10⁻² | +1.5529339054184×10⁻² | ~0 |
| H2 | −2.6793545123835×10⁻⁴ | +1.7809178231828×10⁻³ | ~0 |

xtb also printed TOTAL ENERGY **−5.058625058504 Eh** and
\(\|\nabla E\|=0.109121448851\) Eh/\(a_0\).

### Oracle decomposition (xtb 6.7.1, `--gfn 2 --grad`)

| Quantity | Value | Unit |
| -------- | ----: | ---- |
| Total energy | **−5.058625058504** | Eh |
| SCC | −5.080060971331 | Eh |
| isotropic ES | +0.031320691734 | Eh |
| anisotropic ES | +0.001306928885 | Eh |
| anisotropic XC | −0.000150463849 | Eh |
| dispersion | −0.000143706605 | Eh |
| repulsion | +0.021435912826 | Eh |
| HOMO–LUMO | 10.605534434298 | eV |
| Wall (xtb, 1 thread) | 0.018 | s |

**Hard-coded gates (regression):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−5.058625058504 Eh** | \(1\times10^{-2}\) eV | offeq stretch residual ~8.9 meV |
| \(F_{O,x}\) | `kAtom_fx[0]` | **+3.869155970641 eV/Å** | \(0.15\) eV/Å | \(-\nabla E\) converted; stretch component |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-10}\) eV/Å | translation invariance, each Cartesian |
| \(F_{O,x}\), \(F_{H1,y}\) vs energy FD | identity | — | \(3\times10^{-2}\) eV/Å | \(h=10^{-4}\) Å; unit test same |
| \(F_{H1,y}\) vs xtb | printed, **not gated** | **−0.7986 eV/Å** | — | CAMM-vs-\(R\) / quadrupole-CAMM hole |

**Tolerance rationale.** This slug’s AES kernel is post-SCC (the Fock
deepen is [`gfn2-aes-fock`](gfn2-aes-fock.md)). The offeq stretch residual
is **8.9 meV** vs 26 µeV on the 0.96 Å energy pin — energy floor
\(10^{-2}\) eV. The stretch force \(F_{O,x}\) matches xtb to
\(\sim 0.09\) eV/Å (analytic vs ATV energy FD \(\sim 0.02\) eV/Å); product
tol \(0.15\) eV/Å sits above that. The angular \(F_{H1,y}\) is a
CAMM-vs-\(R\) / quadrupole-CAMM hole (xtb \(\approx-0.80\) eV/Å vs ATV energy FD
\(\approx+0.06\)) and is **not** a hard pin; it is the next deepen, not a
defect of the \(H^0\)/IES/rep/D4 closed-form path. The FD identity (tol
\(3\times10^{-2}\) eV/Å) is the gradient-consistency gate, including the
angular component against ATV’s own \(E\).

## Performance

Source: `REFERENCE.md`. Fair wall asserted in `reg_gfn2-xtb-forces`
(\(r\le 10\)). Analytic path is one SCC + one gradient (replaces 18 energy
solves of a \(6N\) FD).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN2 `--gfn 2 --grad` | **0.018** (1 thread) | measured in `reg_gfn2-xtb-forces` | **asserted \(\le 10\)** | ✓ | xtb cold wall from archived `xtb.out.txt`; ATV tables are static |

`perf_exempt`: **false**. Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\)
when a twin exists. No xtb binary on the SE tutorial host for a fresh fair
wall; the regression fails if the ratio exceeds 10.

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** Odd / non-integer valence count throws
  (`spin-gfn-xtb`).
- **AES on this page is post-SCC.** Charge–dipole / dipole–dipole /
  charge–quadrupole kernel gradient at frozen CAMM is in this path.
  Sibling [`gfn2-aes-fock`](gfn2-aes-fock.md) puts AES in the Fock; xtb’s
  angular \(F_{H1,y}\) remains a **CAMM-vs-\(R\)** hole, not a failed
  stretch pin.
- **No Hessian / PBC / spin / solvent / IPEA / sTDA** on this page —
  Hessian is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md)
  (GFN2 \(\omega\) carries the CAMM-vs-\(R\) hole); IPEA is
  [`ipea-xtb-small`](ipea-xtb-small.md) /
  [`ipea-xtb-forces`](ipea-xtb-forces.md) /
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md) /
  [`ipea-xtb-vip`](ipea-xtb-vip.md). GFN-FF is
  classical (never pending on this loop).
- **SE-only cmake** for this loop — do not enable MD/MINIMIZER/HF/PW/GPW/CUDA
  for the TB product gate; do not build `reg_am1_water_bomd` /
  `reg_am1_water_minimize` under this configure.
- Family lock: **xtb only** for GFN rows — never DFTB+ or OpenMOPAC as a
  GFN2 oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: GFN2 energy + GFN2 \(G\) + AES-in-Fock on CHNO is not
  the GFN product closed (ALPB, CPCM-X, PBC, spin, sTDA, beyond-CHNO,
  CAMM-vs-\(R\) remain ○).

## References

- Offline pin: [`regressions/semiempirical/gfn2-xtb-forces/REFERENCE.md`](../../../regressions/semiempirical/gfn2-xtb-forces/REFERENCE.md)
- Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/` (`mol.xyz`,
  `xtb.out.txt`, `gradient.txt`)
- Energy siblings: `regressions/semiempirical/gfn2-xtb-water/REFERENCE.md`,
  `regressions/semiempirical/gfn2-quadrupole-aes/REFERENCE.md`
- Force siblings: `gfn0-xtb-forces`, `gfn1-xtb-forces` (xtb `--gfn 0/1 --grad`)
- Oracle README: `tools/tb_oracle/README.md`
- Queue / handoff / loop: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`,
  `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Spec history (closed, deleted at done): git `wf/gfn2-xtb-forces`
  (`b8e86743` / `37b9a008`)
- C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory Comput.* **15**, 1652
  (2019), DOI [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
- tblite `get_hamiltonian_gradient`, `get_multipole_gradient_0d`
