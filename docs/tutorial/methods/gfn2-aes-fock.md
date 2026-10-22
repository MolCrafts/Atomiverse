# GFN2-xTB AES in the SCC Fock — off-eq H₂O

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn2-aes-fock` |
| Public API | `cpu::semiempirical::GFN2` (`with_forces = true`); `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; GFN-xTB is a sibling family) |
| Queue | TB method loop — done |
| Roadmap | `docs/roadmap/semiempirical.md` §3 GFN2-xTB; AES and self-consistent D4 **in the SCC Fock**; analytic \(G\) is the exact derivative of the SCC energy |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

GFN2-xTB anisotropic electrostatics (AES) — charge–dipole, dipole–dipole,
charge–quadrupole, and dipole/quadrupole AXC on the cumulative atomic
multipoles of every atom (H included) — and the self-consistent D4 enter
the **SCC Fock** as \(\partial E/\partial P\). The SCC energy is therefore
stationary in \(P\), and the closed-form \(\mathbf{F}=-\nabla E\) of
[`gfn2-xtb-forces`](gfn2-xtb-forces.md) is the exact derivative of the
energy. This is the tblite `get_potential` + `add_pot_to_h1` /
`add_vmp_to_h1` module boundary: multipoles \(\to\) potential
\(\{v, \mathbf{v}^\mu, \mathbf{v}^\Theta\}\) \(\to\) \(H\) every SCC
iteration. Public surface is unchanged —
`cpu::semiempirical::GFN2{charge}` with members (`scc_tol`, `with_forces`);
no public `*Config`, no `make_*` / `build_*`. Product gate:
`reg_gfn2-aes-fock`. Same off-eq H₂O twin as the force sibling
(TB oracle (OpenMOPAC / DFTB+ / xtb)). Fair wall asserted
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\le 10\) (xtb archived wall 0.018 s).

Analytic vs central FD of ATV’s own energy agrees to
\(\le 3\times10^{-7}\) eV/Å; against xtb 6.7.1 the residuals are 11 µeV on
\(E\) and \(1.5\times10^{-5}\) eV/Å on \(F_{O,x}\). \(F_{H1,y}\) is printed,
not gated: the O / H1 \(y\)-gradients in the xtb table are an xtb 6.7.1
orientation artifact (see Validation).

## Method

**GFN2-xTB energy** (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.*
**15**, 1652 (2019), DOI
[10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)) —
closed-shell molecular SCC, already assembled by
`gfn2-xtb-water` /
[`gfn2-quadrupole-aes`](gfn2-quadrupole-aes.md):

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
| \(\operatorname{Tr}(P H^{0})\) | Occupied density on the extended-Hückel \(H^0\) (CN'-shifted self-energies, distance polynomial \(\Pi(R)\)); \(S\) from STO-\(n\)G |
| \(E_{\mathrm{IES}}\) | Shell-resolved isotropic electrostatics, \(\gamma_{AB}\) with \(g_{\exp}=2\) |
| \(E^{(3)}\) | On-site cubic \(\frac13\sum_l \Gamma_l q_l^3\) in the shell charge \(q_l\) |
| \(E_{\mathrm{rep}}\) | Screened \(Z_{\mathrm{eff}}\) pair repulsion |
| \(E_{\mathrm{D4}}\) | Self-consistent Grimme D4 (\(C_6\) scaled by the SCC Mulliken charges) — **Fock potential** \(\partial E/\partial q\); GFN2 damping \(s_6=1\), \(s_8=2.7\), \(a_1=0.52\), \(a_2=5\), \(s_9=5\) |
| \(E_{\mathrm{AES}}\) | Anisotropic electrostatics + AXC on the cumulative multipoles of every atom — **Fock potential** |

Parameter path: static tables → double-exponential CN' (\(k_a=10\),
\(k_b=20\), 2 Bohr shift, \(4/3\)-scaled Pyykkö radii) → \(H^0\) with
\(\Pi(R)=\prod\bigl(1+k\,(R/(R_A+R_B))^{1/2}\bigr)\) (\(\Pi\) radius
O = 0.64 Å) and \(S\) → generalized eigen → Mulliken shell \(q\) **and
multipoles** \(\boldsymbol{\mu}, \Theta\) → SCC mix of IES, \(E^{(3)}\),
**AES and D4** potentials → band + IES + \(E^{(3)}\) + repulsion + D4 + AES.
In ATV this is
`Gfn2ElementRow → CN' → H/S → eigen → P → SCC(γ, Γ, AES, D4) → E/F`
with parameters as members of `GFN2` and tables in
`basis_data/gfn2_xtb_tables.h`. Unsupported \(Z\) throws.

\(H^{0}\) is the CN-shifted STO-\(n\)G valence Hamiltonian (tblite
`get_hamiltonian` scales \(k_{s}=1.85\), \(k_{p}=2.23\),
\(w_{\mathrm{exp}}=0.5\)). Occupation is restricted integer filling (8
valence electrons on this H₂O). GFN-xTB is **not** DFTB3++ — different
Hamiltonian, tables, and oracle (**xtb** only).

### Cumulative atomic multipoles

Moments are cumulative atomic multipoles of \(P\) on **every** atom (H
included), with the operator centred on the bra atom \(A\) of \(\kappa\):

\[
q_A = -\sum_{l\in A}\Delta n_l,
\qquad
\mu_{A,\alpha}
  = -\sum_{\kappa\in A}\sum_{\nu} P_{\kappa\nu}
    \langle\kappa|r_\alpha-R_{A,\alpha}|\nu\rangle,
\]

\[
\Theta_{A,\alpha\beta}
  = -\sum_{\kappa\in A}\sum_{\nu} P_{\kappa\nu}
    \bigl\langle\kappa\bigr|
      \tfrac32(r-R_A)_\alpha(r-R_A)_\beta
      -\tfrac12\delta_{\alpha\beta}\lvert r-R_A\rvert^2
    \bigl|\nu\bigr\rangle,
\]

with \(\Delta n_l\) the Mulliken shell population minus its reference
occupation. The \(R_A\)-centred integrals come from origin-0 \(S\),
\(D^\alpha\), \(Q_{\alpha\beta}\): \(D^\alpha-R_{A,\alpha}S\) and the
Cartesian origin shift
\(q^{\mathrm{shift}}_{\alpha\beta}
 = Q_{\alpha\beta}-R_{\alpha}D_{\beta}-R_{\beta}D_{\alpha}+R_{\alpha}R_{\beta}S\),
mapped to the traceless form
\(\tfrac32 q^{\mathrm{shift}}_{\alpha\beta}
 -\tfrac12\delta_{\alpha\beta}\operatorname{Tr}q^{\mathrm{shift}}\)
(tblite packed \((xx,xy,yy,xz,yz,zz)\)).

Pair damping uses a CN'-dependent AES radius (tblite `damped_multipole`
0d), \(k_{\mathrm{aes}}=4\), \(\mathrm{shift}=1.2\), \(R_{\max}=5\) Bohr,
\(f_n=(1+6(R_{0,AB}/r)^n)^{-1}\) with \(n=3\) / \(n=5\), and
\(g_n=f_n/r^n\). For every pair, \(\mathbf{R}=\mathbf{R}_A-\mathbf{R}_B\):

\[
E_{AB}
  = g_3\,(q_A\boldsymbol{\mu}_B-q_B\boldsymbol{\mu}_A)\!\cdot\mathbf{R}
  + g_5\bigl[q_A\,\mathbf{R}\!\cdot\Theta_B\!\cdot\mathbf{R}
           + q_B\,\mathbf{R}\!\cdot\Theta_A\!\cdot\mathbf{R}
           + (\boldsymbol{\mu}_A\!\cdot\boldsymbol{\mu}_B)\,r^2
           - 3(\boldsymbol{\mu}_A\!\cdot\mathbf{R})(\boldsymbol{\mu}_B\!\cdot\mathbf{R})\bigr],
\]

plus on-site AXC \(\sum_A d_A\lvert\boldsymbol{\mu}_A\rvert^2
+ q^{\mathrm{kern}}_A\,\Theta_A:\Theta_A\). This is the same kernel as
[`gfn2-quadrupole-aes`](gfn2-quadrupole-aes.md).

### AES and D4 potential in the Fock

tblite `coulomb/multipole.f90` `get_potential` differentiates
\(E_{\mathrm{AES}}+E_{\mathrm{AXC}}\) w.r.t. \(\{q, \boldsymbol{\mu},
\Theta\}\) at frozen geometry / damping radii; the self-consistent D4 adds
\(\partial E_{\mathrm{D4}}/\partial q_A\) to the charge channel. Per pair
(and symmetrically for \(B\) with \(\mathbf{R}\to-\mathbf{R}\)):

\[
\begin{aligned}
v_A
  &\mathrel{+}= g_3\,\boldsymbol{\mu}_B\!\cdot\mathbf{R}
              + g_5\,\mathbf{R}\!\cdot\Theta_B\!\cdot\mathbf{R},\\
\mathbf{v}^\mu_A
  &\mathrel{+}= -g_3\,q_B\,\mathbf{R}
              + g_5\bigl(\boldsymbol{\mu}_B\,r^2
                - 3\,\mathbf{R}\,(\boldsymbol{\mu}_B\!\cdot\mathbf{R})\bigr),\\
\mathbf{v}^\Theta_A
  &\mathrel{+}= g_5\,q_B\,\mathbf{R}\mathbf{R},
\end{aligned}
\]

and on site \(v_A \mathrel{+}= \partial E_{\mathrm{D4}}/\partial q_A\),
\(\mathbf{v}^\mu_A \mathrel{+}= 2d_A\boldsymbol{\mu}_A\),
\(\mathbf{v}^\Theta_A \mathrel{+}= \partial E_{\mathrm{AXC}}/\partial\Theta_A\).
AXC on packed \(\Theta\) uses the tblite off-diagonal scale 2 in the
energy, hence 4 in the potential (diag \(2 q^{\mathrm{kern}}\Theta_{ii}\)).

tblite `scf/potential.f90` `add_vmp_to_h1` then folds those channels into
the SCC Hamiltonian (ATV: `build_fock` in `solve_gfn2`), for
\(\kappa\in A\) in shell \(l\), \(\nu\in B\) in shell \(l'\):

\[
\begin{aligned}
H_{\kappa\nu}
  &= H^0_{\kappa\nu}
  + \tfrac12 S_{\kappa\nu}\,(v_l+v_{l'})\\
  &\quad - \tfrac12\sum_\alpha
      \bigl[
        v^\mu_{A,\alpha}\langle\kappa|r_\alpha-R_{A,\alpha}|\nu\rangle
      + v^\mu_{B,\alpha}\langle\nu|r_\alpha-R_{B,\alpha}|\kappa\rangle
      \bigr]\\
  &\quad - \tfrac12\sum_k
      \bigl[
        v^\Theta_{A,k}\,\theta^{A}_{k,\kappa\nu}
      + v^\Theta_{B,k}\,\theta^{B}_{k,\nu\kappa}
      \bigr],
\end{aligned}
\]

with \(\theta^A\) the traceless \(R_A\)-centred quadrupole integrals and
\(v_l=\partial E/\partial\Delta n_l
=\sum_{l''}\gamma_{ll''}\Delta n_{l''}-\Gamma_l\Delta n_l^2-v_A\) (IES,
\(E^{(3)}\) in the shell charge, AES + D4 charge channel). Every atom
carries all three channels. After the generalized eigen, \(\Delta n\),
\(\boldsymbol{\mu}\) and \(\Theta\) are rebuilt from the new \(P\), mixed
together, and the loop repeats until their largest change is below
`scc_tol`. \(H=\partial E/\partial P\), so the converged energy is
stationary in \(P\).

**Ionic forces** are the closed-form path of
[`gfn2-xtb-forces`](gfn2-xtb-forces.md). Stationarity gives
\(\mathrm{d}E/\mathrm{d}R=\partial E/\partial R|_P-\operatorname{Tr}(W\,\partial S/\partial R)\):
Hellmann–Feynman + Pulay, CN' and \(\Pi\) chains, IES
\(\partial\gamma/\partial R\), repulsion, fixed-charge D4
(`scf::dftd4_fixed_charges`), the explicit AES kernel geometry and the
frozen-\(P\) \(\partial\boldsymbol{\mu}/\partial R\),
\(\partial\Theta/\partial R\). Analytic vs central FD of the energy is
\(\le 3\times10^{-7}\) eV/Å. xtb prints \(\nabla E\) [Eh/\(a_0\)]; ATV stores
\(\mathbf{F}=-\nabla E\) [eV/Å].

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, **off-equilibrium** (O–H stretched 0.96 → **1.10** Å) so \(\|F\|\neq 0\) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | GFN2-xTB, closed shell, `charge = 0` (even valence-electron count) |
| Tables | Static H/C/N/O from tblite `gfn2.f90` / xtb `param_gfn2-xtb.txt` |
| AES / D4 | In the SCC Fock (`build_fock` every iteration) |
| Forces | `with_forces = true` → closed-form path (not re-SCC FD) |
| Accuracy | xtb `--acc 1.0` (default GFN2) |
| Units | \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å] |
| Conversion | \(a_0=1/1.8897261245650618\) Å (CODATA); \(1\,\mathrm{Ha}=27.211386245988\) eV |
| Observable | Finite \(E\); net \(\sum_A\mathbf{F}_A=0\); analytic vs xtb \(\nabla E\) on \(F_{O,x}\); analytic vs energy FD |

**Geometry [Å]** (TB oracle (OpenMOPAC / DFTB+ / xtb)):

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H1 | 1 | **1.10** | 0.00 | 0.00 |
| H2 | 1 | −0.24 | 0.93 | 0.00 |

This is **not** the 0.96 Å energy geometry of `gfn2-xtb-water` /
`gfn2-quadrupole-aes`. Same stretch as `gfn2-xtb-forces`. xtb 6.7.1
reports 6 AO, 4 shells, 8 electrons; \(\|\nabla E\|=0.109121448851\)
Eh/\(a_0\).

**Key exclusions (this page / acceptance):**

- Near-eq **energy** vs xtb `--gfn 2` — siblings `gfn2-xtb-water` and
  `gfn2-quadrupole-aes` (same 0.96 Å molecule, residual 11 µeV).
- Gating \(F_{H1,y}\) against the xtb table — the tabled O / H1
  \(y\)-gradients are an xtb 6.7.1 orientation artifact (see Validation).
- Dipole–quadrupole / quadrupole–quadrupole pair AES ranks (not GFN2 terms).
- **GFN0 / GFN1** forces — siblings `gfn0-xtb-forces`, `gfn1-xtb-forces`.
- GFN-xTB **Hessian** is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md).
  IPEA energy / \(G\) / FD Hessian / VIP/VEA are [`ipea-xtb-small`](ipea-xtb-small.md) /
  [`ipea-xtb-forces`](ipea-xtb-forces.md) /
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md) /
  [`ipea-xtb-vip`](ipea-xtb-vip.md).
  PBC, spin, sTDA, ALPB/GBSA, CPCM-X — other queue rows. GFN-FF is
  classical (scope-out of this loop).
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

AES-in-Fock is **not** a public switch — it is the production GFN2 SCC.
The same `GFN2` object is the Evaluate energy source and the force
source.

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
        reg_gfn2-aes-fock
  ctest --test-dir build -R "^(test_cpu_gfn2_params)$" --output-on-failure
  cmake --build build --target run_reg_gfn2-aes-fock
'
```

| Path | Role |
| ---- | ---- |
| `gfn2.h` | Public `GFN2` (`scc_tol`, `charge`, `with_forces`); AES / D4 in the Fock documented |
| `gfn2_xtb_tables.h` | Static H/C/N/O GFN2 rows + AES/D4 globals |
| `src/cpu/semiempirical/gfn2.cpp` | SCC with AES + self-consistent D4 in the Fock (`aes_eval`, `build_fock`); analytic \(\nabla E\) |
| `src/cpu/scf/dftd4.cpp` | `dftd4_fixed_charges`: D4 at given charges, \(\partial E/\partial q\), fixed-\(q\) forces |
| `reg_gfn2-aes-fock.cpp` | Product gate |
| case `gfn2-aes-fock` pin | Offline pins + oracle line |
| TB oracle (OpenMOPAC / DFTB+ / xtb) | xtb 6.7.1 twin (`mol.xyz`, `xtb.out.txt`, `gradient.txt`) |
| `tests/cpu/semiempirical/test_gfn2_params.cpp` | Unit analytic-vs-FD (tol \(10^{-4}\) eV/Å) + net-force invariance |

## Validation (xtb)

Source:
case `gfn2-aes-fock` pin.
Oracle: **`xtb`** `--gfn 2 --grad` (Bannwarth et al., *J. Chem. Theory Comput.*
**15**, 1652 (2019); Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493
(2021)). Formula twin: tblite `get_potential`, `add_vmp_to_h1`.
`oracle_version`: **6.7.1**.
Hard-coded gates in the regression binary — **never** live xtb from ctest.

```text
oracle: xtb
oracle_version: 6.7.1
```

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (identity) | ATV GFN2-xTB closed shell, H/C/N/O tables, AES + D4 in Fock |
| Oracle family | xtb (not OpenMOPAC, not DFTB+) |
| Offline twin | TB oracle (OpenMOPAC / DFTB+ / xtb) |
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
| \(\|\nabla E\|\) | 0.109121448851 | Eh/\(a_0\) |
| Wall (xtb, 1 thread) | 0.018 | s |

**Hard-coded gates (regression):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−5.058625058504 Eh** | \(1\times10^{-2}\) eV | measured residual **11 µeV** (D4 term) |
| \(F_{O,x}\) | `kAtom_fx[0]` | **+3.869155970641 eV/Å** | \(0.35\) eV/Å | \(-\nabla E\) converted; stretch; residual \(1.5\times10^{-5}\) eV/Å |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-10}\) eV/Å | translation invariance, each Cartesian |
| \(F_{O,x}\), \(F_{H1,y}\) vs energy FD | identity | — | \(0.25\) eV/Å | \(h=10^{-4}\) Å; measured \(\le 3\times10^{-7}\) eV/Å |
| \(F_{H1,y}\) vs xtb | printed, **not gated** | **−0.7986 eV/Å** | — | xtb 6.7.1 orientation artifact; ATV \(-0.0083\) eV/Å |

**Tolerance rationale.** The energy floor and the product \(F_{O,x}\) tol
sit far above the measured residuals (11 µeV; \(1.5\times10^{-5}\) eV/Å).
The remaining energy difference is the D4 term (ATV
\(-1.40847\times10^{-4}\) vs xtb \(-1.41270\times10^{-4}\) Eh). The identity
gate is the gradient-consistency check against ATV’s own \(E\), angular
component included; it holds to \(\le 3\times10^{-7}\) eV/Å because AES and
D4 are in the Fock as \(\partial E/\partial P\). \(F_{H1,y}\) is not gated
against xtb: at this orientation (O at the origin, H1 on the \(x\) axis)
the O / H1 \(y\)-gradients xtb 6.7.1 prints are not the derivative of its
own energy. Central FD of xtb 6.7.1 energies (\(h=10^{-3}\) Å) gives
\(g_{O,y}=-1.942\times10^{-3}\), \(g_{H1,y}=+1.620\times10^{-4}\)
Eh/\(a_0\); xtb on the same molecule rotated 0.5 rad about \(z\) and
translated prints \(-1.943\times10^{-3}\) / \(+1.620\times10^{-4}\)
(rotated back); tblite 0.4.0 prints \(-1.9425\times10^{-3}\) /
\(+1.6183\times10^{-4}\); ATV gives \(-1.9428\times10^{-3}\) /
\(+1.6203\times10^{-4}\). The tabled \(-1.731\times10^{-2}\) /
\(+1.553\times10^{-2}\) are not GFN2 gradients.

## Performance

Source: pin notes. Fair wall asserted in `reg_gfn2-aes-fock`
(\(r\le 10\)). AES / D4 in the Fock is one potential fill per SCC iteration
on the same closed-form gradient (replaces 18 energy solves of a \(6N\)
FD).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN2 `--gfn 2 --grad` | **0.018** (1 thread) | measured in `reg_gfn2-aes-fock` | **asserted \(\le 10\)** | ✓ | xtb cold wall from archived `xtb.out.txt`; ATV tables are static |

`perf_exempt`: **false**. Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\)
when a twin exists. No xtb binary on the SE tutorial host for a fresh fair
wall; the regression fails if the ratio exceeds 10.

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** Odd / non-integer valence count throws
  (`spin-gfn-xtb`).
- **\(F_{H1,y}\) vs xtb is not gated.** The xtb 6.7.1 table’s O / H1
  \(y\)-gradients are an orientation artifact; ATV matches FD of xtb’s own
  energies, rotated-frame xtb and tblite 0.4.0 to \(\le 1\times10^{-6}\)
  Eh/\(a_0\).
- **No dipole–quadrupole / quadrupole–quadrupole pair ranks** (not GFN2
  terms).
- **No Hessian / PBC / spin / solvent / IPEA / sTDA** on this page —
  Hessian is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md);
  IPEA is
  [`ipea-xtb-small`](ipea-xtb-small.md) /
  [`ipea-xtb-forces`](ipea-xtb-forces.md) /
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md) /
  [`ipea-xtb-vip`](ipea-xtb-vip.md). GFN-FF is
  classical (never pending on this loop).
- **SE-only cmake** for this loop — do not enable MD/MINIMIZER/HF/PW/GPW/CUDA
  for the TB product gate.
- Family lock: **xtb only** for GFN rows — never DFTB+ or OpenMOPAC as a
  GFN2 oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: GFN2 energy + AES-in-Fock + analytic \(G\) + FD
  Hessian on CHNO is not the GFN product closed (ALPB, CPCM-X, PBC, spin,
  sTDA, beyond-CHNO remain ○).

## References

- Offline pin: case `gfn2-aes-fock` pin
- Twin: TB oracle (OpenMOPAC / DFTB+ / xtb) (`mol.xyz`,
  `xtb.out.txt`, `gradient.txt`)
- Energy siblings: case `gfn2-xtb-water` pin,
  case `gfn2-quadrupole-aes` pin
- Force sibling: [`gfn2-xtb-forces`](gfn2-xtb-forces.md) (same geometry)
- Hessian sibling: [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md)
- IPEA Hessian sibling: [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md)
- IPEA VIP/VEA sibling: [`ipea-xtb-vip`](ipea-xtb-vip.md)
- Oracle README: TB oracle (OpenMOPAC / DFTB+ / xtb)
- Roadmap: `docs/roadmap/semiempirical.md`
- C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory Comput.* **15**, 1652
  (2019), DOI [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
- tblite `coulomb/multipole.f90` `get_potential`; `scf/potential.f90`
  `add_vmp_to_h1`; `get_multipole_gradient_0d`
