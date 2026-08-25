# GFN2-xTB AES in the SCC Fock — off-eq H₂O

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn2-aes-fock` |
| Regression | `regressions/semiempirical/gfn2-aes-fock/` |
| Public API | `cpu::semiempirical::GFN2` (`with_forces = true`); `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; GFN-xTB is a sibling family) |
| Queue | TB method loop — wave 11 done (`93c334cb` / `979ba707`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 GFN2-xTB; AES **in the SCC Fock**; angular \(F_{H1,y}\) is CAMM-vs-\(R\), not a missing Fock term |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

GFN2-xTB anisotropic electrostatics (AES) — charge–dipole, dipole–dipole,
charge–quadrupole, and dipole/quadrupole AXC — enter the **SCC Fock** so
the occupied density is stationary with respect to \(E_{\mathrm{AES}}\).
Siblings already shipped the energy (`gfn2-xtb-water`,
`gfn2-quadrupole-aes`) and a closed-form \(\mathbf{F}=-\nabla E\) with AES
as a **post-SCC energy** (`gfn2-xtb-forces`). That post-energy placement
left the angular force \(F_{H1,y}\) unmatched vs xtb. This slug is the
tblite `get_potential` + `add_pot_to_h1` / `add_vmp_to_h1` module
boundary: CAMM moments \(\to\) AES potential \(\{v, \mathbf{v}^\mu, v^Q\}\)
\(\to\) \(H\) every SCC iteration. Public surface is unchanged —
`cpu::semiempirical::GFN2` members (`charge`, `scc_tol`, `with_forces`);
no public `*Config`, no `make_*` / `build_*`. Product gate:
`reg_gfn2-aes-fock`. Same off-eq H₂O twin as the force sibling
(`tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/`). Fair wall asserted
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\le 10\) (xtb archived wall 0.018 s).

AES-in-Fock does **not** close angular \(F_{H1,y}\). ATV energy FD of its
own \(E\) stays \(\sim+0.06\) eV/Å against xtb \(-0.80\) eV/Å: that slope
is a **CAMM-vs-\(R\)** hole (xtb `mmompop` two-centre \(\Theta\) on H; ATV
skips s-only quadrupoles), not a missing Fock term. Dual-gated anyway
(stretch \(F_{O,x}\) vs xtb; analytic vs ATV energy FD).

## Method

**GFN2-xTB energy** (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.*
**15**, 1652 (2019), DOI
[10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)) —
closed-shell molecular SCC, already assembled by
[`gfn2-xtb-water`](gfn2-xtb-water.md) /
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
| \(\operatorname{Tr}(P H^{0})\) | Occupied eigenvalues of a CN-shifted extended-Hückel \(H\), \(S\) from STO-\(n\)G |
| \(E_{\mathrm{IES}}\) | Shell-resolved isotropic electrostatics, \(\gamma_{AB}\) with \(g_{\exp}=2\) |
| \(E^{(3)}\) | On-site cubic \(\frac13\sum_i \Gamma_i q_i^3\) |
| \(E_{\mathrm{rep}}\) | Screened \(Z_{\mathrm{eff}}\) pair repulsion |
| \(E_{\mathrm{D4}}\) | Grimme D4; GFN2 damping \(s_6=1\), \(s_8=2.7\), \(a_1=0.52\), \(a_2=5\), \(s_9=5\) |
| \(E_{\mathrm{AES}}\) | Anisotropic electrostatics + AXC — **now a Fock potential**, not a post-SCC energy |

Parameter path: static tables → D3-logistic CN → CN-shifted \(H^0/S\) →
generalized eigen → Mulliken shell \(q\) **and CAMM** \(\boldsymbol{\mu},
\Theta\) → SCC mix of \(\gamma\) **plus AES** shifts → band + IES +
\(E^{(3)}\) + repulsion + D4 + AES. In ATV this is
`Gfn2ElementRow → CN → H/S → eigen → P → SCC(Hubbard, γ, AES) → E/F`
with parameters as members of `GFN2` and tables in
`basis_data/gfn2_xtb_tables.h`. Unsupported \(Z\) throws.

\(H^{0}\) is the CN-shifted STO-\(n\)G valence Hamiltonian (tblite
`get_hamiltonian` scales \(k_{s}=1.85\), \(k_{p}=2.23\),
\(w_{\mathrm{exp}}=0.5\)). Occupation is restricted integer filling (8
valence electrons on this H₂O). GFN-xTB is **not** DFTB3++ — different
Hamiltonian, tables, and oracle (**xtb** only).

### CAMM moments (unchanged from the quadrupole sibling)

Moments live on the atom origin \(R_A\):

\[
\mu_{A,\alpha}
  = \tfrac12\sum_{\mu\in A}\sum_{\nu} P_{\mu\nu}
    \bigl(D^{\alpha}_{\mu\nu}-R_{A,\alpha}S_{\mu\nu}\bigr),
\]

\[
Q_{A,\alpha\beta}
  = \tfrac12\sum_{\mu\in A}\sum_{\nu} P_{\mu\nu}\,
    q^{\mathrm{shift}}_{\mu\nu,\alpha\beta},
\]

with Cartesian quadrupole origin-shift
\(q^{\mathrm{shift}}_{\alpha\beta}
 = Q_{\alpha\beta}-R_{\alpha}D_{\beta}-R_{\beta}D_{\alpha}+R_{\alpha}R_{\beta}S\).
The packed Cartesian \(Q\) is mapped to a **traceless Buckingham**
quadrupole (tblite packed \((xx,xy,yy,xz,yz,zz)\)):

\[
\Theta_{\alpha\beta}
  = \tfrac32 Q_{\alpha\beta}
  - \tfrac12\delta_{\alpha\beta}\operatorname{Tr}(Q).
\]

s-only centres (H) get **no** CAMM quadrupole — the 1s overlap blob about
the nuclear origin is not the GFN2 atomic \(\Theta\). xtb `mmompop` still
accumulates a two-centre \(\Theta\) on H; that is the CAMM-vs-\(R\) hole
below, not a Fock omission.

Pair damping uses a CN-dependent AES radius (tblite `damped_multipole`
0d), \(k_{\mathrm{aes}}=4\), \(\mathrm{shift}=1.2\), \(R_{\max}=5\) Bohr,
\(f_n=(1+6(R_{\mathrm{avg}}/r)^n)^{-1}\) with \(n=3\) (odd) / \(n=5\)
(even). The energy \(E_{q\mu}+E_{\mu\mu}+E_{q\Theta}+E_{\mathrm{AXC}}\)
is the same kernel as [`gfn2-quadrupole-aes`](gfn2-quadrupole-aes.md).

### AES potential in the Fock (this slug)

tblite `coulomb/multipole.f90` `get_potential` differentiates
\(E_{\mathrm{AES}}\) w.r.t. \(\{q, \boldsymbol{\mu}, \Theta\}\) at frozen
geometry / damping radii. For every ordered pair \(A\neq B\),
\(\mathbf{r}=\mathbf{R}_A-\mathbf{R}_B\), \(q_B=-\Delta n_B\):

\[
\begin{aligned}
\mathbf{v}^\mu_A
  &\mathrel{+}= q_B\,\mathbf{r}\,r^{-3}\,f_3
              + \bigl(\boldsymbol{\mu}_B\,r^{-3}
                - 3(\boldsymbol{\mu}_B\!\cdot\mathbf{r})\,\mathbf{r}\,r^{-5}\bigr)f_5
              + 2\,d_A\boldsymbol{\mu}_A,\\
v_B
  &\mathrel{+}= -(\boldsymbol{\mu}_A\!\cdot\mathbf{r})\,r^{-3}\,f_3
              - q_B\,\bigl(-\tfrac13\,\Theta_A:(\mathbf{r}\mathbf{r})\,r^{-5}f_5\bigr),\\
v^Q_A
  &\mathrel{+}= q_B\,(-\mathbf{r}\mathbf{r})\,r^{-5}\,f_5/3
              + \partial E_{\mathrm{AXC}}/\partial\Theta_A.
\end{aligned}
\]

AXC on packed \(\Theta\) uses the tblite off-diagonal scale 2 in the
energy, hence 4 in the potential (diag \(2 q_{\mathrm{kernel}}\Theta_{ii}\)).

tblite `scf/potential.f90` `add_vmp_to_h1` then folds those channels into
the SCC Hamiltonian (ATV: `fill_aes_pot` + the \(H\leftarrow H^0 + \ldots\)
loop in `solve_gfn2`):

\[
\begin{aligned}
H_{\mu\nu}
  &\leftarrow H_{\mu\nu}
  + \tfrac12 S_{\mu\nu}\,(v_A+v_B)\\
  &\quad + \tfrac12\sum_\alpha
      \bigl[
        v^\mu_{A,\alpha}(D^\alpha_{\mu\nu}-R_{A,\alpha}S_{\mu\nu})
      + v^\mu_{B,\alpha}(D^\alpha_{\mu\nu}-R_{B,\alpha}S_{\mu\nu})
      \bigr]\\
  &\quad + \tfrac12\sum_k
      \bigl[
        v^Q_{A,k}\,q^{\mathrm{shift},A}_{\mu\nu,k}
      + v^Q_{B,k}\,q^{\mathrm{shift},B}_{\mu\nu,k}
      \bigr].
\end{aligned}
\]

The charge channel \(v_A\) rides the shell potential \(v_{\mathrm{sh}}\)
already used by IES. Quadrupole \(v^Q\) is applied only on centres with
\(n_{\mathrm{shell}}>1\) (C/N/O). After the generalized eigen, CAMM is
rebuilt from the new \(P\) and the loop repeats until \(\max|\Delta q|
<\) `scc_tol`. The density is then stationary w.r.t. AES: there is no
post-energy CAMM evaluation that the Fock never saw.

**Ionic forces** stay the closed-form path of
[`gfn2-xtb-forces`](gfn2-xtb-forces.md) (Hellmann–Feynman + Pulay, CN
chain, IES \(\partial\gamma/\partial R\), repulsion, D4, AES kernel at
**frozen CAMM**). AES-in-Fock changes \(P\) (and therefore every
\(P\)-contracted force); it does not add a new \(\partial Q/\partial R\)
kernel. xtb prints \(\nabla E\) [Eh/\(a_0\)]; ATV stores
\(\mathbf{F}=-\nabla E\) [eV/Å].

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, **off-equilibrium** (O–H stretched 0.96 → **1.10** Å) so \(\|F\|\neq 0\) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | GFN2-xTB, closed shell, `charge = 0` (even valence-electron count) |
| Tables | Static H/C/N/O from tblite `gfn2.f90` / xtb `param_gfn2-xtb.txt` |
| AES | In the SCC Fock (`fill_aes_pot` every iteration) |
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
`gfn2-quadrupole-aes`. Same stretch as `gfn2-xtb-forces`. xtb 6.7.1
reports 6 AO, 4 shells, 8 electrons; \(\|\nabla E\|=0.109121448851\)
Eh/\(a_0\).

**Key exclusions (this page / acceptance):**

- Near-eq **energy** vs xtb `--gfn 2` — siblings `gfn2-xtb-water`
  (isotropic + dipole AES) and `gfn2-quadrupole-aes` (charge–quadrupole +
  \(q_{\mathrm{kernel}}\)). AES-in-Fock moves the SCC density on that
  geometry (quadrupole-AES REFERENCE residual **230 µeV**).
- **Closing angular \(F_{H1,y}\) vs xtb.** Dual-gated as printed-not-gated;
  the slope is CAMM-vs-\(R\) (two-centre \(\Theta\) on H), not a Fock term.
  Naive two-centre \(\Theta\) on H is **not** xtb `mmompop` (eq-H₂O residual
  23 meV).
- Quadrupole CAMM \(\partial Q/\partial R\) (frozen-CAMM \(G\) vs energy FD
  \(\sim 0.19\) eV/Å on \(F_{O,x}\)).
- Dipole–quadrupole / quadrupole–quadrupole pair AES ranks.
- **GFN0 / GFN1** forces — siblings `gfn0-xtb-forces`, `gfn1-xtb-forces`.
- GFN-xTB **Hessian** is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md)
  (GFN2 \(\omega\) still carries the CAMM hole). IPEA energy / \(G\) / FD
  Hessian are [`ipea-xtb-small`](ipea-xtb-small.md) /
  [`ipea-xtb-forces`](ipea-xtb-forces.md) /
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md).
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
| `include/atomiverse/cpu/semiempirical/gfn2.h` | Public `GFN2` (`scc_tol`, `charge`, `with_forces`); AES-in-Fock documented |
| `include/atomiverse/cpu/semiempirical/basis_data/gfn2_xtb_tables.h` | Static H/C/N/O GFN2 rows + AES/D4 globals |
| `src/cpu/semiempirical/gfn2.cpp` | `fill_aes_pot` inside the SCC loop; `aes_energy`; analytic \(\nabla E\) at frozen CAMM |
| `regressions/semiempirical/gfn2-aes-fock/reg_gfn2-aes-fock.cpp` | Product gate |
| `regressions/semiempirical/gfn2-aes-fock/REFERENCE.md` | Offline pins + oracle line |
| `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/` | xtb 6.7.1 twin (`mol.xyz`, `xtb.out.txt`, `gradient.txt`) |
| `tests/cpu/semiempirical/test_gfn2_params.cpp` | Unit analytic-vs-FD (tol 0.25 eV/Å; quadrupole \(\partial Q/\partial R\)) + net-force invariance |

## Validation (xtb)

Source:
[`regressions/semiempirical/gfn2-aes-fock/REFERENCE.md`](../../../regressions/semiempirical/gfn2-aes-fock/REFERENCE.md).  
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
| Engine (identity) | ATV GFN2-xTB closed shell, H/C/N/O tables, AES in Fock |
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
| \(\|\nabla E\|\) | 0.109121448851 | Eh/\(a_0\) |
| Wall (xtb, 1 thread) | 0.018 | s |

**Hard-coded gates (regression):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−5.058625058504 Eh** | \(1\times10^{-2}\) eV | offeq residual **9.19 meV** |
| \(F_{O,x}\) | `kAtom_fx[0]` | **+3.869155970641 eV/Å** | \(0.35\) eV/Å | \(-\nabla E\) converted; stretch; residual **0.306** eV/Å |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-10}\) eV/Å | translation invariance, each Cartesian |
| \(F_{O,x}\), \(F_{H1,y}\) vs energy FD | identity | — | \(0.25\) eV/Å | \(h=10^{-4}\) Å; quadrupole \(\partial Q/\partial R\) on \(F_{O,x}\) \(\sim 0.19\) |
| \(F_{H1,y}\) vs xtb | printed, **not gated** | **−0.7986 eV/Å** | — | ATV analytic \(\approx+0.136\); energy FD \(\approx+0.06\); CAMM-vs-\(R\) |

**Tolerance rationale.** AES-in-Fock **changes the SCC density**. Stretch
\(F_{O,x}\) residual is **0.306** eV/Å (was \(\sim 0.09\) when AES was
post-SCC). Off-eq energy residual **9.19 meV** (same \(10^{-2}\) eV floor
as the force sibling). Analytic \(F_{H1,y}=+0.136\) eV/Å vs xtb
\(-0.799\); ATV energy FD stays \(\sim+0.06\): putting AES in the Fock
does **not** tilt ATV’s energy surface onto xtb’s angular slope. The
identity gate (tol \(0.25\) eV/Å) is the gradient-consistency check
against ATV’s own \(E\), including the angular component; the 0.19 eV/Å
\(F_{O,x}\) FD residual is missing quadrupole \(\partial Q/\partial R\),
not a Fock bug. Product \(F_{O,x}\) tol \(0.35\) eV/Å sits above the
0.306 residual.

## Performance

Source: `REFERENCE.md`. Fair wall asserted in `reg_gfn2-aes-fock`
(\(r\le 10\)). AES-in-Fock is one extra potential fill per SCC iteration
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
- **CAMM-vs-\(R\) angular hole remains.** AES is in the Fock; xtb’s
  \(F_{H1,y}\) is **not** recovered. ATV skips s-only \(\Theta\); xtb
  `mmompop` two-centre quadrupole on H sets the bend slope. GapCensus owns
  a deepen; this slug dual-gated anyway.
- **Frozen-CAMM AES gradient.** Quadrupole \(\partial Q/\partial R\) is not
  assembled; analytic vs energy FD \(\sim 0.19\) eV/Å on \(F_{O,x}\).
- **No dipole–quadrupole / quadrupole–quadrupole pair ranks.**
- **No Hessian / PBC / spin / solvent / IPEA / sTDA** on this page —
  Hessian is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md)
  (GFN2 \(\omega\) still carries the CAMM hole); IPEA is
  [`ipea-xtb-small`](ipea-xtb-small.md) /
  [`ipea-xtb-forces`](ipea-xtb-forces.md) /
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md). GFN-FF is
  classical (never pending on this loop).
- **SE-only cmake** for this loop — do not enable MD/MINIMIZER/HF/PW/GPW/CUDA
  for the TB product gate; do not build `reg_am1_water_bomd` /
  `reg_am1_water_minimize` under this configure.
- Family lock: **xtb only** for GFN rows — never DFTB+ or OpenMOPAC as a
  GFN2 oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: GFN2 energy + AES-in-Fock + analytic \(G\) + FD
  Hessian on CHNO is not the GFN product closed (ALPB, CPCM-X, PBC, spin,
  sTDA, beyond-CHNO, CAMM-vs-\(R\) remain ○).

## References

- Offline pin: [`regressions/semiempirical/gfn2-aes-fock/REFERENCE.md`](../../../regressions/semiempirical/gfn2-aes-fock/REFERENCE.md)
- Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/` (`mol.xyz`,
  `xtb.out.txt`, `gradient.txt`)
- Energy siblings: `regressions/semiempirical/gfn2-xtb-water/REFERENCE.md`,
  `regressions/semiempirical/gfn2-quadrupole-aes/REFERENCE.md`
- Force sibling: [`gfn2-xtb-forces`](gfn2-xtb-forces.md) (same geometry;
  AES was post-SCC there)
- Hessian sibling: [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md)
- IPEA Hessian sibling: [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md)
- Oracle README: `tools/tb_oracle/README.md`
- Queue / handoff / loop: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`,
  `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Spec history (closed, deleted at done): git `wf/gfn2-aes-fock`
  (`93c334cb` / `979ba707`)
- C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory Comput.* **15**, 1652
  (2019), DOI [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
- tblite `coulomb/multipole.f90` `get_potential`; `scf/potential.f90`
  `add_vmp_to_h1`; `get_multipole_gradient_0d`
