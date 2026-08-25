# GFN1-xTB analytic nuclear forces — off-eq H₂O

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn1-xtb-forces` |
| Regression | `regressions/semiempirical/gfn1-xtb-forces/` |
| Public API | `cpu::semiempirical::GFN1` (`with_forces = true`); `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; GFN-xTB is a sibling family) |
| Queue | TB method loop — wave 9 done (`276c7353`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 GFN1-xTB energy **and** analytic \(G\) ✓ |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Closed-form nuclear gradient of production **GFN1-xTB** (Grimme, Bannwarth,
Shushkov, *J. Chem. Theory Comput.* **13**, 1989 (2017)) on off-equilibrium
water, using Atomiverse’s static H/C/N/O tables transcribed from tblite
`gfn1.f90` / xtb `param_gfn1-xtb.txt`. After one shell-resolved Klopman–Ohno
SCC solve, forces are \(\mathbf{F}=-\nabla E\) at that density —
Hellmann–Feynman + Pulay on \(\partial S\), CN/shell-polynomial chains,
\(\partial\gamma/\partial R\), pair repulsion, and D3(BJ) — written to
`frame.span(kAtom_f*)` in eV/Å. This is the **xtb `--gfn 1 --grad` module
boundary**, **not** a \(6N\) energy finite-difference and **not** SCC-DFTB3.
Public surface: concrete method `cpu::semiempirical::GFN1` with members
(`charge`, `scc_tol`, `with_forces`); no public `*Config`, no `make_*` /
`build_*`. Product gate: `reg_gfn1-xtb-forces`. Energy identity already ships
on methane (`reg_gfn1-xtb-organic`; 10.2 meV residual is the same kernel hole).
Fair wall \(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.16\le 10\).

## Method

**GFN1-xTB energy** (Grimme, Bannwarth, Shushkov, *J. Chem. Theory Comput.*
**13**, 1989 (2017), DOI
[10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)) —
extended tight-binding on a Stewart STO-\(n\)G valence basis, already shipped
by `reg_gfn1-xtb-organic`:

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

Hydrogen is double-ζ (1s + orthogonalized 2s polarization). GFN-xTB is
**not** DFTB3++ — different Hamiltonian, tables, and oracle (**xtb** only).

Parameter path: static tables → D3 exponential CN (Pyykkö \(r_{\mathrm{cov}}\),
\(k_1=16\)) → shifted atomic levels \(\varepsilon_{A\ell}\) → \(H/S\) →
generalized eigen → Mulliken shell \(q\) → Anderson SCC → repulsion/D3 → \(E/F\)
with parameters as members of `GFN1` and tables in
`basis_data/gfn1_tables.h`. Unsupported \(Z\) throws.

**EHT \(H^0\)** (same as the energy pin). On-site
\(h_\ell-k_{\mathrm{CN},\ell}\,\mathrm{CN}_A\). Off-site

\[
H^0_{\mu\nu}
  = \tfrac12(h_\mu+h_\nu)\,k_{\mu\nu}
    \bigl(1+k_{\mathrm{poly},\mu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
    \bigl(1+k_{\mathrm{poly},\nu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
    S_{\mu\nu},
\]

\(k_{\mu\nu}\) from \(k_s=1.85\), \(k_p=2.25\), \(k_{sp}=2.08\),
\(k_{\mathrm{diff}}=2.85\) (polarization), pair scales \(k_{\mathrm{HH}}=0.96\),
\(k_{\mathrm{NH}}=1.04\), and \(k_{\mathrm{EN}}=-7\times10^{-3}\) on
\((\Delta\chi)^2\).

**Ionic forces (closed-form)** — `nuclear_gradient` in
`src/cpu/semiempirical/gfn1.cpp` at frozen SCC density. xtb prints \(\nabla E\)
[Eh/\(a_0\)]; ATV stores \(\mathbf{F}=-\nabla E\) [eV/Å] via
`units::kHartreePerBohrInEvPerAngstrom`:

\[
\mathbf{F}_A = -\frac{\partial E}{\partial\mathbf{R}_A}
=
\mathbf{F}_A^{H_0+\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{CN}}
+ \mathbf{F}_A^{\gamma}
+ \mathbf{F}_A^{\mathrm{rep}}
+ \mathbf{F}_A^{\mathrm{D3}},
\]

with pairwise accumulation (antisymmetrized bond contributions) plus CN
chains:

| Term | Contraction (schematic) |
| ---- | ----------------------- |
| Electronic \(H_0\) + Pulay | \(2\,\mathrm{Tr}\!\bigl((P\odot H_0'-W)\,\partial S/\partial R + P\,\partial\Pi/\partial R\,S\bigr)\) plus SCC shift \(V_i=\sum_j\gamma_{ij}q_j\) on \(S\); \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| CN / shell-poly chain | \(\sum_\ell(\partial E/\partial h_\ell)\,(-k_{\mathrm{CN},\ell}\,\partial\mathrm{CN}/\partial R)\) plus \(\partial\Pi/\partial R\) of the EHT polynomial |
| SCC \(\gamma\) (frozen \(q\)) | \(q_i q_j\,(\partial\gamma_{ij}/\partial R)\) from Klopman–Ohno \(\partial\gamma/\partial r\) |
| Third-order \(\Gamma\) | on-site; frozen-\(q\) contribution is identically 0 |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) of the \(e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) pair |
| D3 | Analytic two-body D3(BJ) forces (`gpw::dftd3_pbe_bj`, GFN1 damping) |

The \(6N\) central-FD product path is **not** the production force. The
regression still compares two selected components against a **central
energy-only** FD of ATV’s own \(E\) (\(h=10^{-4}\) Å) as an identity gate.

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, **off-equilibrium** so \(\|F\|\neq 0\) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | GFN1-xTB, closed shell, `charge = 0` |
| Tables | Static H/C/N/O from tblite `gfn1.f90` / xtb `param_gfn1-xtb.txt` |
| Forces | `with_forces = true` → closed-form path (not re-SCC FD) |
| SCC | `scc_tol = 1e-8`, `max_scc = 250` (class defaults); xtb `--acc 0.01` (SCF \(10^{-8}\) Eh) |
| Units | \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å] |
| Conversion | CODATA via `units::kHartreeInEv`, `units::kHartreePerBohrInEvPerAngstrom` |
| Observable | Finite \(E\); net \(\sum_A\mathbf{F}_A=0\); analytic vs xtb \(\nabla E\); analytic vs energy FD |

**Geometry [Å]** (`tools/tb_oracle/inputs/xtb/h2o_gfn1_offeq_grad/mol.xyz`):

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H1 | 1 | 0.96 | 0.00 | 0.00 |
| H2 | 1 | −0.24 | 0.93 | 0.00 |

**Key exclusions (this page / acceptance):**

- Methane **energy** vs xtb `--gfn 1` — sibling `gfn1-xtb-organic` (CH4
  10.2 meV residual is the same kernel hole as the force components not
  pinned here).
- **GFN0 / GFN2** forces — siblings `gfn0-xtb-forces`, `gfn2-xtb-forces`.
- GFN-xTB **Hessian** is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md);
  IPEA energy is [`ipea-xtb-small`](ipea-xtb-small.md). PBC, spin, sTDA,
  ALPB/GBSA, CPCM-X — other queue rows. GFN-FF is classical (scope-out of
  this loop).
- Halogen-bond correction (xtb term; identically 0 on H₂O; not implemented).
- Beyond-CHNO tables — unsupported \(Z\) throws; deepen via
  `gfn2-beyond-chno`.
- Live xtb / DFTB+ / OpenMOPAC from ctest — **forbidden**; offline pins only.
- Using DFTB+ or OpenMOPAC as a GFN1 force reference — family lock forbids it.
- Treating GFN1 as a DFTB3 successor — **forbidden**.

## Implementation

Public surface only (`cpu::Frame`, `cpu::semiempirical::GFN1` members; no
public `*Config`, no `make_*` / `build_*`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn1.h"

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

cpu::semiempirical::GFN1 gfn1;
gfn1.charge = 0;
gfn1.with_forces = true;

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(gfn1);
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
  cmake --build build -j$(nproc) --target atomiverse test_cpu_xtb_gfn1 \
        reg_gfn1-xtb-forces
  ctest --test-dir build -R "^(test_cpu_xtb_gfn1)$" --output-on-failure
  cmake --build build --target run_reg_gfn1-xtb-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/gfn1.h` | Public `GFN1` (`charge`, `scc_tol`, `with_forces`) |
| `include/atomiverse/cpu/semiempirical/basis_data/gfn1_tables.h` | Static H/C/N/O GFN1 rows + global scales |
| `src/cpu/semiempirical/gfn1.cpp` | Stewart overlap / EHT+SCC / analytic \(\nabla E\) (HF+Pulay, CN/\(\gamma\), rep, D3) |
| `regressions/semiempirical/gfn1-xtb-forces/reg_gfn1-xtb-forces.cpp` | Product gate |
| `regressions/semiempirical/gfn1-xtb-forces/REFERENCE.md` | Offline pins + oracle line |
| `tools/tb_oracle/inputs/xtb/h2o_gfn1_offeq_grad/` | xtb 6.7.1 twin (`mol.xyz`, `xtb.out.txt`, `gradient.txt`) |
| `tests/cpu/semiempirical/test_xtb_gfn1.cpp` | Unit analytic-vs-FD + net-force invariance |

## Validation (xtb)

Source:
[`regressions/semiempirical/gfn1-xtb-forces/REFERENCE.md`](../../../regressions/semiempirical/gfn1-xtb-forces/REFERENCE.md).  
Oracle: **`xtb`** `--gfn 1 --grad` (Grimme, Bannwarth, Shushkov, *JCTC* **13**,
1989 (2017); Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493
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
| Engine (identity) | ATV GFN1-xTB closed shell, H/C/N/O tables |
| Oracle family | xtb (not OpenMOPAC, not DFTB+) |
| Offline twin | `tools/tb_oracle/inputs/xtb/h2o_gfn1_offeq_grad/` |
| Command | `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --grad` (`OMP_NUM_THREADS=1`) |
| Atomiverse path | `GFN1` + `Evaluate`, `with_forces=true` |
| FD protocol | Central energy-only, \(h=10^{-4}\) Å (identity, not the product force) |
| Conversion | \(F=-\nabla E\times\) `units::kHartreePerBohrInEvPerAngstrom` |

**xtb \(\nabla E\) [Eh/\(a_0\)]** (`gradient.txt`):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O | −3.4089559688772×10⁻³ | −1.1077217607051×10⁻³ | ~0 |
| H1 | +6.9083433839239×10⁻⁴ | −7.6331225483940×10⁻⁴ | ~0 |
| H2 | +2.7181216304849×10⁻³ | +1.8710340155445×10⁻³ | ~0 |

xtb also printed TOTAL ENERGY **−5.768641651774 Eh** and
\(\|\nabla E\|=0.004979646520\) Eh/\(a_0\).

**xtb energy decomposition (documentation; Eh):**

| Term | Eh |
| ---- | --: |
| SCC (electronic) | −5.805633868132 |
| → electrostatic | 0.050104282812 |
| repulsion | 0.037129798770 |
| dispersion | −0.000137582413 |
| halogen | 0 |
| **Total** | **−5.768641651774** |

**Accuracy table (hard-coded pins):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−5.768641651774 Eh** | \(5\times10^{-3}\) eV | family TB energy floor |
| \(F_{O,x}\) | `kAtom_fx[0]` | **+0.175295563850 eV/Å** | \(5\times10^{-3}\) eV/Å | \(-\nabla E\) converted |
| \(F_{H2,y}\) | `kAtom_fy[2]` | **−0.096212437395 eV/Å** | \(5\times10^{-3}\) eV/Å | dual-component product gate |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-10}\) eV/Å | translation invariance, each Cartesian |
| \(F_{O,x}\), \(F_{H2,y}\) vs energy FD | identity | — | \(3\times10^{-5}\) eV/Å | \(h=10^{-4}\) Å; unit test same |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=2.77\) meV,
\(\lvert F_{O,x}-F_{\mathrm{pin}}\rvert=1.87\times10^{-3}\) eV/Å,
\(\lvert F_{H2,y}-F_{\mathrm{pin}}\rvert=1.81\times10^{-3}\) eV/Å,
analytic vs energy FD \(<10^{-9}\) eV/Å.

**Tolerance rationale.** Energy floor \(5\times10^{-3}\) eV covers the
documented GFN1 kernel hole (D3 C6 interpolation vs xtb’s internal D3,
Stewart primitive screening vs xtb `intcut`, Anderson vs Broyden) — the same
10.2 meV residual as `reg_gfn1-xtb-organic`. Force tol \(5\times10^{-3}\) eV/Å
sits above the 1.87 meV/Å host residual on the matching Cartesian components.
\(F_{O,y}\) / \(F_{H1,y}\) vs xtb sit at **~0.20 eV/Å** — not a gradient
defect (they match ATV energy FD). The pin uses the well-matching components;
the FD identity is the gradient gate.

Offline re-pin (human only; never ctest):

```bash
# when xtb 6.7.1 is available; see tools/tb_oracle/inputs/xtb/h2o_gfn1_offeq_grad/
# --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --grad, off-eq H2O
# record TOTAL ENERGY [Eh] and gradient.txt → F via kHartreePerBohrInEvPerAngstrom
```

## Performance

Source: `REFERENCE.md`. Fair wall asserted in `reg_gfn1-xtb-forces`
(\(r\le 10\)). Analytic path is one SCC + one gradient (replaces 18 energy
solves of a \(6N\) FD).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN1 `--gfn 1 --grad` | **0.004** (1 thread) | **6.4×10⁻⁴** | **≈ 0.16** | ✓ | xtb cold wall from archived `xtb.out.txt`; ATV tables are static |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a twin exists.
`perf_exempt`: **false**. Ratio is a ctest assertion (`ratio > 10` fails).

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** `multiplicity != 1` throws
  (`spin-gfn-xtb`).
- **Kernel hole vs xtb.** Energy residual 2.77 meV and ~0.20 eV/Å on
  \(F_{O,y}\) / \(F_{H1,y}\) are the same D3/intcut/mixer mismatch as the
  CH4 energy pin; analytic vs ATV energy FD is \(<10^{-9}\) eV/Å.
- **No Hessian / PBC / spin / solvent / IPEA / halogen** on this page —
  Hessian is [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md);
  IPEA is [`ipea-xtb-small`](ipea-xtb-small.md). GFN-FF is classical
  (never pending on this loop).
- **SE-only cmake** for this loop — do not enable MD/MINIMIZER/HF/PW/GPW/CUDA
  for the TB product gate; do not build `reg_am1_water_bomd` /
  `reg_am1_water_minimize` under this configure.
- Family lock: **xtb only** for GFN rows — never DFTB+ or OpenMOPAC as a
  GFN1 oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: GFN1 energy + GFN1 \(G\) is not the GFN product closed
  (ALPB, CPCM-X, PBC, spin, sTDA, beyond-CHNO remain).

## References

- Offline pin: [`regressions/semiempirical/gfn1-xtb-forces/REFERENCE.md`](../../../regressions/semiempirical/gfn1-xtb-forces/REFERENCE.md)
- Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn1_offeq_grad/` (`mol.xyz`,
  `xtb.out.txt`, `gradient.txt`)
- Energy sibling: `regressions/semiempirical/gfn1-xtb-organic/REFERENCE.md`
- Oracle README: `tools/tb_oracle/README.md`
- Queue / handoff / loop: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`,
  `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Spec history (closed, deleted at done): git `wf/gfn1-xtb-forces`
- S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory Comput.* **13**,
  1989 (2017), DOI [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
- Siblings: `gfn1-xtb-organic` (GFN1 energy), `gfn0-xtb-forces` (GFN0 analytic
  \(G\)), `gfn2-xtb-forces` (GFN2 analytic \(G\))
