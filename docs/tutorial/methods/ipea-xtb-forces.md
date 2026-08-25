# IPEA-xTB analytic nuclear forces — off-eq CH4

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ipea-xtb-forces` |
| Regression | `regressions/semiempirical/ipea-xtb-forces/` |
| Public API | `cpu::semiempirical::IPEA` (`with_forces = true`); `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; IPEA is a GFN-xTB sibling) |
| Queue | TB method loop — wave 11 done (`b7388af2` / `ae6b48df`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 IPEA-xTB energy **and** analytic \(G\) ✓; FD Hessian is [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md); `--vip` stays ○ |
| Oracle | **xtb 6.7.1** only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Closed-form nuclear gradient of production **IPEA-xTB** (Ásgeirsson, Bauer,
Grimme, *Chem. Sci.* **8**, 4879 (2017)) on off-equilibrium methane, using
Atomiverse’s static H/C/N/O tables transcribed from tblite `ipea1.f90` /
xtb `param_ipea-xtb.txt`. After one shell-resolved Klopman–Ohno SCC
(\(g_{\exp}=1\)), forces are \(\mathbf{F}=-\nabla E\) at that density —
Hellmann–Feynman + Pulay on \(\partial S\), CN/shell-polynomial chains,
\(\partial\gamma/\partial R\), pair repulsion, and D3(BJ) — written to
`frame.span(kAtom_f*)` in eV/Å. This is the **xtb IPEA `--grad` module
boundary** (not GFN1, not a \(6N\) energy finite-difference). Public
surface: concrete method `cpu::semiempirical::IPEA` with members
(`charge`, `scc_tol`, `with_forces`); no public `*Config`, no `make_*` /
`build_*`. Product gate: `reg_ipea-xtb-forces`. Energy identity already
ships on methane (`reg_ipea-xtb-small`; 8.90 meV residual is the same
kernel hole). Off-eq water FD identity lives in `test_cpu_xtb_ipea`
(polar ~0.5 eV hole vs xtb is not a gradient defect). Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.26\le 10\).

## Method

**IPEA-xTB energy** (V. Ásgeirsson, C. Bauer, S. Grimme, *Chem. Sci.* **8**,
4879 (2017), DOI
[10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)) — GFN1-shaped
extended tight-binding **reparametrized for IPs and EAs**, already shipped
by `reg_ipea-xtb-small`:

\[
E = \mathrm{Tr}(P H_0)
  + \tfrac12 q\,\gamma\,q
  + \tfrac13 \Gamma q^3
  + E_{\mathrm{rep}}
  + E_{\mathrm{D3}}.
\]

| Term | Role |
| ---- | ---- |
| \(E_{H^0}=\mathrm{Tr}(P H_0)\) | Occupied density on a CN-shifted EHT Hamiltonian; extra \((n+1)s\) on C/N/O; \(S\) from Stewart STO-\(n\)G |
| \(E_\gamma=\tfrac12 q\,\gamma\,q\) | Shell-resolved Klopman–Ohno SCC (**\(g_{\exp}=1\)**, harmonic-mean hardness) |
| \(E_\Gamma=\tfrac13\sum_A\Gamma_A q_A^3\) | On-site third-order (atomic, not shell-resolved); frozen-\(q\) \(\partial E_\Gamma/\partial R=0\) |
| \(E_{\mathrm{rep}}\) | Classical pair \(Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B\,e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) |
| \(E_{\mathrm{D3}}\) | Molecular D3(BJ); GFN1 set \(s_6=1\), \(s_8=2.4\), \(a_1=0.63\), \(a_2=5.0\) (xtb \(s_9=0\)) |

Hydrogen is double-ζ (1s + orthogonalized 2s polarization). C/N/O carry an
extra 3s (Gram–Schmidt against valence s). IPEA is **not** GFN1, **not**
GFN2, **not** DFTB3 — different Hamiltonian, tables, and oracle (**xtb**
only).

Parameter path: static tables → D3 exponential CN (Pyykkö \(r_{\mathrm{cov}}\),
\(k_1=16\)) → shifted atomic levels \(\varepsilon_{A\ell}\) → \(H/S\) →
generalized eigen → Mulliken shell \(q\) → Anderson SCC → repulsion/D3 → \(E/F\)
with parameters as members of `IPEA` and tables in
`basis_data/ipea_tables.h`. Unsupported \(Z\) throws.

**EHT \(H^0\)** (same as the energy pin). On-site
\(h_\ell-k_{\mathrm{CN},\ell}\,\mathrm{CN}_A\). Off-site

\[
H^0_{\mu\nu}
  = \tfrac12(h_\mu+h_\nu)\,k_{\mu\nu}
    \bigl(1+k_{\mathrm{poly},\mu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
    \bigl(1+k_{\mathrm{poly},\nu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
    S_{\mu\nu},
\]

\(k_{\mu\nu}\) from \(k_s=2.15\), \(k_p=2.30\), \(k_{sp}=2.25\),
\(k_{\mathrm{diff}}=2.25\) (polarization), and \(k_{\mathrm{EN}}=-7\times10^{-3}\)
on \((\Delta\chi)^2\). No GFN1 pair scales \(k_{\mathrm{HH}}\) /
\(k_{\mathrm{NH}}\).

**IPEA vs GFN1 on methane (same geometry class, different Hamiltonian):**

| Knob | IPEA-xTB | GFN1-xTB (`gfn1-xtb-organic` / `gfn1-xtb-forces`) |
| ---- | -------- | ------------------------------------------------ |
| Extra AO | \((n+1)s\) on C/N/O | H 2s polarization only |
| CH₄ AO / shells | 13 / 11 | 12 / 10 |
| \(k_s, k_p, k_{sp}, k_{\mathrm{diff}}\) | 2.15, 2.30, 2.25, 2.25 | 1.85, 2.25, 2.08, 2.85 |
| \(g_{\exp}\) | 1 | 2 |
| Pair \(k_{\mathrm{HH}}, k_{\mathrm{NH}}\) | none | 0.96, 1.04 |
| D3(BJ) | same GFN1 set | same |
| eq-CH4 xtb \(E_{\mathrm{tot}}\) | **−4.670476737294 Eh** | −4.274232224875 Eh |

**Ionic forces (closed-form)** — `nuclear_gradient` in
`src/cpu/semiempirical/ipea.cpp` at frozen SCC density. xtb prints \(\nabla E\)
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
| SCC \(\gamma\) (frozen \(q\)) | \(q_i q_j\,(\partial\gamma_{ij}/\partial R)\) from Klopman–Ohno \(\partial\gamma/\partial r\) (\(g_{\exp}=1\)) |
| Third-order \(\Gamma\) | on-site; frozen-\(q\) contribution is identically 0 |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) of the \(e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) pair |
| D3 | Analytic two-body D3(BJ) forces (`gpw::dftd3_pbe_bj`, GFN1 damping) |

The \(6N\) central-FD product path is **not** the production force. The
regression still compares two selected components against a **central
energy-only** FD of ATV’s own \(E\) (\(h=10^{-4}\) Å) as an identity gate.

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated CH₄, **off-equilibrium** (one C–H stretched 1.09 → 1.20 Å) so \(\|F\|\neq 0\) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | IPEA-xTB, closed shell, `charge = 0` |
| Tables | Static H/C/N/O from tblite `ipea1.f90` / xtb `param_ipea-xtb.txt` |
| Forces | `with_forces = true` → closed-form path (not re-SCC FD) |
| SCC | `scc_tol = 1e-8`, `max_scc = 250` (class defaults); xtb `--acc 0.01` (SCF \(10^{-8}\) Eh) |
| Units | \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å] |
| Conversion | CODATA via `units::kHartreeInEv`, `units::kHartreePerBohrInEvPerAngstrom` |
| Observable | Finite \(E\); net \(\sum_A\mathbf{F}_A=0\); analytic vs xtb \(\nabla E\); analytic vs energy FD |

**Geometry [Å]** (`tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/mol.xyz`):
same tetrahedral CH₄ as `reg_ipea-xtb-small` (\(s=1.09/\sqrt{3}\)) except H1
scaled so \(r(\mathrm{C{-}H1})=1.20\) Å.

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| C | 6 | 0 | 0 | 0 |
| H1 (stretched) | 1 | 0.692820323028 | 0.692820323028 | 0.692820323028 |
| H2 | 1 | 0.629311793417 | −0.629311793417 | −0.629311793417 |
| H3 | 1 | −0.629311793417 | 0.629311793417 | −0.629311793417 |
| H4 | 1 | −0.629311793417 | −0.629311793417 | 0.629311793417 |

**Key exclusions (this page / acceptance):**

- Methane **energy** vs xtb IPEA — sibling `ipea-xtb-small` (CH4
  8.90 meV residual is the same kernel hole as the force components not
  pinned here).
- **GFN0 / GFN1 / GFN2** forces — siblings `gfn0-xtb-forces`,
  `gfn1-xtb-forces`, `gfn2-xtb-forces`.
- xtb **`--vip` / `--vea`** ΔSCC IP/EA — queue `ipea-xtb-vip` (empirical
  shift; not this closed-shell gradient).
- Off-eq **water** vs xtb — polar Δq still carries a ~0.5 eV kernel hole
  (named in `reg_ipea-xtb-small` Gaps). Analytic \(F\) on water is gated by
  ATV energy FD in `test_cpu_xtb_ipea`, not this product pin.
- IPEA **FD Hessian** (Td CH₄) — sibling [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md).
  PBC, spin, sTDA, ALPB/GBSA, CPCM-X — other queue rows.
  GFN-FF is classical (scope-out of this loop).
- Halogen-bond correction (xtb term; identically 0 on CH₄; not implemented).
- Beyond-CHNO tables — unsupported \(Z\) throws; deepen via
  `gfn2-beyond-chno`.
- Live xtb / DFTB+ / OpenMOPAC from ctest — **forbidden**; offline pins only.
- Using DFTB+ or OpenMOPAC as an IPEA force reference — family lock forbids it.
- Treating IPEA / GFN-xTB as a DFTB3 successor — **forbidden**.

## Implementation

Public surface only (`cpu::Frame`, `cpu::semiempirical::IPEA` members; no
public `*Config`, no `make_*` / `build_*`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/ipea.h"

using namespace ATV;

cpu::Frame frame(5);
// Off-eq CH4 (Å): tetrahedral r(C–H)=1.09 Å except H1 stretched to 1.20 Å
const double s = 0.629311793417;          // 1.09 / sqrt(3)
const double h1 = s * (1.20 / 1.09);
const double xyz[5][3] = {
    {0.0, 0.0, 0.0},
    {h1, h1, h1},
    {s, -s, -s},
    {-s, s, -s},
    {-s, -s, s},
};
const int Z[5] = {6, 1, 1, 1, 1};
const double mass[5] = {12.011, 1.008, 1.008, 1.008, 1.008};
for (int i = 0; i < 5; ++i) {
    frame.span(kAtom_type)[i] = Z[i];
    frame.span(kAtom_x)[i] = xyz[i][0];
    frame.span(kAtom_y)[i] = xyz[i][1];
    frame.span(kAtom_z)[i] = xyz[i][2];
    frame.span(kAtom_mass)[i] = mass[i];
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
        frame.span(kAtom_fz)[i] = 0;
}
frame.box = cpu::Box::cubic(20.0);
frame.meta.get(kEnergy) = 0;

cpu::semiempirical::IPEA ipea;
ipea.charge = 0;
ipea.with_forces = true;

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(ipea);
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
  cmake --build build -j$(nproc) --target atomiverse test_cpu_xtb_ipea \
        reg_ipea-xtb-forces
  ctest --test-dir build -R "^(test_cpu_xtb_ipea)$" --output-on-failure
  cmake --build build --target run_reg_ipea-xtb-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/ipea.h` | Public `IPEA` (`charge`, `scc_tol`, `with_forces`) |
| `include/atomiverse/cpu/semiempirical/basis_data/ipea_tables.h` | Static H/C/N/O IPEA rows + global scales |
| `src/cpu/semiempirical/ipea.cpp` | Stewart overlap / extra 3s / EHT+SCC / analytic \(\nabla E\) (HF+Pulay, CN/\(\gamma\), rep, D3) |
| `regressions/semiempirical/ipea-xtb-forces/reg_ipea-xtb-forces.cpp` | Product gate |
| `regressions/semiempirical/ipea-xtb-forces/REFERENCE.md` | Offline pins + oracle line |
| `tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/` | xtb 6.7.1 twin (`mol.xyz`, `xtb.out.txt`, `gradient.txt`) |
| `tests/cpu/semiempirical/test_xtb_ipea.cpp` | Unit analytic-vs-FD (water) + net-force invariance |

## Validation (xtb)

Source:
[`regressions/semiempirical/ipea-xtb-forces/REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-forces/REFERENCE.md).  
Oracle: **`xtb`** IPEA (`param_ipea-xtb.txt` as GFN1) `--grad` (Ásgeirsson,
Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017); Bannwarth et al., *WIREs
Comput. Mol. Sci.* **11**, e01493 (2021)).  
`oracle_version`: **6.7.1** (compiled 2026-06-09).  
Hard-coded gates in the regression binary — **never** live xtb from ctest.

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (identity) | ATV IPEA-xTB closed shell, H/C/N/O tables |
| Oracle family | xtb (not OpenMOPAC, not DFTB+) |
| Offline twin | `tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/` |
| Command | `XTBPATH=… xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --grad` (`OMP_NUM_THREADS=1`; `param_gfn1-xtb.txt` ← `param_ipea-xtb.txt`; banner `I P E A - x T B`) |
| Atomiverse path | `IPEA` + `Evaluate`, `with_forces=true` |
| FD protocol | Central energy-only, \(h=10^{-4}\) Å (identity, not the product force) |
| Conversion | \(F=-\nabla E\times\) `units::kHartreePerBohrInEvPerAngstrom` |

**xtb \(\nabla E\) [Eh/\(a_0\)]** (`gradient.txt`):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| C | −3.5026444745746×10⁻² | −3.5026444745746×10⁻² | −3.5026444745746×10⁻² |
| H1 (stretched) | +4.0716066407528×10⁻² | +4.0716066407528×10⁻² | +4.0716066407528×10⁻² |
| H2 | −1.7357154065799×10⁻³ | −1.9769531276008×10⁻³ | −1.9769531276009×10⁻³ |
| H3 | −1.9769531276008×10⁻³ | −1.7357154065799×10⁻³ | −1.9769531276008×10⁻³ |
| H4 | −1.9769531276008×10⁻³ | −1.9769531276007×10⁻³ | −1.7357154065798×10⁻³ |

xtb also printed TOTAL ENERGY **−4.662477307157 Eh**,
\(\|\nabla E\|=0.093201061537\) Eh/\(a_0\), 13 AO / 11 shells / 8 electrons,
HOMO–LUMO 7.618 eV, 10 SCC iterations.

**xtb energy decomposition (documentation; Eh):**

| Term | Eh |
| ---- | --: |
| SCC (electronic) | −4.745157981515 |
| → electrostatic | 0.000977581394 |
| repulsion | 0.083136092009 |
| dispersion | −0.000455417651 |
| halogen | 0 |
| **Total** | **−4.662477307157** |

**Accuracy table (hard-coded pins):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−4.662477307157 Eh** | \(5\times10^{-2}\) eV | same kernel-residual floor as `reg_ipea-xtb-small` |
| \(F_{C,x}\) | `kAtom_fx[0]` | **+1.801132205114 eV/Å** | \(5\times10^{-3}\) eV/Å | \(-\nabla E\) converted; (111) stretch axis |
| \(F_{H1,x}\) | `kAtom_fx[1]` | **−2.093704314111 eV/Å** | \(5\times10^{-3}\) eV/Å | dual-component product gate |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-9}\) eV/Å | translation invariance, each Cartesian |
| \(F_{C,x}\), \(F_{H1,x}\) vs energy FD | identity | — | \(3\times10^{-5}\) eV/Å | \(h=10^{-4}\) Å; unit test (water) same |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=8.49\) meV,
\(\lvert F_{C,x}-F_{\mathrm{pin}}\rvert=5.12\times10^{-4}\) eV/Å,
\(\lvert F_{H1,x}-F_{\mathrm{pin}}\rvert=1.86\times10^{-3}\) eV/Å,
analytic vs energy FD \(<2\times10^{-8}\) eV/Å.

**Tolerance rationale.** Energy floor \(5\times10^{-2}\) eV is the
`reg_ipea-xtb-small` kernel-residual floor (D3 C6 interpolation vs xtb’s
internal D3, Stewart primitive screening vs xtb `intcut`, Anderson vs
Broyden) on a ~127 eV total — the same 8.90 meV eq-CH4 hole. Force tol
\(5\times10^{-3}\) eV/Å sits above the 0.51 / 1.86 meV/Å host residual on
the matching (111) Cartesian components. Off-eq **water** vs xtb is **not**
this pin (polar ~0.5 eV energy hole); the FD identity in
`test_cpu_xtb_ipea` is the water gradient gate.

Discriminates IPEA from GFN1: eq CH4 GFN1 is −4.274232224875 Eh vs IPEA
−4.670476737294 Eh.

Offline re-pin (human only; never ctest):

```bash
# when xtb 6.7.1 is available; see tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/
# XTBPATH with param_ipea as GFN1; --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --grad
# off-eq CH4 (one C–H = 1.20 Å); record TOTAL ENERGY [Eh] and gradient.txt
# → F via units::kHartreePerBohrInEvPerAngstrom
```

## Performance

Source: `REFERENCE.md`. Fair wall asserted in `reg_ipea-xtb-forces`
(\(r\le 10\)). Analytic path is one SCC + one gradient (replaces 30 energy
solves of a \(6N\) FD on 5 atoms).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ IPEA `--grad` | **0.004** (1 thread) | **1.035×10⁻³** | **≈ 0.26** | ✓ | xtb cold wall from archived `xtb.out.txt`; ATV tables are static |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a twin exists.
`perf_exempt`: **false**. Ratio is a ctest assertion (`ratio > 10` fails).

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** `multiplicity != 1` throws
  (`spin-gfn-xtb`).
- **Kernel hole vs xtb.** Energy residual 8.49 meV on this off-eq CH4 is
  the same D3/intcut/mixer mismatch as the eq-CH4 8.90 meV energy pin;
  analytic vs ATV energy FD is \(<2\times10^{-8}\) eV/Å. Polar-molecule
  residual vs water (~0.5 eV) is an energy-path hole, not a gradient
  defect.
- **No Hessian / PBC / spin / solvent / `--vip` / halogen** on this page —
  FD Hessian is [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md);
  `--vip`/`--vea` is queue `ipea-xtb-vip`. GFN-FF is classical
  (never pending on this loop).
- **SE-only cmake** for this loop — do not enable MD/MINIMIZER/HF/PW/GPW/CUDA
  for the TB product gate; do not build `reg_am1_water_bomd` /
  `reg_am1_water_minimize` under this configure.
- Family lock: **xtb only** for IPEA / GFN rows — never DFTB+ or OpenMOPAC as
  an IPEA oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: IPEA energy + IPEA \(G\) + FD Hessian is not the GFN
  product closed (`--vip`/`--vea`, ALPB, CPCM-X, PBC, spin, sTDA,
  beyond-CHNO remain).

## References

- Offline pin: [`regressions/semiempirical/ipea-xtb-forces/REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-forces/REFERENCE.md)
- Twin: `tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/` (`mol.xyz`,
  `xtb.out.txt`, `gradient.txt`)
- Energy sibling: `regressions/semiempirical/ipea-xtb-small/REFERENCE.md`
- Oracle README: `tools/tb_oracle/README.md`
- Queue / handoff / loop: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`,
  `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Spec history (closed, deleted at done): git `wf/ipea-xtb-forces`
- V. Ásgeirsson, C. Bauer, S. Grimme, *Chem. Sci.* **8**, 4879 (2017),
  DOI [10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
- Siblings: `ipea-xtb-small` (IPEA energy),
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md) (IPEA FD Hessian),
  `gfn1-xtb-forces` (GFN1 analytic \(G\)), `gfn0-xtb-forces` /
  `gfn2-xtb-forces` (GFN0 / GFN2 analytic \(G\))
