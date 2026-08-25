# IPEA-xTB vertical IP/EA — Td CH₄ vs xtb `--vipea`

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ipea-xtb-vip` |
| Regression | `regressions/semiempirical/ipea-xtb-vip/` |
| Public API | `cpu::semiempirical::IPEA` (`compute_vip` / `compute_vea`); `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; IPEA is a GFN-xTB sibling, not NDDO) |
| Queue | TB method loop — harness wave 13 done (`0a7e8b2a` / `5a140845`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 IPEA-xTB VIP/VEA ✓; extra-Z tables stay ○ |
| Oracle | **xtb 6.7.1** (compiled 2026-06-09) only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Vertical ionization potential and electron affinity of production **IPEA-xTB**
(Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017)) on isolated
tetrahedral methane. This is the **xtb `--vip` / `--vea` / `--vipea` module
boundary**: a closed-shell N-electron SCC, then two ionized ΔSCC solves at
charge \(\pm 1\) with degenerate-shell occupations (xtb 300 K Fermi on a
Td-degenerate t2) minus the empirical `ipeashift` **4.8455 eV**. Not a Koopmans
orbital energy, not UHF, not GFN1. Public surface: concrete method
`cpu::semiempirical::IPEA` with members `compute_vip` / `compute_vea`; results
on `last_vip()` / `last_vea()` [eV]. No public `*Config`, no `make_*` /
`build_*`. Native static H/C/N/O tables — **no** `libxtb`. Product gate
`reg_ipea-xtb-vip` asserts

\[
\bigl|\mathrm{IP}_v - 13.0088\,\mathrm{eV}\bigr| \le 0.15\,\mathrm{eV},
\qquad
\bigl|\mathrm{EA}_v - (-1.4559\,\mathrm{eV})\bigr| \le 0.15\,\mathrm{eV}
\]

against an offline **xtb 6.7.1** IPEA `--vipea` pin, plus the same N-electron
energy window as [`ipea-xtb-small`](ipea-xtb-small.md) (0.05 eV). Measured
ionized residuals are **0.103 eV** (IP) and **0.125 eV** (EA); the 0.15 eV
window is that kernel hole plus margin. Polar water is **not** this pin
(0.5 eV hole). Fair wall \(T_{\mathrm{ATV}}/T_{\mathrm{xtb}} = 2.0 \le 10\).

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
| \(E_\Gamma=\tfrac13\sum_A\Gamma_A q_A^3\) | On-site third-order (atomic, not shell-resolved) |
| \(E_{\mathrm{rep}}\) | Classical pair \(Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B\,e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) |
| \(E_{\mathrm{D3}}\) | Molecular D3(BJ); GFN1 set \(s_6=1\), \(s_8=2.4\), \(a_1=0.63\), \(a_2=5.0\) (xtb \(s_9=0\)) |

Hydrogen is double-ζ (1s + orthogonalized 2s polarization). C/N/O carry an
extra 3s (Gram–Schmidt against valence s). CH₄: **13 AO**, 11 shells, 8
electrons. IPEA is **not** GFN1, **not** GFN2, **not** DFTB3 — different
Hamiltonian, tables, and oracle (**xtb** only). Parameters as members of
`IPEA`; tables in `basis_data/ipea_tables.h` (tblite `ipea1.f90` / xtb
`param_ipea-xtb.txt`). Unsupported \(Z\) throws.

**Vertical IP/EA (this page).** xtb `--vip` / `--vea` / `--vipea`:

\[
\mathrm{IP}_v = \bigl(E(N-1)-E(N)\bigr)[\mathrm{eV}] - \Delta_{\mathrm{IPEA}},
\qquad
\mathrm{EA}_v = \bigl(E(N)-E(N+1)\bigr)[\mathrm{eV}] - \Delta_{\mathrm{IPEA}}.
\]

\(\Delta_{\mathrm{IPEA}} = 4.8455\,\mathrm{eV}\) is `ipeashift` from
`param_ipea-xtb.txt`. In-tree constant

\[
\texttt{kIpeaShiftEv}
  = 1.78069 \times 0.1 \times 27.21138505
  \approx 4.8455\,\mathrm{eV}
\]

(`kIpeaEvToHa = 1/27.21138505`; tblite GAM3-style scale). Neutral CH₄ stays
integer double occupation (HOMO–LUMO 7.751 eV; 300 K Fermi is integer here).
Ionized SCCs are **restricted** (not UHF) and share remaining electrons
equally across MOs within a 0.05 Eh degeneracy window — that recovers xtb’s
300 K Fermi occupations on the Td-degenerate t2. ATV’s Stewart/intcut residual
splits that t2 by \(\gg kT\), so a literal 300 K Fermi would collapse to
\(2+2+1\) and drop \(E(N-1)\) by \(\sim 0.1\) eV; the product path is the
degenerate-shell share, not a second Fermi search.

Cation (7 e): occupations \(2.0000 + 1.6667\times 3\) (t2 hole). Anion (9 e):
\(2.0000\times 4 + 1.0000\). xtb totals: \(E(N)=\) **−4.670476737294 Eh**,
\(E(N-1)=\) **−4.014341473228 Eh**, \(E(N+1)=\) **−4.795043035470 Eh**.

**IPEA vs GFN1 on this geometry (same CH₄, different Hamiltonian):**

| Knob | IPEA-xTB | GFN1-xTB (`gfn1-xtb-organic`) |
| ---- | -------- | ----------------------------- |
| Extra AO | \((n+1)s\) on C/N/O | H 2s polarization only |
| CH₄ AO / shells | 13 / 11 | 12 / 10 |
| \(k_s, k_p, k_{sp}, k_{\mathrm{diff}}\) | 2.15, 2.30, 2.25, 2.25 | 1.85, 2.25, 2.08, 2.85 |
| \(g_{\exp}\) | 1 | 2 |
| Pair \(k_{\mathrm{HH}}, k_{\mathrm{NH}}\) | none | 0.96, 1.04 |
| D3(BJ) | same GFN1 set | same |
| xtb \(E_{\mathrm{tot}}\) | **−4.670476737294 Eh** | −4.274232224875 Eh |
| Product | ΔSCC IP/EA + `ipeashift` | energy / \(G\) / FD \(H\) only |

**Twin settings (binding for this page / regression):**

| Knob | Value |
| ---- | ----- |
| System | Isolated CH₄, **tetrahedral** (same xyz as `reg_ipea-xtb-small`; **not** the off-eq stretch of `reg_ipea-xtb-forces`) |
| Cell | Cubic 20 Å (isolated; min-image present, images not engaged) |
| Method | IPEA-xTB; N-electron closed shell, `charge = 0` |
| Tables | Static H/C/N/O from tblite `ipea1.f90` / xtb `param_ipea-xtb.txt` |
| VIP/VEA | `compute_vip = true`, `compute_vea = true` (forces off) |
| Ionized SCC | Restricted; degenerate-shell occ; \(T_{\mathrm{el}}=300\) K (`kIpeaEtempK`) |
| Shift | `kIpeaShiftEv` = **4.8455 eV** |
| SCC | `scc_tol = 1e-8`, `max_scc = 250` (class defaults); xtb default `--acc 1.0` |
| Units | \(E\) [eV] on `kEnergy`; \(\mathrm{IP}_v\), \(\mathrm{EA}_v\) [eV] on `last_vip` / `last_vea` |
| Conversion | \(1\,\mathrm{Ha}=27.211386245988\) eV (`units::kHartreeInEv`); shift uses tblite 27.21138505 |

**Geometry [Å]** (`tools/tb_oracle/inputs/xtb/ch4_ipea_vip/mol.xyz`):
same tetrahedral CH₄ as `reg_ipea-xtb-small` (\(s=1.09/\sqrt{3}\)).

| Atom | \(Z\) | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: | ----: |
| C | 6 | 0 | 0 | 0 |
| H | 1 | \(s\) | \(s\) | \(s\) |
| H | 1 | \(s\) | \(-s\) | \(-s\) |
| H | 1 | \(-s\) | \(s\) | \(-s\) |
| H | 1 | \(-s\) | \(-s\) | \(s\) |

IPEA is selected on the oracle by pointing `XTBPATH` at a directory whose
`param_gfn1-xtb.txt` is a copy of `param_ipea-xtb.txt` (xtb then prints the
`I P E A - x T B` banner).

**Key exclusions (this page / acceptance):**

- IPEA **energy** (this geometry) — sibling [`ipea-xtb-small`](ipea-xtb-small.md)
  (8.90 meV residual is the same kernel hole, amplified by \(\Delta q=\pm 1\)
  on the ionized SCCs).
- IPEA **analytic \(G\)** (off-eq stretch) — sibling [`ipea-xtb-forces`](ipea-xtb-forces.md).
- IPEA **FD Hessian** (Td CH₄) — sibling [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md).
- **Koopmans** \(-\varepsilon_{\mathrm{HOMO}}\) — not the xtb `--vip` product.
- **UHF** ionized SCC (`--uhf`) — `multiplicity != 1` throws; queue
  `spin-gfn-xtb`.
- Polar **water** vs xtb — polar Δq still carries a ~0.5 eV kernel hole
  (named in `reg_ipea-xtb-small` Gaps). Not this pin.
- PBC, sTDA, ALPB/GBSA, CPCM-X — other queue rows. GFN-FF is classical
  (scope-out of this loop).
- Halogen-bond correction (xtb term; identically 0 on CH₄; not implemented).
- Beyond-CHNO tables — unsupported \(Z\) throws; deepen via
  `ipea-beyond-chno`.
- Live xtb / DFTB+ / OpenMOPAC from ctest — **forbidden**; offline pins only.
- Using DFTB+ or OpenMOPAC as an IPEA IP/EA reference — family lock forbids it.
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
const double s = 1.09 / std::sqrt(3.0);
const double x[5] = {0, s, s, -s, -s};
const double y[5] = {0, s, -s, s, -s};
const double z[5] = {0, s, -s, -s, s};
const int Z[5] = {6, 1, 1, 1, 1};
for (int i = 0; i < 5; ++i) {
    frame.span(kAtom_type)[i] = Z[i];
    frame.span(kAtom_x)[i] = x[i];
    frame.span(kAtom_y)[i] = y[i];
    frame.span(kAtom_z)[i] = z[i];
    frame.span(kAtom_mass)[i] = (i == 0) ? 12.011 : 1.008;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::cubic(20.0);
frame.meta.get(kEnergy) = 0;

cpu::Evaluate eval;
cpu::semiempirical::IPEA ipea;
ipea.with_forces = false;
ipea.compute_vip = true;
ipea.compute_vea = true;
ipea.scc_tol = 1e-8;
eval.on(evaluate::Phase::evaluate).add_ref(ipea);
eval.build();
eval.run(frame);

const double e_eV = frame.meta.get(kEnergy);  // N-electron [eV]
const double vip = ipea.last_vip();          // IP_v [eV]
const double vea = ipea.last_vea();          // EA_v [eV]
```

**SE-only configure + scoped build/run** (never bare `cmake --build build`;
never unfiltered `ctest`; VIB stays OFF — this slug is Evaluate, not Hessian):

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
        reg_ipea-xtb-vip
  ./build/regressions/reg_ipea-xtb-vip
  # or: cmake --build build --target run_reg_ipea-xtb-vip
  ctest --test-dir build -R "^(test_cpu_xtb_ipea)$" --output-on-failure
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/ipea.h` | Public `IPEA` (`compute_vip`, `compute_vea`, `last_vip`, `last_vea`) |
| `include/atomiverse/cpu/semiempirical/basis_data/ipea_tables.h` | Static H/C/N/O + `kIpeaShiftEv` / `kIpeaEtempK` |
| `src/cpu/semiempirical/ipea.cpp` | N-electron SCC; ionized ΔSCC + degenerate-shell occ; IP/EA minus shift |
| `tests/cpu/semiempirical/test_xtb_ipea.cpp` | Unit: shift constant, finite CH₄ VIP/VEA, flags-off zeros |
| `regressions/semiempirical/ipea-xtb-vip/reg_ipea-xtb-vip.cpp` | Product gate |
| `tools/tb_oracle/inputs/xtb/ch4_ipea_vip/` | Offline xtb 6.7.1 twin (`mol.xyz` + `xtb.out.txt`) |

## Validation (vs xtb)

Source:
[`regressions/semiempirical/ipea-xtb-vip/REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-vip/REFERENCE.md).
Oracle: **`xtb`** IPEA (`param_ipea-xtb.txt` as GFN1) `--vipea` (Ásgeirsson,
Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017); Bannwarth et al., *WIREs
Comput. Mol. Sci.* **11**, e01493 (2021)).
`oracle_version`: **6.7.1** (compiled 2026-06-09).
Hard-coded gates in the regression binary — **never** live xtb from ctest.

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

Offline pin only. Twin protocol: `tools/tb_oracle/README.md`, inputs under
`tools/tb_oracle/inputs/xtb/ch4_ipea_vip/`.

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (identity) | ATV IPEA-xTB closed-shell N-electron + ionized ΔSCC |
| Oracle family | xtb (not OpenMOPAC, not DFTB+) |
| Offline twin | `tools/tb_oracle/inputs/xtb/ch4_ipea_vip/` |
| Command | `XTBPATH=… xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --vipea` (`OMP_NUM_THREADS=1`; `param_gfn1-xtb.txt` ← `param_ipea-xtb.txt`; banner `I P E A - x T B`) |
| Atomiverse path | `IPEA` on `Evaluate` at `evaluate::Phase::evaluate`; `compute_vip` / `compute_vea` |
| Conversion | \(E[\mathrm{eV}]=E[\mathrm{Ha}]\times 27.211386245988\) |

xtb 6.7.1 (Arrhenius; EasyBuild `xtb/6.7.1-gcc-2025b-eb` when present):

```bash
# XTBPATH dir whose param_gfn1-xtb.txt is a copy of param_ipea-xtb.txt
export XTBPATH=<twin xtbpath>
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --vipea
```

xtb prints `I P E A - x T B` and cites DOI 10.1039/c7sc00601b. Transcript:
13 AO, 8 electrons, HOMO–LUMO 7.751 eV, 8 SCC iterations (neutral); cation
occupations \(2.0000 + 1.6667\times 3\); anion \(2.0000\times 4 + 1.0000\).

**xtb energy decomposition (documentation; Eh):** same geometry as
`reg_ipea-xtb-small`.

| Term | Eh |
| ---- | --: |
| SCC (electronic) | −4.766828456418 |
| → electrostatic | 0.001132005712 |
| repulsion | 0.096806468732 |
| dispersion | −0.000454749608 |
| halogen | 0 |
| **Total \(E(N)\)** | **−4.670476737294** |
| \(E(N-1)\) cation | **−4.014341473228** |
| \(E(N+1)\) anion | **−4.795043035470** |

**Hard-coded gates (regression):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| empirical shift | `kIpeaShiftEv` | **4.8455 eV** | — | `ipeashift`; unit-test \(\lvert\Delta\rvert\le 5\times 10^{-4}\) eV |
| \(\mathrm{IP}_v\) | `IPEA::last_vip()` | **13.0088 eV** | **0.15 eV** | \(\bigl(E(N-1)-E(N)\bigr)-\Delta_{\mathrm{IPEA}}\) |
| \(\mathrm{EA}_v\) | `IPEA::last_vea()` | **−1.4559 eV** | **0.15 eV** | \(\bigl(E(N)-E(N+1)\bigr)-\Delta_{\mathrm{IPEA}}\) |
| \(E(N)\) | `kEnergy` | **−4.670476737294 Eh** (−127.09014645 eV) | **0.05 eV** | same floor as `reg_ipea-xtb-small` |
| Finite IP/EA | identity | finite | hard | `last_vip` / `last_vea` when flags set |
| SCC residual | identity | 8 iter (xtb) | `1e-8` e | N-electron product gate |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=8.90\) meV,
\(\lvert\mathrm{IP}_{\mathrm{ATV}}-\mathrm{IP}_{\mathrm{pin}}\rvert=0.103\) eV,
\(\lvert\mathrm{EA}_{\mathrm{ATV}}-\mathrm{EA}_{\mathrm{pin}}\rvert=0.125\) eV.
Cation t2 occupations match xtb \(2+1.6667\times 3\).

**Tolerance rationale.** The 0.05 eV energy window is the
`reg_ipea-xtb-small` kernel-residual floor (D3 CN vs xtb ncoord, Stewart
contraction vs `intcut`, Anderson vs Broyden). IP/EA are ΔSCC of a
**charged** species: the same kernel hole is amplified by \(\Delta q=\pm 1\).
The 0.15 eV IP/EA window is that ionized residual plus margin — **not** a
missing occupation or shift term. Polar-molecule residual vs water (~0.5 eV)
is **not** this pin.

Discriminates IPEA from GFN1: eq CH4 GFN1 is −4.274232224875 Eh vs IPEA
−4.670476737294 Eh; GFN1 has no `ipeashift` VIP/VEA product.

Offline re-pin (human only; never ctest):

```bash
# when xtb 6.7.1 is available; see tools/tb_oracle/inputs/xtb/ch4_ipea_vip/
# XTBPATH with param_ipea as GFN1; --gfn 1 --chrg 0 --uhf 0 --vipea
# Td CH4 r_CH=1.09 Å; record TOTAL ENERGY [Eh], "delta SCC IP/EA (eV)",
# "empirical IP/EA shift (eV)"
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-vip/REFERENCE.md).
Fair 1-thread wall of `reg_ipea-xtb-vip` (Release, `OMP_NUM_THREADS=1`).
Gate convention (TB loop): \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}} \le 10\)
unless `perf_exempt`. Same *product* (CH₄ ΔSCC IP/EA); ATV is three SCCs
(N, N−1, N+1) on static tables; xtb `--vipea` is the same three solves plus
I/O.

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ IPEA `--vipea` | **0.005** (xtb 6.7.1 cpu clock; transcript wall 0.006 s; `/usr/bin/time` 0.02–0.03 s) | **0.01** (Release, 1 thread) | **2.0** | ✓ | Tiny molecule; both walls are I/O-scale |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a twin exists.
`perf_exempt`: **false**. Ratio is recorded, not a ctest assertion.

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip. Extra-Z is queue `ipea-beyond-chno`.
- **Closed-shell even-electron N-electron state.** `multiplicity != 1` throws
  (`spin-gfn-xtb`). VIP/VEA start from that closed-shell ground state; ionized
  SCCs stay restricted.
- **Degenerate-shell occupations, not a second Fermi search.** Literal 300 K
  Fermi on ATV’s split t2 would be the wrong ionized density vs xtb.
- **Kernel hole vs xtb.** Energy residual 8.90 meV on this Td CH₄ is the same
  D3/intcut/mixer mismatch as `reg_ipea-xtb-small`; ionized ΔSCC amplifies it
  to 0.103 / 0.125 eV. Polar-molecule residual vs water (~0.5 eV) is **not**
  this pin.
- **No PBC / spin / solvent / sTDA / halogen** on this page. GFN-FF is
  classical (never pending on this loop).
- **SE-only cmake** for this loop — do not enable MD/MINIMIZER/HF/PW/GPW/CUDA
  for the TB product gate; do not build `reg_am1_water_bomd` /
  `reg_am1_water_minimize` under this configure. VIB stays OFF (Hessian
  exception is other slugs).
- Family lock: **xtb only** for IPEA / GFN rows — never DFTB+ or OpenMOPAC as
  an IPEA oracle; never treat GFN-xTB as a DFTB3 successor.
- Foothold ≠ closed: IPEA energy + \(G\) + FD \(H\) + VIP/VEA is not the GFN
  product closed (ALPB, CPCM-X, PBC, spin, sTDA, beyond-CHNO, extra-Z IPEA,
  `--etemp`, CAMM-vs-\(R\) remain).

## References

- Offline pin: [`regressions/semiempirical/ipea-xtb-vip/REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-vip/REFERENCE.md)
- Twin: `tools/tb_oracle/inputs/xtb/ch4_ipea_vip/` (`mol.xyz`, `xtb.out.txt`)
- Energy sibling: [`ipea-xtb-small.md`](ipea-xtb-small.md) /
  `regressions/semiempirical/ipea-xtb-small/REFERENCE.md`
- Force sibling: [`ipea-xtb-forces.md`](ipea-xtb-forces.md) /
  `regressions/semiempirical/ipea-xtb-forces/REFERENCE.md`
- Hessian sibling: [`ipea-xtb-analytic-hessian.md`](ipea-xtb-analytic-hessian.md) /
  `regressions/semiempirical/ipea-xtb-analytic-hessian/REFERENCE.md`
- Oracle README: `tools/tb_oracle/README.md`
- Queue / handoff / loop: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`,
  `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Roadmap: `docs/roadmap/semiempirical.md` §3
- Spec history (closed, deleted at done): git `wf/ipea-xtb-vip`
- V. Ásgeirsson, C. Bauer, S. Grimme, *Chem. Sci.* **8**, 4879 (2017),
  DOI [10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
- Related narrative: [ch17-semiempirical-am1](../ch17-semiempirical-am1.md)
  (NDDO/AM1; different family — OpenMOPAC oracle)
