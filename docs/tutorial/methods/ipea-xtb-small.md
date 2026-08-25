# IPEA-xTB — CH₄ energy foothold (H/C/N/O)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ipea-xtb-small` |
| Regression | `regressions/semiempirical/ipea-xtb-small/` |
| Public API | `cpu::semiempirical::IPEA` + `cpu::Evaluate` |
| Related narrative | [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface; IPEA is a GFN-family sibling, not NDDO) |
| Queue | TB method loop — harness wave 10 done (`50066df7` / `4c278a20`) |
| Roadmap | `docs/roadmap/semiempirical.md` §3 IPEA-xTB energy ✓; analytic \(G\) is [`ipea-xtb-forces`](ipea-xtb-forces.md); FD Hessian is [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md); `--vip` stays ○ |
| Oracle | **xtb 6.7.1** (compiled 2026-06-09) only (family-locked; never OpenMOPAC / DFTB+) |

## Summary

Special-purpose **IPEA-xTB** (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**,
4879 (2017)) total energy on isolated tetrahedral methane, using Atomiverse’s
static H/C/N/O tables transcribed from tblite `ipea1.f90` / xtb
`param_ipea-xtb.txt`. IPEA is a **GFN1 reparametrization for IPs and EAs**,
not GFN1 itself: extra \((n+1)s\) on C/N/O, Klopman–Ohno \(g_{\exp}=1\), and
its own \(k_{s,p,\mathrm{sp},\mathrm{diff}}\) / onsite / repulsion numbers.
Same geometry as `reg_gfn1-xtb-organic` so the two Hamiltonians are
discriminated on one molecule (GFN1 −4.274232224875 Eh; IPEA
**−4.670476737294 Eh**). Public path: concrete `cpu::semiempirical::IPEA`
(members for SCC / charge / multiplicity; no `*Config`, no `make_*` /
`build_*`), bound into `cpu::Evaluate` at `evaluate::Phase::evaluate`. Native
tables only — **no** `libxtb`. Product gate `reg_ipea-xtb-small` asserts

\[
\bigl|E_{\mathrm{ATV}} - (-127.09014645\,\mathrm{eV})\bigr| \le 0.05\,\mathrm{eV}
\]

against an offline **xtb 6.7.1** IPEA-xTB pin. Measured ATV residual at this
geometry is **8.90 meV**. This page is energy-only; analytic \(G\) is
[`ipea-xtb-forces`](ipea-xtb-forces.md); FD Hessian of that \(G\) is
[`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md). Fair wall vs xtb:
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}} = 2.0 \le 10\).

## Method

**IPEA-xTB** (V. Ásgeirsson, C. Bauer, S. Grimme, *Chem. Sci.* **8**, 4879
(2017); DOI [10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)).
Extended tight-binding on a Stewart STO-\(n\)G valence basis, built for
vertical ionization potentials and electron affinities in a QCEIMS mass-spec
context. Hamiltonian skeleton is GFN1 (EHT \(H^0\) from overlap, shell-resolved
Klopman–Ohno SCC, on-site third-order \(\Gamma q^3\), exponential pair
repulsion, D3(BJ) with the GFN1 damping set) but **reparametrized**, with
extra \((n+1)s\) functions on C/N/O. Not GFN1, not GFN2, not DFTB. Family
oracle is **xtb** only.

Closed-shell total energy on the public surface:

\[
E = E_{H^0} + E_{\gamma} + E_{\Gamma} + E_{\mathrm{rep}} + E_{\mathrm{D3}},
\]

with \(E\) written in **eV** to `frame.meta.get(kEnergy)`. Internals work in
Hartree / Bohr.

1. **Basis / overlap.** Per-element Stewart STO-\(n\)G primitives (tblite
   `basis/slater.f90` coefficients). H: 1s (4G, valence) + 2s (3G,
   polarization, Gram–Schmidt against 1s). C/N/O: 2s + 2p (6G) **plus** 3s
   (3G, extra \((n+1)s\), Gram–Schmidt against valence s). Cartesian overlap
   \(S_{\mu\nu}\) from Gaussian product integrals. CH₄: **13 AO**, 11 shells,
   8 electrons (C contributes 5 AO vs GFN1’s 4).
2. **EHT \(H^0\).** On-site \(h_l - k_{\mathrm{CN},l}\,\mathrm{CN}_A\) with D3
   exponential coordination numbers (Pyykkö \(r_{\mathrm{cov}}\), \(k_1=16\)).
   Off-site
   \[
   H^0_{\mu\nu}
     = \tfrac12(h_\mu+h_\nu)\,k_{\mu\nu}
       \bigl(1+k_{\mathrm{poly},\mu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
       \bigl(1+k_{\mathrm{poly},\nu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
       S_{\mu\nu},
   \]
   \(k_{\mu\nu}\) from \(k_s=2.15\), \(k_p=2.30\), \(k_{sp}=2.25\),
   \(k_{\mathrm{diff}}=2.25\) (polarization shells), and
   \(k_{\mathrm{EN}}=-7\times10^{-3}\) on \((\Delta\chi)^2\). No GFN1 pair
   scales \(k_{\mathrm{HH}}\) / \(k_{\mathrm{NH}}\).
3. **SCC.** Shell charges \(q_i = n_i - n_i^0\) (Mulliken). Klopman–Ohno
   \(\gamma_{ij}\) with **\(g_{\exp}=1\)** (GFN1 uses 2) and harmonic-mean
   shell hardness. On-site third-order \(E_{\Gamma}=\frac13\sum_A\Gamma_A q_A^3\)
   (atomic, not shell-resolved). Anderson mixing (`mix_beta`, history 6)
   until \(\max|\Delta q_{\mathrm{shell}}| < \texttt{scc\_tol}\). Integer
   occupation (CH₄ HOMO–LUMO 7.751 eV; xtb’s 300 K Fermi fill is integer
   here).
4. **Repulsion.**
   \[
   E_{\mathrm{rep}}
     = \sum_{A<B}
       \frac{Z_A^{\mathrm{eff}} Z_B^{\mathrm{eff}}}{R}
       \exp\bigl(-\sqrt{\alpha_A\alpha_B}\,R^{1.5}\bigr).
   \]
5. **Dispersion.** Molecular D3(BJ), GFN1 set \(s_6=1\), \(s_8=2.4\),
   \(a_1=0.63\), \(a_2=5.0\) (xtb `s9=0`). Halogen-bond correction is
   identically 0 on CH₄ and is not implemented.

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

**Pin Hamiltonian twin (xtb):**

| Item | Setting |
| ---- | ------- |
| Method | IPEA-xTB (xtb `--gfn 1` with `XTBPATH` pointing at `param_gfn1-xtb.txt` ← `param_ipea-xtb.txt`; banner `I P E A - x T B`) |
| Spin | Closed-shell (`--uhf 0`; ATV `multiplicity = 1`) |
| Charge | 0 (`--chrg 0`) |
| Accuracy | xtb default `--acc 1.0` (`--sp`) |
| System | CH₄, \(r(\mathrm{C{-}H})=1.09\) Å, tetrahedral |
| Cell | cubic 20 Å on ATV (isolated; xtb molecular) |
| Basis | 13 AOs (C 2s2p3s + 4×H 1s2s), 11 shells, 8 electrons |
| Oracle | xtb **6.7.1** (compiled 2026-06-09) |

xtb `--vip` is a **multi-step ΔSCC IP** with an empirical shift; it is
**not** this energy pin.

**What this pin includes**

| Item | Value |
| ---- | ----- |
| Method | IPEA-xTB energy only |
| Tables | static `basis_data/ipea_tables.h` (H/C/N/O) |
| Forces | off (`with_forces = false`; true throws) |
| SCC | `scc_tol = 1e-8`, `max_scc = 250` (default; regression uses 1e-8) |
| Charge | 0 |

**Key exclusions (non-goals of this slug)**

- Analytic nuclear forces (`with_forces` throws; queue deepen).
- xtb `--vip` / `--vea` ΔSCC ionization / affinity (cation/anion; empirical
  shift). Not this closed-shell energy.
- Open shell (`multiplicity != 1` throws; queue `spin-gfn-xtb`).
- Halogen-bond correction (xtb term; identically 0 on CH₄; not implemented).
- ALPB/GBSA, CPCM-X/ddCOSMO, PBC, sTDA.
- GFN0 / GFN1 / GFN2 Hamiltonians (siblings `gfn0-beyond-toy`,
  `gfn1-xtb-organic`, `gfn2-xtb-water`).
- Elements beyond H/C/N/O (unsupported Z throws).
- `libxtb` / runtime `param_ipea-xtb.txt` parser.
- Live xtb (or any third-party oracle) inside ctest.
- OpenMOPAC / DFTB+ as reference for this family (**forbidden** cross-oracle).
- Treating IPEA / GFN-xTB as a DFTB3 successor.
- Public `*Config` bags / `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `IPEA` members; parameters are members /
ctor defaults — iron law):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/ipea.h"

using namespace ATV;

// CH4 tetrahedral, r(C–H) = 1.09 Å; type = Z. Same xyz as GFN1 organic pin.
cpu::Frame frame(5);
frame.span(kAtom_type)[0] = 6;
frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] = 0.0;
const double s = 1.09 / std::sqrt(3.0);
const double hpos[4][3] = {
    {s, s, s}, {s, -s, -s}, {-s, s, -s}, {-s, -s, s},
};
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_type)[i + 1] = 1;
    frame.span(kAtom_x)[i + 1] = hpos[i][0];
    frame.span(kAtom_y)[i + 1] = hpos[i][1];
    frame.span(kAtom_z)[i + 1] = hpos[i][2];
}
frame.box = cpu::Box::cubic(20.0);

cpu::Evaluate eval;
cpu::semiempirical::IPEA ipea;
ipea.with_forces = false;
ipea.scc_tol = 1e-8;
eval.on(evaluate::Phase::evaluate).add_ref(ipea);
eval.build();
eval.run(frame);

const double e_eV = frame.meta.get(kEnergy);  // [eV]
```

**SE-only configure + scoped build/run** (never bare `cmake --build build`;
never unfiltered `ctest`):

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
        reg_ipea-xtb-small
  ./build/regressions/reg_ipea-xtb-small
  # or: cmake --build build --target run_reg_ipea-xtb-small
  ctest --test-dir build -R "^(test_cpu_xtb_ipea)$" --output-on-failure
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/ipea.h` | Public `IPEA` (`scc_tol`, `max_scc`, `mix_beta`, `charge`, `multiplicity`, `with_forces`) |
| `include/atomiverse/cpu/semiempirical/basis_data/ipea_tables.h` | Static IPEA H/C/N/O (tblite `ipea1.f90` / xtb `param_ipea-xtb.txt`) |
| `src/cpu/semiempirical/ipea.cpp` | Stewart overlap / extra 3s / EHT+SCC / rep / D3 (private) |
| `tests/cpu/semiempirical/test_xtb_ipea.cpp` | Unit: tables, Z throw, forces throw, finite H₂O \(E\) |
| `regressions/semiempirical/ipea-xtb-small/reg_ipea-xtb-small.cpp` | Product gate |
| `tools/tb_oracle/inputs/xtb/ch4_ipea/` | Offline xtb 6.7.1 twin (`mol.xyz` + `xtb.out.txt`) |

## Validation (vs xtb)

Source:
[`regressions/semiempirical/ipea-xtb-small/REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-small/REFERENCE.md).

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

Offline pin only — **never** launch xtb from ctest. Twin protocol:
`tools/tb_oracle/README.md`, inputs under
`tools/tb_oracle/inputs/xtb/ch4_ipea/`.

**Setup:**

| Item | Value |
| ---- | ----- |
| Molecule | CH₄, charge 0, closed shell |
| Geometry [Å] | C at origin; H at \(\pm s,\pm s,\pm s\) with \(s=1.09/\sqrt{3}\) |
| \(r(\mathrm{C{-}H})\) | 1.09 Å |
| Cell | cubic \(L = 20\) Å (ATV isolation) |
| Hamiltonian | IPEA-xTB |
| Tables | in-tree `ipea_tables.h` (H, C) |
| Provenance | xtb 6.7.1 `TOTAL ENERGY` −4.670476737294 Ha; banner `I P E A - x T B` |

xtb 6.7.1 (Arrhenius; EasyBuild `xtb/6.7.1-gcc-2025b-eb` when present):

```bash
# XTBPATH dir whose param_gfn1-xtb.txt is a copy of param_ipea-xtb.txt
export XTBPATH=<twin xtbpath>
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --sp
```

xtb prints `I P E A - x T B` and cites DOI 10.1039/c7sc00601b. Transcript:
13 AO, 8 electrons, HOMO–LUMO 7.751 eV, 8 SCC iterations.

**xtb energy decomposition (documentation; Ha):**

| Term | Ha |
| ---- | --: |
| SCC (electronic) | −4.766828456418 |
| → electrostatic | 0.001132006161 |
| repulsion | 0.096806468732 |
| dispersion | −0.000454749608 |
| halogen | 0 |
| **Total** | **−4.670476737294** |

Conversion: \(E[\mathrm{eV}] = E[\mathrm{Ha}] \times 27.211386245988\) (ATV
CODATA). Public pin: **−127.09014645 eV**.

**Accuracy gate (product):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | **−127.09014645 eV** | **−4.670476737294 Ha** | **0.05 eV** | Product gate `\|E − pin\| ≤ 0.05` |
| Measured residual | **8.90 meV** (Release, dual-gate after rebase onto `dev` @ `e276a49d`) | \(E_{\mathrm{ATV}}=-127.08124904\) eV | — | vs this xtb transcript |
| SCC residual | \(<\) `scc_tol` | 8 iter (xtb) | `1e-8` e | Product gate |
| Finite energy | required | — | finite, \(\lvert E\rvert>1\) eV | Product gate |

**Tolerance rationale.** The 0.05 eV window is the same kernel-residual floor
as `reg_gfn1-xtb-organic` (D3 CN vs xtb ncoord, Stewart contraction vs
`intcut`, Anderson vs Broyden) on a ~127 eV total. 8.90 meV is inside that
floor — not a missing Hamiltonian term. Polar-molecule residual vs water
(E3 / charge) is not this pin.

Offline re-pin (human only; never ctest):

```bash
# when xtb 6.7.1 is available; see tools/tb_oracle/inputs/xtb/ch4_ipea/
# XTBPATH with param_ipea as GFN1; --gfn 1 --chrg 0 --uhf 0 --sp
# CH4 r_CH=1.09 Å; record TOTAL ENERGY [Eh] into REFERENCE.md
# → eV via 27.211386245988
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/semiempirical/ipea-xtb-small/REFERENCE.md).
Fair 1-thread wall of `reg_ipea-xtb-small` (Release, `OMP_NUM_THREADS=1`).
Gate convention (TB loop): \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}} \le 10\)
unless `perf_exempt`.

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ IPEA Evaluate (forces off) | **0.005** (xtb 6.7.1 clock; `/usr/bin/time` 0.02 s) | **0.01** (Release, 1 thread) | **2.0** | ✓ | Tiny molecule; both walls are I/O-scale |

`perf_exempt`: **false**. Ratio is recorded, not a ctest assertion.

## Limitations

- **Energy-only IPEA** on H/C/N/O. `with_forces = true` and
  `multiplicity != 1` throw on this page. Analytic \(G\) now ships
  ([`ipea-xtb-forces`](ipea-xtb-forces.md)); FD Hessian is
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md).
- Organic CHNO tables only; other IPEA elements are not in-tree
  (`gfn2-beyond-chno` is the beyond-CHNO deepen, on GFN2 first).
- 8.90 meV residual vs xtb is accepted by the 50 meV gate; not
  bit-identical (D3 C6, integral cutoff, mixer).
- `--vip` ΔSCC IP/EA, halogen correction, continuum solvent, PBC, sTDA:
  out of this slug. AES-in-Fock is [`gfn2-aes-fock`](gfn2-aes-fock.md).
- Cross-oracle misuse forbidden: do not re-pin this energy with OpenMOPAC or
  DFTB+, and do not treat IPEA / GFN-xTB as a DFTB3 successor.
- Follow-ons (queue): `ipea-xtb-vip`; then ALPB / beyond-CHNO / sTDA /
  PBC / spin.

Foothold ≠ closed: IPEA CHNO energy is ✓; IPEA \(G\) is ✓
([`ipea-xtb-forces`](ipea-xtb-forces.md)); IPEA FD Hessian is ✓
([`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md)); `--vip`,
solvent, PBC, and spin stay ○. GFN CHNO energy + analytic G + family
FD Hessian + AES-in-Fock + IPEA E+\(G\)+\(H\) is not the GFN product closed.

## References

- Offline pin: `regressions/semiempirical/ipea-xtb-small/REFERENCE.md`
- Oracle twin: `tools/tb_oracle/inputs/xtb/ch4_ipea/` (`mol.xyz`,
  `xtb.out.txt`)
- Oracle rules: `tools/tb_oracle/README.md` (GFN-xTB / IPEA → xtb only)
- Loop design: `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Queue / handoff: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`
- Roadmap: `docs/roadmap/semiempirical.md` §3
- Literature:
  - V. Ásgeirsson, C. Bauer, S. Grimme, *Chem. Sci.* **8**, 4879 (2017),
    DOI [10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)
  - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
    S. Spicher, S. Grimme, *WIREs Comput. Mol. Sci.* **11**, e01493 (2021)
    (xtb)
- Related narrative: [ch17-semiempirical-am1](../ch17-semiempirical-am1.md)
  (NDDO/AM1; different family — OpenMOPAC oracle)
- Siblings: [`ipea-xtb-forces`](ipea-xtb-forces.md) (analytic \(G\)),
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md) (FD Hessian),
  `gfn1-xtb-organic` (same CH₄, GFN1 Hamiltonian),
  `gfn0-beyond-toy` (GFN0 energy), `gfn2-xtb-water` /
  `gfn2-quadrupole-aes` (GFN2 energy)
