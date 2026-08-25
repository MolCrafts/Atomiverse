# GFN1-xTB — CH₄ organic energy (H/C/N/O)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gfn1-xtb-organic` |
| Regression | `regressions/semiempirical/gfn1-xtb-organic/` |
| Public API | `cpu::semiempirical::GFN1` + `cpu::Evaluate` |
| Roadmap / queue | `docs/roadmap/semiempirical.md` §3 GFN1-xTB; queue **done** (`reg_gfn1-xtb-organic`) |
| Oracle | **xtb 6.7.1** (family-locked; never OpenMOPAC / DFTB+; **not** a DFTB3 successor) |

## Summary

First-generation Grimme **GFN1-xTB** total energy for isolated methane on the
organic H/C/N/O static tables. Public path is concrete
`cpu::semiempirical::GFN1` (members for SCC / charge / multiplicity; no
`*Config` bag, no `make_*` / `build_*`), bound into `cpu::Evaluate` at
`evaluate::Phase::evaluate`. Native tables only — **no** `libxtb`. Product
gate `reg_gfn1-xtb-organic` asserts

\[
\bigl|E_{\mathrm{ATV}} - (-116.30778398\,\mathrm{eV})\bigr| \le 0.02\,\mathrm{eV}
\]

against an offline **xtb 6.7.1** `--gfn 1` pin (−4.274232224875 Ha). Measured
ATV residual at this geometry is **10.2 meV**. Analytic forces are a
sibling pin (`reg_gfn1-xtb-forces`). Fair wall vs xtb:
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}} = 0.5 \le 10\).

## Method

**GFN1-xTB** (Grimme, Bannwarth, Shushkov, *J. Chem. Theory Comput.* **13**,
1989 (2017); DOI
[10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)). Extended
tight-binding on a Stewart STO-\(n\)G valence basis: EHT \(H^0\) from overlap,
shell-resolved Klopman–Ohno SCC, on-site third-order \(\Gamma q^3\),
exponential pair repulsion, and D3(BJ) with the GFN1 damping set. Hydrogen is
double-ζ (1s + orthogonalized 2s polarization). GFN-xTB is **not** DFTB3++ —
different Hamiltonian, tables, and oracle (**xtb** only).

Closed-shell total energy on the public surface:

\[
E = E_{H^0} + E_{\gamma} + E_{\Gamma} + E_{\mathrm{rep}} + E_{\mathrm{D3}},
\]

with \(E\) written in **eV** to `frame.meta.get(kEnergy)`. Internals work in
Hartree / Bohr.

1. **Basis / overlap.** Per-element Stewart STO-\(n\)G primitives
   (tblite `basis/slater.f90` coefficients). H: 1s (4G, valence) + 2s (3G,
   polarization, Gram–Schmidt against 1s). C/N/O: 2s + 2p (6G). Cartesian
   overlap \(S_{\mu\nu}\) from Gaussian product integrals.
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
   \(k_{\mu\nu}\) from \(k_s=1.85\), \(k_p=2.25\), \(k_{sp}=2.08\),
   \(k_{\mathrm{diff}}=2.85\) (polarization), pair scales \(k_{\mathrm{HH}}=0.96\),
   \(k_{\mathrm{NH}}=1.04\), and \(k_{\mathrm{EN}}=-7\times10^{-3}\) on
   \((\Delta\chi)^2\).
3. **SCC.** Shell charges \(q_i = n_i - n_i^0\) (Mulliken). Klopman–Ohno
   \(\gamma_{ij}\) with \(g_{\exp}=2\) and harmonic-mean shell hardness.
   On-site third-order \(E_{\Gamma}=\frac13\sum_A\Gamma_A q_A^3\). Anderson
   mixing (`mix_beta`, history 6) until \(\max|\Delta q_{\mathrm{shell}}| <
   \texttt{scc\_tol}\). Integer occupation (CH₄ HOMO–LUMO gap ~10 eV; xtb's
   300 K Fermi fill is integer here).
4. **Repulsion.**
   \[
   E_{\mathrm{rep}}
     = \sum_{A<B}
       \frac{Z_A^{\mathrm{eff}} Z_B^{\mathrm{eff}}}{R}
       \exp\bigl(-\sqrt{\alpha_A\alpha_B}\,R^{1.5}\bigr).
   \]
5. **Dispersion.** Molecular D3(BJ), GFN1 set \(s_6=1\), \(s_8=2.4\),
   \(a_1=0.63\), \(a_2=5.0\) (xtb `s9=0`).

**Pin Hamiltonian twin (xtb):**

| Item | Setting |
| ---- | ------- |
| Method | GFN1-xTB (`--gfn 1`) |
| Spin | Closed-shell (`--uhf 0`; ATV `multiplicity = 1`) |
| Charge | 0 (`--chrg 0`) |
| Accuracy | `--acc 0.01` (xtb SCF \(10^{-8}\) Eh) |
| System | CH₄, \(r(\mathrm{C{-}H})=1.09\) Å, tetrahedral |
| Cell | cubic 20 Å on ATV (isolated; xtb molecular) |
| Basis | 12 AOs (C 2s2p + 4×H 1s2s), 10 shells, 8 electrons |
| Oracle | xtb **6.7.1** |

**What this pin includes**

| Item | Value |
| ---- | ----- |
| Method | GFN1-xTB energy only |
| Tables | static `basis_data/gfn1_tables.h` (H/C/N/O) |
| Forces | off in this energy pin (`with_forces = false`; analytic G is `reg_gfn1-xtb-forces`) |
| SCC | `scc_tol = 1e-8`, `max_scc = 80` (regression) |
| Charge | 0 |

**Key exclusions (non-goals of this slug)**

- Analytic nuclear forces (this energy pin; see `reg_gfn1-xtb-forces`).
- Open shell (`multiplicity != 1` throws; queue `spin-gfn-xtb`).
- Halogen-bond correction (xtb term; identically 0 on CH₄; not implemented).
- ALPB/GBSA, CPCM-X/ddCOSMO, PBC, sTDA, IPEA.
- GFN0 / GFN2 Hamiltonians (siblings `gfn0-beyond-toy`, `gfn2-xtb-water`).
- Elements beyond H/C/N/O (unsupported Z throws).
- `libxtb` / runtime `param_gfn1-xtb.txt` parser.
- Live xtb (or any third-party oracle) inside ctest.
- OpenMOPAC / DFTB+ as reference for this family (**forbidden** cross-oracle).
- Public `*Config` bags / `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `GFN1` members; parameters are members /
ctor defaults — iron law):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn1.h"

using namespace ATV;

// CH4 tetrahedral, r(C–H) = 1.09 Å; type = Z.
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
cpu::semiempirical::GFN1 gfn1;
gfn1.with_forces = false;
gfn1.scc_tol = 1e-8;
gfn1.max_scc = 80;
eval.on(evaluate::Phase::evaluate).add_ref(gfn1);
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
  cmake --build build -j$(nproc) --target atomiverse test_cpu_xtb_gfn1 \
        reg_gfn1-xtb-organic
  ./build/regressions/reg_gfn1-xtb-organic
  # or: cmake --build build --target run_reg_gfn1-xtb-organic
  ctest --test-dir build -R "^(test_cpu_xtb_gfn1)$" --output-on-failure
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/semiempirical/gfn1.h` | Public `GFN1` |
| `include/atomiverse/cpu/semiempirical/basis_data/gfn1_tables.h` | Static GFN1 H/C/N/O (tblite `gfn1.f90` / xtb `param_gfn1-xtb.txt`) |
| `src/cpu/semiempirical/gfn1.cpp` | Stewart overlap / EHT+SCC / rep / D3 (private) |
| `tests/cpu/semiempirical/test_xtb_gfn1.cpp` | Unit: tables, Z throw, forces throw, finite \(E\) |
| `regressions/semiempirical/gfn1-xtb-organic/reg_gfn1-xtb-organic.cpp` | Product gate |
| `tools/tb_oracle/inputs/xtb/ch4_gfn1/` | Offline xtb 6.7.1 twin (`mol.xyz` + `xtb.out.txt`) |

## Validation (vs xtb)

Source:
[`regressions/semiempirical/gfn1-xtb-organic/REFERENCE.md`](../../../regressions/semiempirical/gfn1-xtb-organic/REFERENCE.md).

```text
oracle: xtb
oracle_version: 6.7.1
```

Offline pin only — **never** launch xtb from ctest. Twin protocol:
`tools/tb_oracle/README.md`, inputs under
`tools/tb_oracle/inputs/xtb/ch4_gfn1/`.

**Setup:**

| Item | Value |
| ---- | ----- |
| Molecule | CH₄, charge 0, closed shell |
| Geometry [Å] | C at origin; H at \(\pm s,\pm s,\pm s\) with \(s=1.09/\sqrt{3}\) |
| \(r(\mathrm{C{-}H})\) | 1.09 Å |
| Cell | cubic \(L = 20\) Å (ATV isolation) |
| Hamiltonian | GFN1-xTB |
| Tables | in-tree `gfn1_tables.h` (H, C) |
| Provenance | xtb 6.7.1 `TOTAL ENERGY` −4.274232224875 Ha |

xtb 6.7.1 (EasyBuild `xtb/6.7.1-gcc-2025b-eb` on Arrhenius):

```bash
module load xtb/6.7.1-gcc-2025b-eb
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01
```

**xtb energy decomposition (documentation; Ha):**

| Term | Ha |
| ---- | --: |
| SCC (electronic) | −4.340066348203 |
| → electrostatic | 0.000089060808 |
| repulsion | 0.066288872935 |
| dispersion | −0.000454749608 |
| halogen | 0 |
| **Total** | **−4.274232224875** |

Conversion: \(E[\mathrm{eV}] = E[\mathrm{Ha}] \times 27.211386245988\) (ATV
CODATA). Public pin: **−116.30778398 eV**.

**Accuracy gate (product):**

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | **−116.30778398 eV** | **−4.274232224875 Ha** | **0.02 eV** | Product gate `\|E − pin\| ≤ 0.02` |
| Measured residual | **10.2 meV** (0.375 mEh) | — | — | vs this xtb transcript |
| SCC residual | \(<\) `scc_tol` | 9 iter (xtb) | `1e-8` e | Product gate |
| Finite energy | required | — | finite | Product gate |

Tolerance rationale (REFERENCE): 20 meV floor covers remaining kernel
residuals (D3 C6 interpolation vs xtb's internal D3, Stewart primitive
screening vs xtb `intcut`, Anderson vs Broyden) on a ~116 eV total. Tighten
once a later force slug forces the kernels bit-identical.

Offline re-pin (human only; never ctest):

```bash
# when xtb 6.7.1 is available; see tools/tb_oracle/inputs/xtb/ch4_gfn1/
# --gfn 1 --chrg 0 --uhf 0 --acc 0.01, CH4 r_CH=1.09 Å
# record TOTAL ENERGY [Eh] into REFERENCE.md → eV via 27.211386245988
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/semiempirical/gfn1-xtb-organic/REFERENCE.md).
Informative only — **not** asserted in ctest. Gate convention (TB loop):
\(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}} \le 10\) unless `perf_exempt`.

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ GFN1 Evaluate (forces off) | **0.02** (xtb 6.7.1, `OMP_NUM_THREADS=1`) | **0.01** (Release, 1 thread) | **0.5** | ✓ | Tiny molecule; both walls are I/O-scale |

`perf_exempt`: **false**. Ratio is recorded, not a ctest assertion.

## Limitations

- **Energy-only GFN1** on H/C/N/O. `with_forces = true` and
  `multiplicity != 1` throw on this page. GFN0/1/2 analytic \(G\) now
  ship (`gfn0-xtb-forces`, [`gfn1-xtb-forces`](gfn1-xtb-forces.md),
  [`gfn2-xtb-forces`](gfn2-xtb-forces.md)).
- Organic CHNO tables only; other GFN1 elements are not in-tree
  (`gfn2-beyond-chno` is the beyond-CHNO deepen, on GFN2).
- 10.2 meV residual vs xtb is accepted by the 20 meV gate; not
  bit-identical (D3 C6, integral cutoff, mixer).
- Halogen correction, continuum solvent, PBC, sTDA: out of this slug.
  IPEA energy / \(G\) / FD Hessian / VIP/VEA are [`ipea-xtb-small`](ipea-xtb-small.md) /
  [`ipea-xtb-forces`](ipea-xtb-forces.md) /
  [`ipea-xtb-analytic-hessian`](ipea-xtb-analytic-hessian.md) /
  [`ipea-xtb-vip`](ipea-xtb-vip.md); family Hessian is
  [`gfn-xtb-analytic-hessian`](gfn-xtb-analytic-hessian.md); AES-in-Fock
  is [`gfn2-aes-fock`](gfn2-aes-fock.md).
- Cross-oracle misuse forbidden: do not re-pin this energy with OpenMOPAC or
  DFTB+, and do not treat GFN1 as a DFTB3 successor.
- Follow-ons (queue): ALPB / sTDA / PBC / spin.

Foothold ≠ closed: GFN1 organic energy is ✓; solvent, PBC, and spin stay ○.

## References

- Offline pin: `regressions/semiempirical/gfn1-xtb-organic/REFERENCE.md`
- Oracle twin: `tools/tb_oracle/inputs/xtb/ch4_gfn1/`
- Oracle rules: `tools/tb_oracle/README.md` (GFN-xTB → xtb only)
- Loop design: `docs/developer/tb-method-loop.md`,
  `docs/developer/oracle-capability-loop.md`
- Queue / handoff: `docs/developer/tb-method-queue.md`,
  `docs/developer/tb-method-handoff.md`
- Roadmap: `docs/roadmap/semiempirical.md` §3
- Literature:
  - S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory Comput.* **13**,
    1989 (2017), DOI [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)
  - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
    S. Spicher, S. Grimme, *WIREs Comput. Mol. Sci.* **11**, e01493 (2021)
- Related narrative: [ch17-semiempirical-am1](../ch17-semiempirical-am1.md)
  (NDDO/AM1; different family — OpenMOPAC oracle)
- Siblings: `gfn0-beyond-toy` (GFN0 energy), `gfn0-xtb-forces` (GFN0 analytic
  G), `gfn2-xtb-water` / `gfn2-quadrupole-aes` (GFN2 energy)
