# EA-ADC(2) — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ea-adc2-h2-sto3g` |
| Regression | `regressions/scf/ea-adc2-h2-sto3g/` |
| Public API | `cpu::scf::RHF::ea_adc2(n_states)` → `EAADC2Result` |
| Twin | PySCF `adc.ADC`, `method='adc(2)'`, `method_type='ea'` |
| Roadmap | `docs/roadmap/excited-states.md` (EA-ADC(2) energy ✓) |

## Summary

Strict EA-ADC(2) (algebraic diagrammatic construction through second order)
computes **electron-attachment** energies on a closed-shell RHF reference via
the intermediate-state representation of the electron-attachment propagator.
This page covers the H₂ / STO-3G product gate: the lowest attachment root
(EA0). Call path is Frame → `RHF` members → `run` → `ea_adc2`; no public
`*Config` bag. Offline pin and fair-bench numbers live in
`regressions/scf/ea-adc2-h2-sto3g/REFERENCE.md`; the regression binary is
`reg_ea-adc2-h2-sto3g`. On this minimal basis the LUMO is unbound relative to
the N-electron ground state (EA0 ≈ 18.62 eV endothermic); the pin is the PySCF
strict-EA-ADC(2) root, not experiment.

## Method

- **Reference:** restricted HF (singlet, charge 0). Canonical MOs + chemist MO
  ERIs after `RHF::run`.
- **Ground-state doubles:** first-order MP2 amplitudes on the N-electron
  reference (same \(t_{ij}^{ab}\) as EE-ADC(2)); MP2 \(E_{\mathrm{corr}}\) is
  reported as a diagnostic on `EAADC2Result`.
- **Strict EA-ADC(2) ISR** (not ADC(2)-x, not ADC(3)):
  - **Singles** = virtual (1p) space with second-order intermediate \(M_{ab}\);
  - **Doubles** = 2p–1h with **zeroth-order** diagonal only;
  - **Singles–doubles coupling:** first order via \((ia|bc)\).
- **Eigenproblem:** non-symmetric matvec Davidson on the 1p + 2p–1h space;
  Ritz values \(\omega\) sorted ascending. Product surface reports \(\omega\) in
  eV via `units::kHartreeInEv` (CODATA 27.211386245988 eV/Ha). Positive
  \(\omega\) means endothermic attachment.
- **Spin / manifold:** closed-shell EA only. No IP, EE, spin-flip, CVS, or
  UHF/ROHF ADC on this surface.
- **Scope exclusions:** no Dyson orbitals / spectroscopic factors, no
  EA-ADC gradients, no ADC(2)-x extended doubles, no ADC(3), no frozen-core
  EA-ADC, no open-shell.

Literature anchors: Schirmer, *Phys. Rev. A* **26**, 2395 (1982). Engine twin
follows PySCF `pyscf.adc` / `radc_ea` (`method='adc(2)'`, `method_type='ea'`;
Loos et al., *J. Chem. Phys.* **158**, 044112 (2023)).

## Implementation

Public surface: `cpu::Frame` + concrete `cpu::scf::RHF`. Parameters are members
(`basis`, `tol`, `max_iter`). Result type `cpu::scf::EAADC2Result` is
value-only (`energy_ev`, `e_corr_mp2_ha`) — not a config bag.

```cpp
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"  // RHF; EAADC2Result via rhf/adc

cpu::Frame frame(2);
// H(0,0,0); H(0,0,0.74) Å — pin_scf h2
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_type)[0] = 1;
frame.span(kAtom_x)[1] = 0.0;
frame.span(kAtom_y)[1] = 0.0;
frame.span(kAtom_z)[1] = 0.74;
frame.span(kAtom_type)[1] = 1;
frame.box = cpu::Box::cubic(20.0);

cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 200;
rhf.run(frame);

const cpu::scf::EAADC2Result ea = rhf.ea_adc2(/*n_states=*/1);
// ea.energy_ev[0]  — EA0 [eV]
// ea.e_corr_mp2_ha — MP2 E_corr [Ha] (diagnostic)
```

Build and run the regression (HF + regressions required):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_ea-adc2-h2-sto3g -j$(nproc)
  ./build/regressions/reg_ea-adc2-h2-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/adc.h` | `EAADC2Result` public result type |
| `include/atomiverse/cpu/scf/rhf.h` | `RHF::ea_adc2` declaration |
| `src/cpu/scf/adc.cpp` | EA ISR \(M_{ab}\), 2p–1h matvec Davidson, MP2 \(t_2\) |
| `regressions/scf/ea-adc2-h2-sto3g/reg_ea-adc2-h2-sto3g.cpp` | product gate |

Requires `-DATV_BUILD_HF=ON`. Call `run(frame)` before `ea_adc2`; throws if no
SCF result, no virtuals, or `n_states` outside
\([1,\,n_{\mathrm{virt}} + n_{\mathrm{occ}} n_{\mathrm{virt}}^2]\).

## Validation (vs PySCF)

Source: [`regressions/scf/ea-adc2-h2-sto3g/REFERENCE.md`](../../../regressions/scf/ea-adc2-h2-sto3g/REFERENCE.md)
(offline pin; **never** live PySCF in ctest). Oracle:
`tools/setup_pyscf_env.sh` + `tools/pyscf_oracle/pin_scf.py`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) (`pin_scf` `h2`) |
| Basis | STO-3G |
| Angular | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| ADC | `pyscf.adc.ADC(mf)`, `method='adc(2)'`, `method_type='ea'`, `conv_tol=1e-12` |
| AO / MO | 2 AO; 1 occ, 1 virt spatial |
| PySCF | 2.14.0 (pinned 2026-07-23, venv `~/work/.x86_64`) |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 Ha | 1e-6 Ha | reference |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.013138073590 Ha | 1e-6 Ha | diagnostic |
| **EA-ADC(2) EA0 \(\omega\)** | **0.684281565504 Ha = 18.620249979933 eV** | **1e-4 eV** | **product gate** |
| EA-ADC(2) EA1 \(\omega\) | 1.920840843660 Ha ≈ 52.269 eV | — | diagnostic |

Repro pin:

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ea-adc2 \
    --geom h2 --basis sto-3g --nstates 2 --conv-tol 1e-12
```

CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\)
(`units::kHartreeInEv`) — both sides of the eV gate use the same factor. The
product gate is the PySCF strict-EA-ADC(2) root above, not experimental EA.

## Performance

Source: fair-bench block in
`regressions/scf/ea-adc2-h2-sto3g/REFERENCE.md` (2026-07-23, login node,
`OMP_NUM_THREADS=8`, Release). Handoff records ratio **0.213**.
no offline comparison table is present in-tree; the numbers below are
from REFERENCE / handoff, not re-benched here.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| `ea-adc2`, nstates=1 | 0.735 | 0.157 | **0.213** | ✓ | single login-node median (≥3); STO-3G toy |

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method ea-adc2 --geom h2 --basis sto-3g --nstates 1 --repeat 3 \
  ./build/regressions/reg_ea-adc2-h2-sto3g
# recorded: ratio_atv_over_pyscf=0.2135, within_10x=true
```

**Twin caveats:** (1) Both sides include RHF setup on the same geom/basis, but
host load and BLAS threading can move medians. (2) H₂ / STO-3G (2 AO) is not a
scaling probe — larger-basis EA-ADC \(O(N^5)\) matvec cost is not characterized.
(3) ATV path is strict EA-ADC(2) only; comparing to EA-ADC(2)-x, EE-ADC, or
EOM-EA-CCSD twins is invalid for the 10× gate.

## Limitations

- Closed-shell RHF EA only; no UHF/ROHF, IP/EE on this call path (use
  `ip_adc2` / `adc2` for those manifolds).
- Strict ADC(2) only — no EA-ADC(2)-x, ADC(3), CVS-ADC, or continuum / complex
  absorbing potential treatments.
- Energies only: no Dyson amplitudes, spectroscopic factors, or analytic
  gradients.
- No frozen-core option on this surface; full valence doubles on the tiny
  STO-3G space (1 occ × 1 virt).
- Dense MO-ERI / naive matvec path — acceptable for the product pin, not
  production large-molecule EA-ADC.
- STO-3G attachment roots for H₂ are unbound and basis-limited; do not treat
  the 18.62 eV pin as a physical electron affinity.
- Roadmap axes left open: properties, EA-ADC(2)-x, frozen core, open-shell,
  and larger-basis scaling work (`docs/roadmap/excited-states.md`: energy ✓,
  properties/gradients ○).

## References

- Offline pin / gates / fair-bench:
  `regressions/scf/ea-adc2-h2-sto3g/REFERENCE.md`
- Regression consumer:
  `regressions/scf/ea-adc2-h2-sto3g/reg_ea-adc2-h2-sto3g.cpp`
- Public API: `include/atomiverse/cpu/scf/adc.h`,
  `include/atomiverse/cpu/scf/rhf.h` (`RHF::ea_adc2`)
- Implementation: `src/cpu/scf/adc.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Related EE pin: `docs/tutorial/methods/adc2-h2co-sto3g.md`
- J. Schirmer, *Phys. Rev. A* **26**, 2395 (1982).
  DOI [10.1103/PhysRevA.26.2395](https://doi.org/10.1103/PhysRevA.26.2395)
- PySCF ADC module: Loos et al., *J. Chem. Phys.* **158**, 044112 (2023).
  DOI [10.1063/5.0135290](https://doi.org/10.1063/5.0135290)
