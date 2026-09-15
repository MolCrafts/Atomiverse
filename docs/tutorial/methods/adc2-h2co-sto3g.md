# ADC(2) — H₂CO / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `adc2-h2co-sto3g` |
| Regression | `regressions/scf/adc2-h2co-sto3g/` |
| Public API | `cpu::scf::RHF::adc2(n_states)` → `ADC2Result` |
| Twin | PySCF `adc.ADC`, `method='adc(2)'`, `method_type='ee'` |

## Summary

Strict ADC(2) (algebraic diagrammatic construction through second order)
computes electronic excitation energies on a closed-shell RHF reference via the
intermediate-state representation of the polarization propagator. This page
covers the formaldehyde / STO-3G product gate: the lowest singlet EE root
(\(n\to\pi^*\)). Call path is Frame → `RHF` members → `run` → `adc2`; no public
`*Config` bag. Offline pin and fair-bench numbers live in
`regressions/scf/adc2-h2co-sto3g/REFERENCE.md`; the regression binary is
`reg_adc2-h2co-sto3g`.

## Method

- **Reference:** restricted HF (singlet, charge 0). Canonical MOs + chemist MO
  ERIs after `RHF::run`.
- **Ground-state doubles:** first-order MP2 amplitudes
  \[
  t_{ij}^{ab}
  = \frac{(ia|bj)}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}.
  \]
- **Strict ADC(2) ISR matrix** (not ADC(2)-x, not ADC(3)):
  - singles block \(M_{ss}\): second-order intermediate;
  - singles–doubles coupling: first order;
  - doubles diagonal: zeroth order only.
- **Eigenproblem:** non-symmetric matvec Davidson on singles+doubles space;
  Ritz values \(\omega\) sorted ascending. Product surface reports \(\omega\) in
  eV via `units::kHartreeInEv` (CODATA 27.211386245988 eV/Ha).
- **Spin / manifold:** closed-shell singlet EE only. No triplet, spin-flip,
  CVS, or UHF/ROHF ADC.
- **Scope exclusions:** no transition moments, no ADC gradients, no ADC(2)-x
  extended doubles, no ADC(3), no frozen-core ADC, no open-shell.

Literature anchors: Schirmer, *Phys. Rev. A* **26**, 2395 (1982); Trofimov &
Schirmer, *J. Phys. B* **28**, 2299 (1995). Engine twin follows PySCF
`pyscf.adc` / `radc_ee` (`method='adc(2)'`, `method_type='ee'`; Loos et al.,
*J. Chem. Phys.* **158**, 044112 (2023)).

## Implementation

Public surface: `cpu::Frame` + concrete `cpu::scf::RHF`. Parameters are members
(`basis`, `tol`, `max_iter`). Result type `cpu::scf::ADC2Result` is
value-only (`energy_ev`, `e_corr_mp2_ha`) — not a config bag.

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"  // RHF; ADC2Result via rhf/adc

cpu::Frame frame(4);
// C(0,0,0); O(0,0,1.21); H(0,±0.943,−0.587) Å — pin_scf h2co
// types: 6, 8, 1, 1; box = Box::cubic(30.0)

cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 200;
rhf.run(frame);

const cpu::scf::ADC2Result adc = rhf.adc2(/*n_states=*/1);
// adc.energy_ev[0]  — S1 [eV]
// adc.e_corr_mp2_ha — MP2 E_corr [Ha] (diagnostic)
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
  cmake --build build --target reg_adc2-h2co-sto3g -j$(nproc)
  ./build/regressions/reg_adc2-h2co-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/adc.h` | `ADC2Result` public result type |
| `include/atomiverse/cpu/scf/rhf.h` | `RHF::adc2` declaration |
| `src/cpu/scf/adc.cpp` | ISR intermediates, matvec Davidson, MP2 \(t_2\) |
| `regressions/scf/adc2-h2co-sto3g/reg_adc2-h2co-sto3g.cpp` | product gate |

Requires `-DATV_BUILD_HF=ON`. Call `run(frame)` before `adc2`; throws if no
SCF result, no virtuals, or `n_states` outside \([1,\,n_{\mathrm{occ}} n_{\mathrm{virt}}]\).

## Validation (vs PySCF)

Source: [`regressions/scf/adc2-h2co-sto3g/REFERENCE.md`](../../../regressions/scf/adc2-h2co-sto3g/REFERENCE.md)
(offline pin; **never** live PySCF in ctest). Oracle:
`tools/setup_pyscf_env.sh` + `tools/pyscf_oracle/pin_scf.py`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂CO, charge 0, singlet |
| Geometry (Å) | C(0,0,0); O(0,0,1.21); H(0,±0.943,−0.587) (`pin_scf` `h2co`) |
| Basis | STO-3G |
| Angular | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| ADC | `pyscf.adc.ADC(mf)`, `method='adc(2)'`, `method_type='ee'`, `conv_tol=1e-12` |
| AO / MO | 12 AO; 8 occ, 4 virt spatial |
| PySCF | 2.14.0 (pinned 2026-07-23, venv `~/work/.x86_64`) |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −112.353990369522 Ha | 1e-6 Ha | reference |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.117214030417 Ha | 1e-6 Ha | diagnostic |
| **ADC(2) S1 \(\omega\)** | **0.166669832218 Ha = 4.535317180049 eV** | **1e-4 eV** | **product gate** |
| ADC(2) S2 \(\omega\) | 0.371358612315 Ha ≈ 10.105 eV | — | diagnostic |
| ADC(2) S3 \(\omega\) | 0.508095678166 Ha ≈ 13.826 eV | — | diagnostic |

Repro pin:

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py adc2 \
    --geom h2co --basis sto-3g --nstates 3 --conv-tol 1e-12
```

Same geometry/basis CIS(D) S1 is 4.545 eV (`reg_cisd-h2co-sto3g`); the ADC
gate is the strict-ADC(2) PySCF root above, not experiment.

## Performance

Source: fair-bench block in
`regressions/scf/adc2-h2co-sto3g/REFERENCE.md` (2026-07-23, login node,
`OMP_NUM_THREADS=8`, Release). Handoff also records ratio **0.267**.
no offline comparison table is present in-tree; the numbers below are
from REFERENCE / worker, not re-benched here.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| `adc2` EE, nstates=1 | 0.247 | 0.066 | **0.267** | ✓ | single login-node median (≥3); STO-3G toy |

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method adc2 --geom h2co --basis sto-3g --nstates 1 --repeat 3 \
  -- -- ./build/regressions/reg_adc2-h2co-sto3g
# recorded: ratio_atv_over_pyscf=0.2669, within_10x=true
```

**Twin caveats:** (1) Both sides include RHF setup on the same geom/basis, but
host load and BLAS threading can move medians. (2) STO-3G / 4-atom is not a
scaling probe — large-basis \(O(N^5)\) ADC matvec cost is not characterized.
(3) ATV path is strict ADC(2) only; comparing to ADC(2)-x or CIS(D) twins is
invalid for the 10× gate.

## Limitations

- Closed-shell RHF singlet EE only; no UHF/ROHF, triplet, or spin-flip ADC.
- Strict ADC(2) only — no ADC(2)-x, ADC(3), CVS-ADC, or core excitations.
- Energies only: no oscillator strengths, densities, or analytic gradients.
- No frozen-core option on this surface; full valence doubles on the tiny
  STO-3G space.
- Dense MO-ERI / naive matvec path — acceptable for the product pin, not
  production large-molecule ADC.
- Roadmap axes left open: open-shell ADC, properties, ADC(2)-x, frozen core,
  and larger-basis scaling work.

## References

- Offline pin / gates / fair-bench:
  `regressions/scf/adc2-h2co-sto3g/REFERENCE.md`
- Regression consumer:
  `regressions/scf/adc2-h2co-sto3g/reg_adc2-h2co-sto3g.cpp`
- Public API: `include/atomiverse/cpu/scf/adc.h`,
  `include/atomiverse/cpu/scf/rhf.h` (`RHF::adc2`)
- Implementation: `src/cpu/scf/adc.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- J. Schirmer, *Phys. Rev. A* **26**, 2395 (1982).
  DOI [10.1103/PhysRevA.26.2395](https://doi.org/10.1103/PhysRevA.26.2395)
- A. B. Trofimov, J. Schirmer, *J. Phys. B* **28**, 2299 (1995).
  DOI [10.1088/0953-4075/28/12/003](https://doi.org/10.1088/0953-4075/28/12/003)
- PySCF ADC module: Loos et al., *J. Chem. Phys.* **158**, 044112 (2023).
  DOI [10.1063/5.0135290](https://doi.org/10.1063/5.0135290)
