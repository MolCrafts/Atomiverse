# ADC(2)-x — H₂CO / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `adc2x-h2co-sto3g` |
| Regression | `regressions/scf/adc2x-h2co-sto3g/` |
| Public API | `cpu::scf::RHF::adc2x(n_states)` → `ADC2Result` |
| Twin | PySCF `adc.ADC`, `method='adc(2)-x'`, `method_type='ee'` |

## Summary

Extended ADC(2)-x (algebraic diagrammatic construction with first-order
doubles–doubles) computes electronic excitation energies on a closed-shell RHF
reference via the intermediate-state representation of the polarization
propagator. This page covers the formaldehyde / STO-3G product gate: the lowest
singlet EE root. Call path is Frame → `RHF` members → `run` → `adc2x`; no public
`*Config` bag. Same result type as strict ADC(2) (`ADC2Result`); the method is
chosen by the API name. Offline pin and fair-bench numbers live in
`regressions/scf/adc2x-h2co-sto3g/REFERENCE.md`; the regression binary is
`reg_adc2x-h2co-sto3g`.

## Method

- **Reference:** restricted HF (singlet, charge 0). Canonical MOs + chemist MO
  ERIs after `RHF::run`.
- **Ground-state doubles:** first-order MP2 amplitudes
  \[
  t_{ij}^{ab}
  = \frac{(ia|bj)}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}.
  \]
- **ADC(2)-x ISR matrix** (not strict ADC(2), not ADC(3)):
  - singles block \(M_{ss}\): second-order intermediate (same as strict ADC(2));
  - singles–doubles coupling: first order (same as strict ADC(2));
  - doubles–doubles: **first-order** two-electron terms (vvvv, oovv/ovvo, oooo)
    — the extension that defines ADC(2)-x.
- **Eigenproblem:** non-symmetric matvec Davidson on singles+doubles space;
  Ritz values \(\omega\) sorted ascending. Product surface reports \(\omega\) in
  eV via `units::kHartreeInEv` (CODATA 27.211386245988 eV/Ha).
- **Spin / manifold:** closed-shell singlet EE only. No triplet, spin-flip,
  CVS, IP/EA channel, or UHF/ROHF ADC.
- **Scope exclusions:** no transition moments, no ADC gradients, no ADC(3), no
  frozen-core ADC, no open-shell.

On the same geometry/basis, strict ADC(2) S1 is 4.535 eV
(`reg_adc2-h2co-sto3g`); the ≈−1.22 eV shift is the first-order doubles
correction. The pin is the PySCF ADC(2)-x root, not an experimental band origin.

Literature anchors: Schirmer, *Phys. Rev. A* **26**, 2395 (1982); Trofimov &
Schirmer, *J. Phys. B* **28**, 2299 (1995). Engine twin follows PySCF
`pyscf.adc` (`method='adc(2)-x'`, `method_type='ee'`; Loos et al.,
*J. Chem. Phys.* **158**, 044112 (2023)).

## Implementation

Public surface: `cpu::Frame` + concrete `cpu::scf::RHF`. Parameters are members
(`basis`, `tol`, `max_iter`). Result type `cpu::scf::ADC2Result` is
value-only (`energy_ev`, `e_corr_mp2_ha`) — shared with `adc2`, not a config bag.

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

const cpu::scf::ADC2Result adc = rhf.adc2x(/*n_states=*/1);
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
  cmake --build build --target reg_adc2x-h2co-sto3g -j$(nproc)
  ./build/regressions/reg_adc2x-h2co-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/adc.h` | `ADC2Result` public result type |
| `include/atomiverse/cpu/scf/rhf.h` | `RHF::adc2x` declaration |
| `src/cpu/scf/adc.cpp` | ISR intermediates (extended doubles), matvec Davidson, MP2 \(t_2\) |
| `regressions/scf/adc2x-h2co-sto3g/reg_adc2x-h2co-sto3g.cpp` | product gate |

Requires `-DATV_BUILD_HF=ON`. Call `run(frame)` before `adc2x`; throws if no
SCF result, no virtuals, or `n_states` outside \([1,\,n_{\mathrm{occ}} n_{\mathrm{virt}}]\).

## Validation (vs PySCF)

Source: [`regressions/scf/adc2x-h2co-sto3g/REFERENCE.md`](../../../regressions/scf/adc2x-h2co-sto3g/REFERENCE.md)
(offline pin; **never** live PySCF in ctest). Oracle:
`tools/setup_pyscf_env.sh` + `tools/pyscf_oracle/pin_scf.py`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂CO, charge 0, singlet |
| Geometry (Å) | C(0,0,0); O(0,0,1.21); H(0,±0.943,−0.587) (`pin_scf` `h2co`) |
| Basis | STO-3G |
| Angular | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| ADC | `pyscf.adc.ADC(mf)`, `method='adc(2)-x'`, `method_type='ee'`, `conv_tol=1e-12` |
| AO / MO | 12 AO; 8 occ, 4 virt spatial |
| PySCF | 2.14.0 (pinned 2026-07-23, venv `~/work/.x86_64`) |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −112.353990369522 Ha | 1e-6 Ha | reference |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.117214030417 Ha | 1e-6 Ha | diagnostic |
| **ADC(2)-x S1 \(\omega\)** | **0.121950512434 Ha = 3.318442496747 eV** | **1e-4 eV** | **product gate** |
| ADC(2)-x S2 \(\omega\) | 0.340446186583 Ha ≈ 9.264 eV | — | diagnostic |
| ADC(2)-x S3 \(\omega\) | 0.418603691330 Ha ≈ 11.391 eV | — | diagnostic |

Repro pin:

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py adc2x \
    --geom h2co --basis sto-3g --nstates 3 --conv-tol 1e-12
```

Strict ADC(2) S1 on the same geometry/basis is 4.535 eV
(`reg_adc2-h2co-sto3g`); the product gate is the PySCF ADC(2)-x root above, not
experiment.

## Performance

Source: fair-bench block in
`regressions/scf/adc2x-h2co-sto3g/REFERENCE.md` (2026-07-23, login node,
`OMP_NUM_THREADS=8`, Release). Handoff also records ratio **0.240**.
no offline comparison table is present in-tree; the numbers below are
from REFERENCE / handoff, not re-benched here.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| `adc2x` EE, nstates=1 | 0.467 | 0.112 | **0.240** | ✓ | single login-node median (≥3); STO-3G toy |

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method adc2x --geom h2co --basis sto-3g --nstates 1 --repeat 3 \
  -- -- ./build/regressions/reg_adc2x-h2co-sto3g
# recorded: ratio_atv_over_pyscf=0.2403, within_10x=true
```

**Twin caveats:** (1) Both sides include RHF setup on the same geom/basis, but
host load and BLAS threading can move medians. (2) STO-3G / 4-atom is not a
scaling probe — large-basis \(O(N^5)\) ADC matvec cost is not characterized.
(3) ATV path for this gate is ADC(2)-x only; comparing to strict ADC(2) or
CIS(D) twins is invalid for the 10× gate.

## Limitations

- Closed-shell RHF singlet EE only; no UHF/ROHF, triplet, or spin-flip ADC.
- ADC(2)-x EE only on this surface — IP/EA channels use separate APIs
  (`ip_adc2` / `ea_adc2`, strict ADC(2)); no ADC(3), CVS-ADC, or core
  excitations.
- Energies only: no oscillator strengths, densities, or analytic gradients.
- No frozen-core option on this surface; full valence doubles on the tiny
  STO-3G space.
- Dense MO-ERI / naive matvec path — acceptable for the product pin, not
  production large-molecule ADC.
- Roadmap axes left open: open-shell ADC, properties, frozen core, and
  larger-basis scaling work.

## References

- Offline pin / gates / fair-bench:
  `regressions/scf/adc2x-h2co-sto3g/REFERENCE.md`
- Regression consumer:
  `regressions/scf/adc2x-h2co-sto3g/reg_adc2x-h2co-sto3g.cpp`
- Public API: `include/atomiverse/cpu/scf/adc.h`,
  `include/atomiverse/cpu/scf/rhf.h` (`RHF::adc2x`)
- Implementation: `src/cpu/scf/adc.cpp`
- Strict ADC(2) sibling page: [adc2-h2co-sto3g](adc2-h2co-sto3g.md)
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- J. Schirmer, *Phys. Rev. A* **26**, 2395 (1982).
  DOI [10.1103/PhysRevA.26.2395](https://doi.org/10.1103/PhysRevA.26.2395)
- A. B. Trofimov, J. Schirmer, *J. Phys. B* **28**, 2299 (1995).
  DOI [10.1088/0953-4075/28/12/003](https://doi.org/10.1088/0953-4075/28/12/003)
- PySCF ADC module: Loos et al., *J. Chem. Phys.* **158**, 044112 (2023).
  DOI [10.1063/5.0135290](https://doi.org/10.1063/5.0135290)
