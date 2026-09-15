# IP-ADC(2) — H₂O / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ip-adc2-h2o-sto3g` |
| Regression | `regressions/scf/ip-adc2-h2o-sto3g/` |
| Public API | `cpu::scf::RHF::ip_adc2(n_states)` → `IPADC2Result` |
| Twin | PySCF `adc.ADC`, `method='adc(2)'`, `method_type='ip'` |

## Summary

Strict ADC(2) ionization potentials (IP manifold) on a closed-shell RHF
reference via the intermediate-state representation of the one-particle Green’s
function / polarization propagator. This page covers the water / STO-3G product
gate: the lowest IP root (valence). Call path is Frame → `RHF` members → `run`
→ `ip_adc2`; no public `*Config` bag. Offline pin and fair-bench numbers live in
`regressions/scf/ip-adc2-h2o-sto3g/REFERENCE.md`; the regression binary is
`reg_ip-adc2-h2o-sto3g`.

## Method

- **Reference:** restricted HF (singlet, charge 0). Canonical MOs + chemist MO
  ERIs after `RHF::run`.
- **Ground-state doubles:** first-order MP2 amplitudes on the \(N\)-electron
  reference (diagnostic \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) on the result).
- **Strict IP-ADC(2) ISR** (not ADC(2)-x, not ADC(3)):
  - singles = **1h** (occupied) space with second-order intermediate
    \(M_{ij}\) (occupied–occupied);
  - 1h–**2h1p** coupling: first order via \((ia|jk)\);
  - 2h1p diagonal: zeroth order only,
    \(D_{ajk} = \varepsilon_j + \varepsilon_k - \varepsilon_a\).
- **Eigenproblem:** eigenvalues of the sign-flipped ISR matrix are the
  ionization potentials, sorted ascending. Product surface reports IP in eV via
  `units::kHartreeInEv` (CODATA 27.211386245988 eV/Ha).
- **Spin / manifold:** closed-shell IP only (1h / 2h1p). No EE, EA, CVS, or
  UHF/ROHF ADC on this call.
- **Scope exclusions:** no Dyson orbitals / spectroscopic factors, no IP-ADC
  gradients, no ADC(2)-x IP, no ADC(3), no frozen-core IP-ADC, no open-shell.

Literature anchors: Schirmer, *Phys. Rev. A* **26**, 2395 (1982); Schirmer,
Cederbaum & Walter, *Phys. Rev. A* **28**, 1237 (1983). Engine twin follows
PySCF `pyscf.adc` (`method='adc(2)'`, `method_type='ip'`; Loos et al.,
*J. Chem. Phys.* **158**, 044112 (2023)).

## Implementation

Public surface: `cpu::Frame` + concrete `cpu::scf::RHF`. Parameters are members
(`basis`, `tol`, `max_iter`). Result type `cpu::scf::IPADC2Result` is
value-only (`energy_ev`, `e_corr_mp2_ha`) — not a config bag.

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"  // RHF; IPADC2Result via rhf/adc
#include "atomiverse/core/units.h"

cpu::Frame frame(3);
// O(0,0,0.1173); H(0,±0.7572,−0.4692) Å — pin_scf water
// types: 8, 1, 1; box = Box::cubic(30.0)

cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 200;
rhf.run(frame);

const cpu::scf::IPADC2Result adc = rhf.ip_adc2(/*n_states=*/1);
// adc.energy_ev[0]  — IP1 [eV]
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
  cmake --build build --target reg_ip-adc2-h2o-sto3g -j$(nproc)
  ./build/regressions/reg_ip-adc2-h2o-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/adc.h` | `IPADC2Result` public result type |
| `include/atomiverse/cpu/scf/rhf.h` | `RHF::ip_adc2` declaration |
| `src/cpu/scf/adc.cpp` | IP ISR intermediates, eigen-solve, MP2 \(t_2\) |
| `regressions/scf/ip-adc2-h2o-sto3g/reg_ip-adc2-h2o-sto3g.cpp` | product gate |

Requires `-DATV_BUILD_HF=ON`. Call `run(frame)` before `ip_adc2`; throws if no
SCF result, no virtuals, or `n_states` outside \([1,\,n_{\mathrm{occ}}]\).

## Validation (vs PySCF)

Source: [`regressions/scf/ip-adc2-h2o-sto3g/REFERENCE.md`](../../../regressions/scf/ip-adc2-h2o-sto3g/REFERENCE.md)
(offline pin; **never** live PySCF in ctest). Oracle:
`tools/setup_pyscf_env.sh` + `tools/pyscf_oracle/pin_scf.py`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂O, charge 0, singlet |
| Geometry (Å) | O(0,0,0.1173); H(0,±0.7572,−0.4692) (`pin_scf` `water`) |
| Basis | STO-3G |
| Angular | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| ADC | `pyscf.adc.ADC(mf)`, `method='adc(2)'`, `method_type='ip'`, `nroots≥1`, `conv_tol=1e-12` |
| AO / MO | 7 AO; 5 occ, 2 virt spatial |
| PySCF | 2.14.0 (pinned 2026-07-23, venv `~/work/.x86_64`) |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −74.963023138463 Ha | 1e-6 Ha | reference |
| \(E_{\mathrm{corr}}^{\mathrm{MP2}}\) | −0.035545651647 Ha | 1e-6 Ha | diagnostic |
| **IP-ADC(2) IP1** | **0.304180883162 Ha = 8.277183500364 eV** | **1e-4 eV** | **product gate** |
| IP-ADC(2) IP2 | 0.395907981066 Ha ≈ 10.773 eV | — | diagnostic |
| IP-ADC(2) IP3 | 0.604352918337 Ha ≈ 16.445 eV | — | diagnostic |
| IP-ADC(2) IP4 | 1.191953547066 Ha ≈ 32.435 eV | — | diagnostic |
| IP-ADC(2) IP5 | 1.403753901455 Ha ≈ 38.198 eV | — | diagnostic |

Repro pin:

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ip-adc2 \
    --geom water --basis sto-3g --nstates 5 --conv-tol 1e-12
```

On this minimal basis the lowest IP is ~8.28 eV (valence). The gate is the
PySCF strict IP-ADC(2) root above, not an experimental vertical IP. Both sides
of the eV comparison use `units::kHartreeInEv`.

## Performance

Source: fair-bench block in
`regressions/scf/ip-adc2-h2o-sto3g/REFERENCE.md` (2026-07-23, login node,
`OMP_NUM_THREADS=8`, Release). Handoff records ratio **0.191**.
no offline comparison table is present in-tree; the numbers below are
from REFERENCE / handoff, not re-benched here.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| `ip-adc2`, nstates=1 | 2.766 | 0.528 | **0.191** | ✓ | single login-node median (≥3); STO-3G toy |

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method ip-adc2 --geom water --basis sto-3g --nstates 1 --repeat 3 \
  -- -- ./build/regressions/reg_ip-adc2-h2o-sto3g
# recorded: ratio_atv_over_pyscf=0.1907, within_10x=true
```

**Twin caveats:** (1) Both sides include RHF setup on the same geom/basis, but
host load and BLAS threading can move medians. (2) STO-3G / 3-atom is not a
scaling probe — large-basis IP-ADC matvec cost is not characterized. (3) ATV
path is strict IP-ADC(2) only; comparing to IP-ADC(2)-x, ΔSCF, or G0W0 twins is
invalid for the 10× gate.

## Limitations

- Closed-shell RHF IP only; no UHF/ROHF, EE, or EA on `ip_adc2` (use `adc2` /
  `ea_adc2` for those manifolds).
- Strict ADC(2) only — no IP-ADC(2)-x, ADC(3), CVS-IP-ADC, or core-valence
  separation.
- Energies only: no Dyson orbitals, spectroscopic factors, densities, or
  analytic gradients.
- No frozen-core option on this surface; full valence 2h1p space on the tiny
  STO-3G MO set (5 occ / 2 virt).
- Dense MO-ERI / naive ISR path — acceptable for the product pin, not
  production large-molecule IP-ADC.
- Roadmap axes left open: open-shell IP-ADC, properties, IP-ADC(2)-x / ADC(3),
  frozen core, and larger-basis scaling work.

## References

- Offline pin / gates / fair-bench:
  `regressions/scf/ip-adc2-h2o-sto3g/REFERENCE.md`
- Regression consumer:
  `regressions/scf/ip-adc2-h2o-sto3g/reg_ip-adc2-h2o-sto3g.cpp`
- Public API: `include/atomiverse/cpu/scf/adc.h`,
  `include/atomiverse/cpu/scf/rhf.h` (`RHF::ip_adc2`)
- Implementation: `src/cpu/scf/adc.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- J. Schirmer, *Phys. Rev. A* **26**, 2395 (1982).
  DOI [10.1103/PhysRevA.26.2395](https://doi.org/10.1103/PhysRevA.26.2395)
- J. Schirmer, L. S. Cederbaum, O. Walter, *Phys. Rev. A* **28**, 1237 (1983).
  DOI [10.1103/PhysRevA.28.1237](https://doi.org/10.1103/PhysRevA.28.1237)
- PySCF ADC module: Loos et al., *J. Chem. Phys.* **158**, 044112 (2023).
  DOI [10.1063/5.0135290](https://doi.org/10.1063/5.0135290)
