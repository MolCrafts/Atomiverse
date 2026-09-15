# ROMP2 — triplet O₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `romp2-o2-sto3g` |
| Regression | `regressions/scf/romp2-o2-sto3g/` |
| Public API | `cpu::scf::ROHF::mp2` → `MP2Result` |
| Roadmap | [post-hf.md](../../roadmap/post-hf.md) — Restricted open-shell MP2 |

## Summary

ROMP2 is second-order Møller–Plesset correlation on a **spin-pure ROHF**
reference: shared spatial orbitals, occupations \(n_\alpha / n_\beta\), and
semi-canonical spin-Fock orbital energies in the UMP2 doubles sum (PySCF
`mp.UMP2(ROHF)` convention). Use it when open-shell spin purity of the
reference matters (triplet O₂ here) and a full UHF/UMP2 path is not required.
Public path: construct `cpu::scf::ROHF`, set members, call `mp2(frame)` —
Frame in, `MP2Result` out; no `*Config` bag. Product gate lives at
`regressions/scf/romp2-o2-sto3g` (`reg_romp2-o2-sto3g`).

## Method

- **Reference:** ROHF (Guest–Saunders / PySCF canonical Roothaan \(F_{\mathrm{eff}}\)).
  Shared spatial \(C\); occupations \(n_\alpha=9\), \(n_\beta=7\) for triplet O₂
  (charge 0, \(2S=2\)).
- **Spin purity:** \(\langle S^2\rangle = S(S+1)\) on the ROHF determinant
  (here \(S=1 \Rightarrow \langle S^2\rangle=2\)).
- **Denominators:** not Roothaan eigenvalues. Semi-canonical
  \(\varepsilon_i^\sigma = (C^\dagger F_\sigma C)_{ii}\) from the **true** spin
  Focks \(F_\alpha\), \(F_\beta\) (Atomiverse: `F_alpha_uncoupled_` /
  `F_beta_uncoupled_`).
- **Correlation:** unprojected UMP2 doubles on that UHF-shaped view of the ROHF
  MOs:

$$
E_{\mathrm{corr}}^{\mathrm{ROMP2}} = E_{\mathrm{SS}} + E_{\mathrm{OS}}, \qquad
E_{\mathrm{tot}} = E_{\mathrm{ROHF}} + E_{\mathrm{corr}}.
$$

- **Not included:** ROMP2 analytic gradient/Hessian; frozen-core ROMP2;
  RI-ROMP2; spin-projected MP2; orbital-optimized ROMP2.

PySCF 2.14.0 has no separate `mp.ROMP2` class: `mp.UMP2(mf_rohf)` converts
ROHF→UHF (`addons.convert_to_uhf`) then runs raw UMP2. Atomiverse
`ROHF::mp2` mirrors that path via the private `romp2_energy` kernel.

## Implementation

Public surface only: `cpu::Frame` + concrete `cpu::scf::ROHF`; parameters are
constructor args / members. No public `*Config` / factories.

```cpp
#include <atomiverse/cpu/frame/frame.h>
#include <atomiverse/cpu/scf/methods.h>

// O₂, r = 1.2075 Å, triplet (2S = 2)
ATV::cpu::Frame frame(2);
// … Z=8, positions, tags, box …
ATV::cpu::scf::ROHF rohf(9, 7);
rohf.basis = ATV::cpu::scf::BasisSet::STO3G;
rohf.tol = 1e-12;
rohf.max_iter = 200;
const ATV::cpu::scf::MP2Result mp2 = rohf.mp2(frame);
// mp2.e_corr, mp2.e_tot [Ha]; rohf.energy() = E_ROHF [Ha]
```

```bash
# configure with HF + regressions (CPU login node sketch)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
cmake --build build --target reg_romp2-o2-sto3g -j"$(nproc)"
export LD_LIBRARY_PATH="${LIBRARY_PATH}:${LD_LIBRARY_PATH}"
./build/regressions/reg_romp2-o2-sto3g
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/rohf.h` | concrete ROHF method |
| `include/atomiverse/cpu/scf/mp2.h` | `MP2Result`; documents `ROHF::mp2` |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `mp2(Frame&)` CRTP entry |
| `include/atomiverse/cpu/scf/internal/mp2_pack.h` | private `romp2_energy` |
| `regressions/scf/romp2-o2-sto3g/reg_romp2-o2-sto3g.cpp` | e2e consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/romp2-o2-sto3g/REFERENCE.md`](../../../regressions/scf/romp2-o2-sto3g/REFERENCE.md)
(offline pin; **never** live PySCF in ctest). Oracle:
`tools/pyscf_oracle/pin_scf.py romp2`; env `tools/setup_pyscf_env.sh`.

| Field | Value |
| ----- | ----- |
| Molecule | O₂, charge 0, spin \(2S=2\) (triplet, \(S=1\)) |
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; STO-3G has no \(d\)) |
| SCF | ROHF (9α, 7β), `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.UMP2(mf_rohf).run()` |
| PySCF | 2.14.0 (pinned 2026-07-23) |

| Quantity | Pin (Ha) | Tol | Notes |
| -------- | -------: | --- | ----- |
| \(E_{\mathrm{ROHF}}\) | −147.632166990682 | 1e-8 Ha | reference consistency (not product gate alone) |
| \(E_{\mathrm{corr}}\) (ROMP2) | −0.096718716392 | **1e-6 Ha** | product gate |
| \(E_{\mathrm{tot}}\) | −147.728885707074 | **1e-6 Ha** | product gate; \(= E_{\mathrm{ROHF}}+E_{\mathrm{corr}}\) |

Cross-check: ROHF matches `o2_rohf_sto3g`; UMP2 on UHF
(`ump2-o2-sto3g`) differs (\(E_{\mathrm{UHF}}\) lower, different \(E_{\mathrm{corr}}\))
because reference orbitals/energies differ.

## Performance

Source: `REFERENCE.md` fair-bench block (2026-07-23); worker / handoff
ratio **1.37**.
```bash
export LD_LIBRARY_PATH="${LIBRARY_PATH}:${LD_LIBRARY_PATH}"
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method romp2 --geom o2 --basis sto-3g --spin 2 --repeat 3 -- \
    ./build/regressions/reg_romp2-o2-sto3g
```

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `romp2` (O₂ / STO-3G) | 0.118 | 0.162 | **1.37** | ✓ | Median ≥3; `OMP_NUM_THREADS=1`; STO-3G wall is noise-dominated |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). Twin caveats: same
geometry/basis/method class; single-thread wall; absolute times are tiny —
ratio is informative only against the matched PySCF twin, not as a large-N
scaling claim.

## Limitations

- Energy-only ROMP2 on CPU; **no** UMP2/ROMP2 nuclear gradients or Hessians.
- No frozen-core ROMP2 (`frozen > 0` rejected); no RI-ROMP2.
- Canonical path materializes the full AO ERI tensor in private workspace —
  fine for STO-3G O₂; scales poorly without RI for larger bases.
- CUDA ROMP2 absent (`docs/roadmap/post-hf.md` CUDA ○).
- Residual open axes: projected/spin-adapted open-shell MP2 variants;
  OO-MP2; ROMP2 response properties.

## References

- Offline pin: `regressions/scf/romp2-o2-sto3g/REFERENCE.md`
- Roadmap: `docs/roadmap/post-hf.md` (ROMP2 row)
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Related handbook: [ump2-o2-sto3g](ump2-o2-sto3g.md)
- C. Møller, M. S. Plesset, *Phys. Rev.* **46**, 618 (1934)
- PySCF ROHF→UMP2 path: `pyscf.mp.UMP2` on `scf.ROHF` (v2.14.0)
- Guest–Saunders ROHF projectors / constrained-UHF view: see ROHF stage docs
  (`include/atomiverse/cpu/scf/stages/rohf_couple_stage.h`)
