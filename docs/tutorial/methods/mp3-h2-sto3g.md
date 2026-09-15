# RMP3 — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `mp3-h2-sto3g` |
| Regression | `regressions/scf/mp3-h2-sto3g/` |
| Public API | `cpu::scf::RHF::mp3` → `MP3Result` |
| Roadmap | post-HF / MPn (canonical closed-shell) |

## Summary

Canonical closed-shell third-order Møller–Plesset (RMP3) correlation energy on a
converged RHF reference: \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}\). Use for a
cheap post-HF step beyond MP2 when a full CCSD is not required; product gate is
total \(E_{\mathrm{corr}}\) on H₂ / STO-3G. Public path: construct
`cpu::scf::RHF`, set members (`basis`, `tol`, …), call `mp3(frame)` — Frame in,
`MP3Result` out; no `*Config` bag. Offline pin rebuilds Szabo–Ostlund /
Psi4NumPy diagrams on PySCF RHF orbitals (PySCF 2.14 has no `mp.MP3`). Product
regression: `regressions/scf/mp3-h2-sto3g` (`reg_mp3-h2-sto3g`).

## Method

- **Reference:** RHF (closed shell, \(n_\alpha=n_\beta\)). Geometry H₂,
  \(r=0.74\) Å, charge 0, singlet (\(2S=0\)).
- **Order:** Rayleigh–Schrödinger PT through third order with the Møller–Plesset
  partitioning \(H_0=\sum_p \varepsilon_p a_p^\dagger a_p\).
- **Second order** (same as canonical RMP2 doubles):

$$
t_{ij}^{ab}=\frac{\langle ij\|ab\rangle}{D_{ijab}},\qquad
E^{(2)}=\frac14\sum_{ijab} t_{ij}^{ab}\,\langle ij\|ab\rangle,
$$

with \(D_{ijab}=\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b\) and
antisymmetrized spin-orbital integrals \(\langle pq\|rs\rangle\).

- **Third order:** residual \(u_{ijab}\) from ladder + ring (and partner)
  diagrams; \(E^{(3)}=\frac14\sum_{ijab} t_{ij}^{ab}\,u_{ijab}\). Spatial
  reduction of the same diagrams is the Psi4NumPy / Szabo–Ostlund 12-diagram
  sum used by the offline pin.
- **Total:**

$$
E_{\mathrm{corr}}^{\mathrm{MP3}}=E^{(2)}+E^{(3)},\qquad
E_{\mathrm{tot}}=E_{\mathrm{RHF}}+E_{\mathrm{corr}}.
$$

- **Spin:** closed-shell only (spin-orbital residual form; open-shell /
  GHF rejected).
- **Not included:** UMP3 / ROMP3; RI-MP3; SCS-MP3; analytic MP3 gradient /
  Hessian; orbital-optimized MP3; response properties. Frozen-core is
  implementable (`MP3Result::frozen` / member) but the product pin uses
  `frozen=0`.

PySCF 2.14.0 does **not** ship `mp.MP3`. Oracle: rebuild diagrams on
`scf.RHF` + `ao2mo` MO ERIs (`tools/pyscf_oracle/pin_scf.py mp3`); \(E^{(2)}\)
cross-checks `mp.MP2` to machine precision.

## Implementation

Public surface only: `cpu::Frame` + concrete `cpu::scf::RHF`; parameters are
members. No public `*Config` / factories. Pack-based `mp3_energy` stays under
`scf/internal/`.

```cpp
#include <atomiverse/cpu/frame/frame.h>
#include <atomiverse/cpu/scf/methods.h>

// H₂, r = 0.74 Å, singlet
ATV::cpu::Frame frame(2);
// … Z=1, positions, box …
ATV::cpu::scf::RHF rhf;
rhf.basis = ATV::cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 100;
const ATV::cpu::scf::MP3Result mp3 = rhf.mp3(frame);
// mp3.e_corr_mp2, mp3.e_corr_mp3, mp3.e_corr, mp3.e_tot [Ha]
// rhf.energy() = E_RHF [Ha] after the internal fresh SCF
```

```bash
# configure with HF + regressions (CPU login node sketch)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
cmake --build build --target reg_mp3-h2-sto3g -j"$(nproc)"
export LD_LIBRARY_PATH="${LIBRARY_PATH}:${LD_LIBRARY_PATH}"
./build/regressions/reg_mp3-h2-sto3g
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/mp3.h` | `MP3Result` public POD |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `RHF::mp3(Frame&)` CRTP entry |
| `include/atomiverse/cpu/scf/internal/mp3_pack.h` | private `mp3_energy` |
| `src/cpu/scf/mp3.cpp` | spin-orbital doubles residual kernel |
| `src/cpu/scf/mean_field_method.cpp` | public `mp3` (fresh RHF + ERI + call pack) |
| `regressions/scf/mp3-h2-sto3g/reg_mp3-h2-sto3g.cpp` | e2e consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/mp3-h2-sto3g/REFERENCE.md`](../../../regressions/scf/mp3-h2-sto3g/REFERENCE.md)
(offline pin; **never** live PySCF in ctest). Oracle:
`tools/pyscf_oracle/pin_scf.py mp3`; env `tools/setup_pyscf_env.sh`.

| Field | Value |
| ----- | ----- |
| Molecule | H₂, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | H(0, 0, 0); H(0, 0, 0.74) |
| Basis | STO-3G (`sto-3g`) |
| Angular | Cartesian (`cart=True`; STO-3G has no \(d\), cart is a no-op) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | Canonical RMP3, no frozen core |
| AO / MO | 2 / 2 (1 occupied spatial MO) |
| PySCF | 2.14.0 (pinned 2026-07-23) |

| Quantity | Pin (Ha) | Tol | Notes |
| -------- | -------: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 | 1e-7 Ha | reference consistency |
| \(E^{(2)}\) (RMP2) | −0.013138073590 | 1e-6 Ha | diagnostic; matches `mp.MP2` |
| \(E^{(3)}\) | −0.004836072637 | 1e-6 Ha | diagnostic |
| \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}\) | **−0.017974146227** | **1e-6 Ha** | product gate |
| \(E_{\mathrm{tot}}^{\mathrm{MP3}}\) | −1.134733453623 | 1e-6 Ha | product gate |

Cross-check (same geom/basis): FCI/CISD correlation −0.020524527092 Ha
(PySCF `fci.FCI` / `ci.CISD`); MP3 recovers ~87.6% of correlation and lies
between MP2 and FCI.

## Performance

Source: `REFERENCE.md` fair-bench block (2026-07-23); worker / queue ratio
**0.774**.
```bash
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="${LIBRARY_PATH}:${LD_LIBRARY_PATH}"
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method mp3 --geom h2 --basis sto-3g --repeat 5 -- \
    ./build/regressions/reg_mp3-h2-sto3g
```

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `mp3` (H₂ / STO-3G) | 0.01602 | 0.01241 | **0.774** | ✓ | Median of 5; `OMP_NUM_THREADS=1`; oracle = RHF + diagram sum |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). Twin caveats: same
geometry/basis/method class; single-thread wall; absolute times are sub-20 ms —
noise-dominated; ratio is informative only against the matched PySCF oracle
twin (not a large-N scaling claim). ATV side includes RHF + MP3 + regression
asserts.

## Limitations

- Energy-only RMP3 on CPU; **no** MP3 nuclear gradient or Hessian.
- Closed-shell only: open-shell / GHF / UMP3 rejected (no silent fallback).
- No RI-MP3; full AO ERI materialization in private workspace — fine for
  STO-3G H₂; scales poorly without RI for larger bases / molecule sizes.
- No SCS or other empirical MPn reweightings at third order.
- CUDA MP3 absent.
- Residual open axes: efficient spatial (non-spin-orbital) third-order path;
  frozen-core product cases; CCSD comparison ladder on the same pin; larger
  basis / multi-electron systems.

## References

- Offline pin: `regressions/scf/mp3-h2-sto3g/REFERENCE.md`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Pin helper: `tools/pyscf_oracle/pin_scf.py` (`mp3`)
- A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, Dover; Table 6.2 /
  pp. 364–368 (third-order diagrams)
- Psi4NumPy `Moller-Plesset/MP3.py` (spatial diagram reduction used by pin)
- C. Møller, M. S. Plesset, *Phys. Rev.* **46**, 618 (1934)
- Related handbook: [scs-mp2-h2o-sto3g](scs-mp2-h2o-sto3g.md),
  [frozen-core-mp2-h2o](frozen-core-mp2-h2o.md)
