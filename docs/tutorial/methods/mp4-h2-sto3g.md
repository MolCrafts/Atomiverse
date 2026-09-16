# RMP4(SDTQ) — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `mp4-h2-sto3g` |
| Regression | `regressions/scf/mp4-h2-sto3g/` |
| Public API | `cpu::scf::RHF::mp4` → `MP4Result` |
| Roadmap | post-HF / MPn (canonical closed-shell) |

## Summary

Canonical closed-shell fourth-order Møller–Plesset MP4(SDTQ) correlation energy
on a converged RHF reference:
\(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}+E^{(4)}\) with
\(E^{(4)}=E_S^{(4)}+E_D^{(4)}+E_T^{(4)}+E_Q^{(4)}\). Use as the next rung above
RMP3 when full CCSD is not required; product gate is total \(E_{\mathrm{corr}}\)
on H₂ / STO-3G. Public path: construct `cpu::scf::RHF`, set members
(`basis`, `tol`, …), call `mp4(frame)` — Frame in, `MP4Result` out; no
`*Config` bag. Offline pin rebuilds intermediate-normalization RSPT through
order 4 on PySCF RHF orbitals (PySCF 2.14 has no `mp.MP4`). Product regression:
`regressions/scf/mp4-h2-sto3g` (`reg_mp4-h2-sto3g`).

## Method

- **Reference:** RHF (closed shell, \(n_\alpha=n_\beta\)). Geometry H₂,
  \(r=0.74\) Å, charge 0, singlet (\(2S=0\)).
- **Order:** Rayleigh–Schrödinger PT through fourth order with the Møller–Plesset
  partitioning \(H_0=\sum_p \varepsilon_p a_p^\dagger a_p\).
- **Second order** (canonical RMP2 doubles):

$$
t_{ij}^{ab}=\frac{\langle ij\|ab\rangle}{D_{ijab}},\qquad
E^{(2)}=\frac14\sum_{ijab} t_{ij}^{ab}\,\langle ij\|ab\rangle,
$$

with \(D_{ijab}=\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b\) and
antisymmetrized spin-orbital integrals \(\langle pq\|rs\rangle\).

- **Third order:** doubles residual \(u_{ijab}\) (ladder + ring and partners);
  \(E^{(3)}=\frac14\sum_{ijab} t_{ij}^{ab}\,u_{ijab}\) — same diagrams as RMP3.
- **Fourth order (SDTQ):** spin-orbital residual channels for singles (\(S\)),
  doubles (\(D\)), triples (\(T\)), and the disconnected-quadruples /
  renormalization piece

$$
E_Q^{(4)}=-E^{(2)}\,\|\Psi^{(1)}\|^2.
$$

  Total:

$$
E^{(4)}=E_S^{(4)}+E_D^{(4)}+E_T^{(4)}+E_Q^{(4)},\qquad
E_{\mathrm{corr}}^{\mathrm{MP4}}=E^{(2)}+E^{(3)}+E^{(4)},\qquad
E_{\mathrm{tot}}=E_{\mathrm{RHF}}+E_{\mathrm{corr}}.
$$

- **H₂ (2e) specialization:** \(n_{\mathrm{occ}}^{\mathrm{SO}}=2\), so true
  triples and connected quadruples vanish; \(E^{(4)}\) is carried by the doubles
  residual channel plus \(E_Q^{(4)}\).
- **Spin:** closed-shell only (spin-orbital residual form; open-shell /
  GHF rejected).
- **Not included:** UMP4 / ROMP4; RI-MP4; SCS/SOS reweightings; analytic MP4
  gradient / Hessian; orbital-optimized MP4; response properties. Frozen-core is
  implementable (`MP4Result::frozen` / member) but the product pin uses
  `frozen=0`.

PySCF 2.14.0 does **not** ship `mp.MP4`. Oracle: rebuild RSPT-4 on `scf.RHF` +
`ao2mo` MO ERIs (`tools/pyscf_oracle/pin_scf.py mp4`); \(E^{(2)}\) matches
`mp.MP2` to machine precision; \(E^{(2)}+E^{(3)}\) matches the offline MP3 pin.

## Implementation

Public surface only: `cpu::Frame` + concrete `cpu::scf::RHF`; parameters are
members. No public `*Config` / factories. Pack-based `mp4_energy` stays under
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
const ATV::cpu::scf::MP4Result mp4 = rhf.mp4(frame);
// mp4.e_corr_mp2, mp4.e_corr_mp3, mp4.e_corr_mp4 [Ha]
// mp4.e_corr_s / e_corr_d / e_corr_t / e_corr_q  (E4 channels)
// mp4.e_corr, mp4.e_tot [Ha]
// rhf.energy() = E_RHF [Ha] after the internal fresh SCF
```

```bash
# configure with HF + regressions (CPU login node sketch)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
cmake --build build --target reg_mp4-h2-sto3g -j"$(nproc)"
export LD_LIBRARY_PATH="${LIBRARY_PATH}:${LD_LIBRARY_PATH}"
./build/regressions/reg_mp4-h2-sto3g
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/mp4.h` | `MP4Result` public POD |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `RHF::mp4(Frame&)` CRTP entry |
| `include/atomiverse/cpu/scf/internal/mp4_pack.h` | private `mp4_energy` |
| `src/cpu/scf/mp4.cpp` | spin-orbital SDTQ residual kernel |
| `src/cpu/scf/mean_field_method.cpp` | public `mp4` (fresh RHF + ERI + call pack) |
| `regressions/scf/mp4-h2-sto3g/reg_mp4-h2-sto3g.cpp` | e2e consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/mp4-h2-sto3g/REFERENCE.md`](../../../regressions/scf/mp4-h2-sto3g/REFERENCE.md)
(offline pin; **never** live PySCF in ctest). Oracle:
`tools/pyscf_oracle/pin_scf.py mp4`; env `tools/setup_pyscf_env.sh`.

| Field | Value |
| ----- | ----- |
| Molecule | H₂, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | H(0, 0, 0); H(0, 0, 0.74) |
| Basis | STO-3G (`sto-3g`) |
| Angular | Cartesian (`cart=True`; STO-3G has no \(d\), cart is a no-op) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | Canonical RMP4(SDTQ), no frozen core |
| AO / MO | 2 / 2 (1 occupied spatial MO) |
| PySCF | 2.14.0 (pinned 2026-07-23) |

| Quantity | Pin (Ha) | Tol | Notes |
| -------- | -------: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 | 1e-7 Ha | reference consistency |
| \(E^{(2)}\) (RMP2) | −0.013138073590 | 1e-6 Ha | diagnostic; matches `mp.MP2` |
| \(E^{(3)}\) | −0.004836072637 | 1e-6 Ha | diagnostic; matches MP3 pin |
| \(E^{(4)}\) | −0.001711078792 | 1e-6 Ha | diagnostic; \(E_S+E_D+E_T+E_Q\) |
| \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}+E^{(4)}\) | **−0.019685225019** | **1e-6 Ha** | product gate |
| \(E_{\mathrm{tot}}^{\mathrm{MP4}}\) | −1.136444532416 | 1e-6 Ha | product gate |

Cross-check (same geom/basis): FCI/CISD correlation −0.020524527092 Ha
(PySCF `fci.FCI` / `ci.CISD`); MP4 recovers ~95.9% of correlation and lies
between MP3 (\(E_{\mathrm{corr}}=-0.017974146227\) Ha) and FCI. On this 2e
system, triples and true quadruples vanish; \(E^{(4)}\) is doubles residual +
renormalization \(E_Q^{(4)}\).

## Performance

Source: `REFERENCE.md` fair-bench block (2026-07-23); worker / queue ratio
**0.031**.
```bash
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="${LIBRARY_PATH}:${LD_LIBRARY_PATH}"
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method mp4 --geom h2 --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_mp4-h2-sto3g
```

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `mp4` (H₂ / STO-3G) | 0.747 | 0.023 | **0.031** | ✓ | Median of 3; `OMP_NUM_THREADS=1`; oracle = det-space RSPT-4 twin on PySCF RHF+ao2mo |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). Twin caveats: same
geometry/basis/method class; single-thread wall; PySCF twin is the full
determinant-space RSPT-4 rebuild (not a shipped `mp.MP4` module) so absolute
\(T_{\mathrm{PySCF}}\) is higher than algebraic ATV — ratio is informative only
against that matched oracle twin (not a large-N scaling claim). ATV side
includes RHF + MP4 + regression asserts. `perf_exempt=false`.

## Limitations

- Energy-only RMP4(SDTQ) on CPU; **no** MP4 nuclear gradient or Hessian.
- Closed-shell only: open-shell / GHF / UMP4 rejected (no silent fallback).
- No RI-MP4; full AO ERI materialization in private workspace — fine for
  STO-3G H₂; scales poorly without RI for larger bases / molecule sizes.
- No SCS or other empirical MPn reweightings at fourth order.
- CUDA MP4 absent.
- H₂ product gate does not exercise the triples residual channel
  (\(n_{\mathrm{occ}}^{\mathrm{SO}}=2\)); multi-electron / larger-basis
  pins remain residual open axes.
- Residual open axes: efficient spatial (non-spin-orbital) fourth-order path;
  frozen-core product cases; CCSD comparison ladder on the same pin.

## References

- Offline pin: `regressions/scf/mp4-h2-sto3g/REFERENCE.md`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Pin helper: `tools/pyscf_oracle/pin_scf.py` (`mp4`)
- A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, Dover (MPn / PT through
  fourth order)
- R. J. Bartlett, *Ann. Rev. Phys. Chem.* **32**, 359 (1981) (many-body PT /
  SDTQ channels)
- C. Møller, M. S. Plesset, *Phys. Rev.* **46**, 618 (1934)
- Related handbook: [mp3-h2-sto3g](mp3-h2-sto3g.md),
  [ccsd-h2-sto3g](ccsd-h2-sto3g.md),
  [scs-mp2-h2o-sto3g](scs-mp2-h2o-sto3g.md)
