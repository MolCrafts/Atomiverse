# H2O RMP2/STO-3G through the block-assembled dense ERI — e_corr / e_hf pins

Plain (canonical, non-RI) second-order Møller–Plesset on water / STO-3G, driven
through the public `RHF::mp2` path. Canonical RMP2 consumes the full N⁴ AO ERI
tensor materialized by `IntegralWorkspace::eri_dense`; this case is the end-to-end
witness for spec **scf-review-07-posthf-perf item 17** — the dense ERI tensor
is re-assembled from a per-AO scalar `native::eri` loop to a `native::eri_block`
shell-block engine (amortized Boys/R-table + OpenMP, 8-fold-symmetry scatter),
and the correlation energy must reproduce the cited PySCF value unchanged.

## Distinct from `regressions/scf/h2o_mp2_sto3g`

`h2o_mp2_sto3g` uses a different water geometry (O at origin) and only checks a
loose window `e_corr < -1e-4`. This witness uses the PySCF `water` preset
geometry and pins the **exact** PySCF RHF total energy **and** the plain-RMP2
correlation energy (both tighter than the sibling window), so it gates the
block-assembled dense-ERI numbers rather than merely their sign.

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py`.

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) — pin_scf `water` preset |
| Basis | STO-3G (`sto-3g`), 7 AO, Cartesian (no \(d\); `cart=True` is a no-op) |
| Occupied spatial MOs | 5 |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.MP2(mf).run()` (plain RMP2, unit scales) |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23 (RHF + RMP2 channels), venv `~/work/.x86_64` |

Geometry, basis, SCF settings and PySCF version are identical to
`regressions/scf/scs-mp2-h2o-sto3g` (same `water` preset); the plain-RMP2
`e_corr` here is that case's `m2.e_corr` (before the Grimme SCS reweighting).

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py scs-mp2 \
    --geom water --basis sto-3g --conv-tol 1e-12
```

Equivalent explicit script:

```python
import pyscf
from pyscf import gto, scf, mp

mol = gto.M(
    atom="""
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""",
    basis="sto-3g",
    cart=True,
    unit="Angstrom",
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()                 # -74.963023138463 Ha
m2 = mp.MP2(mf).run()
# m2.e_corr -> -0.035545651647 Ha  (plain RMP2 = e_corr_os + e_corr_ss)
# m2.e_tot  -> -74.998568790110 Ha
```

## Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138463 | 1e-7 Ha |
| \(E_{\mathrm{corr}}\) (plain RMP2) | **−0.035545651647** | **1e-7 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) | −74.998568790110 | 1e-6 Ha |

Cross-check: \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) recovers
the pin to machine precision.

## Atomiverse reproduction (PASS today, pre-rewrite scalar `eri_dense`)

Captured 2026-07-24 (GCC 14.3, `-DATV_BUILD_HF=ON`, double precision):

| Quantity | Atomiverse (Ha) | \(\Delta\) vs PySCF pin |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138463072 | ~7e-14 |
| \(E_{\mathrm{corr}}\) | −0.035545651648477 | ~1.5e-12 |
| \(E_{\mathrm{tot}}\) | −74.998568790111548 | ~1.5e-12 |

Both the current scalar `eri_dense` and the block-assembled rewrite must land
inside the pins above (the ERI equivalence judge caps block-vs-scalar drift at
≤1e-12 per element — see `tests/cpu/scf/test_two_electron.cpp`).

## Notes

- Product regression `reg_posthf-eri-block-mp2.cpp` asserts the hard-coded RHF /
  RMP2 pins — no live PySCF, no reading this file at ctest time. Public path:
  ```cpp
  cpu::scf::RHF rhf;
  rhf.basis = cpu::scf::BasisSet::STO3G;
  rhf.tol = 1e-12;
  auto m = rhf.mp2(frame);   // m.e_corr plain RMP2 through the dense ERI tensor
  ```
  Members, not a Config bag (iron law).
- Møller, C.; Plesset, M. S. *Phys. Rev.* **46**, 618 (1934). Values match
  PySCF 2.14.0 `pyscf.mp.MP2`.
- Domain path under `regressions/scf/` matches other Gaussian post-HF cases
  (`h2o_mp2_sto3g`, `scs-mp2-h2o-sto3g`, `frozen-core-mp2-h2o`); slug = queue id
  `posthf-eri-block-mp2`.
