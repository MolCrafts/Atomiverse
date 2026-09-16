# H2O frozen-core RMP2/6-31G* — e_corr + e_tot pins

Closed-shell Møller–Plesset second-order (RMP2) with **frozen O 1s** on
water / 6-31G*. Axis extension after `rmp2-h2o-sto3g`
(`regressions/scf/h2o_mp2_sto3g`) and `ump2-o2-sto3g`: same observable pair
(`e_corr`, `e_tot`) with the frozen-core axis (lowest occupied MO excluded
from the correlation sum).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py mp2-fc`.

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) — pin_scf `water` preset |
| Basis | 6-31G* (`6-31g*`) |
| Angular convention | Cartesian (`cart=True`; Pople `*` → Cartesian \(d\), Atomiverse default) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.MP2(mf, frozen=1).run()` — freeze lowest occupied MO (O 1s) |
| AO count | 19 (Cartesian 6-31G* on H₂O) |
| Occupied MOs | 5; correlating occupied = 4 after freeze |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py mp2-fc \
    --geom water --basis 6-31g* --frozen 1 --conv-tol 1e-12
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
    basis="6-31g*",
    cart=True,
    unit="Angstrom",
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()                    # -76.010504988257 Ha
m2 = mp.MP2(mf, frozen=1).run()       # freeze O 1s
# m2.e_corr  -> -0.186119540118 Ha
# m2.e_tot   -> -76.196624528375 Ha
# Full (unfrozen) RMP2 e_corr ≈ -0.188521600925 Ha for comparison
```

## Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −76.010504988257 | 1e-8 Ha (reference only; not the product gate) |
| \(E_{\mathrm{corr}}\) (frozen-core RMP2) | −0.186119540118 | **1e-6 Ha** |
| \(E_{\mathrm{tot}}\) = \(E_{\mathrm{RHF}}+E_{\mathrm{corr}}\) | −76.196624528375 | **1e-6 Ha** |

Cross-check: unfrozen RMP2 on the same geometry/basis/cart/tol gives
\(E_{\mathrm{corr}}\approx-0.188521600925\,\mathrm{Ha}\)
(\(E_{\mathrm{tot}}\approx-76.199026589182\,\mathrm{Ha}\)); freezing O 1s
raises \(E_{\mathrm{corr}}\) by \(\sim 2.4\,\mathrm{mHa}\), as expected for
excluding core correlation on a single first-row heavy atom.

## Notes

- Product regression `reg_frozen-core-mp2-h2o.cpp` asserts **only** the
  hard-coded pins above — no live PySCF, no reading this file at ctest time.
  Public path: `rhf.frozen = 1; auto r = rhf.mp2(frame);` (member, not a
  Config bag).
- PySCF's `frozen=1` freezes the lowest occupied spatial MO (O 1s on water);
  this matches the usual “frozen core” chemistry convention for H₂O (no
  core orbitals on H).
- Domain path under `regressions/scf/` matches other Gaussian post-HF cases
  (`h2o_mp2_sto3g`, `h2o_ri_mp2`, `ump2-o2-sto3g`); slug = queue id
  `frozen-core-mp2-h2o`.
