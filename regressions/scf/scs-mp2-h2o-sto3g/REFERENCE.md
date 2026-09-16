# H2O SCS-MP2/STO-3G — e_corr pin

Spin-component-scaled second-order Møller–Plesset (SCS-MP2) on water /
STO-3G. Axis extension after `rmp2-h2o-sto3g` (`regressions/scf/h2o_mp2_sto3g`)
and `frozen-core-mp2-h2o`: same RMP2 doubles sum, reweighted by Grimme’s
opposite-spin / same-spin scales.

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py scs-mp2`.

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) — pin_scf `water` preset |
| Basis | STO-3G (`sto-3g`) |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart is a no-op) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.MP2(mf).run()` then Grimme SCS: \(E_{\mathrm{corr}}^{\mathrm{SCS}} = \tfrac13 E_{\mathrm{SS}} + 1.2\,E_{\mathrm{OS}}\) |
| AO count | 7 |
| Occupied spatial MOs | 5 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

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
# m2.e_corr_ss -> -0.002029831562 Ha
# m2.e_corr_os -> -0.033515820085 Ha
# m2.e_corr    -> -0.035545651647 Ha  (plain RMP2)
# m2.emp2_scs  -> -0.040895594623 Ha  (= e_ss/3 + 1.2*e_os)
# m2.e_tot_scs -> -75.003918733086 Ha
```

## Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138463 | 1e-7 Ha (reference only; not the product gate) |
| \(E_{\mathrm{OS}}\) (unscaled) | −0.033515820085 | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{SS}}\) (unscaled) | −0.002029831562 | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{corr}}\) (plain RMP2) | −0.035545651647 | 1e-6 Ha (diagnostic; OS+SS) |
| \(E_{\mathrm{corr}}^{\mathrm{SCS}}\) | **−0.040895594623** | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}^{\mathrm{SCS}}\) | −75.003918733086 | 1e-6 Ha |

Cross-check: \(E_{\mathrm{corr}}^{\mathrm{SCS}} = \tfrac13 E_{\mathrm{SS}} + 1.2\,E_{\mathrm{OS}}\)
recovers the pin to machine precision; SCS is more negative than plain RMP2
on this system by \(\sim 5.3\,\mathrm{mHa}\) (OS up-weighted, SS down-weighted).

## Notes

- Product regression `reg_scs-mp2-h2o-sto3g.cpp` asserts the hard-coded
  SCS \(E_{\mathrm{corr}}\) (and \(E_{\mathrm{tot}}\)) pins — no live PySCF, no
  reading this file at ctest time. Public path:
  ```cpp
  rhf.mp2_os_scale = 1.2;
  rhf.mp2_ss_scale = 1.0 / 3.0;
  auto r = rhf.mp2(frame);   // r.e_corr is SCS; r.e_corr_os/ss unscaled
  ```
  Members, not a Config bag (iron law).
- Grimme, S. *J. Chem. Phys.* **118**, 9095 (2003). Scales match PySCF’s
  `emp2_scs` property (`e_corr_ss/3 + e_corr_os*1.2`).
- Domain path under `regressions/scf/` matches other Gaussian post-HF cases
  (`h2o_mp2_sto3g`, `frozen-core-mp2-h2o`, `ump2-o2-sto3g`); slug = queue id
  `scs-mp2-h2o-sto3g`.
