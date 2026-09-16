# H2O SOS-MP2/STO-3G — e_corr pin

Scaled-opposite-spin second-order Møller–Plesset (SOS-MP2) on water /
STO-3G. Axis extension after `scs-mp2-h2o-sto3g`: same RMP2 doubles sum,
reweighted to opposite-spin only with Jung’s \(c_{\mathrm{OS}}=1.3\),
\(c_{\mathrm{SS}}=0\).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py sos-mp2`.

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) — pin_scf `water` preset |
| Basis | STO-3G (`sto-3g`) |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart is a no-op) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.MP2(mf).run()` then SOS: \(E_{\mathrm{corr}}^{\mathrm{SOS}} = 1.3\,E_{\mathrm{OS}} + 0\,E_{\mathrm{SS}}\) |
| AO count | 7 |
| Occupied spatial MOs | 5 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py sos-mp2 \
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
# SOS: 1.3*e_os + 0*e_ss
# e_corr_sos   -> -0.043570566111 Ha
# e_tot_sos    -> -75.006593704574 Ha
```

## Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138463 | 1e-7 Ha (reference only; not the product gate) |
| \(E_{\mathrm{OS}}\) (unscaled) | −0.033515820085 | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{SS}}\) (unscaled) | −0.002029831562 | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{corr}}\) (plain RMP2) | −0.035545651647 | 1e-6 Ha (diagnostic; OS+SS) |
| \(E_{\mathrm{corr}}^{\mathrm{SOS}}\) | **−0.043570566111** | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}^{\mathrm{SOS}}\) | −75.006593704574 | 1e-6 Ha |

Cross-check: \(E_{\mathrm{corr}}^{\mathrm{SOS}} = 1.3\,E_{\mathrm{OS}}\) recovers the
pin to machine precision; SOS drops the same-spin channel and up-weights OS,
so the correlation is more negative than plain RMP2 by \(\sim 8.0\,\mathrm{mHa}\)
on this system.

## Notes

- Product regression `reg_sos-mp2-h2o-sto3g.cpp` asserts the hard-coded
  SOS \(E_{\mathrm{corr}}\) (and \(E_{\mathrm{tot}}\)) pins — no live PySCF, no
  reading this file at ctest time. Public path:
  ```cpp
  rhf.mp2_os_scale = 1.3;
  rhf.mp2_ss_scale = 0.0;
  auto r = rhf.mp2(frame);   // r.e_corr is SOS; r.e_corr_os/ss unscaled
  ```
  Members, not a Config bag (iron law).
- Jung, Y.; Lochan, R. C.; Dutoi, A. D.; Head-Gordon, M. *J. Chem. Phys.*
  **121**, 9793 (2004). PySCF has no `emp2_sos` property; pin scales the OS
  channel offline (`pin_scf.py sos-mp2`).
- Domain path under `regressions/scf/` matches other Gaussian post-HF cases
  (`scs-mp2-h2o-sto3g`, `h2o_mp2_sto3g`); slug = queue id
  `sos-mp2-h2o-sto3g`.
