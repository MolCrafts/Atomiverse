# O2 triplet UMP2/STO-3G — e_corr + e_tot pins

Open-shell Møller–Plesset second-order (UMP2) on a UHF reference for
triplet O₂ / STO-3G. Axis extension after `rmp2-h2o-sto3g`
(`regressions/scf/h2o_mp2_sto3g`): same observable pair (`e_corr`, `e_tot`)
on an unrestricted open-shell reference.

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py ump2`.

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | O₂, charge 0, spin \(2S=2\) (triplet, \(S=1\)) |
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) — same as `o2_rohf_sto3g` |
| Basis | STO-3G |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart flag is inert but recorded) |
| SCF | UHF, `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.UMP2(mf).run()` (no frozen core) |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ump2 \
    --geom o2 --basis sto-3g --spin 2 --conv-tol 1e-12
```

Equivalent explicit script:

```python
import pyscf
from pyscf import gto, scf, mp

mol = gto.M(
    atom="O 0 0 0; O 0 0 1.2075",
    basis="sto-3g",
    spin=2,
    cart=True,
    unit="Angstrom",
)
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()           # -147.633946785450 Ha
m2 = mp.UMP2(mf).run()
# m2.e_corr  -> -0.096393144130 Ha
# m2.e_tot   -> -147.730339929580 Ha
# mf.spin_square() -> (2.003410857681, ~3.002)  # residual UHF contamination
```

## Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{UHF}}\) | −147.633946785450 | 1e-8 Ha (reference only; not the product gate) |
| \(E_{\mathrm{corr}}\) (UMP2) | −0.096393144130 | **1e-6 Ha** |
| \(E_{\mathrm{tot}}\) = \(E_{\mathrm{UHF}}+E_{\mathrm{corr}}\) | −147.730339929580 | **1e-6 Ha** |

Cross-check vs existing ROHF/UHF pins in `o2_rohf_sto3g/REFERENCE.md`:
UHF total energy matches that table (−147.633946785450 Ha) to the shared
digits; ⟨S²⟩ − 2 ≈ 3.410858×10⁻³ (same contamination).

## Notes

- Product regression `reg_ump2-o2-sto3g.cpp` asserts **only** the hard-coded pins
  above — no live PySCF, no reading this file at ctest time. ROHF density
  warm-start is required so UHF lands on the same minimum as PySCF.
- UMP2 is spin-contaminated when the UHF reference is; the pins are the
  raw (unprojected) UMP2 energies as returned by PySCF `mp.UMP2`.
- Domain path under `regressions/scf/` matches other Gaussian post-HF cases
  (`h2o_mp2_sto3g`, `h2o_ri_mp2`); slug = queue id `ump2-o2-sto3g`.
