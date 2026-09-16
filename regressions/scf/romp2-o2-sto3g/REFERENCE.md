# O2 triplet ROMP2/STO-3G — e_corr + e_tot pins

Restricted open-shell Møller–Plesset second-order (ROMP2) on an ROHF
reference for triplet O₂ / STO-3G. Axis extension after `ump2-o2-sto3g`:
same geometry/basis/spin, but the **spin-pure** ROHF determinant instead of
a spin-contaminated UHF one. Product gate: `e_corr` (and consistency of
`e_tot`).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py romp2`.

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | O₂, charge 0, spin \(2S=2\) (triplet, \(S=1\)) |
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) — same as `o2_rohf_sto3g` |
| Basis | STO-3G |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart flag is inert but recorded) |
| SCF | ROHF (9α, 7β), `conv_tol=1e-12` |
| Post-HF | `pyscf.mp.UMP2(mf_rohf).run()` (PySCF converts ROHF→UHF view then UMP2) |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Method note (PySCF convention)

PySCF has no separate `mp.ROMP2` class in 2.14.0. Calling `mp.MP2` /
`mp.UMP2` on an ROHF object:

1. Converts ROHF → UHF (`addons.convert_to_uhf`): shared spatial \(C\),
   occupations \(n_\alpha / n_\beta\), and **semi-canonical** orbital
   energies \(\varepsilon_i^\sigma = (C^\dagger F_\sigma C)_{ii}\) tagged as
   `mo_energy.mo_ea` / `mo_eb` (not the Roothaan \(F_{\mathrm{eff}}\)
   eigenvalues).
2. Runs raw (unprojected) UMP2 on that UHF-shaped view.

Atomiverse `ROHF::mp2` implements the same: true spin Focks
(`F_alpha_uncoupled_` / `F_beta_uncoupled_`) → diagonal MO energies →
`ump2_energy` kernel. Spin purity of the ROHF reference is preserved
(\(\langle S^2\rangle = S(S+1)\)); only the denominators are spin-resolved.

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py romp2 \
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
mf = scf.ROHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()           # -147.632166990682 Ha
m2 = mp.UMP2(mf).run()
# m2.e_corr  -> -0.096718716392 Ha
# m2.e_tot   -> -147.728885707074 Ha
# mf.spin_square() -> (2.0, 3.0)
```

## Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{ROHF}}\) | −147.632166990682 | 1e-8 Ha (reference only; not the product gate) |
| \(E_{\mathrm{corr}}\) (ROMP2) | −0.096718716392 | **1e-6 Ha** |
| \(E_{\mathrm{tot}}\) = \(E_{\mathrm{ROHF}}+E_{\mathrm{corr}}\) | −147.728885707074 | **1e-6 Ha** |

Cross-check vs `o2_rohf_sto3g/REFERENCE.md`: ROHF total matches
(−147.632166990682 Ha). Cross-check vs `ump2-o2-sto3g`: UHF is lower
(−147.633946785450) and UMP2 \(E_{\mathrm{corr}}\) differs
(−0.096393144130) because the reference orbitals/energies differ.

## Performance

Fair wall-time vs PySCF (same geometry/basis/method class; median of ≥3
runs). Command:

```bash
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method romp2 --geom o2 --basis sto-3g --spin 2 --repeat 3 -- \
    ./build/regressions/reg_romp2-o2-sto3g
```

| Side | Median wall (s) | Notes |
|---|---|---|
| PySCF 2.14.0 | 0.118 | `ROHF` + `UMP2` (OMP_NUM_THREADS=1) |
| Atomiverse | 0.162 | `reg_romp2-o2-sto3g` (OMP_NUM_THREADS=1) |
| **ratio** \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **1.37** | target ≤ 10.0 ✓ |

Recorded 2026-07-23 with `fair_bench.py --method romp2 --repeat 3`.
`within_10x=true`. No focused optimize pass required.

Gate: `within_10x` ⇔ ratio ≤ 10.0. Correct-but-slow is **not** shippable.

## Notes

- Product regression `reg_romp2-o2-sto3g.cpp` asserts **only** the hard-coded
  pins above — no live PySCF, no reading this file at ctest time.
- Domain path under `regressions/scf/`; slug = queue id `romp2-o2-sto3g`.
