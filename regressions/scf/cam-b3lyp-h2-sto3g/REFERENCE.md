# `cam-b3lyp-h2-sto3g`

H₂ / STO-3G Cartesian closed-shell **CAM-B3LYP** energy foothold.

## Geometry

| Atom | x / Å | y / Å | z / Å |
| ---- | ----- | ----- | ----- |
| H    | 0     | 0     | 0     |
| H    | 0     | 0     | 0.74  |

Same as `tools/pyscf_oracle/pin_scf.py` preset `h2`.

## Oracle

- **Code:** PySCF 2.14.0 + libcint (offline; never imported by ctest)
- **Date:** 2026-08-17
- **Command:**

```bash
python3 - <<'PY'
from pyscf import gto, dft
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g',
            unit='Angstrom', cart=True, verbose=0)
mf = dft.RKS(mol)
# Same mix as ATV v1: B3LYP semi-local + Yanai (α,β,μ)=(0.19,0.46,0.33)
# on K[1/r] and K[erf(μ r)/r]. LR_HF is erf (libxc/PySCF convention).
mf.xc = ('0.19*HF + 0.46*LR_HF(0.33) + 0.08*LDA_X + 0.72*GGA_X_B88 '
         '+ 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP')
print(mf.kernel())
PY
```

- **Printed energy:** `-1.3067931936345114` Ha
- **Tolerance:** \(1\times 10^{-6}\) Ha

This is **not** stock libxc `cam-b3lyp` (that uses range-separated ITYH
B88 and prints `-1.1561638628742086` Ha on the same geometry). v1 reuses
the existing B3LYP semi-local kernels and only adds the erf-attenuated
exact-exchange operator.

## Product path

`cpu::scf::RKS{XCType::CAMB3LYP}` + `run(frame)`.
