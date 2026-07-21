# Reference — 弥散函数与阴离子:OH⁻ ε_HOMO 束缚性 + Cl⁻/H2O 总能量

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md(外部运行、钉定;
PySCF 绝不进 CMake/ctest)。

## 物理背景与文献

- 阴离子的多余电子结合松散,紧凑基组缺少慢衰减(弥散)函数,HOMO 呈非
  束缚或勉强束缚伪像;增广弥散函数恢复束缚:Kendall, Dunning, Harrison,
  *J. Chem. Phys.* **96**, 6796 (1992), DOI 10.1063/1.462569(aug-cc 增广
  弥散基组的定义文献)。
- 6-31G* 极化函数:Hariharan, Pople, *Theor. Chim. Acta* **28**, 213
  (1973), DOI 10.1007/BF00533485;6-31G 分裂价基:Ditchfield, Hehre,
  Pople, *J. Chem. Phys.* **54**, 724 (1971), DOI 10.1063/1.1674902。
- 本几何(r(OH) = 0.9640 Å)下的实测(PySCF)符号序:STO-3G 的 ε_HOMO
  为 **+0.2519 Ha(非束缚伪像)**,6-31G* 已勉强束缚(−0.0378 Ha),
  6-31+G* 弥散 sp 使其加深近 3 倍(−0.1036 Ha)。回归据此断言:STO-3G
  伪像为正、6-31+G* 严格低于 6-31G* 且与钉值同号,逐值对钉 ≤1e-6 Ha。

## 钉定值(全部 PySCF 2.14.0,`conv_tol = 1e-12`)

| 体系 | 基组(约定)| E_RHF [Ha] | ε_HOMO [Ha] | 回归容差 |
|---|---|---|---|---|
| OH⁻ | STO-3G (cart) | −74.056328130466 | **+0.251911715713** | ε ≤1e-6 |
| OH⁻ | 6-31G* (cart) | −75.326596547338 | −0.037774390226 | ε ≤1e-6 |
| OH⁻ | 6-31+G* (cart) | −75.376293363750 | −0.103579773338 | E ≤1e-7, ε ≤1e-6 |
| Cl⁻ | 6-31+G* (cart) | −459.539661044387 | (−0.150252200261) | E ≤1e-7 |
| H2O | aug-cc-pVDZ (sph) | −76.041393519987 | (−0.509406243328) | E ≤1e-7 |

约定:Pople 系(6-31G*/6-31+G*、STO-3G)在 PySCF 侧跑 `cart=True`
(笛卡尔 6d,匹配引擎的 per-set AngularConvention::Cartesian);
aug-cc-pVDZ 跑默认 `cart=False`(球谐,匹配 AngularConvention::Spherical)。

## 几何(Å;C++ 回归与 PySCF 钉定逐字相同)

- OH⁻:O(0, 0, 0); H(0, 0, 0.9640);charge = −1。
- Cl⁻:单原子 Cl(0, 0, 0);charge = −1。
- H2O:O(0, 0, 0.1173); H(0, ±0.7572, −0.4692)(与
  regressions/scf/h2o_rhf_dipole / h2o_rhf_def2svp_sph 同一几何)。

## 钉定命令(逐字;venv `~/work/.x86_64/bin/python`,pyscf 2.14.0)

```python
from pyscf import gto, scf

def run(atom, basis, charge, cart):
    mol = gto.M(atom=atom, basis=basis, charge=charge, cart=cart,
                unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-12
    e = mf.kernel()
    homo = max(mf.mo_energy[mf.mo_occ > 0])
    return e, homo

oh = "O 0 0 0; H 0 0 0.9640"
run(oh, "sto-3g",  -1, True)   # E=-74.056328130466, eps=+0.251911715713
run(oh, "6-31g*",  -1, True)   # E=-75.326596547338, eps=-0.037774390226
run(oh, "6-31+g*", -1, True)   # E=-75.376293363750, eps=-0.103579773338
run("Cl 0 0 0", "6-31+g*", -1, True)   # E=-459.539661044387
h2o = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
run(h2o, "aug-cc-pvdz", 0, False)      # E=-76.041393519987
```

## 已知偏差(有意为之,已文档化)

spec 叙事预期"6-31G* 下 ε_HOMO ≥ 0 伪像";实测该伪像出现在 STO-3G
(+0.2519 Ha),6-31G* 在本几何下已勉强束缚(−0.0378 Ha)。回归遵循
acceptance ac-006 的字面判据(逐基组对钉、6-31+G* 严格更低、与钉值同
号),并加钉 STO-3G 行以保留真实的"非束缚→束缚"符号翻转演示。见 spec
Design 的实现审计注记。
