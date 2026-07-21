# Reference — 半定域有效核势(def2-ECP):I₂ 与 AuH 的 RHF 总能量

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md(外部运行、钉定;
PySCF 绝不进 CMake/ctest)。

## 物理背景与文献

- 半定域 ECP 形式与高斯展开 U(r) = U_L + Σ_{l<L} Σ_m |lm⟩(U_l − U_L)⟨lm|,
  U_l = Σ_k d_k r^{n_k−2} e^{−ζ_k r²}:Kahn, Baybutt, Truhlar,
  *J. Chem. Phys.* **65**, 3826 (1976), DOI 10.1063/1.432900。
- type-1/type-2 积分的径向/角向分解(修正球贝塞尔 i_l(kr),实现用带尺度
  形式 M_l(x)=e^{−x}i_l(x)):McMurchie, Davidson, *J. Comput. Phys.* **44**,
  289 (1981), DOI 10.1016/0021-9991(81)90053-X。
- def2-ECP 参数谱系:Weigend, Ahlrichs, *PCCP* **7**, 3297 (2005),
  DOI 10.1039/B508541A。I ECP28MWB:Bergner 等, *Mol. Phys.* **80**, 1431
  (1993), DOI 10.1080/00268979300103121;Au ECP60MWB:Andrae 等,
  *Theor. Chim. Acta* **77**, 123 (1990), DOI 10.1007/BF01114537。
- 物理意义:重元素芯电子化学惰性,替换为 ECP 后价电子数 I₂ 为
  2×(53−28)=50、AuH 为 (79−60)+1=20;标量相对论效应已烘入 MWB
  参数化(因此引擎的 x2c 与 ECP 元素互斥,双重计数门)。

## 钉定值(全部 PySCF 2.14.0,`conv_tol = 1e-12`,球谐 def2 约定)

| 体系 | R [Å] | 基组 / ECP | E_RHF [Ha] | mol.energy_nuc() [Ha] | 回归容差 |
|---|---|---|---|---|---|
| I₂ | 2.666 | def2-SVP / def2-ECP | −593.3161818794323 | 124.05692304013505 | ≤1e-6 |
| AuH | 1.524 | def2-SVP / def2-ECP | −135.34256700791678 | 6.597353679448819 | ≤1e-6 |

核排斥用 Z_val(I:25,Au:79−60=19):I₂ 的 V_nuc = 25²/R = 124.0569…,
与引擎逐位一致(两侧 Bohr 常数同为 0.52917721092 Å)。

## 几何(Å;C++ 回归与 PySCF 钉定逐字相同)

- I₂:I(0, 0, 0); I(0, 0, 2.666);charge = 0(50 价电子,闭壳层)。
- AuH:Au(0, 0, 0); H(0, 0, 1.524);charge = 0(20 价电子,闭壳层)。

## 钉定命令(逐字;venv `~/work/.x86_64/bin/python`,pyscf 2.14.0)

```python
from pyscf import gto, scf

mol = gto.M(atom="I 0 0 0; I 0 0 2.666", basis="def2-svp", ecp="def2-svp")
mf = scf.RHF(mol); mf.conv_tol = 1e-12
mf.kernel()          # E = -593.3161818794323, converged True

mol = gto.M(atom="Au 0 0 0; H 0 0 1.524", basis="def2-svp", ecp="def2-svp")
mf = scf.RHF(mol); mf.conv_tol = 1e-12
mf.kernel()          # E = -135.34256700791678, converged True
```

(PySCF 的 `ecp="def2-svp"` 即 def2-ECP 家族;H 无 ECP 条目,全电子 —
与引擎的逐元素表命中语义一致。)

## 已知偏差(有意为之,已文档化)

无。ECP 解析梯度不在本环范围(`ScfGradient::compute` 对 ECP pack 抛错,
镜像球谐基组先例);MINAO 初猜对 ECP pack 回退 Hcore(参考密度为全电子,
投影到仅价 pack 会重复计入已删除的芯)。
