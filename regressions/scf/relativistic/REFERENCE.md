# scf/relativistic

One case; every section below is run by `atv_regressions --case scf/relativistic` and keeps its own oracle record.


## Ar / def2-TZVP RHF x2c-1e

### System

- Atom: Ar (Z = 18) at the origin
- Basis: def2-TZVP (spherical, all-electron; Z ≤ 18 covered)
- Method: RHF, closed shell (18 electrons)

### Oracle

PySCF 2.14.0, contracted AO basis (`xuncontract=False` — **required**;
PySCF's default `xuncontract=True` path is out of scope):

```python
from pyscf import gto, scf
mol = gto.M(atom='Ar 0 0 0', basis='def2-TZVP', verbose=0)
e_nr = scf.RHF(mol).run(conv_tol=1e-12).e_tot
mf = scf.RHF(mol).x2c1e()
mf.with_x2c.xuncontract = False
e_x2c = mf.run(conv_tol=1e-12).e_tot
```

| Quantity | Value [Ha] |
|---|---|
| E_nonrel | −526.8027603325265 |
| E_x2c    | −528.6442231685685 |
| Δ = E_x2c − E_nonrel | −1.8414628360419556 |

### Acceptance

- `|E_x2c − pin| ≤ 1e-5 Ha`
- `E_x2c − E_nonrel < 0` and `|Δ|` within 5% of the pinned PySCF difference

### Literature

- Kutzelnigg, Liu, *J. Chem. Phys.* **123**, 241102 (2005), DOI 10.1063/1.2137315
- Peng, Reiher, *Theor. Chem. Acc.* **131**, 1081 (2012), DOI 10.1007/s00214-011-1081-y


## HCl / 6-31G RHF x2c-1e analytic gradient

Geometry (Å): Cl at the origin, H at (0, 0, 1.35) — stretched past the
equilibrium bond so the force is nonzero.

Oracle (PySCF 2.14.0), from the offline tool:

```
python tools/pyscf_oracle/pin_scf.py rhf --geom hcl --basis 6-31g --grad --x2c --conv-tol 1e-12
```

That is `scf.RHF(mol).x2c1e()` with `with_x2c.xuncontract = False` and
`nuc_grad_method().kernel()` (`pyscf.x2c.sfx2c1e_grad.hcore_grad_generator`).
Cartesian 6-31G (`cart=True`).

| Quantity | Pin |
| --- | --- |
| E_x2c | −461.49265438490323 Ha |
| g(Cl)_z | −0.028498915583206497 Ha/Bohr |
| g(H)_z | +0.028498915583214046 Ha/Bohr |

Transverse components are numerical noise (~1e-12) because the molecule
lies on z. The x2c shift of g_z relative to the non-relativistic gradient
is about 3.5e-4 Ha/Bohr, so a non-relativistic force fails the gate.

Acceptance: |E − pin| ≤ 1e-6 Ha; element-wise |g − pin| ≤ 1e-6 Ha/Bohr;
|Σ_A g_A| ≤ 1e-8 Ha/Bohr.


## H2 / STO-3G and H2O / 6-31G RHF x2c-1e analytic gradient

Classic molecular systems with a nonzero nuclear force. An isolated atom
(Ar / def2-TZVP) is an energy pin only: symmetry makes the force zero.

Oracle (PySCF 2.14.0), from the offline tool:

```
python tools/pyscf_oracle/pin_scf.py rhf --geom h2 --basis sto-3g --grad --x2c --conv-tol 1e-12
python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis 6-31g --grad --x2c --conv-tol 1e-12
```

That is `scf.RHF(mol).x2c1e()` with `with_x2c.xuncontract = False` and
`nuc_grad_method().kernel()` (`pyscf.x2c.sfx2c1e_grad.hcore_grad_generator`,
`gen_sf_hfw`). Cartesian bases (`cart=True`). Geometries are the oracle
presets (Å):

- H2: H(0, 0, 0), H(0, 0, 0.74)
- H2O: O(0, 0, 0.1173), H(0, ±0.7572, −0.4692)

| System | Quantity | Pin |
| --- | --- | --- |
| H2 STO-3G | E_x2c | −1.1167713747002186 Ha |
| H2 STO-3G | g(H1)_z | −0.02767894667735915 Ha/Bohr |
| H2 STO-3G | g(H2)_z | +0.027678946677359373 Ha/Bohr |
| H2O 6-31G | E_x2c | −76.03879282908161 Ha |
| H2O 6-31G | g(O)_z | +0.02372121511333436 Ha/Bohr |
| H2O 6-31G | g(H+y)_y, g(H+y)_z | −0.004279885030643804, −0.011860607556359426 Ha/Bohr |
| H2O 6-31G | g(H−y)_y, g(H−y)_z | +0.004279885031505337, −0.011860607556974934 Ha/Bohr |

Transverse components are numerical noise (≤ 1e-12) and are pinned as 0.
Both molecules have |g| ≈ 0.02 Ha/Bohr, so a vanished force fails the gate.

The H2 x2c−nonrelativistic force shift is ~6.5e-7 Ha/Bohr, below the force
tolerance; the energy shift is −1.206e-5 Ha, so a non-relativistic
Hamiltonian fails the H2 energy gate. On H2O the g(O)_z shift is ~9.7e-5
Ha/Bohr, so a non-relativistic gradient fails the force gate.

Acceptance, same gates as the HCl x2c gradient: |E − pin| ≤ 1e-6 Ha;
element-wise |g − pin| ≤ 1e-6 Ha/Bohr; |Σ_A g_A| ≤ 1e-8 Ha/Bohr.


## Reference — 半定域有效核势(def2-ECP):I₂ 与 AuH 的 RHF 总能量

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md(外部运行、钉定;
PySCF 绝不进 CMake/ctest)。

### 物理背景与文献

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

### 钉定值(全部 PySCF 2.14.0,`conv_tol = 1e-12`,球谐 def2 约定)

| 体系 | R [Å] | 基组 / ECP | E_RHF [Ha] | mol.energy_nuc() [Ha] | 回归容差 |
|---|---|---|---|---|---|
| I₂ | 2.666 | def2-SVP / def2-ECP | −593.3161818794323 | 124.05692304013505 | ≤1e-6 |
| AuH | 1.524 | def2-SVP / def2-ECP | −135.34256700791678 | 6.597353679448819 | ≤1e-6 |

核排斥用 Z_val(I:25,Au:79−60=19):I₂ 的 V_nuc = 25²/R = 124.0569…,
与引擎逐位一致(两侧 Bohr 常数同为 0.52917721092 Å)。

### 几何(Å;C++ 回归与 PySCF 钉定逐字相同)

- I₂:I(0, 0, 0); I(0, 0, 2.666);charge = 0(50 价电子,闭壳层)。
- AuH:Au(0, 0, 0); H(0, 0, 1.524);charge = 0(20 价电子,闭壳层)。

### 钉定命令(逐字;venv `~/work/.x86_64/bin/python`,pyscf 2.14.0)

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

### 已知偏差(有意为之,已文档化)

无。ECP 解析梯度不在本环范围(`ScfGradient::compute` 对 ECP pack 抛错,
镜像球谐基组先例);MINAO 初猜对 ECP pack 回退 Hcore(参考密度为全电子,
投影到仅价 pack 会重复计入已删除的芯)。


## Reference — I₂ RHF/def2-SVP+def2-ECP nuclear gradient

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).
Energy sibling: regressions/scf/relativistic (same I₂ geometry/basis/ECP).

| 字段 | 值 |
|---|---|
| 几何 (Å) | I(0, 0, 0); I(0, 0, 2.666) |
| 基组 / ECP | def2-SVP (spherical, 5d/7f) / def2-ECP (I ECP28MWB, 50 价电子) |
| 收敛 | `conv_tol = 1e-12`, RHF |
| PySCF 版本 | 2.14.0 |
| 命令 | `gto.M(atom="I 0 0 0; I 0 0 2.666", basis="def2-svp", ecp="def2-svp", cart=False)`; `mf=scf.RHF(mol); mf.conv_tol=1e-12; mf.kernel()`; `g=mf.nuc_grad_method().kernel()` |
| **E_RHF** | **−593.3161818794325 Ha** (matches the energy sibling) |
| 对照容差 | 元素级 1e-5 Ha/Bohr;平移不变性 1e-8 Ha/Bohr |

### Pinned analytic gradient dE/dR [Ha/Bohr]

`mf.nuc_grad_method().kernel()` (PySCF has **no** `kick_start`; the entry point
is `.kernel()`). Homonuclear diatomic along z → equal-and-opposite, x/y ~1e-16.

| 原子 | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| I (0) | 0 | 0 | +5.752712116941e-03 |
| I (1) | 0 | 0 | −5.752712116951e-03 |

Σ over atoms ≈ 1.1e-14 Ha/Bohr(平移不变)。`max|∂E/∂R| = 5.752712e-03`。

### 为什么 1e-5 阈值 + 半数值 dU/dR 说明

Atomiverse 的 ECP 梯度是**半数值**的:除 ECP 单电子块 dU/dR 外全部解析;
dU/dR 用中心差分(h=1e-4 Bohr)对 ECP 矩阵求导——U 是每几何构建一次的廉价
单电子矩阵(非 O(N⁴)),且引擎的 U 与 PySCF `ECPscalar` 一致到 ~3e-13,故中心
差分误差(截断 O(h²) + 舍入)约 1e-8,远低于 1e-5 阈值。ECP 之外的解析项
(dS/dT/dV_ne/dERI,均已用 Z_val)与 PySCF 解析梯度一致到 ~1e-9。完整解析
dU/dR 递推是未来升级(见 spec `ponytail:` ceiling)。

文献:Kahn, Baybutt, Truhlar, *J. Chem. Phys.* **65**, 3826 (1976),
DOI 10.1063/1.432900(半定域 ECP 形式);McMurchie, Davidson,
*J. Comput. Phys.* **44**, 289 (1981), DOI 10.1016/0021-9991(81)90053-X
(type-1/2 积分);Weigend & Ahlrichs, *PCCP* **7**, 3297 (2005)
(def2-ECP 谱系);PySCF `pyscf/grad/rhf.py`(数值 oracle)。
