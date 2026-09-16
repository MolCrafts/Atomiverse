# Reference — H₂O RHF/STO-3G dipole-derivative tensor (atomic polar tensor)

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| 基组 | STO-3G (Cartesian) |
| 收敛 | `conv_tol = 1e-13`, RHF |
| PySCF 版本 | 2.14.0 |
| oracle | **finite difference** of `mf.dip_moment(unit='au')`, central, h = 1e-3 Bohr |
| 对照容差 | 元素级 1e-4 au(受 FD oracle 截断误差限制,非引擎);平移不变性 1e-6 au |

## 为什么 oracle 是 FD,不是解析

PySCF 的解析偶极导数在 `pyscf.prop.infrared` 模块,该 venv **未安装**。因此
参考值用 PySCF SCF 偶极的有限差分(每个核坐标 ±h,中心差分)。FD 截断误差
O(h²)~4e-7 au 就是 1e-4 容差的下限来源——引擎侧(CPHF 密度响应解析 + 偶极积分
块半数值导数,h=1e-4)比这个 FD oracle 更准,实测最大偏差 4.0e-7 au,即完全
落在 oracle 的 FD 噪声内。

## Pinned ∂μ_a/∂R_{A,j} [au] — [atom A][axis j][dipole a]

```
A0(O) j0: [-0.61244432,  0,           0         ]
A0(O) j1: [ 0,          -0.05805634,  0         ]
A0(O) j2: [ 0,           0,           0.03881088]
A1(H) j0: [ 0.30622222,  0,           0         ]
A1(H) j1: [ 0,           0.02902815,  0.15747119]
A1(H) j2: [ 0,           0.21470465, -0.01940554]
A2(H) j0: [ 0.30622222,  0,           0         ]
A2(H) j1: [ 0,           0.02902815, -0.15747119]
A2(H) j2: [ 0,          -0.21470465, -0.01940554]
```

Σ over atoms per (j,a) ≈ 2e-7 au (translational invariance; the FD floor).

## 物理与实现

∂μ_a/∂R_{A,j} = −Tr[∂P/∂R_{A,j}·r_a] − Tr[P·∂r_a/∂R_{A,j}] + Z_A δ_{aj}。
∂P/∂R 是 CPHF 密度响应(`CphfPack::density_response`,已在 test_cphf.cpp 对
FD 验证);∂⟨μ|r_a|ν⟩/∂R 用偶极积分块中心差分(半数值,同 ECP 梯度模式)。
IR 强度 I_k ∝ Σ_a (Σ_x ∂μ_a/∂R_x · L_{x,k})²(投影到简正模 `normal_modes`);
本回归门控 APT 张量本身——IR 强度的载重物理。逐模强度的 km/mol 投影需
`pyscf.prop` oracle 方能验证,留作后续。

文献:Wilson, Decius, Cross, *Molecular Vibrations* (1955)(APT / IR 强度);
Helgaker, Jørgensen, Olsen §10(CPHF / 分子性质导数);PySCF
`pyscf/prop/infrared`(解析实现参考)。
