# Reference — N2 RHF/6-31G* 电离能:Koopmans 与 ΔSCF

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md(外部运行、钉定);
钉定脚本:`tools/pin_charges_oracle.py`(只在外部 venv 里跑,绝不进
CMake/ctest)。

| 字段 | 值 |
|---|---|
| 几何 (Å) | N(0,0,0); N(0,0,1.0977)(实验键长)|
| 基组 | 6-31G* (cartesian-6d) |
| PySCF 版本 | 2.14.0,`conv_tol = 1e-12` |
| E(N₂) RHF | −108.942662347983 Ha |
| **−ε_HOMO** | **0.611756029712 Ha = 16.64672961 eV**(`mf.mo_energy[nocc−1]`)|
| E(N₂⁺) UHF | −108.365971102534 Ha(`charge=1, spin=1`;⟨S²⟩ ≈ 0.766)|
| **ΔSCF IP** | **15.69256822 eV** = (E₊ − E₀)×27.211386245988 |
| 对照容差 | 各 ≤1e-3 eV |

物理背景(Szabo & Ostlund, *Modern Quantum Chemistry*, §3.3):HF 极限的
N₂ Koopmans IP ≈ 17.3 eV,实验 15.58 eV —— 冻结轨道高估;6-31G* 下
−ε_HOMO = 16.65 eV(基组未到极限,数值随基组走,故对照钉在**同基组**的
PySCF 值上,文献带只作宽松 sanity:15.5–17.5 eV)。ΔSCF 让阳离子轨道弛豫,
15.69 eV 低于 Koopmans 且更接近实验 —— 回归同时断言 ΔSCF < Koopmans。

阳离子只经既有成员配置:`uhf.charge = 1; uhf.n_alpha = 7; uhf.n_beta = 6`
—— 无新 Config 类型、无工厂函数(验收 ac-005 的 grep 条款)。

引文:
- Koopmans, *Physica* **1**, 104 (1934), DOI 10.1016/S0031-8914(34)90011-2。
- Szabo & Ostlund, *Modern Quantum Chemistry* (Dover 1996), §3.3(N₂ 讨论:
  −ε_HOMO ≈ 17.3 eV vs 实验 15.6 eV)。
- 实验垂直 IP 15.58 eV:NIST WebBook (N₂)。
