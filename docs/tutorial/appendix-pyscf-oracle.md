# 附录:PySCF 对照方法学(oracle 协议)

本附录是 HF 教程弧(ch02–ch04、ch18–ch20)全部"与 PySCF 对照"表格的方法学总纲。
四条铁律,任何一章、任何回归用例、任何基准数字都必须遵守:

## 1. PySCF 只在外部运行——永不成为构建/测试依赖

禁的是**依赖**,不是知识。PySCF(以及 CP2K 等其它参考引擎)的数字由开发者在
oracle venv 里离线运行一次,然后**钉入**(pin)文档、回归源码注释或
`REFERENCE.md`。CMake、ctest、单元测试在任何配置下都不得 import、调用或读取
PySCF 及其数据文件。守卫:`grep -ri pyscf CMakeLists.txt cmake/ tests/` 只允许
命中注释/出处说明。

环境搭建(幂等,登录 shell 安全):

```bash
bash tools/setup_pyscf_env.sh
```

装入 `~/work/.x86_64` venv:pip(ensurepip 引导)、pyscf、molvis(本地
`~/work/molcrafts/molvis/python` 的 editable 安装)。
**molplot 现状(2026-07)**:molplot 是 molvis 仓库内的 TypeScript 子包,
尚无 Python 发行;教程图形工具链 = molvis(3D 分子/场景渲染,Python)+
molplot(曲线/图表,TS,经 molvis 页面工具产出)。Python 侧出现可安装的
molplot 后,本附录与 setup 脚本同步更新。

## 2. 每个钉定数字都携带完整复算元数据

pin 进任何地方的 PySCF 参考值必须同时记录:

| 字段 | 示例 |
|---|---|
| 几何(坐标 + 单位) | H2O,O(0,0,0.1173)/H(0,±0.7572,−0.4692) Å |
| 基组(+ cart/sph 约定) | 6-31G*(cartesian-6d)/ def2-SVP(spherical) |
| 收敛设置 | `conv_tol=1e-12`, `max_cycle=100` |
| PySCF 版本 | 2.14.0 |
| 完整命令/脚本 | `mf = scf.RHF(mol); mf.kernel()` |

任何人拿这五项应能离线复算出同一数字。引文:Sun et al.,
*WIREs Comput. Mol. Sci.* 8, e1340 (2018), DOI 10.1002/wcms.1340。

## 3. 单位与容差分级

- 能量:内部 Hartree;Frame/教程报告 eV(`core/units.h`)。表格标明单位。
- 容差分级(章末表格用):能量对账 ≤1e-8…1e-6 Ha(按环 spec);性质
  (偶极等)≤1e-6…1e-5 a.u.;频率 ≤1 cm⁻¹;电荷 ≤1e-6…0.01 e(方法依赖);
  溶剂化能 ≤0.1 kcal/mol。每张表沿用其所属环 acceptance 的 pass_when 容差。

## 4. 标量只进表格,禁止柱状图

标量结果(能量、偶极、迭代数、墙钟)一律用 Markdown 表格呈现——**禁止柱状图**。
曲线类数据(收敛轨迹、基组收敛、标度)允许作图,图一律由 molvis/molplot 产出。

## 回归口径(字节一致性守卫)

基组静态表与 PySCF 基组库的逐位一致由生成器的 `--check` 模式守卫:

```bash
~/work/.x86_64/bin/python tools/gen_basis_data.py --check \
    include/atomiverse/cpu/scf/basis_data/sto3g.h sto-3g sto3g H He Li Be B C N O
```

### 漂移基线(2026-07-20,PySCF 2.14.0)

| 表 | `--check` 结果 | 说明 |
|---|---|---|
| sto3g.h | 逐字节一致 | 验收门 |
| pople_631gs.h | 逐字节一致 | — |
| def2_svp.h | 仅空白差 | 入库时经 clang-format 重折行;数值逐 token 一致 |
| def2_tzvp.h | 仅空白差 | 同上 |
| pople_631g.h | 有差 | He 行系 PySCF 2.13.1 手工钉入(见该文件头注);其余为空白差 |

漂移的处置(重排 or 重生成)不在 01 环范围;后续动这些表的环(05-basis)
以重生成收敛到生成器格式。
