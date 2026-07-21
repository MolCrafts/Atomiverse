---
title: "HF capability gaps 01 — PySCF 对照环境、oracle 协议与基组生成器恢复 (env bootstrap)"
slug: hf-capability-gaps-01-env
status: code-complete
chain: hf-capability-gaps
link: 01/21
created: 2026-07-20
---

# HF capability gaps 01 — PySCF 对照环境、oracle 协议与基组生成器恢复

## Summary

本链启动 HF 能力补全链（共 21 环）的基础设施环节：把 `~/work/.x86_64` 这个缺 pip 的 venv 修复为可用的 Python 对照环境（ensurepip 引导、安装 PySCF、以可编辑方式安装本地姊妹仓库 `~/work/molcrafts/molvis` 及其 molplot 子包），交付一个幂等、登录 shell 安全、带文档的脚本 `tools/setup_pyscf_env.sh`；成文 PySCF-as-oracle 对照协议（PySCF 只在外部运行，永不成为构建/测试依赖——禁的是依赖，不是知识）；恢复被 `basis_table.h` 与全部 `basis_data/*.h` 头注释引用但树中缺失的生成器 `tools/gen_basis_data.py`，验收标准为重新生成的 `sto3g.h` 与已入库版本逐字节一致；并搭好 Jensen 风格 HF 教程弧的骨架（ch18–ch20 章节桩 + oracle 附录 + index 弧描述），供后续各环逐节填充。

## Domain basis

本环无物理方程，Domain basis 为可复现性与对照方法学：

- **Oracle 数值可追溯**：每个 pin 进文档/回归的 PySCF 参考值必须附几何（坐标 + 单位）、基组、收敛设置（`conv_tol`、`max_cycle`）与 PySCF 版本，使任何人可离线复算。PySCF: Sun et al., *WIREs Comput. Mol. Sci.* 8, e1340 (2018), DOI: 10.1002/wcms.1340。
- **数据逐位一致**：`basis_data/*.h` 的数值取自 PySCF `mol._basis`（非归一化收缩系数约定，见 `basis_table.h`），生成器必须逐字节复现已入库头文件，才能证明数据通路无漂移。
- **标量对照永远用表格，禁止柱状图**；教程示意图仅允许 molvis/molplot（既有 index.md 规则的延伸）。

## Design

- `tools/setup_pyscf_env.sh`（bash，`set -euo pipefail`）：以绝对路径调用 `~/work/.x86_64/bin/python`，依次 `-m ensurepip --upgrade`、`-m pip install pyscf`、`-m pip install -e ~/work/molcrafts/molvis`（molplot 随 molvis 一并可导入)；每步先探测（`import` 成功即跳过）实现幂等；不依赖 Lmod module，可在 `bash -lc` 与普通 bash 下运行；结尾自检 `python -c "import pyscf, molvis, molplot"` 并打印版本。
- `tools/gen_basis_data.py`：CLI 为头注释中记载的既有约定 `python tools/gen_basis_data.py <pyscf-basis-name> <header-stem> <元素...>`（如 `sto-3g sto3g H He Li Be B C N O`），从 PySCF 基组库取 `mol._basis`，按已入库头文件的精确格式（doxygen 头、`AtomBasis{{...}}` 布局、Python `repr` 浮点、80 列折行）输出到 stdout；增加 `--check <header路径>` 模式：与指定头文件逐字节比对，不一致时退出非零并打印 diff。生成器只被人工在 venv 中运行，绝不接入 CMake/ctest。
- 教程弧：`appendix-pyscf-oracle.md` 承载对照方法学（单位、容差分级、表格格式规则、pin 元数据字段清单、外部运行流程）；ch18/ch19/ch20 为 Jensen 风格章节桩——方法叙事为主干、算法技巧内嵌，每章末尾预置"与 PySCF 对照"累积表占位；`index.md` 增补该弧的分组与链接。

## Files to create or modify

- `tools/setup_pyscf_env.sh` (new)
- `tools/gen_basis_data.py` (new)
- `docs/tutorial/appendix-pyscf-oracle.md` (new)
- `docs/tutorial/ch18-scf-techniques.md` (new)
- `docs/tutorial/ch19-hf-large-systems.md` (new)
- `docs/tutorial/ch20-hf-properties-environment.md` (new)
- `docs/tutorial/index.md`

## Tasks

- [x] Write failing byte-identity check: 确认 `~/work/.x86_64/bin/python tools/gen_basis_data.py --check include/atomiverse/cpu/scf/basis_data/sto3g.h sto-3g sto3g H He Li Be B C N O` 当前失败（生成器缺失），把该命令记入 `docs/tutorial/appendix-pyscf-oracle.md` 草稿作为回归口径
- [x] Implement `tools/setup_pyscf_env.sh`：ensurepip 引导 + pyscf + molvis/molplot 可编辑安装，幂等、登录 shell 安全，脚本头部注释说明用法与前提
- [x] Verify 环境：连续执行两次 `tools/setup_pyscf_env.sh` 均退出 0，且 `~/work/.x86_64/bin/python -c "import pyscf, molvis, molplot"` 成功
- [x] Implement `tools/gen_basis_data.py`：PySCF `mol._basis` → 头文件文本（stdout）+ `--check` 逐字节比对模式，格式与已入库头完全一致
- [x] Verify sto3g 逐字节一致：上述 `--check` 命令退出 0；再对 pople_631g/pople_631gs/def2_svp/def2_tzvp 各跑一次 `--check`，仅记录漂移（不修改这四个头文件）
- [x] Write `docs/tutorial/appendix-pyscf-oracle.md` 定稿：外部运行铁律、pin 元数据字段（几何/基组/收敛/PySCF 版本）、单位与容差、标量只用表格、molvis/molplot 为唯一绘图工具
- [x] Write 章节桩 `ch18-scf-techniques.md` / `ch19-hf-large-systems.md` / `ch20-hf-properties-environment.md`（Jensen 风格小节骨架 + 章末对照表占位）并 Update `docs/tutorial/index.md` 增补该弧描述与链接
- [x] Run full check + test suite：确认本环未向 CMake/ctest 注册任何目标，单元套件通过且用时不变

## Testing strategy

- **Happy path（外部手动/评估器执行，均不进 ctest）**：setup 脚本首跑成功；`--check` sto3g 逐字节通过；三个 import 全部成功。
- **边界**：脚本二次执行幂等（不重复安装、不报错）；`--check` 对被篡改一个字符的临时副本退出非零；生成器对不支持的元素抛出明确错误（无 fallback）。
- **领域验证（science.required）**：sto3g 逐字节一致即证明 C++ 表与 PySCF 基组库数值逐位相同；其余四个基组头的 `--check` 结果记录在案，为后续环提供漂移基线。
- **不回归**：`grep -ri pyscf CMakeLists.txt cmake/ tests/` 无构建/测试期依赖；ctest 全绿且无新目标。

## Out of scope

- 修复其余四个基组头文件与生成器输出的任何漂移（本环只记录；若有漂移由后续环出专门 spec 处理）。
- `tools/gen_minao_data.py`、MSU 校验脚本等其他缺失工具。
- ch18–ch20 的正文内容与对照数据（由链的后续环逐节填充）。
- `benchmarks/scf/validation/` 的外部引擎 parity 扫描（专门环负责）。
- 任何 C++ 代码、`regressions/` 用例或 CMake 改动。

<!-- simplify: ran clean 2026-07-20 — link adds 2 tools scripts + 5 docs files only; dead helper removed from gen_basis_data.py during impl; no C++/CMake surface. -->
