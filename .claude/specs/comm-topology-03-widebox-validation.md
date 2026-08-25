---
title: 多跳落地后的宽域实机验证（24k GAFF / 256 rank）
slug: comm-topology-03-widebox-validation
status: in-progress
created: 2026-08-25
grilled: true
deps: comm-topology-01-swaplist, comm-topology-02-multihop
---

# 多跳落地后的宽域实机验证（24k GAFF / 256 rank）

## Summary

在 `comm-topology-02-multihop` 让 ghost 壳层可以跨越一个以上子域之后，本 spec 交付这条链的**验证层**：一个可重复的仓库内多跳几何证据（tests/mpi 的强制窄网格用例 + regressions/ 的宽域 parity 案例），以及一条可脚本执行的实机判据——把外部 MIL0PEGM18 24k 原子 GAFF 体系放到单节点 256 rank（8×8×4，子域跨度 < cutoff+skin，正是多跳几何）上跑单点，逐项能量与力必须与 np=1 在数值窗口内一致。性能数字（steps/s、分项计时、与 LAMMPS 同节点同精度档对比）只写进 `tools/bench_gaff_lammps/README.md` 的日期快照块，**不构成验收判据**。不动 `src/`/`include/` 任何引擎代码：这是消费层的验证 spec，多跳实现本身属于 -02。

## Domain basis

- **域分解的 ghost 壳层条件。** 短程分解式 MD 中，每个 rank 必须持有厚度 ≥ `r_cut + skin`（多体/bonded 还要加各自的延伸量）的 ghost 壳层，才能在本地重建完整的相互作用集合：S. Plimpton, *Fast Parallel Algorithms for Short-Range Molecular Dynamics*, J. Comput. Phys. **117**, 1–19 (1995), DOI 10.1006/jcph.1995.1039。当某轴子域跨度 `L_sub = L_d / dims_d` 小于所需壳层厚度 `r_ghost` 时，单跳无法填满壳层，需要的跳数为 `n_hop = ceil(r_ghost / L_sub)`。仓库现状把这个条件写成了硬拒绝：`src/cpu/communication/halo_exchange.cpp:498-517` 在 `need > sub` 时抛出 "ghost shell wider than this rank's subdomain"。本 spec 的所有用例都精确落在 `r_ghost / L_sub ∈ (1, 2]`（即 2 跳）这一档，**-02 未合入的树上必须是这条 throw 触发的 RED**。
- **验收所用的物理不变量。** 势能与力是体系 Hamiltonian 的函数，与 rank 数、与处理器网格无关：`E(np=1, 1×1×1) = E(np=P, dims)` 在精确算术下逐位相等，实数运算下只允许双精度求和次序（reassociation）带来的差异。这是本 spec 的判据来源——不是"冻结 ATV 当前输出"，而是分解不变性。
- **独立 oracle（仓库内案例）。** 最小镜像直接求和：M. P. Allen & D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed., Oxford (2017), §1.6.3（minimum image convention）。O(N²) 最小镜像和是 cutoff 模型 Hamiltonian 的**定义**，因此可以作为 `regressions/` 要求的独立参考，与 `tests/mpi/test_lj_energy_bruteforce.cpp` / `test_bonded_energy_bruteforce.cpp` 同一族。
- **实机体系的力场与长程。** GAFF：J. Wang et al., J. Comput. Chem. **25**, 1157 (2004), DOI 10.1002/jcc.20035；SPME/PPPM 倒空间：U. Essmann et al., J. Chem. Phys. **103**, 8577 (1995), DOI 10.1063/1.470117。MIL0PEGM18 体系**没有**文献参考值，所以实机判据的 oracle 只能是"同 α / 同网格下 ATV 自身的 np=1 单点"（跨 rank parity，不是外部 oracle pin）——不得以 `oracle: lammps` / `oracle: cp2k` 的形式记入任何 pin 台账。
- **单位。** eV、Å、fs、eV/Å；能量窗口以相对量给出，分母取 `max(1 eV, |E|)`；力窗口以 `fmax`（全局 max |f_i|）与 `fnorm`（全局 Σ|f_i|² 的平方根）两个 rank 无关标量表达。

## Design

**touched entities（全部在验证消费层）**

1. `tools/bench_gaff_lammps/run_atv_gaff.cpp` — 已有 `--single-point` 打印逐项能量（已做 `MPI_Allreduce`）。本 spec 只补两条**同样 rank 无关**的输出：
   - `[atv] SINGLE-POINT FORCES: fmax=<%.10e> fnorm=<%.10e>` —— `fmax` 用 `MPI_Allreduce(MAX)`，`fnorm` 用 `MPI_Allreduce(SUM)` 后开方，只统计 owner 行（`nlocal`），ghost 不计。
   - `[atv] GRID dims=<dx,dy,dz> span=<sx,sy,sz> halo=<r_cut+skin> hops=<hx,hy,hz>` —— 由公共静态函数 `DomainDecomposition::choose_dims`（`include/atomiverse/cpu/communication/domain.h:260`）纯函数地推出，因此自动反映 `ATV_FORCE_DIMS` 覆盖；`span_d = L_d/dims_d`，未分解轴（`dims_d == 1`）打印 `hops=0`。注释里必须写明 halo 只算 `cutoff + skin`、bonded 延伸由引擎再加，所以这是几何**下界**证据。
   不新增子命令、不新增可执行文件、不改任何默认行为（无 `--single-point` 时输出逐字不变）。
2. `tools/bench_gaff_lammps/widebox_parity.sh`（new）— 唯一的新"符号"。形状照抄同目录 `run.sh`：`set -euo pipefail` + 环境变量配置（`BUILD/DATA/FF/NP/DIMS/LAUNCHER/OUT`）+ `tee` 日志。做三件事：(a) 用**显式** `--alpha`/`--grid-spacing`/`--cutoff`/`--tail`/`--shift` 跑一次 np=1 单点，(b) 用同一组参数、`ATV_FORCE_DIMS=$DIMS` 跑一次 np=$NP 单点，(c) 解析两份日志的 `[atv] GRID` / `SINGLE-POINT (eV)` / `SINGLE-POINT FORCES` 行，逐项比对并按 `got / want / tol` 打印 `PASS`/`FAIL`，任一项越界即非零退出。窗口为脚本默认值、可用 flag 覆盖，验收绑定的是**默认值**：

   | 量 | 窗口 |
   |---|---|
   | `bond` / `angle` / `dihedral` 各自 | 相对 ≤ `1e-9` |
   | `pair+kspace` | 相对 ≤ `1e-8` |
   | 总能 `E` | 相对 ≤ `1e-8` **且** \|ΔE\|/N ≤ `1e-6` eV/atom |
   | `fmax` / `fnorm` | 相对 ≤ `1e-6` |

   分母一律 `max(1, |参考值|)`。脚本还必须断言 np=$NP 日志里 `[atv] GRID` 的 `hops` 至少有一个分量 ≥ 2，否则判 FAIL——跑了一个单跳几何却报 PASS 是这条链最危险的假绿。
3. `tests/mpi/test_halo_dual_image_ghosts.cpp` + `tests/CMakeLists.txt:560-562` — 在既有 `HaloDualImageGhosts` pin 上**就地推广**出生产形状：`ATV_FORCE_DIMS=4,2,1` 的 8-rank 网格（x 轴 4 刀、y 轴 2 刀、z 不分解），box 24 × 14 × 14 Å、`cutoff = 6.0`、`skin = 0.6` ⇒ `halo = 6.6 Å`：x 轴 span 6.0 进入 2 跳区（≤ dims−1=3），y 轴 span 7.0 单跳且 `2·6.6 > 7.0` 踩中 dual-image 分支（**几何修正**：原 24×12×12 在 dims_y=2 下 6.6 Å 壳违反最小镜像上界 `shell ≤ L−w_self = 6.0`，物理不可行）——这正是 8×8×4 全轴窄子域在小 rank 数上的最近代理。用例在 `world != 8` 时 `GTEST_SKIP`，进程内 `setenv` 后收尾 `unsetenv`，判据沿用文件里既有的 O(N²) 最小镜像 brute force 与 `kEnergyRelativeTolerance = 1e-9`。
4. `regressions/md/multihop-widebox-parity/`（new，`reg_multihop-widebox-parity.cpp` + `REFERENCE.md`）+ `regressions/CMakeLists.txt` — 仓库内、不依赖任何外部数据集的宽域 parity 案例，形状照抄最近的两个先例：`regressions/md/const-pot-charge-halo-mpi`（自重启二进制、编译期 `ATV_MPIEXEC` pin、`ATV_FORCE_DIMS` 定几何）与 `regressions/pw/mpi-pw-scaling-4rank`（parent 起 np=1 与 np=N 两次 child、child 写一行结果、parent 判窗口）。体系是**带电 + 有 bonded 拓扑**的分子晶格（三原子分子 × 若干，整体电中性），势能 = `PairLJCut` + `PairCoulCut` + `BondHarmonic` + `AngleHarmonic`；不引 PPPM，故只需 `ATV_BUILD_MD + ATV_USE_MPI`，无 FFTW 门。这与 -02 自带的最小 LJ 多跳用例明确不重叠：这里同时跨多跳搬运 **ghost 电荷** 与 **bonded 延伸**，正是 24k GAFF 的形状。两级判据：先 np=1 对最小镜像 brute force（验证 oracle 的 special-bond 排除规则镜像正确），再 np=8（`ATV_FORCE_DIMS=4,2,1`）对 np=1（多跳主张本身）。一个扁平 `main()`，`got/want/tol` 打印，失败非零退出，≤ 60 s。
5. `tools/bench_gaff_lammps/README.md` — 追加一个日期快照块，沿用文件里已有的 LAMMPS 对齐计时表惯例。

**Reuse decision（逐条结算 librarian 候选）**

- `reuse` `run_atv_gaff --single-point` —— 实机判据直接复用它的逐项能量与已对齐的 LAMMPS 精度档；只在同一分支内补两条 reduce 后的输出行，不另起 bench 二进制。
- `reuse` `ATV_FORCE_DIMS` / `DomainDecomposition::choose_dims`（`src/cpu/communication/domain.cpp:74`、`include/atomiverse/cpu/communication/domain.h:260`）—— 窄子域几何一律靠这个既有 env 覆盖构造，不新增 dims 接口、不给 bench 工具加 `--dims` flag；几何打印也走同一个纯函数，保证打印的网格与 DD 实际用的网格同源。
- `generalize` `HaloDualImageGhosts`（`tests/mpi/test_halo_dual_image_ghosts.cpp`）—— 按 librarian 落位建议**扩这个文件与它的注册**，不新建 `tests/mpi/test_multihop_*.cpp`；该 pin 的主题就是"子域跨度 vs ghost 壳层"，生产网格是它的自然推广。
- `pattern` `regressions/md/const-pot-charge-halo-mpi` + `regressions/pw/mpi-pw-scaling-4rank` —— 新 regression 的自重启结构、`ATV_MPIEXEC` 编译期 pin、child 单行结果、parent 判窗口，全部照这两个案例的写法。
- `new — widebox_parity.sh`：`run.sh` 是"单个 np + LAMMPS 对照计时"的驱动，仓库里没有任何"同参数两 np 单点 + 逐项窗口判定 + 非零退出"的驱动；复用 `run.sh` 只会把两个语义塞进一个脚本。新增脚本、并沿用 `run.sh` 的环境变量与日志约定。
- MIL0PEGM18 数据集（`/home/jicli594/work/salamander/...`）：**不入仓**，由 `DATA`/`FF` 环境变量传入；missing 输入即失败，脚本不得回退到别的体系。

**Why（决策记录）**

- 性能不进验收：仓库法则是 spec 验收只绑正确性与能力，性能归 bench 记录追踪。因此本 spec 没有任何 `type: performance` 判据，README 快照块只被要求"存在且写明运行元数据"，不绑任何阈值。
- 256-rank 验收 = **能跑 + parity**：这两件事都是二值可判的；"跑得够快"不是。
- oracle 归属：实机跑的参考是 ATV 自身 np=1，属于跨 rank 自洽，不是外部 pin；仓库内 regression 才带独立（最小镜像）oracle。两者不可互相冒充。
- 窗口越界 = 缺陷，不是"放宽窗口"的理由（CLAUDE.md iron law 9）：若实机 parity 越界，按 `path:line` 报告并停，缺陷修复归 -02 或其后继 spec，不在本 spec 内偷偷放宽 tol、也不在本 spec 内改 `src/`。

## Files to create or modify

- `tests/mpi/test_halo_dual_image_ghosts.cpp`
- `tests/CMakeLists.txt`
- `tools/bench_gaff_lammps/widebox_parity.sh` (new)
- `tools/bench_gaff_lammps/run_atv_gaff.cpp`
- `regressions/md/multihop-widebox-parity/reg_multihop-widebox-parity.cpp` (new)
- `regressions/md/multihop-widebox-parity/REFERENCE.md` (new)
- `regressions/CMakeLists.txt`
- `tools/bench_gaff_lammps/README.md`

## Tasks

- [x] Generalize `HaloDualImageGhosts` in `tests/mpi/test_halo_dual_image_ghosts.cpp` 覆盖生产窄网格（`ATV_FORCE_DIMS=4,2,1`、box 24×14×14 Å、cutoff 6.0 / skin 0.6 ⇒ x 2 跳、y 单跳双镜像、`world != 8` 时 skip、判据沿用 O(N²) 最小镜像 brute force 与 rel < 1e-9），并在 `tests/CMakeLists.txt` 注册 `test_mpi_halo_dual_image_ghosts_n8`
- [x] Add `tools/bench_gaff_lammps/widebox_parity.sh`：同参数跑 np=1 与 np=$NP（`ATV_FORCE_DIMS=$DIMS`）单点，解析 `[atv] GRID` / `SINGLE-POINT (eV)` / `SINGLE-POINT FORCES` 行，按默认窗口逐项 `got/want/tol` 判定，并断言 `hops` 至少一个分量 ≥ 2，失败非零退出（在下一条落地前应因缺少这两行而 FAIL）
- [x] Implement rank 无关的 `fmax`/`fnorm`（`MPI_Allreduce` MAX/SUM，仅 owner 行）与 `[atv] GRID dims/span/halo/hops`（经 `DomainDecomposition::choose_dims`）输出于 `tools/bench_gaff_lammps/run_atv_gaff.cpp` 的 `--single-point` 分支
- [x] Add regression example `regressions/md/multihop-widebox-parity/reg_multihop-widebox-parity.cpp` + `REFERENCE.md`（自重启 np=1 / np=8、`ATV_FORCE_DIMS=4,2,1`、带电 + bonded 分子晶格、最小镜像 brute force + 跨 rank parity 双判据、硬编码 tol、非零退出）并在 `regressions/CMakeLists.txt` 注册
- [ ] Verify MIL0PEGM18 24k 体系在单节点 256 rank（`ATV_FORCE_DIMS=8,8,4`）下 `widebox_parity.sh` 全项 PASS，并保存 np=1 / np=256 两份日志
- [ ] Record 日期快照块于 `tools/bench_gaff_lammps/README.md`（256-rank steps/s、分项计时表、同节点同精度档 LAMMPS 对照、srun 命令与数据集路径，并注明这些数字不是验收判据）
- [ ] Run full check + test suite

## Testing strategy

- **单测（`tests/mpi/`，秒级、单一关注点）。** 本 spec 只加一个用例，落在既有 `HaloDualImageGhosts` 套件里：
  - happy path：`world == 8`、`ATV_FORCE_DIMS=4,2,1`，一步冻结（`VelocityVerlet{0.0}`）后总能对 O(N²) 最小镜像 brute force 相对误差 < `1e-9`。
  - edge：`world ∈ {1,2}` 时新用例 skip、原两个用例照跑（注册的 n1/n2 必须仍然全绿）。
  - **RED 条件明确**：没有 -02 的树上 `HaloSend::build` 因 `need(6.6) > sub(6.0)` 抛异常，用例失败——这就是这条链的红。
  - 不在 `tests/` 里放任何 256-rank、任何外部数据集、任何计时、任何 LAMMPS 对照（iron law 1）。
- **Regression（`regressions/md/multihop-widebox-parity/`，本 spec 的 e2e 唯一落点）。** 两级硬编码判据，全部 `got/want/tol` 打印、失败非零退出、≤ 60 s：
  1. np=1 总能与逐原子力 vs 案例内 O(N²) 最小镜像直接求和（LJ + cutoff Coulomb + 谐振键角，special-bond 排除权重逐条镜像）：能量相对 ≤ `1e-9`，力分量绝对 ≤ `1e-9` eV/Å。
  2. np=8（`ATV_FORCE_DIMS=4,2,1`，两条被分解轴都在 2 跳区）vs np=1：能量相对 ≤ `1e-10`，按 tag 对齐后逐原子力分量最大差 ≤ `1e-9` eV/Å。
  `REFERENCE.md` 写明：oracle 是最小镜像直接求和（Allen & Tildesley §1.6.3）与 Plimpton (1995) 的 ghost 壳层条件，tol 的物理理由是"两种分解只差双精度求和次序"；**不**冻结 ATV 输出当金值。
- **实机验证（不入 ctest、不入 regressions）。** `widebox_parity.sh` 在计算节点上以 sbatch + `srun --cpu-bind=cores` 驱动：`NP=256 DIMS=8,8,4`，`--alpha`/`--grid-spacing`/`--cutoff`/`--tail`/`--shift` 两次运行显式取同值。判据即 Design 表中的默认窗口 + `hops` ≥ 2 断言。若任一项越界：按 `path:line` 报告为缺陷并停，不放宽窗口、不改判据。
- **脚本自身的门要能咬人。** 以不可能满足的窗口（`--tol-total 0`）跑一次，脚本必须 FAIL 且非零退出——防止"永远打印 PASS"的假绿。
- **性能采集**（同一批 sbatch 作业内跑 `--steps`）只产出 README 快照块，不产生任何判据。

## Out of scope

- **性能达标承诺**：256-rank steps/s、与 LAMMPS 同节点比值、强扩展曲线只记入 README 快照块，不作验收；当前已知基线（LAMMPS 256 rank 1523 步/s；ATV 125 rank ~290–375 步/s）只是对照背景。
- **多跳 halo 的实现本身**：属于 `comm-topology-02-multihop`；本 spec 不动 `src/`/`include/`。实机跑出的引擎缺陷按 iron law 9 报告并另开 spec，不在此就地改。
- **OpenMP 混合并行**（每 rank 多线程）：本 spec 全程 `OMP_NUM_THREADS=1`。
- **kspace 子通信域 / pencil 分解**：PPPM 在 256 rank 下的通信结构优化是独立轨道。
- **SHAKE / 生产协议 runner**：`--npt`、约束、长时程协议不在验证范围。
- **MIL0PEGM18 数据集入仓**：外部工件，永远经环境变量传入；仓库内只保留不依赖外部数据的代理案例。
- **外部 oracle pin**：本 spec 不产生任何 `oracle: lammps` / `oracle: cp2k` 台账行；实机 oracle 是 ATV 自身 np=1。
