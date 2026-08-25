---
spec: comm-topology-01-swaplist
created: 2026-08-25
criteria:
  - id: ac-001
    summary: "Swap table owns sendlist/nrecv/firstrecv; DD no longer does"
    type: code
    pass_when: |
      include/atomiverse/cpu/communication/domain.h 中不再出现标识符
      `send_lists` 或 `recv_counts`；仓内（排除 build*/）对这两个 DD 成员
      的引用为 0；include/atomiverse/cpu/communication/swap_list.h 定义
      `struct Swap` 含 sendproc/recvproc/sendself/pbc[3]/slablo/slabhi/
      sendlist/nrecv/firstrecv，且 swap_list.cpp 不含任何 MPI_ 调用。
    status: pending
  - id: ac-002
    summary: "Old cascade deleted with no fallback path"
    type: code
    pass_when: |
      src/cpu/communication/halo_exchange.cpp 中不存在符号
      `rebuild_send_lists_for_axis`、`forward_axis_stage`、
      `pbc_shift_for_face`；文件内没有任何在新旧 halo 路径之间切换的
      `#ifdef`、环境变量或运行时布尔开关。
    status: pending
  - id: ac-003
    summary: "Ownership/publication mirrors NeighborList→PairList; public API unchanged"
    type: code
    pass_when: |
      HaloSend 私有持有 SwapList 并在 launch 中把 `const comm::SwapList*`
      赋给 ctx.md.swap_list（context.h 的 DynamicsServices 新增该字段，
      非拥有）；HaloSend 构造签名仍为 `HaloSend(double cutoff, int every = 10)`；
      组件名 halo_send/halo_recv/halo_refresh_{x,v,q} 与默认绑定
      pre_neighbor@0 / post_force@-1000 未变；HaloRefresh::Field 仍为
      {positions, velocities, charges}；md.h 的三处自动注入 phase/order
      (post_integrate@-100, final_integrate@+150, pre_force@+900) 未变；
      新增符号中无 *Config / make_* / build_* / create_*。
    status: pending
  - id: ac-004
    summary: "SwapList setup/borders unit tests pass without mpirun"
    type: runtime
    evaluator_hint: "ctest -R test_cpu_comm_swap_list"
    pass_when: |
      `ctest --test-dir build -R test_cpu_comm_swap_list --output-on-failure`
      返回 0 failures，且该文件断言（硬编码期望值）：dims={2,1,1}、
      box 20×10×16 Å、cutoff 3.0 Å 下 nswap==6、maxneed(d)==1、swap 序为
      (0,lo),(0,hi),(1,lo),(1,hi),(2,lo),(2,hi)；coords[0]==0 的 lo swap
      pbc=={+1,0,0}；swap 2..5 sendself==true 且 nrecv==0；
      commit_recv 对 nrecv={3,2,0,0,0,0}、n_local=10 给出
      firstrecv=={10,13,15,15,15,15} 与 total_recv()==5；
      rebuild_sendlist 的索引集合与半开判据 (q−sublo<cutoff / subhi−q<cutoff)
      逐元素相同。测试进程不调用 mpirun/mpiexec。
    status: pending
  - id: ac-005
    summary: "setup() throws with how-to when derived maxneed exceeds 1"
    type: runtime
    evaluator_hint: "ctest -R test_cpu_comm_swap_list"
    pass_when: |
      单元测试中，halo_cutoff 大于 L_d/dims_d 的构造使 SwapList::setup
      抛出 std::runtime_error，消息同时含「哪一维」与三选一的 how-to
      （更大的盒 / 该轴更少 rank / 更小 cutoff）；反向地，
      halo_cutoff 恰等于 L_d/dims_d − 1e-9 的构造不抛错且 maxneed==1。
    status: pending
  - id: ac-006
    summary: "All 20 tests/mpi cases stay green at n=1/2/8 (behaviour equivalence)"
    type: runtime
    evaluator_hint: "ctest -L mpi"
    pass_when: |
      `ctest --test-dir build -L mpi --output-on-failure` 报告 0 failures 且
      执行的用例数不少于重构前（tests/mpi/ 下 20 个源文件对应的全部
      n=1/2/4/8 注册项），其中 test_mpi_lj_energy_bruteforce、
      test_mpi_coul_energy_bruteforce、test_mpi_bonded_energy_bruteforce、
      test_mpi_eam_multirank、test_mpi_sw_multirank 的 O(N²) 暴力对比
      容差未被放宽（diff 中这些文件的容差常量不变）。
    status: pending
  - id: ac-007
    summary: "dims==2 dual-image ghosts survive; no tag dedup introduced"
    type: runtime
    evaluator_hint: "ctest -R test_mpi_halo_dual_image_ghosts"
    pass_when: |
      `ctest --test-dir build -R test_mpi_halo_dual_image_ghosts --output-on-failure`
      在 n=1 与 n=2 均通过；且新增/更新的断言证明在 2*halo_cutoff > 子域跨度
      的几何下，至少一个 owner tag 在 ghost 行中出现两次（两个周期镜像），
      即 forward 路径未按 tag 去重、未把 sendself swap 变成本地 memcpy。
    status: pending
  - id: ac-008
    summary: "HaloRefresh seam unchanged and idempotent over the swap table"
    type: runtime
    evaluator_hint: "ctest -R 'halo_refresh'"
    pass_when: |
      `ctest --test-dir build -R halo_refresh --output-on-failure` 覆盖
      test_cpu_comm_halo_refresh_charges 与 test_mpi_halo_refresh_charges_n1/n2
      且 0 failures；连续两次 launch 同一 HaloRefresh 后 ghost 行数值不变
      （幂等），positions 携带周期位移而 velocities/charges 不携带。
    status: pending
  - id: ac-009
    summary: "Regression regressions/md/comm-swaplist-halo-mpi meets its hard-coded gates"
    type: runtime
    evaluator_hint: "ctest -R reg_comm-swaplist-halo-mpi"
    pass_when: |
      `ctest --test-dir build -R reg_comm-swaplist-halo-mpi --output-on-failure`
      退出码 0 且用时 ≤ 60 s。regressions/md/comm-swaplist-halo-mpi/
      reg_comm-swaplist-halo-mpi.cpp 只包含公开头并链接 atomiverse，经
      ATV_MPIEXEC 自重启为 mpiexec -n 2，在 8³=512 原子 SC 晶格
      (a0=3.617 Å, box 28.94 Å, ε=0.010341 eV, σ=3.4 Å, r_c=8.5 Å,
      一个原子预位移 0.35 Å) 上跑 21 步后，以文件内 O(N²) min-image 求和为
      判据断言全部硬编码门：|E_dd − E_bruteforce|/N ≤ 1e-9 eV/atom、
      max_i|F_i,dd − F_i,bruteforce| ≤ 1e-9 eV/Å、|Σ_i F_i| ≤ 1e-10 eV/Å、
      nswap == 6、Σ nrecv == n_ghost > 0、全量重建次数 ≥ 2；
      运行期不 import、不 subprocess、不读取任何第三方引擎产物。
    status: pending
  - id: ac-010
    summary: "Doxygen on new public symbols + stale halo prose synced"
    type: docs
    pass_when: |
      include/atomiverse/cpu/communication/swap_list.h 中每个公开符号
      (Swap 的每个字段、SwapList 的每个公开方法) 都有 @brief，带参方法有
      @param 且长度/坐标参数标注单位 [Angstrom]，有返回值的有 @return；
      Swap::sendself 的注释写明「跳过而非 LAMMPS memcpy」及其理由
      (cell_list.cpp 的 axis_extended_ min-image 双计)；
      halo_exchange.h、domain.h、docs/developer/kspace-long-range.md 中
      不再有 `send_lists` / `recv_counts` 的残留描述。
    status: pending
out_of_scope:
  - "maxneed > 1 多跳 halo（-02）"
  - "256-rank 强扩展验证与任何 wall-clock/加速比声明（-03）"
  - "OpenMP 混合并行、kspace 子通信域、Kokkos"
  - "Migrate/Alltoallv 交换路径与 RCB 负载均衡算法"
  - "CUDA 后端 cuda/communication/"
---

# Acceptance — comm-topology-01-swaplist

「done」= halo 的 **结构** 换成了 LAMMPS CommBrick 的 swap 表（一次派生、按表收发、旧级联删干净、无 fallback），而 **数值与可见行为一字未变**：既有 20 个多 rank 用例在 n=1/2/8 全绿且容差未被放宽，dims==2 的双镜像 ghost 与 `HaloRefresh` 的接缝原样保留，新加的纯逻辑单测把 swap 展开/PBC 表/sendself/前缀和钉死，回归示例用就地解析判据证明域分解结果与单机求和恒等。本 spec 不声明任何性能改进——没有 `performance` 判据是有意的。

## AC-001 — Swap 表接管索引所有权

`dd->send_lists` / `dd->recv_counts` 是「派生索引挂在热路径几何结构上」的典型，且让每次 comm 形状变化重编整个含 `domain.h` 的世界。判据是二值的：这两个字段在 `domain.h` 中消失，全仓无引用，取而代之的 `Swap` 结构体字段齐全，并且 `swap_list.cpp` 不含 MPI 调用（证明 SwapList 是纯本地数据结构，通信留在 Component 侧）。

## AC-002 — 旧路径删干净

本仓无兼容层。逐步扫描的三个符号必须从 `halo_exchange.cpp` 消失，且不得留任何新旧切换开关——留一个 `#ifdef` 就等于留两套语义，等价性也就无从验证。

## AC-003 — 所有权形状与公开面冻结

`SwapList` 由 `HaloSend` 拥有、以非拥有 const 指针发布到 `ctx.md.swap_list`，与 `NeighborList → ctx.md.pair_list`（CLAUDE.md § Frame 点名认可）同形。同时冻结用户可见面：构造签名、组件名、默认绑定、`HaloRefresh::Field`、`md.h` 三处自动注入的 phase/order——任何一处变动都会让「行为等价」这句话失真。

## AC-004 / AC-005 — 纯逻辑单测钉死 swap 展开

`SwapList::setup` 与 `rebuild_sendlist` 都是位置/几何的纯函数，可以在没有 mpirun 的情况下被单方法单测覆盖，这正是把它们从 Component 里抽出来的理由。期望值全部硬编码，并把「`need_d > 1` 抛错」当作 -02 的正式交接点固定下来：它与今天 `HaloSend::build()` 的几何守卫代数等价，因此不是新增限制。

## AC-006 — 多 rank 套件是等价性的主判据

20 个 `tests/mpi/` 用例是这次重构唯一有意义的「之前=之后」证据。判据包含「容差未被放宽」这一条，因为让套件变绿的最省事方式恰恰是调松容差。

## AC-007 — 双镜像不得被「顺手修好」

`dims==2` 时同一原子经两个 swap 到达同一 rank、形成两份周期镜像，是正确的现行行为：`cell_list.cpp` 只对非 extended 轴做 min-image，靠真实距离拒绝远镜像。重构中最诱人的错误就是按 tag 去重、或按 LAMMPS 给 `sendself` swap 加本地 memcpy——两者都会让 ghost 集合改变并触发静默双计（历史上是 ~16% LJ 能量误差）。这条判据单独存在，就是为了让这个诱惑显性化。

## AC-008 — HaloRefresh 接缝

`HaloRefresh` 不只服务 RATTLE：`md.h` 有三处自动注入（positions / velocities / charges）。其公开 `Field` 枚举与幂等契约不许变，函数体在 swap 表下简化为「按序遍历 swap、重打包、在 firstrecv 处 unpack」。CPU 单测与 n=1/2 的 MPI 用例共同守这条线。

## AC-009 — 回归示例

`regressions/md/comm-swaplist-halo-mpi/` 是本 spec 唯一的 e2e，作为库外消费者只用公开头。判据不是「等于我们记下的某个数」，而是解析恒等式：域分解只是同一个 min-image 求和的划分，故 DD 结果必须等于同一构型上的 O(N²) 暴力解；`|Σ F| ≤ 1e-10 eV/Å` 则单独抓 reverse 跑了零次或两次。几何被刻意选成 `need = 1` 但 `2·halo > 子域`，使双镜像路径在回归里真实发生；21 步跨过 ≥2 次全量重建，使 borders 路径与步内 forward 路径都被覆盖。运行期不接触任何第三方软件。

## AC-010 — 文档

新公开符号按 Doxygen 约定带单位；`Swap::sendself` 必须把「偏离 LAMMPS」的理由写在代码旁（否则下一个读者会「修好」它）；`halo_exchange.h` / `domain.h` / `docs/developer/kspace-long-range.md` 里描述旧 `send_lists` 结构的段落同步重写，不留指向已删字段的文字。
