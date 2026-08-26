---
slug: lammps-comm-port
created: 2026-08-26
criteria:
  - id: ac-001
    summary: No MPI collective on the per-step halo path
    type: code
    pass_when: |
      In src/cpu/communication/halo_exchange.cpp no MPI_Allreduce / MPI_Bcast /
      MPI_Barrier / MPI_Allgather call is reachable from HaloSend::launch,
      HaloRecv::launch or HaloRefresh::launch on a non-full-rebuild step: every
      remaining collective sits either in a build() body or inside the
      full_rebuild branch (HaloRecv gates on a change of dd->halo_generation).
      mpi::Reduce (src/cpu/communication/reduce.cpp:70) is untouched and is
      explicitly not counted.
    status: verified
    last_checked: 2026-08-26
  - id: ac-002
    summary: Rounds reconciliation is a pure local sweep, SwapList stays MPI-free
    type: code
    pass_when: |
      src/cpu/communication/swap_list.cpp includes no <mpi.h> and references no
      MPI_ symbol; SwapList::derive_uniform_rounds exists, is declared in
      include/atomiverse/cpu/communication/swap_list.h with Doxygen @param /
      @return carrying [Angstrom] units, and setup_swaps in halo_exchange.cpp
      builds its rounds[3] from it with no MPI_Allreduce left in that function.
    status: verified
    last_checked: 2026-08-26
  - id: ac-003
    summary: derive_uniform_rounds unit suite green on hard-coded geometries
    type: runtime
    evaluator_hint: "ctest --test-dir build -R test_cpu_comm_swap_list_uniform_rounds --output-on-failure"
    pass_when: |
      ctest -R test_cpu_comm_swap_list_uniform_rounds passes, covering: dims
      {8,8,4} / cubic box 63.0016 A / shell 11.0 A yields rounds {2,2,1}; an
      axis with dims<=1 yields 0; a shell exactly one slab wide yields 1; the
      dims {2,1,1} / L 10 A / shell 9 A case throws std::runtime_error naming
      the minimum-image bound at EVERY coordinate of that axis; and on both a
      uniform and a cuts-based grid the result equals the max over all
      coordinates of SwapList::derive_hops.
    status: verified
    last_checked: 2026-08-26
  - id: ac-004
    summary: kColComm is exactly x, y, z; kColBorder unchanged
    type: code
    pass_when: |
      In include/atomiverse/core/frame_schema.h only the x, y and z rows of
      ATV_ATOM_FIELDS carry kColComm; vx, vy, vz and charge no longer do, and
      every one of vx, vy, vz, mass, charge, type, tag, image still carries
      kColBorder. No new column flag and no new make_*_plan free function was
      introduced.
    status: verified
    last_checked: 2026-08-26
  - id: ac-005
    summary: comm plan is 3 columns / 24 B per ghost row
    type: runtime
    evaluator_hint: "ctest --test-dir build -R test_cpu_comm_atom_pack_comm_plan --output-on-failure"
    pass_when: |
      ctest -R test_cpu_comm_atom_pack_comm_plan passes: make_comm_plan on the
      canonical schema yields exactly 3 fields with wire_ids x, y, z and
      comm_bytes(plan, 1) == 3 * sizeof(StorageReal) (24 B at the default
      double storage tier), while make_border_plan on the same frame still
      contains vx, vy, vz, mass, charge, type, tag and image.
    status: verified
    last_checked: 2026-08-26
  - id: ac-006
    summary: HaloRefresh velocity/charge paths survive the comm-plan narrowing
    type: runtime
    evaluator_hint: "ctest -R test_mpi_halo_refresh_charges; run_reg const-pot-charge-halo-mpi"
    pass_when: |
      HaloRefresh::launch sources its plan from make_border_plan
      (halo_exchange.cpp, formerly line 1349); test_mpi_halo_refresh_charges
      passes at every registered rank count and
      regressions/md/const-pot-charge-halo-mpi exits 0 with max |q_ghost -
      q_owner| = 0 within its pinned 1e-15 e tolerance and a non-zero count of
      electrode ghosts with |q| > 1e-12 e.
    status: verified
    last_checked: 2026-08-26
  - id: ac-007
    summary: Existing MPI suites stay green with unchanged assertions
    type: runtime
    evaluator_hint: "ctest --test-dir build -R 'test_mpi_' --output-on-failure"
    pass_when: |
      test_mpi_halo_dual_image_ghosts (n1/n2/n4/n8), test_mpi_halo_exchange,
      test_mpi_evaluate_halo_parity, test_mpi_pppm_brick_parity,
      test_mpi_pppm_subcomm_parity, test_mpi_atom_migration and the three
      *_energy_bruteforce suites all pass, and `git diff` shows no change to any
      assertion in tests/mpi/ (the only allowed edit there is the stale
      kColComm comment at tests/mpi/test_halo_refresh_charges.cpp:94).
    status: verified
    last_checked: 2026-08-26
  - id: ac-008
    summary: Designated regression examples exit 0 with unchanged pins
    type: runtime
    evaluator_hint: "run the five reg_* binaries standalone"
    pass_when: |
      regressions/md/comm-swaplist-halo-mpi, regressions/md/comm-topology-
      multihop, regressions/md/multihop-widebox-parity, regressions/md/const-
      pot-charge-halo-mpi and regressions/longrange/pppm-subcomm-parity-mpi all
      exit 0 within their 60 s budget, reproducing the hard-coded gates already
      in their REFERENCE.md files (for comm-swaplist-halo-mpi: |E_dd -
      E_brute|/N <= 1e-9 eV/atom, max_i |F_i,dd - F_i,brute| <= 1e-9 eV/A,
      ||Sum_i F_i|| <= 1e-10 eV/A, nswap = 6, Sum nrecv = n_ghost = 384,
      halo rebuilds >= 2), with no REFERENCE.md pin value edited.
    status: verified
    last_checked: 2026-08-26
  - id: ac-009
    summary: np=256 widebox single-point parity all-green
    type: runtime
    evaluator_hint: "NP=256 DIMS=8,8,4 tools/bench_gaff_lammps/widebox_parity.sh"
    pass_when: |
      DATA/FF set to the MIL0PEGM18 bench inputs, `NP=256 DIMS=8,8,4
      tools/bench_gaff_lammps/widebox_parity.sh` exits 0 with PASS on every
      component (bond/angle/dihedral rel < 1e-9, pair+kspace and total E rel <
      1e-8 plus |dE|/N < 1e-6 eV/atom, fmax/fnorm rel < 1e-6) and its own
      geometry assertion confirms the requested 8,8,4 grid with at least one
      hops component >= 2.
    status: verified
    last_checked: 2026-08-26
  - id: ac-010
    summary: Per-step recv buffers are sized, not zeroed; ghost forces still zeroed
    type: code
    pass_when: |
      Neither the forward recv buffer (halo_exchange.cpp, formerly :776-781)
      nor the reverse recv_force buffer (formerly :1530-1534) is filled with
      std::byte{0} on a per-step path — both only set size — while
      zero_ghost_forces and its full-rebuild call site remain unchanged, and
      unpack still validates the received byte count.
    status: verified
    last_checked: 2026-08-26
  - id: ac-011
    summary: LAMMPS provenance header + corrected steady-state contract
    type: docs
    pass_when: |
      include/atomiverse/cpu/communication/halo_exchange.h and
      src/cpu/communication/halo_exchange.cpp each carry a header naming
      LAMMPS src/comm_brick.cpp (forward_comm / reverse_comm / borders) as the
      ported discipline in the twin+DOI form used by swap_list.h:2,20, retaining
      the Sandia copyright notice and citing DOI 10.1016/j.cpc.2021.108171; and
      the steady-state paragraph at halo_exchange.h:92-93 now states the new
      contract (zero collectives, zero realloc, zero memset on the refresh
      path) with no sentence left contradicting the code.
    status: verified
    last_checked: 2026-08-26
  - id: ac-012
    summary: Bench snapshot records the new Comm section, no threshold asserted
    type: code
    pass_when: |
      tools/bench_gaff_lammps/README.md contains a dated (2026-08 or later)
      np=256 MIL0PEGM18 section-timing row whose Comm value is the newly
      measured one alongside the LAMMPS 0.441 s reference, and the surrounding
      text states no pass/fail performance threshold.
    status: verified
    last_checked: 2026-08-26
out_of_scope:
  - "QEq updates_owned_charges() declaration / DD-correctness (separate /mol:fix slug)"
  - "Reverse comm_f_only zero-copy (measure-first)"
  - "Dense tag map replacing tag_to_local hashing"
  - "Reneighbor-only coordinate wrapping (blocked on Migrate slab compare)"
  - "kspace remap collective shape (next slug)"
  - "InfraScan-style ghost-velocity consumer probing (no second consumer yet)"
  - "Any CUDA halo change; any performance threshold as a gate"
---

# Acceptance — lammps-comm-port

"Done" 的含义：ATV 的 CPU halo 在稳态步进路径上与 LAMMPS `CommBrick` 同纪律——不发起集合通信、每 ghost 只搬 24 B 坐标、不做接收缓冲 memset——而**任何一个物理数字都没有动**。因此验收由两半组成：静态半（ac-001/002/004/010/011/012）确认纪律真的落在代码里而不是被搬到别处；运行半（ac-003/005/006/007/008/009）确认所有既有 MPI 套件、五个指定 regression 与 np=256 单点 parity 在 pin 与断言完全不变的前提下仍然绿。性能只进 README 快照，不构成任何门槛。

## AC-001 — 每步零集合通信

判定方式是读代码路径而非跑 profiler：从三个 `launch()` 出发的调用图上，非 full-rebuild 分支不得到达任何集合调用。`HaloSend` 的 schema 校验允许留在 `build()` 与 full-rebuild 分支（该分支已 bump `dd->halo_generation`）；`HaloRecv` 允许用 `halo_generation` 变化作为门；`HaloRefresh` 只允许在 `build()` 校验一次。`mpi::Reduce` 明确豁免——NVT 每步需要全局动能，LAMMPS `fix_nh` 同样如此。

## AC-002 — 纯函数化的 rounds 归约

`SwapList` 的"构造性无 MPI"是它的既有契约（`swap_list.cpp:1-6`），新函数不得破坏它。`derive_uniform_rounds` 必须逐轴扫过全部坐标复用 `derive_hops`，复杂度 `O(Σ_d dims[d])`；`setup_swaps` 里 4 元素 `MPI_Allreduce(MAX)` 及其"未触发 rank 收到降级消息"的分支应一并消失。

## AC-003 — 跳数纯函数单测

生产几何的期望值来自闭式 `ceil(shell·dims/L)`：`11·8/63.0016 = 1.397 → 2`（x、y）、`11·4/63.0016 = 0.698 → 1`（z）。拒绝用例的关键不是"抛"，而是"**每个坐标都抛**"——这正是删掉携带 refusal flag 的 Allreduce 之后仍然 rank 一致的证明。与 `derive_hops` 全坐标最大值的一致性检查是旧行为的等价 pin。

## AC-004 / AC-005 — `comm_x_only` 载荷

`kColComm` 从 7 列收缩为 3 列是**成员关系纠正**，不是新开关：ghost 的 v/q 仍由 `kColBorder` 在每次 full rebuild 落地，与 `AtomVec::pack_border` 同构。单测在 `atom_pack` 层直接钉住列集合与行字节数，不需要 MPI，秒级完成。

## AC-006 — HaloRefresh 重指向（承重）

`refresh_columns` 从 plan 中挑字段，挑不到即抛；所以窄化 `kColComm` 与把 refresh 的 plan 源改为 `make_border_plan` 必须同一次落地。`const-pot-charge-halo-mpi` 是这条的行为门：没有 refresh，ghost 电荷会停在上一次解出的值（首步为 0），该 regression 会以 ~1e-2 e 的 `max|dq|` 失败。

## AC-007 / AC-008 / AC-009 — 不变性

本 spec 不产生新物理，因此不新建 oracle 案例（那会制造无来源的 pin）。指定的 regression example 是既有的 `regressions/md/comm-swaplist-halo-mpi`：它的 golden 全部硬编码在 `REFERENCE.md`，oracle 是文件内 O(N²) 最小镜像求和，运行期不导入也不 spawn 任何第三方引擎。widebox parity 用 ATV 自己的 np=1 结果作参考，属于跨 rank 自洽，不得记为 `oracle:` pin。

## AC-010 — memset 与 ghost 力清零的区分

删除的是线上字节缓冲的 `assign(..., byte{0})`；`zero_ghost_forces` 是反向部分和累加器的物理要求，删掉它会静默丢力。审阅时两者必须分别确认。

## AC-011 / AC-012 — 溯源与文档诚实

GPL 溯源头用 swap_list 已有的"孪生 + DOI"形态（按结构重新实现），不用 `rys_roots.h` 的命名空间副本形态。`halo_exchange.h:92-93` 那段自述稳态契约今天被每步集合通信否定，属于第 9 条法则下必须当场修正的文档债。
