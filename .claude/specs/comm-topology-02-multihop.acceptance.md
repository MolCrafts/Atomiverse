---
slug: comm-topology-02-multihop
criteria:
  - id: ac-001
    summary: SwapList derives per-axis hop counts for uniform and RCB layouts
    type: code
    pass_when: |
      SwapList exposes a pure hop-derivation entry taking dims/periods/coords/
      cuts/box-lengths/shell-width and returning max_hops(d); the uniform branch
      computes ceil(need/w) in closed form with no MPI call, and the non-uniform
      branch accumulates target-side slab widths along the send direction and
      reconciles with one MPI_Allreduce(MAX).
    status: verified
    last_checked: 2026-08-26
  - id: ac-002
    summary: Hop-derivation unit test passes with hard-coded expectations
    type: runtime
    evaluator_hint: "ctest -R test_cpu_comm_swap_list_maxneed"
    pass_when: |
      tests/cpu/communication/test_swap_list_maxneed.cpp passes and asserts
      max_hops == 2 for (L=15.78, dims=4, need=6.6), == 4 for dims=8, == 1 for
      dims=2, == 0 for dims=1, and == 3 for the non-uniform case
      (L=20, dims=4, cuts={0,0.5,0.6,0.7,1.0}, need=5.0).
    status: verified
    last_checked: 2026-08-26
  - id: ac-003
    summary: Narrow-subdomain rejection removed; only the min-image bound throws
    type: code
    pass_when: |
      Neither src/cpu/communication/halo_exchange.cpp build() nor launch() throws
      on "shell wider than this rank's subdomain"; the only remaining throw on
      that path fires when max_hops(d) > dims[d]-1 and its message names the
      minimum-image bound (need > L_d - w_self), not the rank count.
    status: verified
    last_checked: 2026-08-26
  - id: ac-004
    summary: A box shrink raises the hop count instead of throwing
    type: runtime
    evaluator_hint: "ctest -R test_cpu_comm_swap_list_maxneed"
    pass_when: |
      The unit test drives setup() twice on one SwapList instance with L=15.78
      then L=10.0 (same dims=4, need=6.6) and observes max_hops 2 -> 3 with no
      exception; repeating the second call is idempotent.
    status: verified
    last_checked: 2026-08-26
  - id: ac-005
    summary: Cell grid pads extended axes by the actual ghost shell width
    type: code
    pass_when: |
      CellList::compute_grid pads extended axes by max(cutoff_skin_,
      dd_->ghost_shell_width), grid_stale() compares the captured shell width,
      DomainDecomposition carries ghost_shell_width [Angstrom] written by
      HaloSend, and neighbor_list.h passes it at the reinit_strategy call site.
    status: verified
    last_checked: 2026-08-26
  - id: ac-006
    summary: Multi-hop ghosts bin inside the grid, never clamped to edge cells
    type: runtime
    evaluator_hint: "ctest -R test_cpu_md_cell_list_halo_pad"
    pass_when: |
      tests/cpu/dynamics/test_cell_list_halo_pad.cpp passes: with cutoff=2.0,
      skin=0.2, ghost_shell_width=6.6, rows at sublo-6.5 and sublo-6.5+cs bin to
      distinct cells with index > 0, ghost_shell_width=0 reproduces the 2.2 A
      pad, and raising the shell width alone makes grid_stale() return true.
    status: verified
    last_checked: 2026-08-26
  - id: ac-007
    summary: Narrow-subdomain MPI runs match the brute-force oracle
    type: runtime
    evaluator_hint: "ctest -R test_mpi_halo_dual_image_ghosts"
    pass_when: |
      test_mpi_halo_dual_image_ghosts_n4 (ATV_FORCE_DIMS=4,1,1) and _n8
      (8,1,1) on the 3^3 / a0=5.26 A box at cutoff 6 A match the in-file O(N^2)
      minimum-image LJ oracle to relative error < 1e-9 with dd->n_ghost > 0, and
      the existing _n1 / _n2 registrations still pass unchanged.
    status: verified
    last_checked: 2026-08-26
  - id: ac-008
    summary: Decomposed axes are still never minimum-imaged
    type: code
    pass_when: |
      CellList::min_image_unextended_ and the axis_extended_[d] = (dims[d] > 1)
      predicate are unchanged, and no new call site min-images a separation on a
      decomposed axis.
    status: verified
    last_checked: 2026-08-26
  - id: ac-009
    summary: Bonded skin and halo generation semantics preserved
    type: code
    pass_when: |
      The shell width stays max(pair cutoff, bonded_extension) with the existing
      MPI_Allreduce(MAX) at halo_exchange.cpp:478-485, ++dd->halo_generation
      still runs on every full rebuild, and no new symbol is named halo_cutoff().
    status: verified
    last_checked: 2026-08-26
  - id: ac-010
    summary: Regression reproduces single-rank energy and forces at 4x1x1
    type: runtime
    evaluator_hint: "ctest -R reg_comm-topology-multihop"
    pass_when: |
      regressions/md/comm-topology-multihop/reg_comm-topology-multihop.cpp
      self-launches np=4 with ATV_FORCE_DIMS=4,1,1 on the 64-atom L=15.78 A LJ
      box (eps=0.0104 eV, sigma=3.4 A, cutoff=6 A) and gates, against
      hard-coded literals only: energy vs the in-file O(N^2) oracle rel <= 1e-9,
      max per-atom force deviation <= 1e-9 eV/A, observed x-axis max_hops >= 2,
      np=4 vs np=1 energy rel <= 1e-12; wall time <= 60 s; no third-party binary
      or import at run time.
    status: verified
    last_checked: 2026-08-26
  - id: ac-011
    summary: REFERENCE.md pins the measured value and cites the ghost semantics
    type: docs
    pass_when: |
      regressions/md/comm-topology-multihop/REFERENCE.md declares
      oracle: literature, pins the measured np=1 energy as a literal table entry
      with its tolerance, and cites Plimpton J. Comput. Phys. 117, 1 (1995),
      DOI 10.1006/jcph.1995.1039 and Thompson et al. Comput. Phys. Commun. 271,
      108171 (2022), DOI 10.1016/j.cpc.2021.108171.
    status: verified
    last_checked: 2026-08-26
  - id: ac-012
    summary: New public symbols carry doxygen with units
    type: docs
    pass_when: |
      SwapList::max_hops / nswap / shell_width, setup(), and
      DomainDecomposition::ghost_shell_width each have @brief, @param with
      Angstrom units where applicable, and @return; the HaloSend header contract
      no longer documents the removed narrow-subdomain rejection.
    status: verified
    last_checked: 2026-08-26
  - id: ac-013
    summary: Full build and test suite green
    type: runtime
    pass_when: |
      cmake --build build -j$(nproc) succeeds and
      ctest --test-dir build --output-on-failure reports zero failures,
      including every MPI registration in tests/CMakeLists.txt.
    status: verified
    last_checked: 2026-08-26
---

# Acceptance criteria

- **ac-001 / ac-002** —— 跳数派生是本 spec 的算法核心，且必须可在串行下判定；纯函数 + 硬编码整数期望使第三方无需读 spec 即可判是非。
- **ac-003** —— "删除拒绝"必须精确：窄子域拒绝消失，最小镜像上界的拒绝保留；两者混淆会把物理错误当成配置错误放行。
- **ac-004** —— 覆盖 NPT/deform 运行时收缩这一条 launch 侧路径：升轮数，而非抛异常、也非静默截断。
- **ac-005 / ac-006** —— cell-list 耦合点。`bin_atoms` 的边缘钳位是静默失效面，因此判据落在"落到不同的非边缘格"这一可观察状态上。
- **ac-007** —— 窄盒下的能量正确性，复用既有 O(N²) 暴力 oracle 与既有窄盒几何；`_n1`/`_n2` 同时作为不回归护栏。
- **ac-008 / ac-009** —— 三条不可动的既有不变量（分解轴不折叠、键接壳宽、halo 世代计数），以源码事实判定。
- **ac-010 / ac-011** —— 交付级端到端证据，位于 `regressions/`，全部为硬编码字面量与第一方 oracle；不含任何性能数字（性能属 -03）。
- **ac-012 / ac-013** —— 文档口径与全量绿灯。
