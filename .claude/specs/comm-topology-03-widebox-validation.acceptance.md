---
slug: comm-topology-03-widebox-validation
criteria:
  - id: ac-001
    summary: Forced narrow production grid is pinned as an 8-rank MPI unit case
    type: code
    pass_when: |
      tests/mpi/test_halo_dual_image_ghosts.cpp contains a case that sets
      ATV_FORCE_DIMS=4,2,1 on a 24x14x14 A box at cutoff 6.0 / skin 0.6
      (halo 6.6 A: x span 6.0 in the 2-hop band, y span 7.0 single-hop
      dual-image; the originally-specified 24x12x12 violates the
      minimum-image bound at dims_y=2 and cannot exist), skips unless
      world == 8, and asserts total energy vs the file's O(N^2) minimum-image
      brute force within rel < 1e-9; tests/CMakeLists.txt registers
      test_mpi_halo_dual_image_ghosts_n8, and
      `ctest -R test_mpi_halo_dual_image_ghosts` is green at n1, n2 and n8.
    status: verified
    last_checked: 2026-08-26
    evidence: ProductionGridTwoNarrowAxesMatchBruteForce (4,2,1 / 24x14x14) in test_halo_dual_image_ghosts.cpp; n1/n2/n8 registered, full suite green
  - id: ac-002
    summary: run_atv_gaff --single-point reports rank-invariant forces and DD geometry
    type: code
    pass_when: |
      A build configured with -DATV_USE_MPI=ON -DATV_BUILD_KSPACE=ON
      -DATV_BUILD_BENCH_GAFF=ON compiles target run_atv_gaff, and one
      `mpirun -np 2 run_atv_gaff --data <any> --single-point` run prints both
      `[atv] SINGLE-POINT FORCES: fmax=<v> fnorm=<v>` (MPI_Allreduce MAX / SUM
      over owner rows only) and `[atv] GRID dims=<dx,dy,dz> span=<sx,sy,sz>
      halo=<r> hops=<hx,hy,hz>` derived from DomainDecomposition::choose_dims;
      runs without --single-point print byte-identical output to the
      pre-change binary.
    status: verified
    last_checked: 2026-08-26
    evidence: FORCES+GRID lines land in bench logs (parity256/log.*); non-single-point path unchanged
  - id: ac-003
    summary: widebox_parity.sh fails loudly instead of always printing PASS
    type: runtime
    pass_when: |
      tools/bench_gaff_lammps/widebox_parity.sh run twice on the same input:
      (a) with an impossible window (--tol-total 0) exits non-zero and prints a
      FAIL line carrying got / want / tol; (b) with default windows on an
      np=1 vs np=1 pair exits 0 and prints PASS for every component
      (bond, angle, dihedral, pair+kspace, total, fmax, fnorm); and a run whose
      np>1 log reports every `hops` component <= 1 is reported FAIL as
      "not a multi-hop geometry".
    status: verified
    last_checked: 2026-08-26
    evidence: bite-verified: --tol-total 0 FAILs nonzero; hops<=1 rejected as non-multi-hop
  - id: ac-004
    summary: Multi-hop wide-box regression reproduces brute force and rank parity
    type: runtime
    evaluator_hint: "target: run_reg_multihop-widebox-parity"
    pass_when: |
      regressions/md/multihop-widebox-parity/reg_multihop-widebox-parity.cpp
      (built with -DATV_BUILD_REGRESSIONS=ON -DATV_USE_MPI=ON) runs to exit 0
      in <= 60 s, having gated, with values hard-coded in the case (no external
      data, no third party at run time):
      (1) np=1 total energy vs the in-case O(N^2) minimum-image direct sum,
          relative <= 1e-9, and per-atom force components <= 1e-9 eV/A;
      (2) np=8 under ATV_FORCE_DIMS=4,2,1 (24x14x14 A geometry) vs np=1,
          energy relative <= 1e-10 and tag-aligned per-atom force
          components <= 1e-9 eV/A;
      and regressions/md/multihop-widebox-parity/REFERENCE.md cites Plimpton
      J. Comput. Phys. 117, 1 (1995) and Allen & Tildesley 2nd ed. §1.6.3 with
      the physical justification of those tolerances.
    status: verified
    last_checked: 2026-08-26
    evidence: run_reg_multihop-widebox-parity exit 0 in regression suite (E=-1.9390130589549823 eV family, hard-coded)
  - id: ac-005
    summary: 24k GAFF MIL0PEGM18 runs at 256 ranks and matches the np=1 single point
    type: runtime
    pass_when: |
      One sbatch job on a single node runs
      `NP=256 DIMS=8,8,4 tools/bench_gaff_lammps/widebox_parity.sh` against
      DATA=/home/jicli594/work/salamander/MIL0PEGM18/aMIL0PEGM18_n20_LiTFSI_n108/eq2/eq.data
      and FF=<same dir>/system.ff with --alpha / --grid-spacing / --cutoff /
      --tail / --shift passed explicitly and identically to both runs; the
      saved np=256 log shows `[atv] GRID` with at least one `hops` component
      >= 2, and the script exits 0 with every default window met:
      bond/angle/dihedral relative <= 1e-9, pair+kspace relative <= 1e-8,
      total energy relative <= 1e-8 and |dE|/N <= 1e-6 eV/atom,
      fmax and fnorm relative <= 1e-6 (denominators max(1, |reference|)).
    status: verified
    last_checked: 2026-08-26
    evidence: sbatch 1692800: NP=256 DIMS=8,8,4 on MIL0PEGM18 eq.data, hops=2,2,1, all windows dev=0.0, exit 0; logs atv-mil-bench/parity256/
  - id: ac-006
    summary: Bench README carries a dated 256-rank snapshot marked non-binding
    type: docs
    pass_when: |
      tools/bench_gaff_lammps/README.md gains a `## Snapshot (<ISO date>, ...)`
      block, in the file's existing LAMMPS-aligned section-name format, that
      records for the 256-rank MIL0PEGM18 run: atom count, box, processor grid
      8x8x4, subdomain span vs cutoff+skin, steps/s, the Pair/Bond/Kspace/
      Neigh/Comm/Output/Modify/Other timing table, the same-node same-precision
      LAMMPS reference, the exact srun command and the dataset path, plus one
      sentence stating these numbers are a record and not an acceptance
      criterion. No numeric performance threshold is asserted anywhere in the
      spec's criteria.
    status: verified
    last_checked: 2026-08-26
    evidence: README.md Snapshot (2026-08-26) block: grid 8x8x4, span vs halo, steps/s, 8-section table, LAMMPS ref, srun cmd, record-not-acceptance sentence
---

# Acceptance criteria

- **ac-001 / ac-002 (`code`)** — 仓库内的可重复证据。ac-001 是这条链在 ctest 里的红绿开关：没有 -02 的多跳 halo，`HaloSend::build` 会抛 "ghost shell wider than this rank's subdomain"（`src/cpu/communication/halo_exchange.cpp:498`），用例必红。ac-002 只绑 bench 工具新增的两条 rank 无关输出行与"非 `--single-point` 路径输出不变"。
- **ac-003 (`runtime`)** — 门本身要能咬人。一个永远打印 PASS 的 parity 脚本比没有脚本更糟；`--tol-total 0` 必须 FAIL，`hops <= 1` 必须被判为"不是多跳几何"。
- **ac-004 (`runtime`)** — 本 spec 唯一的 regression 判据，全部参考值与容差硬编码在案例内，不依赖外部数据集、不在运行期调用任何第三方；oracle 是最小镜像直接求和（模型 Hamiltonian 的定义）加分解不变性，不是被冻结的 ATV 输出。
- **ac-005 (`runtime`)** — 实机能力判据：**能跑** + **与 np=1 逐项一致**。geometry 断言（`hops >= 2`）保证跑的确实是多跳几何而不是被 `choose_dims` 悄悄换成了单跳网格。窗口越界按缺陷处理并上报，不得在本 spec 内放宽。
- **ac-006 (`docs`)** — 性能只留档不设阈值：这是仓库法则"验收只绑正确性与能力"的显式落实，因此没有任何 `type: performance` 判据。
