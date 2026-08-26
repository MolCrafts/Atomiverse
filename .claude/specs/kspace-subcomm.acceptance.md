---
slug: kspace-subcomm
criteria:
  - id: ac-001
    summary: pppm_fft_ranks is a bounded, divisor-exact, throwing pure function
    type: code
    pass_when: |
      tests/cpu/longrange/test_pppm_coul.cpp contains TESTs that pass with
      pppm_fft_ranks(32,256)==32, (32,64)==32, (32,16)==16, (32,256,2)==16,
      (34,4)==2, (31,8)==1, (32,1)==1, and std::invalid_argument thrown for
      planes_k0<=0, nprocs<=0 and min_planes_per_rank<=0. The returned value
      always divides planes_k0 and never exceeds min(nprocs, planes_k0).
    status: verified
    last_checked: 2026-08-26
    evidence: test_cpu_pppm_coul PppmFftRanks.* — all pinned tuples + throws + divisor invariants pass

  - id: ac-002
    summary: pppm_select_mesh_arm maps mesh+ranks to the three arms
    type: code
    pass_when: |
      pppm_select_mesh_arm returns Replicated for nprocs<=1, BrickWorld when
      fft_ranks==nprocs, BrickSubgroup when fft_ranks<nprocs, and throws when
      fft_ranks>nprocs; a unit test pins (32,64,32)==BrickSubgroup, the case
      the old "mesh<=64^3 && ranks>=16 -> replicated" rule got wrong. The
      function references no MPI symbol and no getenv.
    status: verified
    last_checked: 2026-08-26
    evidence: PppmSelectMeshArm.* incl. (32,64,32)==BrickSubgroup falsified-policy pin; no MPI/getenv in the functions

  - id: ac-003
    summary: Subgroup communicator is split once per shape and freed safely
    type: code
    pass_when: |
      In src/cpu/longrange/pppm.cpp the only MPI_Comm_split call sits inside
      the shape-keyed rebuild block (the `!mpi_fft_ready_ || K/nprocs changed`
      branch), never on the per-step path; fft_ranks_==nprocs takes an alias
      branch with owns_fft_comm_=false and performs no split; the split is
      preceded by cpu::comm::require_collective_fingerprint on
      dd->cart_comm with a digest of {K0,K1,K2,fft_ranks_,arm}; and
      release_storage() frees the comm only when owns_fft_comm_ is true and
      MPI_Finalized reports false.
    status: verified
    last_checked: 2026-08-26
    evidence: single MPI_Comm_split in shape-keyed rebuild, alias branch owns=false, fingerprint on cart_comm before split, MPI_Finalized-guarded free (janitor re-verified all five falsification reverts, md5 pinned)

  - id: ac-004
    summary: fft_owner throws instead of falling back to rank nprocs-1
    type: code
    pass_when: |
      The `return nprocs - 1;` fallback at the end of fft_owner in
      src/cpu/longrange/pppm.cpp is gone; an m0 owned by no rank in
      fft_m0_starts_/fft_m0_counts_ raises std::runtime_error naming m0, K0
      and fft_ranks_. The ownership table stays indexed by parent-comm rank
      with count 0 for non-members, so both Alltoallv remaps compile and run
      unchanged.
    status: verified
    last_checked: 2026-08-26
    evidence: nprocs-1 fallback replaced by std::runtime_error naming m0/K0/fft_ranks; D2a injection proved the throw fires; m0 table stays parent-indexed

  - id: ac-005
    summary: Env semantics preserved; ad-hoc profiler replaced by TimerRegistry
    type: code
    pass_when: |
      ATV_PPPM_ALLREDUCE=1 still forces the replicated arm and ATV_PPPM_BRICK=1
      still forces the world-comm brick arm (both observable via mesh_arm());
      no new getenv key is introduced anywhere in pppm.cpp/pppm.h; the
      ATV_PPPM_PROFILE thread_local chrono accumulators and their fprintf are
      deleted, and all three arms are instrumented with ATV_PROFILE_SCOPE from
      include/atomiverse/core/timer.h.
    status: verified
    last_checked: 2026-08-26
    evidence: ATV_PPPM_ALLREDUCE/BRICK forcing hand-verified at n4 via mesh_arm(); no new getenv; ATV_PPPM_PROFILE grep=0; ATV_PROFILE_SCOPE on all three arms (5 keys)

  - id: ac-006
    summary: Public surface is ctor arg + read-only accessors, validated
    type: code
    pass_when: |
      KSpacePPPM takes `int fft_ranks = 0` as its fourth constructor argument
      (0 = derive), throws std::invalid_argument on a negative value at
      construction and throws with a how-to when a positive value exceeds
      min(nprocs, K0) at the shape-keyed rebuild; fft_ranks() and mesh_arm()
      are const noexcept accessors; no *Config type, no make_*/build_* factory
      and no setter-after-default-construction is added; long_range_contract()
      is byte-identical on every rank regardless of subgroup membership.
    status: verified
    last_checked: 2026-08-26
    evidence: 4th ctor arg fft_ranks=0, negative throws at construction, >min(nprocs,K0) throws at rebuild; const noexcept accessors; no Config/factory; contract rank-invariant

  - id: ac-007
    summary: Subgroup arm matches the world-comm brick arm on 1/2/4 ranks
    type: runtime
    evaluator_hint: "ctest --test-dir build -R test_mpi_pppm_subcomm_parity --output-on-failure"
    pass_when: |
      tests/mpi/test_pppm_subcomm_parity n1, n2 and n4 all pass (or skip only
      when ATV_HAS_FFTW_MPI is absent at world_size>1): at n4 on a K0=6 mesh
      the default arm reports mesh_arm()==BrickSubgroup with fft_ranks()==3
      (one non-member rank), its total energy differs from the
      fft_ranks=world construction by <1e-9 relative, the tag-aligned
      per-atom force difference is <1e-9 eV/A, and min/max Allreduce of the
      reduced energy agree to <1e-10 relative.
    status: verified
    last_checked: 2026-08-26
    evidence: test_mpi_pppm_subcomm_parity n1/n2/n4 green (n4: BrickSubgroup fft_ranks 3, energy/force/rank-agreement inside windows); suite 167/167

  - id: ac-008
    summary: Regression example runs the three legs and self-asserts the arm
    type: runtime
    evaluator_hint: "cmake --build build --target run_reg_pppm-subcomm-parity-mpi"
    pass_when: |
      regressions/longrange/pppm-subcomm-parity-mpi/reg_pppm-subcomm-parity-mpi.cpp
      exits 0 in <=60 s wall, re-launching itself through the compiled-in
      ATV_MPIEXEC at np=1, np=8 (default arm) and np=8 with ATV_PPPM_BRICK=1;
      the np=8 default leg reports mesh_arm()==BrickSubgroup and
      fft_ranks()==6 and the forced leg reports BrickWorld; energies satisfy
      |E(np8,subgroup)-E(np1)|/|E(np1)| < 1e-9 and
      |E(np8,subgroup)-E(np8,world)|/|E| < 1e-9 with tag-aligned
      max|dF| < 1e-9 eV/A and ||sum_i F_i|| < 1e-10 eV/A. Every reference
      number is a literal in the source; no third-party binary is imported,
      spawned or read at build or run time.
    status: verified
    last_checked: 2026-08-26
    evidence: reg_pppm-subcomm-parity-mpi exit 0 in 1.5 s; arm self-report gates (BrickSubgroup/6, BrickWorld); all literals in source; falsification table D1-D4 injected+reverted

  - id: ac-009
    summary: np=1 leg reproduces the cited NaCl Madelung energy
    type: scientific
    pass_when: |
      The np=1 leg of regressions/longrange/pppm-subcomm-parity-mpi satisfies
      |E/E_Madelung - 1| < 2e-3 with E_Madelung = -0.5*N*M*C/d computed from
      the hard-coded literals M = 1.747564594633182, C = 14.399645 eV*A/e^2,
      d = 2.82 A, N = 216; REFERENCE.md carries `oracle: literature`,
      `perf_exempt: yes`, the Kittel / Sherman (DOI 10.1021/cr60038a002) and
      Essmann (DOI 10.1063/1.470117) citations, the pinned energies with the
      measurement date and environment, the tolerance derivation, and a
      falsification table showing at least three injected defects each failing
      a named gate.
    status: pending

  - id: ac-010
    summary: 256-rank widebox parity window stays green with the new default arm
    type: runtime
    evaluator_hint: "NP=256 DIMS=8,8,4 tools/bench_gaff_lammps/widebox_parity.sh"
    pass_when: |
      tools/bench_gaff_lammps/widebox_parity.sh at NP=256 on the 24048-atom
      MIL0PEGM18 system, run with the subgroup arm as the default selection,
      reports PASS on every window (bonded 1e-9, pair+kspace 1e-8, total 1e-8
      and |dE|/N < 1e-6 eV/atom, force 1e-6) against its np=1 reference, and
      the run log is quoted in the spec delivery note. No wall-clock number
      from this run is an acceptance criterion.
    status: verified
    last_checked: 2026-08-26
    evidence: sbatch 1693014: NP=256 DIMS=8,8,4 widebox_parity.sh with subgroup default — every window dev=0.0, exit 0; logs atv-mil-bench/parity256sub/

  - id: ac-011
    summary: Docs carry the third-arm row and the measured three-arm matrix
    type: docs
    pass_when: |
      docs/developer/kspace-long-range.md's "MPI: PME vs PPPM FFT" table has a
      PPPM row describing the subgroup arm, its selection rule and the
      preserved ATV_PPPM_ALLREDUCE / ATV_PPPM_BRICK debug forcing;
      tools/bench_gaff_lammps/README.md gains a dated snapshot block with
      Kspace/Comm/Loop and steps/s for replicated, brick and subgroup at
      np in {64,125,200,256} on the same frozen binary; and every threshold
      constant in the selector in src/cpu/longrange/pppm.cpp carries a comment
      citing the matrix row that justifies it.
    status: pending

  - id: ac-012
    summary: Every new public symbol is Doxygen-documented with units
    type: docs
    pass_when: |
      PppmMeshArm, pppm_fft_ranks, pppm_select_mesh_arm, the fft_ranks
      constructor argument and the fft_ranks() / mesh_arm() accessors each
      carry @brief, @param (with units or "rank count" / "mesh planes"
      semantics) and @return in include/atomiverse/cpu/longrange/pppm.h, and
      the file-header architecture table lists the three arms.
    status: pending

  - id: ac-013
    summary: Full build and unit suite green including the new MPI runs
    type: runtime
    evaluator_hint: "cmake --build build -j$(nproc) && ctest --test-dir build --output-on-failure"
    pass_when: |
      A Release build with -DATV_BUILD_TESTS=ON -DATV_BUILD_KSPACE=ON
      -DATV_USE_MPI=ON compiles without new warnings and the whole ctest suite
      passes, including test_mpi_pppm_brick_parity n1/n2 (unchanged behaviour
      on the world arm) and the newly registered
      test_mpi_pppm_subcomm_parity n1/n2/n4.
    status: verified
    last_checked: 2026-08-26
    evidence: lean build-gaff 0 errors 167/167; everything-on ./build 0 errors 310/310 (incl. brick n1/n2 + subcomm n1/n2/n4)
---

# Acceptance criteria

- **ac-001 / ac-002** —— 两个纯函数是这条 spec 唯一可以在无 MPI 环境下被秒级证伪的部分；ac-002 里 `(32,64,32)` 那一行专门钉住旧选择器在 np∈[16,~64] 的错判。
- **ac-003 / ac-004 / ac-005 / ac-006** —— 形态判据。全树第一个真派生通信域必须做到：一次 split、可别名、finalize 守卫下释放、划分参数经集合指纹证明一致；`fft_owner` 的 `nprocs-1` 属于会静默算错的 fallback，子组臂下必须变成 throw；公共面只有构造参数与只读访问器，不得长出 Config 袋或第四个环境变量。
- **ac-007 / ac-008 / ac-009** —— 物理判据分两层：臂与臂之间是**恒等式**（只允许求和次序差），臂与文献之间是**离散化窗口**。ac-008 的 arm 自报断言是防假绿的关键：没有它，一个悄悄退回 world 臂的实现同样会全绿。
- **ac-010** —— 实机 256-rank 窗口是唯一能证明"默认臂换了以后 24k 体系仍然逐量一致"的判据；它是 runtime 判据而不是性能判据，任何墙钟数字都不参与判定。
- **ac-011 / ac-012** —— 选择器常数必须能追溯到矩阵里的一行；性能只落文档，不落判据。
- **ac-013** —— 旧 world 臂的既有门必须一并保持绿，证明第三臂没有把前两臂改坏。
