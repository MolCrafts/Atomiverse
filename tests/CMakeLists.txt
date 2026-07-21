include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Unit tests assert CODE CORRECTNESS on tiny systems (tens to hundreds of
# atoms); performance and LAMMPS-reference validation live in benchmarks/.
# On many-core nodes the OpenMP default (= all hardware threads, e.g. 288 on
# a GH200 node) makes those tiny systems WALL-DOMINATED by fork/join barriers
# — 20-atom 5000-step runs took minutes, and `ctest -j` multiplied the
# oversubscription. Cap every test's default; a test that genuinely needs a
# specific thread count can still call omp_set_num_threads itself.
set(ATV_TEST_OMP_THREADS 8 CACHE STRING
    "OMP_NUM_THREADS for single-process unit tests")
# MPI tests exercise multi-RANK behaviour (domain decomposition / halo), not
# OpenMP — so 1 thread/rank is both sufficient and correct. An n8 run at 4
# threads/rank spawns 8×4=32 contending threads, which on a shared login node
# oversubscribes badly: test_mpi_npt_multirank_n8 took ~103 s AND its
# thread-reduction-order-sensitive pressure check flaked. At 1 thread/rank it
# passes deterministically in ~20 s.
set(ATV_TEST_MPI_OMP_THREADS 1 CACHE STRING
    "OMP_NUM_THREADS per rank for MPI unit tests")

# Fast-gate discipline: this suite is unit tests only — each exercises one
# function or method at millisecond scale. End-to-end runs (a full SCF/MD/vib
# solve to convergence, a multi-sweep MC equilibration, an external-engine
# parity check) do not belong here at any granularity; their home is
# benchmarks/<domain>/validation, run at acceptance time. There is deliberately
# no opt-in switch to run e2e from ctest.

# CUDA test: compiled as .cu with separable compilation.
# A .cu source registered while the CUDA language is disabled is silently
# dropped by CMake — the binary links, runs zero tests, and reports PASS.
# Refuse the configuration instead of shipping a green that tested nothing.
function(atv_guard_cuda_sources name)
    foreach(src ${ARGN})
        if(src MATCHES "\\.cu$" AND NOT ATV_BUILD_CUDA)
            message(FATAL_ERROR
                "test '${name}' has a CUDA source (${src}) but ATV_BUILD_CUDA "
                "is OFF: CMake would silently drop it and the test binary "
                "would report PASS with zero tests. Enable CUDA or do not "
                "register this test.")
        endif()
    endforeach()
endfunction()

function(atv_add_test name)
    atv_guard_cuda_sources(${name} ${ARGN})
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE atomiverse GTest::gtest_main)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${ATV_TEST_OMP_THREADS}")
endfunction()

# C++ test: compiled as .cpp, no CUDA required.
function(atv_add_cpp_test name)
    atv_guard_cuda_sources(${name} ${ARGN})
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE atomiverse GTest::gtest_main)
    set_target_properties(${name} PROPERTIES CXX_STANDARD 17)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${ATV_TEST_OMP_THREADS}")
endfunction()

# MPI test executable: one build target, can be launched at multiple rank
# counts via `atv_add_mpi_run`. The test source provides its own `main`
# that wraps MPI init/finalize.
function(atv_add_mpi_exec target_name)
    add_executable(${target_name} ${ARGN})
    target_link_libraries(${target_name} PRIVATE atomiverse GTest::gtest MPI::MPI_CXX)
    set_target_properties(${target_name} PROPERTIES CXX_STANDARD 17)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# Register one CTest invocation at a given rank count for an MPI executable.
# Tagged with the "mpi" CTest label so CI can skip them on MPI-less environments.
function(atv_add_mpi_run test_name target_name n)
    add_test(
        NAME ${test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target_name}> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(${test_name} PROPERTIES
        LABELS "mpi"
        RUN_SERIAL TRUE
        RESOURCE_LOCK "mpi_singleton"
        ENVIRONMENT "OMP_NUM_THREADS=${ATV_TEST_MPI_OMP_THREADS}"
        PASS_REGULAR_EXPRESSION "PASSED[^a-zA-Z]"
        FAIL_REGULAR_EXPRESSION "FAILED[^a-zA-Z]")
endfunction()

# ── Core type tests (always available, no CUDA required) ──────────────

atv_add_cpp_test(test_mat3  core/test_mat3.cpp)
atv_add_cpp_test(test_vec3  core/test_vec3.cpp)
atv_add_cpp_test(test_real_harmonics core/test_real_harmonics.cpp)
atv_add_cpp_test(test_box   core/test_box.cpp)
atv_add_cpp_test(test_units core/test_units.cpp)

add_test(
    NAME test_molrs_source_contract_mismatch
    COMMAND ${CMAKE_COMMAND}
            -DATV_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/molrs-contract-mismatch
            -P ${CMAKE_CURRENT_SOURCE_DIR}/architecture/molrs_contract_mismatch.cmake)
set_tests_properties(test_molrs_source_contract_mismatch PROPERTIES
    WILL_FAIL TRUE)

# ── CPU backend tests (always available, no CUDA required) ──────────

atv_add_cpp_test(test_cpu_box cpu/test_box.cpp)
# Context service slices (arch-refactor-05-context): flat core fields + nested
# md/mc/opt/analysis pure-aggregate slices, exact field mapping, all-null
# defaults, trivially-copyable / non-polymorphic guards. Layout-only, no
# physics.
atv_add_cpp_test(test_cpu_component_context_slices cpu/component/test_context_slices.cpp)
atv_add_cpp_test(test_cpu_mc_scaffold cpu/test_mc_scaffold.cpp)
atv_add_cpp_test(test_cpu_mc_move cpu/mc/test_mc_move.cpp)
atv_add_cpp_test(test_cpu_mc_transaction_rollback cpu/mc/test_transaction_rollback.cpp)
atv_add_cpp_test(test_cpu_mc_ensemble_policies cpu/mc/test_ensemble_policies.cpp)
atv_add_cpp_test(test_cpu_mc_gcmc_npt_movers cpu/mc/test_gcmc_npt_movers.cpp)
# The MC physics-validation gates (pair_distribution / acceptance_ratio /
# lj_nist) drive MC::run over many sweeps — e2e, not unit. They now live in
# benchmarks/mc/lj/validation/ and run at acceptance time, never in ctest.
atv_add_cpp_test(test_cpu_md cpu/md/test_md.cpp)
atv_add_cpp_test(test_cpu_md_pe_thermo_schema cpu/md/test_pe_thermo_schema.cpp)
atv_add_cpp_test(test_cpu_md_pe_single_chain cpu/md/test_pe_single_chain.cpp)
atv_add_cpp_test(test_cpu_md_pe_nve               cpu/md/test_pe_nve.cpp)
# test_cpu_md_csvr / test_cpu_md_nvt relocated to benchmarks/md/lj/validation/
# (val_md_csvr / val_md_nvt) — full thermostat equilibrations, not ms unit tests.
atv_add_cpp_test(test_cpu_md_atom_sort            cpu/md/test_atom_sort.cpp)
atv_add_cpp_test(test_cpu_md_step_timing          cpu/md/test_step_timing.cpp)
atv_add_cpp_test(test_cpu_md_timing_report        cpu/md/test_timing_report.cpp)
atv_add_cpp_test(test_cpu_md_pe_fd_force          cpu/md/test_pe_fd_force.cpp)
atv_add_cpp_test(test_cpu_md_pe_multi_chain       cpu/md/test_pe_multi_chain.cpp)
atv_add_cpp_test(test_cpu_md_lj_exclusions        cpu/md/test_lj_exclusions.cpp)
atv_add_cpp_test(test_cpu_md_pair_lj_cut          cpu/md/test_pair_lj_cut.cpp)
atv_add_cpp_test(test_cpu_pair_kernel             cpu/potentials/test_pair_kernel.cpp)
atv_add_cpp_test(test_cpu_pair_coul_cut           cpu/potentials/test_pair_coul_cut.cpp)
atv_add_cpp_test(test_cpu_pair_special_scale      cpu/potentials/test_pair_special_scale.cpp)
atv_add_cpp_test(test_cpu_coul_long_special       cpu/potentials/test_coul_long_special.cpp)
atv_add_cpp_test(test_cpu_pair_lj_cut_coul_long   cpu/potentials/test_pair_lj_cut_coul_long.cpp)
atv_add_cpp_test(test_cpu_dihedral_charmm         cpu/potentials/test_dihedral_charmm.cpp)
atv_add_cpp_test(test_cpu_improper_harmonic       cpu/potentials/test_improper_harmonic.cpp)
atv_add_cpp_test(test_cpu_compiled_forcefield     cpu/io/test_compiled_forcefield.cpp)
atv_add_cpp_test(test_cpu_lammps_data_full        cpu/io/test_lammps_data_full.cpp)
atv_add_cpp_test(test_cpu_lammps_real             cpu/io/test_lammps_real.cpp)
atv_add_cpp_test(test_cpu_bias_distance_cv        cpu/bias/test_distance_cv.cpp)
atv_add_cpp_test(test_cpu_ewald_sum               cpu/longrange/test_ewald_sum.cpp)
atv_add_cpp_test(test_cpu_pme_math                cpu/longrange/test_pme_math.cpp)
atv_add_cpp_test(test_cpu_ewald_coul              cpu/longrange/test_ewald_coul.cpp)
if(ATV_BUILD_KSPACE)
    atv_add_cpp_test(test_cpu_pme_coul            cpu/longrange/test_pme_coul.cpp)
    atv_add_cpp_test(test_cpu_pppm_coul           cpu/longrange/test_pppm_coul.cpp)
endif()
atv_add_cpp_test(test_cpu_am1_scaffold            cpu/semiempirical/test_am1_scaffold.cpp)
atv_add_cpp_test(test_cpu_rm1                     cpu/semiempirical/test_rm1.cpp)

# GPW module tests. All of them run unconditionally on the in-tree GTH + MOLOPT +
# D3 tables (cpu/gpw/basis_data/) — no external data directory is consulted at
# build or test time. (Until 2026-07-11 the 14 tests that touched a real molecule
# only registered when ATV_CP2K_DATA_DIR pointed at a CP2K install, so with the
# gate off GPW had no real-molecule physics and no force validation at all.)
if(ATV_BUILD_GPW)
    # Every GPW test below runs on the in-tree data tables
    # (cpu/gpw/basis_data/gpw_data.h) — no external data directory, no
    # third-party file read. E2E (converged SCF / AIMD / geoopt / anchors) lives
    # in benchmarks/qc/gpw/validation and never runs under ctest.
    atv_add_cpp_test(test_cpu_gpw_gth_integrals    cpu/gpw/test_cpu_gpw_gth_integrals.cpp)
    atv_add_cpp_test(test_cpu_gpw_collocate        cpu/gpw/test_cpu_gpw_collocate.cpp)
    atv_add_cpp_test(test_cpu_gpw_multigrid        cpu/gpw/test_cpu_gpw_multigrid.cpp)
    if(ATV_BUILD_PW)  # FFT transfer (prolong_fft / restrict_fft) needs FFTW3
        target_link_libraries(test_cpu_gpw_multigrid PRIVATE fftw3_stub)
        target_compile_definitions(test_cpu_gpw_multigrid PRIVATE ATV_GPW_HAS_FFT)
    endif()
    atv_add_cpp_test(test_cpu_gpw_xc_grid          cpu/gpw/test_cpu_gpw_xc_grid.cpp)
    atv_add_cpp_test(test_cpu_gpw_one_electron     cpu/gpw/test_cpu_gpw_one_electron.cpp)
    atv_add_cpp_test(test_cpu_xc_pade              cpu/xc/test_cpu_xc_pade.cpp)
    if(ATV_BUILD_PW)  # FFT-backed pieces
        atv_add_cpp_test(test_cpu_gpw_poisson_fft  cpu/gpw/test_cpu_gpw_poisson_fft.cpp)
        atv_add_cpp_test(test_cpu_gpw_hartree_grid cpu/gpw/test_cpu_gpw_hartree_grid.cpp)
        atv_add_cpp_test(test_cpu_gpw_grid_fft     cpu/gpw/test_cpu_gpw_grid_fft.cpp)
        atv_add_cpp_test(test_cpu_gpw_xc_grid_gga  cpu/gpw/test_cpu_gpw_xc_grid_gga.cpp)
        atv_add_cpp_test(test_cpu_gpw_mt_poisson   cpu/gpw/test_cpu_gpw_mt_poisson.cpp)
        atv_add_cpp_test(test_cpu_gpw_solver       cpu/gpw/test_cpu_gpw_solver.cpp)
    endif()

    # Data-table unit tests: grammar/ingest of the in-tree GTH + MOLOPT tables and
    # the D3 parameter tables. No SCF is run here — the converged-SCF checks that
    # used to live alongside these are now benchmarks/qc/gpw/validation.
    atv_add_cpp_test(test_cpu_gpw_data_table    cpu/gpw/test_cpu_gpw_data_table.cpp)
    # Minimal model system for the Γ-point lattice sum: S must be positive definite
    # at every cell size (Gram matrix). Pinned against PySCF.
    atv_add_cpp_test(test_cpu_gpw_periodic_overlap cpu/gpw/test_cpu_gpw_periodic_overlap.cpp)
    # Heavy elements: Zn/Na/Cl GTH-PBE +
    # DZVP-MOLOPT-SR-GTH ingest, incl. the Zn d/f channels.
    atv_add_cpp_test(test_cpu_gpw_heavy_elements cpu/gpw/test_cpu_gpw_heavy_elements.cpp)
    # Grimme DFT-D3 (D3): energy + analytic force incl. the CN-derivative
    # term, against a finite-difference check. Reference-free.
    atv_add_cpp_test(test_cpu_gpw_dftd3         cpu/gpw/test_cpu_gpw_dftd3.cpp)
endif()
atv_add_cpp_test(test_cpu_md_pair_morse_smoke     cpu/md/test_pair_morse_smoke.cpp)
atv_add_cpp_test(test_cpu_md_pe_ff_constants      cpu/md/test_pe_ff_constants.cpp)
atv_add_cpp_test(test_cpu_md_pe_energy_xcheck     cpu/md/test_pe_energy_xcheck.cpp)
# pe_tension_protocol grepped benchmarks/md/pe/... which was deleted; re-add
# when the PE tension app is restored under regressions/ or examples/.
atv_add_cpp_test(test_cpu_md_neighbor_list_arena  cpu/md/test_neighbor_list_arena.cpp)
atv_add_cpp_test(test_cpu_md_neighbor_list_rebuild  cpu/md/test_neighbor_list_rebuild.cpp)
atv_add_cpp_test(test_cpu_md_full_neighbor_list  cpu/md/test_full_neighbor_list.cpp)
atv_add_cpp_test(test_cpu_md_cell_list_grid_stale  cpu/md/test_cell_list_grid_stale.cpp)
atv_add_cpp_test(test_cpu_atom_variable_n  cpu/test_atom_variable_n.cpp)
atv_add_cpp_test(test_cpu_frame_registry  cpu/frame/test_frame_registry.cpp)
atv_add_cpp_test(test_cpu_eam  cpu/potentials/test_eam.cpp)
atv_add_cpp_test(test_cpu_sw  cpu/potentials/test_sw.cpp)
atv_add_cpp_test(test_cpu_pair_lj_types  cpu/potentials/test_pair_lj_types.cpp)
atv_add_cpp_test(test_cpu_md_fix_rattle  cpu/md/test_fix_rattle.cpp)
atv_add_cpp_test(test_cpu_md_write_sets  cpu/md/test_write_sets.cpp)
atv_add_cpp_test(test_cpu_path_replica    cpu/path/test_replica.cpp)
atv_add_cpp_test(test_cpu_path_remd       cpu/path/test_remd.cpp)
atv_add_cpp_test(test_cpu_criterion       cpu/driver/test_criterion.cpp)
atv_add_cpp_test(test_cpu_driver_nesting  cpu/driver/test_driver_nesting.cpp)
# scf-driver-refactor-01-plumb: scf::Phase / PhaseKind::SCF / PhaseTraits<scf::Phase>
# / the Criterion component kind + driver pull-poll. Pure type-foundation plumbing —
# no consumer yet.
atv_add_cpp_test(test_cpu_scf_phase_plumbing cpu/driver/test_scf_phase_plumbing.cpp)
atv_add_cpp_test(test_cpu_component_ownership cpu/driver/test_component_ownership.cpp)
# test_cpu_md_pe_nvt relocated to benchmarks/md/pe/validation/ (val_md_pe_nvt) —
# a full NVT equilibration of the bonded PE melt, not a ms unit test.
atv_add_cpp_test(test_cpu_md_frame_tag_image      cpu/md/test_frame_tag_image.cpp)
atv_add_cpp_test(test_cpu_md_domain_decomposition cpu/md/test_domain_decomposition.cpp)
atv_add_cpp_test(test_cpu_md_mpi_stubs            cpu/md/test_mpi_stubs.cpp)
atv_add_cpp_test(test_cpu_analysis_executor       cpu/test_analysis_executor.cpp)
atv_add_cpp_test(test_cpu_snapshot_publisher      cpu/test_snapshot_publisher.cpp)
atv_add_cpp_test(test_cpu_async_component_base    cpu/test_async_component_base.cpp)
atv_add_cpp_test(test_cpu_observer_rdf_async      cpu/observer/test_rdf_observer.cpp)
atv_add_cpp_test(test_cpu_observer_msd            cpu/observer/test_msd.cpp)
atv_add_cpp_test(test_cpu_observer_vacf           cpu/observer/test_vacf.cpp)
atv_add_cpp_test(test_cpu_md_no_alloc              cpu/test_md_no_alloc.cpp)
atv_add_cpp_test(test_cpu_driver_role_validation cpu/test_cpu_driver_role_validation.cpp)
# Registration-time PhaseKind gate + five-driver plan-shape regression
# (arch-refactor-01-bindings). Uses AM1Evaluator (unconditional, no RHF gate).
atv_add_cpp_test(test_cpu_driver_binding_kind             cpu/test_driver_binding_kind.cpp)
atv_add_cpp_test(test_cpu_evaluate_driver_capabilities    cpu/test_evaluate_driver_capabilities.cpp)
atv_add_cpp_test(test_cpu_optimize_driver_early_stop      cpu/test_optimize_driver_early_stop.cpp)
atv_add_cpp_test(test_cpu_dynamics_full_budget            cpu/test_dynamics_full_budget.cpp)
atv_add_cpp_test(test_cpu_install_infra_hook              cpu/test_install_infra_hook.cpp)
atv_add_cpp_test(test_cpu_md_set_api                      cpu/md/test_md_set_api.cpp)
atv_add_cpp_test(test_cpu_md_plan_equivalence             cpu/md/test_md_plan_equivalence.cpp)
atv_add_cpp_test(test_cpu_minimize_evaluate_set_api       cpu/test_minimize_evaluate_set_api.cpp)
atv_add_cpp_test(test_cpu_materialize                     cpu/test_materialize.cpp)
atv_add_cpp_test(test_cpu_molrs_contract                  cpu/test_molrs_contract.cpp)

# cpu/task infrastructure cluster (mpi-launcher-embedded-python-02-codec):
# TaskSpec data model + MPI-free byte codec + rank-agnostic materializer.
# Serial, no mpirun; the AM1 single point is the ms-scale H2 solve already
# established by test_cpu_am1_scaffold.
atv_add_cpp_test(test_cpu_task_codec                      cpu/test_task_codec.cpp)
atv_add_cpp_test(test_cpu_task_materializer               cpu/test_task_materializer.cpp)

# LauncherGate (mpi-launcher-embedded-python-04, ac-002): the PID-qualified
# ATV_LAUNCHER marker probe. Serial gtest — MPI is never initialized in this
# binary, which is exactly the state the degradation contract is about.
atv_add_cpp_test(test_cpu_launcher_gate                   cpu/task/test_launcher_gate.cpp)

# LauncherArgs (same link, ac-002/ac-006): header-only class in
# app/atomiverse/, which is NOT part of libatomiverse — the test compiles the
# header directly via this include dir.
atv_add_cpp_test(test_app_launcher_args                   app/test_launcher_args.cpp)
target_include_directories(test_app_launcher_args PRIVATE
    ${CMAKE_SOURCE_DIR}/app/atomiverse)

# InterpreterLocator (mpi-launcher-embedded-python-05, ac-004/ac-005): pure
# path-resolution + version-string logic — Python.h-free, no popen/exec, no
# interpreter ever executed. The TU lives in app/atomiverse/ (NOT part of
# libatomiverse), so this target compiles it directly as a second source.
atv_add_cpp_test(test_app_interpreter_locator
    app/test_interpreter_locator.cpp
    ${CMAKE_SOURCE_DIR}/app/atomiverse/interpreter_locator.cpp)
target_include_directories(test_app_interpreter_locator PRIVATE
    ${CMAKE_SOURCE_DIR}/app/atomiverse)

# ExtXYZ loading through molrs — inline fixtures only, no external file.
atv_add_cpp_test(test_cpu_io_extxyz                       cpu/io/test_cpu_io_extxyz.cpp)

# ── MPI tests (launched under `mpirun`) ─────────────────────────────────

if(ATV_USE_MPI AND MPIEXEC_EXECUTABLE)
    atv_add_mpi_exec(test_mpi_topology mpi/test_topology.cpp)
    atv_add_mpi_run(test_mpi_topology_n1 test_mpi_topology 1)
    atv_add_mpi_run(test_mpi_topology_n2 test_mpi_topology 2)
    atv_add_mpi_run(test_mpi_topology_n4 test_mpi_topology 4)
    atv_add_mpi_run(test_mpi_topology_n8 test_mpi_topology 8)

    atv_add_mpi_exec(test_mpi_halo_dual_image_ghosts mpi/test_halo_dual_image_ghosts.cpp)
    atv_add_mpi_run(test_mpi_halo_dual_image_ghosts_n1 test_mpi_halo_dual_image_ghosts 1)
    atv_add_mpi_run(test_mpi_halo_dual_image_ghosts_n2 test_mpi_halo_dual_image_ghosts 2)

    atv_add_mpi_exec(test_mpi_bcast_frame mpi/test_bcast_frame.cpp)
    atv_add_mpi_run(test_mpi_bcast_frame_n1 test_mpi_bcast_frame 1)
    atv_add_mpi_run(test_mpi_bcast_frame_n2 test_mpi_bcast_frame 2)
    atv_add_mpi_run(test_mpi_bcast_frame_n4 test_mpi_bcast_frame 4)

    atv_add_mpi_exec(test_mpi_halo_exchange mpi/test_halo_exchange.cpp)
    atv_add_mpi_run(test_mpi_halo_exchange_n1 test_mpi_halo_exchange 1)
    atv_add_mpi_run(test_mpi_halo_exchange_n2 test_mpi_halo_exchange 2)
    atv_add_mpi_run(test_mpi_halo_exchange_n8 test_mpi_halo_exchange 8)

    atv_add_mpi_exec(test_mpi_lj_nve mpi/test_lj_nve.cpp)
    atv_add_mpi_run(test_mpi_lj_nve_n1 test_mpi_lj_nve 1)
    atv_add_mpi_run(test_mpi_lj_nve_n8 test_mpi_lj_nve 8)

    atv_add_mpi_exec(test_mpi_lj_energy_bruteforce
                     mpi/test_lj_energy_bruteforce.cpp)
    atv_add_mpi_run(test_mpi_lj_energy_bruteforce_n1
                    test_mpi_lj_energy_bruteforce 1)
    atv_add_mpi_run(test_mpi_lj_energy_bruteforce_n8
                    test_mpi_lj_energy_bruteforce 8)

    atv_add_mpi_exec(test_mpi_coul_energy_bruteforce
                     mpi/test_coul_energy_bruteforce.cpp)
    atv_add_mpi_run(test_mpi_coul_energy_bruteforce_n1
                    test_mpi_coul_energy_bruteforce 1)
    atv_add_mpi_run(test_mpi_coul_energy_bruteforce_n8
                    test_mpi_coul_energy_bruteforce 8)

    if(ATV_BUILD_KSPACE)
        atv_add_mpi_exec(test_mpi_pme_nacl_parity
                         mpi/test_pme_nacl_parity.cpp)
        atv_add_mpi_run(test_mpi_pme_nacl_parity_n1
                        test_mpi_pme_nacl_parity 1)
        atv_add_mpi_run(test_mpi_pme_nacl_parity_n2
                        test_mpi_pme_nacl_parity 2)
        atv_add_mpi_exec(test_mpi_pppm_brick_parity
                         mpi/test_pppm_brick_parity.cpp)
        atv_add_mpi_run(test_mpi_pppm_brick_parity_n1
                        test_mpi_pppm_brick_parity 1)
        atv_add_mpi_run(test_mpi_pppm_brick_parity_n2
                        test_mpi_pppm_brick_parity 2)
    endif()

    atv_add_mpi_exec(test_mpi_bonded_energy_bruteforce
                     mpi/test_bonded_energy_bruteforce.cpp)
    atv_add_mpi_run(test_mpi_bonded_energy_bruteforce_n1
                    test_mpi_bonded_energy_bruteforce 1)
    atv_add_mpi_run(test_mpi_bonded_energy_bruteforce_n8
                    test_mpi_bonded_energy_bruteforce 8)

    atv_add_mpi_exec(test_mpi_atom_migration mpi/test_atom_migration.cpp)
    atv_add_mpi_run(test_mpi_atom_migration_n1 test_mpi_atom_migration 1)
    atv_add_mpi_run(test_mpi_atom_migration_n2 test_mpi_atom_migration 2)

    atv_add_mpi_exec(test_mpi_minimize_lj mpi/test_minimize_lj.cpp)
    atv_add_mpi_run(test_mpi_minimize_lj_n1 test_mpi_minimize_lj 1)
    atv_add_mpi_run(test_mpi_minimize_lj_n2 test_mpi_minimize_lj 2)
    atv_add_mpi_run(test_mpi_minimize_lj_n4 test_mpi_minimize_lj 4)
    atv_add_mpi_run(test_mpi_minimize_lj_n8 test_mpi_minimize_lj 8)

    atv_add_mpi_exec(test_mpi_pe_chains_multirank mpi/test_pe_chains_multirank.cpp)
    atv_add_mpi_run(test_mpi_pe_chains_multirank_n2 test_mpi_pe_chains_multirank 2)

    atv_add_mpi_exec(test_mpi_pe_chains_8rank mpi/test_pe_chains_8rank.cpp)
    atv_add_mpi_run(test_mpi_pe_chains_8rank_n8 test_mpi_pe_chains_8rank 8)

    atv_add_mpi_exec(test_mpi_nose_hoover_t_ramp mpi/test_nose_hoover_t_ramp.cpp)
    atv_add_mpi_run(test_mpi_nose_hoover_t_ramp_n2 test_mpi_nose_hoover_t_ramp 2)

    atv_add_mpi_exec(test_mpi_pe_chain_crossing mpi/test_pe_chain_crossing.cpp)
    atv_add_mpi_run(test_mpi_pe_chain_crossing_n2 test_mpi_pe_chain_crossing 2)

    atv_add_mpi_exec(test_mpi_pe_full_multirank mpi/test_pe_full_multirank.cpp)
    atv_add_mpi_run(test_mpi_pe_full_multirank_n2 test_mpi_pe_full_multirank 2)

    atv_add_mpi_exec(test_mpi_pe_chain_drifting mpi/test_pe_chain_drifting.cpp)
    atv_add_mpi_run(test_mpi_pe_chain_drifting_n2 test_mpi_pe_chain_drifting 2)

    atv_add_mpi_exec(test_mpi_npt_multirank mpi/test_npt_multirank.cpp)
    atv_add_mpi_run(test_mpi_npt_multirank_n8 test_mpi_npt_multirank 8)

    atv_add_mpi_exec(test_mpi_nose_hoover_npt_multirank
                     mpi/test_nose_hoover_npt_multirank.cpp)
    atv_add_mpi_run(test_mpi_nose_hoover_npt_multirank_n8
                    test_mpi_nose_hoover_npt_multirank 8)

    atv_add_mpi_exec(test_mpi_deform_multirank mpi/test_deform_multirank.cpp)
    atv_add_mpi_run(test_mpi_deform_multirank_n8 test_mpi_deform_multirank 8)

    atv_add_mpi_exec(test_mpi_eam_multirank mpi/test_eam_multirank.cpp)
    atv_add_mpi_run(test_mpi_eam_multirank_n2 test_mpi_eam_multirank 2)
    atv_add_mpi_run(test_mpi_eam_multirank_n4 test_mpi_eam_multirank 4)

    atv_add_mpi_exec(test_mpi_rattle_multirank mpi/test_rattle_multirank.cpp)
    atv_add_mpi_run(test_mpi_rattle_multirank_n2 test_mpi_rattle_multirank 2)
    atv_add_mpi_run(test_mpi_rattle_multirank_n4 test_mpi_rattle_multirank 4)

    atv_add_mpi_exec(test_mpi_sw_multirank mpi/test_sw_multirank.cpp)
    atv_add_mpi_run(test_mpi_sw_multirank_n2 test_mpi_sw_multirank 2)
    atv_add_mpi_run(test_mpi_sw_multirank_n4 test_mpi_sw_multirank 4)

    atv_add_mpi_exec(test_mpi_pe_msu_pipeline mpi/test_pe_msu_pipeline.cpp)
    atv_add_mpi_run(test_mpi_pe_msu_pipeline_n8 test_mpi_pe_msu_pipeline 8)

    atv_add_mpi_exec(test_mpi_gather mpi/test_gather.cpp)
    atv_add_mpi_run(test_mpi_gather_n1 test_mpi_gather 1)
    atv_add_mpi_run(test_mpi_gather_n2 test_mpi_gather 2)
    atv_add_mpi_run(test_mpi_gather_n8 test_mpi_gather 8)

    atv_add_mpi_exec(test_mpi_rcb mpi/test_rcb.cpp)
    atv_add_mpi_run(test_mpi_rcb_n1 test_mpi_rcb 1)
    atv_add_mpi_run(test_mpi_rcb_n8 test_mpi_rcb 8)

    atv_add_mpi_exec(test_mpi_task_protocol mpi/test_task_protocol.cpp)
    atv_add_mpi_run(test_mpi_task_protocol_n1 test_mpi_task_protocol 1)
    atv_add_mpi_run(test_mpi_task_protocol_n2 test_mpi_task_protocol 2)
    atv_add_mpi_run(test_mpi_task_protocol_n4 test_mpi_task_protocol 4)

    # serial-on-replica seam (mpi-launcher-embedded-python-04): a driver with
    # set_serial_on_replica() must NOT decompose — every rank solves the full
    # replica (AM1 H2 single point) and lands on the identical energy.
    atv_add_mpi_exec(test_mpi_serial_on_replica mpi/test_serial_on_replica.cpp)
    atv_add_mpi_run(test_mpi_serial_on_replica_n1 test_mpi_serial_on_replica 1)
    atv_add_mpi_run(test_mpi_serial_on_replica_n2 test_mpi_serial_on_replica 2)
    atv_add_mpi_run(test_mpi_serial_on_replica_n4 test_mpi_serial_on_replica 4)
endif()

# ── PW infrastructure tests (require ATV_BUILD_PW) ─────────────────────

if(ATV_BUILD_PW)
    # Naming convention: test_{backend}_{domain}_{component}.
    atv_add_cpp_test(test_cpu_pw_g_vector   cpu/pw/test_g_vector.cpp)
    atv_add_cpp_test(test_cpu_pw_upf        cpu/pw/test_upf.cpp)
    atv_add_cpp_test(test_cpu_pw_kpoint     cpu/pw/test_kpoint.cpp)
    atv_add_cpp_test(test_cpu_pw_crystal    cpu/pw/test_crystal.cpp)
    atv_add_cpp_test(test_cpu_pw_ewald      cpu/pw/test_ewald.cpp)
    atv_add_cpp_test(test_cpu_pw_components cpu/pw/test_pw_components.cpp)
    atv_add_cpp_test(test_cpu_pw_si         cpu/pw/test_pw_si.cpp)
    atv_add_cpp_test(test_cpu_pw_evaluator  cpu/pw/test_pw_evaluator.cpp)
    atv_add_cpp_test(test_cpu_pw_forces     cpu/pw/test_pw_forces.cpp)
    atv_add_cpp_test(test_cpu_pw_stress     cpu/pw/test_pw_stress.cpp)

    # FFT test needs FFTW3 linkage
    add_executable(test_cpu_pw_fft cpu/pw/test_fft.cpp)
    target_link_libraries(test_cpu_pw_fft PRIVATE atomiverse GTest::gtest_main fftw3_stub)
    set_target_properties(test_cpu_pw_fft PROPERTIES CXX_STANDARD 17)
    target_include_directories(test_cpu_pw_fft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME test_cpu_pw_fft COMMAND test_cpu_pw_fft)
    set_tests_properties(test_cpu_pw_fft PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${ATV_TEST_OMP_THREADS}")

    # MPI k-point parallel parity (launched under mpirun)
    if(ATV_USE_MPI AND MPIEXEC_EXECUTABLE)
        atv_add_mpi_exec(test_mpi_pw_kpoint mpi/test_pw_kpoint.cpp)
        atv_add_mpi_run(test_mpi_pw_kpoint_n1 test_mpi_pw_kpoint 1)
        atv_add_mpi_run(test_mpi_pw_kpoint_n2 test_mpi_pw_kpoint 2)
        atv_add_mpi_run(test_mpi_pw_kpoint_n4 test_mpi_pw_kpoint 4)
    endif()
endif()

# ── Shared dense linear algebra (cpu/linalg) ──────────────────────────

# Canonical orthogonalization with S-eigenvalue thresholding
# (hf-robust-02). Header-only cpu/linalg/eigensolve.h — no RHF gate.
atv_add_cpp_test(test_cpu_linalg_canonical_orth
                 cpu/linalg/test_canonical_orthogonalization.cpp)

# ── RHF/SCF tests ─────────────────────────────────────────────────────

if(ATV_BUILD_HF)
    atv_add_cpp_test(test_cpu_scf_cart2sph     cpu/scf/test_cart2sph.cpp)
    atv_add_cpp_test(test_cpu_scf_multipole_ints cpu/scf/test_multipole_integrals.cpp)
    atv_add_cpp_test(test_cpu_observer_multipole cpu/observer/test_multipole.cpp)
    # Löwdin populations + MK/CHELPG ESP fit — hand-built shells/determinant
    # and synthetic point-charge potentials only (hf-capability-gaps-04).
    atv_add_cpp_test(test_cpu_observer_lowdin cpu/observer/test_lowdin.cpp)
    atv_add_cpp_test(test_cpu_observer_esp_charges cpu/observer/test_esp_charges.cpp)
    atv_add_cpp_test(test_cpu_scf_spherical    cpu/scf/test_spherical_basis.cpp)
    # Static basis-table structure for the third-period + diffuse extension
    # (hf-capability-gaps-05-basis) — table pins vs PySCF, no SCF solve.
    atv_add_cpp_test(test_basis_tables         cpu/scf/test_basis_tables.cpp)
    # Semi-local def2-ECP: static table pins, EcpPotential lookup,
    # EcpIntegrals U-matrix vs external PySCF ECPscalar pins, IntegralPack
    # valence wiring, x2c double-counting gate (hf-capability-gaps-06-ecp).
    atv_add_cpp_test(test_ecp                  cpu/scf/test_ecp.cpp)
    # hf-capability-gaps-18-x2c: spin-free pVp kernel pins, X2cWorkspace
    # nonrel/exact-decoupling identities, ECP↔x2c mutual exclusion both
    # ways, forces() intercept. No heavy-atom SCF in this unit binary.
    atv_add_cpp_test(test_cpu_scf_x2c           cpu/scf/test_x2c.cpp)
    atv_add_cpp_test(test_cpu_scf_uhf_solver   cpu/scf/test_uhf_solver.cpp)
    # hf-capability-gaps-10-rohf: ROHF (Roothaan effective Fock, PySCF
    # canonical / Guest-Saunders convention) vs externally pinned PySCF.
    atv_add_cpp_test(test_cpu_scf_rohf_solver  cpu/scf/test_rohf_solver.cpp)
    # hf-capability-gaps-11-ghf: real two-component GHF — spinor
    # SlaterDeterminant invariants, H2 collapse to UHF, noncollinear win on
    # the frustrated H3 triangle vs externally pinned PySCF, collinear trap.
    atv_add_cpp_test(test_cpu_scf_ghf          cpu/scf/test_ghf.cpp)
    atv_add_cpp_test(test_cpu_scf_uks_solver   cpu/scf/test_uks_solver.cpp)
    atv_add_cpp_test(test_cpu_scf_xc_functional cpu/scf/test_xc_functional.cpp)
    atv_add_cpp_test(test_cpu_scf_evaluator     cpu/scf/test_scf_evaluator.cpp)
    # scf-driver-refactor-04b-wire: method-driver nesting + MeanFieldOutputs.
    atv_add_cpp_test(test_cpu_scf_method_driver_wire
                     cpu/scf/test_method_driver_wire.cpp)
    # hf-capability-gaps-08-stability: MO orbital-Hessian A/B kernel +
    # Seeger-Pople stability analysis. NO SCF loop: the converged H2/STO-3G
    # determinant is symmetry-exact in closed form (one Fock build, no
    # iteration); the full RHF.run()+stability() path is
    # regressions/scf/h2_rhf_stability.
    atv_add_cpp_test(test_cpu_scf_orbital_hessian cpu/scf/test_orbital_hessian.cpp)
    atv_add_cpp_test(test_cpu_scf_stability       cpu/scf/test_stability.cpp)
    # hf-capability-gaps-14-tdhf: TDA + TDHF/RPA lowest roots over the same
    # A/B kernel (dense vs full-supermatrix vs Davidson parity, the
    # triplet-instability throw) + the RHF::tda()/tdhf() method surface.
    # Formaldehyde oracle e2e lives in regressions/scf/ch2o_tdhf.
    atv_add_cpp_test(test_cpu_scf_tdhf_pack       cpu/scf/test_tdhf_pack.cpp)
    atv_add_cpp_test(test_cpu_scf_tdhf_method     cpu/scf/test_tdhf_method.cpp)
    atv_add_cpp_test(test_cpu_scf_embedding       cpu/scf/test_embedding.cpp)
    # hf-capability-gaps-16-solvent: C-PCM cavity/Born oracle, f(ε), H2O
    # solvent vs gas, solvent_eps=0 vacuum identity, forces throw.
    atv_add_cpp_test(test_cpu_scf_pcm             cpu/scf/test_pcm.cpp)
    # hf-capability-gaps-17-bsse: ghost AtomList/ShellList/IntegralPack
    # workspace semantics + Boys–Bernardi Counterpoise 5-SCF identities
    # (He–He/STO-3G; locked bsse ≤ 0). Water-dimer e2e lives in
    # regressions/scf/h2o_dimer_cp_hf631gs.
    atv_add_cpp_test(test_ghost_basis             cpu/scf/test_ghost_basis.cpp)
    atv_add_cpp_test(test_counterpoise            cpu/scf/test_counterpoise.cpp)
    # hf-capability-gaps-13-cphf: second/skeleton derivative-integral FD
    # gates, B^x sum rules, CPHF iterative-vs-dense parity + the link-08
    # sigma_plus LHS identity (micro HeH+), and the H2/STO-3G analytic-
    # Hessian-vs-FD-of-analytic-forces first gate. The H2O/6-31G* e2e lives
    # in regressions/scf/h2o_rhf_freq.
    atv_add_cpp_test(test_cpu_scf_cphf            cpu/scf/test_cphf.cpp)
    atv_add_cpp_test(test_cpu_scf_analytic_hessian
                     cpu/scf/test_analytic_hessian.cpp)
    # hf-capability-gaps-09-soscf: CIAH stepper units — exp(κ) unitarity,
    # gradient vs FD, matrix-free CIAH vs dense augmented eigensolve, trust
    # region, MixStage latch; plus the allowed micro H2/STO-3G default-off
    # invariance pin (milliseconds). The hard-case contrast is
    # regressions/scf/soscf_hard_case.
    atv_add_cpp_test(test_soscf                   cpu/scf/test_soscf.cpp)
    atv_add_cpp_test(test_cpu_scf_mp2           cpu/scf/test_mp2.cpp)
    atv_add_cpp_test(test_cpu_scf_fci           cpu/scf/test_fci.cpp)
    atv_add_cpp_test(test_cpu_scf_koopmans      cpu/scf/test_koopmans.cpp)
    atv_add_cpp_test(test_cpu_scf_delta_scf     cpu/scf/test_delta_scf.cpp)
    atv_add_cpp_test(test_cpu_scf_integrals     cpu/scf/test_integrals.cpp)
    atv_add_cpp_test(test_cpu_scf_rys_roots     cpu/scf/test_rys_roots.cpp)
    atv_add_cpp_test(test_cpu_scf_rys_eri       cpu/scf/test_rys_eri.cpp)
    # hf-capability-gaps-12-df: def2-JKFIT table, g-shell 2c/3c integrals,
    # metric whitening, fitted J/K vs dense (single calls, no SCF).
    atv_add_cpp_test(test_density_fitting       cpu/scf/test_density_fitting.cpp)
    atv_add_cpp_test(test_cpu_scf_primitives    cpu/scf/test_primitives.cpp)
    atv_add_cpp_test(test_cpu_scf_accelerators  cpu/scf/test_accelerators.cpp)
    # hf-capability-gaps-07-guess: Hückel/SAP guess matrices + EDIIS/ADIIS
    # far-field mixers (single-call math only; iteration-count contrast is
    # regressions/scf/h2o_scf_convergence).
    atv_add_cpp_test(test_cpu_scf_init_guess    cpu/scf/test_init_guess.cpp)
    atv_add_cpp_test(test_cpu_scf_ediis_adiis   cpu/scf/test_ediis_adiis.cpp)
    # Charge/multiplicity resolver + method run() validity guard
    # (hf-robust-01); tutorial ch02/ch03 regression anchor.
    atv_add_cpp_test(test_cpu_scf_spin_resolution cpu/scf/test_spin_resolution.cpp)
    # SlaterDeterminant value type: validating ctors, per-spin density,
    # expect/homo/lumo, with_occupation, validate(S)
    # (scf-slater-determinant-01-type).
    atv_add_cpp_test(test_cpu_scf_slater_determinant cpu/scf/test_slater_determinant.cpp)
    # Public SCF vocabulary types re-homed into dedicated headers: scf_enums.h,
    # gradient_result.h, scf_guess.h — without including scf_result.h
    # (scf-slater-determinant-02-rehome).
    atv_add_cpp_test(test_scf_type_homes cpu/scf/test_scf_type_homes.cpp)
    # SCF linear-dependence handling via canonical orthogonalization
    # (hf-robust-02); tutorial ch02 regression anchor.
    atv_add_cpp_test(test_cpu_scf_lindep         cpu/scf/test_scf_lindep.cpp)
    # Virtual-orbital level shift, PySCF get_fock ordering parity
    # (hf-robust-03); tutorial ch02 regression anchor.
    atv_add_cpp_test(test_cpu_scf_level_shift    cpu/scf/test_level_shift.cpp)
    atv_add_cpp_test(test_cpu_scf_solver        cpu/scf/test_scf_solver.cpp)
    # scf-driver-refactor-02-stages: five scf::Phase stage components wrapping
    # the internal SCF helpers, isolated bit-identical parity (unit tier).
    atv_add_cpp_test(test_cpu_scf_stages cpu/scf/stages/test_scf_stages.cpp)
    # Bit-identical parity compares two independent builds of the same kernels;
    # parallel reduction order (OpenMP / BLAS) is ULP-non-deterministic, so this
    # test pins serial execution. Do NOT loosen the assertions to tolerances —
    # the spec mandates bit-identity.
    set_tests_properties(test_cpu_scf_stages PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    # scf-driver-refactor-03a-convert: RHF/UHF/RKS/UKS as CPUDriver<scf::Phase>,
    # regression-pinned against the 03a-parity energies/eigenvalues. Pins serial
    # execution for the same ULP-determinism reason as test_cpu_scf_stages.
    atv_add_cpp_test(test_cpu_scf_method_driver_parity
                     cpu/scf/test_method_driver_parity.cpp)
    set_tests_properties(test_cpu_scf_method_driver_parity PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    atv_add_cpp_test(test_cpu_scf_method_driver_no_alloc
                     cpu/scf/test_method_driver_no_alloc.cpp)
    atv_add_cpp_test(test_cpu_evaluate_driver        cpu/test_evaluate_driver.cpp)
    atv_add_cpp_test(test_cpu_scf_minao         cpu/scf/test_minao.cpp)
    atv_add_cpp_test(test_cpu_scf_rys_eri_deriv cpu/scf/test_rys_eri_deriv.cpp)
    # test_cpu_minimizer_lbfgs relocated to benchmarks/qc/rhf/validation/
    # (val_min_lbfgs_geoopt) — full SCF-driven H2 geometry optimizations.
    atv_add_cpp_test(test_cpu_minimizer_line_search cpu/minimizer/test_line_search.cpp)
    atv_add_cpp_test(test_cpu_minimizer_recorders cpu/minimizer/test_trajectory_writer.cpp)
    atv_add_cpp_test(test_cpu_minimize_driver_swap_optimizer cpu/test_minimize_driver_swap_optimizer.cpp)
    atv_add_cpp_test(test_cpu_scf_orbital_info_observer cpu/scf/test_orbital_info_observer.cpp)
    # Vibrational analysis (finite-difference Hessian) is an end-to-end /
    # PySCF-parity suite — too slow for the unit gate. RELOCATED to
    # benchmarks/qc/vib/validation/ (run under -DATV_BUILD_BENCHMARKS=ON).
    atv_add_cpp_test(test_cpu_recorder_density_cube cpu/recorder/test_density_cube.cpp)
    atv_add_cpp_test(test_xc_reference         test_xc_reference.cpp)
    # libxc test oracle (sub-spec 02 of libxc-xc-fix creates the oracle/
    # subdirectory; this block lands ahead of it so the option flips on
    # cleanly the moment the directory exists).
    if(ATV_BUILD_XC_ORACLE)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpu/scf/oracle/CMakeLists.txt")
            add_subdirectory(cpu/scf/oracle)
        else()
            message(STATUS
                "ATV_BUILD_XC_ORACLE=ON but tests/cpu/scf/oracle/ does "
                "not exist yet (sub-spec 02 of libxc-xc-fix will create it). "
                "Skipping oracle build.")
        endif()
    endif()
endif()

# ── CUDA backend tests ──────────────────────────────────────────────

if(ATV_BUILD_CUDA)
    atv_add_test(test_cuda_device_manager cuda/test_device_manager.cu)
    atv_add_test(test_cuda_domain_decomposition cuda/test_domain_decomposition.cu)
    atv_add_test(test_cuda_multi_gpu  cuda/md/test_multi_gpu.cu)
    atv_add_test(test_cuda_frame      cuda/test_frame.cu)
    atv_add_test(test_cuda_materialize cuda/test_materialize.cu)
    atv_add_test(test_cuda_backend    cuda/test_backend.cu)
    atv_add_test(test_facade_plan     facade/test_plan.cu)
    atv_add_test(test_cuda_md         cuda/md/test_md.cu)
    atv_add_test(test_cuda_md_thermostat cuda/md/test_thermostat.cu)
    atv_add_test(test_cuda_md_zero_momentum cuda/md/test_zero_momentum.cu)
    atv_add_test(test_cuda_md_components cuda/md/test_cuda_md_components.cu)
    atv_add_test(test_cuda_install_infra_capability cuda/md/test_install_infra_capability.cu)
    atv_add_test(test_cuda_neighborlist_skin_gating cuda/md/test_neighborlist_skin_gating.cu)
    atv_add_test(test_cuda_atom_cell_sort cuda/md/test_atom_cell_sort.cu)
    # Pulls in the private primitive header src/cuda/neighbor/atom_cell_sort.cuh
    target_include_directories(test_cuda_atom_cell_sort PRIVATE
        ${CMAKE_SOURCE_DIR}/src/cuda/neighbor)
    atv_add_test(test_cuda_pair_lj_no_d2h cuda/test_pair_lj_no_d2h.cu)
    target_compile_definitions(test_cuda_pair_lj_no_d2h PRIVATE
        ATV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    atv_add_test(test_cuda_driver_no_per_phase_sync cuda/test_driver_no_per_phase_sync.cu)
    target_compile_definitions(test_cuda_driver_no_per_phase_sync PRIVATE
        ATV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    atv_add_test(test_cuda_md_nve_limit cuda/md/test_nve_limit.cu)
    atv_add_test(test_cuda_md_fix_langevin cuda/md/test_fix_langevin.cu)
    atv_add_test(test_cuda_md_nose_hoover_npt cuda/md/test_cuda_nose_hoover_npt.cu)
    # cuda::RDF on the molrs streaming accumulator (arch-refactor-07-molrs):
    # static-trajectory g(r) parity vs a host-fed cpu::compute::RDFAccumulator
    # + fail-fast ctor edges. Needs a GPU node (runs a cuda::md::MD).
    atv_add_test(test_cuda_observer_rdf_molrs cuda/observer/test_rdf_molrs.cu)
    atv_add_test(test_cuda_pair_lj_special_bonds cuda/potentials/test_pair_lj_special_bonds.cu)
    atv_add_test(test_cuda_potentials_bonded cuda/potentials/test_bonded.cu)
    atv_add_test(test_cuda_ewald_coul cuda/kspace/test_ewald_coul.cu)
    if(ATV_BUILD_KSPACE)
        atv_add_test(test_cuda_pme_coul cuda/kspace/test_pme_coul.cu)
    endif()
    if(ATV_BUILD_PW)
        atv_add_test(test_cuda_pw_fft cuda/pw/test_fft.cu)
        target_link_libraries(test_cuda_pw_fft PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_poisson cuda/pw/test_poisson.cu)
        target_link_libraries(test_cuda_pw_poisson PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_grid_map cuda/pw/test_grid_map.cu)
        target_link_libraries(test_cuda_pw_grid_map PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_xc_lda cuda/pw/test_xc_lda.cu)
        target_link_libraries(test_cuda_pw_xc_lda PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_veff cuda/pw/test_veff.cu)
        target_link_libraries(test_cuda_pw_veff PRIVATE CUDA::cufft)
    endif()
    atv_add_test(test_cuda_minimizer_sd cuda/minimizer/test_sd.cu)
    atv_add_test(test_cuda_minimizer_cg cuda/minimizer/test_cg.cu)
    atv_add_test(test_cuda_minimizer_lbfgs cuda/minimizer/test_lbfgs.cu)
    atv_add_test(test_cuda_scf        cuda/scf/test_scf.cu)
    atv_add_test(test_cuda_scf_rhf    cuda/scf/test_rhf.cu)
    atv_add_test(test_cuda_scf_device_integrals cuda/scf/test_device_integrals.cu)
    if(ATV_BUILD_PW)
        atv_add_test(test_cuda_pw_si  cuda/pw/test_pw_si.cu)
        # Parity needs a gapped system: uses the checked-in ONCV LDA Si
        # pseudopotential from the pw benchmark reference data.
        target_compile_definitions(test_cuda_pw_si PRIVATE
            ATV_PW_UPF_PATH="${CMAKE_SOURCE_DIR}/tests/data/pw/Si_lda_nc.upf")
    endif()
    atv_add_test(test_cuda_driver_role_validation cuda/test_cuda_driver_role_validation.cu)
    # Registration-time PhaseKind gate, CUDA mirror
    # (arch-refactor-01-bindings). add()-only — needs no CUDA device.
    atv_add_test(test_cuda_driver_binding_kind cuda/test_driver_binding_kind.cu)

    # AsyncRecordBase / AsyncBridge contract guards. The subdir registers
    # benchmark-independent tests unconditionally and gates the
    # polyethylene_tension-dependent ones on ATV_BUILD_BENCHMARKS internally.
    add_subdirectory(cuda/async)

endif()

# Grep gate: the frame-block registry is THE single per-atom field
# enumeration (frame-block-01, ac-001). set_capacity / copy_row /
# pack_exchange / unpack_exchange / max_exchange_doubles iterate the
# registry; a hand-written member name inside any of them fails the build.
add_test(
    NAME test_frame_field_list_scrub
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cpu/frame/field_list_scrub.sh
            ${CMAKE_SOURCE_DIR})

# Grep gate: the direct `atoms.<field>[` / `atoms.<field>(` per-atom column
# access shape (frame-block-02, ac-001) must not remain in src/cpu/ or
# include/atomiverse/cpu/ -- every consumer resolves a typed
# `Frame::span(kAtom_*)` once at the launch boundary instead. Pure grep —
# runs in every build configuration.
add_test(
    NAME test_frame_direct_access_scrub
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cpu/frame/direct_access_scrub.sh
            ${CMAKE_SOURCE_DIR})

# Grep gate: no hand-written static_cast<int> on a Phase enum. The seven workflow
# Phase enums all start at 0, so an int phase cannot be told apart across
# workflows; ATV::slot_of is the one sanctioned conversion and the typed entry
# points (slot_of / slot_count / ctx.at / run_stage / make_binding) are the only
# way in. Pure grep — runs in every build configuration.
add_test(
    NAME test_stage_typing_scrub
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/facade/stage_typing_scrub.sh
            ${CMAKE_SOURCE_DIR})

# Grep gate: domain objects are constructed via class constructors — never
# free-function factories (build_integrals / *_config / make_atom_list / …).
# Pure grep — runs in every build configuration. See AGENTS.md OOP rule.
# Product-module DAG + source gating (python-on-demand-install-01)
add_test(
    NAME test_module_dag_auto_enable
    COMMAND ${CMAKE_COMMAND}
            -DATV_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/architecture/module_dag_auto_enable.cmake)
set_tests_properties(test_module_dag_auto_enable PROPERTIES
    LABELS "architecture;cmake"
    TIMEOUT 300)

add_test(
    NAME test_module_off_excludes_sources
    COMMAND ${CMAKE_COMMAND}
            -DATV_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/architecture/module_off_excludes_sources.cmake)
set_tests_properties(test_module_off_excludes_sources PROPERTIES
    LABELS "architecture;cmake"
    TIMEOUT 300)

add_test(
    NAME test_no_factory_scrub
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/architecture/no_factory_scrub.sh
            ${CMAKE_SOURCE_DIR})

# Grep gate: compatibility-only CPU aliases/overloads stay deleted and the
# TaskSpec wire format is strict v3 with an explicit ComponentId mapping.
# Deliberately excludes cpu/frame/ and CUDA/Torch sources.
add_test(
    NAME test_cpu_compatibility_surface_scrub
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cpu/compatibility_surface_scrub.sh
            ${CMAKE_SOURCE_DIR})

# Grep gate: cuda::RDF must be migrated off the deleted legacy host histogram
# helpers (arch-refactor-07-molrs) — rdf.cuh must consume
# cpu::compute::RDFAccumulator via trajectory_metrics.h; only rdf_write_curve
# (the one g(r) emitter) may remain. The forbidden-token list lives in the
# gate script. Pure grep — runs in every build configuration, no CUDA needed.
add_test(
    NAME test_cuda_rdf_molrs_migration
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cuda/observer/rdf_molrs_migration.sh
            ${CMAKE_SOURCE_DIR})

# ── Torch backend tests ─────────────────────────────────────────────

if(ATV_BUILD_TORCH)
    atv_add_test(test_torch_frame          torch/test_frame.cu)
    atv_add_test(test_torch_materialize    torch/test_materialize.cu)
    atv_add_test(test_torch_transfer       torch/test_transfer.cu)
    atv_add_test(test_torch_lj             torch/test_lj.cu)
    atv_add_test(test_torch_harmonic_bond  torch/test_harmonic_bond.cu)
    atv_add_test(test_torch_neighbor_list  torch/test_neighbor_list.cu)
endif()

# Diagnostic helper (not registered as a test): fixed-V_eff band parity.
if(ATV_BUILD_CUDA AND ATV_BUILD_PW)
    add_executable(diag_band_parity cuda/pw/diag_band_parity.cu)
    target_link_libraries(diag_band_parity PRIVATE atomiverse)
    set_target_properties(diag_band_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Every end-to-end / parity check has been physically relocated to
# benchmarks/{qc,md,mc,pw}/**/validation/ (val_scf_*, val_vib_*, val_md_*,
# val_min_lbfgs_geoopt, val_md_pe_nvt_multirank, the MC physics gates), built
# with -DATV_BUILD_BENCHMARKS=ON and run at acceptance time. Nothing is
# disabled-but-compiled here any more: if it is registered above, it runs.
