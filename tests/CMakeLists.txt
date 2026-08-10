include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

# Unit tests assert CODE CORRECTNESS on tiny systems (tens to hundreds of
# atoms); performance and LAMMPS-reference validation live in benchmarks/.
# On many-core nodes the OpenMP default (= all hardware threads, e.g. 288 on
# a GH200 node) makes those tiny systems WALL-DOMINATED by fork/join barriers
# — 20-atom 5000-step runs took minutes, and `ctest -j` multiplied the
# oversubscription. Cap every test's default; a test that genuinely needs a
# specific thread count can still call omp_set_num_threads itself.
# Tiny systems: few threads beat oversubscription. 8 threads on 20-atom
# tests was wall-dominated by OpenMP fork/join.
set(ATV_TEST_OMP_THREADS 2 CACHE STRING
    "OMP_NUM_THREADS for single-process unit tests")
# MPI unit tests: 1 thread/rank — multi-rank campaigns are not unit tests.
set(ATV_TEST_MPI_OMP_THREADS 1 CACHE STRING
    "OMP_NUM_THREADS per rank for MPI unit tests")
# HARD CAP: every ctest registration dies at this wall time. If a "unit"
# needs longer, it is not a unit — move to regressions/ or delete.
set(ATV_TEST_TIMEOUT_SECONDS 10 CACHE STRING
    "Maximum wall seconds for any unit test under ctest (hard kill)")

# Pre-push builds only non-heavy targets so the gate stays minutes, not hours.
# Full method campaigns (CCSD/FCI/MP2/BOMD/CPMD/full GPW/PW solves, …) match
# this regex and are skipped unless ATV_TEST_HEAVY=ON.
option(ATV_TEST_HEAVY
    "Build/register multi-second method tests (default OFF — pre-push gate)"
    OFF)
# Not CACHE: must track the source list (a sticky CACHE goes stale).
# Wave-2 (2026-08-10): residual multi-iter e2e under tests/ deleted; false-heavy
# tokens that gated true units (method_driver, eom_, tdhf, dispersion→phonon,
# pw_evaluator→gpw_evaluator) removed. Empty regex = no name-based heavy skip.
set(ATV_TEST_HEAVY_REGEX "^$")

# Fast-gate discipline (HARD RULE):
#   • tests/ = one component / one method / ≤ ATV_TEST_TIMEOUT_SECONDS.
#   • FORBIDDEN under ctest: multi-step Driver loops, equilibration,
#     multi-rank production MD, full post-HF/CPMD/BOMD campaigns, oracles.
#   • Those live in regressions/ (NOT ctest) or benchmarks/**.

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
    if(NOT ATV_TEST_HEAVY AND name MATCHES "${ATV_TEST_HEAVY_REGEX}")
        message(STATUS "unit: skip heavy ${name} (ATV_TEST_HEAVY=OFF)")
        return()
    endif()
    atv_guard_cuda_sources(${name} ${ARGN})
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE atomiverse GTest::gtest_main)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        TIMEOUT ${ATV_TEST_TIMEOUT_SECONDS}
        ENVIRONMENT "OMP_NUM_THREADS=${ATV_TEST_OMP_THREADS}")
endfunction()

# C++ test: compiled as .cpp, no CUDA required.
function(atv_add_cpp_test name)
    if(NOT ATV_TEST_HEAVY AND name MATCHES "${ATV_TEST_HEAVY_REGEX}")
        message(STATUS "unit: skip heavy ${name} (ATV_TEST_HEAVY=OFF)")
        return()
    endif()
    atv_guard_cuda_sources(${name} ${ARGN})
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE atomiverse GTest::gtest_main)
    set_target_properties(${name} PROPERTIES CXX_STANDARD 20)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        TIMEOUT ${ATV_TEST_TIMEOUT_SECONDS}
        ENVIRONMENT "OMP_NUM_THREADS=${ATV_TEST_OMP_THREADS}")
endfunction()

# MPI test executable: one build target, can be launched at multiple rank
# counts via `atv_add_mpi_run`. The test source provides its own `main`
# that wraps MPI init/finalize.
function(atv_add_mpi_exec target_name)
    add_executable(${target_name} ${ARGN})
    target_link_libraries(${target_name} PRIVATE atomiverse GTest::gtest MPI::MPI_CXX)
    set_target_properties(${target_name} PROPERTIES CXX_STANDARD 20)
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
        TIMEOUT ${ATV_TEST_TIMEOUT_SECONDS}
        ENVIRONMENT "OMP_NUM_THREADS=${ATV_TEST_MPI_OMP_THREADS}"
        PASS_REGULAR_EXPRESSION "PASSED[^a-zA-Z]"
        FAIL_REGULAR_EXPRESSION "FAILED[^a-zA-Z]")
endfunction()

# ── Core type tests (always available, no CUDA required) ──────────────

atv_add_cpp_test(test_mat3  core/test_mat3.cpp)
atv_add_cpp_test(test_vec3  core/test_vec3.cpp)
atv_add_cpp_test(test_real_harmonics core/test_real_harmonics.cpp)
atv_add_cpp_test(test_units core/test_units.cpp)

# ── CPU backend tests (always available, no CUDA required) ──────────

# Mirrors include/atomiverse/cpu/frame/box.h (+ box_simd.h).
atv_add_cpp_test(test_cpu_box cpu/frame/test_box.cpp)
# Mirrors include/atomiverse/cpu/device/chunked_range.h — the fixed, thread-count
# independent partition every reproducible OpenMP reduction is built on.
atv_add_cpp_test(test_cpu_chunked_range cpu/device/test_chunked_range.cpp)
# Context service slices (arch-refactor-05-context): flat core fields + nested
# md/mc/opt/analysis pure-aggregate slices, exact field mapping, all-null
# defaults, trivially-copyable / non-polymorphic guards. Layout-only, no
# physics.
atv_add_cpp_test(test_cpu_component_context_slices cpu/component/test_context_slices.cpp)
if(ATV_BUILD_HF)
    atv_add_cpp_test(test_cpu_qm_engine cpu/component/test_qm_engine.cpp)
    # Bit-for-bit erased-vs-direct RHF comparison: OMP reduction order must be
    # identical in both runs, so pin the thread count (parallel ctest otherwise
    # oversubscribes and the comparison flakes).
    set_tests_properties(test_cpu_qm_engine PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1")
endif()
# esmm-contract-05-accumulate-switch (U8 + U7b): the kEnergyElectronic per-term
# key — ZeroEnergy must clear it in lockstep with kEnergy, and a co-bound
# Evaluate pass must leave the QM share alone in it. Registered UNCONDITIONALLY
# (ZeroEnergy is always-on infrastructure); the two co-binding cases inside are
# #ifdef ATV_HAS_HF. Serial: the RHF pin must not drift with reduction order.
atv_add_cpp_test(test_cpu_zero_energy_electronic cpu/component/test_zero_energy_electronic.cpp)
if(TARGET test_cpu_zero_energy_electronic)
set_tests_properties(test_cpu_zero_energy_electronic PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()
atv_add_cpp_test(test_cpu_mc_scaffold cpu/test_mc_scaffold.cpp)
atv_add_cpp_test(test_cpu_mc_move cpu/montecarlo/test_mc_move.cpp)
atv_add_cpp_test(test_cpu_mc_transaction_rollback cpu/montecarlo/test_transaction_rollback.cpp)
atv_add_cpp_test(test_cpu_mc_ensemble_policies cpu/montecarlo/test_ensemble_policies.cpp)
atv_add_cpp_test(test_cpu_mc_gcmc_npt_movers cpu/montecarlo/test_gcmc_npt_movers.cpp)
atv_add_cpp_test(test_cpu_mc_row_order cpu/montecarlo/test_mc_row_order.cpp)
# The MC physics-validation gates (pair_distribution / acceptance_ratio /
# lj_nist) drive MC::run over many sweeps — e2e, not unit. They now live in
# benchmarks/mc/lj/validation/ and run at acceptance time, never in ctest.
atv_add_cpp_test(test_cpu_md_pe_thermo_schema cpu/dynamics/test_pe_thermo_schema.cpp)
atv_add_cpp_test(test_cpu_md_pe_single_chain cpu/dynamics/test_pe_single_chain.cpp)
# Multi-step PE bonded NVE conservation is a regression
# (regressions/md/pe_bonded_nve), not a unit test.
# test_cpu_md_csvr / test_cpu_md_nvt relocated to benchmarks/md/lj/validation/
# (val_md_csvr / val_md_nvt) — full thermostat equilibrations, not ms unit tests.
atv_add_cpp_test(test_cpu_md_atom_sort            cpu/dynamics/test_atom_sort.cpp)
atv_add_cpp_test(test_cpu_md_timing_report        cpu/dynamics/test_timing_report.cpp)
atv_add_cpp_test(test_cpu_md_pe_fd_force          cpu/dynamics/test_pe_fd_force.cpp)
# Multi-chain multi-step NVE was e2e; multi-chain unit surface is
# special-bonds / net-force one-launch only in pe_multi_chain.
atv_add_cpp_test(test_cpu_md_pe_multi_chain       cpu/dynamics/test_pe_multi_chain.cpp)
atv_add_cpp_test(test_cpu_md_lj_exclusions        cpu/dynamics/test_lj_exclusions.cpp)
atv_add_cpp_test(test_cpu_md_pair_lj_cut          cpu/dynamics/test_pair_lj_cut.cpp)
atv_add_cpp_test(test_cpu_md_const_pot_bo         cpu/dynamics/test_const_pot_bo.cpp)
atv_add_cpp_test(test_cpu_md_const_charge_electrode cpu/dynamics/test_const_charge_electrode.cpp)
atv_add_cpp_test(test_cpu_md_qeq                  cpu/dynamics/test_qeq.cpp)
atv_add_cpp_test(test_cpu_md_const_pot_cg         cpu/dynamics/test_const_pot_cg.cpp)
atv_add_cpp_test(test_cpu_md_const_pot_per_eta    cpu/dynamics/test_const_pot_per_eta.cpp)
atv_add_cpp_test(test_cpu_md_charge_updater_probe cpu/dynamics/test_charge_updater_probe.cpp)
atv_add_cpp_test(test_cpu_comm_halo_refresh_charges cpu/communication/test_halo_refresh_charges.cpp)
atv_add_cpp_test(test_cpu_comm_tag_index_lifetime cpu/communication/test_tag_index_lifetime.cpp)
atv_add_cpp_test(test_cpu_md_const_pot_ffield     cpu/dynamics/test_const_pot_ffield.cpp)
atv_add_cpp_test(test_cpu_md_const_pot_maze_symplectic cpu/dynamics/test_const_pot_maze_symplectic.cpp)
atv_add_cpp_test(test_cpu_md_const_pot_site_volume cpu/dynamics/test_const_pot_site_volume.cpp)
atv_add_cpp_test(test_cpu_voronoi_volume          cpu/observe/test_voronoi_volume.cpp)
atv_add_cpp_test(test_cpu_md_dipole_electrode_joint cpu/dynamics/test_dipole_electrode_joint.cpp)
atv_add_cpp_test(test_cpu_classical_polarization  cpu/observe/test_classical_polarization.cpp)
atv_add_cpp_test(test_cpu_pair_coul_cut           cpu/potentials/test_pair_coul_cut.cpp)
atv_add_cpp_test(test_cpu_charging_work           cpu/electrodes/test_charging_work.cpp)
atv_add_cpp_test(test_cpu_electrode_charge_aspc   cpu/electrodes/test_charge_aspc.cpp)
atv_add_cpp_test(test_cpu_pair_lj_cut_tip4p_cut   cpu/potentials/test_pair_lj_cut_tip4p_cut.cpp)
atv_add_cpp_test(test_cpu_pair_lj_soft            cpu/potentials/test_pair_lj_soft.cpp)
atv_add_cpp_test(test_cpu_pair_coul_soft          cpu/potentials/test_pair_coul_soft.cpp)
atv_add_cpp_test(test_cpu_lambda_ti               cpu/potentials/test_lambda_ti.cpp)
atv_add_cpp_test(test_cpu_pair_special_scale      cpu/potentials/test_pair_special_scale.cpp)
atv_add_cpp_test(test_cpu_coul_long_special       cpu/potentials/test_coul_long_special.cpp)
atv_add_cpp_test(test_cpu_pair_lj_cut_coul_long   cpu/potentials/test_pair_lj_cut_coul_long.cpp)
atv_add_cpp_test(test_cpu_dihedral_charmm         cpu/potentials/test_dihedral_charmm.cpp)
atv_add_cpp_test(test_cpu_improper_harmonic       cpu/potentials/test_improper_harmonic.cpp)
atv_add_cpp_test(test_cpu_lammps_data_full        cpu/io/test_lammps_data_full.cpp)
atv_add_cpp_test(test_cpu_lammps_real             cpu/io/test_lammps_real.cpp)
atv_add_cpp_test(test_cpu_fix_efield              cpu/dynamics/test_fix_efield.cpp)
atv_add_cpp_test(test_cpu_fix_dfield              cpu/dynamics/test_fix_dfield.cpp)
atv_add_cpp_test(test_cpu_fix_addforce            cpu/dynamics/test_fix_addforce.cpp)
atv_add_cpp_test(test_cpu_fix_oscillating_efield  cpu/dynamics/test_fix_oscillating_efield.cpp)
atv_add_cpp_test(test_cpu_fix_wall_lj1043         cpu/dynamics/test_fix_wall_lj1043.cpp)
atv_add_cpp_test(test_cpu_bias_distance_cv        cpu/bias/test_distance_cv.cpp)
atv_add_cpp_test(test_cpu_bias_angle_cv           cpu/bias/test_angle_cv.cpp)
atv_add_cpp_test(test_cpu_bias_dihedral_cv        cpu/bias/test_dihedral_cv.cpp)
atv_add_cpp_test(test_cpu_bias_metadynamics_1d    cpu/bias/test_metadynamics_1d.cpp)
atv_add_cpp_test(test_cpu_bias_metadynamics_nd    cpu/bias/test_metadynamics_nd.cpp)
# Requires molrs WhamCompute (gated with ATV_MOLRS_HAS_WHAM / ATV_HAS_MOLRS_WHAM).
if(ATV_MOLRS_HAS_WHAM)
    atv_add_cpp_test(test_cpu_wham                cpu/observe/test_wham.cpp)
endif()
atv_add_cpp_test(test_cpu_pair_yukawa             cpu/potentials/test_pair_yukawa.cpp)
atv_add_cpp_test(test_cpu_gaussian_charge         cpu/potentials/test_gaussian_charge.cpp)
atv_add_cpp_test(test_cpu_pair_buck               cpu/potentials/test_pair_buck.cpp)
atv_add_cpp_test(test_cpu_pair_born               cpu/potentials/test_pair_born.cpp)
atv_add_cpp_test(test_cpu_pair_virial             cpu/potentials/test_pair_virial.cpp)
atv_add_cpp_test(test_cpu_pim_induced_dipole      cpu/potentials/test_pim_induced_dipole.cpp)
atv_add_cpp_test(test_cpu_pim_xl_dipole           cpu/potentials/test_pim_xl_dipole.cpp)
atv_add_cpp_test(test_cpu_minimize_aniso_cell     cpu/minimize/test_anisotropic_cell.cpp)
# GFN2 sources land with the SE domain; gate the unit test the same way.
if(ATV_BUILD_SEMIEMPIRICAL AND EXISTS
   "${CMAKE_CURRENT_SOURCE_DIR}/cpu/semiempirical/test_gfn2_params.cpp")
    atv_add_cpp_test(test_cpu_gfn2_params         cpu/semiempirical/test_gfn2_params.cpp)
endif()
# QM/MM units. Mechanical stays under SE; electrostatic engine-slot path needs HF.
if(ATV_BUILD_SEMIEMPIRICAL)
    atv_add_cpp_test(test_cpu_qmmm_mechanical     cpu/multiscale/test_qmmm_mechanical.cpp)
    atv_add_cpp_test(test_cpu_qmmm_oniom          cpu/multiscale/test_qmmm_oniom.cpp)
    atv_add_cpp_test(test_cpu_classical_nonbonded cpu/multiscale/test_classical_nonbonded.cpp)
    if(ATV_BUILD_HF)
        atv_add_cpp_test(test_cpu_qmmm_polarizable    cpu/multiscale/test_qmmm_polarizable.cpp)
    endif()
endif()
if(ATV_BUILD_HF)
    atv_add_cpp_test(test_cpu_qm_mm_vdw           cpu/multiscale/test_qm_mm_vdw.cpp)
    atv_add_cpp_test(test_cpu_qm_subsystem        cpu/multiscale/test_qm_subsystem.cpp)
    atv_add_cpp_test(test_cpu_qmmm_electrostatic  cpu/multiscale/test_qmmm_electrostatic.cpp)
    atv_add_cpp_test(test_cpu_link_atom           cpu/multiscale/test_link_atom.cpp)
endif()
atv_add_cpp_test(test_cpu_minimize_triclinic      cpu/minimize/test_triclinic_cell.cpp)
atv_add_cpp_test(test_cpu_bond_fene               cpu/potentials/test_bond_fene.cpp)
atv_add_cpp_test(test_cpu_pair_wca                cpu/potentials/test_pair_wca.cpp)
atv_add_cpp_test(test_cpu_pair_dpd                cpu/potentials/test_pair_dpd.cpp)
atv_add_cpp_test(test_cpu_path_hremd              cpu/path/test_hremd.cpp)
atv_add_cpp_test(test_cpu_ewald_sum               cpu/longrange/test_ewald_sum.cpp)
atv_add_cpp_test(test_cpu_pme_math                cpu/longrange/test_pme_math.cpp)
atv_add_cpp_test(test_cpu_ewald_coul              cpu/longrange/test_ewald_coul.cpp)
atv_add_cpp_test(test_cpu_slab_ewald              cpu/longrange/test_slab_ewald.cpp)
# Backend-agnostic FMM operator math (absorbed into cpu/longrange/, always on)
atv_add_cpp_test(test_multipole_expansion cpu/longrange/test_multipole_expansion.cpp)
atv_add_cpp_test(test_fmm_operators cpu/longrange/test_fmm_operators.cpp)
atv_add_cpp_test(test_fmm_tree cpu/longrange/test_fmm_tree.cpp)
# hf-capability-gaps-21-cfmm: WJGH Gaussian extent (header-only facade)
atv_add_cpp_test(test_gaussian_extent cpu/longrange/test_gaussian_extent.cpp)
if(ATV_BUILD_KSPACE)
    atv_add_cpp_test(test_cpu_pme_coul            cpu/longrange/test_pme_coul.cpp)
    atv_add_cpp_test(test_cpu_pppm_coul           cpu/longrange/test_pppm_coul.cpp)
endif()
atv_add_cpp_test(test_cpu_nddo_calpar_derived    cpu/semiempirical/test_nddo_calpar_derived.cpp)
atv_add_cpp_test(test_cpu_rm1                     cpu/semiempirical/test_rm1.cpp)
atv_add_cpp_test(test_cpu_dftb2_sk                cpu/semiempirical/test_dftb2_sk.cpp)
atv_add_cpp_test(test_cpu_scc_mixer               cpu/semiempirical/test_scc_mixer.cpp)

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
    atv_add_cpp_test(test_cpu_gpw_gamma_k_identity cpu/gpw/test_cpu_gpw_gamma_k_identity.cpp)
    atv_add_cpp_test(test_cpu_gpw_multigrid        cpu/gpw/test_cpu_gpw_multigrid.cpp)
    if(ATV_BUILD_PW)  # FFT transfer (prolong_fft / restrict_fft) needs FFTW3
        target_link_libraries(test_cpu_gpw_multigrid PRIVATE fftw3_stub)
        target_compile_definitions(test_cpu_gpw_multigrid PRIVATE ATV_GPW_HAS_FFT)
    endif()
    atv_add_cpp_test(test_cpu_gpw_xc_grid          cpu/gpw/test_cpu_gpw_xc_grid.cpp)
    atv_add_cpp_test(test_cpu_gpw_one_electron     cpu/gpw/test_cpu_gpw_one_electron.cpp)
    atv_add_cpp_test(test_cpu_gpw_workspace        cpu/gpw/test_cpu_gpw_workspace.cpp)
    atv_add_cpp_test(test_cpu_gpw_driver_node     cpu/gpw/test_cpu_gpw_driver_node.cpp)
    if(ATV_BUILD_PW)
        atv_add_cpp_test(test_cpu_gpw_ks_grid_build cpu/gpw/test_cpu_gpw_ks_grid_build.cpp)
    endif()
    atv_add_cpp_test(test_cpu_xc_pade              cpu/xc/test_cpu_xc_pade.cpp)
    if(ATV_BUILD_PW)  # FFT-backed pieces
        atv_add_cpp_test(test_cpu_gpw_poisson_fft  cpu/gpw/test_cpu_gpw_poisson_fft.cpp)
        atv_add_cpp_test(test_cpu_gpw_hartree_grid cpu/gpw/test_cpu_gpw_hartree_grid.cpp)
        atv_add_cpp_test(test_cpu_gpw_grid_fft     cpu/gpw/test_cpu_gpw_grid_fft.cpp)
        atv_add_cpp_test(test_cpu_gpw_xc_grid_gga  cpu/gpw/test_cpu_gpw_xc_grid_gga.cpp)
        atv_add_cpp_test(test_cpu_gpw_mt_poisson   cpu/gpw/test_cpu_gpw_mt_poisson.cpp)
        # Gaussian DOS free math + multi-k mesh guards (no full SCF e2e).
        atv_add_cpp_test(test_cpu_gpw_dos          cpu/gpw/test_cpu_gpw_dos.cpp)
        # Bitwise-identical grid kernels at 1/2/4/8/16 OpenMP threads. Guards the
        # reduction-order defect that made the OT minimizer take 30-200 SCF
        # iterations for the same input on the same binary.
        atv_add_cpp_test(test_cpu_gpw_thread_reproducibility
                                                   cpu/gpw/test_cpu_gpw_thread_reproducibility.cpp)
        atv_add_cpp_test(test_cpu_gpw_ksv
                                                   cpu/gpw/test_cpu_gpw_ksv.cpp)
        # Name is bandpath (historical); wave-2 emptied HEAVY_REGEX.
        # `pw_bands` which would skip a naive test_cpu_gpw_bands name.
        atv_add_cpp_test(test_cpu_gpw_bandpath
                                                   cpu/gpw/test_cpu_gpw_bandpath.cpp)
        # Cell-precondition unit (orthorhombic-only refuse) — no SCF.
        if(ATV_BUILD_MD)
            atv_add_cpp_test(test_cpu_gpw_evaluator
                                                  cpu/gpw/test_gpw_evaluator.cpp)
        endif()
    endif()

    # Data-table / lattice-sum / volume-force units — no full SCF e2e here.
    # Method-level pins live in regressions/gpw/.
    atv_add_cpp_test(test_cpu_gpw_data_table    cpu/gpw/test_cpu_gpw_data_table.cpp)
    atv_add_cpp_test(test_cpu_gpw_periodic_overlap cpu/gpw/test_cpu_gpw_periodic_overlap.cpp)
    atv_add_cpp_test(test_cpu_gpw_heavy_elements cpu/gpw/test_cpu_gpw_heavy_elements.cpp)
    atv_add_cpp_test(test_cpu_gpw_vol_forces    cpu/gpw/test_cpu_gpw_vol_forces.cpp)
endif()
atv_add_cpp_test(test_cpu_md_pair_morse_smoke     cpu/dynamics/test_pair_morse_smoke.cpp)
atv_add_cpp_test(test_cpu_md_pe_ff_constants      cpu/dynamics/test_pe_ff_constants.cpp)
atv_add_cpp_test(test_cpu_md_pe_energy_xcheck     cpu/dynamics/test_pe_energy_xcheck.cpp)
# pe_tension_protocol grepped benchmarks/md/pe/... which was deleted; re-add
# when the PE tension app is restored under regressions/ or examples/.
atv_add_cpp_test(test_cpu_md_neighbor_list_arena  cpu/dynamics/test_neighbor_list_arena.cpp)
atv_add_cpp_test(test_cpu_md_full_neighbor_list  cpu/dynamics/test_full_neighbor_list.cpp)
atv_add_cpp_test(test_cpu_md_cell_list_grid_stale  cpu/dynamics/test_cell_list_grid_stale.cpp)
atv_add_cpp_test(test_cpu_atom_variable_n  cpu/test_atom_variable_n.cpp)
atv_add_cpp_test(test_cpu_frame_registry  cpu/frame/test_frame_registry.cpp)
atv_add_cpp_test(test_cpu_eam  cpu/potentials/test_eam.cpp)
atv_add_cpp_test(test_cpu_sw  cpu/potentials/test_sw.cpp)
if(ATV_BUILD_VIB)
    atv_add_cpp_test(test_cpu_vib_phonon_dispersion cpu/vibration/test_phonon_dispersion.cpp)
    # esmm-contract-01-zeroing: the Hessian driver must zero kAtom_f* / kEnergy
    # per displacement pass, or an accumulating source halves the curvature.
    atv_add_cpp_test(test_cpu_vib_hessian_zeroing
        cpu/vibration/test_hessian_zeroing.cpp)
    # GPW force-consumer wiring for Phonon (no NeighborList). Full FD spectrum
    # lives in regressions/gpw/gpw-fd-phonon-si (6N SCF — not unit budget).
    if(ATV_BUILD_GPW)
        atv_add_cpp_test(test_cpu_vib_phonon_gpw_wiring
            cpu/vibration/test_phonon_gpw_wiring.cpp)
    endif()
endif()
atv_add_cpp_test(test_cpu_pair_lj_types  cpu/potentials/test_pair_lj_types.cpp)
atv_add_cpp_test(test_cpu_md_fix_freeze  cpu/dynamics/test_fix_freeze.cpp)
atv_add_cpp_test(test_cpu_md_piston_electrode cpu/dynamics/test_piston_electrode.cpp)
atv_add_cpp_test(test_cpu_md_const_potential_bo  cpu/dynamics/test_const_potential_bo.cpp)
atv_add_cpp_test(test_cpu_md_fix_settle  cpu/dynamics/test_fix_settle.cpp)
atv_add_cpp_test(test_cpu_md_virtual_sites  cpu/dynamics/test_virtual_sites.cpp)
atv_add_cpp_test(test_cpu_md_brownian     cpu/dynamics/test_brownian.cpp)
atv_add_cpp_test(test_cpu_md_write_sets  cpu/dynamics/test_write_sets.cpp)
atv_add_cpp_test(test_cpu_path_replica    cpu/path/test_replica.cpp)
atv_add_cpp_test(test_cpu_path_remd       cpu/path/test_remd.cpp)
atv_add_cpp_test(test_cpu_path_string     cpu/path/test_string_method.cpp)
atv_add_cpp_test(test_cpu_path_irc        cpu/path/test_irc.cpp)
atv_add_cpp_test(test_cpu_path_rpmd_est   cpu/path/test_rpmd_estimator.cpp)
# esmm-contract-01-zeroing: PathDriver must zero kAtom_f* / kEnergy before each
# per-replica force pass, or an accumulating source grows with the iteration.
atv_add_cpp_test(test_cpu_path_zeroing    cpu/path/test_path_zeroing.cpp)
# esmm-contract-02-validation-walks: PathDriver::validate_components must see a
# nested `Driver<scf::Phase> + HF` method node as the band's force source.
atv_add_cpp_test(test_cpu_path_validation cpu/path/test_path_validation.cpp)
# esmm-contract-03-runslot-nested: Minimize's StrongWolfe trials must re-enter a
# nested `Driver<scf::Phase> + HF` force source (run_stage → Runtime::run_slot_).
# Stub physics (harmonic well) — no SCF, milliseconds.
atv_add_cpp_test(test_cpu_minimizer_line_search_nested_node
                 cpu/minimize/test_line_search_nested_node.cpp)
atv_add_cpp_test(test_cpu_criterion       cpu/driver/test_criterion.cpp)
atv_add_cpp_test(test_cpu_driver_nesting  cpu/driver/test_driver_nesting.cpp)
# scf-driver-refactor-01-plumb: scf::Phase / PhaseKind::SCF / PhaseTraits<scf::Phase>
# / the Criterion component kind + driver pull-poll. Pure type-foundation plumbing —
# no consumer yet.
atv_add_cpp_test(test_cpu_scf_phase_plumbing cpu/driver/test_scf_phase_plumbing.cpp)
atv_add_cpp_test(test_cpu_component_ownership cpu/driver/test_component_ownership.cpp)
# esmm-contract-04-forces-requested: `forces_requested` as a compiled Plan-slot
# property — MC's energy phase is energy-only, MD/Minimize/Evaluate still ask for
# forces, and the veto ANDs through a nested driver's own slots. Stubs only, no
# physics, microseconds; the QM-in-MC e2e is regressions/mc/mc_bias_es.
atv_add_cpp_test(test_cpu_forces_requested cpu/driver/test_forces_requested.cpp)
# test_cpu_md_pe_nvt relocated to benchmarks/md/pe/validation/ (val_md_pe_nvt) —
# a full NVT equilibration of the bonded PE melt, not a ms unit test.
atv_add_cpp_test(test_cpu_md_frame_tag_image      cpu/dynamics/test_frame_tag_image.cpp)
atv_add_cpp_test(test_cpu_md_domain_decomposition cpu/dynamics/test_domain_decomposition.cpp)
atv_add_cpp_test(test_cpu_analysis_executor       cpu/test_analysis_executor.cpp)
atv_add_cpp_test(test_cpu_observer_msd            cpu/observe/test_msd.cpp)
atv_add_cpp_test(test_cpu_observer_vacf           cpu/observe/test_vacf.cpp)
atv_add_cpp_test(test_cpu_observer_widom          cpu/observe/test_widom.cpp)
atv_add_cpp_test(test_cpu_observer_electrode_charge cpu/observe/test_electrode_charge_observer.cpp)
atv_add_cpp_test(test_cpu_observer_capacitance_charge_fluctuation
                 cpu/observe/test_capacitance_charge_fluctuation.cpp)
atv_add_cpp_test(test_cpu_observer_potential_profile
                 cpu/observe/test_potential_profile.cpp)
atv_add_cpp_test(test_cpu_observer_species_density_profile
                 cpu/observe/test_species_density_profile.cpp)

atv_add_cpp_test(test_cpu_driver_role_validation cpu/test_cpu_driver_role_validation.cpp)
# Registration-time PhaseKind gate + five-driver plan-shape regression
# (arch-refactor-01-bindings). Uses AM1Evaluator (unconditional, no RHF gate).
atv_add_cpp_test(test_cpu_driver_binding_kind             cpu/test_driver_binding_kind.cpp)
atv_add_cpp_test(test_cpu_evaluate_driver_capabilities    cpu/test_evaluate_driver_capabilities.cpp)
atv_add_cpp_test(test_cpu_optimize_driver_early_stop      cpu/test_optimize_driver_early_stop.cpp)
atv_add_cpp_test(test_cpu_dynamics_full_budget            cpu/test_dynamics_full_budget.cpp)
atv_add_cpp_test(test_cpu_install_infra_hook              cpu/test_install_infra_hook.cpp)
atv_add_cpp_test(test_cpu_md_set_api                      cpu/dynamics/test_md_set_api.cpp)
atv_add_cpp_test(test_cpu_md_plan_equivalence             cpu/dynamics/test_md_plan_equivalence.cpp)
# TimestepProvider/TimestepConsumer opt-in protocol + the MD build-time push
# (esmm-cpmd-02-timestep-protocol): dt is single-sourced from the integrator,
# conflicting/absent/invalid publications throw at build. Local stubs only —
# no GPW dependency, one no-op MD step to fire the component build() pass.
atv_add_cpp_test(test_cpu_md_timestep_protocol            cpu/dynamics/test_timestep_protocol.cpp)
# Car-Parrinello conserved-energy assembler (esmm-cpmd-03-observability):
# kConserved = kEnergy + kKinetic + the two electronic keys, end_of_step @ -90,
# throw when the electronic keys are absent. Pure meta arithmetic on a seeded
# one-atom frame — no GPW, no SCF, no MD step.
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

# Unreadable input paths must throw std::runtime_error, not abort in Rust.
atv_add_cpp_test(test_cpu_io_missing_file                 cpu/io/test_cpu_io_missing_file.cpp)

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

    # Ghost-charge refresh after a mid-step Q rewrite (const-pot MPI DD).
    atv_add_mpi_exec(test_mpi_halo_refresh_charges mpi/test_halo_refresh_charges.cpp)
    atv_add_mpi_run(test_mpi_halo_refresh_charges_n1 test_mpi_halo_refresh_charges 1)
    atv_add_mpi_run(test_mpi_halo_refresh_charges_n2 test_mpi_halo_refresh_charges 2)


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


    atv_add_mpi_exec(test_mpi_nose_hoover_npt_multirank
                     mpi/test_nose_hoover_npt_multirank.cpp)
    atv_add_mpi_run(test_mpi_nose_hoover_npt_multirank_n8
                    test_mpi_nose_hoover_npt_multirank 8)

    atv_add_mpi_exec(test_mpi_deform_multirank mpi/test_deform_multirank.cpp)
    atv_add_mpi_run(test_mpi_deform_multirank_n8 test_mpi_deform_multirank 8)

    atv_add_mpi_exec(test_mpi_eam_multirank mpi/test_eam_multirank.cpp)
    atv_add_mpi_run(test_mpi_eam_multirank_n2 test_mpi_eam_multirank 2)
    atv_add_mpi_run(test_mpi_eam_multirank_n4 test_mpi_eam_multirank 4)


    atv_add_mpi_exec(test_mpi_sw_multirank mpi/test_sw_multirank.cpp)
    atv_add_mpi_run(test_mpi_sw_multirank_n2 test_mpi_sw_multirank 2)
    atv_add_mpi_run(test_mpi_sw_multirank_n4 test_mpi_sw_multirank 4)


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
    atv_add_cpp_test(test_cpu_pw_g_vector   cpu/planewave/test_g_vector.cpp)
    atv_add_cpp_test(test_cpu_pw_upf        cpu/planewave/test_upf.cpp)
    atv_add_cpp_test(test_cpu_pw_gth_upf    cpu/planewave/test_gth_upf.cpp)
    atv_add_cpp_test(test_cpu_pw_kpoint     cpu/planewave/test_kpoint.cpp)
    atv_add_cpp_test(test_cpu_pw_crystal    cpu/planewave/test_crystal.cpp)
    atv_add_cpp_test(test_cpu_pw_ewald      cpu/planewave/test_ewald.cpp)
    atv_add_cpp_test(test_cpu_pw_driver_node cpu/planewave/test_cpu_pw_driver_node.cpp)
    # ctest runs from the build tree, so the in-tree Si UPF must be located
    # absolutely — a relative path silently GTEST_SKIPped the only UPF force
    # coverage in the suite.
    if(TARGET test_cpu_pw_forces)
        target_compile_definitions(test_cpu_pw_forces
            PRIVATE ATV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    endif()

    # FFT test needs FFTW3 linkage
    add_executable(test_cpu_pw_fft cpu/planewave/test_fft.cpp)
    target_link_libraries(test_cpu_pw_fft PRIVATE atomiverse GTest::gtest_main fftw3_stub)
    set_target_properties(test_cpu_pw_fft PROPERTIES CXX_STANDARD 20)
    target_include_directories(test_cpu_pw_fft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME test_cpu_pw_fft COMMAND test_cpu_pw_fft)
    if(TARGET test_cpu_pw_fft)
    set_tests_properties(test_cpu_pw_fft PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${ATV_TEST_OMP_THREADS}")
    endif()

    # MPI PW/GPW k-point multi-iter SCF: e2e-in-tests wave-2 deleted the unit
    # sources. Industrial gates live in regressions/pw/mpi-pw-scaling-4rank and
    # regressions/gpw/mpi-gpw-scaling-4rank (not under ctest default gate).
endif()

# ── Shared dense linear algebra (cpu/linalg) ──────────────────────────

# Canonical orthogonalization with S-eigenvalue thresholding
# (hf-robust-02). Header-only cpu/linalg/eigensolve.h — no RHF gate.
atv_add_cpp_test(test_cpu_linalg_canonical_orth
                 cpu/linalg/test_canonical_orthogonalization.cpp)

# Symmetric matrix power + nonsymmetric (dgeev) eigensolver
# (scf-review-01-linalg). Header-only cpu/linalg/eigensolve.h — no RHF gate.
atv_add_cpp_test(test_cpu_linalg_eigensolve
                 cpu/linalg/test_eigensolve.cpp)

# Preconditioned conjugate gradient (scf-review-01-linalg). Header-only
# cpu/linalg/linear_solve.h — no RHF gate.
atv_add_cpp_test(test_cpu_linalg_linear_solve
                 cpu/linalg/test_linear_solve.cpp)

# ── Electronic-state propagator tests ─────────────────────────────────
#
# esmm-cpmd-01-electronic-home: Cpmd/CpmdCplx (Car-Parrinello RATTLE),
# Cpmd/CpmdCplx and ASPCPredictor(Cplx) live in cpu/electronic/ —
# method-agnostic dense algebra over cpu/linalg, consumed by Gaussian SCF
# *and* GPW. Their sources compile under ATV_BUILD_HF (see the root
# CMakeLists HF source block), so these units gate on HF only — deliberately
# NOT on ATV_BUILD_GPW. Every test here is pure propagator algebra; the GPW
# build()/guess_P integration guards stay in the ATV_BUILD_GPW section above.
# XlBomd/XlBomdCplx compile in the always-on set (classical PIM XL dipoles
# consume the recursion in MD-only builds), so its unit is unconditional.


if(ATV_BUILD_HF)
    atv_add_cpp_test(test_cpu_electronic_aspc_predictor
                     cpu/electronic/test_aspc_predictor.cpp)
endif()

# ── RHF/SCF tests ─────────────────────────────────────────────────────

if(ATV_BUILD_HF)
    atv_add_cpp_test(test_cpu_scf_cart2sph     cpu/scf/test_cart2sph.cpp)
    atv_add_cpp_test(test_cpu_scf_multipole_ints cpu/scf/test_multipole_integrals.cpp)
    atv_add_cpp_test(test_cpu_observer_multipole cpu/observe/test_multipole.cpp)
    # Löwdin populations + MK/CHELPG ESP fit — hand-built shells/determinant
    # and synthetic point-charge potentials only (hf-capability-gaps-04).
    atv_add_cpp_test(test_cpu_observer_lowdin cpu/observe/test_lowdin.cpp)
    atv_add_cpp_test(test_cpu_observer_esp_charges cpu/observe/test_esp_charges.cpp)
    atv_add_cpp_test(test_cpu_scf_spherical    cpu/scf/test_spherical_basis.cpp)
    atv_add_cpp_test(test_cpu_scf_ecp_gradient cpu/scf/test_ecp_gradient.cpp)
    # Static basis-table structure for the third-period + diffuse extension
    # (hf-capability-gaps-05-basis) — table pins vs PySCF, no SCF solve.
    atv_add_cpp_test(test_basis_tables         cpu/scf/test_basis_tables.cpp)
    # Semi-local def2-ECP: static table pins, EcpPotential lookup,
    # EcpIntegrals U-matrix vs external PySCF ECPscalar pins, IntegralWorkspace
    # valence wiring, x2c double-counting gate (hf-capability-gaps-06-ecp).
    atv_add_cpp_test(test_ecp                  cpu/scf/test_ecp.cpp)
    # hf-capability-gaps-18-x2c: spin-free pVp kernel pins, X2cWorkspace
    # nonrel/exact-decoupling identities, ECP↔x2c mutual exclusion both
    # ways, forces() intercept. No heavy-atom SCF in this unit binary.
    # hf-capability-gaps-10-rohf: ROHF (Roothaan effective Fock, PySCF
    # canonical / Guest-Saunders convention) vs externally pinned PySCF.
    # hf-capability-gaps-11-ghf: real two-component GHF — spinor
    # SlaterDeterminant invariants, H2 collapse to UHF, noncollinear win on
    # the frustrated H3 triangle vs externally pinned PySCF, collinear trap.
    atv_add_cpp_test(test_cpu_scf_xc_functional cpu/scf/test_xc_functional.cpp)
    # scf-driver-refactor-04b-wire: method-driver is DriverNode + outer Plan
    # bind/build only (no full SCF under tests/; wave-2). Nested energy pins:
    # regressions/architecture/lj_rhf_coupled + scf/*rhf*.
    atv_add_cpp_test(test_cpu_scf_method_driver_wire
                     cpu/scf/test_method_driver_wire.cpp)
    # esmm-cpmd-04-ao-cpmd: AO Car-Parrinello on the mean-field methods —
    # the refusal lattice, the TimestepConsumer seam, ScfGradient::
    # compute_at_density parity at a converged determinant, and the
    # zero-SCF-iteration production signature. H2/STO-3G only (2 AOs); the
    # E_cons drift physics is regressions/dynamics/cpmd-{rhf-h2,rks-water,
    # uhf-o2}.
    # BOMD+d3bj/d4 conservativeness: `MeanFieldMethod` adds E_disp to the total
    # energy, so `internal::ScfGradient` must add −∂E_disp/∂R to the force. That
    # reduces to whether D3BJ/D4 `forces()` differentiates their own `energy()`
    # — measured here by central difference, no SCF (ms). The end-to-end
    # conservation witness is regressions/scf/bomd-d3bj-ar2-nve.
    # esmm-cpmd-05-scf-xlbomd: XL-BOMD (Niklasson JCP 130, 214109) as the
    # Gaussian-SCF guess — the xl_bomd/xl_bomd_k member surface, the
    # exclusivity gates (ASPC predictor, cpmd, Table-I order K in [3,9]),
    # auxiliary-history integrity (advance only on converged nested runs), and
    # the guess ladder explicit > xl > ASPC > last-P. H2O/STO-3G on a
    # ballistic nested-MD trajectory, ~1 s; the warm-step iteration margin and
    # the drift envelope are regressions/dynamics/xl-bomd-rhf-water.
    # esmm-contract-04-forces-requested: the ASPC-predictor-under-MC guard on
    # MeanFieldMethod::run_nested_erased — extrapolated density histories are
    # MD-only (rejected trials break the continuity the extrapolation assumes),
    # while plain last-P reuse under MC stays legal. H2/STO-3G, one sweep.
    # esmm-contract-05-accumulate-switch: the ONE entry that runs two SCF
    # epilogues — the ASPC corrector-miss fallback — must leave exactly one
    # contribution in the driver-owned accumulators (kEnergy /
    # kEnergyElectronic / kAtom_f*) now that the in-plan entry accumulates.
    # Four 6-step H2/STO-3G nested trajectories, ~20 ms. Serial for the same
    # ULP-determinism reason as test_cpu_scf_method_driver_parity: the pin is
    # bit-identity against the method's own energy() readback.
    if(TARGET test_cpu_scf_aspc_fallback_accumulate)
    set_tests_properties(test_cpu_scf_aspc_fallback_accumulate PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    endif()
    # hf-capability-gaps-08-stability: MO orbital-Hessian A/B kernel +
    # Seeger-Pople stability analysis. NO SCF loop: the converged H2/STO-3G
    # determinant is symmetry-exact in closed form (one Fock build, no
    # iteration); the full RHF.run()+stability() path is
    # regressions/scf/h2_rhf_stability.
    atv_add_cpp_test(test_cpu_scf_orbital_hessian cpu/scf/test_orbital_hessian.cpp)
    atv_add_cpp_test(test_cpu_scf_stability       cpu/scf/test_stability.cpp)
    # hf-capability-gaps-14-tdhf: TDA + TDHF/RPA pack units (no SCF loop).
    # Formaldehyde / H2 TDA-force / NAC / ALDA e2e: regressions/scf/{ch2o_tdhf,
    # h2_tda_force,h2_analytic_tda_force,h2_tda_nac,h2_nac_vectors,h2_full_grid_alda}.
    atv_add_cpp_test(test_cpu_scf_tdhf_pack       cpu/scf/test_tdhf_pack.cpp)
    # hf-capability-gaps-16-solvent: C-PCM cavity/Born oracle, f(ε), H2O
    # solvent vs gas, solvent_eps=0 vacuum identity, forces throw.
    # hf-capability-gaps-17-bsse: ghost AtomList/ShellList/IntegralWorkspace
    # workspace semantics + Boys–Bernardi Counterpoise 5-SCF identities
    # (He–He/STO-3G; locked bsse ≤ 0). Water-dimer e2e lives in
    # regressions/scf/h2o_dimer_cp_hf631gs.
    atv_add_cpp_test(test_ghost_basis             cpu/scf/test_ghost_basis.cpp)
    # hf-capability-gaps-13-cphf: Hessian multi-SCF first-gates removed in
    # e2e-in-tests wave-2 (were full RHF/UHF.hessian campaigns under tests/).
    # E2e: regressions/scf/{h2o_rhf_freq,uhf-hessian-o2,uks-hessian-o2-sto3g,
    # rks-hessian-h2o-sto3g,rohf-hessian-o2}.
    # scf-review-04-meanfield-dedup: the FD nuclear-Hessian skeleton hoisted to
    # the protected CRTP template
    # MeanFieldMethod<Derived>::central_difference_nuclear_hessian, pinned in
    # isolation against an analytic quadratic mock force field (no SCF); and the
    # Ha/Bohr->eV/Angstrom gradient glue hoisted onto the tag ctor
    # GradientResult(FromHartreeBohr, ...), pinned against hand-computed
    # eV/Angstrom + eV literals (1e-12).
    atv_add_cpp_test(test_cpu_scf_fd_hessian_template
                     cpu/scf/test_fd_hessian_template.cpp)
    atv_add_cpp_test(test_cpu_scf_gradient_result
                     cpu/scf/test_gradient_result.cpp)
    # hf-capability-gaps-09-soscf: CIAH stepper units — exp(κ) unitarity,
    # gradient vs FD, matrix-free CIAH vs dense augmented eigensolve, trust
    # region, MixStage latch; KS+soscf throw. No full RHF.run loop (wave-2).
    # Hard-case contrast: regressions/scf/soscf_hard_case.
    atv_add_cpp_test(test_soscf                   cpu/scf/test_soscf.cpp)
    # scf-review-02-eri-hoists: hosted spin-orbital ERI helpers
    # (SpinOrbital / antisym_eri / mp_denominator) in internal/spin_orbital_eri.h.
    atv_add_cpp_test(test_cpu_scf_spin_orbital_eri cpu/scf/test_spin_orbital_eri.cpp)
    # scf-review-03-posthf-dedup: canonical determinant-CI evaluators
    # (choose/enumerate_strings/jw_single_sign/Slater-Condon elements/ci_element)
    # hoisted to internal/ci_engine.h; hand-computed 2-orbital pins (1e-12).
    atv_add_cpp_test(test_cpu_scf_ci_engine     cpu/scf/test_ci_engine.cpp)
    # scf-review-03-posthf-dedup: the single hoisted make_tau in
    # internal/eom_ccsd_common.h; hand-computed tau pins (1e-12, fac=1 and 0.5).
    atv_add_cpp_test(test_cpu_scf_eom_ccsd_common cpu/scf/test_eom_ccsd_common.cpp)
    atv_add_cpp_test(test_cpu_scf_integrals     cpu/scf/test_integrals.cpp)
    # scf-review-07-posthf-perf: block-assembled dense ERI (item 17) equivalence
    # guard — IntegralWorkspace::eri_dense vs per-AO native::eri reference (≤1e-12) +
    # hard-coded anchors. Mirrors src/cpu/scf/two_electron.cpp.
    atv_add_cpp_test(test_cpu_scf_two_electron  cpu/scf/test_two_electron.cpp)
    atv_add_cpp_test(test_cpu_scf_rys_roots     cpu/scf/test_rys_roots.cpp)
    atv_add_cpp_test(test_cpu_scf_rys_eri       cpu/scf/test_rys_eri.cpp)
    # scf-review-06-scf-perf: fused one-sweep jk_rhf vs two-pass g_rhf+k
    # (≤1e-12 Ha F gate, d-shell fixture) + coulomb_exchange_fused thread
    # invariance. Mirrors cpu/scf/operators.cpp.
    atv_add_cpp_test(test_cpu_scf_operators     cpu/scf/test_operators.cpp)
    # hf-capability-gaps-12-df: def2-JKFIT table, g-shell 2c/3c integrals,
    # metric whitening, fitted J/K vs dense (single calls, no SCF).
    atv_add_cpp_test(test_density_fitting       cpu/scf/test_density_fitting.cpp)
    # hf-capability-gaps-21-cfmm: shell-pair distributions, near/far tables,
    # single H2O/STO-3G j(P) vs dense direct J (no converged SCF).
    atv_add_cpp_test(test_cfmm_j                cpu/scf/test_cfmm_j.cpp)
    atv_add_cpp_test(test_cpu_scf_primitives    cpu/scf/test_primitives.cpp)
    atv_add_cpp_test(test_cpu_scf_accelerators  cpu/scf/test_accelerators.cpp)
    # scf-review-08-ccsd-spatial-diis: CC amplitude DIIS accelerator
    # (internal/cc_diis.h) over flattened [t1;t2] vectors — Pulay B-matrix on a
    # contractive linear toy + single/near-singular/state edges (pure algebra).
    atv_add_cpp_test(test_cpu_scf_cc_diis       cpu/scf/test_cc_diis.cpp)
    # scf-review-08-ccsd-spatial-diis: closed-shell spatial-orbital CCSD kernel
    # (internal/ccsd_spatial.h) — H2/STO-3G E_corr pin (same PySCF pin as
    # regressions/scf/ccsd-h2-sto3g) + He/STO-3G zero-virtual edge.
    # hf-capability-gaps-07-guess: Hückel/SAP guess matrices + EDIIS/ADIIS
    # far-field mixers (single-call math only; iteration-count contrast is
    # regressions/scf/h2o_scf_convergence).
    atv_add_cpp_test(test_cpu_scf_init_guess    cpu/scf/test_init_guess.cpp)
    atv_add_cpp_test(test_cpu_scf_ediis_adiis   cpu/scf/test_ediis_adiis.cpp)
    # Charge/multiplicity resolver + occupation-guard throws (hf-robust-01).
    # Converged UHF e2e is regressions/scf/*uhf*; no multi-iter here (wave-2).
    atv_add_cpp_test(test_cpu_scf_spin_resolution cpu/scf/test_spin_resolution.cpp)
    # SlaterDeterminant value type: validating ctors, per-spin density,
    # expect/homo/lumo, with_occupation, validate(S)
    # (scf-slater-determinant-01-type).
    atv_add_cpp_test(test_cpu_scf_slater_determinant cpu/scf/test_slater_determinant.cpp)
    # Public SCF vocabulary types re-homed into dedicated headers: scf_enums.h,
    # gradient_result.h, scf_guess.h — without including scf_result.h
    # (scf-slater-determinant-02-rehome).
    atv_add_cpp_test(test_scf_type_homes cpu/scf/test_scf_type_homes.cpp)
    # scf-driver-refactor-02-stages: five scf::Phase stage components wrapping
    # the internal SCF helpers, isolated bit-identical parity (unit tier).
    atv_add_cpp_test(test_cpu_scf_stages cpu/scf/stages/test_scf_stages.cpp)
    # Bit-identical parity compares two independent builds of the same kernels;
    # parallel reduction order (OpenMP / BLAS) is ULP-non-deterministic, so this
    # test pins serial execution. Do NOT loosen the assertions to tolerances —
    # the spec mandates bit-identity.
    if(TARGET test_cpu_scf_stages)
    set_tests_properties(test_cpu_scf_stages PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    endif()
    # scf-driver-refactor-03a-convert: RHF/UHF/RKS/UKS as Driver<scf::Phase>,
    # regression-pinned against the 03a-parity energies/eigenvalues. Pins serial
    # execution for the same ULP-determinism reason as test_cpu_scf_stages.
    if(TARGET test_cpu_scf_method_driver_parity)
    set_tests_properties(test_cpu_scf_method_driver_parity PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    endif()
    atv_add_cpp_test(test_cpu_evaluate_driver        cpu/test_evaluate_driver.cpp)
    # esmm-contract-05-accumulate-switch (U7): two accumulating stubs co-bound
    # into ONE cpu::Evaluate pass (no SCF; wave-2). QM×MM pin lives in
    # regressions/architecture/lj_rhf_coupled.
    atv_add_cpp_test(test_cpu_evaluate_source_cobinding
                     cpu/evaluate/test_source_cobinding.cpp)
    atv_add_cpp_test(test_cpu_scf_minao         cpu/scf/test_minao.cpp)
    atv_add_cpp_test(test_cpu_scf_rys_eri_deriv cpu/scf/test_rys_eri_deriv.cpp)
    # test_cpu_minimizer_lbfgs relocated to benchmarks/qc/rhf/validation/
    # (val_min_lbfgs_geoopt) — full SCF-driven H2 geometry optimizations.
    atv_add_cpp_test(test_cpu_minimizer_line_search cpu/minimize/test_line_search.cpp)
    atv_add_cpp_test(test_cpu_minimize_driver_swap_optimizer cpu/test_minimize_driver_swap_optimizer.cpp)
    # Vibrational analysis (finite-difference Hessian) is an end-to-end /
    # PySCF-parity suite — too slow for the unit gate. RELOCATED to
    # benchmarks/qc/vib/validation/ (run under -DATV_BUILD_BENCHMARKS=ON).
    atv_add_cpp_test(test_cpu_recorder_density_cube cpu/record/test_density_cube.cpp)
    # Mirrors cpu/scf/xc_functional.h — analytical hard-coded goldens (no live libxc).
    atv_add_cpp_test(test_xc_reference         cpu/scf/test_xc_reference.cpp)
    # scf-review-06-scf-perf: geometry-fixed XCGridBasis cache equivalence —
    # cached-basis vs rebuild-every-call evaluate_xc_batched are bitwise-equal
    # for LDA and PBE. Mirrors cpu/scf/xc_grid_engine.h.
    atv_add_cpp_test(test_xc_grid_engine       cpu/scf/test_xc_grid_engine.cpp)
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
    atv_add_test(test_cuda_multi_gpu  cuda/dynamics/test_multi_gpu.cu)
    atv_add_test(test_cuda_frame      cuda/test_frame.cu)
    atv_add_test(test_cuda_materialize cuda/test_materialize.cu)
    atv_add_test(test_cuda_backend    cuda/test_backend.cu)
    atv_add_test(test_core_plan       core/test_plan.cu)
    atv_add_test(test_cuda_md_zero_momentum cuda/dynamics/test_zero_momentum.cu)
    atv_add_test(test_cuda_neighborlist_skin_gating cuda/dynamics/test_neighborlist_skin_gating.cu)
    atv_add_test(test_cuda_cluster_contract cuda/dynamics/test_cuda_cluster_contract.cu)
    atv_add_test(test_cuda_cluster_mask cuda/dynamics/test_cuda_cluster_mask.cu)
    atv_add_test(test_pair_lj_tile_step0 cuda/potentials/test_pair_lj_tile_step0.cu)
    atv_add_test(test_cuda_atom_cell_sort cuda/dynamics/test_atom_cell_sort.cu)
    # Pulls in the private primitive header src/cuda/neighbor/atom_cell_sort.cuh
    target_include_directories(test_cuda_atom_cell_sort PRIVATE
        ${CMAKE_SOURCE_DIR}/src/cuda/neighbor)
    atv_add_test(test_cuda_pair_lj_no_d2h cuda/test_pair_lj_no_d2h.cu)
    target_compile_definitions(test_cuda_pair_lj_no_d2h PRIVATE
        ATV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    atv_add_test(test_cuda_driver_no_per_phase_sync cuda/test_driver_no_per_phase_sync.cu)
    target_compile_definitions(test_cuda_driver_no_per_phase_sync PRIVATE
        ATV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    # cuda::RDF on the molrs streaming accumulator (arch-refactor-07-molrs):
    # static-trajectory g(r) parity vs a host-fed cpu::compute::RDFAccumulator
    # + fail-fast ctor edges. Needs a GPU node (runs a cuda::md::MD).
    atv_add_test(test_cuda_observer_rdf_molrs cuda/observe/test_rdf_molrs.cu)
    atv_add_test(test_cuda_pair_lj_special_bonds cuda/potentials/test_pair_lj_special_bonds.cu)
    # pair_kernel is cuda-owned (backend-mirror-02-absorb); its scalar-math test
    # moved here from the former cpu/potentials/test_pair_kernel.cpp.
    atv_add_test(test_cuda_pair_kernel cuda/potentials/test_pair_kernel.cu)
    atv_add_test(test_cuda_ewald_coul cuda/longrange/test_ewald_coul.cu)
    if(ATV_BUILD_KSPACE)
        atv_add_test(test_cuda_pme_coul cuda/longrange/test_pme_coul.cu)
    endif()
    # Manual A/B (not ctest): OpenMM cluster vs LAMMPS Verlet nonbonded path.
    # Run: ATV_CUDA_NONBONDED=openmm|lammps ./cuda_nonbonded_style_bench [n_side] [steps]
    add_executable(cuda_nonbonded_style_bench
                   ${CMAKE_SOURCE_DIR}/tools/cuda_nonbonded_style_bench.cu)
    target_link_libraries(cuda_nonbonded_style_bench PRIVATE atomiverse)
    set_target_properties(cuda_nonbonded_style_bench PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    # Manual (not ctest): CPU vs CUDA RHF wall + AO scaling.
    # Run: OMP_NUM_THREADS=8 ./bin/bench_cpu_cuda_rhf [n_rep_e] [n_rep_f]
    if(ATV_BUILD_HF)
        add_executable(bench_cpu_cuda_rhf
                       ${CMAKE_SOURCE_DIR}/tools/bench_cpu_cuda_rhf.cu)
        target_link_libraries(bench_cpu_cuda_rhf PRIVATE atomiverse)
        set_target_properties(bench_cpu_cuda_rhf PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON
                              RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        # OpenMP for fair multi-thread CPU leg on the GPU node.
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(bench_cpu_cuda_rhf PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
    if(ATV_BUILD_PW)
        atv_add_test(test_cuda_pw_fft cuda/planewave/test_fft.cu)
        target_link_libraries(test_cuda_pw_fft PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_poisson cuda/planewave/test_poisson.cu)
        target_link_libraries(test_cuda_pw_poisson PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_grid_map cuda/planewave/test_grid_map.cu)
        target_link_libraries(test_cuda_pw_grid_map PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_xc_lda cuda/planewave/test_xc_lda.cu)
        target_link_libraries(test_cuda_pw_xc_lda PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_xc_pbe cuda/planewave/test_xc_pbe.cu)
        target_link_libraries(test_cuda_pw_xc_pbe PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_xc_pbe_spin cuda/planewave/test_xc_pbe_spin.cu)
        target_link_libraries(test_cuda_pw_xc_pbe_spin PRIVATE CUDA::cufft)
        atv_add_test(test_cuda_pw_veff cuda/planewave/test_veff.cu)
        target_link_libraries(test_cuda_pw_veff PRIVATE CUDA::cufft)
        # cuda::pw::PW surface + honesty guards. No SCF, no device work:
        # every assertion fires before setup_pw_problem and before CUDA.
        atv_add_test(test_cuda_pw_guards cuda/planewave/test_guards.cu)
        if(TARGET test_cuda_pw_guards)
            target_link_libraries(test_cuda_pw_guards PRIVATE CUDA::cufft)
        endif()
        # DeviceDensity stage parity vs the CPU stage helpers. Unit-only:
        # no SCF loop, no Davidson — the end-to-end parity solve lives in
        # regressions/pw/pw-cuda-lda-si.
        atv_add_test(test_cuda_pw_density cuda/planewave/test_density.cu)
        if(TARGET test_cuda_pw_density)
            target_link_libraries(test_cuda_pw_density PRIVATE CUDA::cufft)
        endif()
        # Device local HF force kernel + public API symmetry (unit-only).
        # Name keeps historical token; full SCF product e2e is regressions.
        # analytic path on CPU); this is a kernel/API unit gate.
        atv_add_test(test_cuda_pw_force_local cuda/planewave/test_pw_force_local.cu)
        if(TARGET test_cuda_pw_force_local)
            target_link_libraries(test_cuda_pw_force_local PRIVATE CUDA::cufft)
        endif()
        # Cell stress public API + cheap H isotropy/CUDA↔CPU (unit-only).
        # Name keeps historical token; full SCF product e2e is regressions.
        atv_add_test(test_cuda_pw_cell_stress cuda/planewave/test_pw_cell_stress.cu)
        if(TARGET test_cuda_pw_cell_stress)
            target_link_libraries(test_cuda_pw_cell_stress PRIVATE CUDA::cufft)
        endif()
    endif()
    # CUDA GPW Γ Poisson foothold — needs CPU solve_poisson oracle (GPW+PW)
    # and device kernels.
    if(ATV_BUILD_GPW AND ATV_BUILD_PW)
        atv_add_test(test_cuda_gpw_device_poisson cuda/gpw/test_device_poisson.cu)
        target_link_libraries(test_cuda_gpw_device_poisson PRIVATE CUDA::cufft)
        # CUDA GPW Padé LDA XC foothold — vs cpu::xc::pade_lsd / xc_grid_pade.
        atv_add_test(test_cuda_gpw_xc_lda cuda/gpw/test_xc_lda.cu)
        # CUDA GPW product ρ collocate foothold — vs CPU collint oracle.
        atv_add_test(test_cuda_gpw_product_collocate cuda/gpw/test_product_collocate.cu)
        # CUDA GPW product V·shape integrate foothold — vs CPU collint oracle.
        atv_add_test(test_cuda_gpw_product_integrate cuda/gpw/test_product_integrate.cu)
        # CUDA GPW composed KS chain stage — vs CPU ks_grid_build_rks.
        atv_add_test(test_cuda_gpw_ks_chain cuda/gpw/test_ks_chain.cu)
        if(TARGET test_cuda_gpw_ks_chain)
            target_link_libraries(test_cuda_gpw_ks_chain PRIVATE CUDA::cufft)
        endif()
        atv_add_test(test_cuda_gpw_ks_gga cuda/gpw/test_ks_gga.cu)
        if(TARGET test_cuda_gpw_ks_gga)
            target_link_libraries(test_cuda_gpw_ks_gga PRIVATE CUDA::cufft)
        endif()
        # CUDA multi-grid product collocate/integrate KS stage vs CPU Γ
        # multigrid fragment (prolong_fft / restrict_fft + ks_grid_build_rks).
        atv_add_test(test_cuda_gpw_ks_multigrid cuda/gpw/test_ks_multigrid.cu)
        if(TARGET test_cuda_gpw_ks_multigrid)
            target_link_libraries(test_cuda_gpw_ks_multigrid PRIVATE CUDA::cufft)
        endif()
        # Public cuda::gpw::GpwDft first product (Γ RKS DIIS + device KS).
        atv_add_test(test_cuda_gpw_full_product cuda/gpw/test_gpw_full_product.cu)
        if(TARGET test_cuda_gpw_full_product)
            target_link_libraries(test_cuda_gpw_full_product PRIVATE CUDA::cufft)
        endif()
    endif()
    atv_add_test(test_cuda_scf        cuda/scf/test_scf.cu)
    atv_add_test(test_cuda_scf_rhf    cuda/scf/test_rhf.cu)
    atv_add_test(test_cuda_scf_device_integrals cuda/scf/test_device_integrals.cu)
    # CPU/CUDA PW parity is an end-to-end SCF solve: it lives in
    # regressions/pw/pw-cuda-lda-si (--device=both), never under ctest.
    atv_add_test(test_cuda_driver_role_validation cuda/test_cuda_driver_role_validation.cu)
    # Registration-time PhaseKind gate, CUDA mirror
    # (arch-refactor-01-bindings). add()-only — needs no CUDA device.
    atv_add_test(test_cuda_driver_binding_kind cuda/test_driver_binding_kind.cu)

    # AsyncRecordBase / AsyncBridge contract guards. The subdir registers
    # benchmark-independent tests unconditionally and gates the
    # polyethylene_tension-dependent ones on ATV_BUILD_BENCHMARKS internally.
    add_subdirectory(cuda/asynchronous)

endif()


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
    add_executable(diag_band_parity cuda/planewave/diag_band_parity.cu)
    target_link_libraries(diag_band_parity PRIVATE atomiverse)
    set_target_properties(diag_band_parity PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ctest policy reminder: multi-step Driver pipelines (MD/Minimize/Path
# run(N) for equilibration / production-style trajectories, multi-rank
# PE/MD campaigns) are DELETED from this tree — not "disabled". Public
# end-to-end scenarios live under regressions/ (ATV_BUILD_REGRESSIONS,
# never ctest). Acceptance / parity gates live under benchmarks/** when
# that tree is rebuilt. If a target is registered above, it must be a
# single-component unit test.

# Enforce the unit timeout on every registration in this directory (including
# raw add_test scrub scripts). Hard kill — no 300s architecture greps on pre-push.
get_property(_atv_all_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
if(_atv_all_tests)
  set_tests_properties(${_atv_all_tests} PROPERTIES TIMEOUT ${ATV_TEST_TIMEOUT_SECONDS})
endif()
