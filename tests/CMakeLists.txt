include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# Unit tests assert CODE CORRECTNESS on tiny systems; one OpenMP thread keeps
# them free of fork/join overhead and makes reductions bit-reproducible (some
# tests compare two runs bit for bit).
set(ATV_TEST_OMP_THREADS 1 CACHE STRING
    "OMP_NUM_THREADS for single-process unit tests")
# MPI unit tests: 1 thread/rank — multi-rank campaigns are not unit tests.
set(ATV_TEST_MPI_OMP_THREADS 1 CACHE STRING
    "OMP_NUM_THREADS per rank for MPI unit tests")
# HARD CAP: every ctest registration dies at this wall time. If a "unit"
# needs longer, it is not a unit — move to regressions/ or delete.
set(ATV_TEST_TIMEOUT_SECONDS 10 CACHE STRING
    "Maximum wall seconds for any unit test under ctest (hard kill)")

# Fast-gate discipline (HARD RULE):
#   • tests/ = one component / one method / ≤ ATV_TEST_TIMEOUT_SECONDS.
#   • FORBIDDEN here: multi-step Driver loops, equilibration, multi-rank
#     production MD, full post-HF/CPMD/BOMD campaigns, oracles.
#   • Those live in regressions/ (ctest label `regression`) or benchmarks/**.

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
    foreach(src IN LISTS ARGN)
        if(src MATCHES "\\.cu$")
            set_property(GLOBAL APPEND PROPERTY ATV_UNIT_CUDA_SOURCES
                         ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        else()
            set_property(GLOBAL APPEND PROPERTY ATV_UNIT_SOURCES
                         ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        endif()
    endforeach()
endfunction()

# Thread pins for every single-process unit test. The `;` must reach
# gtest_discover_tests escaped: an unescaped list splits the PROPERTIES
# key/value pairs and silently drops OPENBLAS_NUM_THREADS, leaving OpenBLAS
# on every core in each concurrently running test.
set(ATV_TEST_ENVIRONMENT
    "OMP_NUM_THREADS=${ATV_TEST_OMP_THREADS}\\;OPENBLAS_NUM_THREADS=${ATV_TEST_OMP_THREADS}")

# A unit test that needs its own link line or compile definitions (stub
# libraries, extra sources) keeps a dedicated executable.
function(atv_add_unit_exe name)
    atv_guard_cuda_sources(${name} ${ARGN})
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE atomiverse GTest::gtest_main)
    set_target_properties(${name} PROPERTIES CXX_STANDARD 20
                          CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    set(labels unit)
    foreach(src IN LISTS ARGN)
        if(src MATCHES "\\.cu$")
            set(labels cuda)
        endif()
    endforeach()
    gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST
        PROPERTIES TIMEOUT ${ATV_TEST_TIMEOUT_SECONDS} LABELS ${labels}
                   ENVIRONMENT "${ATV_TEST_ENVIRONMENT}")
endfunction()

# C++ test: compiled as .cpp, no CUDA required.
function(atv_add_cpp_test name)
    atv_add_test(${name} ${ARGN})
endfunction()

# MPI test executable: one build target, can be launched at multiple rank
# counts via `atv_add_mpi_run`. The test source provides its own `main`
# that wraps MPI init/finalize.
function(atv_add_mpi_exec target_name src)
    set_property(GLOBAL APPEND PROPERTY ATV_UNIT_MPI_SOURCES
                 ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    # The gtest suites of this source select its tests inside atv_tests_mpi.
    file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/${src} lines
         REGEX "^(TEST|TEST_F|TEST_P)\\(")
    set(suites "")
    foreach(line IN LISTS lines)
        string(REGEX REPLACE "^TEST[_A-Z]*\\( *([A-Za-z0-9_]+).*" "\\1.*" suite "${line}")
        list(APPEND suites ${suite})
    endforeach()
    list(REMOVE_DUPLICATES suites)
    list(JOIN suites ":" filter)
    set_property(GLOBAL PROPERTY ATV_MPI_FILTER_${target_name} "${filter}")
endfunction()

# Register one CTest invocation at a given rank count for an MPI executable.
# Tagged with the "mpi" CTest label so CI can skip them on MPI-less environments.
function(atv_add_mpi_run test_name target_name n)
    get_property(filter GLOBAL PROPERTY ATV_MPI_FILTER_${target_name})
    add_test(
        NAME ${test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:atv_tests_mpi> ${MPIEXEC_POSTFLAGS}
                --gtest_filter=${filter}
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
atv_add_cpp_test(test_unit_system core/test_unit_system.cpp)
atv_add_cpp_test(test_thermo_style core/test_thermo_style.cpp)
atv_add_cpp_test(test_gto core/test_gto.cpp)
atv_add_cpp_test(test_driver_contract core/test_driver_contract.cpp)
atv_add_cpp_test(test_evaluation_history core/test_evaluation_history.cpp)
atv_add_cpp_test(test_plan_lifecycle core/test_plan_lifecycle.cpp)
atv_add_cpp_test(test_lj_type_table core/test_lj_type_table.cpp)
atv_add_cpp_test(test_pppm_stencil core/test_pppm_stencil.cpp)
atv_add_cpp_test(test_pppm_accuracy core/test_pppm_accuracy.cpp)
atv_add_cpp_test(test_cpu_electronic_aimd_method
                 cpu/electronic/test_aimd_method.cpp)

# ── CPU backend tests (always available, no CUDA required) ──────────

# Mirrors include/atomiverse/cpu/frame/box.h (+ box_simd.h).
atv_add_cpp_test(test_cpu_box cpu/frame/test_box.cpp)
# Mirrors include/atomiverse/cpu/frame/pbc.h (owned-row wrap + image flags).
atv_add_cpp_test(test_cpu_pbc cpu/frame/test_pbc.cpp)
# Mirrors include/atomiverse/cpu/device/chunked_range.h — the fixed, thread-count
# independent partition every reproducible OpenMP reduction is built on.
atv_add_cpp_test(test_cpu_chunked_range cpu/device/test_chunked_range.cpp)
# Context service slices (arch-refactor-05-context): flat core fields + nested
# md/mc/opt/analysis pure-aggregate slices, exact field mapping, all-null
# defaults, trivially-copyable / non-polymorphic guards. Layout-only, no
# physics.
atv_add_cpp_test(test_cpu_component_context_slices cpu/component/test_context_slices.cpp)
atv_add_cpp_test(test_cpu_component_pack_forward_comm cpu/component/test_pack_forward_comm.cpp)
if(ATV_BUILD_VIB) # instantiates vib::HessianAccumulator
    atv_add_cpp_test(test_cpu_component_provides_probes cpu/component/test_provides_probes.cpp)
endif()
if(ATV_BUILD_HF)
    atv_add_cpp_test(test_cpu_qm_engine cpu/component/test_qm_engine.cpp)
    # Bit-for-bit erased-vs-direct RHF comparison: OMP reduction order must be
    # identical in both runs, so pin the thread count (parallel ctest otherwise
    # oversubscribes and the comparison flakes).
endif()
# esmm-contract-05-accumulate-switch (U8 + U7b): the kEnergyElectronic per-term
# key — ZeroEnergy must clear it in lockstep with kEnergy, and a co-bound
# Evaluate pass must leave the QM share alone in it. Registered UNCONDITIONALLY
# (ZeroEnergy is always-on infrastructure); the two co-binding cases inside are
# #ifdef ATV_HAS_HF. Serial: the RHF pin must not drift with reduction order.
if(ATV_BUILD_HF) # drives the electronic-energy zeroing through scf methods
    atv_add_cpp_test(test_cpu_zero_energy_electronic cpu/component/test_zero_energy_electronic.cpp)
endif()
if(ATV_BUILD_MC) # montecarlo sources are excluded when MC is off
    atv_add_cpp_test(test_cpu_mc_scaffold cpu/test_mc_scaffold.cpp)
    atv_add_cpp_test(test_cpu_mc_move cpu/montecarlo/test_mc_move.cpp)
    atv_add_cpp_test(test_cpu_mc_transaction_rollback cpu/montecarlo/test_transaction_rollback.cpp)
    atv_add_cpp_test(test_cpu_mc_ensemble_policies cpu/montecarlo/test_ensemble_policies.cpp)
    atv_add_cpp_test(test_cpu_mc_gcmc_npt_movers cpu/montecarlo/test_gcmc_npt_movers.cpp)
    atv_add_cpp_test(test_cpu_mc_row_order cpu/montecarlo/test_mc_row_order.cpp)
endif()
# The MC physics-validation gates (pair_distribution / acceptance_ratio /
# lj_nist) drive MC::run over many sweeps — e2e, not unit; they are not
# registered here.
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_md_pe_thermo_schema cpu/dynamics/test_pe_thermo_schema.cpp)
endif()
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_md_pe_single_chain cpu/dynamics/test_pe_single_chain.cpp)
endif()
# Multi-step PE bonded NVE conservation is a regression
# (regressions/md/nve), not a unit test.
# Full thermostat equilibrations (CSVR / NVT) are e2e, not ms unit tests.
if(ATV_BUILD_NEIGHBOR)
    atv_add_cpp_test(test_cpu_md_atom_sort            cpu/dynamics/test_atom_sort.cpp)
endif()
atv_add_cpp_test(test_cpu_md_timing_report        cpu/dynamics/test_timing_report.cpp)
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_md_pe_fd_force          cpu/dynamics/test_pe_fd_force.cpp)
endif()
# Multi-chain multi-step NVE was e2e; multi-chain unit surface is
# special-bonds / net-force one-launch only in pe_multi_chain.
if(ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
    atv_add_cpp_test(test_cpu_md_pe_multi_chain       cpu/dynamics/test_pe_multi_chain.cpp)
    atv_add_cpp_test(test_cpu_md_lj_exclusions        cpu/dynamics/test_lj_exclusions.cpp)
endif()
if(ATV_BUILD_MD) # VelocityVerlet / QEq / ConstPot* / FixRattle are MD-domain sources
    atv_add_cpp_test(test_cpu_md_pair_lj_cut          cpu/dynamics/test_pair_lj_cut.cpp)
    atv_add_cpp_test(test_cpu_md_const_pot_bo         cpu/dynamics/test_const_pot_bo.cpp)
    atv_add_cpp_test(test_cpu_md_const_charge_electrode cpu/dynamics/test_const_charge_electrode.cpp)
    atv_add_cpp_test(test_cpu_md_qeq                  cpu/dynamics/test_qeq.cpp)
    atv_add_cpp_test(test_cpu_md_const_pot_cg         cpu/dynamics/test_const_pot_cg.cpp)
    atv_add_cpp_test(test_cpu_md_const_pot_per_eta    cpu/dynamics/test_const_pot_per_eta.cpp)
    atv_add_cpp_test(test_cpu_md_charge_updater_probe cpu/dynamics/test_charge_updater_probe.cpp)
    atv_add_cpp_test(test_cpu_md_fix_rattle_packer    cpu/dynamics/test_fix_rattle_packer.cpp)
    atv_add_cpp_test(test_cpu_md_fix_rattle           cpu/dynamics/test_fix_rattle.cpp)
endif()
atv_add_cpp_test(test_cpu_md_charge_packer        cpu/dynamics/test_charge_packer.cpp)
atv_add_cpp_test(test_cpu_comm_tag_index_lifetime cpu/communication/test_tag_index_lifetime.cpp)
atv_add_cpp_test(test_cpu_comm_halo               cpu/communication/test_halo.cpp)
atv_add_cpp_test(test_cpu_comm_halo_maxneed       cpu/communication/test_halo_maxneed.cpp)
atv_add_cpp_test(test_cpu_comm_halo_queries       cpu/communication/test_halo_queries.cpp)
atv_add_cpp_test(test_cpu_comm_halo_rounds        cpu/communication/test_halo_rounds.cpp)
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_ghost_contract           cpu/frame/test_ghost_contract.cpp)
endif()
atv_add_cpp_test(test_cpu_comm_halo_uniform_rounds cpu/communication/test_halo_uniform_rounds.cpp)
atv_add_cpp_test(test_cpu_comm                    cpu/communication/test_comm.cpp)
atv_add_cpp_test(test_cpu_comm_brick              cpu/communication/test_comm_brick.cpp)
atv_add_cpp_test(test_cpu_comm_atom_pack_comm_plan cpu/communication/test_atom_pack_comm_plan.cpp)
atv_add_cpp_test(test_cpu_comm_atom_pack_reverse  cpu/communication/test_atom_pack_reverse.cpp)
atv_add_cpp_test(test_cpu_comm_atom_pack_generic  cpu/communication/test_atom_pack_generic_wire.cpp)
atv_add_cpp_test(test_cpu_comm_packer_bytes       cpu/communication/test_packer_bytes.cpp)
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_md_const_pot_ffield     cpu/dynamics/test_const_pot_ffield.cpp)
    atv_add_cpp_test(test_cpu_md_const_pot_maze_symplectic cpu/dynamics/test_const_pot_maze_symplectic.cpp)
    atv_add_cpp_test(test_cpu_md_const_pot_site_volume cpu/dynamics/test_const_pot_site_volume.cpp)
endif()
atv_add_cpp_test(test_cpu_voronoi_volume          cpu/observe/test_voronoi_volume.cpp)
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_md_dipole_electrode_joint cpu/dynamics/test_dipole_electrode_joint.cpp)
endif()
atv_add_cpp_test(test_cpu_classical_polarization  cpu/observe/test_classical_polarization.cpp)
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_pair_coul_cut           cpu/potentials/test_pair_coul_cut.cpp)
endif()
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_charging_work           cpu/electrodes/test_charging_work.cpp)
    atv_add_cpp_test(test_cpu_electrode_charge_aspc   cpu/electrodes/test_charge_aspc.cpp)
endif()
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_pair_lj_cut_tip4p_cut   cpu/potentials/test_pair_lj_cut_tip4p_cut.cpp)
    atv_add_cpp_test(test_cpu_pair_lj_soft            cpu/potentials/test_pair_lj_soft.cpp)
    atv_add_cpp_test(test_cpu_pair_coul_soft          cpu/potentials/test_pair_coul_soft.cpp)
    atv_add_cpp_test(test_cpu_lambda_ti               cpu/potentials/test_lambda_ti.cpp)
    atv_add_cpp_test(test_cpu_pair_special_scale      cpu/potentials/test_pair_special_scale.cpp)
    atv_add_cpp_test(test_cpu_pair_partition          cpu/potentials/test_pair_partition.cpp)
    atv_add_cpp_test(test_cpu_coul_long_special       cpu/potentials/test_coul_long_special.cpp)
    atv_add_cpp_test(test_cpu_pair_lj_cut_coul_long   cpu/potentials/test_pair_lj_cut_coul_long.cpp)
    atv_add_cpp_test(test_cpu_dihedral_charmm         cpu/potentials/test_dihedral_charmm.cpp)
    atv_add_cpp_test(test_cpu_improper_harmonic       cpu/potentials/test_improper_harmonic.cpp)
    atv_add_cpp_test(test_cpu_dihedral_fourier        cpu/potentials/test_dihedral_fourier.cpp)
    atv_add_cpp_test(test_cpu_improper_cvff           cpu/potentials/test_improper_cvff.cpp)
endif()
atv_add_cpp_test(test_cpu_lammps_data_full        cpu/io/test_lammps_data_full.cpp)
atv_add_cpp_test(test_cpu_lammps_real             cpu/io/test_lammps_real.cpp)
atv_add_cpp_test(test_cpu_lammps_coeffs           cpu/io/test_lammps_coeffs.cpp)
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_fix_efield              cpu/dynamics/test_fix_efield.cpp)
    atv_add_cpp_test(test_cpu_fix_dfield              cpu/dynamics/test_fix_dfield.cpp)
    atv_add_cpp_test(test_cpu_fix_addforce            cpu/dynamics/test_fix_addforce.cpp)
    atv_add_cpp_test(test_cpu_fix_oscillating_efield  cpu/dynamics/test_fix_oscillating_efield.cpp)
    atv_add_cpp_test(test_cpu_fix_wall_lj1043         cpu/dynamics/test_fix_wall_lj1043.cpp)
endif()
if(ATV_BUILD_PATH) # bias sources ride the PATH domain flag
    atv_add_cpp_test(test_cpu_bias_distance_cv    cpu/bias/test_distance_cv.cpp)
    atv_add_cpp_test(test_cpu_bias_angle_cv       cpu/bias/test_angle_cv.cpp)
    atv_add_cpp_test(test_cpu_bias_dihedral_cv    cpu/bias/test_dihedral_cv.cpp)
    atv_add_cpp_test(test_cpu_bias_metadynamics_1d cpu/bias/test_metadynamics_1d.cpp)
    atv_add_cpp_test(test_cpu_bias_metadynamics_nd cpu/bias/test_metadynamics_nd.cpp)
endif()
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_pair_yukawa             cpu/potentials/test_pair_yukawa.cpp)
    atv_add_cpp_test(test_cpu_gaussian_charge         cpu/potentials/test_gaussian_charge.cpp)
    atv_add_cpp_test(test_cpu_pair_buck               cpu/potentials/test_pair_buck.cpp)
    atv_add_cpp_test(test_cpu_pair_born               cpu/potentials/test_pair_born.cpp)
    atv_add_cpp_test(test_cpu_pair_virial             cpu/potentials/test_pair_virial.cpp)
    atv_add_cpp_test(test_cpu_pim_induced_dipole      cpu/potentials/test_pim_induced_dipole.cpp)
    atv_add_cpp_test(test_cpu_pim_xl_dipole           cpu/potentials/test_pim_xl_dipole.cpp)
endif()
if(ATV_BUILD_MINIMIZER)
    atv_add_cpp_test(test_cpu_optimizer_metrics cpu/minimize/test_optimizer_metrics.cpp)
    atv_add_cpp_test(test_cpu_minimize_aniso_cell cpu/minimize/test_anisotropic_cell.cpp)
    atv_add_cpp_test(test_cpu_lbfgs_history cpu/minimize/test_lbfgs_history.cpp)
endif()
if(ATV_BUILD_SEMIEMPIRICAL)
    atv_add_cpp_test(test_cpu_gfn2_params         cpu/semiempirical/test_gfn2_params.cpp)
endif()
# QM/MM units. Mechanical stays under SE; electrostatic engine-slot path needs HF.
if(ATV_BUILD_SEMIEMPIRICAL)
    atv_add_cpp_test(test_cpu_qm_lifecycle        cpu/multiscale/test_qm_lifecycle.cpp)
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
if(ATV_BUILD_MINIMIZER)
    atv_add_cpp_test(test_cpu_minimize_triclinic cpu/minimize/test_triclinic_cell.cpp)
endif()
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_bond_fene               cpu/potentials/test_bond_fene.cpp)
    atv_add_cpp_test(test_cpu_angle_cosine            cpu/potentials/test_angle_cosine.cpp)
    atv_add_cpp_test(test_cpu_pair_wca                cpu/potentials/test_pair_wca.cpp)
    atv_add_cpp_test(test_cpu_pair_dpd                cpu/potentials/test_pair_dpd.cpp)
endif()
if(ATV_BUILD_PATH)
    atv_add_cpp_test(test_cpu_path_hremd          cpu/path/test_hremd.cpp)
endif()
atv_add_cpp_test(test_cpu_ewald_sum               cpu/longrange/test_ewald_sum.cpp)
atv_add_cpp_test(test_cpu_pme_math                cpu/longrange/test_pme_math.cpp)
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_ewald_coul              cpu/longrange/test_ewald_coul.cpp)
    atv_add_cpp_test(test_cpu_slab_ewald              cpu/longrange/test_slab_ewald.cpp)
endif()
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
if(ATV_BUILD_SEMIEMPIRICAL)
    atv_add_cpp_test(test_cpu_nddo_calpar_derived cpu/semiempirical/test_nddo_calpar_derived.cpp)
    atv_add_cpp_test(test_cpu_rm1                 cpu/semiempirical/test_rm1.cpp)
    atv_add_cpp_test(test_cpu_nddo_derivative cpu/semiempirical/test_nddo_derivative.cpp)
    atv_add_cpp_test(test_cpu_nddo_driver_node    cpu/semiempirical/test_nddo_driver_node.cpp)
    if(ATV_BUILD_MINIMIZER)
        atv_add_cpp_test(test_cpu_nddo_trial_reuse
                         cpu/semiempirical/test_nddo_trial_reuse.cpp)
    endif()
    atv_add_cpp_test(test_cpu_nddo_solver         cpu/semiempirical/test_nddo_solver.cpp)
    atv_add_cpp_test(test_cpu_nddo_geometry       cpu/semiempirical/test_nddo_geometry.cpp)
    atv_add_cpp_test(test_cpu_dftb2_sk            cpu/semiempirical/test_dftb2_sk.cpp)
    atv_add_cpp_test(test_cpu_dftb_gamma_lattice  cpu/semiempirical/test_dftb_gamma_lattice.cpp)
    atv_add_cpp_test(test_cpu_dftb_published_constants cpu/semiempirical/test_dftb_published_constants.cpp)
    atv_add_cpp_test(test_cpu_scc_mixer           cpu/semiempirical/test_scc_mixer.cpp)
    atv_add_cpp_test(test_cpu_scc_driver          cpu/semiempirical/test_scc_driver.cpp)
    atv_add_cpp_test(test_cpu_am1_bcc_charges     cpu/semiempirical/test_am1_bcc_charges.cpp)
endif()
if(ATV_BUILD_SEMIEMPIRICAL)
    atv_add_cpp_test(test_cpu_xtb_gfn0            cpu/semiempirical/test_xtb_gfn0.cpp)
    atv_add_cpp_test(test_cpu_xtb_gfn1            cpu/semiempirical/test_xtb_gfn1.cpp)
    atv_add_cpp_test(test_cpu_xtb_ipea            cpu/semiempirical/test_xtb_ipea.cpp)
    if(ATV_BUILD_VIB)
        atv_add_cpp_test(test_cpu_xtb_gfn0_hessian
            cpu/semiempirical/test_xtb_gfn0_hessian.cpp)
        atv_add_cpp_test(test_cpu_xtb_gfn_hessian
            cpu/semiempirical/test_xtb_gfn_hessian.cpp)
        atv_add_cpp_test(test_cpu_xtb_ipea_hessian
            cpu/semiempirical/test_xtb_ipea_hessian.cpp)
    endif()
endif()

# GPW module tests. All of them run unconditionally on the in-tree GTH + MOLOPT +
# D3 tables (cpu/gpw/basis_data/) — no external data directory is consulted at
# build or test time. (Until 2026-07-11 the 14 tests that touched a real molecule
# only registered when ATV_CP2K_DATA_DIR pointed at a CP2K install, so with the
# gate off GPW had no real-molecule physics and no force validation at all.)
if(ATV_BUILD_GPW)
    # Every GPW test below runs on the in-tree data tables
    # (cpu/gpw/basis_data/gpw_data.h) — no external data directory, no
    # third-party file read. E2E (converged SCF / AIMD / geoopt / anchors) lives
    # outside the unit gate and never runs under ctest.
    atv_add_cpp_test(test_cpu_gpw_gth_integrals    cpu/gpw/test_cpu_gpw_gth_integrals.cpp)
    atv_add_cpp_test(test_cpu_gpw_collocate        cpu/gpw/test_cpu_gpw_collocate.cpp)
    atv_add_cpp_test(test_cpu_gpw_gamma_k_identity cpu/gpw/test_cpu_gpw_gamma_k_identity.cpp)
    atv_add_unit_exe(test_cpu_gpw_multigrid        cpu/gpw/test_cpu_gpw_multigrid.cpp)
    if(ATV_BUILD_PW)  # FFT transfer (prolong_fft / restrict_fft) needs FFTW3
        target_link_libraries(test_cpu_gpw_multigrid PRIVATE fftw3_stub)
        target_compile_definitions(test_cpu_gpw_multigrid PRIVATE ATV_GPW_HAS_FFT)
    endif()
    atv_add_cpp_test(test_cpu_gpw_xc_grid          cpu/gpw/test_cpu_gpw_xc_grid.cpp)
    atv_add_cpp_test(test_cpu_gpw_one_electron     cpu/gpw/test_cpu_gpw_one_electron.cpp)
    atv_add_cpp_test(test_cpu_gpw_workspace        cpu/gpw/test_cpu_gpw_workspace.cpp)
    atv_add_cpp_test(test_cpu_gpw_driver_node     cpu/gpw/test_cpu_gpw_driver_node.cpp)
    atv_add_cpp_test(test_cpu_gpw_diis            cpu/gpw/test_diis.cpp)
    atv_add_cpp_test(test_cpu_gpw_density_stack   cpu/gpw/test_kpoints_density_stack.cpp)
    atv_add_cpp_test(test_cpu_gpw_fourier_overlap cpu/gpw/test_kpoints_fourier_overlap.cpp)
    atv_add_cpp_test(test_cpu_gpw_grid_contractions cpu/gpw/test_kpoints_grid_contractions.cpp)
    atv_add_cpp_test(test_cpu_gpw_admission       cpu/gpw/test_solver_admission.cpp)
    atv_add_cpp_test(test_cpu_gpw_core_overlap    cpu/gpw/test_core_overlap.cpp)
    atv_add_cpp_test(test_cpu_gpw_kpoints_natural_orbitals
                     cpu/gpw/test_kpoints_natural_orbitals.cpp)
    if(ATV_BUILD_PW)
        atv_add_cpp_test(test_cpu_gpw_ks_grid_build cpu/gpw/test_cpu_gpw_ks_grid_build.cpp)
    endif()
    atv_add_cpp_test(test_cpu_xc_pade              cpu/xc/test_cpu_xc_pade.cpp)
    if(ATV_BUILD_PW)  # FFT-backed pieces
        atv_add_cpp_test(test_cpu_gpw_nonlocal_vdw cpu/gpw/test_nonlocal_vdw.cpp)
        atv_add_cpp_test(test_cpu_gpw_poisson_fft  cpu/gpw/test_cpu_gpw_poisson_fft.cpp)
        atv_add_cpp_test(test_cpu_gpw_hartree_grid cpu/gpw/test_cpu_gpw_hartree_grid.cpp)
        atv_add_cpp_test(test_cpu_gpw_grid_fft     cpu/gpw/test_cpu_gpw_grid_fft.cpp)
        atv_add_cpp_test(test_cpu_gpw_xc_grid_gga  cpu/gpw/test_cpu_gpw_xc_grid_gga.cpp)
        atv_add_cpp_test(test_cpu_gpw_mt_poisson   cpu/gpw/test_cpu_gpw_mt_poisson.cpp)
        # Multi-k spectral G0W0 kernels: Bloch-MO gauge invariance (no SCF).
        atv_add_cpp_test(test_cpu_gpw_g0w0_kpoints cpu/gpw/test_cpu_gpw_g0w0_kpoints.cpp)
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
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_md_pair_morse_smoke     cpu/dynamics/test_pair_morse_smoke.cpp)
endif()
atv_add_cpp_test(test_cpu_md_pe_ff_constants      cpu/dynamics/test_pe_ff_constants.cpp)
if(ATV_BUILD_NEIGHBOR)
    atv_add_cpp_test(test_cpu_md_neighbor_list_arena  cpu/dynamics/test_neighbor_list_arena.cpp)
    atv_add_cpp_test(test_cpu_md_neighbor_decide      cpu/dynamics/test_neighbor_decide.cpp)
endif()
atv_add_cpp_test(test_cpu_driver_verlet_comm      cpu/driver/test_verlet_comm.cpp)
if(ATV_BUILD_MD AND ATV_BUILD_EVALUATE AND ATV_BUILD_MINIMIZER AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
    atv_add_cpp_test(test_cpu_driver_preset_comm_plan cpu/driver/test_preset_comm_plan.cpp)
endif()
if(ATV_BUILD_NEIGHBOR)
    atv_add_cpp_test(test_cpu_md_full_neighbor_list  cpu/dynamics/test_full_neighbor_list.cpp)
    atv_add_cpp_test(test_cpu_md_cell_list_grid_stale  cpu/dynamics/test_cell_list_grid_stale.cpp)
    atv_add_cpp_test(test_cpu_md_cell_list_halo_pad  cpu/dynamics/test_cell_list_halo_pad.cpp)
endif()
atv_add_cpp_test(test_cpu_atom_variable_n  cpu/test_atom_variable_n.cpp)
atv_add_cpp_test(test_cpu_frame_registry  cpu/frame/test_frame_registry.cpp)
# EAM/SW table lookups key on molrs::Element; sources excluded when MOLPY off.
if(ATV_BUILD_MOLPY AND ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_eam  cpu/potentials/test_eam.cpp)
    atv_add_cpp_test(test_cpu_sw  cpu/potentials/test_sw.cpp)
endif()
if(ATV_BUILD_VIB)
    atv_add_cpp_test(test_cpu_vib_phonon_dispersion cpu/vibration/test_phonon_dispersion.cpp)
    # esmm-contract-01-zeroing: the Hessian driver must zero kAtom_f* / kEnergy
    # per displacement pass, or an accumulating source halves the curvature.
    atv_add_cpp_test(test_cpu_vib_hessian_zeroing
        cpu/vibration/test_hessian_zeroing.cpp)
    # GPW force-consumer wiring for Phonon (no NeighborList). Full FD spectrum
    # lives in regressions/response/phonons (6N SCF — not unit budget).
    if(ATV_BUILD_GPW)
        atv_add_cpp_test(test_cpu_vib_phonon_gpw_wiring
            cpu/vibration/test_phonon_gpw_wiring.cpp)
    endif()
endif()
if(ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_pair_lj_types  cpu/potentials/test_pair_lj_types.cpp)
    if(ATV_BUILD_EVALUATE)
        atv_add_cpp_test(test_cpu_pair_lj_typed_cutoff
            cpu/potentials/test_pair_lj_typed_cutoff.cpp)
    endif()
endif()
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_md_fix_freeze  cpu/dynamics/test_fix_freeze.cpp)
    atv_add_cpp_test(test_cpu_md_piston_electrode cpu/dynamics/test_piston_electrode.cpp)
    atv_add_cpp_test(test_cpu_md_const_potential_bo  cpu/dynamics/test_const_potential_bo.cpp)
    atv_add_cpp_test(test_cpu_md_fix_settle  cpu/dynamics/test_fix_settle.cpp)
    atv_add_cpp_test(test_cpu_md_virtual_sites  cpu/dynamics/test_virtual_sites.cpp)
    atv_add_cpp_test(test_cpu_md_berendsen    cpu/dynamics/test_berendsen.cpp)
    atv_add_cpp_test(test_cpu_md_sllod_flow   cpu/dynamics/test_sllod_flow.cpp)
    atv_add_cpp_test(test_cpu_md_nvt_sllod    cpu/dynamics/test_nvt_sllod.cpp)
    if(ATV_BUILD_EVALUATE AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
        atv_add_cpp_test(test_cpu_md_nvt_sllod_flip_lj
            cpu/dynamics/test_nvt_sllod_flip_lj.cpp)
    endif()
    atv_add_cpp_test(test_cpu_md_npt_scalars cpu/dynamics/test_npt_scalars.cpp)
    atv_add_cpp_test(test_cpu_md_brownian     cpu/dynamics/test_brownian.cpp)
    atv_add_cpp_test(test_cuda_md_brownian cuda/dynamics/test_brownian.cpp)
    atv_add_cpp_test(test_cpu_md_write_sets  cpu/dynamics/test_write_sets.cpp)
endif()
if(ATV_BUILD_PATH)
    atv_add_cpp_test(test_cpu_path_replica    cpu/path/test_replica.cpp)
    atv_add_cpp_test(test_cpu_path_remd       cpu/path/test_remd.cpp)
    atv_add_cpp_test(test_cpu_path_string     cpu/path/test_string_method.cpp)
    atv_add_cpp_test(test_cpu_path_irc        cpu/path/test_irc.cpp)
    atv_add_cpp_test(test_cpu_path_rpmd_est   cpu/path/test_rpmd_estimator.cpp)
    # esmm-contract-01-zeroing: PathDriver must zero kAtom_f* / kEnergy before
    # each per-replica force pass, or an accumulating source grows with the
    # iteration.
    atv_add_cpp_test(test_cpu_path_zeroing    cpu/path/test_path_zeroing.cpp)
    # esmm-contract-02-validation-walks: PathDriver::validate_components must
    # see a nested `Driver<scf::Phase> + HF` method node as the band's force
    # source.
    atv_add_cpp_test(test_cpu_path_validation cpu/path/test_path_validation.cpp)
endif()
# esmm-contract-03-runslot-nested: Minimize's StrongWolfe trials must re-enter a
# nested `Driver<scf::Phase> + HF` force source (run_stage → Runtime::run_slot_).
# Stub physics (harmonic well) — no SCF, milliseconds.
if(ATV_BUILD_MINIMIZER)
    atv_add_cpp_test(test_cpu_minimizer_line_search_nested_node
                     cpu/minimize/test_line_search_nested_node.cpp)
    # Trial-force reuse: the accepted trial's forces stand in for the
    # committed force phase when every force op is trial-invariant.
    atv_add_cpp_test(test_cpu_minimizer_trial_force_reuse
                     cpu/minimize/test_trial_force_reuse.cpp)
endif()
atv_add_cpp_test(test_cpu_criterion       cpu/driver/test_criterion.cpp)
if(ATV_BUILD_EVALUATE)
    atv_add_cpp_test(test_cpu_driver_nesting  cpu/driver/test_driver_nesting.cpp)
endif()
# scf-driver-refactor-01-plumb: scf::Phase / PhaseKind::SCF / PhaseTraits<scf::Phase>
# / the Criterion component kind + driver pull-poll. Pure type-foundation plumbing —
# no consumer yet.
atv_add_cpp_test(test_cpu_scf_phase_plumbing cpu/driver/test_scf_phase_plumbing.cpp)
atv_add_cpp_test(test_cpu_component_ownership cpu/driver/test_component_ownership.cpp)
if(ATV_BUILD_MD AND ATV_BUILD_MINIMIZER AND ATV_BUILD_EVALUATE)
    atv_add_cpp_test(test_cpu_require_match cpu/driver/test_require_match.cpp)
endif()
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_borrowed_lifetime cpu/driver/test_borrowed_lifetime.cpp)
    atv_add_cpp_test(test_cpu_input_updates cpu/driver/test_input_updates.cpp)
endif()
atv_add_cpp_test(test_driver_live_table cpu/driver/test_driver_live_table.cpp)
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_driver_unfix cpu/driver/test_unfix.cpp)
endif()
# Mirrors include/atomiverse/cpu/driver/outcome_policy.h — cpu::Metropolis'
# verb contract (the canonical policy owns Displace/None and refuses the
# Box/Insert/Delete trials that need the NptMc / Gcmc presets). Formula-level,
# hand-built LoopState, microseconds.
if(ATV_BUILD_MC) # exercises mc::Transaction verbs
    atv_add_cpp_test(test_cpu_outcome_policy cpu/driver/test_outcome_policy.cpp)
endif()
# esmm-contract-04-forces-requested: `forces_requested` as a compiled Plan-slot
# property — MC's energy phase is energy-only, MD/Minimize/Evaluate still ask for
# forces, and the veto ANDs through a nested driver's own slots. Stubs only, no
# physics, microseconds; the QM-in-MC e2e is regressions/mc/mc_bias_es.
if(ATV_BUILD_MC) # instantiates the cpu::mc::MC preset
    atv_add_cpp_test(test_cpu_forces_requested cpu/driver/test_forces_requested.cpp)
endif()
# A full NVT equilibration of the bonded PE melt is e2e, not a ms unit test.
atv_add_cpp_test(test_cpu_md_frame_tag_image      cpu/dynamics/test_frame_tag_image.cpp)
atv_add_cpp_test(test_cpu_md_domain_decomposition cpu/dynamics/test_domain_decomposition.cpp)

# The shared lossless capture buffer of asynchronous consumers (LJ MD).
if(ATV_BUILD_MD AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
    atv_add_cpp_test(test_cpu_capture_buffer      cpu/asynchronous/test_capture_buffer.cpp)
endif()
# MSD/VACF stream into molpy accumulators (cpu/observe/trajectory_metrics.h).
if(ATV_BUILD_ANALYSIS)
    atv_add_cpp_test(test_cpu_observer_msd        cpu/observe/test_msd.cpp)
    atv_add_cpp_test(test_cpu_observer_vacf       cpu/observe/test_vacf.cpp)
    if(ATV_BUILD_MD AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
        atv_add_cpp_test(test_cpu_async_observers_lossless
                         cpu/observe/test_async_observers_lossless.cpp)
    endif()
endif()
atv_add_cpp_test(test_cpu_observer_widom          cpu/observe/test_widom.cpp)
atv_add_cpp_test(test_cpu_observer_electrode_charge cpu/observe/test_electrode_charge_observer.cpp)
atv_add_cpp_test(test_cpu_observer_capacitance_charge_fluctuation
                 cpu/observe/test_capacitance_charge_fluctuation.cpp)
atv_add_cpp_test(test_cpu_observer_potential_profile
                 cpu/observe/test_potential_profile.cpp)
atv_add_cpp_test(test_cpu_observer_species_density_profile
                 cpu/observe/test_species_density_profile.cpp)

if(ATV_BUILD_MD AND ATV_BUILD_EVALUATE AND ATV_BUILD_MINIMIZER)
    atv_add_cpp_test(test_cpu_driver_role_validation cpu/test_cpu_driver_role_validation.cpp)
endif()
# Registration-time PhaseKind gate + five-driver plan-shape regression
# (arch-refactor-01-bindings). Mounts AM1 (unconditional, no RHF gate).
# Walks default bindings across MC / PATH / SE / VIB component kinds.
if(ATV_BUILD_MC AND ATV_BUILD_PATH AND ATV_BUILD_SEMIEMPIRICAL AND ATV_BUILD_VIB)
    atv_add_cpp_test(test_cpu_driver_binding_kind         cpu/test_driver_binding_kind.cpp)
endif()
if(ATV_BUILD_MD AND ATV_BUILD_EVALUATE)
    atv_add_cpp_test(test_cpu_evaluate_driver_capabilities    cpu/test_evaluate_driver_capabilities.cpp)
endif()
if(ATV_BUILD_MINIMIZER)
    atv_add_cpp_test(test_cpu_optimize_driver_early_stop      cpu/test_optimize_driver_early_stop.cpp)
endif()
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_dynamics_full_budget            cpu/test_dynamics_full_budget.cpp)
    atv_add_cpp_test(test_cpu_install_infra_hook              cpu/test_install_infra_hook.cpp)
endif()
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_md_set_api                      cpu/dynamics/test_md_set_api.cpp)
    atv_add_cpp_test(test_cpu_md_plan_equivalence             cpu/dynamics/test_md_plan_equivalence.cpp)
endif()
# `MD::run(molrs::FrameRef&, n)` independence contract: consecutive calls are
# separate simulations (step index restarts at 0), NOT a continued trajectory —
# the cpu::Frame& overload is the continuable one. 2 x 3 steps of 4-atom LJ, ms.
if(ATV_BUILD_IO AND ATV_BUILD_MD AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
    atv_add_cpp_test(test_cpu_md_molpy_run                cpu/dynamics/test_md_molpy_run.cpp)
endif()
# TimestepProvider/TimestepConsumer opt-in protocol + the MD build-time push
# (esmm-cpmd-02-timestep-protocol): dt is single-sourced from the integrator,
# conflicting/absent/invalid publications throw at build. Local stubs only —
# no GPW dependency, one no-op MD step to fire the component build() pass.
if(ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_md_timestep_protocol            cpu/dynamics/test_timestep_protocol.cpp)
endif()
# Car-Parrinello conserved-energy assembler (esmm-cpmd-03-observability):
# kConserved = kEnergy + kKinetic + the two electronic keys, end_of_step @ -90,
# throw when the electronic keys are absent. Pure meta arithmetic on a seeded
# one-atom frame — no GPW, no SCF, no MD step.
if(ATV_BUILD_MINIMIZER)
    atv_add_cpp_test(test_cpu_minimize_evaluate_set_api cpu/test_minimize_evaluate_set_api.cpp)
endif()
if(ATV_BUILD_IO) # from_molpy/to_molpy boundary lives in the IO block
    atv_add_cpp_test(test_cpu_materialize                 cpu/test_materialize.cpp)
endif()
if(ATV_BUILD_MOLPY)
    # Drift fence: every molrs:: call pattern ATV production code uses.
    atv_add_cpp_test(test_molpy_element                   test_molpy/test_element.cpp)
    atv_add_cpp_test(test_molpy_frame_columns             test_molpy/test_frame_columns.cpp)
endif()
if(ATV_BUILD_MOLPY AND ATV_BUILD_IO)
    atv_add_cpp_test(test_molpy_frame_exchange            test_molpy/test_frame_exchange.cpp)
    atv_add_cpp_test(test_molpy_xyz_io                    test_molpy/test_xyz_io.cpp)
endif()
if(ATV_BUILD_MOLPY AND ATV_BUILD_ANALYSIS)
    atv_add_cpp_test(test_molpy_observe                   test_molpy/test_observe.cpp)
endif()
if(ATV_BUILD_MOLPY AND ATV_BUILD_SEMIEMPIRICAL)
    atv_add_cpp_test(test_molpy_am1_bcc                   test_molpy/test_am1_bcc.cpp)
endif()
if(ATV_BUILD_MOLPY AND ATV_BUILD_IO)
    atv_add_cpp_test(test_molpy_mrec                      test_molpy/test_mrec.cpp)
endif()

# cpu/task infrastructure cluster (mpi-launcher-embedded-python-02-codec):
# TaskSpec data model + MPI-free byte codec + rank-agnostic materializer.
# Serial, no mpirun; the AM1 single point and the 3-step AM1 BOMD both run on
# the ms-scale H2 fixture this file owns (mirrored by test_mpi_serial_on_replica).
if(ATV_BUILD_MOLPY) # cpu/task block follows MOLPY (launcher-only surface)
    atv_add_cpp_test(test_cpu_task_codec                  cpu/test_task_codec.cpp)
    # Replays MD and Minimize tasks; each replay is compiled only with its domain.
    if(ATV_BUILD_MD AND ATV_BUILD_MINIMIZER)
        atv_add_cpp_test(test_cpu_task_materializer       cpu/test_task_materializer.cpp)
    endif()
endif()

# LauncherGate (mpi-launcher-embedded-python-04, ac-002): the PID-qualified
# ATV_LAUNCHER marker probe. Serial gtest — MPI is never initialized in this
# binary, which is exactly the state the degradation contract is about.
if(ATV_BUILD_MOLPY) # cpu/task block follows MOLPY (launcher-only surface)
    atv_add_cpp_test(test_cpu_launcher_gate               cpu/task/test_launcher_gate.cpp)
endif()

# LauncherArgs (same link, ac-002/ac-006): header-only class in
# app/atomiverse/, which is NOT part of libatomiverse — the test compiles the
# header directly via this include dir.
atv_add_unit_exe(test_app_launcher_args                   app/test_launcher_args.cpp)
target_include_directories(test_app_launcher_args PRIVATE
    ${CMAKE_SOURCE_DIR}/app/atomiverse)

# InterpreterLocator (mpi-launcher-embedded-python-05, ac-004/ac-005): pure
# path-resolution + version-string logic — Python.h-free, no popen/exec, no
# interpreter ever executed. The TU lives in app/atomiverse/ (NOT part of
# libatomiverse), so this target compiles it directly as a second source.
atv_add_unit_exe(test_app_interpreter_locator
    app/test_interpreter_locator.cpp
    ${CMAKE_SOURCE_DIR}/app/atomiverse/interpreter_locator.cpp)
target_include_directories(test_app_interpreter_locator PRIVATE
    ${CMAKE_SOURCE_DIR}/app/atomiverse)

# ExtXYZ loading through molpy — inline fixtures only, no external file.
# Gated: cpu/io/extxyz.h reaches molpy_cxx/bridge.h, absent in MOLPY-off builds.
if(ATV_BUILD_IO)
    atv_add_cpp_test(test_cpu_io_extxyz                   cpu/io/test_cpu_io_extxyz.cpp)
endif()

# cpu::XyzRecorder (py-recorder-xyz, ac-003): stride cadence, truncate-then-
# append, frame content. The component itself is compiled only in the
# ATV_BUILD_IO source block and the test drives it through cpu::md::MD, so the
# registration is gated on both domains (iron law 8: OFF excludes the tests
# too) — an IO-less / SE-only build must not try to link cpu::XyzRecorder.
if(ATV_BUILD_IO AND ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_xyz_recorder                cpu/record/test_xyz_recorder.cpp)
endif()

# cpu::Thermo: the LAMMPS thermo step convention (setup line, completed-step
# labels, last step off cadence, continued runs) against independent runs.
if(ATV_BUILD_MD AND ATV_BUILD_NEIGHBOR AND ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_thermo                      cpu/record/test_thermo.cpp)
endif()

# The LAMMPS `minimize` output convention of the minimizer-loop writers
# (TrajectoryWriter, OptLog, Thermo, XyzRecorder, MrecRecorder): initial
# state, completed-iteration labels, forced final state on an early stop;
# plus the single-pass Evaluate (LAMMPS `run 0`) labelling.
if(ATV_BUILD_IO AND ATV_BUILD_MINIMIZER AND ATV_BUILD_EVALUATE
   AND ATV_BUILD_NEIGHBOR AND ATV_BUILD_POTENTIALS)
    atv_add_cpp_test(test_cpu_minimize_output             cpu/minimize/test_minimize_output.cpp)
endif()

# cpu::MrecRecorder — the streaming `*.mrec` sibling: constructor guards,
# stride cadence + commit at teardown, frame read-back through the bridge,
# the LAMMPS step convention against independent runs.
# Same IO+MD gate as the XYZ recorder.
if(ATV_BUILD_IO AND ATV_BUILD_MD)
    atv_add_cpp_test(test_cpu_mrec_recorder               cpu/record/test_mrec_recorder.cpp)
endif()

# Unreadable input paths must throw std::runtime_error, not abort in Rust.
if(ATV_BUILD_IO) # extxyz + XyzRecorder halves reach molpy
    atv_add_cpp_test(test_cpu_io_missing_file             cpu/io/test_cpu_io_missing_file.cpp)
endif()

# ── MPI tests (launched under `mpirun`) ─────────────────────────────────

if(ATV_USE_MPI AND MPIEXEC_EXECUTABLE)
    if(ATV_BUILD_MD)
        atv_add_mpi_exec(test_mpi_init_velocity mpi/test_init_velocity.cpp)
        atv_add_mpi_run(test_mpi_init_velocity_n2 test_mpi_init_velocity 2)
        atv_add_mpi_run(test_mpi_init_velocity_n4 test_mpi_init_velocity 4)
    endif()
    atv_add_mpi_exec(test_mpi_criterion mpi/test_criterion.cpp)
    atv_add_mpi_run(test_mpi_criterion_n2 test_mpi_criterion 2)
    atv_add_mpi_run(test_mpi_criterion_n4 test_mpi_criterion 4)

    atv_add_mpi_exec(test_mpi_input_updates mpi/test_input_updates.cpp)
    atv_add_mpi_run(test_mpi_input_updates_n2 test_mpi_input_updates 2)

    atv_add_mpi_exec(test_mpi_topology mpi/test_topology.cpp)
    atv_add_mpi_run(test_mpi_topology_n1 test_mpi_topology 1)
    atv_add_mpi_run(test_mpi_topology_n2 test_mpi_topology 2)
    atv_add_mpi_run(test_mpi_topology_n4 test_mpi_topology 4)
    atv_add_mpi_run(test_mpi_topology_n8 test_mpi_topology 8)

    atv_add_mpi_exec(test_mpi_halo_dual_image_ghosts mpi/test_halo_dual_image_ghosts.cpp)
    atv_add_mpi_run(test_mpi_halo_dual_image_ghosts_n1 test_mpi_halo_dual_image_ghosts 1)
    atv_add_mpi_run(test_mpi_halo_dual_image_ghosts_n2 test_mpi_halo_dual_image_ghosts 2)
    # Narrow subdomains (ghost shell wider than one slab => multi-hop swaps,
    # spec comm-topology-02): sub = 3.945 A at n4, 1.9725 A at n8.
    atv_add_mpi_run(test_mpi_halo_dual_image_ghosts_n4 test_mpi_halo_dual_image_ghosts 4)
    atv_add_mpi_run(test_mpi_halo_dual_image_ghosts_n8 test_mpi_halo_dual_image_ghosts 8)

    atv_add_mpi_exec(test_mpi_bcast_frame mpi/test_bcast_frame.cpp)
    atv_add_mpi_run(test_mpi_bcast_frame_n1 test_mpi_bcast_frame 1)
    atv_add_mpi_run(test_mpi_bcast_frame_n2 test_mpi_bcast_frame 2)
    atv_add_mpi_run(test_mpi_bcast_frame_n4 test_mpi_bcast_frame 4)

    atv_add_mpi_exec(test_mpi_halo_exchange mpi/test_halo_exchange.cpp)
    atv_add_mpi_run(test_mpi_halo_exchange_n1 test_mpi_halo_exchange 1)
    atv_add_mpi_run(test_mpi_halo_exchange_n2 test_mpi_halo_exchange 2)
    atv_add_mpi_run(test_mpi_halo_exchange_n8 test_mpi_halo_exchange 8)

    atv_add_mpi_exec(test_mpi_comm_brick mpi/test_comm_brick.cpp)
    atv_add_mpi_run(test_mpi_comm_brick_n1 test_mpi_comm_brick 1)
    atv_add_mpi_run(test_mpi_comm_brick_n2 test_mpi_comm_brick 2)

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
        atv_add_mpi_exec(test_mpi_pppm_brick_parity
                         mpi/test_pppm_brick_parity.cpp)
        atv_add_mpi_run(test_mpi_pppm_brick_parity_n1
                        test_mpi_pppm_brick_parity 1)
        atv_add_mpi_run(test_mpi_pppm_brick_parity_n2
                        test_mpi_pppm_brick_parity 2)
        # PPPM mesh-arm selector: Replicated (n1), aliased BrickWorld (n2),
        # and the FFT subgroup (n4: K0=6 → 3 of 4 ranks, one non-member).
        atv_add_mpi_exec(test_mpi_pppm_subcomm_parity
                         mpi/test_pppm_subcomm_parity.cpp)
        atv_add_mpi_run(test_mpi_pppm_subcomm_parity_n1
                        test_mpi_pppm_subcomm_parity 1)
        atv_add_mpi_run(test_mpi_pppm_subcomm_parity_n2
                        test_mpi_pppm_subcomm_parity 2)
        atv_add_mpi_run(test_mpi_pppm_subcomm_parity_n4
                        test_mpi_pppm_subcomm_parity 4)
    endif()

    atv_add_mpi_exec(test_mpi_bonded_energy_bruteforce
                     mpi/test_bonded_energy_bruteforce.cpp)
    atv_add_mpi_run(test_mpi_bonded_energy_bruteforce_n1
                    test_mpi_bonded_energy_bruteforce 1)
    atv_add_mpi_run(test_mpi_bonded_energy_bruteforce_n8
                    test_mpi_bonded_energy_bruteforce 8)

    atv_add_mpi_exec(test_mpi_owner_exchange mpi/test_owner_exchange.cpp)
    atv_add_mpi_run(test_mpi_owner_exchange_n1 test_mpi_owner_exchange 1)
    atv_add_mpi_run(test_mpi_owner_exchange_n2 test_mpi_owner_exchange 2)

    # cpu::Evaluate wraps ctx.md.comm around the evaluate slot (same
    # identity as MD/Minimize). n1 is the serial guardrail, n2 the
    # multi-rank statement.
    if(ATV_BUILD_EVALUATE AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
        atv_add_mpi_exec(test_mpi_evaluate_halo_parity
                         mpi/test_evaluate_halo_parity.cpp)
        atv_add_mpi_run(test_mpi_evaluate_halo_parity_n1
                        test_mpi_evaluate_halo_parity 1)
        atv_add_mpi_run(test_mpi_evaluate_halo_parity_n2
                        test_mpi_evaluate_halo_parity 2)
    endif()

    if(ATV_BUILD_MINIMIZER)
        atv_add_mpi_exec(test_mpi_optimizer_metrics mpi/test_optimizer_metrics.cpp)
        atv_add_mpi_run(test_mpi_optimizer_metrics_n2 test_mpi_optimizer_metrics 2)
        atv_add_mpi_run(test_mpi_optimizer_metrics_n4 test_mpi_optimizer_metrics 4)
    endif()

    if(ATV_BUILD_MOLPY) # EAM/SW table lookups key on molrs::Element
        atv_add_mpi_exec(test_mpi_eam_multirank mpi/test_eam_multirank.cpp)
        atv_add_mpi_run(test_mpi_eam_multirank_n2 test_mpi_eam_multirank 2)
        atv_add_mpi_run(test_mpi_eam_multirank_n4 test_mpi_eam_multirank 4)
    endif()


    if(ATV_BUILD_MOLPY) # EAM/SW table lookups key on molrs::Element
        atv_add_mpi_exec(test_mpi_sw_multirank mpi/test_sw_multirank.cpp)
        atv_add_mpi_run(test_mpi_sw_multirank_n2 test_mpi_sw_multirank 2)
        atv_add_mpi_run(test_mpi_sw_multirank_n4 test_mpi_sw_multirank 4)
    endif()


    atv_add_mpi_exec(test_mpi_gather mpi/test_gather.cpp)
    atv_add_mpi_run(test_mpi_gather_n1 test_mpi_gather 1)
    atv_add_mpi_run(test_mpi_gather_n2 test_mpi_gather 2)
    atv_add_mpi_run(test_mpi_gather_n8 test_mpi_gather 8)

    atv_add_mpi_exec(test_mpi_rebalance mpi/test_rebalance.cpp)
    atv_add_mpi_run(test_mpi_rebalance_n1 test_mpi_rebalance 1)
    atv_add_mpi_run(test_mpi_rebalance_n8 test_mpi_rebalance 8)

    # cpu/task block follows MOLPY; the protocol test runs MD and Minimize tasks.
    if(ATV_BUILD_MOLPY AND ATV_BUILD_MD AND ATV_BUILD_MINIMIZER)
        atv_add_mpi_exec(test_mpi_task_protocol mpi/test_task_protocol.cpp)
        atv_add_mpi_run(test_mpi_task_protocol_n1 test_mpi_task_protocol 1)
        atv_add_mpi_run(test_mpi_task_protocol_n2 test_mpi_task_protocol 2)
        atv_add_mpi_run(test_mpi_task_protocol_n4 test_mpi_task_protocol 4)
    endif()

    # serial-on-replica seam (mpi-launcher-embedded-python-04): a driver with
    # set_serial_on_replica() must NOT decompose — every rank solves the full
    # replica (AM1 H2 single point) and lands on the identical energy.
    if(ATV_BUILD_SEMIEMPIRICAL) # AM1 H2 fixture
        atv_add_mpi_exec(test_mpi_serial_on_replica mpi/test_serial_on_replica.cpp)
        atv_add_mpi_run(test_mpi_serial_on_replica_n1 test_mpi_serial_on_replica 1)
        atv_add_mpi_run(test_mpi_serial_on_replica_n2 test_mpi_serial_on_replica 2)
        atv_add_mpi_run(test_mpi_serial_on_replica_n4 test_mpi_serial_on_replica 4)
    endif()
endif()

# ── PW infrastructure tests (require ATV_BUILD_PW) ─────────────────────

if(ATV_BUILD_PW)
    # Naming convention: test_{backend}_{domain}_{component}.
    atv_add_cpp_test(test_cpu_pw_g_vector   cpu/planewave/test_g_vector.cpp)
    atv_add_cpp_test(test_cpu_pw_smearing   cpu/planewave/test_smearing.cpp)
    atv_add_cpp_test(test_cpu_pw_davidson   cpu/planewave/test_davidson.cpp)
    atv_add_cpp_test(test_cpu_pw_density    cpu/planewave/test_density_builder.cpp)
    atv_add_cpp_test(test_cpu_pw_setup      cpu/planewave/test_pw_setup.cpp)
    atv_add_cpp_test(test_cpu_pw_response_mixer cpu/planewave/test_response_mixer.cpp)
    atv_add_cpp_test(test_cpu_pw_xc_builder cpu/planewave/test_xc_builder.cpp)
    atv_add_cpp_test(test_cpu_pw_upf        cpu/planewave/test_upf.cpp)
    atv_add_cpp_test(test_cpu_pw_nc         cpu/planewave/test_nc_potential.cpp)
    atv_add_cpp_test(test_cpu_pw_gth_radial    cpu/planewave/test_gth_radial.cpp)
    atv_add_cpp_test(test_cpu_pw_kpoint     cpu/planewave/test_kpoint.cpp)
    atv_add_cpp_test(test_cpu_pw_crystal    cpu/planewave/test_crystal.cpp)
    atv_add_cpp_test(test_cpu_pw_ewald      cpu/planewave/test_ewald.cpp)
    atv_add_cpp_test(test_cpu_pw_radial_derivatives cpu/planewave/test_pw_radial_derivatives.cpp)
    atv_add_cpp_test(test_cpu_pw_xc_virial   cpu/planewave/test_xc_virial.cpp)
    atv_add_cpp_test(test_cpu_pw_xc_energy_precision cpu/planewave/test_xc_energy_precision.cpp)
    atv_add_cpp_test(test_cpu_pw_frozen_operator_virial cpu/planewave/test_frozen_operator_virial.cpp)
    atv_add_cpp_test(test_cpu_pw_frozen_functional_virial cpu/planewave/test_frozen_functional_virial.cpp)
    atv_add_cpp_test(test_cpu_pw_final_state cpu/planewave/test_pw_final_state.cpp)
    atv_add_cpp_test(test_cpu_pw_nc_projector_gauge cpu/planewave/test_nc_projector_gauge.cpp)
    atv_add_cpp_test(test_cpu_pw_driver_node cpu/planewave/test_cpu_pw_driver_node.cpp)
    atv_add_cpp_test(test_cpu_pw_dfpt_admission cpu/planewave/test_dfpt_admission.cpp)
    # ctest runs from the build tree, so the in-tree Si UPF must be located
    # absolutely — a relative path silently GTEST_SKIPped the only UPF force
    # coverage in the suite.
    if(TARGET test_cpu_pw_forces)
        target_compile_definitions(test_cpu_pw_forces
            PRIVATE ATV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    endif()

    # FFT test links the FFTW3 stub.
    atv_add_unit_exe(test_cpu_pw_fft cpu/planewave/test_fft.cpp)
    target_link_libraries(test_cpu_pw_fft PRIVATE fftw3_stub)

    # MPI PW/GPW k-point multi-iter SCF: e2e-in-tests wave-2 deleted the unit
    # sources. Industrial gates live in regressions/pw/rks and
    # regressions/gpw/rks (not under ctest default gate).
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
atv_add_cpp_test(test_cpu_linalg_pulay cpu/linalg/test_pulay.cpp)
atv_add_cpp_test(test_cpu_linalg_diis cpu/linalg/internal/test_diis_mixer.cpp)

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
    # hf-cuda-lgt1: Cartesian shell pack and CUDA route. No SCF.
    atv_add_cpp_test(test_cuda_lgt1_pack       cpu/scf/test_cuda_lgt1_pack.cpp)
    # Semi-local def2-ECP: static table pins, EcpPotential lookup,
    # EcpIntegrals U-matrix vs external PySCF ECPscalar pins, IntegralWorkspace
    # valence wiring, x2c double-counting gate (hf-capability-gaps-06-ecp).
    atv_add_cpp_test(test_ecp                  cpu/scf/test_ecp.cpp)
    # x2c-1e nuclear gradient of h^X2C vs a central difference. No SCF.
    atv_add_cpp_test(test_cpu_scf_x2c_hcore_grad cpu/scf/test_x2c_hcore_grad.cpp)
    # Complex spin-orbital hcore (sigma_dot, spnucsp, X2C-1e). No SCF.
    atv_add_cpp_test(test_cpu_scf_complex_ghf cpu/scf/test_complex_ghf.cpp)
    # Dirac–Hartree–Fock driver shape. No SCF solve (regressions/scf/ghf).
    atv_add_cpp_test(test_cpu_scf_dhf_driver cpu/scf/test_dhf_driver.cpp)
    # Closed-shell CCSD driver shape. No amplitude solve.
    atv_add_cpp_test(test_cpu_scf_ccsd_driver cpu/scf/test_ccsd_driver.cpp)
    # Closed-shell CC2 driver shape. No amplitude solve.
    atv_add_cpp_test(test_cpu_scf_cc2_driver cpu/scf/test_cc2_driver.cpp)
    # Closed-shell CASSCF driver shape. No orbital solve.
    atv_add_cpp_test(test_cpu_scf_casscf_driver cpu/scf/test_casscf_driver.cpp)
    # hf-capability-gaps-18-x2c: spin-free pVp kernel pins, X2cWorkspace
    # nonrel/exact-decoupling identities, ECP↔x2c mutual exclusion both
    # ways, forces() intercept. No heavy-atom SCF in this unit binary.
    # hf-capability-gaps-10-rohf: ROHF (Roothaan effective Fock, PySCF
    # canonical / Guest-Saunders convention) vs externally pinned PySCF.
    # hf-capability-gaps-11-ghf: real two-component GHF — spinor
    # SlaterDeterminant invariants, H2 collapse to UHF, noncollinear win on
    # the frustrated H3 triangle vs externally pinned PySCF, collinear trap.
    atv_add_cpp_test(test_cpu_scf_xc_functional cpu/scf/test_xc_functional.cpp)
    atv_add_cpp_test(test_cpu_scf_xc_extension cpu/scf/test_xc_extension.cpp)
    # scf-driver-refactor-04b-wire: method-driver is DriverNode + outer Plan
    # bind/build only (no full SCF under tests/; wave-2). Nested energy pins:
    # regressions/multiscale/qmmm + scf/*rhf*.
    if(ATV_BUILD_MD AND ATV_BUILD_MINIMIZER AND ATV_BUILD_EVALUATE)
        atv_add_cpp_test(test_cpu_scf_method_driver_wire
                         cpu/scf/test_method_driver_wire.cpp)
    endif()
    # RHF ASPC density predictor in nested MD: a 100-step H2/STO-3G NVT run
    # with predictor(3) stays bound and on the plain-SCF trajectory (Kolafa
    # weights; Lagrange weights dissociate it). Coefficients themselves are
    # pinned in cpu/electronic/test_aspc_predictor.cpp.
    if(ATV_BUILD_MD)
        atv_add_cpp_test(test_cpu_scf_rhf_aspc_md cpu/scf/test_rhf_aspc_md.cpp)
    endif()
    # Minimizer line search keeps an SCF method's accepted trial instead of
    # re-solving it (DriverNode::trial_invariant through the nested slot).
    if(ATV_BUILD_MINIMIZER)
        atv_add_cpp_test(test_cpu_scf_method_trial_reuse
                         cpu/scf/test_method_trial_reuse.cpp)
    endif()
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
    # KS response: f^xc kernel (singlet/triplet), alpha_hf + kernel TDHFPack
    # (dense == Davidson) and the range-separated admission guard. No SCF.
    atv_add_cpp_test(test_cpu_scf_xc_response_kernel cpu/scf/test_xc_response_kernel.cpp)
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
    # scf-review-04-meanfield-dedup: the FD nuclear-Hessian skeleton is
    # central_difference_nuclear_hessian (not a MeanFieldMethod member), pinned in
    # isolation against an analytic quadratic mock force field (no SCF); and the
    # Ha/Bohr->eV/Angstrom gradient glue hoisted onto the tag ctor
    # GradientResult(FromHartreeBohr, ...), pinned against hand-computed
    # eV/Angstrom + eV literals (1e-12).
    atv_add_cpp_test(test_cpu_scf_fd_hessian_template
                     cpu/scf/test_fd_hessian_template.cpp)
    atv_add_cpp_test(test_cpu_scf_gradient_result
                     cpu/scf/test_gradient_result.cpp)
    atv_add_cpp_test(test_cpu_scf_uhf_ctor cpu/scf/test_uhf_ctor.cpp)
    # hf-capability-gaps-09-soscf: CIAH stepper units — exp(κ) unitarity,
    # gradient vs FD, matrix-free CIAH vs dense augmented eigensolve, trust
    # region, MixStage latch; KS+soscf throw. No full RHF.run loop (wave-2).
    # Hard-case contrast: regressions/scf/soscf_hard_case.
    atv_add_cpp_test(test_soscf                   cpu/scf/test_soscf.cpp)
    atv_add_cpp_test(test_orbital_canonicalization cpu/scf/internal/test_orbital_canonicalization.cpp)
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
    atv_add_cpp_test(test_cpu_scf_rys_eri_erf   cpu/scf/test_rys_eri_erf.cpp)
    atv_add_cpp_test(test_cpu_scf_periodic_exchange_lattice
                     cpu/scf/test_periodic_exchange_lattice.cpp)
    atv_add_cpp_test(test_cpu_scf_kpoint_hf
                     cpu/scf/test_kpoint_hf.cpp)
    atv_add_cpp_test(test_cpu_scf_xc_cam_b3lyp_params cpu/scf/test_xc_cam_b3lyp_params.cpp)
    # coulomb_exchange_fused thread invariance. Mirrors cpu/scf/operators.cpp.
    atv_add_cpp_test(test_cpu_scf_operators     cpu/scf/test_operators.cpp)
    atv_add_cpp_test(test_cpu_scf_exact_jk      cpu/scf/test_exact_jk.cpp)
    # Every HF/KS Fock branch through ExactJK vs literal J/K (fixed densities).
    atv_add_cpp_test(test_cpu_scf_hamiltonian_builder cpu/scf/test_hamiltonian_builder.cpp)
    # hf-capability-gaps-12-df: def2-JKFIT table, g-shell 2c/3c integrals,
    # metric whitening, fitted J/K vs dense (single calls, no SCF).
    atv_add_cpp_test(test_density_fitting       cpu/scf/test_density_fitting.cpp)
    # hf-capability-gaps-21-cfmm: shell-pair distributions, near/far tables,
    # ExactJK J + far correction vs direct J on H2O/STO-3G (no SCF).
    atv_add_cpp_test(test_cfmm_j                cpu/scf/test_cfmm_j.cpp)
    atv_add_cpp_test(test_cpu_scf_primitives    cpu/scf/test_primitives.cpp)
    atv_add_cpp_test(test_cpu_scf_accelerators  cpu/scf/test_accelerators.cpp)
    # scf-review-08-ccsd-spatial-diis: CC amplitude DIIS accelerator
    # (internal/cc_diis.h) over flattened [t1;t2] vectors — Pulay B-matrix on a
    # contractive linear toy + single/near-singular/state edges (pure algebra).
    atv_add_cpp_test(test_cpu_scf_cc_diis       cpu/scf/test_cc_diis.cpp)
    # scf-review-08-ccsd-spatial-diis: closed-shell spatial-orbital CCSD kernel
    # (internal/ccsd_spatial.h) — H2/STO-3G E_corr pin (same PySCF pin as
    # regressions/scf/coupled-cluster) + He/STO-3G zero-virtual edge.
    # hf-capability-gaps-07-guess: Hückel/SAP guess matrices + EDIIS/ADIIS
    # far-field mixers (single-call math only; iteration-count contrast is
    # regressions/scf/rhf).
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
    # scf-driver-refactor-03a-convert: RHF/UHF/RKS/UKS as Driver<scf::Phase>,
    # regression-pinned against the 03a-parity energies/eigenvalues. Pins serial
    # execution for the same ULP-determinism reason as test_cpu_scf_stages.
    atv_add_cpp_test(test_cpu_evaluate_driver        cpu/test_evaluate_driver.cpp)
    # esmm-contract-05-accumulate-switch (U7): two accumulating stubs co-bound
    # into ONE cpu::Evaluate pass (no SCF; wave-2). QM×MM pin lives in
    # regressions/architecture/lj_rhf_coupled.
    atv_add_cpp_test(test_cpu_evaluate_source_cobinding
                     cpu/evaluate/test_source_cobinding.cpp)
    atv_add_cpp_test(test_cpu_scf_minao         cpu/scf/test_minao.cpp)
    atv_add_cpp_test(test_cpu_scf_rys_eri_deriv cpu/scf/test_rys_eri_deriv.cpp)
    # Full SCF-driven geometry optimizations are e2e, not unit tests.
    atv_add_cpp_test(test_cpu_minimizer_line_search cpu/minimize/test_line_search.cpp)
    if(ATV_BUILD_MINIMIZER)
        atv_add_cpp_test(test_cpu_minimize_driver_swap_optimizer cpu/test_minimize_driver_swap_optimizer.cpp)
    endif()
    # Vibrational analysis (finite-difference Hessian) is an end-to-end /
    # PySCF-parity suite — too slow for the unit gate; not registered here.
    if(ATV_BUILD_MINIMIZER)
        atv_add_cpp_test(test_cpu_recorder_density_cube cpu/record/test_density_cube.cpp)
    endif()
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
    atv_add_test(test_cuda_spatial_grid cuda/test_spatial_grid.cu)
    if(ATV_BUILD_MD)
        atv_add_test(test_cuda_multi_gpu  cuda/dynamics/test_multi_gpu.cu)
    endif()
    if(ATV_BUILD_NEIGHBOR)
        atv_add_test(test_cuda_hilbert cuda/dynamics/test_hilbert.cu)
    endif()
    atv_add_test(test_cuda_timer cuda/device/test_timer.cu)
    atv_add_test(test_cuda_frame      cuda/test_frame.cu)
    if(ATV_BUILD_MOLPY)
        atv_add_test(test_cuda_materialize cuda/test_materialize.cu)
    endif()
    atv_add_test(test_cuda_backend    cuda/test_backend.cu)
    atv_add_test(test_cuda_nested_requests cuda/driver/test_nested_requests.cu)
    atv_add_test(test_cuda_input_updates cuda/driver/test_input_updates.cu)
    atv_add_test(test_cuda_borrowed_execution cuda/driver/test_borrowed_execution.cu)
    if(ATV_BUILD_EVALUATE)
        atv_add_test(test_cuda_evaluate cuda/evaluate/test_evaluate.cu)
    endif()
    if(ATV_BUILD_MD AND ATV_BUILD_NEIGHBOR AND ATV_BUILD_POTENTIALS)
        atv_add_test(test_cuda_capture_buffer cuda/asynchronous/test_capture_buffer.cu)
    endif()
    if(ATV_BUILD_SEMIEMPIRICAL)
        atv_add_test(test_cuda_nddo_fock cuda/semiempirical/test_nddo_fock.cu)
        atv_add_test(test_cuda_nddo_two_electron cuda/semiempirical/test_nddo_two_electron.cu)
        atv_add_test(test_cuda_nddo_residual cuda/semiempirical/test_nddo_residual.cu)
    endif()
    atv_add_test(test_core_plan       core/test_plan.cu)
    if(ATV_BUILD_MD)
        atv_add_test(test_cuda_md_primitives cuda/dynamics/test_md_primitives.cu)
        atv_add_test(test_cuda_precision cuda/frame/test_precision.cu)
    endif()
    if(ATV_BUILD_MD)
        atv_add_test(test_cuda_md_zero_momentum cuda/dynamics/test_zero_momentum.cu)
    endif()
    if(ATV_BUILD_MD)
        atv_add_test(test_cuda_nose_hoover_npt cuda/dynamics/test_nose_hoover_npt.cu)
        atv_add_test(test_cuda_fix_efield cuda/dynamics/test_fix_efield.cu)
        atv_add_test(test_cuda_image_flags cuda/dynamics/test_image_flags.cu)
        atv_add_test(test_cuda_nvt_sllod cuda/dynamics/test_nvt_sllod.cu)
        atv_add_test(test_cuda_deform cuda/dynamics/test_deform.cu)
        if(ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
            atv_add_test(test_cuda_nvt_sllod_flip_lj
                cuda/dynamics/test_nvt_sllod_flip_lj.cu)
        endif()
    endif()
    if(ATV_BUILD_MINIMIZER)
        atv_add_test(test_cuda_minimizer_image_flags
            cuda/minimize/test_minimizer_image_flags.cu)
    endif()
    if(ATV_BUILD_NEIGHBOR)
        atv_add_test(test_cuda_neighborlist_skin_gating cuda/dynamics/test_neighborlist_skin_gating.cu)
        atv_add_test(test_cuda_evaluation_demand cuda/dynamics/test_evaluation_demand.cu)
    endif()
    if(ATV_BUILD_MD)
        atv_add_test(test_cuda_cluster_contract cuda/dynamics/test_cuda_cluster_contract.cu)
    endif()
    if(ATV_BUILD_NEIGHBOR)
        atv_add_test(test_cuda_cluster_mask cuda/dynamics/test_cuda_cluster_mask.cu)
    endif()
    if(ATV_BUILD_MD)
        atv_add_test(test_pair_lj_tile_step0 cuda/potentials/test_pair_lj_tile_step0.cu)
    endif()
    if(ATV_BUILD_NEIGHBOR)
        atv_add_unit_exe(test_cuda_atom_cell_sort cuda/dynamics/test_atom_cell_sort.cu)
        # Pulls in the private primitive header src/cuda/neighbor/atom_cell_sort.cuh
        target_include_directories(test_cuda_atom_cell_sort PRIVATE
            ${CMAKE_SOURCE_DIR}/src/cuda/neighbor)
    endif()
    # cuda::RDF on the molpy streaming accumulator (arch-refactor-07-molpy):
    # static-trajectory g(r) parity vs a host-fed cpu::compute::RDFAccumulator
    # + fail-fast ctor edges. Needs a GPU node (runs a cuda::md::MD).
    if(ATV_BUILD_ANALYSIS AND ATV_BUILD_MD)
        atv_add_test(test_cuda_observer_rdf_molpy cuda/observe/test_rdf_molpy.cu)
    endif()
    # cuda::MrecRecorder: count / cell / images, CPU parity, a held writer
    # never holds the run, device-decided drops. Same IO gate as the
    # component (iron law 8).
    if(ATV_BUILD_MD AND ATV_BUILD_IO AND ATV_BUILD_NEIGHBOR
       AND ATV_BUILD_POTENTIALS)
        atv_add_test(test_cuda_mrec_recorder cuda/record/test_mrec_recorder.cu)
    endif()
    # The LAMMPS output step convention of cuda::XyzRecorder, cuda::Thermo
    # and cuda::md::PressureSeries against independent runs (the XYZ writer
    # reaches molpy IO).
    if(ATV_BUILD_MD AND ATV_BUILD_IO AND ATV_BUILD_NEIGHBOR
       AND ATV_BUILD_POTENTIALS)
        atv_add_test(test_cuda_recorder_step_convention
                     cuda/record/test_recorder_step_convention.cu)
    endif()
    # The LAMMPS `minimize` output convention on the CUDA minimizer loop
    # (initial state, completed-iteration labels, forced final state on an
    # early stop) against independent CUDA minimizations.
    if(ATV_BUILD_MINIMIZER AND ATV_BUILD_IO AND ATV_BUILD_NEIGHBOR
       AND ATV_BUILD_POTENTIALS)
        atv_add_test(test_cuda_minimize_output
                     cuda/minimize/test_minimize_output.cu)
    endif()
    if(ATV_BUILD_POTENTIALS)
        atv_add_test(test_cuda_harmonic_coefficients cuda/potentials/test_harmonic_coefficients.cu)
        atv_add_test(test_cuda_angle_cosine cuda/potentials/test_angle_cosine.cu)
        atv_add_test(test_cuda_bond_fene cuda/potentials/test_bond_fene.cu)
        atv_add_test(test_cuda_packed_tiles cuda/potentials/test_packed_tiles.cu)
    endif()
    # Per-term energies (energy_pair/_bond/_angle/_dihedral) of every CUDA
    # force-field family against the same-settings CPU style.
    if(ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR AND ATV_BUILD_EVALUATE)
        atv_add_test(test_cuda_energy_terms cuda/potentials/test_energy_terms.cu)
    endif()
    if(ATV_BUILD_MD)
        atv_add_test(test_cuda_pair_lj_special_bonds cuda/potentials/test_pair_lj_special_bonds.cu)
        if(ATV_BUILD_EVALUATE)
            atv_add_test(test_cuda_pair_lj_typed_cutoff
                cuda/potentials/test_pair_lj_typed_cutoff.cu)
        endif()
    endif()
    # Ewald always; the PME cases compile only with ATV_HAS_KSPACE.
    atv_add_test(test_cuda_kspace_virial cuda/longrange/test_kspace_virial.cu)
    if(ATV_BUILD_KSPACE)
        atv_add_test(test_cuda_kspace_pppm cuda/longrange/test_kspace_pppm.cu)
    endif()
    # Compiles to no tests without ATV_HAS_KSPACE.
    atv_add_test(test_cuda_pppm_stencil cuda/longrange/test_pppm_stencil.cu)
    # pair_kernel is cuda-owned (backend-mirror-02-absorb); its scalar-math test
    # moved here from the former cpu/potentials/test_pair_kernel.cpp.
    if(ATV_BUILD_POTENTIALS)
        atv_add_test(test_cuda_pair_kernel cuda/potentials/test_pair_kernel.cu)
    endif()
    if(ATV_BUILD_PW)
        atv_add_unit_exe(test_cuda_pw_fft cuda/planewave/test_fft.cu)
        target_link_libraries(test_cuda_pw_fft PRIVATE CUDA::cufft)
        atv_add_unit_exe(test_cuda_pw_poisson cuda/planewave/test_poisson.cu)
        target_link_libraries(test_cuda_pw_poisson PRIVATE CUDA::cufft)
        atv_add_unit_exe(test_cuda_pw_grid_map cuda/planewave/test_grid_map.cu)
        target_link_libraries(test_cuda_pw_grid_map PRIVATE CUDA::cufft)
        atv_add_unit_exe(test_cuda_pw_xc_lda cuda/planewave/test_xc_lda.cu)
        target_link_libraries(test_cuda_pw_xc_lda PRIVATE CUDA::cufft)
        atv_add_unit_exe(test_cuda_pw_xc_pbe cuda/planewave/test_xc_pbe.cu)
        target_link_libraries(test_cuda_pw_xc_pbe PRIVATE CUDA::cufft)
        atv_add_unit_exe(test_cuda_pw_xc_pbe_spin cuda/planewave/test_xc_pbe_spin.cu)
        target_link_libraries(test_cuda_pw_xc_pbe_spin PRIVATE CUDA::cufft)
        atv_add_unit_exe(test_cuda_pw_veff cuda/planewave/test_veff.cu)
        target_link_libraries(test_cuda_pw_veff PRIVATE CUDA::cufft)
        # PW device components, followed by public-surface honesty guards.
        atv_add_unit_exe(test_cuda_pw_frozen_operator_virial cuda/planewave/test_frozen_operator_virial.cu)
        atv_add_unit_exe(test_cuda_pw_frozen_spin_packing cuda/planewave/test_frozen_spin_packing.cu)
        if(TARGET test_cuda_pw_frozen_spin_packing)
            target_link_libraries(test_cuda_pw_frozen_spin_packing PRIVATE CUDA::cufft)
        endif()
        atv_add_unit_exe(test_cuda_pw_frozen_spin_domain cuda/planewave/test_frozen_spin_domain.cu)
        if(TARGET test_cuda_pw_frozen_spin_domain)
            target_link_libraries(test_cuda_pw_frozen_spin_domain PRIVATE CUDA::cufft)
        endif()
        atv_add_unit_exe(test_cuda_pw_frozen_derivative_certificate cuda/planewave/test_frozen_derivative_certificate.cu)
        atv_add_unit_exe(test_cuda_pw_frozen_xc_virial cuda/planewave/test_frozen_xc_virial.cu)
        target_link_libraries(test_cuda_pw_frozen_xc_virial PRIVATE CUDA::cufft)
        atv_add_unit_exe(test_cuda_pw_frozen_spin_xc_virial cuda/planewave/test_frozen_spin_xc_virial.cu)
        if(TARGET test_cuda_pw_frozen_spin_xc_virial)
            target_link_libraries(test_cuda_pw_frozen_spin_xc_virial PRIVATE CUDA::cufft)
        endif()
        atv_add_unit_exe(test_cuda_pw_final_density cuda/planewave/test_final_density.cu)
        if(TARGET test_cuda_pw_final_density)
            target_link_libraries(test_cuda_pw_final_density PRIVATE CUDA::cufft)
        endif()
        if(TARGET test_cuda_pw_frozen_operator_virial)
            target_link_libraries(test_cuda_pw_frozen_operator_virial PRIVATE CUDA::cufft)
        endif()
        atv_add_unit_exe(test_cuda_pw_nc_projector_gauge cuda/planewave/test_nc_projector_gauge.cu)
        if(TARGET test_cuda_pw_nc_projector_gauge)
            target_link_libraries(test_cuda_pw_nc_projector_gauge PRIVATE CUDA::cufft)
        endif()
        atv_add_unit_exe(test_cuda_pw_guards cuda/planewave/test_guards.cu)
        if(TARGET test_cuda_pw_guards)
            target_link_libraries(test_cuda_pw_guards PRIVATE CUDA::cufft)
        endif()
        # DeviceDensity stage parity vs the CPU stage helpers. Unit-only:
        # no SCF loop, no Davidson — the end-to-end parity solve lives in
        # regressions/pw/pw-cuda-lda-si.
        atv_add_unit_exe(test_cuda_pw_density cuda/planewave/test_density.cu)
        if(TARGET test_cuda_pw_density)
            target_link_libraries(test_cuda_pw_density PRIVATE CUDA::cufft)
        endif()
        # Device local HF force kernel + public API symmetry (unit-only).
        # Name keeps historical token; full SCF product e2e is regressions.
        # analytic path on CPU); this is a kernel/API unit gate.
        atv_add_unit_exe(test_cuda_pw_force_local cuda/planewave/test_pw_force_local.cu)
        if(TARGET test_cuda_pw_force_local)
            target_link_libraries(test_cuda_pw_force_local PRIVATE CUDA::cufft)
        endif()
        # Cell stress public API + cheap H isotropy/CUDA↔CPU (unit-only).
        # Name keeps historical token; full SCF product e2e is regressions.
        atv_add_unit_exe(test_cuda_pw_cell_stress cuda/planewave/test_pw_cell_stress.cu)
        if(TARGET test_cuda_pw_cell_stress)
            target_link_libraries(test_cuda_pw_cell_stress PRIVATE CUDA::cufft)
        endif()
    endif()
    # CUDA GPW Γ Poisson foothold — needs CPU solve_poisson oracle (GPW+PW)
    # and device kernels.
    if(ATV_BUILD_GPW AND ATV_BUILD_PW)
        atv_add_unit_exe(test_cuda_gpw_device_poisson cuda/gpw/test_device_poisson.cu)
        target_link_libraries(test_cuda_gpw_device_poisson PRIVATE CUDA::cufft)
        # CUDA GPW Padé LDA XC foothold — vs cpu::xc::pade_lsd / xc_grid_pade.
        atv_add_test(test_cuda_gpw_xc_lda cuda/gpw/test_xc_lda.cu)
        # CUDA GPW product ρ collocate foothold — vs CPU collint oracle.
        atv_add_test(test_cuda_gpw_product_collocate cuda/gpw/test_product_collocate.cu)
        # CUDA GPW product V·shape integrate foothold — vs CPU collint oracle.
        atv_add_test(test_cuda_gpw_product_integrate cuda/gpw/test_product_integrate.cu)
        # CUDA GPW composed KS chain stage — vs CPU ks_grid_build_rks.
        atv_add_unit_exe(test_cuda_gpw_ks_chain cuda/gpw/test_ks_chain.cu)
        if(TARGET test_cuda_gpw_ks_chain)
            target_link_libraries(test_cuda_gpw_ks_chain PRIVATE CUDA::cufft)
        endif()
        atv_add_unit_exe(test_cuda_gpw_ks_gga cuda/gpw/test_ks_gga.cu)
        if(TARGET test_cuda_gpw_ks_gga)
            target_link_libraries(test_cuda_gpw_ks_gga PRIVATE CUDA::cufft)
        endif()
        # CUDA multi-grid product collocate/integrate KS stage vs CPU Γ
        # multigrid fragment (prolong_fft / restrict_fft + ks_grid_build_rks).
        atv_add_unit_exe(test_cuda_gpw_ks_multigrid cuda/gpw/test_ks_multigrid.cu)
        if(TARGET test_cuda_gpw_ks_multigrid)
            target_link_libraries(test_cuda_gpw_ks_multigrid PRIVATE CUDA::cufft)
        endif()
        atv_add_unit_exe(test_cuda_gpw_stress cuda/gpw/test_gpw_stress.cu)
        if(TARGET test_cuda_gpw_stress)
            target_link_libraries(test_cuda_gpw_stress PRIVATE CUDA::cufft)
        endif()
        # Device grid force kernels + free-fn guards (cuda-gpw-device-forces).
        atv_add_unit_exe(test_cuda_gpw_forces cuda/gpw/test_gpw_forces.cu)
        if(TARGET test_cuda_gpw_forces)
            target_link_libraries(test_cuda_gpw_forces PRIVATE CUDA::cufft)
        endif()
        # Device AO DIIS + cuSOLVER eigen vs host DIISMixer / diagonalize_fock
        # (cuda-gpw-device-diis-eigen).
        atv_add_unit_exe(test_cuda_gpw_diis_eigen cuda/gpw/test_diis_eigen.cu)
        if(TARGET test_cuda_gpw_diis_eigen)
            target_link_libraries(test_cuda_gpw_diis_eigen PRIVATE CUDA::cufft
                                  CUDA::cublas CUDA::cusolver)
        endif()
        # Device P→weight + acc→V scatter vs host oracles
        # (cuda-gpw-device-weight-pack).
        atv_add_test(test_cuda_gpw_weight_pack cuda/gpw/test_weight_pack.cu)
        # Device multi-k χ^k density collocate vs host collocate_density_k
        # (cuda-gpw-kpoint-device-collocate).
        atv_add_unit_exe(test_cuda_gpw_kpoint_collocate
                     cuda/gpw/test_kpoint_collocate.cu)
        if(TARGET test_cuda_gpw_kpoint_collocate)
            target_link_libraries(test_cuda_gpw_kpoint_collocate
                                  PRIVATE CUDA::cufft)
        endif()
        # Device multi-k V→H_k integrate vs host integrate_potential_k
        # (cuda-gpw-kpoint-device-integrate).
        atv_add_unit_exe(test_cuda_gpw_kpoint_integrate
                     cuda/gpw/test_kpoint_integrate.cu)
        if(TARGET test_cuda_gpw_kpoint_integrate)
            target_link_libraries(test_cuda_gpw_kpoint_integrate
                                  PRIVATE CUDA::cufft)
        endif()
        # Multi-k device Pulay DIIS + zheevd vs host KDiis/diagonalize_k
        # (cuda-gpw-kpoint-device-diis).
        atv_add_unit_exe(test_cuda_gpw_kpoint_diis_eigen
                     cuda/gpw/test_kpoint_diis_eigen.cu)
        if(TARGET test_cuda_gpw_kpoint_diis_eigen)
            target_link_libraries(test_cuda_gpw_kpoint_diis_eigen
                                  PRIVATE CUDA::cufft CUDA::cublas
                                          CUDA::cusolver)
        endif()
        # Device multi-k H_k = H0 + Vmat assemble
        # (cuda-gpw-kpoint-h-assemble).
        atv_add_unit_exe(test_cuda_gpw_kpoint_h_assemble
                     cuda/gpw/test_kpoint_h_assemble.cu)
        if(TARGET test_cuda_gpw_kpoint_h_assemble)
            target_link_libraries(test_cuda_gpw_kpoint_h_assemble
                                  PRIVATE CUDA::cufft)
        endif()
        # Multi-k device phased STVnl (cuda-gpw-kpoint-device-stvnl).
        atv_add_unit_exe(test_cuda_gpw_kpoint_stvnl
                     cuda/gpw/test_kpoint_stvnl.cu)
        if(TARGET test_cuda_gpw_kpoint_stvnl)
            target_link_libraries(test_cuda_gpw_kpoint_stvnl
                                  PRIVATE CUDA::cufft)
        endif()
        # Device H residual residency: assemble → ingest → residual_from_device
        # (cuda-gpw-kpoint-h-residency).
        atv_add_unit_exe(test_cuda_gpw_kpoint_h_residency
                     cuda/gpw/test_kpoint_h_residency.cu)
        if(TARGET test_cuda_gpw_kpoint_h_residency)
            target_link_libraries(test_cuda_gpw_kpoint_h_residency
                                  PRIVATE CUDA::cufft CUDA::cublas
                                          CUDA::cusolver)
        endif()
    endif()
    atv_add_test(test_cuda_scf        cuda/scf/test_scf.cu)
    if(ATV_BUILD_HF)
        atv_add_test(test_cuda_scf_rhf cuda/scf/test_rhf.cu)
        atv_add_test(test_cuda_scf_device_integrals cuda/scf/test_device_integrals.cu)
        atv_add_test(test_cuda_nuclear_identity cuda/scf/test_nuclear_identity.cu)
        atv_add_test(test_cuda_scf_diis_mixer cuda/scf/test_diis_mixer.cu)
        atv_add_test(test_cuda_scf_eigen_solver cuda/scf/test_eigen_solver.cu)
        atv_add_test(test_cuda_hf_convergence cuda/scf/test_convergence_check.cu)
        atv_add_test(test_cuda_gaussian_math cuda/scf/test_boys.cu)
        atv_add_test(test_cuda_mp2_device cuda/scf/test_mp2_device.cu)
        atv_add_test(test_cuda_uhf_fock cuda/scf/test_uhf_fock.cu)
        atv_add_test(test_cuda_uhf_guess cuda/scf/test_uhf_guess.cu)
    endif()
    # CPU/CUDA PW parity is an end-to-end SCF solve: it lives in
    # regressions/pw/rks (--device=both), never under ctest.
    if(ATV_BUILD_MD AND ATV_BUILD_MINIMIZER)
        atv_add_test(test_cuda_driver_role_validation cuda/test_cuda_driver_role_validation.cu)
    endif()
    if(ATV_BUILD_MD)
        atv_add_test(test_cuda_component_provides_probes
            cuda/component/test_provides_probes.cu)
    endif()
    if(ATV_BUILD_MD AND ATV_BUILD_MINIMIZER)
        atv_add_test(test_cuda_driver_require_match cuda/driver/test_require_match.cu)
    endif()
    if(ATV_BUILD_MINIMIZER)
        atv_add_test(test_cuda_minimize_contract cuda/driver/test_minimize_contract.cu)
    endif()
    if(ATV_BUILD_MD)
        atv_add_test(test_cuda_driver_unfix cuda/driver/test_unfix.cu)
    endif()
    atv_add_test(test_cuda_driver_live_table cuda/driver/test_driver_live_table.cu)
    if(ATV_BUILD_EVALUATE AND ATV_BUILD_NEIGHBOR AND ATV_BUILD_POTENTIALS)
        atv_add_test(test_cuda_driver_long_range cuda/driver/test_long_range_contract.cu)
    endif()
    # Registration-time PhaseKind gate, CUDA mirror
    # (arch-refactor-01-bindings). add()-only — needs no CUDA device.
    if(ATV_BUILD_MD AND ATV_BUILD_MINIMIZER)
        atv_add_test(test_cuda_driver_binding_kind cuda/test_driver_binding_kind.cu)
    endif()

    # Compile site of the HF-gated asynchronous consumers (MullikenObs,
    # DensityRecorder).
    add_subdirectory(cuda/asynchronous)

endif()



if(ATV_BUILD_MD)
    atv_add_unit_exe(test_app_input_script
        app/test_input_script.cpp
        ${CMAKE_SOURCE_DIR}/app/atomiverse/input_script.cpp)
    target_include_directories(test_app_input_script PRIVATE
        ${CMAKE_SOURCE_DIR}/app/atomiverse)
endif()

# ctest policy reminder: multi-step Driver pipelines (MD/Minimize/Path
# run(N) for equilibration / production-style trajectories, multi-rank
# PE/MD campaigns) are DELETED from this tree — not "disabled". Public
# end-to-end scenarios live under regressions/ (ATV_BUILD_REGRESSIONS,
# never ctest). Acceptance / parity gates live under benchmarks/** when
# that tree is rebuilt. If a target is registered above, it must be a
# single-component unit test.

# ── Single executables ─────────────────────────────────────────────────
get_property(_atv_unit_sources GLOBAL PROPERTY ATV_UNIT_SOURCES)
add_executable(atv_tests ${_atv_unit_sources})
target_link_libraries(atv_tests PRIVATE atomiverse GTest::gtest_main)
target_include_directories(atv_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(atv_tests PROPERTIES CXX_STANDARD 20)
target_precompile_headers(atv_tests PRIVATE <gtest/gtest.h>
    "${CMAKE_SOURCE_DIR}/include/atomiverse/cpu/frame/frame.h")
gtest_discover_tests(atv_tests DISCOVERY_MODE PRE_TEST
    PROPERTIES TIMEOUT ${ATV_TEST_TIMEOUT_SECONDS}
               ENVIRONMENT "${ATV_TEST_ENVIRONMENT}")

get_property(_atv_unit_cuda_sources GLOBAL PROPERTY ATV_UNIT_CUDA_SOURCES)
if(_atv_unit_cuda_sources)
    add_executable(atv_tests_cuda ${_atv_unit_cuda_sources})
    target_link_libraries(atv_tests_cuda PRIVATE atomiverse GTest::gtest_main)
    target_include_directories(atv_tests_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(atv_tests_cuda PROPERTIES CXX_STANDARD 20
                          CUDA_SEPARABLE_COMPILATION ON)
    gtest_discover_tests(atv_tests_cuda DISCOVERY_MODE PRE_TEST
        PROPERTIES TIMEOUT ${ATV_TEST_TIMEOUT_SECONDS} LABELS cuda
                   ENVIRONMENT "${ATV_TEST_ENVIRONMENT}")
endif()

get_property(_atv_unit_mpi_sources GLOBAL PROPERTY ATV_UNIT_MPI_SOURCES)
if(_atv_unit_mpi_sources)
    add_executable(atv_tests_mpi mpi/main.cpp ${_atv_unit_mpi_sources})
    target_link_libraries(atv_tests_mpi PRIVATE atomiverse GTest::gtest MPI::MPI_CXX)
    target_include_directories(atv_tests_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(atv_tests_mpi PROPERTIES CXX_STANDARD 20)
endif()
