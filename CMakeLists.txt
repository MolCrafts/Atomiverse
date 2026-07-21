cmake_minimum_required(VERSION 3.18)
option(ATV_DOCS_ONLY "Configure documentation targets without the C++/CUDA toolchain" OFF)

if(ATV_DOCS_ONLY)
    project(Atomiverse LANGUAGES NONE)
else()
    project(Atomiverse LANGUAGES CXX)
endif()

find_package(Doxygen QUIET)

if(NOT ATV_DOCS_ONLY)
    # Default to an optimized build. An empty CMAKE_BUILD_TYPE compiles at
    # -O0, which makes the ERI / SCF / PW hot paths 10-30x slower — easy to
    # ship by accident since -march=native alone carries no -O level.
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                     Debug Release RelWithDebInfo MinSizeRel)
    endif()

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    option(ATV_BUILD_CUDA "Build CUDA backend (requires CUDA toolkit)" ON)
    if(ATV_BUILD_CUDA)
        enable_language(CUDA)
        # CUDA TUs that transitively include xtensor (via cpu/frame.h) need
        # C++20: xtensor 0.27.1 uses `<concepts>` / `std::input_iterator`
        # on toolchains where `<concepts>` is available (GCC 13), which
        # nvcc cannot parse at -std=c++17. The CPU side stays at C++17.
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
    endif()

    set(ATV_CUDA_REAL "float" CACHE STRING "CUDA per-atom storage precision (float or double)")
    set_property(CACHE ATV_CUDA_REAL PROPERTY STRINGS float double)

    set(ATV_CUDA_WORK "double" CACHE STRING "CUDA kernel computation precision (float or double)")
    set_property(CACHE ATV_CUDA_WORK PROPERTY STRINGS float double)

    set(ATV_CUDA_PREC "double" CACHE STRING "CUDA accumulator/SCF precision (float or double)")
    set_property(CACHE ATV_CUDA_PREC PROPERTY STRINGS float double)

    # CPU/shared numerical precision — LAMMPS-style tiered control (see
    # include/atomiverse/core/precision.h). Storage and kernel-work reals may be
    # lowered to float independently; accumulators stay double (never economized).
    # Replaces the former single-switch -DATV_REAL / -DATV_INT, which were never
    # wired and silently no-op'd.
    set(ATV_REAL_STORAGE "double" CACHE STRING "Per-atom storage real precision (float or double)")
    set_property(CACHE ATV_REAL_STORAGE PROPERTY STRINGS float double)

    set(ATV_REAL_WORK "double" CACHE STRING "Transient kernel-work real precision (float or double)")
    set_property(CACHE ATV_REAL_WORK PROPERTY STRINGS float double)

    # Default bigbig (64-bit IDs) preserves the legacy always-int64 tag width, so
    # the default build is byte-identical to pre-migration. smallbig (32-bit IDs,
    # the LAMMPS default + molrs's on-disk tag width) is the memory-saving opt-in.
    set(ATV_ID_SIZE "bigbig" CACHE STRING "Global-ID integer width: bigbig (64-bit IDs, default) or smallbig (32-bit IDs)")
    set_property(CACHE ATV_ID_SIZE PROPERTY STRINGS smallbig bigbig)

    option(ATV_BUILD_HF "Build RHF electronic structure code" OFF)
    option(ATV_BUILD_PW "Build plane-wave DFT module (requires FFTW3)" OFF)
    option(ATV_BUILD_GPW "Build GPW DFT module (Gaussian + uniform-grid Hartree/XC + GTH PP)" OFF)
    option(ATV_BUILD_KSPACE "Build Particle-Mesh Ewald (PME) electrostatics (CPU: FFTW3, CUDA: cuFFT)" OFF)
    option(ATV_BUILD_TORCH "Build LibTorch backend (requires LibTorch)" OFF)
    option(ATV_BUILD_LIBCINT "Fetch libcint for integral validation tests" OFF)
    option(ATV_BUILD_XC_ORACLE
           "Build libxc-backed XC oracle (test-only scaffolding; deletable after native parity is locked in)"
           OFF)
    # Default: git submodule at ${CMAKE_SOURCE_DIR}/molrs (see .gitmodules).
    # Override with -DATV_MOLRS_PATH=... only for local experiments; CI and
    # normal builds use the submodule pin.
    set(ATV_MOLRS_PATH "${CMAKE_SOURCE_DIR}/molrs" CACHE PATH
        "Path to molrs source tree (default: git submodule ./molrs)")
    include("${CMAKE_SOURCE_DIR}/cmake/MolrsContract.cmake")
    if(NOT EXISTS "${ATV_MOLRS_PATH}/Cargo.toml")
        message(FATAL_ERROR
            "molrs not found at ATV_MOLRS_PATH='${ATV_MOLRS_PATH}'.\n"
            "  Init the submodule:  git submodule update --init --recursive molrs\n"
            "  Or point explicitly: -DATV_MOLRS_PATH=/path/to/molrs")
    endif()
    atv_validate_molrs_source_contract("${ATV_MOLRS_PATH}")
    option(ATV_BUILD_PYTHON "Build Python bindings (nanobind)" OFF)
    # The nanobind module is a shared object linking the static atomiverse
    # and fetched-dependency archives (spdlog, ...): everything must be PIC.
    # Set BEFORE FetchContent so the fetched targets inherit it. macOS
    # compiles PIC by default, which is why this was never needed there.
    if(ATV_BUILD_PYTHON)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    endif()
    option(ATV_BUILD_TESTS "Build tests" OFF)
    option(ATV_BUILD_REGRESSIONS
        "Build end-to-end textbook regression cases (regressions/)" OFF)
    option(ATV_BUILD_BENCHMARKS "Build benchmarks" OFF)
    option(ATV_BUILD_EXAMPLES "Build examples" OFF)
    option(ATV_BUILD_LAUNCHER
           "Build the atomiverse MPI launcher executable (rank-0 embedded CPython)" OFF)
    option(ATV_USE_MATPLOT "Enable matplotplusplus for plotting in examples (requires gnuplot)" OFF)

    # Product-module gates (MD/MC/MINIMIZER/… + POTENTIALS/NEIGHBOR DAG).
    # Must run after ATV_BUILD_KSPACE is declared (KSPACE → POTENTIALS edge).
    include("${CMAKE_SOURCE_DIR}/cmake/AtvProductModules.cmake")

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/deps")

    # ── Dependencies ──

    include(FetchContent)
    FetchContent_Declare(
        xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
        GIT_TAG 0.8.2
    )
    FetchContent_Declare(
        xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
        GIT_TAG 0.27.1
    )
    FetchContent_Declare(
        xtensor-blas
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
        GIT_TAG 0.23.0
    )
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.17.0
    )
    FetchContent_Declare(
        xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
        GIT_TAG 14.2.0
    )
    FetchContent_MakeAvailable(xtl xtensor xtensor-blas spdlog xsimd)

    # ── spdlog bundled fmt: Apple Clang 21+ consteval workaround ───────
    # spdlog 1.15's bundled fmt base.h defines FMT_USE_CONSTEVAL=1 for
    # Apple Clang >= 14, but Apple Clang 21 has a stricter consteval
    # enforcement that rejects valid fmt code. Patch the version boundary.
    if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        execute_process(COMMAND sed -i ""
            "s/__apple_build_version__ < 14000029L/__apple_build_version__ < 14000029L || __apple_build_version__ >= 21000000L/"
            "${spdlog_SOURCE_DIR}/include/spdlog/fmt/bundled/base.h"
        )
    endif()

    # ── Library ──

    # ── OpenMP ──
    # Apple Clang doesn't ship OpenMP; use Homebrew libomp if available.
    if(APPLE AND NOT OpenMP_CXX_FOUND)
        find_library(LIBOMP_LIB omp PATHS /opt/homebrew/opt/libomp/lib NO_DEFAULT_PATH)
        if(LIBOMP_LIB)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_omp_LIBRARY "${LIBOMP_LIB}" CACHE STRING "" FORCE)
            include_directories(/opt/homebrew/opt/libomp/include)
        endif()
    endif()
    find_package(OpenMP)

    # ── MPI (optional) ──
    find_package(MPI QUIET)
    option(ATV_USE_MPI "Enable MPI domain decomposition for CPU MD" ${MPI_CXX_FOUND})
    if(ATV_USE_MPI AND NOT MPI_CXX_FOUND)
        message(FATAL_ERROR
            "ATV_USE_MPI=ON but no MPI installation was found. "
            "Install an MPI implementation (OpenMPI, MPICH, MVAPICH) or "
            "configure with -DATV_USE_MPI=OFF.")
    endif()

    # Always-on core + product-module gated domain sources (see AtvProductModules.cmake).
    set(ATOMIVERSE_SRCS
        # Cross-cutting core (logging registry, metrics)
        src/core/log.cpp
        src/core/real_harmonics.cpp
        src/core/molrs_contract.cpp
        src/cpu/scf/solve.cpp
        src/cpu/io/frame_writer.cpp
        src/cpu/io/lammps_data.cpp
        src/cpu/io/extxyz.cpp
        src/cpu/observer/rdf.cpp
        src/cpu/observer/rdf_curve.cpp
        src/cpu/observer/msd.cpp
        src/cpu/observer/vacf.cpp
        src/cpu/observer/trajectory_metrics.cpp
        src/cpu/recorder/thermo.cpp
        src/cpu/recorder/xyz.cpp
        src/cpu/recorder/timing_report.cpp
        # CPU backend infrastructure
        src/cpu/frame/frame.cpp
        src/cpu/driver/backend.cpp
        src/cpu/driver/dof.cpp
        src/cpu/frame/box.cpp
        src/cpu/component/zero.cpp
        src/cpu/frame/materialize.cpp
        src/cpu/driver/driver.cpp
        # CPU task infrastructure (TaskSpec + codec + materializer + protocol)
        src/cpu/task/task_spec.cpp
        src/cpu/task/task_codec.cpp
        src/cpu/task/task_materializer.cpp
        src/cpu/task/task_protocol.cpp
        src/cpu/task/launcher_gate.cpp
        src/cpu/async/analysis_executor.cpp
        src/cpu/async/async_component_base.cpp
        src/cpu/async/snapshot_publisher.cpp
        src/cpu/observer/virial_stress.cpp
        # CPU MPI communication subsystem (LAMMPS-shaped, ATV::cpu::comm::*)
        src/cpu/comm/atom_pack.cpp
        src/cpu/comm/bcast_frame.cpp
        src/cpu/comm/halo_exchange.cpp
        src/cpu/comm/atom_migration.cpp
        src/cpu/comm/domain.cpp
        src/cpu/comm/reduce.cpp
        src/cpu/comm/gather.cpp
        src/cpu/comm/gather_for_io.cpp
        src/cpu/comm/rcb.cpp
        # Direct Ewald always (no external deps); PME gated by ATV_BUILD_KSPACE below
        src/cpu/longrange/ewald.cpp
    )

    if(ATV_BUILD_SEMIEMPIRICAL)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/semiempirical/am1_parameters.cpp
            src/cpu/semiempirical/rm1_parameters.cpp
            src/cpu/semiempirical/nddo_density.cpp
            src/cpu/semiempirical/nddo_fock.cpp
            src/cpu/semiempirical/nddo_hcore.cpp
            src/cpu/semiempirical/nddo_matrix.cpp
            src/cpu/semiempirical/nddo_overlap.cpp
            src/cpu/semiempirical/nddo_system.cpp
            src/cpu/semiempirical/nddo_two_electron.cpp
            src/cpu/semiempirical/nddo_solver.cpp
            src/cpu/semiempirical/am1_evaluator.cpp
            src/cpu/semiempirical/rm1_evaluator.cpp
            src/cpu/semiempirical/am1_bcc_charge_assigner.cpp
        )
    endif()

    if(ATV_BUILD_MC)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/mc/translate_move.cpp
            src/cpu/mc/transaction.cpp
            src/cpu/mc/volume_move.cpp
            src/cpu/mc/insert_delete_move.cpp
        )
    endif()

    if(ATV_BUILD_NEIGHBOR)
        list(APPEND ATOMIVERSE_SRCS src/cpu/neighbor/cell_list.cpp)
    endif()

    if(ATV_BUILD_MD)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/md/init.cpp
            src/cpu/md/atom_sort.cpp
            src/cpu/md/velocity_verlet.cpp
            src/cpu/md/nve_limit.cpp
            src/cpu/md/nose_hoover.cpp
            src/cpu/md/nose_hoover_npt.cpp
            src/cpu/md/fix_langevin.cpp
            src/cpu/md/fix_csvr.cpp
            src/cpu/md/fix_rattle.cpp
            src/cpu/md/barostat.cpp
            src/cpu/md/deform.cpp
        )
    endif()

    if(ATV_BUILD_VIB)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/vib/hessian.cpp
            src/cpu/vib/mass_weighting.cpp
            src/cpu/vib/hessian_diagonalizer.cpp
            src/cpu/vib/thermochem.cpp
            src/cpu/vib/frequency_report.cpp
            src/cpu/vib/normal_mode_writer.cpp
        )
    endif()

    if(ATV_BUILD_MINIMIZER)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/minimizer/lbfgs.cpp
            src/cpu/minimizer/lbfgs_history.cpp
            src/cpu/minimizer/cg.cpp
            src/cpu/minimizer/trajectory_writer.cpp
            src/cpu/minimizer/opt_log.cpp
            src/cpu/minimizer/final_geometry.cpp
        )
    endif()

    if(ATV_BUILD_PATH)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/replica/replica_expansion.cpp
            src/cpu/path/neb.cpp
            src/cpu/path/remd.cpp
            src/cpu/bias/distance_cv.cpp
            src/cpu/bias/harmonic_bias.cpp
        )
    endif()

    # CPU potentials — globbed so new pair/bond/angle/dihedral styles drop in.
    if(ATV_BUILD_POTENTIALS)
        file(GLOB CPU_POTENTIALS_SRCS CONFIGURE_DEPENDS
            "${CMAKE_SOURCE_DIR}/src/cpu/potentials/*.cpp"
        )
        list(APPEND ATOMIVERSE_SRCS ${CPU_POTENTIALS_SRCS})
    endif()

    if(ATV_BUILD_KSPACE)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/longrange/pme.cpp
            src/cpu/longrange/pppm.cpp) # true Hockney–Eastwood PPPM (r2c + HE green)

    endif()

    if(ATV_BUILD_CUDA)
        list(APPEND ATOMIVERSE_SRCS
            src/cuda/device/log_callback.cpp
            src/cuda/device/device_manager.cu
            src/cuda/comm/domain_decomposition.cu
            src/cuda/async/async_bridge.cu
            src/cuda/async/record_base.cu
            src/cuda/frame/transfer.cu
            src/cuda/frame/materialize.cu
            src/cuda/device/timer.cu
            src/cuda/driver/driver.cu
            src/cuda/comm/multi_gpu.cu
            src/cuda/observer/stress.cu
            src/cuda/scf/hamiltonian.cu
            src/cuda/scf/eigen_solver.cu
            src/cuda/scf/density_builder.cu
            src/cuda/scf/convergence_check.cu
            src/cuda/scf/scf_solver.cu
            src/cuda/scf/scf_evaluator.cu
            src/cuda/kspace/ewald.cu
        )
        if(ATV_BUILD_MD)
            list(APPEND ATOMIVERSE_SRCS
                src/cuda/md/velocity_verlet.cu
                src/cuda/md/nve_limit.cu
                src/cuda/md/fix_langevin.cu
                src/cuda/md/nose_hoover_npt.cu
                src/cuda/md/init.cu
                src/cuda/md/kinetic_energy.cu
                src/cuda/md/nose_hoover.cu
                src/cuda/md/deform.cu
                src/cuda/md/barostat.cu
            )
        endif()
        if(ATV_BUILD_NEIGHBOR)
            list(APPEND ATOMIVERSE_SRCS
                src/cuda/neighbor/neighbor_list.cu
                src/cuda/neighbor/sort_and_cluster.cu
                src/cuda/neighbor/grid.cu
                src/cuda/neighbor/pair_search.cu
                src/cuda/neighbor/fused_search.cu
                src/cuda/neighbor/nlist_direct.cu
                src/cuda/neighbor/atom_cell_sort.cu
                src/cuda/neighbor/setup.cu
                src/cuda/neighbor/export_pairs.cu
            )
        endif()
        if(ATV_BUILD_MINIMIZER)
            list(APPEND ATOMIVERSE_SRCS
                src/cuda/minimizer/sd.cu
                src/cuda/minimizer/cg.cu
                src/cuda/minimizer/lbfgs.cu
                src/cuda/minimizer/conv_check.cu
            )
        endif()
        if(ATV_BUILD_POTENTIALS)
            file(GLOB CUDA_POTENTIALS_SRCS CONFIGURE_DEPENDS
                "${CMAKE_SOURCE_DIR}/src/cuda/potentials/*.cu"
            )
            list(APPEND ATOMIVERSE_SRCS ${CUDA_POTENTIALS_SRCS})
        endif()
        if(ATV_BUILD_KSPACE)
            list(APPEND ATOMIVERSE_SRCS src/cuda/kspace/pme.cu)
        endif()
    else()
        # CUDA-free build still links against cuda::Timer (PIMPL in facade/driver.h).
        list(APPEND ATOMIVERSE_SRCS src/cuda/device/timer_cpu_stub.cpp)
    endif()

    add_library(atomiverse ${ATOMIVERSE_SRCS})
    target_compile_definitions(atomiverse PRIVATE
        ATV_MOLRS_CXX_API_VERSION=${ATV_MOLRS_CXX_API_VERSION}
        ATV_MOLRS_CXX_REQUIRED_CAPABILITIES=${ATV_MOLRS_CXX_REQUIRED_CAPABILITIES})
    atv_apply_product_module_defs(atomiverse)

    if(ATV_BUILD_HF)
        target_sources(atomiverse PRIVATE
            src/cpu/scf/basis_set.cpp
            src/cpu/scf/one_electron.cpp
            src/cpu/scf/two_electron.cpp
            src/cpu/scf/integrals.cpp
            src/cpu/scf/ecp_potential.cpp
            src/cpu/scf/ecp_integrals.cpp
            src/cpu/scf/native_integrals.cpp
            src/cpu/scf/rys_roots.cpp
            src/cpu/scf/rys_roots_iterative.cpp
            src/cpu/scf/rys_eri.cpp
            src/cpu/scf/shell_pair.cpp
            src/cpu/scf/operators.cpp
            src/cpu/observer/multipole.cpp
            src/cpu/observer/lowdin.cpp
            src/cpu/observer/esp_charges.cpp
            src/cpu/scf/internal/cart2sph.cpp
            src/cpu/scf/internal/diis_mixer.cpp
            src/cpu/scf/internal/ediis_adiis.cpp
            src/cpu/scf/internal/hueckel_guess.cpp
            src/cpu/scf/internal/sap_guess.cpp
            src/cpu/scf/internal/hamiltonian_builder.cpp
            src/cpu/scf/internal/density_fitting.cpp
            src/cpu/scf/internal/external_field.cpp
            src/cpu/scf/internal/level_shift.cpp
            src/cpu/scf/internal/point_charges.cpp
            src/cpu/scf/internal/pcm.cpp
            src/cpu/scf/internal/soscf.cpp
            src/cpu/scf/internal/x2c.cpp
            src/cpu/scf/stages/fock_stage.cpp
            src/cpu/scf/stages/mix_stage.cpp
            src/cpu/scf/stages/density_stage.cpp
            src/cpu/scf/mean_field_method.cpp
            src/cpu/scf/orbital_hessian.cpp
            src/cpu/scf/stability.cpp
            src/cpu/scf/tdhf.cpp
            src/cpu/scf/rhf.cpp
            src/cpu/scf/rohf.cpp
            src/cpu/scf/uhf.cpp
            src/cpu/scf/ghf.cpp
            src/cpu/scf/rks.cpp
            src/cpu/scf/uks.cpp
            src/cpu/scf/minao_data.cpp
            src/cpu/scf/minao.cpp
            src/cpu/scf/xc_functional.cpp
            src/cpu/scf/xc_gga.cpp
            src/cpu/scf/grid.cpp
            src/cpu/scf/integral_derivatives.cpp
            src/cpu/scf/cphf_pack.cpp
            src/cpu/scf/gradient.cpp
            src/cpu/scf/mp2.cpp
            src/cpu/scf/fci.cpp
            src/cpu/scf/libcint_bridge.cpp
            src/cpu/scf/aspc_predictor.cpp
            src/cpu/scf/orbital_info_observer.cpp
            src/cpu/recorder/cube.cpp
            src/cpu/recorder/density_cube_writer.cpp
            src/cpu/recorder/orbital_cube_writer.cpp
            src/cpu/observer/mulliken.cpp
            # eval_density_grid resolves shells via scf::build_shell_list
            # (basis_set.cpp above), so it is gated with the SCF sources —
            # same as the cube writers and mulliken. Its only consumer,
            # cuda::DensityRecorder, is #ifdef ATV_HAS_HF anyway.
            src/cpu/observer/density.cpp
        )
        if(ATV_BUILD_CUDA)
            target_sources(atomiverse PRIVATE
                src/cuda/scf/rhf_fock.cu
                src/cuda/scf/rhf_density.cu
                src/cuda/scf/device_integrals.cu
            )
        endif()
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_HF)
    endif()

    if(ATV_BUILD_GPW)
        if(NOT ATV_BUILD_HF)
            message(FATAL_ERROR
                "ATV_BUILD_GPW=ON requires ATV_BUILD_HF=ON (Gaussian SCF).")
        endif()
        if(NOT ATV_BUILD_PW)
            message(FATAL_ERROR
                "ATV_BUILD_GPW=ON requires ATV_BUILD_PW=ON (FFTW plane-wave stack).")
        endif()
        # GPW = Gaussian basis + density collocated on a uniform FFT grid +
        # analytic GTH pseudopotentials, targeting CP2K Quickstep parity.
        # Incremental build-out: the CP2K data-file readers come first
        # (dependency-free, std-only); grid collocation / GTH integrals / MT
        # Poisson / SCF land in later tasks.
        target_sources(atomiverse PRIVATE
            src/cpu/gpw/internal/local_pp.cpp
            src/cpu/gpw/gaussian_basis.cpp
            src/cpu/gpw/gth_potential.cpp
            src/cpu/gpw/internal/gth_integrals.cpp
            src/cpu/gpw/internal/diis.cpp
            src/cpu/gpw/internal/ot.cpp
            src/cpu/gpw/internal/collocate.cpp
            src/cpu/gpw/internal/multigrid.cpp
            src/cpu/gpw/internal/xc_grid.cpp
            # PADE (GTH Padé) LDA XC — CP2K-native functional used by the first
            # GPW examples; lives under cpu/xc/, built with the GPW module until
            # a dedicated ATV_BUILD_XC flag exists.
            src/cpu/xc/pade.cpp
            src/cpu/gpw/internal/one_electron.cpp
            # Grimme DFT-D3 dispersion (zero damping, PBE) — additive,
            # FFT-free; a standalone correction over the GPW DFT.
            src/cpu/gpw/internal/dftd3_params.cpp
            src/cpu/gpw/internal/dftd3.cpp
        )
        # FFT Poisson needs FFTW, supplied by the ATV_BUILD_PW block (fftw3_stub
        # is linked into atomiverse there, so the fftw3.h include propagates).
        if(ATV_BUILD_PW)
            target_sources(atomiverse PRIVATE
                src/cpu/gpw/internal/poisson_fft.cpp
                src/cpu/gpw/internal/hartree_grid.cpp
                src/cpu/gpw/internal/grid_fft.cpp
                src/cpu/gpw/internal/xc_grid_gga.cpp
                src/cpu/gpw/internal/mt_poisson.cpp
                src/cpu/gpw/gpw_solver.cpp
                src/cpu/gpw/internal/gpw_forces.cpp
                src/cpu/gpw/gpw_evaluator.cpp
            )
            target_compile_definitions(atomiverse PUBLIC ATV_GPW_HAS_FFT)
            # Optional FFTW3 OpenMP threading — the FFTs over the (large) density
            # grid dominate GPW's cost. fftw_plan_with_nthreads is global, so one
            # GPWSolver initialization threads every plan. Link libfftw3_omp if present.
            find_library(FFTW3_OMP_LIB fftw3_omp HINTS
                $ENV{FFTW3_ROOT}/lib $ENV{FFTW3_ROOT}/lib64
                $ENV{FFTW_ROOT}/lib $ENV{FFTW_ROOT}/lib64)
            if(FFTW3_OMP_LIB AND OpenMP_CXX_FOUND)
                target_link_libraries(atomiverse PUBLIC ${FFTW3_OMP_LIB} OpenMP::OpenMP_CXX)
                target_compile_definitions(atomiverse PUBLIC ATV_GPW_FFTW_THREADS)
                message(STATUS "GPW: FFTW3 OpenMP threading enabled (${FFTW3_OMP_LIB})")
            endif()
        endif()
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_GPW)
    endif()

    if(ATV_BUILD_PW)
        # $FFTW3_ROOT / $FFTW_ROOT come from the HPC buildenv Lmod modules
        # (no hardcoded paths); homebrew/usr-local hints cover macOS.
        find_library(FFTW3_LIB fftw3 HINTS
            $ENV{FFTW3_ROOT}/lib $ENV{FFTW3_ROOT}/lib64
            $ENV{FFTW_ROOT}/lib $ENV{FFTW_ROOT}/lib64
            /opt/homebrew/opt/fftw/lib /usr/local/lib)
        find_path(FFTW3_INC fftw3.h HINTS
            $ENV{FFTW3_ROOT}/include $ENV{FFTW_ROOT}/include
            /opt/homebrew/opt/fftw/include /usr/local/include)
        if(NOT FFTW3_LIB OR NOT FFTW3_INC)
            message(FATAL_ERROR "FFTW3 required for ATV_BUILD_PW but not found. "
                               "Install via: brew install fftw")
        endif()
        add_library(fftw3_stub INTERFACE)
        target_link_libraries(fftw3_stub INTERFACE ${FFTW3_LIB})
        target_include_directories(fftw3_stub INTERFACE ${FFTW3_INC})

        target_sources(atomiverse PRIVATE
            src/cpu/pw/internal/g_vector.cpp
            src/cpu/pw/internal/fft.cpp
            src/cpu/pw/internal/upf_reader.cpp
            src/cpu/pw/internal/kpoint.cpp
            src/cpu/pw/internal/ewald.cpp
            src/cpu/pw/internal/pw_setup.cpp
            src/cpu/pw/internal/pw_forces.cpp
            src/cpu/pw/internal/pw_stress.cpp
            src/cpu/pw/internal/pw_subspace.cpp
            src/cpu/pw/internal/poisson.cpp
            src/cpu/pw/internal/xc_builder.cpp
            src/cpu/pw/internal/veff_builder.cpp
            src/cpu/pw/internal/davidson.cpp
            src/cpu/pw/internal/density_builder.cpp
            src/cpu/pw/internal/convergence.cpp
            src/cpu/pw/pw_solver.cpp
            src/cpu/pw/pw_evaluator.cpp
        )
        if(NOT ATV_BUILD_HF)
            target_sources(atomiverse PRIVATE
                src/cpu/scf/xc_functional.cpp
                src/cpu/scf/xc_gga.cpp
            )
        endif()
        target_link_libraries(atomiverse PUBLIC fftw3_stub)
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_PW)

        # CUDA plane-wave backend — device-resident kernels mirroring the CPU
        # pw module (cuFFT FFT layer, G-space Hartree solver, sparse-G ↔ dense
        # grid bridge, LDA/Veff stages) plus the full pw SCF band solver
        # (cuFFT + cuBLAS Davidson). Only built when CUDA is also enabled.
        if(ATV_BUILD_CUDA)
            target_sources(atomiverse PRIVATE
                src/cuda/pw/internal/fft.cu
                src/cuda/pw/internal/poisson.cu
                src/cuda/pw/internal/grid_map.cu
                src/cuda/pw/internal/xc_lda.cu
                src/cuda/pw/internal/veff.cu
                src/cuda/pw/internal/device_solver.cu
                src/cuda/pw/pw_scf.cu
            )
            target_link_libraries(atomiverse PUBLIC CUDA::cufft)
        endif()
    endif()

    if(ATV_BUILD_KSPACE)
        # CPU PME needs FFTW3 (shared `fftw3_stub` with the PW module).
        if(NOT TARGET fftw3_stub)
            find_library(FFTW3_LIB fftw3 HINTS
                $ENV{FFTW3_ROOT}/lib $ENV{FFTW_ROOT}/lib
                /opt/homebrew/opt/fftw/lib /usr/local/lib)
            find_path(FFTW3_INC fftw3.h HINTS
                $ENV{FFTW3_ROOT}/include $ENV{FFTW_ROOT}/include
                /opt/homebrew/opt/fftw/include /usr/local/include)
            if(NOT FFTW3_LIB OR NOT FFTW3_INC)
                message(FATAL_ERROR "FFTW3 required for ATV_BUILD_KSPACE but not found.")
            endif()
            add_library(fftw3_stub INTERFACE)
            target_link_libraries(fftw3_stub INTERFACE ${FFTW3_LIB})
            target_include_directories(fftw3_stub INTERFACE ${FFTW3_INC})
        endif()
        target_link_libraries(atomiverse PUBLIC fftw3_stub)
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_KSPACE)
        # FFTW3 MPI (slab/pencil distributed FFT for multi-rank PPPM).
        if(ATV_USE_MPI)
            find_library(FFTW3_MPI_LIB fftw3_mpi HINTS
                $ENV{FFTW3_ROOT}/lib $ENV{FFTW3_ROOT}/lib64
                $ENV{FFTW_ROOT}/lib $ENV{FFTW_ROOT}/lib64)
            if(FFTW3_MPI_LIB)
                target_link_libraries(atomiverse PUBLIC ${FFTW3_MPI_LIB})
                target_compile_definitions(atomiverse PUBLIC ATV_HAS_FFTW_MPI)
                message(STATUS "KSPACE: FFTW3 MPI enabled (${FFTW3_MPI_LIB})")
            else()
                message(WARNING "KSPACE: libfftw3_mpi not found — multi-rank "
                                "PPPM falls back to replicated-grid Allreduce")
            endif()
        endif()
        # CUDA PME (cuFFT) — only when the CUDA backend is also enabled.
        if(ATV_BUILD_CUDA)
            target_link_libraries(atomiverse PUBLIC CUDA::cufft)
        endif()
    endif()

    if(ATV_BUILD_TORCH)
        # The torch backend is CUDA-only: every source hardcodes
        # `::torch::kCUDA` and test_transfer.cu calls cudaStreamCreate. With
        # CUDA disabled, `enable_language(CUDA)` never ran, and CMake then
        # SILENTLY DROPS every .cu source from the build — no warning, no
        # error. The library ends up with zero torch objects, every
        # test_torch_* binary links to an empty gtest_main, runs "0 tests from
        # 0 test suites", exits 0, and ctest reports a full green. A build
        # configuration that cannot possibly work must fail at configure time,
        # not hand back a passing test suite that tested nothing.
        if(NOT ATV_BUILD_CUDA)
            message(FATAL_ERROR
                "ATV_BUILD_TORCH=ON requires ATV_BUILD_CUDA=ON: the torch "
                "backend is a CUDA-only differentiable layer (it hardcodes "
                "torch::kCUDA and calls the CUDA runtime), and with the CUDA "
                "language disabled CMake silently excludes every .cu source — "
                "producing a library with no torch code and test binaries with "
                "no tests, which then report PASS. Build it on the GPU node "
                "with a CUDA-enabled LibTorch.")
        endif()
        find_package(Torch REQUIRED)
        target_sources(atomiverse PRIVATE
            src/torch/frame/transfer.cu
            src/torch/frame/materialize.cu
            src/torch/device/stream_guard.cu
            src/torch/potentials/lj.cu
            src/torch/neighbor/neighbor_list.cu
            src/torch/potentials/harmonic_bond.cu
        )
        target_link_libraries(atomiverse PUBLIC ${TORCH_LIBRARIES})
        target_include_directories(atomiverse PUBLIC ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_TORCH)
    endif()

    # Optional: fetch libcint as a test-only ground-truth reference
    if(ATV_BUILD_LIBCINT)
        FetchContent_Declare(
            libcint
            GIT_REPOSITORY https://github.com/sunqm/libcint.git
            GIT_TAG v6.1.2
        )
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(libcint)
        target_include_directories(cint PUBLIC
            ${libcint_BINARY_DIR}/include
            ${libcint_SOURCE_DIR}/include
        )
        target_link_libraries(atomiverse PUBLIC cint)
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_LIBCINT)
    endif()

    # Optional: libxc as test-only oracle (NOT a runtime dependency).
    # libxc is found here and consumed only by test targets under
    # tests/cpu/scf/oracle/ (see sub-spec 02). libatomiverse never
    # links against libxc and exposes no libxc-related compile macro.
    # After native XC parity is locked in (sub-spec 03 of the
    # libxc-xc-fix chain), this option and the oracle directory are
    # deletable in a single commit without touching libatomiverse.
    if(ATV_BUILD_XC_ORACLE)
        # No version pinned in find_package -- libxc's ConfigVersion uses
        # SameMajorVersion compat, so requesting "6" rejects 7+ and vice
        # versa. Accept any installed major; verify minimum below.
        find_package(Libxc CONFIG QUIET)
        if(NOT Libxc_FOUND OR Libxc_VERSION VERSION_LESS 6.0)
            message(FATAL_ERROR
                "\nATV_BUILD_XC_ORACLE=ON but libxc >= 6 was not found "
                "(found: ${Libxc_VERSION}).\n"
                "Install libxc 6+ on the build host:\n"
                "  macOS:  brew install libxc\n"
                "  Linux:  apt install libxc-dev\n"
                "No FetchContent fallback is provided -- libxc is a test-only "
                "oracle and must be installed system-wide to exercise the XC "
                "parity tests.\n"
            )
        endif()
        message(STATUS "libxc test oracle enabled: ${Libxc_VERSION} (${Libxc_DIR})")
    endif()

    if(ATV_BUILD_CUDA)
        find_package(CUDAToolkit REQUIRED)
    endif()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    if(ATV_BUILD_CUDA)
        target_include_directories(atomiverse PRIVATE src/cuda/neighbor)
    endif()
    target_include_directories(atomiverse PUBLIC include)
    target_link_libraries(atomiverse PUBLIC xtensor xtensor-blas xsimd spdlog::spdlog ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    # xsimd is linked for direct use in native_integrals.cpp ERI hot loops.
    # XTENSOR_USE_XSIMD is NOT set globally: enabling it triggers SIMD-backed
    # xtensor storage which breaks reshape_view in xc_grid_engine.h.
    # Native CPU tuning: unlock NEON on ARM64, AVX2/AVX-512 on x86-64.
    # Language-guarded to prevent NVCC from seeing -march=native on CUDA files.
    target_compile_options(atomiverse PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>)

    # ── Phase 5 Track B.8 — `-ffast-math` policy ─────────────────────
    #
    # `-ffast-math` (and its weaker cousin `-funsafe-math-optimizations`)
    # reorder floating-point additions and break IEEE-strict semantics.
    # That is acceptable in pure force-evaluator translation units (LJ,
    # bonded kernels, ERI inner loops) where reductions over many pair
    # interactions hide reorderings inside accumulator noise, but it
    # **must not be applied** to integrators or thermostats (Velocity
    # Verlet, Nose-Hoover) because:
    #
    #   * Velocity Verlet is **time-reversible** by construction — the
    #     update sequence v(t+dt/2) = v(t) + a(t)·dt/2 ; r(t+dt) = r(t)
    #     + v(t+dt/2)·dt ; v(t+dt) = v(t+dt/2) + a(t+dt)·dt/2 must
    #     produce, after FP rounding, the *exact* state when run forward
    #     and then backward. `-ffast-math` reorders the multiply-add
    #     into a fused-multiply-add or a different summation tree,
    #     destroying that property.
    #   * Nose-Hoover thermostat updates `xi += dt/2 · (2·KE -
    #     n_dof·k_B·T_target)/Q` and then rescales velocities by
    #     `exp(-dt/2 · xi)`. Reordering moves the `exp(...)` argument
    #     and silently changes the conserved Hamiltonian.
    #
    # **Policy** (keep this block in sync with
    # `.claude/specs/parallel-md-engine.md` Architectural rules and
    # `.claude/specs/parallel-md-engine.acceptance.md` ac-021):
    #
    #   - `atomiverse` library default = strict FP (no `-ffast-math`).
    #   - Per-file opt-in for FF kernels via
    #     `set_source_files_properties(<src> PROPERTIES COMPILE_FLAGS
    #     "-ffast-math")` after measuring a real benefit AND verifying
    #     bit-identical NVE energy (≤1e-12 relative) on the targeted
    #     test_lj_nve / test_pe_small_box test.
    #   - Integrator translation units (`src/cpu/md/velocity_verlet.cpp`,
    #     `src/cpu/md/nose_hoover.cpp`, `src/cpu/md/init.cpp`) MUST NOT
    #     be in any per-file opt-in list. Document any opt-in here and
    #     in the relevant run's notes under `benchmarks/md/<sys>/performance/runs/`
    #     with the exact compiler + flag string used to produce the
    #     reported `bench_md_step` numbers (ac-022).
    #
    # No file currently opts in. When the first FF kernel does, add
    # `set_source_files_properties(... PROPERTIES COMPILE_FLAGS ...)`
    # immediately below this comment block.
    # Apple Clang 21+ enforces consteval strictly; spdlog 1.15 fmt bundled lib
    # triggers a false consteval-in-non-constexpr-context error. Disable it.
    if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_definitions(atomiverse PUBLIC FMT_USE_CONSTEVAL=0)
    endif()
    if(ATV_BUILD_CUDA)
        target_link_libraries(atomiverse PUBLIC CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
        set_target_properties(atomiverse PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_CUDA)
    endif()
    target_compile_definitions(atomiverse PUBLIC ATV_CUDA_REAL_TYPE=${ATV_CUDA_REAL} ATV_CUDA_WORK_TYPE=${ATV_CUDA_WORK} ATV_CUDA_PREC_TYPE=${ATV_CUDA_PREC})

    # CPU/shared tiered precision (core/precision.h). Accumulators are always
    # double and intentionally not configurable here.
    target_compile_definitions(atomiverse PUBLIC
        ATV_REAL_STORAGE_TYPE=${ATV_REAL_STORAGE}
        ATV_REAL_WORK_TYPE=${ATV_REAL_WORK})
    if(ATV_ID_SIZE STREQUAL "bigbig")
        target_compile_definitions(atomiverse PUBLIC ATV_ID_BIGBIG)
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(atomiverse PUBLIC OpenMP::OpenMP_CXX)
    endif()

    if(ATV_USE_MPI)
        target_link_libraries(atomiverse PUBLIC MPI::MPI_CXX)
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_MPI)
    endif()

    # ── molrs (hard dependency via CXX FFI) ──
    FetchContent_Declare(Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion
        GIT_TAG v0.6.1
    )
    FetchContent_MakeAvailable(Corrosion)
    corrosion_import_crate(
        MANIFEST_PATH "${ATV_MOLRS_PATH}/Cargo.toml"
        CRATES molcrafts-molrs-cxxapi
    )
    # molrs-core hardcodes F = f64.  Bridge always uses double.
    # CPU backend uses double; zero-copy across the FFI boundary.
    # Package is `molcrafts-molrs-cxxapi` in Cargo.toml, but `[lib] name` is
    # `molrs_cxxapi` -- corrosion exposes the target under the lib name.
    corrosion_add_cxxbridge(molrs_cxx
        CRATE molrs_cxxapi
        FILES bridge.rs
    )

    # corrosion v0.6.1's cxxbridge target carries the generated cxx runtime
    # header `rust/cxx.h` as a source without the GENERATED property. On a
    # fresh configure (no cached build tree) the file does not exist yet, so
    # CMake's generate step aborts in every directory that links `molrs_cxx`
    # transitively (e.g. tests/). Mark the generated cxxbridge outputs
    # GENERATED so CMake trusts the build step to produce them. CMP0118 (NEW)
    # makes the GENERATED property visible across directories.
    if(POLICY CMP0118)
        cmake_policy(SET CMP0118 NEW)
    endif()
    set(_molrs_cxx_gen_dir
        "${CMAKE_BINARY_DIR}/corrosion_generated/cxxbridge/molrs_cxx")
    set_source_files_properties(
        "${_molrs_cxx_gen_dir}/include/rust/cxx.h"
        "${_molrs_cxx_gen_dir}/include/molrs_cxx/src/bridge.rs.h"
        "${_molrs_cxx_gen_dir}/src/bridge.rs.cc"
        PROPERTIES GENERATED TRUE)

    target_link_libraries(atomiverse PUBLIC molrs_cxx)

    if(ATV_USE_MATPLOT)
        FetchContent_Declare(
            matplotplusplus
            GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
            GIT_TAG v1.2.1
        )
        FetchContent_MakeAvailable(matplotplusplus)
    endif()

    # ── Python bindings (nanobind) ──
    # When built via scikit-build-core (pip install), nanobind is installed into
    # the build venv before CMake runs, so find_package(nanobind) resolves via
    # `python -m nanobind --cmake_dir`.  For standalone cmake builds, nanobind
    # must be installed in the active Python environment (pip install nanobind).
    if(ATV_BUILD_PYTHON)
        find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

        # Locate nanobind from the active Python environment
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE NB_CMAKE_DIR
            RESULT_VARIABLE NB_RESULT
        )
        if(NOT NB_RESULT EQUAL 0)
            message(FATAL_ERROR "nanobind not found in Python environment. "
                "Run: pip install nanobind>=2.4")
        endif()
        list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
        find_package(nanobind CONFIG REQUIRED)

        set(_ATV_NB_SRCS python/src/bindings/bind.cpp)
        if(ATV_BUILD_MD)
            list(APPEND _ATV_NB_SRCS python/src/bindings/bind_md.cpp)
        endif()
        if(ATV_BUILD_MINIMIZER)
            list(APPEND _ATV_NB_SRCS python/src/bindings/bind_minimizer.cpp)
        endif()
        if(ATV_BUILD_PW)
            list(APPEND _ATV_NB_SRCS python/src/bindings/bind_pw.cpp)
        endif()
        if(ATV_BUILD_SEMIEMPIRICAL)
            list(APPEND _ATV_NB_SRCS python/src/bindings/bind_semiempirical.cpp)
        endif()
        nanobind_add_module(_atomiverse ${_ATV_NB_SRCS})
        target_link_libraries(_atomiverse PRIVATE atomiverse)
        target_include_directories(_atomiverse PRIVATE include python/src/bindings)
        if(ATV_BUILD_PW)
            target_link_libraries(_atomiverse PRIVATE ${FFTW3_LIB})
        endif()
        install(TARGETS _atomiverse LIBRARY DESTINATION atomiverse)
    endif()

    # ── Plain `cmake --install` layout (wheel builds carry none of this;
    #    scikit-build-core owns their install tree) ──
    if(NOT SKBUILD)
        include(GNUInstallDirs)
        install(TARGETS atomiverse
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
        install(DIRECTORY include/atomiverse TYPE INCLUDE)
    endif()

    # ── MPI launcher executable (rank-0 embedded CPython) ──
    if(ATV_BUILD_LAUNCHER)
        if(NOT ATV_USE_MPI)
            message(FATAL_ERROR
                "ATV_BUILD_LAUNCHER=ON requires ATV_USE_MPI=ON: the launcher "
                "is the MPI process model (rank 0 embeds Python, workers run "
                "the task protocol). Configure with -DATV_USE_MPI=ON or drop "
                "-DATV_BUILD_LAUNCHER.")
        endif()
        if(NOT ATV_BUILD_PYTHON)
            message(FATAL_ERROR
                "ATV_BUILD_LAUNCHER=ON requires ATV_BUILD_PYTHON=ON: the "
                "launcher hosts the Python bindings built from this tree. "
                "Configure with -DATV_BUILD_PYTHON=ON or drop "
                "-DATV_BUILD_LAUNCHER.")
        endif()
        add_subdirectory(app/atomiverse)
    endif()

    # ctest is used by both the unit-test suite and the benchmark validation
    # binaries (relocated end-to-end / parity checks); enable it for either.
    if(ATV_BUILD_TESTS OR ATV_BUILD_BENCHMARKS)
        enable_testing()
    endif()

    if(ATV_BUILD_TESTS)
        # Surface the build option as a compile definition so library code
        # that wants to enable test-only hooks (e.g. the
        # cuda::stream_synchronize counter in src/cuda/async/record_base.cu
        # used by msu-pe-cuda-components-07's hot-path zero-sync gate) can
        # compile in those hooks. PUBLIC so test executables that link
        # atomiverse pick up the same definition.
        target_compile_definitions(atomiverse PUBLIC ATV_BUILD_TESTS)
        add_subdirectory(tests)
    endif()

    # End-to-end textbook regressions. Deliberately NOT registered with ctest —
    # tests/ is unit-only and "if it is registered, it runs" (see CLAUDE.md).
    if(ATV_BUILD_REGRESSIONS)
        add_subdirectory(regressions)
    endif()

    # Benchmarks tree removed (rebuild later under a fresh layout).
    # option ATV_BUILD_BENCHMARKS kept OFF-only so existing CI scripts don't break.
    if(ATV_BUILD_BENCHMARKS)
        message(WARNING "ATV_BUILD_BENCHMARKS=ON but benchmarks/ was removed; skipping.")
    endif()

    if(ATV_BUILD_EXAMPLES)
        add_subdirectory(examples)
    endif()

    option(ATV_BUILD_DEBUG_AMORPHOUS
        "Build debug_amorphous LAMMPS-comparison runner" OFF)
    if(ATV_BUILD_DEBUG_AMORPHOUS)
        add_subdirectory(tools/debug_amorphous)
    endif()
    option(ATV_BUILD_BENCH_GAFF
        "Build tools/bench_gaff_lammps (GAFF vs LAMMPS timing)" OFF)
    if(ATV_BUILD_BENCH_GAFF)
        add_subdirectory(tools/bench_gaff_lammps)
    endif()
endif()

if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating Doxygen documentation"
        VERBATIM
    )

    add_custom_target(docs-open
        COMMAND ${CMAKE_COMMAND}
                -DINPUT_FILE=${CMAKE_SOURCE_DIR}/docs/html/index.html
                -P ${CMAKE_SOURCE_DIR}/cmake/utils/open_generated_file.cmake
        DEPENDS docs
        COMMENT "Opening generated documentation"
        VERBATIM
    )
else()
    message(STATUS "Doxygen not found; docs targets are disabled.")
endif()
