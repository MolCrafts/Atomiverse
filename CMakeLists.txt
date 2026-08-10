cmake_minimum_required(VERSION 3.18)
option(ATV_DOCS_ONLY "Configure documentation targets without the C++/CUDA toolchain" OFF)

# Single source of truth for the release version. python/pyproject.toml,
# python/src/atomiverse/__init__.py and Doxyfile must match it; the Python
# native-install CLI additionally requires the git tag to be v${VERSION}.
if(ATV_DOCS_ONLY)
    project(Atomiverse VERSION 0.1.0 LANGUAGES NONE)
else()
    project(Atomiverse VERSION 0.1.0 LANGUAGES CXX)
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

    # C++20 is not optional: xtensor 0.27.1 uses UNGUARDED C++20 `requires`
    # clauses (e.g. xtensor/views/xview.hpp:1511) and declares
    # `target_compile_features(xtensor INTERFACE cxx_std_20)`, which CMake
    # propagates to every target linking xtensor. Before this line said 20 the
    # `atomiverse` target was ALREADY compiling at -std=gnu++20 via that
    # interface requirement while this variable claimed 17 — declaring 20 makes
    # the build honest and also lifts the non-xtensor targets (molrs_cxx) that
    # were still getting gnu++17.
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    option(ATV_BUILD_CUDA "Build CUDA backend (requires CUDA toolkit)" ON)
    if(ATV_BUILD_CUDA)
        # Prefer an explicit arch when building for GH200 (sm_90). `native`
        # only works when configure+compile run on the target GPU node; a
        # login-node or cross configure silently omits ISA 90 and produces
        # runtime "No device code available for GPU ISA 90" warnings.
        # This MUST precede enable_language(CUDA): compiler detection caches
        # the nvcc default (sm_75) into CMAKE_CUDA_ARCHITECTURES, after which
        # `NOT DEFINED` can never fire — sm_75 cubins on an sm_90 GH200 then
        # pay a >10 s driver PTX JIT on the first kernel launch, which is
        # enough to blow the 10 s ctest budget of the first test binary.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            if(DEFINED ENV{ATV_CUDA_ARCHITECTURES} AND NOT "$ENV{ATV_CUDA_ARCHITECTURES}" STREQUAL "")
                set(CMAKE_CUDA_ARCHITECTURES "$ENV{ATV_CUDA_ARCHITECTURES}")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "90")
            endif()
        endif()
        enable_language(CUDA)
        # CUDA TUs that transitively include xtensor (via cpu/frame.h) need
        # C++20: xtensor 0.27.1 uses `<concepts>` / `std::input_iterator`
        # on toolchains where `<concepts>` is available (GCC 13), which
        # nvcc cannot parse at -std=c++17. The CPU side is C++20 too (above).
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    # CUDA precision — aligned with LAMMPS Kokkos `KOKKOS_PREC` (one knob, three
    # modes). LAMMPS uses two types only: KK_FLOAT (storage + force math) and
    # KK_ACC_FLOAT (force/energy/virial accumulation). We mirror that as
    # real_t/work_t (= KK_FLOAT) and prec_t (= KK_ACC_FLOAT). There is no free
    # independent "work" axis — WORK always tracks REAL.
    #   double (default, KK default): real=work=prec=double
    #   mixed:                        real=work=float, prec=double
    #   single:                       real=work=prec=float
    set(ATV_CUDA_PREC_MODE "double" CACHE STRING
        "CUDA precision mode (LAMMPS KOKKOS_PREC): double | mixed | single")
    set_property(CACHE ATV_CUDA_PREC_MODE PROPERTY STRINGS double mixed single)
    string(TOLOWER "${ATV_CUDA_PREC_MODE}" _atv_cuda_prec_mode)
    if(_atv_cuda_prec_mode STREQUAL "double")
        set(ATV_CUDA_REAL "double")
        set(ATV_CUDA_WORK "double")
        set(ATV_CUDA_PREC "double")
    elseif(_atv_cuda_prec_mode STREQUAL "mixed")
        set(ATV_CUDA_REAL "float")
        set(ATV_CUDA_WORK "float")
        set(ATV_CUDA_PREC "double")
    elseif(_atv_cuda_prec_mode STREQUAL "single")
        set(ATV_CUDA_REAL "float")
        set(ATV_CUDA_WORK "float")
        set(ATV_CUDA_PREC "float")
    else()
        message(FATAL_ERROR
            "ATV_CUDA_PREC_MODE must be double|mixed|single (got '${ATV_CUDA_PREC_MODE}')")
    endif()
    message(STATUS
        "CUDA precision mode=${_atv_cuda_prec_mode} "
        "(real_t=${ATV_CUDA_REAL} work_t=${ATV_CUDA_WORK} prec_t=${ATV_CUDA_PREC}; "
        "mirrors LMP KK_FLOAT/KK_ACC_FLOAT)")

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
    option(ATV_BUILD_XC_ORACLE
           "Build libxc-backed XC oracle (test-only scaffolding; deletable after native parity is locked in)"
           OFF)
    # molrs path (optional product — gated by ATV_BUILD_MOLRS after module DAG).
    # Submodule pin at ./molrs; -DATV_MOLRS_PATH only for local experiments.
    set(ATV_MOLRS_PATH "${CMAKE_SOURCE_DIR}/molrs" CACHE PATH
        "Path to molrs source tree (default: git submodule ./molrs)")
    include("${CMAKE_SOURCE_DIR}/cmake/MolrsContract.cmake")
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

    # Product-module gates (MD/MC/… + ANALYSIS/IO/MOLRS DAG).
    # Must run after ATV_BUILD_KSPACE / HF / GPW / PYTHON are declared.
    include("${CMAKE_SOURCE_DIR}/cmake/AtvProductModules.cmake")

    # molrs contract only when the DAG left MOLRS on.
    set(ATV_MOLRS_HAS_WHAM OFF)
    if(ATV_BUILD_MOLRS)
        if(NOT EXISTS "${ATV_MOLRS_PATH}/Cargo.toml")
            message(FATAL_ERROR
                "ATV_BUILD_MOLRS=ON but molrs not found at "
                "ATV_MOLRS_PATH='${ATV_MOLRS_PATH}'.\n"
                "  Init: git submodule update --init --recursive molrs\n"
                "  Or turn off ANALYSIS/IO/PYTHON/GPW/HF/SEMIEMPIRICAL "
                "(those products enable molrs).")
        endif()
        if(NOT ATV_MOLRS_PATH STREQUAL "${CMAKE_SOURCE_DIR}/molrs")
            message(WARNING
                "ATV_MOLRS_PATH='${ATV_MOLRS_PATH}' overrides the ./molrs submodule.")
        endif()
        atv_validate_molrs_source_contract("${ATV_MOLRS_PATH}")
        atv_detect_molrs_wham("${ATV_MOLRS_PATH}" ATV_MOLRS_HAS_WHAM)
        if(ATV_MOLRS_HAS_WHAM)
            message(STATUS "molrs WhamCompute: available")
        else()
            message(STATUS "molrs WhamCompute: not in this pin (wham skipped)")
        endif()
    else()
        message(STATUS "molrs CXX: OFF (no Rust link; pure engine build)")
    endif()

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
        GIT_TAG 14.3.0
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
        # molrs_contract.cpp — only with ATV_BUILD_MOLRS (below)
        # Backend-agnostic longrange operator math (Ewald/PME headers are
        # header-only; FMM needs translation-operator / tree sources). Always
        # compiled — not gated by ATV_BUILD_*.
        src/cpu/longrange/multipole_expansion.cpp
        src/cpu/longrange/fmm_operators.cpp
        src/cpu/longrange/fmm_tree.cpp
        src/cpu/scf/solve.cpp
        # Shared dense linear algebra. S-metric orbital primitives (projected
        # band gradient, tangent projection, Löwdin re-orthonormalization) are
        # used by the GPW OT minimizer and by the electronic-state propagators,
        # so they are never gated on a method flag.
        src/cpu/linalg/orbital_tangent.cpp
        # Niklasson dissipative XL recursion (cpu/electronic/xl_bomd.cpp):
        # depends only on cpu/linalg header types. Consumed by Gaussian SCF /
        # GPW (ATV_BUILD_HF) *and* by the classical extended-Lagrangian PIM
        # dipoles (cpu/potentials/pim_xl_dipole.*, MD-only builds), so it is
        # never method-gated.
        src/cpu/electronic/xl_bomd.cpp
        src/cpu/io/lammps_data.cpp
        # molrs-backed IO / ANALYSIS / exchange — gated below (ATV_BUILD_*)
        src/cpu/record/thermo.cpp
        src/cpu/record/timing_report.cpp
        # CPU runtime infrastructure (Runtime + composed sub-runtimes)
        src/cpu/frame/frame.cpp
        src/cpu/driver/runtime.cpp
        src/cpu/communication/domain_runtime.cpp
        src/cpu/device/device_runtime.cpp
        src/cpu/asynchronous/analysis_runtime.cpp
        src/cpu/driver/dof.cpp
        src/cpu/frame/box.cpp
        src/cpu/component/zero.cpp
        src/cpu/driver/driver.cpp
        # Multiscale shared machinery (QmEngine slot; used by SE and HF couplings)
        src/cpu/multiscale/internal/qm_subsystem.cpp
        src/cpu/multiscale/internal/qm_mm_vdw.cpp
        # CPU task infrastructure (TaskSpec + codec + materializer + protocol)
        src/cpu/task/task_spec.cpp
        src/cpu/task/task_codec.cpp
        src/cpu/task/task_materializer.cpp
        src/cpu/task/task_protocol.cpp
        src/cpu/task/launcher_gate.cpp
        src/cpu/asynchronous/analysis_executor.cpp
        src/cpu/asynchronous/async_component_base.cpp
        src/cpu/asynchronous/snapshot_publisher.cpp
        src/cpu/observe/virial_stress.cpp
        src/cpu/observe/widom.cpp
        src/cpu/observe/electrode_charge.cpp
        src/cpu/observe/classical_polarization.cpp
        src/cpu/observe/capacitance_charge_fluctuation.cpp
        src/cpu/observe/potential_profile.cpp
        src/cpu/observe/species_density_profile.cpp
        src/cpu/observe/charge_autocorr.cpp
        src/cpu/observe/elastance_observer.cpp
        src/cpu/observe/voronoi_volume.cpp

        # CPU MPI communication subsystem (LAMMPS-shaped, ATV::cpu::comm::*)
        src/cpu/communication/atom_pack.cpp
        src/cpu/communication/bcast_frame.cpp
        src/cpu/communication/halo_exchange.cpp
        src/cpu/communication/atom_migration.cpp
        src/cpu/communication/domain.cpp
        src/cpu/communication/reduce.cpp
        src/cpu/communication/gather.cpp
        src/cpu/communication/gather_for_io.cpp
        src/cpu/communication/rcb.cpp
        # Direct Ewald always (no external deps); PME gated by ATV_BUILD_KSPACE below
        src/cpu/longrange/ewald.cpp
    )

    # ── molrs surfaces (IO / ANALYSIS / exchange / Element tables) ─────────
    if(ATV_BUILD_MOLRS)
        list(APPEND ATOMIVERSE_SRCS
            src/core/molrs_contract.cpp
            src/cpu/frame/molrs_exchange.cpp
        )
    endif()
    # Products own features; MOLRS is already derived (AtvProductModules).
    if(ATV_BUILD_IO)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/io/frame_writer.cpp
            src/cpu/io/extxyz.cpp
            src/cpu/record/xyz.cpp
        )
    endif()
    if(ATV_BUILD_ANALYSIS)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/observe/rdf.cpp
            src/cpu/observe/rdf_curve.cpp
            src/cpu/observe/msd.cpp
            src/cpu/observe/vacf.cpp
            src/cpu/observe/trajectory_metrics.cpp
        )
    endif()

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
            src/cpu/semiempirical/am1_solver.cpp
            src/cpu/semiempirical/rm1_solver.cpp
            src/cpu/semiempirical/am1_evaluator.cpp
            src/cpu/semiempirical/rm1_evaluator.cpp
            src/cpu/semiempirical/dftb2.cpp
            src/cpu/semiempirical/dftb0.cpp
            src/cpu/semiempirical/dftb3.cpp
            src/cpu/semiempirical/dftb3_scc.cpp
            src/cpu/semiempirical/mio_sk_block.cpp
            src/cpu/semiempirical/scc_mixer.cpp

            src/cpu/multiscale/qmmm_mechanical.cpp
            src/cpu/multiscale/qmmm_oniom.cpp
            src/cpu/multiscale/internal/classical_nonbonded.cpp
        )
        if(ATV_BUILD_MOLRS)
            list(APPEND ATOMIVERSE_SRCS
                src/cpu/semiempirical/am1_bcc_charge_assigner.cpp)
        endif()
        if(ATV_BUILD_HF)
            list(APPEND ATOMIVERSE_SRCS
                src/cpu/multiscale/qmmm_polarizable.cpp
            )
        endif()
        # GFN2 (optional while sources land on the branch)
        if(EXISTS "${CMAKE_SOURCE_DIR}/src/cpu/semiempirical/gfn2.cpp")
            list(APPEND ATOMIVERSE_SRCS
                src/cpu/semiempirical/gfn2.cpp
                src/cpu/semiempirical/gfn2_xtb_kernel.cpp)
        endif()
        # Grimme D3 engine for DFTB2::d3 (and peers) under SE-only cmake.
        # When HF is on, the same sources are listed under ATV_BUILD_HF — do
        # not double-compile.
        if(NOT ATV_BUILD_HF)
            list(APPEND ATOMIVERSE_SRCS
                src/cpu/gpw/internal/dftd3_params.cpp
                src/cpu/gpw/internal/dftd3.cpp
            )
        endif()
    endif()

    if(ATV_BUILD_MC)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/montecarlo/translate_move.cpp
            src/cpu/montecarlo/transaction.cpp
            src/cpu/montecarlo/volume_move.cpp
            src/cpu/montecarlo/insert_delete_move.cpp
        )
    endif()

    if(ATV_BUILD_NEIGHBOR)
        list(APPEND ATOMIVERSE_SRCS src/cpu/neighbor/cell_list.cpp)
    endif()

    if(ATV_BUILD_MD)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/dynamics/init.cpp
            src/cpu/dynamics/atom_sort.cpp
            src/cpu/dynamics/cpmd_conserved.cpp
            src/cpu/dynamics/velocity_verlet.cpp
            src/cpu/dynamics/nve_limit.cpp
            src/cpu/dynamics/nose_hoover.cpp
            src/cpu/dynamics/nose_hoover_npt.cpp
            src/cpu/dynamics/parrinello_rahman_npt.cpp
            src/cpu/dynamics/nph_integrator.cpp
            src/cpu/dynamics/fix_langevin.cpp
            src/cpu/dynamics/fix_efield.cpp
            src/cpu/dynamics/fix_addforce.cpp
            src/cpu/dynamics/fix_move.cpp
            src/cpu/dynamics/fix_dfield.cpp
            src/cpu/dynamics/fix_oscillating_efield.cpp
            src/cpu/dynamics/fix_wall_lj1043.cpp
            src/cpu/dynamics/fix_csvr.cpp
            src/cpu/dynamics/fix_rattle.cpp
            src/cpu/dynamics/fix_settle.cpp
            src/cpu/dynamics/virtual_sites.cpp
            src/cpu/dynamics/const_pot_bo.cpp
            src/cpu/dynamics/const_pot_cg.cpp
            src/cpu/dynamics/qeq.cpp
            src/cpu/dynamics/const_pot_maze.cpp
            src/cpu/dynamics/dipole_electrode_joint.cpp
            src/cpu/dynamics/fix_freeze.cpp
            src/cpu/dynamics/piston_electrode.cpp
            src/cpu/dynamics/const_potential.cpp
            src/cpu/dynamics/brownian.cpp
            src/cpu/dynamics/barostat.cpp
            src/cpu/dynamics/deform.cpp
            # Electrochemical electrode bookkeeping (MetalWalls E1/E2 map)
            src/cpu/electrodes/charging_work.cpp
            src/cpu/electrodes/charge_aspc.cpp
        )
    endif()

    if(ATV_BUILD_VIB)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/vibration/hessian.cpp
            src/cpu/vibration/mass_weighting.cpp
            src/cpu/vibration/hessian_diagonalizer.cpp
            src/cpu/vibration/thermochem.cpp
            src/cpu/vibration/frequency_report.cpp
            src/cpu/vibration/normal_mode_writer.cpp
            src/cpu/vibration/supercell.cpp
            src/cpu/vibration/phonon.cpp
            src/cpu/vibration/phonon_diagonalizer.cpp
            src/cpu/vibration/phonon_dispersion.cpp
        )
    endif()

    if(ATV_BUILD_MINIMIZER)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/minimize/lbfgs.cpp
            src/cpu/minimize/lbfgs_history.cpp
            src/cpu/minimize/cg.cpp
            src/cpu/minimize/fire.cpp
            src/cpu/minimize/cell_relax.cpp
            src/cpu/minimize/anisotropic_cell_relax.cpp
            src/cpu/minimize/triclinic_cell_relax.cpp
            src/cpu/minimize/joint_cell_relax.cpp
            src/cpu/minimize/trajectory_writer.cpp
            src/cpu/minimize/opt_log.cpp
            src/cpu/minimize/final_geometry.cpp
        )
    endif()

    if(ATV_BUILD_PATH)
        list(APPEND ATOMIVERSE_SRCS
            src/cpu/replica/replica_expansion.cpp
            src/cpu/path/neb.cpp
            src/cpu/path/remd.cpp
            src/cpu/path/hremd.cpp
            src/cpu/path/pimd.cpp
            src/cpu/path/ring_polymer_spring.cpp
            src/cpu/path/dimer.cpp
            src/cpu/path/string_method.cpp
            src/cpu/path/irc.cpp
            src/cpu/bias/distance_cv.cpp
            src/cpu/bias/angle_cv.cpp
            src/cpu/bias/dihedral_cv.cpp
            src/cpu/bias/harmonic_bias.cpp
            src/cpu/bias/metadynamics_1d.cpp
            src/cpu/bias/metadynamics.cpp
        )
    endif()

    # CPU potentials — globbed so new pair/bond/angle/dihedral styles drop in.
    # EAM/SW table lookups use molrs::Element — omit when MOLRS is off.
    if(ATV_BUILD_POTENTIALS)
        file(GLOB CPU_POTENTIALS_SRCS CONFIGURE_DEPENDS
            "${CMAKE_SOURCE_DIR}/src/cpu/potentials/*.cpp"
        )
        if(NOT ATV_BUILD_MOLRS)
            list(FILTER CPU_POTENTIALS_SRCS EXCLUDE REGEX ".*/eam\\.cpp$")
            list(FILTER CPU_POTENTIALS_SRCS EXCLUDE REGEX ".*/sw\\.cpp$")
        endif()
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
            src/cuda/communication/domain_decomposition.cu
            src/cuda/asynchronous/async_bridge.cu
            src/cuda/asynchronous/record_base.cu
            src/cuda/frame/transfer.cu
            src/cuda/device/timer.cu
            src/cuda/driver/driver.cu
            src/cuda/observe/stress.cu
            src/cuda/scf/hamiltonian.cu
            src/cuda/scf/eigen_solver.cu
            src/cuda/scf/density_builder.cu
            src/cuda/scf/convergence_check.cu
            src/cuda/scf/synthetic_scf_cycle.cu
            src/cuda/scf/scf_evaluator.cu
            src/cuda/longrange/ewald.cu
        )
        if(ATV_BUILD_MOLRS)
            list(APPEND ATOMIVERSE_SRCS src/cuda/frame/molrs_exchange.cu)
        endif()
        if(ATV_BUILD_MD)
            # multi_gpu.cu owns multi-rank NVE and links launch_nve_limit_*;
            # must stay under MD (MD=OFF was a silent link hole vs nve_limit).
            list(APPEND ATOMIVERSE_SRCS
                src/cuda/communication/multi_gpu.cu
                src/cuda/dynamics/velocity_verlet.cu
                src/cuda/dynamics/nve_limit.cu
                src/cuda/dynamics/fix_langevin.cu
                src/cuda/dynamics/nose_hoover_npt.cu
                src/cuda/dynamics/init.cu
                src/cuda/dynamics/kinetic_energy.cu
                src/cuda/dynamics/nose_hoover.cu
                src/cuda/dynamics/deform.cu
                src/cuda/dynamics/barostat.cu
            )
        endif()
        if(ATV_BUILD_NEIGHBOR)
            list(APPEND ATOMIVERSE_SRCS
                src/cuda/neighbor/neighbor_list.cu
                src/cuda/neighbor/sort_and_cluster.cu
                src/cuda/neighbor/grid.cu
                src/cuda/neighbor/pair_search.cu
                src/cuda/neighbor/fused_search.cu
                src/cuda/neighbor/atom_cell_sort.cu
                src/cuda/neighbor/setup.cu
                src/cuda/neighbor/export_pairs.cu
            )
        endif()
        if(ATV_BUILD_MINIMIZER)
            list(APPEND ATOMIVERSE_SRCS
                src/cuda/minimize/sd.cu
                src/cuda/minimize/cg.cu
                src/cuda/minimize/lbfgs.cu
                src/cuda/minimize/conv_check.cu
            )
        endif()
        if(ATV_BUILD_POTENTIALS)
            file(GLOB CUDA_POTENTIALS_SRCS CONFIGURE_DEPENDS
                "${CMAKE_SOURCE_DIR}/src/cuda/potentials/*.cu"
            )
            list(APPEND ATOMIVERSE_SRCS ${CUDA_POTENTIALS_SRCS})
        endif()
        if(ATV_BUILD_KSPACE)
            list(APPEND ATOMIVERSE_SRCS src/cuda/longrange/pme.cu)
        endif()
    else()
        # CUDA-free build still links against cuda::Timer (PIMPL in core/driver_template.h).
        list(APPEND ATOMIVERSE_SRCS src/cuda/device/timer_cpu_stub.cpp)
    endif()

    add_library(atomiverse ${ATOMIVERSE_SRCS})
    if(ATV_BUILD_MOLRS)
        target_compile_definitions(atomiverse PRIVATE
            ATV_MOLRS_CXX_API_VERSION=${ATV_MOLRS_CXX_API_VERSION}
            ATV_MOLRS_CXX_REQUIRED_CAPABILITIES=${ATV_MOLRS_CXX_REQUIRED_CAPABILITIES})
    endif()
    atv_apply_product_module_defs(atomiverse)

    # Multi-window WHAM — needs ANALYSIS + molrs pin with WhamCompute.
    if(ATV_BUILD_ANALYSIS AND ATV_BUILD_MOLRS AND ATV_MOLRS_HAS_WHAM)
        target_sources(atomiverse PRIVATE src/cpu/observe/wham.cpp)
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_MOLRS_WHAM=1)
    endif()

    if(ATV_BUILD_HF)
        target_sources(atomiverse PRIVATE
            src/cpu/scf/basis_set.cpp
            src/cpu/scf/internal/one_electron.cpp
            src/cpu/scf/two_electron.cpp
            src/cpu/scf/internal/integrals.cpp
            src/cpu/scf/ecp_potential.cpp
            src/cpu/scf/internal/ecp_integrals.cpp
            src/cpu/scf/native_integrals.cpp
            src/cpu/scf/rys_roots.cpp
            src/cpu/scf/rys_roots_iterative.cpp
            src/cpu/scf/rys_eri.cpp
            src/cpu/scf/internal/shell_pair.cpp
            src/cpu/scf/operators.cpp
            src/cpu/observe/multipole.cpp
            src/cpu/scf/wannier.cpp
            src/cpu/observe/lowdin.cpp
            src/cpu/observe/esp_charges.cpp
            src/cpu/scf/internal/cart2sph.cpp
            src/cpu/scf/internal/diis_mixer.cpp
            src/cpu/scf/internal/ediis_adiis.cpp
            src/cpu/scf/internal/hueckel_guess.cpp
            src/cpu/scf/internal/sap_guess.cpp
            src/cpu/scf/internal/hamiltonian_builder.cpp
            src/cpu/scf/internal/cfmm_j.cpp
            src/cpu/scf/internal/density_fitting.cpp
            src/cpu/scf/internal/alda_kernel.cpp
            src/cpu/scf/internal/tda_gradient.cpp
            src/cpu/scf/internal/mp2_gradient.cpp
            src/cpu/scf/internal/ccsd_gradient.cpp
            src/cpu/scf/internal/eom_ccsd_common.cpp
            src/cpu/scf/internal/cc_diis.cpp
            src/cpu/scf/internal/ci_engine.cpp
            src/cpu/scf/internal/external_field.cpp
            src/cpu/scf/internal/level_shift.cpp
            src/cpu/scf/internal/point_charges.cpp
            src/cpu/scf/internal/pcm.cpp
            src/cpu/scf/internal/soscf.cpp
            src/cpu/scf/internal/x2c.cpp
            # Multiscale electrostatic QM/MM (esmm-qmmm-03) — HF engine required.
            src/cpu/multiscale/qmmm_electrostatic.cpp
            src/cpu/scf/stages/fock_stage.cpp
            src/cpu/scf/stages/mix_stage.cpp
            src/cpu/scf/stages/density_stage.cpp
            src/cpu/scf/internal/mean_field_energy.cpp
            src/cpu/scf/mean_field_method.cpp
            src/cpu/scf/internal/orbital_hessian.cpp
            src/cpu/scf/internal/stability.cpp
            src/cpu/scf/tdhf.cpp
            src/cpu/scf/cisd.cpp
            src/cpu/scf/adc.cpp
            src/cpu/scf/g0w0.cpp
            src/cpu/scf/rpa.cpp
            src/cpu/scf/bse.cpp
            src/cpu/scf/rhf.cpp
            src/cpu/scf/gf2.cpp
            src/cpu/scf/rohf.cpp
            src/cpu/scf/uhf.cpp
            src/cpu/scf/ghf.cpp
            src/cpu/scf/rks.cpp
            src/cpu/scf/uks.cpp
            src/cpu/scf/roks.cpp
            src/cpu/scf/minao_data.cpp
            src/cpu/scf/internal/minao.cpp
            src/cpu/scf/xc_functional.cpp
            src/cpu/scf/xc_gga.cpp
            src/cpu/scf/grid.cpp
            src/cpu/scf/internal/integral_derivatives.cpp
            src/cpu/scf/internal/cphf_pack.cpp
            src/cpu/scf/internal/ucphf_pack.cpp
            src/cpu/scf/internal/gradient.cpp

            src/cpu/scf/mp2.cpp
            src/cpu/scf/mp3.cpp
            src/cpu/scf/mp4.cpp
            src/cpu/scf/ccsd.cpp
            src/cpu/scf/ccsd_spatial.cpp
            src/cpu/scf/eom_ee_ccsd.cpp
            src/cpu/scf/eom_ea_ccsd.cpp
            src/cpu/scf/eom_ip_ccsd.cpp
            src/cpu/scf/fci.cpp
            src/cpu/scf/casscf.cpp
            src/cpu/scf/nevpt2.cpp
            src/cpu/scf/orbital_info_observer.cpp
            # Electronic-state propagators (cpu/electronic/): Kolafa ASPC
            # extrapolation and Car–Parrinello fictitious-mass dynamics. They
            # depend only on cpu/linalg, and every consumer (Gaussian SCF
            # here, GPW below — GPW requires HF) is behind this flag, so they
            # build once here. The Niklasson XL recursion (xl_bomd.cpp) moved
            # to the always-on set: classical PIM XL dipoles consume it in
            # MD-only builds.
            src/cpu/electronic/aspc_predictor.cpp
            src/cpu/electronic/cpmd.cpp
            # Molecular DFT-D3 (BJ) + shared Grimme tables (also used by GPW
            # zero-damping when ATV_BUILD_GPW=ON — sources live once under HF
            # because GPW requires HF).
            src/cpu/gpw/internal/dftd3_params.cpp
            src/cpu/gpw/internal/dftd3.cpp
            src/cpu/gpw/internal/ts_dispersion.cpp
            src/cpu/gpw/internal/xdm_dispersion.cpp
            src/cpu/gpw/internal/mbd_dispersion.cpp
            src/cpu/gpw/internal/vdw_df.cpp
            src/cpu/gpw/internal/vdw_df_kernel_data.cpp
            src/cpu/gpw/internal/rvv10.cpp
            src/cpu/gpw/internal/rvv10_kernel_data.cpp
            src/cpu/gpw/internal/hirshfeld_volumes.cpp
            src/cpu/scf/dftd3.cpp
            src/cpu/scf/dftd4.cpp
            # The one flag -> dispersion-evaluator switch every published
            # mean-field gradient shares (forces / mp2_forces / ccsd_forces).
            src/cpu/scf/internal/dispersion.cpp
            src/cpu/record/cube.cpp
            src/cpu/record/density_cube_writer.cpp
            src/cpu/record/orbital_cube_writer.cpp
            src/cpu/observe/mulliken.cpp
            # eval_density_grid resolves shells via scf::build_shell_list
            # (basis_set.cpp above), so it is gated with the SCF sources —
            # same as the cube writers and mulliken. Its only consumer,
            # cuda::DensityRecorder, is #ifdef ATV_HAS_HF anyway.
            src/cpu/observe/density.cpp
        )
        if(ATV_BUILD_CUDA)
            target_sources(atomiverse PRIVATE
                src/cuda/scf/rhf.cu
                src/cuda/scf/rhf_scf_cycle.cu
                src/cuda/scf/rhf_fock.cu
                src/cuda/scf/rhf_density.cu
                src/cuda/scf/diis_mixer.cu
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
            # The electronic-state propagators GPWEvaluator drives live in
            # src/cpu/electronic/: CPMD + ASPC compile under ATV_BUILD_HF
            # (which GPW requires); the XL recursion is in the always-on set.
            # DFT-D3 engine sources are compiled under ATV_BUILD_HF (shared
            # with molecular D3BJ); GPW zero-damping calls the same symbols.
        )
        # FFT Poisson needs FFTW, supplied by the ATV_BUILD_PW block (fftw3_stub
        # is linked into atomiverse there, so the fftw3.h include propagates).
        if(ATV_BUILD_PW)
            target_sources(atomiverse PRIVATE
                src/cpu/gpw/internal/fft_cache.cpp
                src/cpu/gpw/internal/poisson_fft.cpp
                src/cpu/gpw/internal/hartree_grid.cpp
                src/cpu/gpw/internal/grid_fft.cpp
                src/cpu/gpw/internal/xc_grid_gga.cpp
                src/cpu/gpw/internal/mt_poisson.cpp
                src/cpu/gpw/internal/kpoints.cpp
                src/cpu/gpw/internal/ks_grid_build.cpp
                src/cpu/gpw/internal/product_collint.cpp
                src/cpu/gpw/gpw_solver.cpp
                src/cpu/gpw/bands_dos.cpp
                src/cpu/gpw/periodic_wannier.cpp
                src/cpu/gpw/berry_polarization.cpp
                src/cpu/gpw/internal/gpw_forces.cpp
                src/cpu/gpw/internal/gpw_stress.cpp
                src/cpu/gpw/internal/mp2_kernels.cpp
                src/cpu/gpw/mp2.cpp
                src/cpu/gpw/rpa.cpp
                src/cpu/gpw/g0w0.cpp
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
            # CUDA GPW foothold: Γ-point device Poisson/FFT + Padé LDA XC grid
            # + product ρ collocate / V·shape integrate + composed KS chain
            # stage with closed-shell PBE-GGA (spectral ∇ρ / −∇·flux) +
            # multi-grid product collocate/integrate (spectral prolong/restrict;
            # not full method product).
            # Gated on CUDA so OFF builds exclude sources entirely (iron law 8).
            if(ATV_BUILD_CUDA)
                target_sources(atomiverse PRIVATE
                    src/cuda/gpw/internal/poisson_fft.cu
                    src/cuda/gpw/internal/xc_lda.cu
                    src/cuda/gpw/internal/product_collocate.cu
                    src/cuda/gpw/internal/product_integrate.cu
                    src/cuda/gpw/internal/spectral_fft.cu
                    src/cuda/gpw/internal/multigrid_transfer.cu
                    src/cuda/gpw/internal/ks_chain.cu
                    src/cuda/gpw/gpw_solver.cu
                )
                target_link_libraries(atomiverse PUBLIC CUDA::cufft)
                message(STATUS "GPW: CUDA Γ Poisson + Padé/PBE KS chain + multigrid + public GpwDft product enabled")
            endif()
        endif()
        # vdW-DF kernel tables ship zlib-compressed (float32); inflate at first use.
        find_package(ZLIB REQUIRED)
        target_link_libraries(atomiverse PUBLIC ZLIB::ZLIB)
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
            src/cpu/planewave/internal/g_vector.cpp
            src/cpu/planewave/internal/fft.cpp
            src/cpu/planewave/internal/upf_reader.cpp
            # Analytic GTH pseudopotentials. gth_potential.cpp is shared data
            # (GPW consumes it too) and is built here because PW is the wider
            # of the two flags; its home under cpu/gpw/ is a directory-layering
            # debt, not a dependency — it pulls in only its own table header.
            src/cpu/gpw/gth_potential.cpp
            src/cpu/planewave/internal/gth_upf.cpp
            src/cpu/planewave/internal/kpoint.cpp
            src/cpu/planewave/internal/ewald.cpp
            src/cpu/planewave/internal/pw_setup.cpp
            src/cpu/planewave/internal/scf_loop.cpp
            src/cpu/planewave/internal/pw_forces.cpp
            src/cpu/planewave/internal/pw_stress.cpp
            src/cpu/planewave/internal/dfpt.cpp
            src/cpu/planewave/internal/pw_subspace.cpp
            src/cpu/planewave/internal/poisson.cpp
            src/cpu/planewave/internal/xc_builder.cpp
            src/cpu/planewave/internal/veff_builder.cpp
            src/cpu/planewave/internal/davidson.cpp
            src/cpu/planewave/internal/density_builder.cpp
            src/cpu/planewave/internal/convergence.cpp
            src/cpu/planewave/internal/smearing.cpp
            src/cpu/planewave/internal/restricted_scf.cpp
            src/cpu/planewave/pw_solver.cpp
            src/cpu/planewave/pw_evaluator.cpp
            src/cpu/planewave/bands_dos.cpp
            src/cpu/planewave/dfpt_phonon.cpp
            src/cpu/planewave/berry_polarization.cpp
            src/cpu/planewave/periodic_wannier.cpp
        )
        if(NOT ATV_BUILD_HF)
            target_sources(atomiverse PRIVATE
            )
        endif()
        target_link_libraries(atomiverse PUBLIC fftw3_stub)
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_PW)

        # CUDA plane-wave backend — device-resident kernels mirroring the CPU
        # pw module (cuFFT FFT layer, G-space Hartree solver, sparse-G ↔ dense
        # grid bridge, LDA/PBE XC and Veff stages) plus the full pw SCF band
        # solver (cuFFT + cuBLAS Davidson). Only built when CUDA is enabled.
        if(ATV_BUILD_CUDA)
            target_sources(atomiverse PRIVATE
                src/cuda/planewave/internal/fft.cu
                src/cuda/planewave/internal/poisson.cu
                src/cuda/planewave/internal/grid_map.cu
                src/cuda/planewave/internal/xc_lda.cu
                src/cuda/planewave/internal/xc_pbe.cu
                src/cuda/planewave/internal/veff.cu
                src/cuda/planewave/internal/device_solver.cu
                src/cuda/planewave/internal/device_density.cu
                src/cuda/planewave/internal/pw_forces.cu
                src/cuda/planewave/internal/pw_stress.cu
                src/cuda/planewave/pw_solver.cu
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
            src/torch/device/stream_guard.cu
            src/torch/potentials/lj.cu
            src/torch/neighbor/neighbor_list.cu
            src/torch/potentials/harmonic_bond.cu
        )
        if(ATV_BUILD_MOLRS)
            target_sources(atomiverse PRIVATE src/torch/frame/molrs_exchange.cu)
        endif()
        target_link_libraries(atomiverse PUBLIC ${TORCH_LIBRARIES})
        target_include_directories(atomiverse PUBLIC ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(atomiverse PUBLIC ATV_HAS_TORCH)
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

    # ── GFN2 private kernel: gfortran-built libxtb (C API) ─────────────
    # Wave-02 energy path. Prefer ATV_XTB_ROOT / env; default to the local
    # gfortran install next to the prebuilt Intel binary (which cannot link).
    # Not a black-box subprocess: linked library, Frame I/O stays on ATV.
    if(ATV_BUILD_SEMIEMPIRICAL)
        set(_ATV_XTB_DEFAULT
            "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/software/xtb-6.7.1-gfortran")
        if(DEFINED ENV{ATV_XTB_ROOT} AND NOT "$ENV{ATV_XTB_ROOT}" STREQUAL "")
            set(_ATV_XTB_DEFAULT "$ENV{ATV_XTB_ROOT}")
        endif()
        set(ATV_XTB_ROOT "${_ATV_XTB_DEFAULT}" CACHE PATH
            "Root of gfortran-built xtb (include/ + lib64/libxtb.so)")
        find_path(ATV_XTB_INCLUDE_DIR xtb.h
            HINTS "${ATV_XTB_ROOT}/include" NO_DEFAULT_PATH)
        find_library(ATV_XTB_LIBRARY NAMES xtb
            HINTS "${ATV_XTB_ROOT}/lib64" "${ATV_XTB_ROOT}/lib" NO_DEFAULT_PATH)
        if(ATV_XTB_INCLUDE_DIR AND ATV_XTB_LIBRARY)
            target_include_directories(atomiverse PRIVATE "${ATV_XTB_INCLUDE_DIR}")
            target_link_libraries(atomiverse PUBLIC "${ATV_XTB_LIBRARY}" gfortran gomp)
            target_compile_definitions(atomiverse PUBLIC ATV_HAS_XTB_KERNEL)
            message(STATUS "GFN2 xtb kernel: ${ATV_XTB_LIBRARY}")
        else()
            message(STATUS
                "GFN2 xtb kernel: NOT FOUND under ATV_XTB_ROOT=${ATV_XTB_ROOT} "
                "(GFN2::launch will throw until a gfortran libxtb is installed)")
        endif()
    endif()

    # xsimd is linked for direct use in native_integrals.cpp ERI hot loops.
    # XTENSOR_USE_XSIMD is NOT set globally. Precise reason (traced 2026-07-26,
    # reported upstream as xtensor-stack/xtensor#2929): it is a hard COMPILE
    # error, not a runtime break, and it is not reshape_view in general — it is
    # `reshape_view(<lazy xfunction>, {G,1})` followed by a broadcast assign, the
    # pattern used ~12× in cpu/scf/xc_grid_engine.h. xstrided_view gates its SIMD
    # interface on the *wrapped expression* (xstrided_view.hpp:183) but then does
    # `&(storage()[i])` (:688); for a lazy operand the flat storage returns
    # prvalues, so taking `&` is ill-formed.
    # WORKAROUND if we ever want the global flag: pre-evaluate the operand
    # (`xt::xtensor<double,1> tmp = w * v;` then reshape_view(tmp, …)) — that
    # makes the view container-backed and compiles under xsimd. Note this flag is
    # NOT needed to vectorize a hot loop: use xsimd directly on raw spans the way
    # native_integrals.cpp does.
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
    #   - Integrator translation units (`src/cpu/dynamics/velocity_verlet.cpp`,
    #     `src/cpu/dynamics/nose_hoover.cpp`, `src/cpu/dynamics/init.cpp`) MUST NOT
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

    # ── molrs CXX (only when products derived ATV_BUILD_MOLRS) ──
    if(ATV_BUILD_MOLRS)
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
        # Package is `molcrafts-molrs-cxxapi` in Cargo.toml, but `[lib] name` is
        # `molrs_cxxapi` -- corrosion exposes the target under the lib name.
        corrosion_add_cxxbridge(molrs_cxx
            CRATE molrs_cxxapi
            FILES bridge.rs
        )

        # corrosion v0.6.1: mark generated cxxbridge outputs GENERATED (CMP0118).
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
        # cuda::stream_synchronize counter in src/cuda/asynchronous/record_base.cu
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
