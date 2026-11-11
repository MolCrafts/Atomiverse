# Native performance harnesses, one subdirectory per domain. Built only with
# -DATV_BUILD_BENCHMARKS=ON; binaries land in <build>/bin. Each binary prints
# wall times; a perf claim is the number it printed on a named node.

find_package(OpenMP)

# atv_add_bench(<name> <source> [MPI] [OPENMP] [SEPARABLE])
function(atv_add_bench name src)
    cmake_parse_arguments(ARG "MPI;OPENMP;SEPARABLE" "" "" ${ARGN})
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE atomiverse)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    if(ARG_SEPARABLE)
        set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    if(ARG_MPI)
        target_link_libraries(${name} PRIVATE MPI::MPI_CXX)
    endif()
    if(ARG_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${name} PRIVATE ATV_BENCH_HAS_OPENMP)
    endif()
endfunction()

# Public C++ participant for the PW/QE and GPW/CP2K accuracy campaigns.
if(ATV_BUILD_PW)
    if(ATV_BUILD_CUDA)
        set_source_files_properties(periodic/periodic_dft.cpp PROPERTIES LANGUAGE CUDA)
        atv_add_bench(atv_periodic_dft periodic/periodic_dft.cpp SEPARABLE)
    else()
        atv_add_bench(atv_periodic_dft periodic/periodic_dft.cpp)
    endif()
endif()

# ── md ──────────────────────────────────────────────────────────────────────
if(ATV_BUILD_MD AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
    # CPU LJ/EAM/SW MD; compare/lj-lammps/compare.py checks it against LAMMPS.
    atv_add_bench(bench_lj_cpu md/bench_lj_cpu.cpp)
    if(ATV_USE_MPI AND ATV_BUILD_KSPACE)
        # MPI GAFF (PPPM/PME) timing and finite-difference force check;
        # driven by compare/gaff-liquid and compare/gaff-peo-lammps.
        atv_add_bench(run_atv_gaff md/run_atv_gaff.cpp MPI)
        atv_add_bench(fd_check md/fd_check.cpp MPI)
    endif()
    if(ATV_BUILD_CUDA)
        # OpenMM cluster/tile LJ engine: bench_nonbonded_style [N] [steps] [warmup]
        atv_add_bench(bench_nonbonded_style md/bench_nonbonded_style.cu)
        atv_add_bench(bench_multi_gpu_force md/bench_multi_gpu_force.cu)
        if(ATV_BUILD_KSPACE)
            # Steady-step CUDA MD: no-D2H Nsight audit, bitwise trajectory
            # comparison and the md/mps_pack.sh MPS packing driver.
            atv_add_bench(md-residency md/residency/run.cu)
        endif()
        if(ATV_BUILD_KSPACE AND ATV_BUILD_EVALUATE)
            atv_add_bench(bench_gaff md/bench_gaff.cu)
        endif()
    endif()
endif()

# ── minimize ────────────────────────────────────────────────────────────────
if(ATV_BUILD_MINIMIZER AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
    # CPU minimizer overhead + force-call counts (Rosenbrock / LJ fcc).
    atv_add_bench(bench_minimize_cpu minimize/bench_minimize_cpu.cpp)
endif()

# ── scf ─────────────────────────────────────────────────────────────────────
if(ATV_BUILD_HF)
    # CPU HF diagnostic with J/K phase logs (profile_hf_cpu.py drives it).
    atv_add_bench(bench_hf_cpu scf/bench_hf_cpu.cpp)
    if(ATV_BUILD_CUDA)
        atv_add_bench(bench_mp2_cpu_vs_cuda scf/bench_mp2_cpu_vs_cuda.cu
                      SEPARABLE OPENMP)
    endif()
endif()

# ── pw ──────────────────────────────────────────────────────────────────────
# CPU vs CUDA plane-wave wall over Si supercells.
# Run: bench_pw_large_cell <upf> <n_rep> <cpu|cuda|both> [ecut_ry] [k_mesh]
if(ATV_BUILD_PW AND ATV_BUILD_CUDA)
    atv_add_bench(bench_pw_large_cell pw/bench_pw_large_cell.cu SEPARABLE OPENMP)
endif()

# ── semiempirical ───────────────────────────────────────────────────────────
if(ATV_BUILD_SEMIEMPIRICAL)
    atv_add_bench(bench_nddo_allocations
                  semiempirical/nddo_forces/count_allocations.cpp)
    if(ATV_BUILD_CUDA AND ATV_BUILD_EVALUATE)
        atv_add_bench(bench_nddo_forces semiempirical/nddo_forces/run.cu)
        atv_add_bench(bench_nddo_energy semiempirical/nddo_energy/run.cu)
    endif()
endif()
