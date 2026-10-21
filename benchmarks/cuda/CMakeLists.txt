# CUDA performance harnesses (manual, never ctest). Built only with
# -DATV_BUILD_BENCHMARKS=ON -DATV_BUILD_CUDA=ON; binaries land in <build>/bin.

# OpenMM cluster/tile LJ engine. Run: bench_nonbonded_style [N] [steps] [warmup]
add_executable(bench_nonbonded_style bench_nonbonded_style.cu)
target_link_libraries(bench_nonbonded_style PRIVATE atomiverse)
set_target_properties(bench_nonbonded_style PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(ATV_BUILD_MD AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
    add_executable(bench_multi_gpu_force bench_multi_gpu_force.cu)
    target_link_libraries(bench_multi_gpu_force PRIVATE atomiverse)
    set_target_properties(bench_multi_gpu_force PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

if(ATV_BUILD_KSPACE AND ATV_BUILD_MD AND ATV_BUILD_EVALUATE
   AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
    add_executable(bench_gaff bench_gaff.cu)
    target_link_libraries(bench_gaff PRIVATE atomiverse)
    set_target_properties(bench_gaff PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

find_package(OpenMP)

# CPU vs CUDA RHF wall + AO scaling.
# Run: OMP_NUM_THREADS=8 ./bin/bench_rhf_cpu_vs_cuda [n_rep_e] [n_rep_f]
if(ATV_BUILD_HF)
    add_executable(bench_rhf_cpu_vs_cuda bench_rhf_cpu_vs_cuda.cu)
    target_link_libraries(bench_rhf_cpu_vs_cuda PRIVATE atomiverse)
    set_target_properties(bench_rhf_cpu_vs_cuda PROPERTIES
                          CUDA_SEPARABLE_COMPILATION ON
                          RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    # OpenMP for a fair multi-thread CPU leg on the GPU node.
    if(OpenMP_CXX_FOUND)
        target_link_libraries(bench_rhf_cpu_vs_cuda PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# CPU vs CUDA plane-wave wall over Si supercells.
# Run: ./bin/bench_pw_large_cell <upf> <n_rep> <cpu|cuda|both> [ecut_ry] [k_mesh]
if(ATV_BUILD_PW)
    add_executable(bench_pw_large_cell bench_pw_large_cell.cu)
    target_link_libraries(bench_pw_large_cell PRIVATE atomiverse)
    set_target_properties(bench_pw_large_cell PROPERTIES
                          CUDA_SEPARABLE_COMPILATION ON
                          RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(bench_pw_large_cell PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
