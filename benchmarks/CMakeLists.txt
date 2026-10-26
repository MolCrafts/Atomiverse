# Standing performance harnesses. Deliberately NOT registered with ctest —
# tests/ is unit-only and "if it is registered, it runs" (see docs/developer/project-contract.md).
# Each binary prints walls; a perf claim is the number the harness printed on a
# named node, recorded in the matching README.
if(ATV_BUILD_CUDA)
    add_subdirectory(cuda)
endif()

if(ATV_BUILD_HF)
    add_executable(bench_hf_cpu scf/bench_cpu.cpp)
    target_link_libraries(bench_hf_cpu PRIVATE atomiverse)
endif()

if(ATV_BUILD_MD)
    add_subdirectory(lj)
endif()

if(ATV_BUILD_SEMIEMPIRICAL)
    add_executable(bench_nddo_allocations
        semiempirical/nddo_forces/count_allocations.cpp)
    target_link_libraries(bench_nddo_allocations PRIVATE atomiverse)
    if(ATV_BUILD_CUDA AND ATV_BUILD_EVALUATE)
        add_executable(bench_nddo_forces semiempirical/nddo_forces/run.cu)
        target_link_libraries(bench_nddo_forces PRIVATE atomiverse)
        set_target_properties(bench_nddo_forces PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        add_executable(bench_nddo_energy semiempirical/nddo_energy/run.cu)
        target_link_libraries(bench_nddo_energy PRIVATE atomiverse)
        set_target_properties(bench_nddo_energy PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endif()
