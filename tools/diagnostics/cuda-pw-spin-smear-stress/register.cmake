add_executable(cuda-pw-spin-smear-stress "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
target_link_libraries(cuda-pw-spin-smear-stress PRIVATE atomiverse)
target_compile_features(cuda-pw-spin-smear-stress PRIVATE cxx_std_20)
set_target_properties(cuda-pw-spin-smear-stress PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Original implementation, renamed only for a strict stage negative control.
if(ATV_PW_SPIN_BASELINE)
    add_executable(cuda-pw-spin-residual-before
        "${CMAKE_SOURCE_DIR}/tests/cuda/planewave/test_density.cu"
        "${ATV_PW_SPIN_BASELINE}/device_density.cu")
    target_compile_definitions(cuda-pw-spin-residual-before PRIVATE
        DeviceDensity=DeviceDensityBefore)
    target_include_directories(cuda-pw-spin-residual-before BEFORE PRIVATE
        "${ATV_PW_SPIN_BASELINE}/include")
    target_link_libraries(cuda-pw-spin-residual-before PRIVATE
        atomiverse GTest::gtest_main)
    set_target_properties(cuda-pw-spin-residual-before PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# One make traversal shares the Cargo/ABI dependency visit across all selected
# executables. Building six top-level Make goals separately repeats that work.
if(ATV_BUILD_TESTS)
    cmake_language(DEFER CALL add_custom_target cuda-pw-spin-stress-checks)
    cmake_language(DEFER CALL add_dependencies cuda-pw-spin-stress-checks
        cuda-pw-spin-smear-stress test_cuda_pw_density test_cuda_pw_guards
        test_cuda_pw_cell_stress test_cuda_pw_force_local)
    if(ATV_PW_SPIN_BASELINE)
        cmake_language(DEFER CALL add_dependencies cuda-pw-spin-stress-checks
            cuda-pw-spin-residual-before)
    endif()
endif()
