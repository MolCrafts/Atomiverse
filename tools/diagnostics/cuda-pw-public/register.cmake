# Opt-in full-method diagnostic; not registered as a CTest unit.
# Configure the repository with CMAKE_PROJECT_Atomiverse_INCLUDE pointing here.
add_executable(cuda-pw-public "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
target_link_libraries(cuda-pw-public PRIVATE atomiverse)
target_compile_features(cuda-pw-public PRIVATE cxx_std_20)
set_target_properties(cuda-pw-public PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Optional negative control: compile an explicitly supplied earlier density
# implementation/header against the same strict stage tests and current CPU.
if(ATV_PW_QUADRATURE_BASELINE)
    add_executable(cuda-pw-density-before
        "${CMAKE_SOURCE_DIR}/tests/cuda/planewave/test_density.cu"
        "${ATV_PW_QUADRATURE_BASELINE}/device_density.cu")
    # Keep the original class distinct from the current library's CUDA
    # registration objects, which device-link may also pull from the archive.
    target_compile_definitions(cuda-pw-density-before PRIVATE
        DeviceDensity=DeviceDensityBefore)
    target_include_directories(cuda-pw-density-before BEFORE PRIVATE
        "${ATV_PW_QUADRATURE_BASELINE}/include")
    target_link_libraries(cuda-pw-density-before PRIVATE atomiverse GTest::gtest_main)
    set_target_properties(cuda-pw-density-before PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
