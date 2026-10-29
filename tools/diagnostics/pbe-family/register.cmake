# Opt-in complete-method validation, outside the unit-test suite.
add_executable(pbe-family "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
target_link_libraries(pbe-family PRIVATE atomiverse)
target_compile_features(pbe-family PRIVATE cxx_std_20)
set_target_properties(pbe-family PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(ATV_BUILD_GPW)
    add_executable(pbe-family-gpw "${CMAKE_CURRENT_LIST_DIR}/gpw.cpp")
    target_link_libraries(pbe-family-gpw PRIVATE atomiverse)
    target_compile_features(pbe-family-gpw PRIVATE cxx_std_20)
    set_target_properties(pbe-family-gpw PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

if(ATV_BUILD_CUDA AND ATV_BUILD_PW AND ATV_BUILD_TESTS)
    # Complete SCF baselines are deliberately not registered with CTest.
    add_executable(pw-cell-stress-baselines
        "${CMAKE_CURRENT_LIST_DIR}/legacy_pw_cell_stress.cu")
    target_link_libraries(pw-cell-stress-baselines PRIVATE
        atomiverse GTest::gtest_main CUDA::cufft)
    set_target_properties(pw-cell-stress-baselines PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

if(ATV_BUILD_CUDA AND ATV_BUILD_PW)
    # Actual nested PW solves are opt-in diagnostics, never CTest units.
    add_executable(pw-driver-baselines
        "${CMAKE_CURRENT_LIST_DIR}/pw_driver.cpp")
    target_link_libraries(pw-driver-baselines PRIVATE atomiverse)
    target_compile_features(pw-driver-baselines PRIVATE cxx_std_20)
    set_target_properties(pw-driver-baselines PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    if(NOT TARGET cuda-pw-public)
        include("${CMAKE_CURRENT_LIST_DIR}/../cuda-pw-public/register.cmake")
    endif()
endif()
