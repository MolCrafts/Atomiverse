# Opt-in full-method checks, deliberately outside the CTest unit tier.
add_executable(pw-spin-smear-stress "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
target_link_libraries(pw-spin-smear-stress PRIVATE atomiverse)
target_compile_features(pw-spin-smear-stress PRIVATE cxx_std_20)
set_target_properties(pw-spin-smear-stress PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(ATV_PW_STRESS_BASELINE)
    add_executable(pw-spin-smear-stress-before
        "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
        "${ATV_PW_STRESS_BASELINE}")
    target_compile_definitions(pw-spin-smear-stress-before PRIVATE
        ATV_PW_STRESS_BEFORE
        pw_stress_fd_impl=pw_stress_fd_impl_before
        pw_stress_analytic_impl=pw_stress_analytic_impl_before)
    target_link_libraries(pw-spin-smear-stress-before PRIVATE atomiverse)
    target_compile_features(pw-spin-smear-stress-before PRIVATE cxx_std_20)
    set_target_properties(pw-spin-smear-stress-before PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
