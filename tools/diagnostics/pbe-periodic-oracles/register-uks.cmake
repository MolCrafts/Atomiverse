# Opt-in actual independent Gamma UKS reference comparison; never a CTest unit.
add_executable(pbe-periodic-uks-oracles "${CMAKE_CURRENT_LIST_DIR}/uks.cpp")
target_link_libraries(pbe-periodic-uks-oracles PRIVATE atomiverse)
target_compile_features(pbe-periodic-uks-oracles PRIVATE cxx_std_20)
set_target_properties(pbe-periodic-uks-oracles PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
