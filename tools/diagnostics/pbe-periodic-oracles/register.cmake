# Opt-in actual external-reference comparison, never a CTest unit.
add_executable(pbe-periodic-oracles "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
target_link_libraries(pbe-periodic-oracles PRIVATE atomiverse)
target_compile_features(pbe-periodic-oracles PRIVATE cxx_std_20)
set_target_properties(pbe-periodic-oracles PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
