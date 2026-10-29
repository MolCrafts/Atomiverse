# Opt-in independent CP2K comparison, never a CTest unit.
add_executable(pbe-periodic-cp2k-oracles "${CMAKE_CURRENT_LIST_DIR}/cp2k.cpp")
target_link_libraries(pbe-periodic-cp2k-oracles PRIVATE atomiverse)
target_compile_features(pbe-periodic-cp2k-oracles PRIVATE cxx_std_20)
set_target_properties(pbe-periodic-cp2k-oracles PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
