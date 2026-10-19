# Standing benchmark only; no ctest registration.
add_executable(bench_lj_cpu bench_lj_cpu.cpp)
target_link_libraries(bench_lj_cpu PRIVATE atomiverse)
set_target_properties(bench_lj_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
