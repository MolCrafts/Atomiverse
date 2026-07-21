# tests/cpu/scf/oracle/CMakeLists.txt
#
# libxc test-only oracle for XC parity validation.
#
# This file is added via `add_subdirectory(cpu/scf/oracle)` in
# tests/CMakeLists.txt ONLY when -DATV_BUILD_XC_ORACLE=ON, so every
# target declared here is implicitly gated. The root CMakeLists.txt's
# ATV_BUILD_XC_ORACLE block has already invoked find_package(Libxc 6
# CONFIG QUIET) and bailed out via FATAL_ERROR if libxc was missing,
# so `Libxc::xc` is guaranteed to exist by the time this file runs.

# Oracle static library: contains every call to libxc in the codebase.
add_library(xc_libxc_oracle STATIC
    xc_libxc_oracle.cpp
)
target_include_directories(xc_libxc_oracle PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(xc_libxc_oracle
    PUBLIC  Libxc::xc       # propagates libxc to test targets that link this
    PRIVATE atomiverse      # for ATV::cpu::scf::XCOutput definition
)
set_target_properties(xc_libxc_oracle PROPERTIES CXX_STANDARD 17)

# Smoke test for the oracle itself. Verifies init/teardown,
# spin-polarized vs unpolarized, density cutoff, and the Slater
# textbook value (cross-library reference for LDA exchange).
atv_add_cpp_test(test_xc_libxc_oracle test_xc_libxc_oracle.cpp)
target_link_libraries(test_xc_libxc_oracle PRIVATE xc_libxc_oracle)

# Native-vs-libxc parity test (the binding scientific gate of sub-spec 03).
# Lives in the parent tests/cpu/scf/ directory but registered here so it
# is only built when the oracle is available.
atv_add_cpp_test(test_cpu_scf_xc_vs_libxc ../test_xc_vs_libxc.cpp)
target_link_libraries(test_cpu_scf_xc_vs_libxc PRIVATE xc_libxc_oracle)
