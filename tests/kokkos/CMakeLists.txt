# Unit test for the Kokkos Serial LJ kernel. Registered from the top-level
# CMakeLists only when ATV_BUILD_KOKKOS=ON, so an OFF build never compiles it.
# Not hooked from tests/CMakeLists.txt.

add_executable(test_kokkos_pair_lj_cut potentials/test_pair_lj_cut.cpp)
target_link_libraries(test_kokkos_pair_lj_cut PRIVATE atomiverse_kokkos GTest::gtest_main)
set_target_properties(test_kokkos_pair_lj_cut PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)
add_test(NAME test_kokkos_pair_lj_cut COMMAND test_kokkos_pair_lj_cut)
set_tests_properties(test_kokkos_pair_lj_cut PROPERTIES
    TIMEOUT ${ATV_TEST_TIMEOUT_SECONDS}
    ENVIRONMENT "OMP_NUM_THREADS=${ATV_TEST_OMP_THREADS}")
