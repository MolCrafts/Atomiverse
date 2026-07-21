# tools/bench_gaff_lammps — MPI GAFF vs LAMMPS timing harness.
# Requires ATV_USE_MPI + ATV_BUILD_KSPACE (PPPM/PME path).

if(NOT ATV_USE_MPI)
    message(FATAL_ERROR "bench_gaff_lammps needs -DATV_USE_MPI=ON")
endif()
if(NOT ATV_BUILD_KSPACE)
    message(FATAL_ERROR "bench_gaff_lammps needs -DATV_BUILD_KSPACE=ON (FFTW3)")
endif()

add_executable(run_atv_gaff run_atv_gaff.cpp)
target_link_libraries(run_atv_gaff PRIVATE atomiverse MPI::MPI_CXX)
target_compile_features(run_atv_gaff PRIVATE cxx_std_20)
