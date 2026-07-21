# tools/debug_amorphous/CMakeLists.txt
#
# Builds the amorphous-PE LAMMPS-comparison runner. Pulled in from the
# top-level CMakeLists.txt only when -DATV_BUILD_DEBUG_AMORPHOUS=ON is set.
# Requires MPI (the runner is the multi-rank story; we don't pretend
# otherwise).

if(NOT ATV_USE_MPI)
    message(FATAL_ERROR
        "tools/debug_amorphous/run_atomiverse needs MPI. Configure with "
        "-DATV_USE_MPI=ON.")
endif()

add_executable(run_atomiverse_amorphous run_atomiverse.cpp)
target_link_libraries(run_atomiverse_amorphous
    PRIVATE atomiverse MPI::MPI_CXX)
target_include_directories(run_atomiverse_amorphous PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(run_atomiverse_amorphous PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)


add_executable(prof_amorphous prof_amorphous.cpp)
target_link_libraries(prof_amorphous PRIVATE atomiverse MPI::MPI_CXX)
target_include_directories(prof_amorphous PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(prof_amorphous PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
