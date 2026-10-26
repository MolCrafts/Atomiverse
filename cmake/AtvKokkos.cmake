# Kokkos backend foothold. OFF excludes the library and its test.
# Kokkos itself is not vendored: find_package, or add_subdirectory an
# external tree (the LAMMPS checkout's lib/kokkos). Never copy it in.

if(NOT ATV_BUILD_KOKKOS)
    return()
endif()

set(ATV_KOKKOS_SOURCE "" CACHE PATH
    "External Kokkos source tree, used only when find_package(Kokkos) fails. Not copied into the repo.")

find_package(Kokkos QUIET CONFIG)
if(Kokkos_FOUND)
    if(DEFINED Kokkos_ENABLE_SERIAL AND NOT Kokkos_ENABLE_SERIAL)
        message(FATAL_ERROR
            "ATV_BUILD_KOKKOS=ON but the installed Kokkos has "
            "Kokkos_ENABLE_SERIAL=OFF. The foothold kernel launches "
            "Kokkos::RangePolicy<Kokkos::Serial>. Install a Serial build "
            "or set Kokkos_ROOT to one.")
    endif()
    message(STATUS "ATV_BUILD_KOKKOS: using installed Kokkos ${Kokkos_VERSION}")
else()
    set(_atv_kokkos_src "${ATV_KOKKOS_SOURCE}")
    if(NOT _atv_kokkos_src)
        set(_atv_kokkos_candidates)
        if(DEFINED ENV{ATV_KOKKOS_SOURCE} AND NOT "$ENV{ATV_KOKKOS_SOURCE}" STREQUAL "")
            list(APPEND _atv_kokkos_candidates "$ENV{ATV_KOKKOS_SOURCE}")
        endif()
        # docs/developer/project-contract.md names ../../lammps-src from the repo root. A worktree
        # sits deeper, so walk parents for lib/kokkos (not copied in).
        set(_atv_walk "${CMAKE_SOURCE_DIR}")
        foreach(_up RANGE 0 8)
            foreach(_rel IN ITEMS
                    "lammps-src/lib/kokkos"
                    "lammps-src/src/lib/kokkos"
                    "lammps/lib/kokkos"
                    "lammps/src/lib/kokkos")
                list(APPEND _atv_kokkos_candidates "${_atv_walk}/${_rel}")
            endforeach()
            get_filename_component(_atv_walk "${_atv_walk}" DIRECTORY)
        endforeach()
        foreach(_c IN LISTS _atv_kokkos_candidates)
            if(EXISTS "${_c}/CMakeLists.txt" AND EXISTS "${_c}/core/src")
                set(_atv_kokkos_src "${_c}")
                break()
            endif()
        endforeach()
    endif()
    if(NOT _atv_kokkos_src OR NOT EXISTS "${_atv_kokkos_src}/CMakeLists.txt")
        message(FATAL_ERROR
            "ATV_BUILD_KOKKOS=ON but Kokkos was not found, and Atomiverse "
            "does not vendor it.\n"
            "  Install Kokkos with the Serial backend and set Kokkos_ROOT,\n"
            "  or set ATV_KOKKOS_SOURCE (cache or environment) to an external "
            "Kokkos tree (LAMMPS lib/kokkos).")
    endif()
    message(STATUS "ATV_BUILD_KOKKOS: building external Kokkos from ${_atv_kokkos_src}")
    # Serial only. Do not inherit a CUDA or OpenMP execution space for this
    # foothold; the kernel names Kokkos::Serial.
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL
        "ATV Kokkos foothold: Serial CPU execution space" FORCE)
    set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL
        "ATV Kokkos foothold does not build the OpenMP execution space" FORCE)
    set(Kokkos_ENABLE_CUDA OFF CACHE BOOL
        "ATV Kokkos foothold does not build the CUDA execution space" FORCE)
    set(Kokkos_ENABLE_HIP OFF CACHE BOOL
        "ATV Kokkos foothold does not build the HIP execution space" FORCE)
    set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Do not build Kokkos tests" FORCE)
    set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "Do not build Kokkos examples" FORCE)
    set(Kokkos_ENABLE_BENCHMARKS OFF CACHE BOOL "Do not build Kokkos benchmarks" FORCE)
    add_subdirectory("${_atv_kokkos_src}" "${CMAKE_BINARY_DIR}/kokkos" EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET Kokkos::kokkos)
    message(FATAL_ERROR
        "ATV_BUILD_KOKKOS=ON but target Kokkos::kokkos does not exist after "
        "find_package / add_subdirectory.")
endif()

add_library(atomiverse_kokkos STATIC
    src/kokkos/potentials/pair_lj_cut.cpp)
target_compile_features(atomiverse_kokkos PUBLIC cxx_std_20)
target_include_directories(atomiverse_kokkos PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>")
target_link_libraries(atomiverse_kokkos PUBLIC Kokkos::kokkos)
target_compile_definitions(atomiverse_kokkos PUBLIC ATV_HAS_KOKKOS)
target_compile_options(atomiverse_kokkos PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wunused-function -Wunused-variable>)
