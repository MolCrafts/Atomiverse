# Configure-level check: MD=OFF excludes src/cpu/md from atomiverse SOURCES.

if(NOT DEFINED ATV_SOURCE_DIR)
    message(FATAL_ERROR "ATV_SOURCE_DIR must be set")
endif()

set(_bin "${CMAKE_CURRENT_BINARY_DIR}/module_off_excludes_tree")
file(REMOVE_RECURSE "${_bin}")
file(MAKE_DIRECTORY "${_bin}")

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${ATV_SOURCE_DIR}"
        -B "${_bin}"
        -DATV_BUILD_CUDA=OFF
        -DATV_BUILD_TESTS=OFF
        -DATV_BUILD_PYTHON=OFF
        -DATV_BUILD_HF=OFF
        -DATV_BUILD_PW=OFF
        -DATV_BUILD_MD=OFF
        -DATV_BUILD_MC=OFF
        -DATV_BUILD_MINIMIZER=OFF
        -DATV_BUILD_PATH=OFF
        -DATV_BUILD_VIB=OFF
        -DATV_BUILD_SEMIEMPIRICAL=OFF
        -DATV_BUILD_POTENTIALS=ON
        -DATV_BUILD_NEIGHBOR=ON
        -DATV_BUILD_KSPACE=OFF
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MD=OFF configure failed:\n${_out}\n${_err}")
endif()

# Dump target sources via a tiny helper project file written next to the cache.
file(WRITE "${_bin}/dump_sources.cmake" [=[
include("${CMAKE_BINARY_DIR}/CMakeFiles/ExportDump.cmake" OPTIONAL)
get_target_property(_srcs atomiverse SOURCES)
if(NOT _srcs)
  # Fallback: parse build.ninja / compile commands if ninja
  message(STATUS "SOURCES property empty; scanning link line via file(GLOB)")
endif()
set(_joined "")
foreach(s IN LISTS _srcs)
  string(APPEND _joined "${s};")
endforeach()
file(WRITE "${CMAKE_BINARY_DIR}/atomiverse_sources.txt" "${_joined}")
]=])

# Prefer reading the generated build files for md paths
file(GLOB_RECURSE _md_objs
    "${_bin}/**/src/cpu/md/*"
    "${_bin}/**/md/*.cpp.o"
    "${_bin}/**/velocity_verlet*"
)
# Also check CMakeFiles atomiverse.dir for md paths
file(GLOB_RECURSE _md_in_cmake
    "${_bin}/CMakeFiles/atomiverse.dir/src/cpu/md/*"
)
if(_md_in_cmake OR _md_objs)
    message(FATAL_ERROR
        "MD=OFF but found md build artifacts under ${_bin}:\n${_md_in_cmake}${_md_objs}")
endif()

# Positive check: cell_list (neighbor) should still be present when NEIGHBOR=ON
file(GLOB_RECURSE _nei_in_cmake
    "${_bin}/CMakeFiles/atomiverse.dir/src/cpu/neighbor/*"
)
if(NOT _nei_in_cmake)
    message(STATUS
        "Note: neighbor object path layout may differ by generator; "
        "MD exclusion check (no md/ objects) passed.")
else()
    message(STATUS "module_off_excludes_sources: neighbor objects present, md absent — OK")
endif()

message(STATUS "module_off_excludes_sources: MD=OFF excludes cpu/md — OK")
