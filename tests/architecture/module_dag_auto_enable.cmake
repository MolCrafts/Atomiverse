# Configure-level check: MD / PATH / KSPACE auto-enable their DAG edges.
# Invoked as a CTest cmake script from tests/CMakeLists.txt.

if(NOT DEFINED ATV_SOURCE_DIR)
    message(FATAL_ERROR "ATV_SOURCE_DIR must be set")
endif()

set(_bin "${CMAKE_CURRENT_BINARY_DIR}/module_dag_auto_enable_tree")
file(REMOVE_RECURSE "${_bin}")
file(MAKE_DIRECTORY "${_bin}")

# MD=ON should force POTENTIALS+NEIGHBOR even if initially OFF in cache seed.
# We pass -D after the project options; AtvProductModules FORCE-overwrites.
execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${ATV_SOURCE_DIR}"
        -B "${_bin}"
        -DATV_BUILD_CUDA=OFF
        -DATV_BUILD_TESTS=OFF
        -DATV_BUILD_PYTHON=OFF
        -DATV_BUILD_HF=OFF
        -DATV_BUILD_PW=OFF
        -DATV_BUILD_MD=ON
        -DATV_BUILD_POTENTIALS=OFF
        -DATV_BUILD_NEIGHBOR=OFF
        -DATV_BUILD_MC=OFF
        -DATV_BUILD_MINIMIZER=OFF
        -DATV_BUILD_PATH=OFF
        -DATV_BUILD_VIB=OFF
        -DATV_BUILD_SEMIEMPIRICAL=OFF
        -DATV_BUILD_KSPACE=OFF
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MD auto-enable configure failed:\n${_out}\n${_err}")
endif()

# Read forced values from the build cache
set(_cache "${_bin}/CMakeCache.txt")
file(STRINGS "${_cache}" _pot REGEX "^ATV_BUILD_POTENTIALS:BOOL=")
file(STRINGS "${_cache}" _nei REGEX "^ATV_BUILD_NEIGHBOR:BOOL=")
if(NOT _pot MATCHES "=ON$" OR NOT _nei MATCHES "=ON$")
    message(FATAL_ERROR
        "Expected POTENTIALS+NEIGHBOR forced ON by MD; got\n  ${_pot}\n  ${_nei}")
endif()

message(STATUS "module_dag_auto_enable: MD → POTENTIALS+NEIGHBOR OK")
