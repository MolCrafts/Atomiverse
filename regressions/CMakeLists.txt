# ── regressions/ — end-to-end capability pins ─────────────────────────────
#
# Contract: regressions/README.md. Every case directory carries a
# `case.cmake` that calls atv_regression(); this file only discovers them.
# All cases link into one executable, `atv_regressions`; each is one ctest
# test labelled `regression`:
#
#   ctest -L regression                       # every case
#   ctest -R '^scf/coupled-cluster$'            # one case
#   atv_regressions --case scf/ccsd-h2-sto3g  # same, without ctest
#   ctest -LE regression                      # unit tests only

# atv_regression(SECTION <name> REQUIRES <tokens…>|NONE [DEVICES cpu|cuda|both]
#                [THREADS n] [GPUS n] [MPIEXEC] [LIBS targets…])
#
# One call per section, from regressions/<domain>/<method>/case.cmake; the
# method directory is one case (one ctest test) running its enabled sections.
#   SECTION   section source <name>.cpp, or <name>.cu on CUDA (see DEVICES).
#   REQUIRES  product modules (ATV_BUILD_<X>) plus WHAM (molrs WhamCompute) and
#             FFTW_MPI (multi-rank FFTW); any unmet token removes the section.
#   DEVICES   cpu: <name>.cpp. cuda: <name>.cu, needs ATV_BUILD_CUDA.
#             both: <name>.cu run with --device=both when CUDA is on,
#             otherwise <name>.cpp (absent when there is none).
#   THREADS   OMP/OpenBLAS width (default 1); the case runs at the widest.
#   MPIEXEC   the section relaunches itself through mpiexec (ATV_MPIEXEC).
#   GPUS      minimum visible GPUs (test label).
function(atv_regression)
    cmake_parse_arguments(R "MPIEXEC" "SECTION;DEVICES;THREADS;GPUS"
                          "REQUIRES;LIBS" ${ARGN})
    set(case ${_ATV_REG_DOMAIN}/${_ATV_REG_CASE})
    set(dir ${CMAKE_CURRENT_SOURCE_DIR}/${case})
    if(NOT R_SECTION)
        message(FATAL_ERROR "regression ${case}: SECTION is mandatory")
    endif()
    if(NOT R_REQUIRES)
        message(FATAL_ERROR "regression ${case}/${R_SECTION}: REQUIRES is mandatory (use NONE)")
    endif()
    foreach(token IN LISTS R_REQUIRES)
        if(token STREQUAL "NONE")
            continue()
        elseif(token STREQUAL "WHAM")
            set(met ${ATV_MOLRS_HAS_WHAM})
        elseif(token STREQUAL "FFTW_MPI")
            set(met ${FFTW3_MPI_LIB})
        elseif(DEFINED ATV_BUILD_${token})
            set(met ${ATV_BUILD_${token}})
        else()
            message(FATAL_ERROR "regression ${case}/${R_SECTION}: unknown REQUIRES token '${token}'")
        endif()
        if(NOT met)
            return()
        endif()
    endforeach()
    if(R_MPIEXEC AND NOT (ATV_USE_MPI AND MPIEXEC_EXECUTABLE))
        return()
    endif()

    if(NOT R_DEVICES)
        set(R_DEVICES cpu)
    endif()
    set(both OFF)
    if(R_DEVICES STREQUAL "cpu")
        set(source ${R_SECTION}.cpp)
    elseif(R_DEVICES STREQUAL "cuda")
        if(NOT ATV_BUILD_CUDA)
            return()
        endif()
        set(source ${R_SECTION}.cu)
    elseif(R_DEVICES STREQUAL "both")
        if(ATV_BUILD_CUDA)
            set(source ${R_SECTION}.cu)
            set(both ON)
        elseif(EXISTS ${dir}/${R_SECTION}.cpp)
            set(source ${R_SECTION}.cpp)
        else()
            return()
        endif()
    else()
        message(FATAL_ERROR "regression ${case}/${R_SECTION}: DEVICES must be cpu|cuda|both")
    endif()
    if(NOT EXISTS ${dir}/${source})
        message(FATAL_ERROR "regression ${case}/${R_SECTION}: missing ${source}")
    endif()
    if(NOT R_THREADS)
        set(R_THREADS 1)
    endif()

    set_property(GLOBAL APPEND PROPERTY ATV_REG_SOURCES ${dir}/${source})
    set_property(GLOBAL APPEND PROPERTY ATV_REG_LIBS ${R_LIBS})
    if(R_MPIEXEC)
        set_property(GLOBAL PROPERTY ATV_REG_MPIEXEC ON)
    endif()
    # Accumulate the case: widest thread count, sections run with
    # --device=both, GPU label.
    set(_atv_case_enabled ON PARENT_SCOPE)
    if(R_THREADS GREATER _atv_case_threads)
        set(_atv_case_threads ${R_THREADS} PARENT_SCOPE)
    endif()
    if(both)
        set(_atv_case_both ${_atv_case_both} ${R_SECTION} PARENT_SCOPE)
    endif()
    if(R_GPUS)
        set(_atv_case_gpus ${R_GPUS} PARENT_SCOPE)
    endif()
endfunction()

# Register one discovered method directory as one ctest test.
function(atv_regression_case_test case)
    add_test(NAME ${case}
             COMMAND $<TARGET_FILE:atv_regressions> --case ${case})
    set(labels regression)
    if(_atv_case_gpus)
        list(APPEND labels "gpus=${_atv_case_gpus}")
    endif()
    list(JOIN _atv_case_both ":" both)
    set_tests_properties(${case} PROPERTIES
        LABELS "${labels}"
        PROCESSORS ${_atv_case_threads}
        ENVIRONMENT "OMP_NUM_THREADS=${_atv_case_threads};OPENBLAS_NUM_THREADS=${_atv_case_threads};ATV_REG_BOTH_SECTIONS=${both}")
endfunction()

file(GLOB _atv_reg_cases RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*/*/case.cmake)
list(SORT _atv_reg_cases)
foreach(_atv_reg_file IN LISTS _atv_reg_cases)
    get_filename_component(_atv_reg_path ${_atv_reg_file} DIRECTORY)
    get_filename_component(_ATV_REG_CASE ${_atv_reg_path} NAME)
    get_filename_component(_ATV_REG_DOMAIN ${_atv_reg_path} DIRECTORY)
    set(_atv_case_enabled OFF)
    set(_atv_case_threads 1)
    set(_atv_case_both "")
    set(_atv_case_gpus "")
    include(${CMAKE_CURRENT_SOURCE_DIR}/${_atv_reg_file})
    if(_atv_case_enabled)
        atv_regression_case_test(${_ATV_REG_DOMAIN}/${_ATV_REG_CASE})
    endif()
endforeach()

# One executable for every enabled case. Unity batches parse the shared
# headers once per batch; each case lives in its own namespace, so batches
# cannot collide.
get_property(_atv_reg_sources GLOBAL PROPERTY ATV_REG_SOURCES)
get_property(_atv_reg_libs GLOBAL PROPERTY ATV_REG_LIBS)
list(REMOVE_DUPLICATES _atv_reg_libs)
add_executable(atv_regressions harness/main.cpp ${_atv_reg_sources})
target_include_directories(atv_regressions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(atv_regressions PRIVATE atomiverse ${_atv_reg_libs})
target_compile_definitions(atv_regressions PRIVATE
    ATV_REG_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
get_property(_atv_reg_mpiexec GLOBAL PROPERTY ATV_REG_MPIEXEC)
if(_atv_reg_mpiexec)
    target_compile_definitions(atv_regressions PRIVATE
        ATV_MPIEXEC="${MPIEXEC_EXECUTABLE}")
endif()
set_target_properties(atv_regressions PROPERTIES
    CXX_STANDARD 20
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16)
foreach(_src IN LISTS _atv_reg_sources)
    if(_src MATCHES "\\.cu$")
        set_source_files_properties(${_src} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
        set_target_properties(atv_regressions PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
endforeach()

add_custom_target(regressions
    COMMAND ${CMAKE_CTEST_COMMAND} -L regression --output-on-failure
    DEPENDS atv_regressions
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all end-to-end regression cases")
