# ── regressions/ — end-to-end capability pins ─────────────────────────────
#
# Contract: regressions/README.md. Every case directory carries a
# `case.cmake` that calls atv_regression(); this file only discovers them.
# Regressions are never registered with add_test() (CLAUDE.md iron law 1):
#
#   cmake --build build --target regressions       # build and run every case
#   cmake --build build --target run_reg_<case>    # one case

add_custom_target(regressions
    COMMENT "Running all end-to-end regression cases")

# atv_regression(REQUIRES <tokens…>|NONE [DEVICES cpu|cuda|both] [THREADS n]
#                [RANKS n] [GPUS n] [MPIEXEC] [LIBS targets…] [ARGS args…])
#
# Called from regressions/<domain>/<case>/case.cmake with _ATV_REG_DOMAIN and
# _ATV_REG_CASE set by the discovery loop below.
#   REQUIRES  product modules (ATV_BUILD_<X>) plus WHAM (molrs WhamCompute) and
#             FFTW_MPI (multi-rank FFTW); any unmet token removes the case.
#   DEVICES   cpu: reg_<case>.cpp. cuda: reg_<case>.cu, needs ATV_BUILD_CUDA.
#             both: reg_<case>.cu run with --device=both when CUDA is on,
#             otherwise reg_<case>.cpp (absent when there is none).
#   THREADS   OMP/OpenBLAS width (default 1); raise only on a measurement.
#   RANKS     run under mpiexec -n <n>.
#   MPIEXEC   the case relaunches itself through mpiexec; it gets ATV_MPIEXEC.
#   GPUS      minimum visible GPUs (recorded in the run comment).
function(atv_regression)
    cmake_parse_arguments(R "MPIEXEC" "DEVICES;THREADS;RANKS;GPUS"
                          "REQUIRES;LIBS;ARGS" ${ARGN})
    set(domain ${_ATV_REG_DOMAIN})
    set(case ${_ATV_REG_CASE})
    set(dir ${CMAKE_CURRENT_SOURCE_DIR}/${domain}/${case})
    if(NOT R_REQUIRES)
        message(FATAL_ERROR "regression ${domain}/${case}: REQUIRES is mandatory (use NONE)")
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
            message(FATAL_ERROR "regression ${domain}/${case}: unknown REQUIRES token '${token}'")
        endif()
        if(NOT met)
            return()
        endif()
    endforeach()
    if((R_MPIEXEC OR R_RANKS) AND NOT (ATV_USE_MPI AND MPIEXEC_EXECUTABLE))
        return()
    endif()

    if(NOT R_DEVICES)
        set(R_DEVICES cpu)
    endif()
    set(device_args "")
    if(R_DEVICES STREQUAL "cpu")
        set(source reg_${case}.cpp)
    elseif(R_DEVICES STREQUAL "cuda")
        if(NOT ATV_BUILD_CUDA)
            return()
        endif()
        set(source reg_${case}.cu)
    elseif(R_DEVICES STREQUAL "both")
        if(ATV_BUILD_CUDA)
            set(source reg_${case}.cu)
            set(device_args --device=both)
        elseif(EXISTS ${dir}/reg_${case}.cpp)
            set(source reg_${case}.cpp)
        else()
            return()
        endif()
    else()
        message(FATAL_ERROR "regression ${domain}/${case}: DEVICES must be cpu|cuda|both")
    endif()
    if(NOT EXISTS ${dir}/${source})
        message(FATAL_ERROR "regression ${domain}/${case}: missing ${source}")
    endif()
    if(NOT R_THREADS)
        set(R_THREADS 1)
    endif()

    set(target reg_${case})
    add_executable(${target} ${dir}/${source})
    target_link_libraries(${target} PRIVATE atomiverse ${R_LIBS})
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20)
    if(source MATCHES "\\.cu$")
        set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    # Absolute, compiled-in input roots: a case resolves its inputs the same
    # way from the build tree, the source tree or a bare ./reg_<case>.
    target_compile_definitions(${target} PRIVATE
        ATV_REG_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data"
        ATV_REG_CASE_DIR="${dir}")
    if(R_MPIEXEC)
        target_compile_definitions(${target} PRIVATE
            ATV_MPIEXEC="${MPIEXEC_EXECUTABLE}")
    endif()

    set(launcher "")
    if(R_RANKS)
        set(launcher ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${R_RANKS})
    endif()
    set(note "${R_THREADS}t")
    if(R_GPUS)
        string(APPEND note ", needs ${R_GPUS} GPUs")
    endif()
    add_custom_target(run_${target}
        COMMAND ${CMAKE_COMMAND} -E env
                OMP_NUM_THREADS=${R_THREADS} OPENBLAS_NUM_THREADS=${R_THREADS}
                ${launcher} $<TARGET_FILE:${target}> ${device_args} ${R_ARGS}
        DEPENDS ${target}
        COMMENT "regression [${domain}/${case}] (${note})"
        VERBATIM)
    add_dependencies(regressions run_${target})
endfunction()

file(GLOB _atv_reg_cases RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*/*/case.cmake)
list(SORT _atv_reg_cases)
foreach(_atv_reg_file IN LISTS _atv_reg_cases)
    get_filename_component(_atv_reg_path ${_atv_reg_file} DIRECTORY)
    get_filename_component(_ATV_REG_CASE ${_atv_reg_path} NAME)
    get_filename_component(_ATV_REG_DOMAIN ${_atv_reg_path} DIRECTORY)
    include(${CMAKE_CURRENT_SOURCE_DIR}/${_atv_reg_file})
endforeach()
