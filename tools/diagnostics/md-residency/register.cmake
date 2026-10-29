# Opt-in CUDA MD residency audit and before/after trajectory comparison;
# never register with CTest. Configure any tree (also a frozen baseline) with
#   -DCMAKE_PROJECT_Atomiverse_INCLUDE=<repo>/tools/diagnostics/md-residency/register.cmake
# Targets are declared after the top-level CMakeLists has enabled CUDA and
# defined `atomiverse`.
function(_atv_md_residency_targets)
    if(NOT ATV_BUILD_CUDA)
        return()
    endif()
    set(here "${_atv_md_residency_dir}")
    add_executable(md-residency "${here}/main.cu")
    target_link_libraries(md-residency PRIVATE atomiverse)
    target_compile_features(md-residency PRIVATE cxx_std_20)
    set_target_properties(md-residency PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()
set(_atv_md_residency_dir "${CMAKE_CURRENT_LIST_DIR}")
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
               CALL _atv_md_residency_targets)
