# Numbered C++ examples. Each target is gated on every product module its
# source uses; binaries land in <build>/examples.

# atv_add_example(<NN_name> <condition>...) — builds NN_name.cpp when every
# listed option is ON.
function(atv_add_example name)
    foreach(opt IN LISTS ARGN)
        if(NOT ${opt})
            return()
        endif()
    endforeach()
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE atomiverse)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
endfunction()

atv_add_example(01_rhf                    ATV_BUILD_HF)
atv_add_example(02_open_shell             ATV_BUILD_HF)
atv_add_example(03_ionization             ATV_BUILD_HF)
atv_add_example(04_correlation            ATV_BUILD_HF)
atv_add_example(05_density_cube           ATV_BUILD_HF ATV_BUILD_EVALUATE)
atv_add_example(06_geometry_optimization  ATV_BUILD_HF ATV_BUILD_MINIMIZER)
atv_add_example(07_aimd                   ATV_BUILD_HF ATV_BUILD_MD)
atv_add_example(08_md_lj                  ATV_BUILD_MD ATV_BUILD_POTENTIALS
                                          ATV_BUILD_NEIGHBOR)
atv_add_example(09_pw                     ATV_BUILD_PW ATV_BUILD_EVALUATE)
atv_add_example(10_gpw                    ATV_BUILD_GPW ATV_BUILD_EVALUATE)
atv_add_example(11_am1                    ATV_BUILD_SEMIEMPIRICAL
                                          ATV_BUILD_MINIMIZER ATV_BUILD_EVALUATE)
