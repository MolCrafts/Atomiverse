# Opt-in complete-method residency acceptance; never register with CTest.
include("${CMAKE_CURRENT_LIST_DIR}/../gpw-driver/register.cmake")
if(ATV_BUILD_CUDA AND ATV_BUILD_GPW)
    add_executable(gpw-residency "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
    target_link_libraries(gpw-residency PRIVATE atomiverse)
    target_compile_features(gpw-residency PRIVATE cxx_std_20)
    set_target_properties(gpw-residency PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
