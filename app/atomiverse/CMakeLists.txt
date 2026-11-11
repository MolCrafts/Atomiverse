# The `atomiverse` driver binary.
#
# Built when either path is on:
# ATV_BUILD_MD: native LAMMPS-shaped `-in` runner (no Python, no MPI).
# ATV_BUILD_LAUNCHER: hosts Python scripts under MPI (rank 0 embeds CPython).
#
# CMake target is `atomiverse_cli` (the library already owns `atomiverse`);
# the on-disk name is `atomiverse`.
# The library never depends on this directory.

include(GNUInstallDirs)

set(_ATV_CLI_SRCS main.cpp)

if(ATV_BUILD_MD)
    list(APPEND _ATV_CLI_SRCS input_script.cpp)
endif()

if(ATV_BUILD_LAUNCHER)
    list(APPEND _ATV_CLI_SRCS python_host.cpp interpreter_locator.cpp)
endif()

add_executable(atomiverse_cli ${_ATV_CLI_SRCS})
set_target_properties(atomiverse_cli PROPERTIES
    OUTPUT_NAME atomiverse
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

target_include_directories(atomiverse_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(atomiverse_cli PRIVATE atomiverse)

if(ATV_BUILD_LAUNCHER)
    find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development.Embed)
    target_compile_definitions(atomiverse_cli PRIVATE ATV_HAS_LAUNCHER=1)
    set_target_properties(atomiverse_cli PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib;${Python3_LIBRARY_DIRS}")
    target_link_libraries(atomiverse_cli PRIVATE Python3::Python MPI::MPI_CXX)
endif()

if(SKBUILD)
    install(TARGETS atomiverse_cli
            RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}" COMPONENT python)
else()
    install(TARGETS atomiverse_cli
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
