# The `atomiverse` MPI launcher: every rank native, rank 0 embeds CPython.
# ONLY this executable links libpython — libatomiverse and _atomiverse.so
# stay untouched (the wheel's serial `python atv.py` path is unaffected).
#
# The CMake target is `atomiverse_launcher` (the library target already owns
# the name `atomiverse`); the on-disk binary is still `atomiverse`.

find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development.Embed)

include(GNUInstallDirs)

add_executable(atomiverse_launcher
    main.cpp
    python_host.cpp
    interpreter_locator.cpp
)
set_target_properties(atomiverse_launcher PROPERTIES
    OUTPUT_NAME atomiverse
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    # 05-packaging: conda/prefix layouts keep libpython at ../lib; uv-managed
    # python-build-standalone venvs have NO lib/ dir, so also carry the
    # build-time libpython dir absolutely — the wheel is environment-bound
    # by design (never manylinux), so an absolute toolchain path is fair.
    INSTALL_RPATH "$ORIGIN/../lib;${Python3_LIBRARY_DIRS}")
target_link_libraries(atomiverse_launcher PRIVATE
    atomiverse
    Python3::Python
    MPI::MPI_CXX)
# Install: wheel builds land the REAL ELF in the wheel's scripts dir
# (pip copies binaries verbatim — the patchelf precedent — so it reaches
# the environment's bin/ on PATH, never a console-script shim); plain
# `cmake --install` uses the conventional bindir.
if(SKBUILD)
    install(TARGETS atomiverse_launcher
            RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}")
else()
    install(TARGETS atomiverse_launcher
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
