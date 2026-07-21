# Exact Atomiverse -> molrs CXX consumer contract.
#
# This is Atomiverse's single CMake-readable source of truth. The actual molrs
# version lives in molrs-cxxapi/CXX_API_VERSION and is checked even when a
# developer supplies ATV_MOLRS_PATH.
#
# Pin policy:
#   - molrs is a **git submodule** at ./molrs (.gitmodules → MolCrafts/molrs).
#     The recorded submodule commit is the authoritative source pin.
#   - ATV_MOLRS_CXX_API_VERSION must match molrs-cxxapi/CXX_API_VERSION exactly.
#   - ATV_MOLRS_GIT_REF documents the intended release tag for that pin
#     (informational; bump it when advancing the submodule to a new tag).
#   - Local -DATV_MOLRS_PATH overrides do not bypass the CXX API version check.
set(ATV_MOLRS_CXX_API_VERSION 1)
set(ATV_MOLRS_CXX_REQUIRED_CAPABILITIES 7) # Frame-block v2 | Element | AM1-BCC
set(ATV_MOLRS_GIT_REF "v0.8.2" CACHE STRING
    "Documented MolCrafts/molrs tag matching the ./molrs submodule pin")

function(atv_validate_molrs_source_contract molrs_root)
    set(version_file "${molrs_root}/molrs-cxxapi/CXX_API_VERSION")
    if(NOT CMAKE_SCRIPT_MODE_FILE)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     "${version_file}")
    endif()
    if(NOT EXISTS "${version_file}")
        message(FATAL_ERROR
            "Atomiverse requires molrs CXX API v${ATV_MOLRS_CXX_API_VERSION}, "
            "but ${version_file} does not exist. ATV_MOLRS_PATH='${molrs_root}' "
            "is not a compatible molrs source tree. "
            "If using the default path, run: "
            "git submodule update --init --recursive molrs")
    endif()

    file(READ "${version_file}" actual_version)
    string(STRIP "${actual_version}" actual_version)
    if(NOT actual_version MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "Invalid molrs CXX API version in ${version_file}: "
            "expected one unsigned decimal integer, found '${actual_version}'.")
    endif()
    if(NOT actual_version STREQUAL "${ATV_MOLRS_CXX_API_VERSION}")
        message(FATAL_ERROR
            "molrs CXX API mismatch: Atomiverse requires v"
            "${ATV_MOLRS_CXX_API_VERSION}; found v${actual_version} at "
            "ATV_MOLRS_PATH='${molrs_root}'. Local path overrides do not "
            "bypass the contract.")
    endif()
endfunction()
