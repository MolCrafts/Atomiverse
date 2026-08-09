# Exact Atomiverse -> molrs CXX consumer contract.
#
# This is Atomiverse's single CMake-readable source of truth. The actual molrs
# version lives in molrs-cxxapi/CXX_API_VERSION and is checked even when a
# developer supplies ATV_MOLRS_PATH.
#
# Pin policy:
#   - molrs is a **git submodule** at ./molrs (.gitmodules → MolCrafts/molrs).
#     The recorded submodule commit (gitlink mode 160000) is the authoritative
#     source pin. Always: `git submodule update --init --recursive molrs`.
#   - ATV_MOLRS_PATH defaults to ${CMAKE_SOURCE_DIR}/molrs (the submodule).
#     Overrides are for local experiments only; CI must use the submodule.
#   - ATV_MOLRS_CXX_API_VERSION must match molrs-cxxapi/CXX_API_VERSION exactly.
#   - ATV_MOLRS_GIT_REF documents the intended release / commit for that pin
#     (informational; bump it when advancing the submodule gitlink).
#   - Local -DATV_MOLRS_PATH overrides do not bypass the CXX API version check.
set(ATV_MOLRS_CXX_API_VERSION 1)
set(ATV_MOLRS_CXX_REQUIRED_CAPABILITIES 7) # Frame-block v2 | Element | AM1-BCC
set(ATV_MOLRS_GIT_REF "b2be0b836000278159d8354602932b1a64035451" CACHE STRING
    "Documented MolCrafts/molrs ref matching the ./molrs submodule pin (dev)")

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

# Optional molrs surface: multi-window WHAM (WhamCompute) is not yet on every
# pin. Detect from the CXX bridge source so Atomiverse compiles against the
# submodule even before WHAM lands in molrs.
function(atv_detect_molrs_wham molrs_root out_var)
    set(_bridge "${molrs_root}/molrs-cxxapi/src/bridge.rs")
    set(_has OFF)
    if(EXISTS "${_bridge}")
        file(READ "${_bridge}" _bridge_txt)
        if(_bridge_txt MATCHES "WhamCompute")
            set(_has ON)
        endif()
    endif()
    set(${out_var} ${_has} PARENT_SCOPE)
endfunction()
