# Exact Atomiverse -> molrs CXX consumer contract.
#
# This is Atomiverse's single CMake-readable source of truth. The actual molrs
# version lives in molrs-cxxapi/CXX_API_VERSION and is checked even when a
# developer supplies ATV_MOLRS_PATH.
#
# Pin policy:
#   - molrs is a **git submodule** at ./molrs (.gitmodules → MolCrafts/molrs).
#     The recorded submodule commit (gitlink mode 160000) is the CI source pin
#     (cmake/AtvUnitGate.cmake configures without ATV_MOLRS_PATH, so it always
#     builds the submodule). Always: `git submodule update --init --recursive
#     molrs`.
#   - ATV_MOLRS_PATH defaults to ${CMAKE_SOURCE_DIR}/molrs (the submodule).
#     Every developer preset in CMakePresets.json sets it to the sibling
#     MolCrafts checkout ${sourceDir}/../molrs instead: molrs' dev branch is
#     consumed with UNCOMMITTED changes in its working tree (the streaming
#     TrajectoryWriter, release-0-14-15+), which no gitlink can capture. The
#     gitlink tracks the newest committed dev commit the preset build was
#     verified against.
#   - ATV_MOLRS_CXX_API_VERSION must match molrs-cxxapi/CXX_API_VERSION exactly.
#   - ATV_MOLRS_CXX_REQUIRED_CAPABILITIES is the mask Atomiverse cannot run
#     without; bit 3 (CXX_CAP_TRAJECTORY_WRITER, 0x8) is optional and detected
#     per pin (atv_detect_molrs_trajectory_writer) so the submodule pin still
#     links.
#   - ATV_MOLRS_GIT_REF documents the intended release / commit for that pin
#     (informational; bump it when advancing the submodule gitlink).
#   - -DATV_MOLRS_PATH overrides (preset or manual) never bypass the CXX API
#     version check.
set(ATV_MOLRS_CXX_API_VERSION 1)
set(ATV_MOLRS_CXX_REQUIRED_CAPABILITIES 7) # Frame-block v2 | Element | AM1-BCC
set(ATV_MOLRS_GIT_REF "f33996297e3bafa2c4d2e7f1288552514699886a" CACHE STRING
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

# Optional molrs surface: the streaming `*.mrec` trajectory writer
# (`TrajectoryWriterRef`, CXX capability bit 3). Detect from the bridge SOURCE
# (`build.rs` generates `src/bridge.rs`) so cpu::MrecRecorder and its test are
# compiled only against a pin that declares it, and a pin that gains it
# re-triggers configure.
function(atv_detect_molrs_trajectory_writer molrs_root out_var)
    set(_bridge_src "${molrs_root}/molrs-cxxapi/build.rs")
    if(NOT CMAKE_SCRIPT_MODE_FILE)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     "${_bridge_src}")
    endif()
    set(_has OFF)
    if(EXISTS "${_bridge_src}")
        file(READ "${_bridge_src}" _bridge_txt)
        if(_bridge_txt MATCHES "fn trajectory_writer_create")
            set(_has ON)
        endif()
    endif()
    set(${out_var} ${_has} PARENT_SCOPE)
endfunction()
