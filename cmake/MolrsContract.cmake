# molrs source pin and optional-surface detection.
#
# molrs has no API version before 1.0.0: every commit may break the previous
# one, so there is no version/capability gate — the pinned tree either compiles
# against Atomiverse or it does not.
#
# Pin policy:
#   - molrs is a **git submodule** at ./molrs (.gitmodules → MolCrafts/molrs).
#     The recorded submodule commit (gitlink mode 160000) is the CI source pin
#     (cmake/AtvUnitGate.cmake configures without ATV_MOLRS_PATH, so it always
#     builds the submodule). Always: `git submodule update --init --recursive
#     molrs`.
#   - ATV_MOLRS_PATH defaults to ${CMAKE_SOURCE_DIR}/molrs (the submodule).
#     Every developer preset in CMakePresets.json sets it to the sibling
#     MolCrafts checkout ${sourceDir}/../molrs instead: molrs is consumed with
#     UNCOMMITTED changes in its working tree, which no gitlink can capture.
#     The gitlink tracks the newest committed molrs commit the preset build was
#     verified against.
#   - Surfaces that only some molrs trees carry (WHAM, the streaming
#     `*.mrec` writer `MrecWriterRef`) are detected per tree below, so the
#     submodule pin still links. The gitlink tracks molrs dev; see
#     `git ls-tree HEAD molrs` for the committed source pin.

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

# Optional molrs surface: the streaming `*.mrec` writer (`MrecWriterRef`).
# molrs 0.16 declares `fn mrec_writer_create` in the CXX bridge source.
function(atv_detect_molrs_trajectory_writer molrs_root out_var)
    set(_bridge_src "${molrs_root}/molrs-cxxapi/src/bridge.rs")
    if(NOT CMAKE_SCRIPT_MODE_FILE)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     "${_bridge_src}")
    endif()
    set(_has OFF)
    if(EXISTS "${_bridge_src}")
        file(READ "${_bridge_src}" _bridge_txt)
        if(_bridge_txt MATCHES "fn mrec_writer_create")
            set(_has ON)
        endif()
    endif()
    set(${out_var} ${_has} PARENT_SCOPE)
endfunction()
