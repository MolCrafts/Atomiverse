# Local merged molpy bridge capability detection. No revision pin is imposed.

function(atv_detect_molpy_wham molpy_root out_var)
    set(_bridge "${molpy_root}/molcxx/src/bridge.rs")
    set(_has OFF)
    if(EXISTS "${_bridge}")
        file(READ "${_bridge}" _bridge_txt)
        if(_bridge_txt MATCHES "WhamCompute")
            set(_has ON)
        endif()
    endif()
    set(${out_var} ${_has} PARENT_SCOPE)
endfunction()

# Optional molpy surface: the streaming `*.mrec` writer (`MrecWriterRef`).
# molcxx declares `fn mrec_writer_create` in the CXX bridge source.
function(atv_detect_molpy_trajectory_writer molpy_root out_var)
    set(_bridge_src "${molpy_root}/molcxx/src/bridge.rs")
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
