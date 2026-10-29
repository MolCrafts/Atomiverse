set(_pw_final_state_wraps
    _ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE
    _ZN3ATV3cpu2pw8internal19extract_final_bandsERKNS2_9PWProblemERKNS1_2PWERKNS1_8PWResultEi
    _ZN3ATV3cpu2pw8internal33assemble_frozen_functional_virialERKNS2_9PWProblemERKNS2_14FinalBandStateENS0_2xc6XCTypeERKNS2_14EwaldSummationERNS2_9XCBuilderE
    _ZN3ATV3cpu2pw8internal23pw_stress_analytic_implERKNS0_5FrameERKNS1_2PWEPNS1_8PWResultEPNS2_9PWProblemE
    _ZN3ATV3cpu2pw8internal20pw_forces_from_bandsERKNS0_5FrameERKNS2_9PWProblemERKNS2_14FinalBandStateEPKNS2_14EwaldSummationE
)

# Complete scientific solves are opt-in and deliberately outside CTest.
add_executable(pw-analytic-stress "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
target_link_libraries(pw-analytic-stress PRIVATE atomiverse)
target_compile_features(pw-analytic-stress PRIVATE cxx_std_20)
set_target_properties(pw-analytic-stress PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Link wrapping counts real internal calls without adding production hooks.
foreach(_target IN ITEMS pw-analytic-stress-general pw-analytic-stress-family)
    if(_target STREQUAL "pw-analytic-stress-general")
        set(_source general.cpp)
    else()
        set(_source family_matrix.cpp)
    endif()
    add_executable(${_target} "${CMAKE_CURRENT_LIST_DIR}/${_source}"
        "${CMAKE_CURRENT_LIST_DIR}/general_trace.cpp")
    target_link_libraries(${_target} PRIVATE atomiverse)
    target_compile_features(${_target} PRIVATE cxx_std_20)
    set_target_properties(${_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()

target_sources(pw-analytic-stress PRIVATE "${CMAKE_CURRENT_LIST_DIR}/general_trace.cpp")
foreach(_target IN ITEMS pw-analytic-stress pw-analytic-stress-general pw-analytic-stress-family)
    foreach(_symbol IN LISTS _pw_final_state_wraps)
        target_link_options(${_target} PRIVATE "LINKER:--wrap=${_symbol}")
    endforeach()
endforeach()

# Opt-in only; complete SCF diagnostics never join CTest.
add_executable(pw-final-state-completion "${CMAKE_CURRENT_LIST_DIR}/completion.cpp")
target_link_libraries(pw-final-state-completion PRIVATE atomiverse)
target_compile_features(pw-final-state-completion PRIVATE cxx_std_20)
set_target_properties(pw-final-state-completion PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(_symbol IN ITEMS
    _ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE
    _ZN3ATV3cpu2pw8internal19extract_final_bandsERKNS2_9PWProblemERKNS1_2PWERKNS1_8PWResultEi)
    target_link_options(pw-final-state-completion PRIVATE "LINKER:--wrap=${_symbol}")
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/register-cuda-total.cmake")

# Shared ensemble harness; complete scientific solves remain outside CTest.
foreach(_target IN ITEMS pw-multi-k-fd pw-gaussian-ensemble pw-fixed-k-ensemble pw-mp-ensemble pw-mp-rks-ensemble)
    if(_target STREQUAL "pw-multi-k-fd")
        set(_source multi_k_fd.cpp)
    elseif(_target STREQUAL "pw-gaussian-ensemble")
        set(_source gaussian_ensemble.cpp)
    elseif(_target STREQUAL "pw-fixed-k-ensemble")
        set(_source fixed_k_ensemble.cpp)
    elseif(_target STREQUAL "pw-mp-rks-ensemble")
        set(_source mp_rks_ensemble.cpp)
    else()
        set(_source mp_ensemble.cpp)
    endif()
    add_executable(${_target} "${CMAKE_CURRENT_LIST_DIR}/${_source}"
        "${CMAKE_CURRENT_LIST_DIR}/general_trace.cpp")
    target_link_libraries(${_target} PRIVATE atomiverse)
    target_compile_features(${_target} PRIVATE cxx_std_20)
    set_target_properties(${_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    foreach(_symbol IN LISTS _pw_final_state_wraps)
        target_link_options(${_target} PRIVATE "LINKER:--wrap=${_symbol}")
    endforeach()
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/register-empty-beta.cmake")
