# Opt-in diagnostics only. Include after register.cmake defines _pw_final_state_wraps.
add_executable(pw-empty-beta "${CMAKE_CURRENT_LIST_DIR}/empty_beta.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/general_trace.cpp")
foreach(_symbol IN LISTS _pw_final_state_wraps)
    target_link_options(pw-empty-beta PRIVATE "LINKER:--wrap=${_symbol}")
endforeach()
foreach(_kind IN ITEMS component zero-paths tangent)
    string(REPLACE "-" "_" _source "${_kind}")
    add_executable(pw-empty-beta-${_kind} "${CMAKE_CURRENT_LIST_DIR}/empty_beta_${_source}.cpp")
    target_link_options(pw-empty-beta-${_kind} PRIVATE
        "LINKER:--wrap=_ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE")
endforeach()
foreach(_symbol IN ITEMS
    _ZN3ATV3cpu2pw8internal19extract_final_bandsERKNS2_9PWProblemERKNS1_2PWERKNS1_8PWResultEi
    _ZN3ATV3cpu2pw8internal33assemble_frozen_functional_virialERKNS2_9PWProblemERKNS2_14FinalBandStateENS0_2xc6XCTypeERKNS2_14EwaldSummationERNS2_9XCBuilderE
    _ZN3ATV3cpu2pw8internal23pw_stress_analytic_implERKNS0_5FrameERKNS1_2PWEPNS1_8PWResultEPNS2_9PWProblemE)
    target_link_options(pw-empty-beta-tangent PRIVATE "LINKER:--wrap=${_symbol}")
endforeach()
foreach(_target IN ITEMS pw-empty-beta pw-empty-beta-component pw-empty-beta-zero-paths pw-empty-beta-tangent)
    target_link_libraries(${_target} PRIVATE atomiverse)
    target_compile_features(${_target} PRIVATE cxx_std_20)
    set_target_properties(${_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()
