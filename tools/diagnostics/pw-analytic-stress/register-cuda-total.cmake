# Opt-in complete scientific diagnostic; deliberately outside CTest.
if(ATV_BUILD_PW AND ATV_BUILD_CUDA)
    foreach(_target IN ITEMS pw-analytic-stress-cuda pw-analytic-stress-cuda-family pw-analytic-stress-cuda-fd pw-analytic-stress-cuda-uks pw-analytic-stress-cuda-domain)
        if(_target STREQUAL "pw-analytic-stress-cuda")
            set(_source cuda_total.cpp)
        elseif(_target STREQUAL "pw-analytic-stress-cuda-family")
            set(_source cuda_total_family.cpp)
        elseif(_target STREQUAL "pw-analytic-stress-cuda-fd")
            set(_source cuda_total_fd.cpp)
        elseif(_target STREQUAL "pw-analytic-stress-cuda-uks")
            set(_source cuda_uks_total.cpp)
        else()
            set(_source cuda_uks_domain.cpp)
        endif()
        add_executable(${_target} "${CMAKE_CURRENT_LIST_DIR}/${_source}")
        target_link_libraries(${_target} PRIVATE atomiverse)
        target_compile_features(${_target} PRIVATE cxx_std_20)
        target_compile_options(${_target} PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Werror=return-type>")
        set_target_properties(${_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        set(_cuda_total_symbols
            _ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace6finishERNS2_13DeviceDensityERNS2_16DeviceBandSolverERKNS_3cpu2pw8PWResultEPSt6vectorISt5arrayIdLm3EESaISF_EEPSE_IdLm6EEPNS0_5FrameE
            _ZN3ATV4cuda2pw8internal16DeviceBandSolver17solve_final_bandsERNS_3cpu2pw8internal11PWWorkspaceENS2_20FinalBandDestinationEi
            cudaMemcpyAsync
            _ZN3ATV4cuda2pw8internal23project_frozen_nonlocalENS2_28FrozenNonlocalDeviceGeometryENS2_20FrozenBandDeviceViewENS2_27FrozenNonlocalDeviceScratchEP13cublasContextP11CUstream_st
            _ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspaceC1ERKNS_3cpu5FrameERNS4_2pw8internal9PWProblemEiNS4_2xc6XCTypeENS_10electronic8SmearingEdSt8optionalINS2_20FixedSpinOccupationsEEP11CUstream_stbb
            _ZN3ATV4cuda2pw8internal9FFTPlan3DC1EiiiP11CUstream_st
            )
        if(_target STREQUAL "pw-analytic-stress-cuda-domain")
            set(_cuda_total_symbols
                _ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace6finishERNS2_13DeviceDensityERNS2_16DeviceBandSolverERKNS_3cpu2pw8PWResultEPSt6vectorISt5arrayIdLm3EESaISF_EEPSE_IdLm6EEPNS0_5FrameE
                _ZN3ATV4cuda2pw8internal16DeviceBandSolver17solve_final_bandsERNS_3cpu2pw8internal11PWWorkspaceENS2_20FinalBandDestinationEi
                cudaMemcpyAsync
                _ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace15commit_fallbackERNS2_13DeviceDensityERKNS_3cpu2pw8PWResultERKSt6vectorISt5arrayIdLm3EESaISD_EERKSC_IdLm6EEPNS0_5FrameE
            )
        endif()
        foreach(_symbol IN LISTS _cuda_total_symbols)
            target_link_options(${_target} PRIVATE "LINKER:--wrap=${_symbol}")
        endforeach()
    endforeach()
endif()
