# Relocatable public C++ package. Dependencies provide their own installed
# targets; generated Rust/CXX archives and headers belong to this install.
include(CMakePackageConfigHelpers)
set(_atv_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Atomiverse")

set(ATV_PACKAGE_FFTW_OMP OFF)
set(ATV_PACKAGE_FFTW_MPI OFF)
if(ATV_BUILD_GPW AND FFTW3_OMP_LIB AND OpenMP_CXX_FOUND)
    set(ATV_PACKAGE_FFTW_OMP ON)
endif()
if(ATV_BUILD_KSPACE AND ATV_USE_MPI AND FFTW3_MPI_LIB)
    set(ATV_PACKAGE_FFTW_MPI ON)
endif()

if(ATV_BUILD_MOLRS)
    install(FILES $<TARGET_FILE:molrs_cxx> $<TARGET_FILE:molrs_cxxapi-static>
            DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY "${_molrs_cxx_gen_dir}/include/"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/AtomiverseConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/AtomiverseConfig.cmake"
    INSTALL_DESTINATION "${_atv_package_dir}"
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AtomiverseConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT AtomiverseTargets NAMESPACE Atomiverse::
        DESTINATION "${_atv_package_dir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/AtomiverseConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/AtomiverseConfigVersion.cmake"
        DESTINATION "${_atv_package_dir}")
