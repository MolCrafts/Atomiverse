# Install product headers using the same ownership gates as root source lists.
# Public counterparts of compiled sources and the shared contracts listed below
# remain available across directory boundaries. No disabled method is restored
# by following a conditional include in an enabled header.
function(atv_install_headers target)
    set(header_root "${CMAKE_SOURCE_DIR}/include/atomiverse")
    set(excluded)
    if(NOT ATV_BUILD_CUDA)
        list(APPEND excluded "^cuda/" "^cpu/frame/transfer\\.h$")
    endif()
    if(NOT ATV_BUILD_KOKKOS)
        list(APPEND excluded "^kokkos/")
    endif()
    if(NOT ATV_BUILD_MD)
        list(APPEND excluded "^(cpu|cuda)/dynamics/" "^cpu/electrodes/"
            "^cuda/communication/multi_gpu\\.cuh$")
    endif()
    if(NOT ATV_BUILD_MC)
        list(APPEND excluded "^cpu/montecarlo/")
    endif()
    if(NOT ATV_BUILD_MINIMIZER)
        list(APPEND excluded "^(cpu|cuda)/minimize/"
            "^cpu/task/internal/minimize_builders\\.h$")
    endif()
    if(NOT ATV_BUILD_VIB)
        list(APPEND excluded "^cpu/vibration/")
    endif()
    if(NOT ATV_BUILD_PATH)
        list(APPEND excluded "^(cpu|cuda)/(path|replica|bias)/")
    endif()
    if(NOT ATV_BUILD_HF)
        list(APPEND excluded "^(cpu|cuda)/scf/"
            "^cpu/observe/(multipole|lowdin|esp_charges|mulliken|density)\\.h$"
            "^cpu/record/(cube|density_cube_writer|orbital_cube_writer)\\.h$"
            "^cpu/multiscale/qmmm_electrostatic\\.h$"
            "^cpu/electronic/cpmd\\.h$"
            "^cuda/observe/mulliken\\.cuh$")
    endif()
    if(NOT ATV_BUILD_PW)
        list(APPEND excluded "^(cpu|cuda)/planewave/")
    endif()
    if(NOT ATV_BUILD_GPW)
        list(APPEND excluded "^(cpu|cuda)/gpw/" "^cpu/xc/"
            "^cpu/electronic/(car_parrinello|second_gen_cp|xl_bomd_force)\\.h$")
    endif()
    if(NOT ATV_BUILD_KSPACE)
        # Direct Ewald/FMM and common operator contracts are always compiled.
        list(APPEND excluded "^cpu/longrange/(pme[^/]*|pppm[^/]*)\\.h$"
            "^cuda/longrange/pme\\.cuh$")
    endif()
    if(NOT ATV_BUILD_POTENTIALS)
        list(APPEND excluded "^(cpu|cuda)/potentials/"
            "^cpu/task/internal/potential_builders\\.h$")
    endif()
    if(NOT ATV_BUILD_NEIGHBOR)
        list(APPEND excluded "^(cpu|cuda)/neighbor/")
    endif()
    if(NOT ATV_BUILD_EVALUATE)
        list(APPEND excluded "^(cpu|cuda)/evaluate/")
    endif()
    if(NOT ATV_BUILD_SEMIEMPIRICAL)
        list(APPEND excluded "^cpu/semiempirical/"
            "^cpu/multiscale/qmmm_(mechanical|oniom)\\.h$")
    endif()
    if(NOT ATV_BUILD_SEMIEMPIRICAL OR NOT ATV_BUILD_HF)
        list(APPEND excluded "^cpu/multiscale/qmmm_polarizable\\.h$")
    endif()
    if(NOT ATV_BUILD_ANALYSIS)
        list(APPEND excluded
            "^cpu/observe/(rdf|rdf_curve|msd|vacf|trajectory_metrics|wham)\\.h$"
            "^cuda/observe/rdf\\.cuh$")
    endif()
    if(NOT ATV_MOLRS_HAS_WHAM)
        list(APPEND excluded "^cpu/observe/wham\\.h$")
    endif()
    if(NOT ATV_BUILD_IO)
        # LAMMPS data/coefficient readers are in the always-on source set.
        list(APPEND excluded "^cpu/io/(frame_writer|extxyz)\\.h$"
            "^cpu/record/(xyz|xyz_recorder|mrec|mrec_recorder)\\.h$"
            "^cpu/minimize/(trajectory_writer|final_geometry)\\.h$")
    endif()
    if(NOT ATV_MOLRS_HAS_TRAJECTORY_WRITER)
        list(APPEND excluded "^cpu/record/(mrec|mrec_recorder)\\.h$")
    endif()
    if(NOT ATV_BUILD_MOLRS)
        list(APPEND excluded "^(cpu|cuda)/frame/molrs_exchange\\.(h|cuh)$"
            "^cpu/task/" "^cpu/multiscale/internal/classical_nonbonded\\.h$"
            "^cpu/potentials/(eam|eam_potential|sw)\\.h$" "^molrs\\.hpp$")
    endif()

    # These are data/protocol contracts, not method implementations. The core
    # Driver template names CUDA-free contracts even in CPU-only builds;
    # OutcomePolicy uses MC's transaction record independently of the MC driver.
    set(shared
        "^cuda/(component/(component|context)\\.cuh|device/timer\\.h|driver/driver_node\\.h|frame/(frame|box|types)\\.cuh)$"
        "^cpu/montecarlo/transaction\\.h$"
        "^cpu/neighbor/pair_list\\.h$"
        "^cpu/potentials/lj_type_coeff\\.h$")
    if(ATV_BUILD_PW)
        list(APPEND shared "^cpu/scf/xc_functional\\.h$")
    endif()
    if(ATV_BUILD_CUDA)
        # The synthetic SCF kernels are in the always-on CUDA source set.
        list(APPEND shared
            "^cuda/scf/(scf_workspace|internal/(synthetic_scf_cycle|convergence_check|density_builder|eigen_solver|hamiltonian|solver_context))\\.cuh$")
    endif()

    file(GLOB_RECURSE headers CONFIGURE_DEPENDS RELATIVE "${header_root}"
        "${header_root}/*.h" "${header_root}/*.cuh" "${header_root}/*.hpp")
    set(selected)
    foreach(header IN LISTS headers)
        set(omit FALSE)
        foreach(pattern IN LISTS excluded)
            if(header MATCHES "${pattern}")
                set(omit TRUE)
                break()
            endif()
        endforeach()
        foreach(pattern IN LISTS shared)
            if(header MATCHES "${pattern}")
                set(omit FALSE)
                break()
            endif()
        endforeach()
        if(NOT omit)
            list(APPEND selected "${header_root}/${header}")
        endif()
    endforeach()

    # Retain public counterparts of compiled sources.
    # This follows actual ownership, including shared implementations that live
    # under another method directory; no extra product is enabled by installation.
    get_target_property(sources ${target} SOURCES)
    foreach(source IN LISTS sources)
        if(NOT IS_ABSOLUTE "${source}")
            set(source "${CMAKE_SOURCE_DIR}/${source}")
        endif()
        file(RELATIVE_PATH source "${CMAKE_SOURCE_DIR}/src" "${source}")
        string(REGEX REPLACE "\\.[^.]+$" "" stem "${source}")
        foreach(extension h cuh hpp)
            if(EXISTS "${header_root}/${stem}.${extension}")
                list(APPEND selected "${header_root}/${stem}.${extension}")
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES selected)
    list(SORT selected)
    foreach(header IN LISTS selected)
        file(RELATIVE_PATH relative "${header_root}" "${header}")
        get_filename_component(directory "${relative}" DIRECTORY)
        install(FILES "${header}"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/atomiverse/${directory}")
    endforeach()
endfunction()
