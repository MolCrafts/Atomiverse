# ── regressions/ — out-of-library textbook end-to-end cases ──────────────
#
# See README.md for the rules. In short: each case is ONE flat main() that
# consumes the PUBLIC library only, checks a cited LITERATURE value, runs in
# <= 60 s, and exits non-zero on failure.
#
# These are deliberately NOT registered with add_test(): CLAUDE.md's rule is
# that tests/ is unit-only and "if it is registered, it runs". Regressions are
# end-to-end, so they get their own target instead:
#
#   cmake -B build -DATV_BUILD_REGRESSIONS=ON
#   cmake --build build --target regressions      # builds AND runs every case

add_custom_target(regressions
    COMMENT "Running all end-to-end textbook regression cases")

# atv_add_regression(<domain> <case>)
#   expects regressions/<domain>/<case>/reg_<case>.cpp
function(atv_add_regression domain case)
    set(target reg_${case})
    add_executable(${target} ${domain}/${case}/${target}.cpp)
    target_link_libraries(${target} PRIVATE atomiverse)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 17)

    add_custom_target(run_${target}
        COMMAND ${target}
        DEPENDS ${target}
        COMMENT "regression [${domain}/${case}]")
    add_dependencies(regressions run_${target})
endfunction()

if(ATV_BUILD_MD)
    atv_add_regression(md lj_argon)
    atv_add_regression(md eam_cu)
    atv_add_regression(md sw_si)
    atv_add_regression(md rigid_water)
    atv_add_regression(md dihedral_charmm)
    atv_add_regression(md improper_harmonic)
    atv_add_regression(md gaff_protocol_smoke)
endif()
if(ATV_BUILD_MC)
    atv_add_regression(mc gcmc_lj)
    atv_add_regression(mc npt_lj)
endif()
if(ATV_BUILD_PW)
    atv_add_regression(pw h_force_fd)
    atv_add_regression(pw h_force_analytic)
    atv_add_regression(pw h_stress_cubic)
endif()
if(ATV_BUILD_HF)
    atv_add_regression(scf h2o_rhf_sto3g)
    atv_add_regression(scf h2o_rhf_def2svp_sph)
    atv_add_regression(scf h2o_rhf_dipole)
    atv_add_regression(scf h2o_embedding_field)
    atv_add_regression(scf h2o_rhf_631gs_cpcm)
    atv_add_regression(scf h2o_charges_pyscf)
    atv_add_regression(scf n2_ionization)
    atv_add_regression(scf diffuse_anion)
    atv_add_regression(scf ecp_heavy_atom)
    atv_add_regression(scf h2o_scf_convergence)
    atv_add_regression(scf h2_rhf_stability)
    atv_add_regression(scf soscf_hard_case)
    atv_add_regression(scf o2_rohf_sto3g)
    atv_add_regression(scf h2o_ri_jk)
    atv_add_regression(scf h3_ghf_sto3g)
    atv_add_regression(scf h2o_rhf_freq)
    atv_add_regression(scf ch2o_tdhf)
    atv_add_regression(scf h2o_dimer_cp_hf631gs)
    atv_add_regression(scf x2c_argon)
endif()
if(ATV_BUILD_PATH)
    atv_add_regression(path neb_muller_brown)
    atv_add_regression(path remd_harmonic)
endif()
atv_add_regression(architecture harmonic_driver_matrix)
atv_add_regression(architecture module_compile_defs)
if(ATV_BUILD_MINIMIZER)
    atv_add_regression(minimizer lj_dimer)
endif()
if(ATV_BUILD_POTENTIALS)
    atv_add_regression(forcefield tip3p_two_water)
    atv_add_regression(forcefield gaff_peo_fragment)
endif()
if(ATV_BUILD_PATH)
    atv_add_regression(sampling umbrella_harmonic)
endif()
if(ATV_BUILD_SEMIEMPIRICAL)
    atv_add_regression(semiempirical am1_water_force)
    atv_add_regression(semiempirical am1_water_minimize)
    atv_add_regression(semiempirical am1_water_bomd)
    atv_add_regression(semiempirical rm1_water_energy)
endif()
if(ATV_BUILD_KSPACE AND ATV_BUILD_MD)
    atv_add_regression(md npt_pme_nacl)
endif()

# ── pure-Python import (python-on-demand-install-02) ──────────────────────
# No C++ link: isolated pure sources only. Needs a Python interpreter.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(run_reg_python_pure_import
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/python/pure_import/reg_pure_import.sh
                ${Python3_EXECUTABLE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/pure_import
        COMMENT "python pure_import: soft import without C++ extension"
        VERBATIM)
    add_dependencies(regressions run_reg_python_pure_import)

    add_custom_target(run_reg_python_install_cli
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/python/install_cli/reg_install_cli.sh
                ${Python3_EXECUTABLE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/install_cli
        COMMENT "python install_cli: help / mutual exclusion / hf / version lock"
        VERBATIM)
    add_dependencies(regressions run_reg_python_install_cli)
endif()

# ── launcher parity (mpi-launcher-embedded-python-04) ─────────────────────
# Not an atv_add_regression() case: it needs mpirun + a Python environment
# with atomiverse/molrs importable (the ~/work/.x86_64 venv per the spec).
# The launcher binary and interpreter come from the configured build.
if(ATV_BUILD_LAUNCHER)
    # regressions/ may be processed before app/atomiverse/ — resolve the
    # interpreter here too (cached, so this is a no-op when already found).
    find_package(Python3 COMPONENTS Interpreter QUIET)
    add_custom_target(run_reg_launcher_lj_parity
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/launcher/lj_parity/reg_lj_parity.sh
                ${Python3_EXECUTABLE}
                $<TARGET_FILE:atomiverse_launcher>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/launcher/lj_parity
        COMMENT "launcher lj_parity: serial-python vs 2-rank launcher (LJ + AM1)"
        VERBATIM)
    add_dependencies(run_reg_launcher_lj_parity atomiverse_launcher)
    add_dependencies(regressions run_reg_launcher_lj_parity)
endif()
