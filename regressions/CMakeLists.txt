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

# atv_add_regression(<domain> <case> [THREADS <n>])
#   expects regressions/<domain>/<case>/reg_<case>.cpp
#
# THREADS <n> — the case's DECLARED thread requirement, default **1**.
#   A regression is a textbook twin of a handful of atoms: for almost all of
#   them the OpenMP fork/join cost dwarfs the physics, and on a fat node that
#   is not a rounding error but the whole wall (mc_bias_es: 7 s at 1 thread,
#   231 s at 8, no completion at the node default of 48 — bit-identical
#   results at every count). Leaving the count ambient made a case's wall a
#   property of whichever machine happened to run it.
#   A few cases have the opposite shape (GPW self-consistency over a real
#   grid) and are genuinely faster wide; those declare `THREADS <n>` HERE,
#   next to the case, so the requirement is reviewable in one file instead of
#   hiding inside a .cpp. A case needing a specific count for physics reasons
#   may still call `set_threads()` — omp_set_num_threads() overrides the
#   environment, so the in-case choice still wins.
#
#   Measured 2026-07-28 (48-core epyc login node), 1-thread wall / 8-thread wall:
#     gpw/*                8x  (gpw-forces-water >70 s -> 14 s; kpoints-si -> 14 s)
#     scf/c10h22_cfmm_j    8x  (80 s -> 10 s)
#     dynamics/cpmd-*      2x  (53 s -> 27 s; cpmd-uks 125 s -> 57 s)
#     pw/*, response/dfpt-*  1.0x — the plane-wave and DFPT solvers do NOT
#         thread on these paths (dfpt-spin-smearing was 2x SLOWER at 8), so
#         they stay at 1: extra width there is footprint with no wall benefit.
#   Do not raise a case on a hunch — measure both widths first.
#
#   No case is pinned to 1 for CORRECTNESS any more. second-gen-cpmd-ot,
#   second-gen-cpmd-uks-ot and periodic-ugw-kpoints were, because the GPW grid
#   kernels reduced in whatever order OpenMP handed them: the KS matrix differed
#   run to run at a fixed thread count, and the OT line search amplified that
#   into a 30-200 iteration spread (1 run in 8 of ot-minimizer-water exhausted
#   its SCF budget outright). Those reductions now use a fixed partition folded
#   in a fixed order, so every one of them is bitwise identical at 1/2/4/8/16
#   threads — pinned by tests/cpu/gpw/test_cpu_gpw_thread_reproducibility.cpp.
#   Measured 2026-07-29, 1-thread -> 8-thread: second-gen-cpmd-ot 107 s -> 30 s,
#   second-gen-cpmd-uks-ot 277 s -> 75 s, periodic-ugw-kpoints 24 s -> 16 s.
#   A thread count is once again a pure cost choice.
function(atv_add_regression domain case)
    cmake_parse_arguments(REG "" "THREADS" "" ${ARGN})
    if(NOT DEFINED REG_THREADS)
        set(REG_THREADS 1)
    endif()

    set(target reg_${case})
    add_executable(${target} ${domain}/${case}/${target}.cpp)
    target_link_libraries(${target} PRIVATE atomiverse)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20)

    # Absolute, compiled-in data root. A case must resolve its inputs the same
    # way from the build tree, the source tree, or a bare `./reg_<case>` under
    # a debugger — never by probing the current working directory. NOTE: we
    # deliberately do NOT set WORKING_DIRECTORY here; that would paper over a
    # CWD-relative path in the aggregate target while leaving every standalone
    # run broken, which is exactly how this defect stayed invisible.
    target_compile_definitions(${target} PRIVATE
        ATV_REG_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")

    add_custom_target(run_${target}
        COMMAND ${CMAKE_COMMAND} -E env
                OMP_NUM_THREADS=${REG_THREADS}
                OPENBLAS_NUM_THREADS=${REG_THREADS}
                $<TARGET_FILE:${target}>
        DEPENDS ${target}
        COMMENT "regression [${domain}/${case}] (${REG_THREADS}t)")
    add_dependencies(regressions run_${target})
endfunction()

# Layout guarantee (always on): out-of-library consumer of the core contract
# headers — backend-mirror-01-core moved these from facade/ into core/.
atv_add_regression(layout core_contract_includes)
atv_add_regression(layout backend_owned_longrange)

if(ATV_BUILD_MD)
    # Layout: full-word backend include paths (backend-mirror-03-paths).
    atv_add_regression(layout path_words)
    atv_add_regression(layout driver_names)
    atv_add_regression(layout no_backend)
    atv_add_regression(layout architecture_gates)
    atv_add_regression(md lj_argon)
    atv_add_regression(md const-pot-bo-parallel-plate)
    atv_add_regression(md per-electrode-eta)
    atv_add_regression(md const-charge-electrode)
    atv_add_regression(md const-pot-cg-large-electrode)
    atv_add_regression(md electrode-charge-predictor)
    atv_add_regression(md pr_npt_lj)
    atv_add_regression(md nph_lj)
    atv_add_regression(md surface-tension-coupling)
    atv_add_regression(md pe_bonded_nve)
    atv_add_regression(md eam_cu)
    atv_add_regression(md sw_si)
    atv_add_regression(md rigid_water)
    atv_add_regression(md dihedral_charmm)
    atv_add_regression(md improper_harmonic)
    atv_add_regression(md gaff_protocol_smoke)
    atv_add_regression(md charging-work-energy)
    atv_add_regression(md gaussian-electrode-charge)
    atv_add_regression(md qeq-fluctuating-charge)
    atv_add_regression(md freeze-electrode-atoms)
    atv_add_regression(md piston-electrode-pz)
    atv_add_regression(md const-pot-charge-observer)
    atv_add_regression(md capacitance-charge-fluctuation)
    atv_add_regression(md potential-profile-observer)
    atv_add_regression(md external-efield-md)
    atv_add_regression(md constant-force-pull-ion)
    atv_add_regression(md oscillating-efield-md)
    atv_add_regression(md four-site-tip4p-foothold)
    atv_add_regression(md virtual-site-multi-topology)
    atv_add_regression(md finite-field-single-electrode)
    atv_add_regression(md pim-induced-dipole-bulk)
    atv_add_regression(md pim-npt-polarizable-il)
    atv_add_regression(md thomas-fermi-electrode)
    atv_add_regression(md maze-symplectic-charge-dynamics)
    atv_add_regression(md voronoi-volume-tf)
    atv_add_regression(md dipole-electrode-joint)
    # Ghost-charge transport after the const-pot Q solve under MPI DD:
    # self-relaunching binary (mpiexec -n 2), oracle=literature invariant.
    if(ATV_USE_MPI AND MPIEXEC_EXECUTABLE)
        atv_add_regression(md const-pot-charge-halo-mpi)
        target_compile_definitions(reg_const-pot-charge-halo-mpi PRIVATE
            ATV_MPIEXEC="${MPIEXEC_EXECUTABLE}")
    endif()
endif()
if(ATV_BUILD_MC)
    atv_add_regression(mc gcmc_lj)
    atv_add_regression(mc npt_lj)
    atv_add_regression(mc widom_hs)
    if(ATV_BUILD_HF)
        # Monte Carlo on an electronic structure (esmm-contract-04 R3): the
        # trial energy is a NESTED cpu::scf::RHF driver node, so the binary
        # needs the Gaussian SCF vtables compiled in — hence the HF nest.
        atv_add_regression(mc mc_bias_es)
    endif()
endif()
if(ATV_BUILD_PW)
    atv_add_regression(pw h_force_fd)
    atv_add_regression(pw h_force_analytic)
    atv_add_regression(pw h_stress_cubic)
    atv_add_regression(pw pw-lda-si)
    atv_add_regression(pw pw-analytic-stress-si)
    # In-tree ONCV Si UPF for NC-PW regressions (compile-time path pin).
    set(_atv_si_upf "${CMAKE_SOURCE_DIR}/tests/data/pw/Si_lda_nc.upf")
    if(TARGET reg_pw-lda-si)
        target_compile_definitions(reg_pw-lda-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    if(TARGET reg_pw-analytic-stress-si)
        target_compile_definitions(reg_pw-analytic-stress-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    if(ATV_BUILD_MINIMIZER)
        atv_add_regression(pw pw-variable-cell-si)
        if(TARGET reg_pw-variable-cell-si)
            target_compile_definitions(reg_pw-variable-cell-si PRIVATE
                ATV_PW_UPF_PATH="${_atv_si_upf}")
        endif()
    endif()
    # DFPT phonons at Γ (response): PW LSDA Si + Sternheimer response.
    atv_add_regression(response dfpt-phonon-si)
    if(TARGET reg_dfpt-phonon-si)
        target_compile_definitions(reg_dfpt-phonon-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Finite-q DFPT Sternheimer at X (response): same Si twin, q = (1/2,0,0).
    atv_add_regression(response dfpt-finite-q-si)
    if(TARGET reg_dfpt-finite-q-si)
        target_compile_definitions(reg_dfpt-finite-q-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Incommensurate-q DFPT: direct Sternheimer on primitive cell, q=(0.3,0,0).
    atv_add_regression(response dfpt-incommensurate-si)
    if(TARGET reg_dfpt-incommensurate-si)
        target_compile_definitions(reg_dfpt-incommensurate-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Γ DFPT with pure GGA (PBE) XC kernel (response).
    atv_add_regression(response dfpt-gga-si)
    if(TARGET reg_dfpt-gga-si)
        target_compile_definitions(reg_dfpt-gga-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Γ DFPT with collinear spin (LSDA, multiplicity=1 Si twin).
    atv_add_regression(response dfpt-spin-si)
    if(TARGET reg_dfpt-spin-si)
        target_compile_definitions(reg_dfpt-spin-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Γ DFPT joint collinear spin × pure GGA (PBE) kernel (response).
    atv_add_regression(response dfpt-gga-spin-si)
    if(TARGET reg_dfpt-gga-spin-si)
        target_compile_definitions(reg_dfpt-gga-spin-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Finite-q DFPT at X with pure GGA (PBE) kernel (response).
    atv_add_regression(response dfpt-gga-finite-q-si)
    if(TARGET reg_dfpt-gga-finite-q-si)
        target_compile_definitions(reg_dfpt-gga-finite-q-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Finite-q DFPT at X with joint collinear spin × pure GGA (PBE) kernel.
    atv_add_regression(response dfpt-gga-spin-finite-q-si)
    if(TARGET reg_dfpt-gga-spin-finite-q-si)
        target_compile_definitions(reg_dfpt-gga-spin-finite-q-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Γ DFPT with metallic occupation smearing (FD, restricted Si twin).
    atv_add_regression(response dfpt-metallic-si)
    if(TARGET reg_dfpt-metallic-si)
        target_compile_definitions(reg_dfpt-metallic-si PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Finite-q DFPT at X with metallic occupation smearing (FD).
    atv_add_regression(response dfpt-finite-q-smearing)
    if(TARGET reg_dfpt-finite-q-smearing)
        target_compile_definitions(reg_dfpt-finite-q-smearing PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Γ DFPT collinear spin + metallic smearing (FD, Si twin).
    atv_add_regression(response dfpt-spin-smearing)
    if(TARGET reg_dfpt-spin-smearing)
        target_compile_definitions(reg_dfpt-spin-smearing PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Finite-q DFPT collinear spin + metallic smear at X (FD).
    atv_add_regression(response dfpt-finite-q-spin-smearing)
    if(TARGET reg_dfpt-finite-q-spin-smearing)
        target_compile_definitions(reg_dfpt-finite-q-spin-smearing PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    # Incommensurate-q DFPT with collinear spin (direct Sternheimer).
    atv_add_regression(response dfpt-spin-incommensurate)
    if(TARGET reg_dfpt-spin-incommensurate)
        target_compile_definitions(reg_dfpt-spin-incommensurate PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}")
    endif()
    atv_add_regression(pw pw-pbe-gga-si)
    atv_add_regression(pw pw-spin-polarized-o2)
    atv_add_regression(pw pw-spin-gga)
    atv_add_regression(pw bands-dos-si)
    atv_add_regression(pw bands-spin-si)
    # Metallic occupation DOS (FD smear on diamond Si; integral to E_F).
    atv_add_regression(pw dos-metallic-si)
    # Metallic occupation bands (FD smear + high-symmetry path).
    atv_add_regression(pw bands-metallic-si)
    # Collinear spin + metallic smear bands (FD + nspin=2 diamond Si).
    atv_add_regression(pw bands-spin-metallic-si)
    # Collinear-spin DOS (fixed occ M=0 diamond Si; integral to joint VBM).
    atv_add_regression(pw dos-spin-si)
    # Collinear spin + metallic FD smear DOS (diamond Si; integral to E_F).
    atv_add_regression(pw dos-spin-metallic-si)
    atv_add_regression(pw berry-polarization)
    target_compile_definitions(reg_berry-polarization PRIVATE
        ATV_PW_UPF_PATH="${_atv_si_upf}")
    atv_add_regression(pw berry-spin-si)
    target_compile_definitions(reg_berry-spin-si PRIVATE
        ATV_PW_UPF_PATH="${_atv_si_upf}")
    atv_add_regression(pw periodic-wannier-si)
    target_compile_definitions(reg_periodic-wannier-si PRIVATE
        ATV_PW_UPF_PATH="${_atv_si_upf}")
    atv_add_regression(pw periodic-wannier-uks)
    target_compile_definitions(reg_periodic-wannier-uks PRIVATE
        ATV_PW_UPF_PATH="${_atv_si_upf}")
    atv_add_regression(pw pw-uks-kmesh-fe)
    atv_add_regression(pw pw-uks-fe-smearing)
    # MPI k-point round-robin scaling gate: the binary re-launches itself
    # through the configured mpiexec at np=1 and np=4 (parity + wall ratio).
    if(ATV_USE_MPI AND MPIEXEC_EXECUTABLE)
        atv_add_regression(pw mpi-pw-scaling-4rank)
        target_compile_definitions(reg_mpi-pw-scaling-4rank PRIVATE
            ATV_PW_UPF_PATH="${_atv_si_upf}"
            ATV_MPIEXEC="${MPIEXEC_EXECUTABLE}")
    endif()
    # pw-smearing-al merged when branch lands
    if(ATV_BUILD_MD)
        atv_add_regression(pw pw-bomd-h)
    endif()
    if(ATV_BUILD_MINIMIZER)
        atv_add_regression(pw h_cell_isotropic)
    atv_add_regression(pw pw-smearing-al)
    endif()
endif()
# FD supercell phonons (response): SW Si twin needs MD potentials; API is vib::Phonon.
if(ATV_BUILD_VIB AND ATV_BUILD_MD)
    atv_add_regression(pw fd-phonon-supercell)
    # Fourier ω(q) path from supercell IFCs (phonopy-style); same SW twin.
    atv_add_regression(response phonon-dispersion-si)
endif()
if(ATV_BUILD_GPW)
    atv_add_regression(gpw gpw-pbe-d3bj-ice THREADS 8)
    atv_add_regression(gpw gpw-d3-atm THREADS 8)
    atv_add_regression(gpw gpw-d4-water THREADS 8)
    atv_add_regression(gpw gpw-d4-atm THREADS 8)
    atv_add_regression(gpw gpw-d3-stress THREADS 8)
    atv_add_regression(gpw gpw-d3bj-stress THREADS 8)
    # Multi-k analytic cell stress with DFT-D3(BJ) (CP2K NUMERICAL Δσ + k-mesh).
    atv_add_regression(gpw gpw-d3bj-kpoint-stress THREADS 8)
    atv_add_regression(gpw gpw-d4-stress THREADS 8)
    atv_add_regression(gpw gpw-d4-forces THREADS 8)
    atv_add_regression(gpw gpw-d4-atm-forces THREADS 8)
    # Multi-k analytic ionic forces with D4 three-body ATM (s9); offline
    # CP2K 2026.1 libdftd4 ENERGY_FORCE FORCE Δ + dftd4 E_disp pin.
    atv_add_regression(gpw gpw-d4-atm-kpoint-forces THREADS 8)
    atv_add_regression(gpw gpw-d4-atm-stress THREADS 8)
    atv_add_regression(gpw gpw-ts-ice THREADS 8)
    atv_add_regression(gpw gpw-ts-forces THREADS 8)
    atv_add_regression(gpw gpw-ts-stress THREADS 8)
    atv_add_regression(gpw gpw-xdm-ice THREADS 8)
    atv_add_regression(gpw gpw-xdm-forces THREADS 8)
    atv_add_regression(gpw gpw-mbd-ice THREADS 8)
    atv_add_regression(gpw gpw-mbd-periodic THREADS 8)
    atv_add_regression(gpw gpw-mbd-forces THREADS 8)
    atv_add_regression(gpw gpw-mbd-kpoint-forces THREADS 8)
    atv_add_regression(gpw gpw-vdw-df-ice THREADS 8)
    atv_add_regression(gpw gpw-vdw-df2-ice THREADS 8)
    atv_add_regression(gpw gpw-rvv10-ice THREADS 8)
    atv_add_regression(gpw gpw-vdw-scf-potential THREADS 8)
    # Nonlocal vdW-DF on the REL_CUTOFF multigrid product path
    # (offline CP2K 2026.1 NGRIDS 4 pin — gpw-vdw-multigrid).
    atv_add_regression(gpw gpw-vdw-multigrid THREADS 8)
    # Nonlocal vdW-DF on the multi-k RKS path (Γ-limit identity + offline
    # CP2K 2026.1 KPOINTS + NON_LOCAL DRSLL pin — gpw-vdw-kpoints).
    atv_add_regression(gpw gpw-vdw-kpoints THREADS 8)
    # Γ collinear-UKS nonlocal vdW: V_nl on the total density in both spin
    # potentials + E_c^nl on e_disp (offline CP2K 2026.1 LSD DRSLL pin).
    atv_add_regression(gpw gpw-vdw-uks THREADS 8)
    # Multi-k collinear-UKS nonlocal vdW: dual-spin V_nl fold on the k path
    # + E_c^nl on e_disp (offline CP2K 2026.1 LSD + KPOINTS + DRSLL pin —
    # gpw-vdw-uks-kpoints).
    atv_add_regression(gpw gpw-vdw-uks-kpoints THREADS 8)
    # Γ UKS × DRSLL × REL_CUTOFF multigrid (NGRIDS 4) product pin — matched
    # offline CP2K 2026.1 NGRIDS+DRSLL+UKS twin (gpw-vdw-uks-multigrid).
    atv_add_regression(gpw gpw-vdw-uks-multigrid THREADS 8)
    # Multi-k UKS × DRSLL × REL_CUTOFF multigrid triple product pin — matched
    # offline CP2K 2026.1 NGRIDS+DRSLL+UKS+MP twin (gpw-vdw-uks-kpoint-multigrid).
    atv_add_regression(gpw gpw-vdw-uks-kpoint-multigrid THREADS 8)
    atv_add_regression(gpw gpw-optb88-vdw THREADS 8)
    # Γ collinear-UKS optB88-vdW (OptB88 + DRSLL) product pin — offline
    # CP2K 2026.1 LSD GGA_X_OPTB88_VDW + PW92 + NON_LOCAL DRSLL twin
    # (gpw-optb88-uks).
    atv_add_regression(gpw gpw-optb88-uks THREADS 8)
    # Multi-k RKS optB88-vdW (OptB88 + DRSLL on k path; offline CP2K
    # 2026.1 KPOINTS + GGA_X_OPTB88_VDW + DRSLL pin — gpw-optb88-kpoints).
    atv_add_regression(gpw gpw-optb88-kpoints THREADS 8)
    # Multi-k collinear-UKS optB88-vdW (OptB88 + DRSLL on k-UKS path;
    # offline CP2K 2026.1 LSD + KPOINTS + GGA_X_OPTB88_VDW + DRSLL pin —
    # gpw-optb88-uks-kpoints).
    atv_add_regression(gpw gpw-optb88-uks-kpoints THREADS 8)
    atv_add_regression(gpw gpw-mbd-hirshfeld THREADS 8)
    atv_add_regression(gpw gpw-hirshfeld-kpoints THREADS 8)
    atv_add_regression(gpw gpw-hirshfeld-uks THREADS 8)
    # Multi-k collinear UKS Hirshfeld volumes for TS/MBD (fold_dispersion_k_uks;
    # offline libMBD / literature pin — no stock CP2K TS/MBD).
    atv_add_regression(gpw gpw-hirshfeld-uks-kpoints THREADS 8)
    atv_add_regression(gpw ot-minimizer-water THREADS 8)
    atv_add_regression(gpw ot-uks-o2 THREADS 8)
    atv_add_regression(gpw ot-kpoints THREADS 8)
    atv_add_regression(gpw ot-smearing THREADS 8)
    atv_add_regression(gpw ot-uks-smearing THREADS 8)
    atv_add_regression(gpw ot-uks-kpoints THREADS 8)
    atv_add_regression(gpw ot-dftu THREADS 8)
    atv_add_regression(gpw ot-dftu-kpoints THREADS 8)
    atv_add_regression(gpw ot-uks-dftu THREADS 8)
    atv_add_regression(gpw ot-dftu-smearing THREADS 8)
    atv_add_regression(gpw ot-uks-dftu-smearing THREADS 8)
    atv_add_regression(gpw gpw-water-pbe-mt THREADS 8)
    atv_add_regression(gpw gpw-blyp-water THREADS 8)
    atv_add_regression(gpw gpw-rpbe-water THREADS 8)
    atv_add_regression(gpw gpw-forces-water THREADS 8)
    atv_add_regression(gpw gpw-stress-water THREADS 8)
    atv_add_regression(gpw gpw-analytic-stress THREADS 8)
    atv_add_regression(gpw gpw-uks-o2-gamma THREADS 8)
    atv_add_regression(gpw gpw-uks-pbe THREADS 8)
    atv_add_regression(gpw gpw-uks-blyp THREADS 8)
    atv_add_regression(gpw gpw-uks-revpbe THREADS 8)
    atv_add_regression(gpw gpw-uks-rpbe THREADS 8)
    atv_add_regression(gpw gpw-uks-forces THREADS 8)
    atv_add_regression(gpw gpw-uks-stress THREADS 8)
    atv_add_regression(gpw gpw-uks-multigrid THREADS 8)
    if(ATV_BUILD_MD)
        atv_add_regression(gpw gpw-bomd-water THREADS 8)
        # Plain multi-k BOMD (full SCF each step, previous-step P_k warm
        # start) — H2 NVE pinned vs CP2K 2026.1 Quickstep AIMD + KPOINTS.
        atv_add_regression(gpw gpw-bomd-kpoints THREADS 8)
        # Plain multi-k collinear UKS BOMD (dual P_k^σ warm start) — H2 NVE
        # pinned vs CP2K 2026.1 Quickstep UKS AIMD + KPOINTS.
        atv_add_regression(gpw gpw-bomd-uks-kpoints THREADS 8)
        # Extended-Lagrangian BOMD (dynamics domain): XL auxiliary-density
        # propagation on the same H2O NVE twin.
        atv_add_regression(dynamics xl-bomd-water THREADS 8)
        # XL-BOMD on Γ-UKS (dual α/β auxiliary densities) — O2 triplet NVE.
        atv_add_regression(dynamics xl-bomd-uks THREADS 8)
        # Multi-k RKS XL-BOMD (complex P_k aux densities) — H2 NVE.
        atv_add_regression(dynamics xl-bomd-kpoints THREADS 8)
        atv_add_regression(dynamics cpmd-kpoints THREADS 8)
        # Multi-k UKS XL-BOMD (dual-spin P_k^sigma) — high-spin H2 NVE.
        atv_add_regression(dynamics xl-bomd-uks-kpoints THREADS 8)
        # Zero-SCF linearized-functional XL-BOMD forces (shadow BO potential).
        atv_add_regression(dynamics xl-bomd-linearized-forces THREADS 8)
        # Zero-SCF linearized XL-BOMD on Γ-UKS (dual-spin shadow forces) —
        # O2 triplet NVE (identical twin of xl-bomd-uks + linearized tag).
        atv_add_regression(dynamics xl-bomd-linearized-uks THREADS 8)
        # Zero-SCF linearized XL-BOMD on the k path (complex shadow D_k) —
        # H2 NVE 1x1x2, SCF-XL vs linearized A/B + UKS degeneracy leg,
        # anchored to the CP2K 2026.1 converged-BOMD .ener trace.
        atv_add_regression(dynamics xl-bomd-linearized-kpoints THREADS 8)
        # Zero-SCF linearized XL-BOMD + analytic fixed-P shadow stress
        # (variable-cell/NPT-facing product path) — H2 NVE Γ + multi-k,
        # CP2K-anchored BO stress trace + dual gate + A/B shadow envelope.
        atv_add_regression(dynamics xl-bomd-linearized-stress THREADS 8)
        # Car–Parrinello MD (dynamics domain): fictitious electronic DOF on
        # the same H2O NVE twin family — not BOMD, not XL-BOMD.
        atv_add_regression(dynamics cpmd-water THREADS 8)
        # CPMD electronic RATTLE (constraint-preserving orbitals) — upgrade
        # of the Löwdin foothold on the same H2O NVE twin family.
        atv_add_regression(dynamics cpmd-rattle-water THREADS 8)
        # CPMD + Nosé–Hoover electronic thermostat on fictitious Ċ (H2O).
        atv_add_regression(dynamics cpmd-electronic-thermostat THREADS 8)
        # Multi-k CPMD + electronic NH per k channel (k-weighted Te/Eth) —
        # periodic H2 NVE 1x1x2, lift of the "NVE electrons only" k throw.
        atv_add_regression(dynamics cpmd-thermostat-kpoints THREADS 8)
        # CPMD on Γ-UKS (dual α/β fictitious electronic DOF) — O2 triplet NVE.
        atv_add_regression(dynamics cpmd-uks THREADS 8)
        # Multi-k UKS CPMD (dual-spin Bloch C_k^σ, one CpmdCplx per (k, spin))
        # — periodic H2 NVE, closed-shell fillings on the UKS path.
        atv_add_regression(dynamics cpmd-uks-kpoints THREADS 8)
        # Multi-k UKS CPMD + electronic NH per (k, σ) channel (k-weighted
        # Te/Eth) — combo pin of cpmd-uks-kpoints × cpmd-thermostat-kpoints.
        atv_add_regression(dynamics cpmd-uks-thermostat-kpoints THREADS 8)
        # Second-generation CP (Kühne ASPC + partial SCF corrector).
        atv_add_regression(dynamics second-gen-cpmd-water THREADS 8)
        # Second-generation CP on Γ-UKS (dual ASPC α/β + partial SCF).
        atv_add_regression(dynamics second-gen-cpmd-uks THREADS 8)
        # Second-generation CP with OT partial-SCF corrector (RKS Γ).
        atv_add_regression(dynamics second-gen-cpmd-ot THREADS 8)
        # Second-generation CP with OT partial-SCF corrector on Γ-UKS.
        atv_add_regression(dynamics second-gen-cpmd-uks-ot THREADS 8)
        # Multi-k second-generation CP (ASPC on complex P_k + partial SCF).
        atv_add_regression(dynamics second-gen-cpmd-kpoints THREADS 8)
        # Multi-k second-generation CP with OT partial-SCF corrector
        # (natural-orbital warm start of the per-k OT manifold).
        atv_add_regression(dynamics second-gen-cpmd-ot-kpoints THREADS 8)
        # Multi-k UKS second-generation CP with OT corrector (dual-spin
        # per-(k, σ) natural-orbital warm start from P_k^σ).
        atv_add_regression(dynamics second-gen-cpmd-ot-uks-kpoints THREADS 8)
    endif()
    atv_add_regression(gpw gpw-pbesol-si THREADS 8)
    atv_add_regression(gpw gpw-revpbe-si THREADS 8)
    atv_add_regression(gpw gpw-dftu-nio THREADS 8)
    atv_add_regression(gpw gpw-dftu-stress THREADS 8)
    atv_add_regression(gpw gpw-dftu-kpoints THREADS 8)
    atv_add_regression(gpw gpw-dftu-kpoint-stress THREADS 8)
    atv_add_regression(gpw gpw-dftu-kpoint-forces THREADS 8)
    atv_add_regression(gpw gpw-kpoints-si THREADS 8)
    # Multi-k RKS with CP2K REL_CUTOFF multigrid product collocation
    # (offline CP2K 2026.1 pin — gpw-multigrid-kpoints).
    atv_add_regression(gpw gpw-multigrid-kpoints THREADS 8)
    # Multi-k collinear UKS with the same REL_CUTOFF multigrid product path
    # (per-spin Bloch D/V stacks — gpw-uks-multigrid-kpoints).
    atv_add_regression(gpw gpw-uks-multigrid-kpoints THREADS 8)
    atv_add_regression(gpw periodic-wannier-gpw THREADS 8)
    # Collinear UKS multi-k Marzari–Vanderbilt MLWF (per-spin bond midpoints).
    atv_add_regression(gpw periodic-wannier-gpw-uks THREADS 8)
    # Dispersion ionic forces vs offline CP2K Quickstep D3 twins.
    atv_add_regression(gpw gpw-d3-forces THREADS 8)
    atv_add_regression(gpw gpw-d3bj-forces THREADS 8)
    # D3 three-body ATM (C9) force pin, isolated + periodic image triples.
    atv_add_regression(gpw gpw-d3-atm-forces THREADS 8)
    # D3 ATM (C9) cell-stress pin: CP2K numerical-stress difference (C9 on-off).
    atv_add_regression(gpw gpw-d3-atm-stress THREADS 8)
    # Multi-k analytic cell stress folding D3 three-body ATM (C9).
    atv_add_regression(gpw gpw-d3-atm-kpoint-stress THREADS 8)
    # Gamma analytic cell stress folding periodic MBD@rsSCS.
    atv_add_regression(gpw gpw-mbd-stress THREADS 8)
    # Multi-k analytic cell stress folding periodic MBD@rsSCS.
    atv_add_regression(gpw gpw-mbd-kpoint-stress THREADS 8)
    # Multi-k analytic cell stress folding DFT-D3 zero damping (CP2K NUMERICAL
    # stress-difference twin — dedicated pin after Γ gpw-d3-stress).
    atv_add_regression(gpw gpw-d3-kpoint-stress THREADS 8)
    # Multi-k analytic cell stress folding DFT-D4 (two-body BJ).
    atv_add_regression(gpw gpw-d4-kpoint-stress THREADS 8)
    # Gamma analytic cell stress folding free-atom XDM (dual-python strain pins).
    atv_add_regression(gpw gpw-xdm-stress THREADS 8)
    # Multi-k analytic cell stress folding free-atom XDM (dual-python strain pins).
    atv_add_regression(gpw gpw-xdm-kpoint-stress THREADS 8)
    atv_add_regression(gpw gpw-metallic-smearing THREADS 8)
    atv_add_regression(gpw gpw-uks-smearing THREADS 8)
    atv_add_regression(gpw gpw-kpoint-forces THREADS 8)
    atv_add_regression(gpw gpw-kpoint-uks-forces THREADS 8)
    atv_add_regression(gpw gpw-kpoint-stress THREADS 8)
    atv_add_regression(gpw gpw-kpoint-uks-stress THREADS 8)
    atv_add_regression(gpw gpw-kpoint-uks-stress-gga THREADS 8)
    atv_add_regression(gpw gpw-kpoints-uks THREADS 8)
    # FD vibrational spectrum from GPW analytic forces (Si2 UKS triplet,
    # isolated, centred MT). Dual-gate vs 1D energy curvature at the
    # ATV-convergent window (8 A / 150 Ry); offline CP2K 10 A/400 Ry pin
    # stays in REFERENCE for a future mixer-overlap twin.
    if(ATV_BUILD_VIB)
        atv_add_regression(gpw gpw-fd-phonon-si THREADS 8)
    endif()
    atv_add_regression(gpw gpw-variable-cell-ice THREADS 8)
    atv_add_regression(gpw gpw-variable-cell-molopt THREADS 8)
    atv_add_regression(gpw periodic-mp2-si THREADS 8)
    atv_add_regression(gpw periodic-ump2-si THREADS 8)
    atv_add_regression(gpw periodic-ri-mp2-si THREADS 8)
    # Γ RI-GPW SCS-MP2 (Grimme c_OS=1.2 / c_SS=1/3) on sc-Si2 — offline CP2K
    # 2026.1 SCALE_S/SCALE_T pin (periodic-scs-mp2-si).
    atv_add_regression(gpw periodic-scs-mp2-si THREADS 8)
    atv_add_regression(gpw periodic-mp2-kpoints THREADS 8)
    atv_add_regression(gpw periodic-ump2-kpoints THREADS 8)
    atv_add_regression(gpw periodic-rpa-si THREADS 8)
    atv_add_regression(gpw periodic-rpa-kpoints THREADS 8)
    atv_add_regression(gpw periodic-rpa-finite-q THREADS 8)
    atv_add_regression(gpw periodic-urpa-si THREADS 8)
    atv_add_regression(gpw periodic-urpa-kpoints THREADS 8)
    atv_add_regression(gpw periodic-gw-si THREADS 8)
    atv_add_regression(gpw periodic-ugw-si THREADS 8)
    # Was pinned to 1 for CORRECTNESS (the UGW Newton QP solve missed its root
    # once in ~7 runs at 8 threads). The cause was the thread-dependent
    # reduction order in integrate_potential_k, now a fixed partition folded in
    # a fixed order; 24 s -> 16 s wide.
    atv_add_regression(gpw periodic-ugw-kpoints THREADS 8)
    # UGW@UKS RPBE spin XC residual (spectral G0W0; xc_grid_rpbe_spin).
    atv_add_regression(gpw ugw-uks-rpbe THREADS 8)
    atv_add_regression(gpw periodic-gw-kpoints THREADS 8)
    atv_add_regression(gpw periodic-gw-finite-q THREADS 8)
    # MPI k-point RR industrial gate (self-launches mpiexec when ATV_USE_MPI).
    if(ATV_USE_MPI AND MPIEXEC_EXECUTABLE)
        atv_add_regression(gpw mpi-gpw-scaling-4rank THREADS 8)
        target_compile_definitions(reg_mpi-gpw-scaling-4rank PRIVATE
            ATV_MPIEXEC="${MPIEXEC_EXECUTABLE}")
        # Parent self-launches mpiexec; link MPI for the --mpi-worker path.
        target_link_libraries(reg_mpi-gpw-scaling-4rank PRIVATE MPI::MPI_CXX)
    endif()
    # remaining green GPW e2e land with their branches
endif()
if(ATV_BUILD_HF)
    # Layout: SCF API hygiene — MolecularGrid ctor links + rys_roots throws
    # catchably (scf-review-05-api-hygiene). Needs the SCF headers, hence HF.
    atv_add_regression(layout scf_api_hygiene)
    # One case per physics pin; multi-device via --device=cpu|cuda|both
    # (PyTorch-style). When ATV_BUILD_CUDA=ON the .cu TU is the target so the
    # same binary can run both backends; otherwise the .cpp (cpu only).
    if(ATV_BUILD_CUDA)
        set(_reg_h2o reg_h2o_rhf_sto3g)
        add_executable(${_reg_h2o} scf/h2o_rhf_sto3g/${_reg_h2o}.cu)
        target_link_libraries(${_reg_h2o} PRIVATE atomiverse)
        set_target_properties(${_reg_h2o} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 20)
        target_compile_definitions(${_reg_h2o} PRIVATE
            ATV_REG_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
        add_custom_target(run_${_reg_h2o}
            COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
                    $<TARGET_FILE:${_reg_h2o}> --device=both
            DEPENDS ${_reg_h2o}
            COMMENT "regression [scf/h2o_rhf_sto3g] devices=both (1t)")
        add_dependencies(regressions run_${_reg_h2o})
        unset(_reg_h2o)
    else()
        atv_add_regression(scf h2o_rhf_sto3g)
    endif()
    atv_add_regression(scf h2o_rhf_def2svp_sph)
    if(ATV_BUILD_CUDA)
        set(_reg_h2o_g reg_h2o_rhf_def2svp_grad)
        add_executable(${_reg_h2o_g} scf/h2o_rhf_def2svp_grad/${_reg_h2o_g}.cu)
        target_link_libraries(${_reg_h2o_g} PRIVATE atomiverse)
        set_target_properties(${_reg_h2o_g} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 20)
        target_compile_definitions(${_reg_h2o_g} PRIVATE
            ATV_REG_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
        add_custom_target(run_${_reg_h2o_g}
            COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
                    $<TARGET_FILE:${_reg_h2o_g}> --device=both
            DEPENDS ${_reg_h2o_g}
            COMMENT "regression [scf/h2o_rhf_def2svp_grad] devices=both (1t)")
        add_dependencies(regressions run_${_reg_h2o_g})
        unset(_reg_h2o_g)
    else()
        atv_add_regression(scf h2o_rhf_def2svp_grad)
    endif()
    # Short BOMD: MD + RHF composition (no AimdDriver); multi-device.
    if(ATV_BUILD_CUDA)
        set(_reg_aimd reg_h2_rhf_sto3g_aimd)
        add_executable(${_reg_aimd} scf/h2_rhf_sto3g_aimd/${_reg_aimd}.cu)
        target_link_libraries(${_reg_aimd} PRIVATE atomiverse)
        set_target_properties(${_reg_aimd} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 20)
        target_compile_definitions(${_reg_aimd} PRIVATE
            ATV_REG_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
        add_custom_target(run_${_reg_aimd}
            COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
                    $<TARGET_FILE:${_reg_aimd}> --device=both
            DEPENDS ${_reg_aimd}
            COMMENT "regression [scf/h2_rhf_sto3g_aimd] devices=both (1t)")
        add_dependencies(regressions run_${_reg_aimd})
        unset(_reg_aimd)
    else()
        atv_add_regression(scf h2_rhf_sto3g_aimd)
    endif()
    atv_add_regression(scf h2o_rhf_dipole)
    atv_add_regression(scf wannier-centers)
    atv_add_regression(scf static-polariz-h2o)
    atv_add_regression(scf h2o_embedding_field)
    atv_add_regression(scf embed_h2o_grad)
    atv_add_regression(scf embed_h2o_mm_forces)
    atv_add_regression(scf field_h2o_grad)
    atv_add_regression(scf h2o_rhf_631gs_cpcm)
    atv_add_regression(scf ief-pcm-h2o)
    atv_add_regression(scf pcm-grad-h2o)
    atv_add_regression(scf h2o_charges_pyscf)
    atv_add_regression(scf n2_ionization)
    atv_add_regression(scf diffuse_anion)
    atv_add_regression(scf ecp_heavy_atom)
    atv_add_regression(scf ecp_i2_grad)
    atv_add_regression(scf h2o_scf_convergence)
    atv_add_regression(scf h2_rhf_stability)
    atv_add_regression(scf soscf_hard_case)
    atv_add_regression(scf o2_rohf_sto3g)
    atv_add_regression(scf o2_rohf_grad)
    atv_add_regression(scf roks-o2-sto3g)
    atv_add_regression(scf roks-grad-o2-sto3g)
    atv_add_regression(scf h2o_ri_jk)
    atv_add_regression(scf c10h22_cfmm_j THREADS 8)
    atv_add_regression(scf h3_ghf_sto3g)
    atv_add_regression(scf ghf-grad-h3-sto3g)
    atv_add_regression(scf h2o_rhf_freq)
    atv_add_regression(scf uhf-hessian-o2)
    atv_add_regression(scf rohf-hessian-o2)
    atv_add_regression(scf rks-hessian-h2o-sto3g)
    atv_add_regression(scf uks-hessian-o2-sto3g)
    atv_add_regression(scf h2o_rhf_ir)
    atv_add_regression(scf ch2o_tdhf)
    atv_add_regression(scf cisd-h2co-sto3g)
    atv_add_regression(scf adc2-h2co-sto3g)
    atv_add_regression(scf adc2x-h2co-sto3g)
    atv_add_regression(scf g0w0-h2-sto3g)
    atv_add_regression(scf direct-rpa-h2-sto3g)
    atv_add_regression(scf scf-review-01-linalg)
    atv_add_regression(scf gf2-h2-sto3g)
    atv_add_regression(scf bse-h2-sto3g)
    atv_add_regression(scf h2co_rks_tda)
    atv_add_regression(scf h2_rks_alda_tda)
    atv_add_regression(scf h2_full_grid_alda)
    atv_add_regression(scf h2-b3lyp-hybrid-tda)
    atv_add_regression(scf h2-b3lyp-rpa-tddft)
    # scf-review-06-scf-perf: restricted B3LYP/def2-SVP H2O single point through
    # the restricted-hybrid Fock build (the fused one-sweep jk_rhf path); pinned
    # PySCF 2.14.0 total energy, no live oracle at ctest time.
    atv_add_regression(scf b3lyp-fused-jk-h2o)
    atv_add_regression(scf uks-tda-o2-lsda)
    atv_add_regression(scf uks-full-lr-o2-lsda)
    atv_add_regression(scf spin-flip-tda-o2)
    atv_add_regression(scf sf-tddft-o2)
    atv_add_regression(scf delta-scf-n2)
    atv_add_regression(scf full-lr-tddft-h2-pbe)
    atv_add_regression(scf blyp-water-sto3g)
    atv_add_regression(scf b2plyp-h2o)
    atv_add_regression(scf d3bj-molecular-pbe)
    atv_add_regression(scf dft-d4-h2o-pbe)
    atv_add_regression(scf h2_analytic_tda_force)
    atv_add_regression(scf h2_tda_force)
    atv_add_regression(scf h2_tda_nac)
    atv_add_regression(scf h2_nac_vectors)
    atv_add_regression(scf h2_alda_vs_cis)
    atv_add_regression(scf h2o_mp2_sto3g)
    # scf-review-07-posthf-perf: H2O/STO-3G plain RMP2 end-to-end through the
    # block-assembled dense ERI tensor (IntegralWorkspace::eri_dense); pinned PySCF
    # 2.14.0 e_hf/e_corr, no live oracle at ctest time.
    atv_add_regression(scf posthf-eri-block-mp2)
    atv_add_regression(scf ccsd-h2-sto3g)
    atv_add_regression(scf ccsd-t-h2o-sto3g)
    atv_add_regression(scf mp4-h2-sto3g)
    atv_add_regression(scf cc2-h2-sto3g)
    atv_add_regression(scf casci-h2-sto3g)
    atv_add_regression(scf ci-engine-equiv-h2-sto3g)
    # scf-review-08-ccsd-spatial-diis: closed-shell CCSD spatial-vs-spin-orbital
    # E_corr parity + DIIS-vs-Jacobi fixed point on H2O/STO-3G, vs PySCF golden.
    atv_add_regression(scf ccsd-spatial-so-parity)
    atv_add_regression(scf casscf-h2-sto3g)
    atv_add_regression(scf nevpt2-h2-sto3g)
    atv_add_regression(scf eom-ee-ccsd)
    atv_add_regression(scf ea-adc2-h2-sto3g)
    atv_add_regression(scf ip-adc2-h2o-sto3g)
    atv_add_regression(scf eom-cc2-h2)
    atv_add_regression(scf eom-ea-ccsd)
    atv_add_regression(scf eom-ip-ccsd)
    atv_add_regression(scf eom-sf-ccsd-o2)
    atv_add_regression(scf h2o_ri_mp2)
    atv_add_regression(scf ump2-o2-sto3g)
    atv_add_regression(scf romp2-o2-sto3g)
    atv_add_regression(scf frozen-core-mp2-h2o)
    atv_add_regression(scf scs-mp2-h2o-sto3g)
    atv_add_regression(scf sos-mp2-h2o-sto3g)
    atv_add_regression(scf mp2-grad-h2o-sto3g)
    atv_add_regression(scf mp2-d3bj-forces-h2o-sto3g)
    atv_add_regression(scf ccsd-grad-h2o-sto3g)
    atv_add_regression(scf ccsd-d3bj-forces-h2o-sto3g)
    atv_add_regression(scf ump2-grad-o2-sto3g)
    atv_add_regression(scf mp3-h2-sto3g)
    atv_add_regression(scf h2o_dimer_cp_hf631gs)
    atv_add_regression(scf x2c_argon)
    # AO-basis Car–Parrinello MD (dynamics domain, esmm-cpmd-04): fictitious
    # electronic DOF on the Gaussian SCF methods. Needs the MD driver as well
    # as HF — the electrons are propagated on the integrator's clock, so these
    # only build when both product modules are on.
    if(ATV_BUILD_MD)
        # RHF/STO-3G H2 — seed / zero-iteration production / kConserved wiring.
        atv_add_regression(dynamics cpmd-rhf-h2)
        # RKS(LSDA)/STO-3G H2O — fixed-density KS branch + XC gradient.
        atv_add_regression(dynamics cpmd-rks-water)
        # UHF triplet O2/STO-3G — dual-channel (alpha + beta) fictitious DOF.
        atv_add_regression(dynamics cpmd-uhf-o2)
        # Extended-Lagrangian BOMD on the same seam (esmm-cpmd-05): the
        # auxiliary density is the SCF guess instead of the propagated
        # coefficients. RHF/STO-3G H2O — warm-step iteration reduction vs a
        # cold-start twin and a plain-BOMD (last-density reuse) twin.
        atv_add_regression(dynamics xl-bomd-rhf-water)
        # Born–Oppenheimer NVE on a purely dispersion-bound dimer: the witness
        # that `e_total_ += e_disp_` and the ScfGradient dispersion leg
        # describe ONE surface. Before the leg existed this case dissociated.
        atv_add_regression(scf bomd-d3bj-ar2-nve)
    endif()
endif()
if(ATV_BUILD_PATH)
    atv_add_regression(path neb_muller_brown)
    atv_add_regression(path remd_harmonic)
    atv_add_regression(path pimd_free_particle)
    atv_add_regression(path dimer_saddle)
    atv_add_regression(path string_muller_brown)
    atv_add_regression(path irc_muller_brown)
    atv_add_regression(path pimd_free_particle_estimators)
endif()
atv_add_regression(architecture harmonic_driver_matrix)
atv_add_regression(architecture module_compile_defs)
if(ATV_BUILD_HF AND ATV_BUILD_POTENTIALS AND ATV_BUILD_NEIGHBOR)
    # esmm-contract-05-accumulate-switch (R1): the first ES x MM co-bound case.
    # One cpu::Evaluate over one Frame — a He pair under PairLJCut plus a
    # nested cpu::scf::RHF H2 30 A away (the He sites are ghosts in the QM
    # Hamiltonian) — must report kEnergy = E_LJ(analytic) + E_RHF(literature).
    # Before the accumulate switch the electronic evaluator overwrote kEnergy
    # and the MM term was erased. HF for the QM source; POTENTIALS/NEIGHBOR for
    # the LJ pair style and its neighbor list (both default ON).
    atv_add_regression(architecture lj_rhf_coupled)
endif()
if(ATV_BUILD_MINIMIZER)
    atv_add_regression(minimizer lj_dimer)
    atv_add_regression(minimizer fire_lj_dimer)
    atv_add_regression(minimizer cell_lj_pressure)
    atv_add_regression(minimizer joint_lj_zero_p)
    atv_add_regression(minimizer aniso_lj_box)
    atv_add_regression(minimizer triclinic_lj_shear)
    if(ATV_BUILD_HF)
        # StrongWolfe line search over a NESTED RHF force source (the
        # Runtime::run_slot_ nested-node replay): H2 relaxes to the CCCBDB
        # RHF/STO-3G equilibrium bond length inside a frozen iteration bound.
        atv_add_regression(minimizer rhf_h2_linesearch)
    endif()
endif()
if(ATV_BUILD_POTENTIALS)
    atv_add_regression(forcefield tip3p_two_water)
    atv_add_regression(forcefield gaff_peo_fragment)
    atv_add_regression(forcefield lj_soft_dimer)
    atv_add_regression(forcefield coul_soft_dimer)
    atv_add_regression(forcefield yukawa_dimer)
    atv_add_regression(forcefield buckingham-pair-lammps)
    atv_add_regression(forcefield born-repulsion-foothold)
    atv_add_regression(forcefield fumi-tosi-nacl)
    atv_add_regression(forcefield steele-wall-argon)
endif()
if(ATV_BUILD_PATH)
    atv_add_regression(sampling umbrella_harmonic)
    atv_add_regression(sampling umbrella_angle)
    atv_add_regression(sampling umbrella_dihedral)
    atv_add_regression(sampling fep_soft_dimer)
    atv_add_regression(sampling metad_double_well)
    atv_add_regression(sampling metad_2d)
    # WHAM needs a molrs pin that ships WhamCompute; the library gates
    # src/cpu/observe/wham.cpp on the same condition, so the case must too.
    if(ATV_MOLRS_HAS_WHAM)
        atv_add_regression(sampling wham_double_well)
    endif()
    atv_add_regression(path hremd_soft_lambda)
endif()
if(ATV_BUILD_MD)
    atv_add_regression(md brownian_einstein)
    atv_add_regression(md settle_tip3p)
    atv_add_regression(md fene_wca_chain)
    atv_add_regression(md dpd_groot_warren)
    atv_add_regression(md dpd_ab_mix)
endif()
if(ATV_BUILD_SEMIEMPIRICAL)
    atv_add_regression(semiempirical am1_water_force)
    atv_add_regression(semiempirical am1-water-mopac-energy)
    atv_add_regression(semiempirical am1_water_minimize)
    atv_add_regression(semiempirical am1_water_bomd)
    atv_add_regression(semiempirical rm1_water_energy)
    atv_add_regression(semiempirical rm1-water-mopac-energy)
    atv_add_regression(semiempirical h2_dftb2)
    atv_add_regression(semiempirical dftb0-h2)
    atv_add_regression(semiempirical h2o_dftb2_mio)
    atv_add_regression(semiempirical dftb2-3ob-organic)
    atv_add_regression(semiempirical dftb2-analytic-forces)
    atv_add_regression(semiempirical dftb2-dftbplus-energy-pin)
    atv_add_regression(semiempirical dftb2-analytic-stress)
    atv_add_regression(semiempirical dftb2-analytic-forces-threeob)
    atv_add_regression(semiempirical dftb2-analytic-stress-threeob)
    atv_add_regression(semiempirical dftb3-water-mio)
    atv_add_regression(semiempirical dftb3-analytic-forces)
    atv_add_regression(semiempirical dftb3-periodic-forces)
    atv_add_regression(semiempirical h2o_gfn0)
    atv_add_regression(semiempirical spin-dftb2-o2)
    atv_add_regression(semiempirical dftb2-periodic-gamma)
    atv_add_regression(semiempirical dftb2-periodic-forces)
    atv_add_regression(semiempirical dftb2-kpoints)
    atv_add_regression(semiempirical dftb2-multi-k-forces)
    atv_add_regression(semiempirical dftb2-dispersion-d3)
    atv_add_regression(semiempirical dftb2-dispersion-d3-periodic)
    atv_add_regression(semiempirical dftb2-spin-analytic-forces)
    atv_add_regression(semiempirical dftb2-spin-analytic-stress)
    atv_add_regression(semiempirical td-dftb-h2)
    atv_add_regression(semiempirical dftb-plus-u)
    if(ATV_BUILD_VIB)
        atv_add_regression(semiempirical se-analytic-hessian)
    endif()
    atv_add_regression(multiscale qmmm_h2_ar)
    atv_add_regression(multiscale qmmm_opt_h2_ar)
    atv_add_regression(multiscale qmmm_oniom_h2_ar)
    if(ATV_BUILD_HF)
        # ES/link/polarizable embedding instantiate cpu::scf::RHF through the
        # QMMM library sources; without ATV_BUILD_HF the RHF vtable is not
        # compiled and these binaries cannot link (MD-only merge gate).
        atv_add_regression(multiscale qmmm_es_h2o_pyscf)
        atv_add_regression(multiscale qmmm_link_ch3oh)
        atv_add_regression(multiscale qmmm_pol_h2o)
        if(ATV_BUILD_MD)
            atv_add_regression(multiscale qmmm_md_conservation)
        endif()
    endif()
endif()
if(ATV_BUILD_KSPACE AND ATV_BUILD_MD)
    atv_add_regression(md npt_pme_nacl)
    atv_add_regression(md slab-2d-ewald-foothold)
    atv_add_regression(md const-pot-md-nvt-short)
    atv_add_regression(md global-electroneutrality-bo)
    atv_add_regression(md electrode-atom-forces)
    atv_add_regression(md maze-charge-dynamics)
    atv_add_regression(md gaussian-coul-neighborlist)
    atv_add_regression(md born-buck-neighborlist-bulk)
    atv_add_regression(md displacement-dfield-md)
    atv_add_regression(md constant-d-polarization-feedback)
    atv_add_regression(md const-pot-tip4p-edl)
    atv_add_regression(md const-pot-il-edl-box)
    atv_add_regression(md charge-charge-autocorr)
    atv_add_regression(md elastance-condition-observer)
    atv_add_regression(md constant-velocity-smd)
    atv_add_regression(md extended-lagrangian-dipoles)
    atv_add_regression(md nanoporous-carbon-const-pot)
    atv_add_regression(md maze-molten-salt-al)
    atv_add_regression(md thermo-integration-capacitance)
endif()
# FMM operator math is always-on (facade longrange/); no product-module gate.
atv_add_regression(longrange fmm_point_charges)

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
