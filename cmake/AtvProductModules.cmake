# AtvProductModules.cmake — product-module options, auto-enable DAG, HAS macros.
#
# Always-on core (no product gate): core/, frame/, driver/, component/, io/,
# recorder/, observer/, async/, comm/, task/, toy scf/solve.cpp, direct Ewald.
#
# Product gates (default ON for monorepo/CI parity):
#   ATV_BUILD_MD, MC, MINIMIZER, VIB, PATH, SEMIEMPIRICAL, POTENTIALS, NEIGHBOR,
#   EVALUATE
#
# Auto-enable DAG (one-way, no cycles):
#   MD, MC, MINIMIZER → POTENTIALS + NEIGHBOR
#   PATH → MINIMIZER  (then inherits minimizer → potentials+neighbor)
#   KSPACE → POTENTIALS
# GPW → RHF ∧ PW, TORCH → CUDA, LAUNCHER → MPI∧PYTHON remain FATAL elsewhere.

option(ATV_BUILD_MD "Build classical MD domain (cpu/md, cuda/md)" ON)
option(ATV_BUILD_MC "Build Monte Carlo domain (cpu/mc)" ON)
option(ATV_BUILD_MINIMIZER "Build geometry optimization (cpu/minimizer, cuda/minimizer)" ON)
option(ATV_BUILD_VIB "Build vibrational analysis (cpu/vib)" ON)
option(ATV_BUILD_PATH "Build path methods + replica + bias (cpu/path, replica, bias)" ON)
option(ATV_BUILD_SEMIEMPIRICAL "Build NDDO/AM1/RM1 semiempirical domain" ON)
option(ATV_BUILD_POTENTIALS "Build real-space force-field potentials" ON)
option(ATV_BUILD_NEIGHBOR "Build neighbor-list infrastructure" ON)
option(ATV_BUILD_EVALUATE "Advertise Evaluate product surface (ATV_HAS_EVALUATE)" ON)

# ── Auto-enable DAG ──────────────────────────────────────────────────────────
macro(atv_force_on _opt _because)
    if(NOT ${_opt})
        set(${_opt} ON CACHE BOOL "Auto-enabled because ${_because}" FORCE)
        message(STATUS "Auto-enabled ${_opt}=ON because ${_because}")
    endif()
endmacro()

if(ATV_BUILD_MD)
    atv_force_on(ATV_BUILD_POTENTIALS "ATV_BUILD_MD")
    atv_force_on(ATV_BUILD_NEIGHBOR "ATV_BUILD_MD")
endif()
if(ATV_BUILD_MC)
    atv_force_on(ATV_BUILD_POTENTIALS "ATV_BUILD_MC")
    atv_force_on(ATV_BUILD_NEIGHBOR "ATV_BUILD_MC")
endif()
if(ATV_BUILD_MINIMIZER)
    atv_force_on(ATV_BUILD_POTENTIALS "ATV_BUILD_MINIMIZER")
    atv_force_on(ATV_BUILD_NEIGHBOR "ATV_BUILD_MINIMIZER")
endif()
if(ATV_BUILD_PATH)
    atv_force_on(ATV_BUILD_MINIMIZER "ATV_BUILD_PATH")
    # PATH → MINIMIZER may have just flipped minimizer; re-apply minimizer edges.
    atv_force_on(ATV_BUILD_POTENTIALS "ATV_BUILD_PATH→MINIMIZER")
    atv_force_on(ATV_BUILD_NEIGHBOR "ATV_BUILD_PATH→MINIMIZER")
endif()
if(ATV_BUILD_KSPACE)
    atv_force_on(ATV_BUILD_POTENTIALS "ATV_BUILD_KSPACE")
endif()

# ── Public compile definitions helper ───────────────────────────────────────
# Call after add_library(atomiverse): atv_apply_product_module_defs(atomiverse)
function(atv_apply_product_module_defs _target)
    if(ATV_BUILD_MD)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_MD)
    endif()
    if(ATV_BUILD_MC)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_MC)
    endif()
    if(ATV_BUILD_MINIMIZER)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_MINIMIZER)
    endif()
    if(ATV_BUILD_VIB)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_VIB)
    endif()
    if(ATV_BUILD_PATH)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_PATH)
    endif()
    if(ATV_BUILD_SEMIEMPIRICAL)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_SEMIEMPIRICAL)
    endif()
    if(ATV_BUILD_POTENTIALS)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_POTENTIALS)
    endif()
    if(ATV_BUILD_NEIGHBOR)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_NEIGHBOR)
    endif()
    if(ATV_BUILD_EVALUATE)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_EVALUATE)
    endif()
endfunction()

message(STATUS "ATV product modules: "
    "MD=${ATV_BUILD_MD} MC=${ATV_BUILD_MC} MINIMIZER=${ATV_BUILD_MINIMIZER} "
    "VIB=${ATV_BUILD_VIB} PATH=${ATV_BUILD_PATH} SE=${ATV_BUILD_SEMIEMPIRICAL} "
    "POTENTIALS=${ATV_BUILD_POTENTIALS} NEIGHBOR=${ATV_BUILD_NEIGHBOR} "
    "EVALUATE=${ATV_BUILD_EVALUATE}")
