# AtvProductModules.cmake — product-module options, auto-enable DAG, HAS macros.
#
# Always-on core (no product gate): core/, frame/ (cpu::Frame only), driver/,
# component/, async/, comm/, task/, toy scf/solve.cpp, direct Ewald,
# classical observers that do not call molrs (virial, profiles, …).
#
# Product gates (default ON for monorepo/CI parity):
#   ATV_BUILD_MD, MC, MINIMIZER, VIB, PATH, SEMIEMPIRICAL, POTENTIALS, NEIGHBOR,
#   EVALUATE, ANALYSIS, IO
#
# molrs (Rust CXX) is **derived**, not a peer product switch:
#   ATV_BUILD_ANALYSIS / IO / PYTHON / GPW / HF / PW / SEMIEMPIRICAL → enable MOLRS
#   User turns ANALYSIS (or IO/…) on/off; MOLRS follows. Never “MOLRS requires
#   ANALYSIS” — the product owns the feature, molrs is the implementation.
#
# Auto-enable DAG (one-way, no cycles):
#   MD, MC, MINIMIZER → POTENTIALS + NEIGHBOR
#   PATH → MINIMIZER  (then inherits minimizer → potentials+neighbor)
#   KSPACE → POTENTIALS
#   PYTHON → IO (exchange for nanobind)
#   ANALYSIS | IO | PYTHON | GPW | HF | PW | SEMIEMPIRICAL → MOLRS (derived)
# GPW → HF ∧ PW, LAUNCHER → MPI∧PYTHON remain FATAL elsewhere.

option(ATV_BUILD_MD "Build classical MD domain (cpu/dynamics, cuda/dynamics)" ON)
option(ATV_BUILD_MC "Build Monte Carlo domain (cpu/montecarlo)" ON)
option(ATV_BUILD_MINIMIZER "Build geometry optimization (cpu/minimizer, cuda/minimizer)" ON)
option(ATV_BUILD_VIB "Build vibrational analysis (cpu/vibration)" ON)
option(ATV_BUILD_PATH "Build path methods + replica + bias (cpu/path, replica, bias)" ON)
option(ATV_BUILD_SEMIEMPIRICAL "Build NDDO/AM1/RM1/DFTB semiempirical domain" ON)
option(ATV_BUILD_POTENTIALS "Build real-space force-field potentials" ON)
option(ATV_BUILD_NEIGHBOR "Build neighbor-list infrastructure" ON)
option(ATV_BUILD_EVALUATE "Advertise Evaluate product surface (ATV_HAS_EVALUATE)" ON)
option(ATV_BUILD_ANALYSIS
    "Trajectory analysis observers (RDF/MSD/VACF/WHAM) — enables molrs" ON)
option(ATV_BUILD_IO
    "File I/O + trajectory recorders (extxyz, XYZ/Zarr, from_molrs/to_molrs) — enables molrs" ON)

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
if(ATV_BUILD_PYTHON)
    atv_force_on(ATV_BUILD_IO "ATV_BUILD_PYTHON (Frame exchange)")
endif()

# ── MOLRS is derived from products (products control molrs, not the reverse) ─
set(_atv_need_molrs FALSE)
set(_atv_molrs_why "")
if(ATV_BUILD_ANALYSIS)
    set(_atv_need_molrs TRUE)
    list(APPEND _atv_molrs_why "ANALYSIS")
endif()
if(ATV_BUILD_IO)
    set(_atv_need_molrs TRUE)
    list(APPEND _atv_molrs_why "IO")
endif()
if(ATV_BUILD_PYTHON)
    set(_atv_need_molrs TRUE)
    list(APPEND _atv_molrs_why "PYTHON")
endif()
if(ATV_BUILD_GPW)
    set(_atv_need_molrs TRUE)
    list(APPEND _atv_molrs_why "GPW")
endif()
# PW consumes the shared analytic GTH tables, which key on molrs::Element.
if(ATV_BUILD_PW)
    set(_atv_need_molrs TRUE)
    list(APPEND _atv_molrs_why "PW")
endif()
if(ATV_BUILD_HF)
    set(_atv_need_molrs TRUE)
    list(APPEND _atv_molrs_why "HF")
endif()
if(ATV_BUILD_SEMIEMPIRICAL)
    set(_atv_need_molrs TRUE)
    list(APPEND _atv_molrs_why "SEMIEMPIRICAL")
endif()

if(_atv_need_molrs)
    string(REPLACE ";" "+" _atv_molrs_why_str "${_atv_molrs_why}")
    set(ATV_BUILD_MOLRS ON CACHE BOOL
        "Derived: link molrs CXX when ANALYSIS/IO/… are on" FORCE)
    message(STATUS "ATV_BUILD_MOLRS=ON (derived from ${_atv_molrs_why_str})")
else()
    set(ATV_BUILD_MOLRS OFF CACHE BOOL
        "Derived: link molrs CXX when ANALYSIS/IO/… are on" FORCE)
    message(STATUS "ATV_BUILD_MOLRS=OFF (no product requested molrs)")
endif()
# EAM/SW table sources compile only when MOLRS is on (see root CMakeLists).

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
    if(ATV_BUILD_ANALYSIS)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_ANALYSIS)
    endif()
    if(ATV_BUILD_IO)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_IO)
    endif()
    if(ATV_BUILD_MOLRS)
        target_compile_definitions(${_target} PUBLIC ATV_HAS_MOLRS)
    endif()
endfunction()

message(STATUS "ATV product modules: "
    "MD=${ATV_BUILD_MD} MC=${ATV_BUILD_MC} MINIMIZER=${ATV_BUILD_MINIMIZER} "
    "VIB=${ATV_BUILD_VIB} PATH=${ATV_BUILD_PATH} SE=${ATV_BUILD_SEMIEMPIRICAL} "
    "POTENTIALS=${ATV_BUILD_POTENTIALS} NEIGHBOR=${ATV_BUILD_NEIGHBOR} "
    "EVALUATE=${ATV_BUILD_EVALUATE} ANALYSIS=${ATV_BUILD_ANALYSIS} "
    "IO=${ATV_BUILD_IO} MOLRS=${ATV_BUILD_MOLRS}")
