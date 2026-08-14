# AtvUnitGate.cmake — path-scoped CPU unit gate (cmake -P).
#
# Usage:
#   cmake -P cmake/AtvUnitGate.cmake
#   cmake -DATV_GATE_ALL=1 -P cmake/AtvUnitGate.cmake   # full classical unit suite
#   cmake -DATV_GATE_MODULES=hf,md -P cmake/AtvUnitGate.cmake  # force modules
#
# Rules:
#   • Optional physics (HF / PW / GPW / KSPACE) is OFF unless the diff needs it.
#   • Product modules (MD/MC/…) stay at CMake defaults (AtvProductModules).
#   • Only build + ctest targets that match the selected modules (not "all tests").
#   • Every test ≤ ATV_TEST_TIMEOUT_SECONDS (default 10). Heavy suite stays OFF.
#
# Override force-all: ATV_GATE_ALL=1 or env ATV_GATE_ALL=1

cmake_minimum_required(VERSION 3.25)

# -P mode: this file lives in cmake/; repo root is the parent.
get_filename_component(_ATV_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(_CMAKE_BIN_DIR "${CMAKE_COMMAND}" DIRECTORY)
set(_CTEST "${_CMAKE_BIN_DIR}/ctest${CMAKE_EXECUTABLE_SUFFIX}")
if(NOT EXISTS "${_CTEST}")
  set(_CTEST "ctest")
endif()

set(_BUILD_DIR "${_ATV_ROOT}/build")
if(DEFINED ATV_CI_BUILD_DIR AND NOT ATV_CI_BUILD_DIR STREQUAL "")
  if(IS_ABSOLUTE "${ATV_CI_BUILD_DIR}")
    set(_BUILD_DIR "${ATV_CI_BUILD_DIR}")
  else()
    set(_BUILD_DIR "${_ATV_ROOT}/${ATV_CI_BUILD_DIR}")
  endif()
endif()

set(_TIMEOUT 10)
if(DEFINED ATV_TEST_TIMEOUT_SECONDS AND NOT ATV_TEST_TIMEOUT_SECONDS STREQUAL "")
  set(_TIMEOUT "${ATV_TEST_TIMEOUT_SECONDS}")
endif()

set(_OMP 2)
if(DEFINED ATV_TEST_OMP_THREADS AND NOT ATV_TEST_OMP_THREADS STREQUAL "")
  set(_OMP "${ATV_TEST_OMP_THREADS}")
endif()

set(_JOBS "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
if(_JOBS STREQUAL "")
  cmake_host_system_information(RESULT _JOBS QUERY NUMBER_OF_LOGICAL_CORES)
endif()
if(NOT _JOBS OR _JOBS STREQUAL "0")
  set(_JOBS 4)
endif()

# ── Optional module set (HF/PW/GPW/KSPACE) ──────────────────────────────────
# Keys used internally; map 1:1 to ATV_BUILD_*.
set(_ALL_OPT hf pw gpw kspace)

# Path → optional modules (substring match on posix path).
# Order does not matter; union is taken.
set(_MAP_hf
  "/scf/" "/electronic/" "/cpu/xc/"
  "include/atomiverse/cpu/scf"
  "include/atomiverse/cpu/electronic"
  "src/cpu/scf" "src/cpu/electronic" "src/cpu/xc"
  "tests/cpu/scf" "tests/cpu/electronic" "tests/cpu/xc"
  "test_cpu_scf" "test_basis_" "test_ecp" "test_cpu_zero_energy_electronic"
)
set(_MAP_pw
  "/planewave/" "include/atomiverse/cpu/planewave"
  "src/cpu/planewave" "tests/cpu/planewave"
  "test_cpu_pw_" "test_pw_" "test_upf" "test_fft" "test_kpoint" "test_g_vector"
  "test_berry" "test_crystal" "test_dfpt" "test_ewald.cpp"
)
set(_MAP_gpw
  "/gpw/" "include/atomiverse/cpu/gpw"
  "src/cpu/gpw" "tests/cpu/gpw" "test_cpu_gpw" "test_gpw_"
)
set(_MAP_kspace
  "longrange/test_pme" "longrange/test_pppm"
  "test_cpu_pme" "test_cpu_pppm" "test_pme_" "test_pppm_"
  "src/cpu/longrange/pme" "src/cpu/longrange/pppm"
  "include/atomiverse/cpu/longrange/pme"
  "include/atomiverse/cpu/longrange/pppm"
)

# ctest -R fragments per optional module (plus always-safe core smoke).
set(_R_core "test_mat3|test_vec3|test_units|test_real_harmonics|test_cpu_box|test_cpu_chunked_range|test_cpu_frame")
set(_R_hf "scf|electronic|basis_|ecp|xc_functional|zero_energy_electronic|cart2sph|wannier|x2c|multipole_ints|spherical_basis|gradient_result|init_guess|density_fitting|cfmm|ediis|accelerators|orbital_hessian|stability|ghost_basis|delta_scf|ci_engine|cc_diis")
set(_R_pw "planewave|test_cpu_pw_|test_pw_|test_upf|test_fft$|test_kpoint|test_g_vector|test_berry|test_crystal|test_dfpt|test_cpu_ewald$")
set(_R_gpw "gpw")
set(_R_kspace "pme|pppm")
# Classical monorepo slice when MD-ish paths change (product modules stay default ON).
set(_R_md "md_|dynamics|neighbor|potential|pair_|bond_|dihedral|improper|eam|yukawa|brownian|settle|virtual_sites|domain_decomposition|halo_|frame_tag|atom_sort|timestep|write_sets|pe_|const_pot|electrode|qeq|bias_|metadynamics|observe|msd|vacf|widom|voronoi|esp_|lowdin|capacitance|multipole$|species_density|potential_profile|montecarlo|mc_|gcmc|minimize|line_search|triclinic|anisotropic|path_|remd|hremd|replica|string_method|irc|rpmd|vib_|phonon|semiempirical|nddo|am1|rm1|dftb|scc_mixer|xtb|qmmm|link_atom|driver|criterion|materialize|task_|evaluate|install_infra|linalg|eigensolve|linear_solve|canonical_orthogonal|io_|lammps|extxyz|fmm_|gaussian_extent|ewald_sum|ewald_coul|slab_ewald|multipole_expansion|gaussian_charge|pim_|lambda_ti|coul_long|force_efield|force_dfield|force_addforce|force_wall|force_freeze|force_oscillating|charging_work|charge_aspc|atom_variable|analysis_executor|optimize_driver|minimize_driver|minimize_evaluate|cpu_driver|binding_kind|forces_requested|component_ownership|driver_nesting|context_slices|source_cobinding|density_cube|classical_polarization")

# Paths that mean "classical product surface" (not optional HF/PW/…).
set(_MAP_md
  "/dynamics/" "/neighbor/" "/potentials/" "/montecarlo/" "/minimize/"
  "/path/" "/bias/" "/replica/" "/vibration/" "/semiempirical/" "/multiscale/"
  "/observe/" "/io/" "/driver/" "/component/" "/evaluate/" "/task/"
  "/communication/" "/device/" "/electrodes/" "/record/" "/linalg/"
  "src/cpu/dynamics" "src/cpu/neighbor" "src/cpu/potentials"
  "tests/cpu/dynamics" "tests/cpu/potentials" "tests/cpu/montecarlo"
  "tests/cpu/minimize" "tests/cpu/path" "tests/cpu/bias" "tests/cpu/observe"
  "tests/cpu/io" "tests/cpu/driver" "tests/cpu/semiempirical" "tests/cpu/vib"
  "python/" "include/atomiverse/cpu/dynamics" "include/atomiverse/cpu/neighbor"
  "include/atomiverse/cpu/potentials"
)

# Paths that force a broader classical unit pass (no HF unless also mapped).
# CI/gate config alone is core smoke — only product/source contracts widen.
set(_MAP_broad
  "CMakeLists.txt" "cmake/AtvProductModules" "cmake/MolrsContract"
  "molrs/" "cmake/deps"
)

function(atv_gate_path_hit out_var path needles_var)
  set(_hit FALSE)
  set(_p "${path}")
  # Normalize to forward slashes for matching.
  string(REPLACE "\\" "/" _p "${_p}")
  foreach(_n IN LISTS ${needles_var})
    string(FIND "${_p}" "${_n}" _idx)
    if(NOT _idx EQUAL -1)
      set(_hit TRUE)
      break()
    endif()
  endforeach()
  set(${out_var} "${_hit}" PARENT_SCOPE)
endfunction()

function(atv_gate_collect_changed out_var)
  set(_files "")
  # Prefer unpushed commits + dirty tree vs upstream; fall back to HEAD~1 / dirty.
  execute_process(
    COMMAND git rev-parse --abbrev-ref --symbolic-full-name @{u}
    WORKING_DIRECTORY "${_ATV_ROOT}"
    RESULT_VARIABLE _urc
    OUTPUT_VARIABLE _upstream
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(_range "")
  if(_urc EQUAL 0 AND NOT _upstream STREQUAL "")
    execute_process(
      COMMAND git merge-base HEAD "${_upstream}"
      WORKING_DIRECTORY "${_ATV_ROOT}"
      RESULT_VARIABLE _mbrc
      OUTPUT_VARIABLE _mb
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_mbrc EQUAL 0 AND NOT _mb STREQUAL "")
      set(_range "${_mb}")
    endif()
  endif()
  if(_range STREQUAL "")
    execute_process(
      COMMAND git rev-parse HEAD~1
      WORKING_DIRECTORY "${_ATV_ROOT}"
      RESULT_VARIABLE _h1rc
      OUTPUT_VARIABLE _range
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _h1rc EQUAL 0)
      set(_range "HEAD")
    endif()
  endif()

  execute_process(
    COMMAND git diff --name-only "${_range}"
    WORKING_DIRECTORY "${_ATV_ROOT}"
    OUTPUT_VARIABLE _diff_commits
    ERROR_QUIET)
  execute_process(
    COMMAND git diff --name-only
    WORKING_DIRECTORY "${_ATV_ROOT}"
    OUTPUT_VARIABLE _diff_work
    ERROR_QUIET)
  execute_process(
    COMMAND git diff --name-only --cached
    WORKING_DIRECTORY "${_ATV_ROOT}"
    OUTPUT_VARIABLE _diff_index
    ERROR_QUIET)
  set(_raw "${_diff_commits}\n${_diff_work}\n${_diff_index}")
  string(REPLACE "\n" ";" _list "${_raw}")
  foreach(_f IN LISTS _list)
    if(NOT _f STREQUAL "")
      list(APPEND _files "${_f}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _files)
  set(${out_var} "${_files}" PARENT_SCOPE)
endfunction()

# ── Resolve modules ─────────────────────────────────────────────────────────
set(_FORCE_ALL FALSE)
if(DEFINED ATV_GATE_ALL AND ATV_GATE_ALL)
  set(_FORCE_ALL TRUE)
endif()
if(DEFINED ENV{ATV_GATE_ALL} AND NOT "$ENV{ATV_GATE_ALL}" STREQUAL "" AND NOT "$ENV{ATV_GATE_ALL}" STREQUAL "0")
  set(_FORCE_ALL TRUE)
endif()

set(_want_hf FALSE)
set(_want_pw FALSE)
set(_want_gpw FALSE)
set(_want_kspace FALSE)
set(_want_md FALSE)
set(_want_core FALSE)
set(_want_broad FALSE)
set(_modules_forced FALSE)

if(DEFINED ATV_GATE_MODULES AND NOT ATV_GATE_MODULES STREQUAL "")
  set(_modules_forced TRUE)
  string(TOLOWER "${ATV_GATE_MODULES}" _mods)
  string(REPLACE "," ";" _mod_list "${_mods}")
  foreach(_m IN LISTS _mod_list)
    string(STRIP "${_m}" _m)
    if(_m STREQUAL "all")
      set(_FORCE_ALL TRUE)
    elseif(_m STREQUAL "hf")
      set(_want_hf TRUE)
    elseif(_m STREQUAL "pw")
      set(_want_pw TRUE)
    elseif(_m STREQUAL "gpw")
      set(_want_gpw TRUE)
      set(_want_hf TRUE)
      set(_want_pw TRUE)
    elseif(_m STREQUAL "kspace")
      set(_want_kspace TRUE)
    elseif(_m STREQUAL "md" OR _m STREQUAL "classical")
      set(_want_md TRUE)
    elseif(_m STREQUAL "core")
      set(_want_core TRUE)
    endif()
  endforeach()
elseif(NOT _FORCE_ALL)
  atv_gate_collect_changed(_changed)
  list(LENGTH _changed _nchanged)
  message(STATUS "AtvUnitGate: ${_nchanged} changed path(s)")
  foreach(_f IN LISTS _changed)
    message(STATUS "  · ${_f}")
    atv_gate_path_hit(_h "${_f}" _MAP_hf)
    if(_h)
      set(_want_hf TRUE)
    endif()
    atv_gate_path_hit(_h "${_f}" _MAP_pw)
    if(_h)
      set(_want_pw TRUE)
    endif()
    atv_gate_path_hit(_h "${_f}" _MAP_gpw)
    if(_h)
      set(_want_gpw TRUE)
      set(_want_hf TRUE)
      set(_want_pw TRUE)
    endif()
    atv_gate_path_hit(_h "${_f}" _MAP_kspace)
    if(_h)
      set(_want_kspace TRUE)
    endif()
    atv_gate_path_hit(_h "${_f}" _MAP_md)
    if(_h)
      set(_want_md TRUE)
    endif()
    atv_gate_path_hit(_h "${_f}" _MAP_broad)
    if(_h)
      set(_want_broad TRUE)
    endif()
  endforeach()
  # Pure docs / unmapped paths → tiny core smoke (no HF, no full MD suite).
  if(NOT _want_hf AND NOT _want_pw AND NOT _want_gpw AND NOT _want_kspace
      AND NOT _want_md AND NOT _want_broad)
    set(_want_core TRUE)
    message(STATUS "AtvUnitGate: no module map hit → core smoke only")
  endif()
endif()

if(_FORCE_ALL)
  # Full classical unit suite + optional modules that tests may need.
  # Still no CUDA. HF/PW/GPW/KSPACE ON so registered slices exist.
  set(_want_hf TRUE)
  set(_want_pw TRUE)
  set(_want_gpw TRUE)
  set(_want_kspace TRUE)
  set(_want_md TRUE)
  set(_want_broad TRUE)
  set(_want_core TRUE)
  message(STATUS "AtvUnitGate: ATV_GATE_ALL — full unit slice")
endif()

# GPW implies HF+PW (CMake FATAL otherwise).
if(_want_gpw)
  set(_want_hf TRUE)
  set(_want_pw TRUE)
endif()

set(_hf OFF)
set(_pw OFF)
set(_gpw OFF)
set(_kspace OFF)
if(_want_hf)
  set(_hf ON)
endif()
if(_want_pw)
  set(_pw ON)
endif()
if(_want_gpw)
  set(_gpw ON)
endif()
if(_want_kspace)
  set(_kspace ON)
endif()

# ctest include regex
set(_R "${_R_core}")
if(_want_md OR _want_broad OR _FORCE_ALL)
  set(_R "${_R}|${_R_md}")
endif()
if(_want_hf)
  set(_R "${_R}|${_R_hf}")
endif()
if(_want_pw)
  set(_R "${_R}|${_R_pw}")
endif()
if(_want_gpw)
  set(_R "${_R}|${_R_gpw}")
endif()
if(_want_kspace)
  set(_R "${_R}|${_R_kspace}")
endif()
# _want_core alone keeps _R_core only (already set).

# Always exclude heavy / MPI multi-rank.
set(_E "no_alloc|^test_mpi_")

message(STATUS "AtvUnitGate: root=${_ATV_ROOT}")
message(STATUS "AtvUnitGate: build=${_BUILD_DIR}")
message(STATUS "AtvUnitGate: HF=${_hf} PW=${_pw} GPW=${_gpw} KSPACE=${_kspace}")
message(STATUS "AtvUnitGate: ctest -R '${_R}'")
message(STATUS "AtvUnitGate: jobs=${_JOBS} timeout=${_TIMEOUT}s OMP=${_OMP}")

# ── Configure ───────────────────────────────────────────────────────────────
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${_ATV_ROOT}"
    -B "${_BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DATV_BUILD_CUDA=OFF
    -DATV_BUILD_TESTS=ON
    -DATV_BUILD_EXAMPLES=OFF
    -DATV_BUILD_REGRESSIONS=OFF
    -DATV_BUILD_BENCHMARKS=OFF
    -DATV_BUILD_HF=${_hf}
    -DATV_BUILD_PW=${_pw}
    -DATV_BUILD_GPW=${_gpw}
    -DATV_BUILD_KSPACE=${_kspace}
    -DATV_REAL_STORAGE=double
    -DATV_TEST_TIMEOUT_SECONDS=${_TIMEOUT}
    -DATV_TEST_OMP_THREADS=${_OMP}
    -DATV_TEST_HEAVY=OFF
  RESULT_VARIABLE _crc
)
if(NOT _crc EQUAL 0)
  message(FATAL_ERROR "AtvUnitGate: configure failed (${_crc})")
endif()

# ── Discover matching test names, build only those targets + atomiverse ─────
execute_process(
  COMMAND "${_CTEST}" --test-dir "${_BUILD_DIR}" -N -R "${_R}" -E "${_E}"
  OUTPUT_VARIABLE _ctest_n
  ERROR_VARIABLE _ctest_n_err
  RESULT_VARIABLE _nrc
)
if(NOT _nrc EQUAL 0)
  message(FATAL_ERROR "AtvUnitGate: ctest -N failed:\n${_ctest_n}\n${_ctest_n_err}")
endif()

set(_targets atomiverse)
set(_test_names "")
string(REGEX MATCHALL "Test +#[0-9]+: +[A-Za-z0-9_]+" _lines "${_ctest_n}")
foreach(_line IN LISTS _lines)
  string(REGEX REPLACE "Test +#[0-9]+: +" "" _tname "${_line}")
  list(APPEND _test_names "${_tname}")
  # Unit executables share the ctest name; script tests (cmake -P / sh) may not.
  list(APPEND _targets "${_tname}")
endforeach()
list(REMOVE_DUPLICATES _targets)
list(LENGTH _test_names _ntests)
if(_ntests EQUAL 0)
  message(FATAL_ERROR "AtvUnitGate: no tests matched -R '${_R}' (broken map?)")
endif()
message(STATUS "AtvUnitGate: ${_ntests} test(s) selected")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${_BUILD_DIR}" -j "${_JOBS}" --target ${_targets}
  RESULT_VARIABLE _brc
)
# Some selected names are not build targets (script tests). Retry atomiverse-only
# then let ctest run; missing binaries will fail clearly.
if(NOT _brc EQUAL 0)
  message(STATUS "AtvUnitGate: selective target build failed; building default test suite targets via 'all'")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_BUILD_DIR}" -j "${_JOBS}"
    RESULT_VARIABLE _brc2
  )
  if(NOT _brc2 EQUAL 0)
    message(FATAL_ERROR "AtvUnitGate: build failed (${_brc2})")
  endif()
endif()

# ── Test ────────────────────────────────────────────────────────────────────
set(ENV{OMP_NUM_THREADS} "${_OMP}")
execute_process(
  COMMAND "${_CTEST}"
    --test-dir "${_BUILD_DIR}"
    -j "${_JOBS}"
    --timeout "${_TIMEOUT}"
    --output-on-failure
    -R "${_R}"
    -E "${_E}"
  RESULT_VARIABLE _trc
)
if(NOT _trc EQUAL 0)
  message(FATAL_ERROR "AtvUnitGate: ctest failed (${_trc})")
endif()

message(STATUS "AtvUnitGate: OK (${_ntests} tests, HF=${_hf} PW=${_pw} GPW=${_gpw} KSPACE=${_kspace})")
