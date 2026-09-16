# ── Example layout ──────────────────────────────────────────────────
# Flat, numbered tutorial sequence (see README.md):
#   NN_topic.cpp   CPU canonical example — one concept per file
#   NN_topic.cu    optional CUDA twin of the same number (target NN_topic_cuda)
# Examples demonstrate API usage only; speed / science comparisons
# lived in benchmarks/ (removed; rebuild later).

# CUDA example (source = <name>.cu, may carry an explicit target name).
function(atv_add_cuda_example name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE atomiverse)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

# C++ example (no CUDA required).
function(atv_add_cpp_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE atomiverse)
    set_target_properties(${name} PROPERTIES CXX_STANDARD 20)
endfunction()

# 01 — bare SCF loop on xtensor only (intentionally does not link atomiverse).
add_executable(01_scf 01_scf.cpp)
target_link_libraries(01_scf PRIVATE xtensor xtensor-blas ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
# xtensor-blas still uses old flat-header paths; route through the compat shim
# that atomiverse also uses.
target_include_directories(01_scf PRIVATE $<BUILD_INTERFACE:${_xt_compat}>)
set_target_properties(01_scf PROPERTIES CXX_STANDARD 20)

if(ATV_BUILD_HF)
    atv_add_cpp_example(02_rhf)
    atv_add_cpp_example(03_uhf)
    atv_add_cpp_example(04_uks)
    atv_add_cpp_example(05_geom_opt)
    atv_add_cpp_example(06_aimd)
    atv_add_cpp_example(11_koopmans)
    atv_add_cpp_example(12_delta_scf)
    atv_add_cpp_example(13_mp2)
    atv_add_cpp_example(14_fci)
    # 16 — Gaussian cube density / HOMO dump for MolVis docs embeds
    atv_add_cpp_example(16_density_cube)
    if(ATV_BUILD_CUDA)
        atv_add_cuda_example(02_rhf_cuda 02_rhf.cu)
    endif()
endif()

# 15 — AM1 semiempirical (always in the CPU library; no RHF/PW gate).
atv_add_cpp_example(15_am1)

# 07 — classical MD. CPU .cpp is the canonical lesson (always built);
# the .cu is the CUDA twin (target 07_md_lj_cuda).
atv_add_cpp_example(07_md_lj)
if(ATV_BUILD_CUDA)
    atv_add_cuda_example(07_md_lj_cuda 07_md_lj.cu)
endif()

# 08 — plane-wave DFT (periodic silicon); needs the PW module (FFTW3).
if(ATV_BUILD_PW)
    atv_add_cpp_example(08_pw)
endif()

# 09 — GPW DFT (isolated water, PBE + MT); needs the GPW module (RHF + PW/FFTW3).
if(ATV_BUILD_GPW AND ATV_BUILD_PW)
    atv_add_cpp_example(09_gpw)
endif()
