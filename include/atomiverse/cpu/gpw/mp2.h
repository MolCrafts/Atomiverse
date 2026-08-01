/// @file cpu/gpw/mp2.h
/// @brief GPW MP2 correlation energy result (public value type).
///
/// User path: `GpwDft::mp2(atoms)` after a converged `run()` — the GPW
/// analogue of molecular `RHF::mp2` / `UHF::mp2` (`cpu/scf/mp2.h`).
///
/// - **RKS:** Γ (real) or multi-k (complex Bloch + momentum conservation).
/// - **UKS:** Γ unrestricted MP2 (UMP2) on α/β orbitals **and multi-k UMP2**
///   (`reg_periodic-ump2-si` / `reg_periodic-ump2-kpoints`), with optional
///   Grimme SCS scales (`reg_periodic-scs-ump2-si` /
///   `reg_periodic-scs-ump2-kpoints`).
///
/// Product default (`mp2_ri = true`) is the RI-GPW plane-wave
/// resolution-of-identity assembly of \((ia|jb)\) (Del Ben, Hutter,
/// VandeVondele, J. Chem. Theory Comput. 8, 4177 (2012); CP2K
/// `&WF_CORRELATION` `METHOD MP2_GPW`). Γ foothold Poisson path remains with
/// `mp2_ri = false`. Pair densities are neutral (\(S_{ia}=0\)), so the
/// G=0-dropped periodic Coulomb is divergence-free.
///
/// This header exposes only the `Mp2Result` value type; kernels live in
/// `internal/mp2_kernels.h`.
#pragma once

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief GPW MP2 correlation energy (RKS/UKS Γ or multi-k; optional SCS).
///
/// Spin-component channels `e_corr_os` / `e_corr_ss` are the **unscaled**
/// opposite-spin and same-spin pieces of the doubles sum (same convention as
/// molecular `cpu::scf::MP2Result` / PySCF). The reported `e_corr` applies the
/// optional Grimme scales from the method members `mp2_os_scale` /
/// `mp2_ss_scale` (defaults 1.0 recover plain MP2).
struct Mp2Result {
    double e_corr = 0.0; ///< Scaled MP2 correlation energy [Hartree] (< 0).
    /// @brief Unscaled opposite-spin correlation [Hartree].
    double e_corr_os = 0.0;
    /// @brief Unscaled same-spin correlation [Hartree].
    double e_corr_ss = 0.0;
    double e_scf = 0.0;   ///< Converged GPW SCF total energy [Hartree].
    double e_total = 0.0; ///< e_scf + e_corr [Hartree].
    /// @brief RKS: correlated occupied spatial orbitals. UMP2: n_α + n_β.
    int n_occ = 0;
    /// @brief RKS: virtual spatial orbitals (after cap). UMP2: n_va + n_vb.
    int n_virt = 0;
    /// @brief true when (ia|jb) used the RI-GPW product path.
    bool ri = false;
    /// @brief Sampled k-points used in the MP2 sum (0 = real Γ path).
    int n_kpts = 0;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
