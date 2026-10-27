/// @file cuda/planewave/pw_solver.cuh
/// @brief cuda::pw::PW — concrete plane-wave DFT method (CUDA).
///
/// The CUDA twin of `cpu::pw::PW`: parameters are public members, `Frame`
/// is the only I/O currency, and the self-consistency loop is owned by
/// this method object (no public `*Config` bag, no `make_*` factory, no
/// Driver, no phase enum, no Plan registration).
///
/// The device path is a strict subset of the CPU path. Members the GPU
/// cannot honour make `run()` throw with a how-to — the class never
/// silently degrades to a different Hamiltonian. Supported today:
/// restricted (nspin = 1) with fixed closed-shell occupations **or**
/// metallic smearing (Fermi–Dirac / Gaussian / Methfessel–Paxton) and
/// `XCType::LSDA` / `XCType::PBE`; collinear UKS (`spin_polarized`,
/// integer multiplicity) with **`XCType::LSDA` or `XCType::PBE`**,
/// fixed occupation **or** metallic smear with shared μ. Methfessel–Paxton
/// corrections admit signed quasi-occupations outside the probability range.
/// Restricted smear forces and restricted smear cell stress are on
/// device; collinear LSDA/PBE spin forces (fixed-occ **and** smear)
/// and LSDA/PBE spin cell stress (fixed-occ or fractional occupations) are
/// supported. Cell stress is a frozen-density NSCF strain finite difference,
/// with both strained spin spectra refilled at one shared chemical potential.
///
/// k-points are NOT distributed across MPI ranks on this backend: every
/// rank computes every k-point of the Monkhorst-Pack set, so `n_kpts_local`
/// always equals `n_kpts`. Run one rank per calculation, or use
/// `cpu::pw::PW` under an MPI launcher.
///
/// @code
///   cuda::pw::PW pw{std::make_shared<cpu::pw::UpfPseudopotential>("Si_lda_nc.upf"),
///                   /*ecut_ry=*/15.0, /*k_mesh=*/2, cpu::xc::XCType::LSDA};
///   pw.build(frame);
///   const auto& result = pw.run(frame);   // energy [Ry] / energy_ha [Ha]
/// @endcode
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/xc/xc_functional.h"
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {

/// @brief Result type shared verbatim with the CPU method (energy in Ry).
using PWResult = ::ATV::cpu::pw::PWResult;

/// @brief Concrete plane-wave DFT method on CUDA — members + `run(frame)`.
///
/// Construction → set members → `build(reference)` → `run(frame)`.
/// `last_result()` is valid after each `run()`. Every geometry-dependent
/// structure (host problem, device buffers, cuFFT plans) is rebuilt inside
/// each `run()`, so the object is safe to reuse across AIMD / optimize steps.
class PW {
  public:
    // ── Identity (constructor arguments, fixed after construction) ────
    /// @brief Shared immutable builtin NC potential; custom CPU providers are
    ///        rejected before any device work.
    const std::shared_ptr<const ::ATV::electronic::NormConservingPotential>
        potential;
    /// @brief Wavefunction cutoff [Ry].
    const double ecut_ry;
    /// @brief Monkhorst-Pack mesh size per dimension (dimensionless).
    const int k_mesh;
    /// @brief Exchange–correlation functional. `XCType::LSDA` and
    ///        `XCType::PBE` have device paths; `run()` throws for anything
    ///        else (including PBEsol/revPBE/RPBE).
    const cpu::xc::XCType xc;

    // ── Knobs ──────────────────────────────────────────────────────────
    /// @brief SCF tolerance on the density-change L2 norm (dimensionless).
    ///
    /// This is the SCF (outer) tolerance only. The device block-Davidson
    /// band solve uses the same fixed residual threshold as the CPU solver
    /// and does not read this member.
    double tol = 1e-6;
    /// @brief Maximum SCF iterations (dimensionless).
    int max_iter = 50;
    /// @brief Density mixing parameter (QE mixing_beta, dimensionless).
    double mix_beta = 0.7;
    /// @brief Pulay mixing history window (dimensionless).
    int pulay_depth = 5;
    /// @brief Collinear spin-polarized KS (fixed occupations). Device path
    ///        supports `XCType::LSDA` and true spin-GGA `XCType::PBE`.
    bool spin_polarized = false;
    /// @brief Spin multiplicity 2S+1. With `spin_polarized`, sets
    ///        N↑ − N↓ = multiplicity − 1. Without spin, must stay 1.
    int multiplicity = 1;
    /// @brief Metallic smearing kind. Restricted and collinear UKS use a
    ///        shared chemical potential over the occupied spin orbitals.
    ATV::electronic::Smearing smearing = ATV::electronic::Smearing::none;
    /// @brief Smearing width [Ry] (QE `degauss`). Required > 0 when
    ///        `smearing != none`.
    double degauss = 0.02;
    /// @brief Methfessel–Paxton order \(N \ge 1\) (ignored for FD/Gaussian).
    int mp_order = 1;
    /// @brief Extra empty bands beyond \(\lceil n_e/2\rceil\) when smearing.
    ///        Fixed-occupation runs ignore this (solve \(n_e/2\) bands).
    int extra_bands = 4;

    /// @brief Plane-wave DFT using a builtin GTH or UPF NC provider.
    /// @param pp Immutable potential; custom providers have no CUDA product path.
    /// @param ecut_ry Wavefunction cutoff [Ry].
    /// @param k_mesh Monkhorst-Pack mesh per dimension.
    /// @param xc Exchange-correlation functional with a device kernel.
    PW(std::shared_ptr<const ::ATV::electronic::NormConservingPotential> pp,
       double ecut_ry, int k_mesh, cpu::xc::XCType xc = cpu::xc::XCType::PBE)
        : potential(std::move(pp)), ecut_ry(ecut_ry), k_mesh(k_mesh), xc(xc) {
        if (!potential
            || (!dynamic_cast<const cpu::pw::GthPseudopotential*>(
                    potential.get())
                && !dynamic_cast<const cpu::pw::UpfPseudopotential*>(
                    potential.get())))
            throw std::invalid_argument("cuda::pw::PW: only builtin GTH/UPF NC "
                                        "providers are supported");
    }
    ~PW() = default;
    PW(PW&&) noexcept = default;
    PW& operator=(PW&&) = delete;
    PW(const PW&) = delete;
    PW& operator=(const PW&) = delete;

    /// @brief Validate Frame species against the builtin potential.
    /// @param reference Frame used to validate every atom's identity.
    /// @throws std::invalid_argument if any species differs from the provider.
    void build(const cpu::Frame& reference);

    /// @brief Plane-wave SCF at the frame's current geometry, on the GPU.
    ///
    /// Rejects every member the device path cannot honour BEFORE any CUDA
    /// call or problem assembly, so an unsupported request costs nothing and
    /// names its own remedy.
    ///
    /// @param frame Atom positions [Å], types (Z), and simulation cell.
    /// @return Owned `PWResult` (valid until the next `run()`).
    /// @throws std::invalid_argument for an `xc` other than `XCType::LSDA` /
    ///         `XCType::PBE`, collinear spin with non-LSDA/PBE XC, multiplicity
    ///         without spin, `degauss ≤ 0` with smear, non-positive
    ///         `tol`/`mix_beta`, `pulay_depth < 1`, `max_iter < 1`,
    ///         incompatible n_elec/multiplicity parity, or (restricted fixed
    ///         occupation only) an odd valence-electron count. Every message
    ///         names the member and the way out.
    PWResult& run(const cpu::Frame& frame);

    /// @brief Same SCF as `run(const Frame&)`, then write `kEnergy` [eV].
    /// @param frame Geometry in, energy meta out.
    /// @return Owned `PWResult` (valid until the next `run()` or `forces()`).
    PWResult& run(cpu::Frame& frame);

    /// @brief Device SCF plus analytic forces into `kAtom_f*` [eV/Å] and
    ///        `kEnergy` [eV].
    /// @param frame Geometry in, forces and energy out.
    /// @return The complete SCF result that produced the forces, including
    ///         convergence, iterations, occupations and k-point counts.
    /// @throws std::invalid_argument for the same members `pw_forces_analytic`
    ///         rejects.
    PWResult& forces(cpu::Frame& frame);

    /// @brief Device SCF plus frozen-density NSCF strain-FD stress (h=2e-3).
    ///        Returns σ [eV/Å³] and writes
    ///        `kStress` = σΩ [eV] and `kEnergy` [eV].
    /// @param frame Geometry in, stress and energy meta out.
    /// @return Intensive stress [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
    ///         `last_result()` describes the reference SCF of this stress call.
    /// @throws std::invalid_argument for the same members `pw_stress` rejects.
    std::array<double, 6> stress(cpu::Frame& frame);

    /// @brief Most recent SCF result.
    /// @return Owned `PWResult` (`energy` [Ry], `energy_ha` [Hartree]);
    ///         zero-initialised before the first `run()` and re-initialised
    ///         at the start of each `run()`. Derivative members also replace
    ///         it with their own SCF result; a truncated solve cannot retain
    ///         a previous call's convergence or occupation metadata.
    const PWResult& last_result() const { return result_; }

  private:
    friend std::vector<std::array<double, 3>>
    pw_forces_analytic(const cpu::Frame& frame, const PW& method,
                       double* energy_ha_out);
    friend std::array<double, 6>
    pw_stress(const cpu::Frame& frame, const PW& method, double* energy_ry_out);
    /// @brief Reject every member with no device path (throws, never warns).
    void validate_members() const;

    // Shared execution for the Frame member and the compatibility free function.
    // The optional result is the very SCF used for these derivatives.
    std::vector<std::array<double, 3>> forces_impl(const cpu::Frame& frame,
                                                   double* energy_ha_out,
                                                   PWResult* result_out) const;
    std::array<double, 6> stress_impl(const cpu::Frame& frame,
                                      double* energy_ry_out,
                                      PWResult* result_out) const;

    bool built_ = false;
    PWResult result_;
};

/// @brief Analytic ionic forces [eV/Å] after a full device SCF.
///
/// CUDA twin of `cpu::pw::pw_forces_analytic`: local Hellmann–Feynman +
/// Ewald + Kleinman–Bylander nonlocal. Local and nonlocal force kernels
/// run on the device; Ewald is the shared host classical sum. Supports
/// restricted closed shell (LSDA/PBE), restricted metallic smear
/// (FD/Gaussian/MP fractional occupations), collinear fixed-occupation
/// UKS (LSDA/PBE), and collinear spin×smear with a shared μ (signed MP
/// quasi-occupations are preserved).
/// Rejects unsupported XC / members with a how-to.
///
/// @param frame  Geometry [Å], types (Z), cell; not mutated.
/// @param method Plane-wave method members (upf/gth, ecut, k_mesh, tol, …).
/// @return Forces [eV/Å], one xyz triple per atom.
/// @throws std::invalid_argument for empty frame or unsupported members.
std::vector<std::array<double, 3>>
pw_forces_analytic(const cpu::Frame& frame, const PW& method,
                   double* energy_ha_out = nullptr);

/// @brief Frozen-density NSCF strain-FD stress σ [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
///
/// CUDA twin of `cpu::pw::pw_stress`: one device SCF + Hellmann–Feynman
/// strain-FD of the non-self-consistent total / free energy under remapped
/// density at strain step 2e-3; this is not a closed-form tensor.
/// Mechanical / CP2K sign:
/// \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\). Restricted metallic
/// smear uses the Mermin free energy \(F=E_{\mathrm{KS}}-TS\). Supports
/// restricted fixed closed shell (LSDA/PBE), restricted metallic smear
/// (FD/Gaussian/MP), and collinear UKS under **LSDA/PBE**, with fixed or
/// fractional occupations (remaps both spin densities). Rejects the same
/// unsupported members as `PW::run`.
///
/// @param frame         Geometry [Å], types (Z), cell; not mutated.
/// @param method        Plane-wave method members.
/// @param energy_ry_out Optional: reference SCF free/total energy [Ry].
/// @return Intensive stress [eV/Å³] Voigt.
/// @throws std::invalid_argument for empty frame, bad cell, or unsupported
///         members.
std::array<double, 6> pw_stress(const cpu::Frame& frame, const PW& method,
                                double* energy_ry_out = nullptr);

} // namespace pw
} // namespace cuda
} // namespace ATV
