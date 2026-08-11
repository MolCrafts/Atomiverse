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
/// `XCType::LSDA` / `XCType::PBE`; collinear fixed-occupation UKS
/// (`spin_polarized`, integer multiplicity) with **`XCType::LSDA` or
/// `XCType::PBE`**. Restricted smear forces and restricted smear cell stress
/// are on device; collinear LSDA spin forces **and** LSDA spin cell stress
/// are on device. spin+smear and spin-PBE forces/stress stay on
/// `cpu::pw::PW` (or deeper CUDA product rows).
///
/// k-points are NOT distributed across MPI ranks on this backend: every
/// rank computes every k-point of the Monkhorst-Pack set, so `n_kpts_local`
/// always equals `n_kpts`. Run one rank per calculation, or use
/// `cpu::pw::PW` under an MPI launcher.
///
/// @code
///   cuda::pw::PW pw;
///   pw.upf_path = "Si_lda_nc.upf";
///   pw.ecut_ry = 15.0;
///   pw.k_mesh = 2;
///   pw.xc = cpu::scf::XCType::LSDA;   // or XCType::PBE (the default)
///   pw.build(frame);
///   const auto& result = pw.run(frame);   // energy [Ry] / energy_ha [Ha]
/// @endcode
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include <array>
#include <string>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {

/// @brief Result type shared verbatim with the CPU method (energy in Ry).
using PWResult = ::ATV::cpu::pw::PWResult;

/// @brief Smearing selector shared verbatim with the CPU method.
using Smearing = ::ATV::cpu::pw::Smearing;

/// @brief Concrete plane-wave DFT method on CUDA — members + `run(frame)`.
///
/// Construction → set members → `build(reference)` → `run(frame)`.
/// `last_result()` is valid after each `run()`. Every geometry-dependent
/// structure (host problem, device buffers, cuFFT plans) is rebuilt inside
/// each `run()`, so the object is safe to reuse across AIMD / optimize steps.
class PW {
  public:
    /// @brief Norm-conserving UPF pseudopotential file. Mutually exclusive
    ///        with `gth_name`; exactly one must be set.
    std::string upf_path;
    /// @brief In-tree analytic GTH parameterization, e.g. "GTH-PBE-q4".
    ///        Needs no external file. Mutually exclusive with `upf_path`;
    ///        leaving both empty throws — there is no default
    ///        pseudopotential. Same surface as `cpu::pw::PW`.
    std::string gth_name;
    /// @brief Wavefunction cutoff [Ry].
    double ecut_ry = 30.0;
    /// @brief Monkhorst-Pack mesh size per dimension (dimensionless).
    int k_mesh = 4;
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
    /// @brief Exchange–correlation functional.
    ///
    /// `XCType::LSDA` (Slater exchange + VWN5 correlation) and `XCType::PBE`
    /// (PBE GGA over a PW92 base) have device paths. The default matches
    /// `cpu::pw::PW` deliberately: identical user code computes the same
    /// functional on both backends. `run()` throws for anything else,
    /// including the PBE-family variants (PBEsol/revPBE/RPBE) whose
    /// constants the device kernel does not carry.
    cpu::scf::XCType xc = cpu::scf::XCType::PBE;
    /// @brief Collinear spin-polarized KS (fixed occupations). Device path
    ///        supports `XCType::LSDA` and true spin-GGA `XCType::PBE`.
    bool spin_polarized = false;
    /// @brief Spin multiplicity 2S+1. With `spin_polarized`, sets
    ///        N↑ − N↓ = multiplicity − 1. Without spin, must stay 1.
    int multiplicity = 1;
    /// @brief Metallic smearing kind. Restricted path only — with
    ///        `spin_polarized` any value other than `none` throws.
    Smearing smearing = Smearing::none;
    /// @brief Smearing width [Ry] (QE `degauss`). Required > 0 when
    ///        `smearing != none`.
    double degauss = 0.02;
    /// @brief Methfessel–Paxton order \(N \ge 1\) (ignored for FD/Gaussian).
    int mp_order = 1;
    /// @brief Extra empty bands beyond \(\lceil n_e/2\rceil\) when smearing.
    ///        Fixed-occupation runs ignore this (solve \(n_e/2\) bands).
    int extra_bands = 4;

    PW() = default;
    ~PW() = default;
    PW(PW&&) noexcept = default;
    PW& operator=(PW&&) noexcept = default;
    PW(const PW&) = delete;
    PW& operator=(const PW&) = delete;

    /// @brief One-time setup: resolves the pseudopotential and validates Z.
    ///
    /// Validates the chosen source up front so a missing file or an unknown
    /// GTH name fails here rather than mid-SCF. Same envelope as
    /// `cpu::pw::PW::build`.
    ///
    /// @param reference Frame used only to validate atom types.
    /// @throws std::invalid_argument if neither or both of `upf_path` and
    ///         `gth_name` are set — a plane-wave run has no default
    ///         pseudopotential.
    /// @throws std::runtime_error if the UPF cannot be parsed, or if
    ///         `gth_name` names no in-tree parameterization for this element
    ///         (no silent fallback).
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
    ///         without spin, spin+smear, `degauss ≤ 0` with smear, non-positive
    ///         `tol`/`mix_beta`, `pulay_depth < 1`, `max_iter < 1`,
    ///         incompatible n_elec/multiplicity parity, or (restricted fixed
    ///         occupation only) an odd valence-electron count. Every message
    ///         names the member and the way out.
    PWResult& run(const cpu::Frame& frame);

    /// @brief Most recent SCF result.
    /// @return Owned `PWResult` (`energy` [Ry], `energy_ha` [Hartree]);
    ///         zero-initialised before the first `run()` and re-initialised
    ///         at the start of each `run()`. Valid until the next `run()`.
    const PWResult& last_result() const { return result_; }

  private:
    friend std::vector<std::array<double, 3>>
    pw_forces_analytic(const cpu::Frame& frame, const PW& method);
    friend std::array<double, 6> pw_stress(const cpu::Frame& frame,
                                           const PW& method,
                                           double* energy_ry_out);
    /// @brief Reject every member with no device path (throws, never warns).
    void validate_members() const;

    bool built_ = false;
    PWResult result_;
};

/// @brief Analytic ionic forces [eV/Å] after a full device SCF.
///
/// CUDA twin of `cpu::pw::pw_forces_analytic`: local Hellmann–Feynman +
/// Ewald + Kleinman–Bylander nonlocal. Local and nonlocal force kernels
/// run on the device; Ewald is the shared host classical sum. Supports
/// restricted closed shell (LSDA/PBE), restricted metallic smear
/// (FD/Gaussian/MP fractional occupations), and collinear fixed-occupation
/// UKS under **LSDA** only. Rejects spin∧smear, spin∧PBE forces, and other
/// unsupported members with a how-to.
///
/// @param frame  Geometry [Å], types (Z), cell; not mutated.
/// @param method Plane-wave method members (upf/gth, ecut, k_mesh, tol, …).
/// @return Forces [eV/Å], one xyz triple per atom.
/// @throws std::invalid_argument for empty frame or unsupported members.
std::vector<std::array<double, 3>> pw_forces_analytic(const cpu::Frame& frame,
                                                      const PW& method);

/// @brief Nielsen–Martin analytic cell stress σ [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
///
/// CUDA twin of `cpu::pw::pw_stress`: one device SCF + Hellmann–Feynman
/// strain-FD of the non-self-consistent total / free energy under remapped
/// density. Mechanical / CP2K sign:
/// \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\). Restricted metallic
/// smear uses the Mermin free energy \(F=E_{\mathrm{KS}}-TS\). Supports
/// restricted fixed closed shell (LSDA/PBE), restricted metallic smear
/// (FD/Gaussian/MP), and collinear fixed-occupation UKS under **LSDA**
/// (remaps both \(\rho_\uparrow\) and \(\rho_\downarrow\)). Rejects spin∧smear,
/// spin∧PBE stress, and other members `PW::run` rejects.
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
