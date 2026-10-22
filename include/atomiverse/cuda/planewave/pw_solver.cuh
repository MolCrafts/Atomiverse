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
/// fixed occupation **or** metallic smear (shared μ, \(f\in[0,1]\)).
/// Restricted smear forces and restricted smear cell stress are on
/// device; collinear LSDA/PBE spin forces (fixed-occ **and** smear)
/// and LSDA/PBE spin cell stress (fixed-occ) are on device.
/// spin∧smear analytic stress stays on `cpu::pw::pw_stress_fd` (or a
/// deeper CUDA product row).
///
/// k-points are NOT distributed across MPI ranks on this backend: every
/// rank computes every k-point of the Monkhorst-Pack set, so `n_kpts_local`
/// always equals `n_kpts`. Run one rank per calculation, or use
/// `cpu::pw::PW` under an MPI launcher.
///
/// @code
///   cuda::pw::PW pw{cpu::pw::UpfPseudopotential{"Si_lda_nc.upf"},
///                   /*ecut_ry=*/15.0, /*k_mesh=*/2, cpu::scf::XCType::LSDA};
///   pw.build(frame);
///   const auto& result = pw.run(frame);   // energy [Ry] / energy_ha [Ha]
/// @endcode
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
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
    // ── Identity (constructor arguments, fixed after construction) ────
    /// @brief UPF file path; empty when a GTH table entry is used.
    const std::string upf_path;
    /// @brief GTH table entry; empty when a UPF file is used.
    const std::string gth_name;
    /// @brief Wavefunction cutoff [Ry].
    const double ecut_ry;
    /// @brief Monkhorst-Pack mesh size per dimension (dimensionless).
    const int k_mesh;
    /// @brief Exchange–correlation functional. `XCType::LSDA` and
    ///        `XCType::PBE` have device paths; `run()` throws for anything
    ///        else (including PBEsol/revPBE/RPBE).
    const cpu::scf::XCType xc;

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
    /// @brief Metallic smearing kind. Restricted and collinear UKS — with
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

    /// @brief Plane-wave DFT from a UPF pseudopotential file (same
    ///        signature as `cpu::pw::PW`).
    /// @param pp Pseudopotential file.
    /// @param ecut_ry Wavefunction cutoff [Ry].
    /// @param k_mesh Monkhorst-Pack mesh per dimension (dimensionless).
    /// @param xc Exchange–correlation functional.
    PW(cpu::pw::UpfPseudopotential pp, double ecut_ry, int k_mesh,
       cpu::scf::XCType xc = cpu::scf::XCType::PBE)
        : upf_path(std::move(pp.path)), ecut_ry(ecut_ry), k_mesh(k_mesh),
          xc(xc) {
        if (upf_path.empty())
            throw std::invalid_argument("cuda::pw::PW: empty UPF path");
    }
    /// @brief Plane-wave DFT from an in-tree GTH pseudopotential.
    /// @param pp GTH table entry.
    /// @param ecut_ry Wavefunction cutoff [Ry].
    /// @param k_mesh Monkhorst-Pack mesh per dimension (dimensionless).
    /// @param xc Exchange–correlation functional.
    PW(cpu::pw::GthPseudopotential pp, double ecut_ry, int k_mesh,
       cpu::scf::XCType xc = cpu::scf::XCType::PBE)
        : gth_name(std::move(pp.name)), ecut_ry(ecut_ry), k_mesh(k_mesh),
          xc(xc) {
        if (gth_name.empty())
            throw std::invalid_argument("cuda::pw::PW: empty GTH name");
    }
    ~PW() = default;
    PW(PW&&) noexcept = default;
    PW& operator=(PW&&) = delete;
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

    /// @brief Same SCF as `run(const Frame&)`, then write `kEnergy` [eV].
    /// @param frame Geometry in, energy meta out.
    /// @return Owned `PWResult` (valid until the next `run()` or `forces()`).
    PWResult& run(cpu::Frame& frame);

    /// @brief Device SCF plus analytic forces into `kAtom_f*` [eV/Å] and
    ///        `kEnergy` [eV].
    /// @param frame Geometry in, forces and energy out.
    /// @return The SCF result that produced the forces.
    /// @throws std::invalid_argument for the same members `pw_forces_analytic`
    ///         rejects.
    PWResult& forces(cpu::Frame& frame);

    /// @brief Device SCF plus analytic stress. Returns σ [eV/Å³] and writes
    ///        `kStress` = σΩ [eV] and `kEnergy` [eV].
    /// @param frame Geometry in, stress and energy meta out.
    /// @return Intensive stress [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
    /// @throws std::invalid_argument for the same members `pw_stress` rejects.
    std::array<double, 6> stress(cpu::Frame& frame);

    /// @brief Most recent SCF result.
    /// @return Owned `PWResult` (`energy` [Ry], `energy_ha` [Hartree]);
    ///         zero-initialised before the first `run()` and re-initialised
    ///         at the start of each `run()`. Valid until the next `run()`.
    const PWResult& last_result() const { return result_; }

  private:
    friend std::vector<std::array<double, 3>>
    pw_forces_analytic(const cpu::Frame& frame, const PW& method,
                       double* energy_ha_out);
    friend std::array<double, 6>
    pw_stress(const cpu::Frame& frame, const PW& method, double* energy_ry_out);
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
/// (FD/Gaussian/MP fractional occupations), collinear fixed-occupation
/// UKS (LSDA/PBE), and collinear spin×smear (shared μ, \(f\in[0,1]\)).
/// Rejects unsupported XC / members with a how-to.
///
/// @param frame  Geometry [Å], types (Z), cell; not mutated.
/// @param method Plane-wave method members (upf/gth, ecut, k_mesh, tol, …).
/// @return Forces [eV/Å], one xyz triple per atom.
/// @throws std::invalid_argument for empty frame or unsupported members.
std::vector<std::array<double, 3>>
pw_forces_analytic(const cpu::Frame& frame, const PW& method,
                   double* energy_ha_out = nullptr);

/// @brief Nielsen–Martin analytic cell stress σ [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
///
/// CUDA twin of `cpu::pw::pw_stress`: one device SCF + Hellmann–Feynman
/// strain-FD of the non-self-consistent total / free energy under remapped
/// density. Mechanical / CP2K sign:
/// \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\). Restricted metallic
/// smear uses the Mermin free energy \(F=E_{\mathrm{KS}}-TS\). Supports
/// restricted fixed closed shell (LSDA/PBE), restricted metallic smear
/// (FD/Gaussian/MP), and collinear fixed-occupation UKS under **LSDA/PBE**
/// (remaps both \(\rho_\uparrow\) and \(\rho_\downarrow\)). Rejects
/// spin∧smear analytic stress (use `cpu::pw::pw_stress_fd`) and other
/// members `PW::run` rejects.
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
