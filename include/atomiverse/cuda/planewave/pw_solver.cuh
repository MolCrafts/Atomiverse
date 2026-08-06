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
/// restricted (nspin = 1), fixed closed-shell occupations, `XCType::LSDA`.
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
///   pw.xc = cpu::scf::XCType::LSDA;   // required: PBE has no device path
///   pw.build(frame);
///   const auto& result = pw.run(frame);   // energy [Ry] / energy_ha [Ha]
/// @endcode
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include <string>

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
    /// @brief UPF v1 pseudopotential path (empty = toy V_loc).
    std::string upf_path;
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
    /// Only `XCType::LSDA` (Slater exchange + VWN5 correlation) has a device
    /// path. The default matches `cpu::pw::PW` deliberately: a CUDA-only
    /// default would make identical user code compute a different functional
    /// on the two backends. `run()` throws for anything else.
    cpu::scf::XCType xc = cpu::scf::XCType::PBE;
    /// @brief Collinear spin-polarized KS. No device path — `true` throws.
    bool spin_polarized = false;
    /// @brief Spin multiplicity 2S+1. No device path — `!= 1` throws.
    int multiplicity = 1;
    /// @brief Metallic smearing kind. No device path — `!= none` throws.
    Smearing smearing = Smearing::none;
    /// @brief Smearing width [Ry]. Inert: `smearing` is rejected first.
    double degauss = 0.02;
    /// @brief Methfessel–Paxton order. Inert: `smearing` is rejected first.
    int mp_order = 1;
    /// @brief Extra bands beyond ceil(n_e/2). Inert: `smearing` is rejected
    ///        first; the device path always solves exactly n_e/2 bands.
    int extra_bands = 4;

    PW() = default;
    ~PW() = default;
    PW(PW&&) noexcept = default;
    PW& operator=(PW&&) noexcept = default;
    PW(const PW&) = delete;
    PW& operator=(const PW&) = delete;

    /// @brief One-time setup: parses the UPF (if any) and validates Z.
    /// @param reference Frame used only to validate atom types.
    /// @throws std::runtime_error if the UPF cannot be parsed.
    /// @throws std::out_of_range if the toy valence table has no entry for
    ///         some atom's Z (no silent zero-electron fallback).
    void build(const cpu::Frame& reference);

    /// @brief Plane-wave SCF at the frame's current geometry, on the GPU.
    ///
    /// Rejects every member the device path cannot honour BEFORE any CUDA
    /// call or problem assembly, so an unsupported request costs nothing and
    /// names its own remedy.
    ///
    /// @param frame Atom positions [Å], types (Z), and simulation cell.
    /// @return Owned `PWResult` (valid until the next `run()`).
    /// @throws std::invalid_argument for `xc != XCType::LSDA`,
    ///         `spin_polarized`,
    ///         `multiplicity != 1`, `smearing != Smearing::none`,
    ///         non-positive `tol`/`mix_beta`, `pulay_depth < 1`,
    ///         `max_iter < 1`, or an odd valence-electron count (the device
    ///         density accumulator occupies every band with exactly 2
    ///         electrons). Every message names the member and the way out.
    PWResult& run(const cpu::Frame& frame);

    /// @brief Most recent SCF result.
    /// @return Owned `PWResult` (`energy` [Ry], `energy_ha` [Hartree]);
    ///         zero-initialised before the first `run()` and re-initialised
    ///         at the start of each `run()`. Valid until the next `run()`.
    const PWResult& last_result() const { return result_; }

  private:
    /// @brief Reject every member with no device path (throws, never warns).
    void validate_members() const;

    bool built_ = false;
    PWResult result_;
};

} // namespace pw
} // namespace cuda
} // namespace ATV
