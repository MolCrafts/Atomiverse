/// @file cuda/planewave/pw_solver.cuh
/// @brief cuda::pw::PW — concrete plane-wave DFT method (CUDA).
///
/// The CUDA twin of `cpu::pw::PW`: parameters are public members, `Frame`
/// is the only I/O currency, and the self-consistency loop is owned by
/// this method object as `cuda::Driver<pw::Phase>`. Standalone and nested
/// calls share the same staged SCF and its accepted device density.
///
/// The device path is a strict subset of the CPU path. Members the GPU
/// cannot honour make `run()` throw with a how-to — the class never
/// silently degrades to a different Hamiltonian. Supported today:
/// restricted (nspin = 1) with fixed closed-shell occupations **or**
/// metallic smearing (Fermi–Dirac / Gaussian / Methfessel–Paxton) and
/// `XCType::LSDA` or the PBE family (`PBE`, `PBEsol`, `RevPBE`, `RPBE`);
/// collinear UKS (`spin_polarized`, integer multiplicity) with the same XC set,
/// fixed occupation **or** metallic smear with shared μ. Methfessel–Paxton
/// corrections admit signed quasi-occupations outside the probability range.
/// Restricted smear forces and restricted smear cell stress are on
/// device; collinear LSDA/PBE-family spin forces (fixed-occ **and** smear)
/// and spin cell stress (fixed-occ or fractional occupations) are
/// supported. Gamma RKS fixed/FD LSDA/PBE-family uses one certified final-band ensemble
/// for closed-form force/stress assembly and one nested device publication.
/// Cold radial/Ewald setup remains host work. Other cell-stress paths use
/// frozen-density NSCF strain differences, with both strained spin spectra
/// refilled at one shared chemical potential.
///
/// k-points are NOT distributed across MPI ranks on this backend: every
/// rank computes every k-point of the Monkhorst-Pack set, so `n_kpts_local`
/// always equals `n_kpts`. Run one rank per calculation, or use
/// `cpu::pw::PW` under an MPI launcher.
///
/// @code
///   cuda::pw::PW pw{{std::make_shared<cpu::pw::UpfPseudopotential>("Si_lda_nc.upf")},
///                   /*ecut_ry=*/15.0, /*k_mesh=*/2, cpu::xc::XCType::LSDA};
///   pw.build(frame);
///   const auto& result = pw.run(frame);   // energy [Ry] / energy_ha [Ha]
/// @endcode
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/driver/driver_node.h"
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ATV::cpu::pw::internal {
struct PWSetup;
struct PWWorkspace;
} // namespace ATV::cpu::pw::internal

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
struct PwLoopState;

class PW : public cuda::Driver<ATV::pw::Phase>, public cuda::DriverNode {
  public:
    // ── Identity (constructor arguments, fixed after construction) ────
    /// @brief Shared immutable builtin NC providers, keyed by atomic number.
    const std::vector<
        std::shared_ptr<const ::ATV::electronic::NormConservingPotential>>
        potentials;
    /// @brief Wavefunction cutoff [Ry].
    const double ecut_ry;
    /// @brief Monkhorst-Pack mesh size per dimension (dimensionless).
    const int k_mesh;
    /// @brief Exchange–correlation functional. Device paths accept LSDA and
    ///        the pure PBE family (PBE, PBEsol, RevPBE, RPBE); `run()` throws
    ///        for every other functional.
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
    /// @brief Collinear spin-polarized KS with fixed or smeared occupations.
    ///        Device XC supports LSDA and the pure PBE-family spin GGAs.
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

    /// @brief Plane-wave DFT using builtin GTH and/or scalar NC-UPF providers.
    /// @param potentials One builtin provider per atomic number in the frame.
    /// @param ecut_ry Wavefunction cutoff [Ry].
    /// @param k_mesh Monkhorst-Pack mesh per dimension.
    /// @param xc Exchange-correlation functional with a device kernel.
    PW(std::vector<
           std::shared_ptr<const ::ATV::electronic::NormConservingPotential>>
           potentials,
       double ecut_ry, int k_mesh, cpu::xc::XCType xc = cpu::xc::XCType::PBE);
    ~PW() override;
    /// @brief Move an unattached method; a live Runtime binding rejects moves.
    PW(PW&&);
    PW& operator=(PW&&) = delete;
    PW(const PW&) = delete;
    PW& operator=(const PW&) = delete;

    /// @brief Validate Frame species against the element providers.
    /// @param reference Frame used to validate every atom's identity.
    /// @throws std::invalid_argument for missing or duplicate element providers.
    void build(const cpu::Frame& reference);

    /// @brief Plane-wave SCF at the frame's current geometry, on the GPU.
    ///
    /// Rejects every member the device path cannot honour BEFORE any CUDA
    /// call or problem assembly, so an unsupported request costs nothing and
    /// names its own remedy.
    ///
    /// @param frame Atom positions [Å], types (Z), and simulation cell.
    /// @return Owned `PWResult` (valid until the next `run()`).
    /// @throws std::invalid_argument for an `xc` outside LSDA and the pure
    ///         PBE family (PBE, PBEsol, RevPBE, RPBE), multiplicity without
    ///         spin, `degauss ≤ 0` with smear, non-positive
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
    /// @throws std::invalid_argument for unsupported members, as in `run`.
    PWResult& forces(cpu::Frame& frame);

    /// @brief One reference SCF plus six-component cell stress.
    ///        Gamma RKS fixed/FD and fixed nonempty UKS LSDA/PBE-family use
    ///        certified unstrained final bands and closed-form assembly.
    ///        UKS checks the orbital XC domain; finite floor/cap domains use
    ///        the existing numerical epilogue after the two spin band solves.
    ///        Other domains use frozen-density NSCF strain differences
    ///        (h=2e-3). Neither path starts another reference SCF.
    ///        Returns σ [eV/Å³] and writes
    ///        `kStress` = σΩ [eV] and `kEnergy` [eV].
    /// @param frame Geometry in, stress and energy meta out.
    /// @return Intensive stress [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
    ///         `last_result()` describes the reference SCF of this stress call.
    /// @throws std::invalid_argument for unsupported members, as in `run`.
    std::array<double, 6> stress(cpu::Frame& frame);

    /// @brief Most recent SCF result.
    /// @return Owned `PWResult` (`energy` [Ry], `energy_ha` [Hartree]);
    ///         zero-initialised before the first `run()` and re-initialised
    ///         at the start of each `run()`. Derivative members also replace
    ///         it with their own SCF result; a truncated solve cannot retain
    ///         a previous call's convergence or occupation metadata.
    const PWResult& last_result() const { return result_; }

    /// @brief Electronic energy/force/virial source for an outer CUDA Driver.
    bool provides_energy() const { return true; }
    bool provides_force() const { return true; }
    bool provides_system_stress() const { return true; }
    int default_budget() const override { return max_iter; }

    /// @brief Mount stages on the parent's existing Backend and device Frame.
    void build_on(cuda::Context& ctx, cuda::Frame& frame) override;
    /// @brief Evaluate on the parent's execution, accumulating requested outputs.
    int run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                          int budget) override;
    /// @brief Detach stages without destroying the parent execution owner.
    void destroy_on(cuda::Context& ctx, cuda::Frame& frame) override;

  protected:
    void validate_components() const override {}

  private:
    void validate_members() const;
    void release_backend_();
    void drive_stages_(cuda::Context& ctx, cuda::Frame& frame, int budget);
    std::array<int, 2>
    prepare_(const cpu::Frame& geometry, cpu::pw::internal::PWSetup& problem,
             cpu::pw::internal::PWWorkspace& workspace) const;
    enum class DerivativePublication { ordinary, device_committed };
    DerivativePublication evaluate_(
        const cpu::Frame& geometry, const cpu::pw::internal::PWSetup& problem,
        cpu::pw::internal::PWWorkspace& workspace,
        const std::array<int, 2>& spin_occupations, cuda::Context& ctx,
        cuda::Frame& frame, int budget,
        std::vector<std::array<double, 3>>* forces_out,
        std::array<double, 6>* stress_out, cuda::Frame* force_target = nullptr);
    PWResult& standalone_(const cpu::Frame& geometry,
                          std::vector<std::array<double, 3>>* forces_out,
                          std::array<double, 6>* stress_out);

    bool built_ = false;
    PWResult result_;
    std::unique_ptr<PwLoopState> loop_;
};

} // namespace pw
} // namespace cuda
} // namespace ATV
