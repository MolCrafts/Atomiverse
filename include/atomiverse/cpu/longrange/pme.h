/// @file cpu/longrange/pme.h
/// @brief Smooth Particle-Mesh Ewald reciprocal sum (LAMMPS `kspace_style
/// pppm`/`pme`) — CPU. Requires `ATV_BUILD_KSPACE` (FFTW3).
///
/// Thin Frame/MPI/FFTW wrapper over `atomiverse/cpu/longrange/{pme_spline,
/// pme_influence}`. A drop-in replacement for `KSpaceEwald`: same real-space
/// `CoulLong`, same self / net-charge terms, but the O(N²) reciprocal sum is
/// replaced by B-spline charge spreading → FFT → influence-function multiply →
/// inverse FFT → force interpolation, giving O(N log N) scaling.
///
/// Smooth PME of Essmann et al., J. Chem. Phys. 103, 8577 (1995):
///   E_recip = units::coulomb/(2πV) · Σ_{m≠0} exp(−π²m²/α²)/m² · B(m) · |F(Q)(m)|²
/// with Q the spline-interpolated charge grid and B(m) the Euler spline
/// modulus. DOI: 10.1063/1.470117.
///
/// **Performance.** Influence geometry tables are rebuilt only when the mesh
/// or cell changes (not every step). FFTW plans use `FFTW_MEASURE`. Under MPI
/// (replicated grid) only rank 0 runs FFT + influence + iFFT; the force grid
/// is broadcast — every rank still spreads owned charges and interpolates.
/// (Distributed slab FFT is a future scalability refinement.)
#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/longrange/kspace.h"
#include "atomiverse/cpu/longrange/pme_influence.h"

namespace ATV {
namespace cpu {

/// @brief Smooth Particle-Mesh Ewald reciprocal-space electrostatics.
class KSpacePME : public cpu::KSpace {
  public:
    /// @param alpha        Ewald splitting parameter α [1/Å]. Must match the
    ///                     real-space `CoulLong` α.
    /// @param grid_spacing Target real-space grid spacing [Å] (grid sizes are
    ///                     derived from the box and rounded to FFT-friendly
    ///                     values). Default 1.0 Å.
    /// @param order        B-spline interpolation order (≥ 3). Default 6.
    explicit KSpacePME(double alpha, double grid_spacing = 1.0, int order = 6);
    ~KSpacePME() override;

    KSpacePME(const KSpacePME&) = delete;
    KSpacePME& operator=(const KSpacePME&) = delete;
    KSpacePME(KSpacePME&& other) noexcept;
    KSpacePME& operator=(KSpacePME&&) = delete;

    std::string name() const override { return "pme"; }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::ReciprocalSpace, alpha_, AccReal(0), AccReal(0)};
    }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::force, 100)};
    }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;
    std::size_t memory_bytes() const override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    void ensure_grid(const cpu::Frame& frame, bool initial_build);
    void ensure_atom_capacity(int required, bool initial_build);
    void release_storage() noexcept;

    double alpha_;
    double spacing_;
    int order_;

    int K_[3] = {0, 0, 0}; ///< Grid dimensions.
    const cpu::Box* box_identity_ = nullptr;
    std::uint64_t cell_revision_ = 0;
    bool have_grid_ = false;

    // Euler spline moduli per grid line, |b_d(m)|².
    std::vector<double> bsp_mod_[3];

    // Charge / convolution grid (complex, size K0·K1·K2) and FFTW plans
    // (opaque to keep fftw3.h out of this header).
    std::vector<std::complex<double>> grid_;
    void* plan_fwd_ = nullptr;
    void* plan_bwd_ = nullptr;

    // Cached influence (geometry). Rebuilt with mesh/cell; applied every step.
    longrange::PmeInfluenceTables influence_;

    // Persistent per-atom spline workspace. Growth is an explicit rebuild.
    int atom_capacity_ = 0;
    std::vector<int> base_;
    std::vector<double> theta_;
    std::vector<double> dtheta_;
};

} // namespace cpu
} // namespace ATV
