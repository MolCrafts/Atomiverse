/// @file cpu/longrange/pppm.h
/// @brief Particle–mesh reciprocal electrostatics — brick PPPM (CPU).
///
/// ## Architecture (LAMMPS-aligned)
///
/// | Step | Action |
/// |---|---|
/// | 1 | Local **density brick** (owned mesh + order ghost layers) |
/// | 2 | Spread **owned atoms only** onto brick |
/// | 3 | **reverse_comm**: sum ghost density into neighbors' owned cells |
/// | 4 | **brick→FFT remap** (Alltoallv) into FFTW-MPI dim-0 slabs |
/// | 5 | Distributed r2c FFT → SPME H(m) → c2r |
/// | 6 | **FFT→brick remap** + **forward_comm** potential ghosts |
/// | 7 | AD force interpolation from local brick |
///
/// Serial (1 rank): dens_full_ + local r2c/c2r.
/// Multi-rank default: brick + reverse/forward + Alltoallv + FFTW-MPI slabs.
/// Debug fallback: `ATV_PPPM_ALLREDUCE=1` uses dens_full Allreduce + rank-0 FFT.
///
/// Requires `ATV_BUILD_KSPACE` (FFTW3). Multi-rank brick needs `ATV_HAS_FFTW_MPI`.
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/longrange/kspace.h"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {

/// @brief Match real-space α to a PPPM accuracy setting (LAMMPS-style guess).
inline double pppm_alpha_from_accuracy(double accuracy, double real_cutoff) {
    if (!(accuracy > 0.0) || !(accuracy < 1.0) || !(real_cutoff > 0.0))
        throw std::invalid_argument("pppm_alpha_from_accuracy: bad args");
    return (1.35 - 0.15 * std::log(accuracy)) / real_cutoff;
}

/// @brief Brick-decomposed particle–mesh kspace.
class KSpacePPPM : public cpu::KSpace {
  public:
    explicit KSpacePPPM(double alpha, double grid_spacing = 1.0, int order = 5);
    ~KSpacePPPM() override;

    KSpacePPPM(const KSpacePPPM&) = delete;
    KSpacePPPM& operator=(const KSpacePPPM&) = delete;
    KSpacePPPM(KSpacePPPM&& other) noexcept;
    KSpacePPPM& operator=(KSpacePPPM&&) = delete;

    static KSpacePPPM from_accuracy(double accuracy, double real_cutoff,
                                    double grid_spacing = -1.0) {
        const double alpha = pppm_alpha_from_accuracy(accuracy, real_cutoff);
        const double spacing =
            grid_spacing > 0.0 ? grid_spacing : (real_cutoff / 2.5);
        return KSpacePPPM(alpha, spacing, /*order=*/5);
    }

    std::string name() const override { return "pppm"; }
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
    void setup_brick_partition_(int coords[3], int dims[3]);

    double alpha_;
    double spacing_;
    int order_;

    int K_[3] = {0, 0, 0};
    const cpu::Box* box_identity_ = nullptr;
    std::uint64_t cell_revision_ = 0;
    bool have_grid_ = false;

    // ── Brick (owned + ghost), global indices inclusive ───────────────
    int nxlo_in_ = 0, nxhi_in_ = 0;
    int nylo_in_ = 0, nyhi_in_ = 0;
    int nzlo_in_ = 0, nzhi_in_ = 0;
    int nxlo_out_ = 0, nxhi_out_ = 0;
    int nylo_out_ = 0, nyhi_out_ = 0;
    int nzlo_out_ = 0, nzhi_out_ = 0;
    int nxl_ = 0, nyl_ = 0, nzl_ = 0; ///< brick dimensions incl. ghosts
    int nghost_ = 0;
    std::vector<double> brick_; ///< density or potential, size nxl*nyl*nzl

    // Serial full-mesh buffers (1-rank path / green tables).
    std::vector<double> dens_full_; ///< full K0·K1·K2 real buffer (serial FFT)
    std::vector<std::complex<double>> work_;
    std::vector<double> green_; ///< H(m) half-spectrum, full K0×K1×n2c
    std::vector<double> bsp_mod_[3];

    void* plan_fwd_ = nullptr;
    void* plan_bwd_ = nullptr;

    // FFTW-MPI slab
    bool mpi_fft_ready_ = false;
    long long mpi_local_n0_ = 0;
    long long mpi_local_0_start_ = 0;
    long long mpi_alloc_ = 0;
    double* mpi_data_ = nullptr;
    void* plan_mpi_fwd_ = nullptr;
    void* plan_mpi_bwd_ = nullptr;
    int mpi_fft_K_[3] = {0, 0, 0};
    int mpi_nprocs_ = 1;
    /// Per-rank FFT m0 ownership (length nprocs+1 prefix starts).
    std::vector<int> fft_m0_starts_;
    std::vector<int> fft_m0_counts_;

    // Remap scratch
    std::vector<double> remap_send_;
    std::vector<double> remap_recv_;
    std::vector<int> sendcounts_, senddispls_, recvcounts_, recvdispls_;

    int atom_capacity_ = 0;
    std::vector<int> base_;
    std::vector<double> theta_;
    std::vector<double> dtheta_;

    /// @brief Flat index into brick_ for global (ix,iy,iz) in out-range.
    int brick_index_(int ix, int iy, int iz) const noexcept {
        return (iz - nzlo_out_) * nyl_ * nxl_ + (iy - nylo_out_) * nxl_
               + (ix - nxlo_out_);
    }
};

} // namespace cpu
} // namespace ATV
