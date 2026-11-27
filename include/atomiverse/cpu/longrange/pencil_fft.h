/// @file cpu/longrange/pencil_fft.h
/// @brief Distributed real 3-D FFT of a brick-decomposed mesh on pencils.
///
/// Ported from LAMMPS `src/KSPACE/fft3d.cpp` (`fft_3d`,
/// `fft_3d_create_plan`, `bifactor`), checkout e8c4c278a8: three sets of 1-D
/// FFTs, each on pencils that hold the whole transform axis, joined by
/// `remap_3d` plans (`GridRemap`) that permute the storage so the next axis
/// is the contiguous one; ranks are split over the two non-transformed axes
/// by `bifactor`.
/// LAMMPS: Copyright (2003) Sandia Corporation. GPL-2.0-only.
///
/// Two departures from `fft_3d`, both inside its structure:
/// - the first stage is real-to-complex (and the last backward stage
///   complex-to-real), so the spectrum is the half spectrum `m2 ≤ K2/2` the
///   PPPM tables are written for (`core/pppm_ik.h`) and the two later stages
///   move half the data;
/// - the spectrum stays on the third-stage pencils (LAMMPS `permute = 2`
///   output), so the forward transform has no post-remap and the backward
///   transform starts without a pre-remap. The brick tiling is the input of
///   the forward and the output of the backward transform, as in LAMMPS
///   `PPPM::brick2fft` and `fft2`.
///
/// **Axes.** Mesh `K = (K0, K1, K2)` along x, y, z. A brick is the inclusive
/// box `{xlo, xhi, ylo, yhi, zlo, zhi}` stored x-outer / z-inner without
/// ghosts. The stages run z, then y, then x.
///
/// **Spectrum layout.** This rank owns modes `m0 ∈ [0, K0)`,
/// `m1 ∈ [m1lo, m1hi]`, `m2 ∈ [m2lo, m2hi]` (`spectrum_box()`), stored at
/// `((m1 - m1lo) * n2 + (m2 - m2lo)) * K0 + m0`, `n2 = m2hi - m2lo + 1`.
/// Transforms are unnormalized, with FFTW's sign convention; forward then
/// backward multiplies by `K0 * K1 * K2`.
#pragma once

#ifdef ATV_HAS_MPI

#include <array>
#include <complex>
#include <cstddef>
#include <memory>

#include <mpi.h>

#include "atomiverse/cpu/longrange/grid_remap.h"

namespace ATV::longrange {

/// @brief Split @p n into two factors as equal as possible (LAMMPS
///        `bifactor`).
/// @param n Rank count, ≥ 1.
/// @return `{n1, n2}` with `n1 * n2 == n` and `n1 ≤ n2`.
std::array<int, 2> pencil_bifactor(int n);

/// @brief One forward / backward plan pair for a fixed mesh and brick
///        tiling (LAMMPS `fft_plan_3d`).
class PencilFft {
  public:
    /// @brief Build the plans. Collective over @p comm.
    /// @param comm   Communicator whose ranks own the brick tiling.
    /// @param mesh   Mesh points `(K0, K1, K2)` (count each).
    /// @param brick  This rank's owned brick, inclusive global indices
    ///               `{xlo, xhi, ylo, yhi, zlo, zhi}`.
    /// @param active Ranks `[0, active)` hold pencils; the others only feed
    ///               and receive their brick. `1 ≤ active ≤ size(comm)`.
    /// @throws std::invalid_argument on a bad mesh or @p active.
    /// @throws std::runtime_error when FFTW cannot plan.
    PencilFft(MPI_Comm comm, const std::array<int, 3>& mesh,
              const std::array<int, 6>& brick, int active);
    ~PencilFft();
    PencilFft(const PencilFft&) = delete;
    PencilFft& operator=(const PencilFft&) = delete;

    /// @brief Modes this rank owns.
    /// @return `{m1lo, m1hi, m2lo, m2hi}` (inclusive; `hi < lo` when none).
    std::array<int, 4> spectrum_box() const { return spectrum_box_; }
    /// @brief Complex values this rank owns in the spectrum (count).
    /// @return `K0 * (m1hi - m1lo + 1) * (m2hi - m2lo + 1)`.
    std::size_t spectrum_size() const { return spectrum_size_; }
    /// @brief The spectrum buffer: output of forward(), input of backward().
    /// @return Pointer to `spectrum_size()` values (may be null when 0).
    std::complex<double>* spectrum() { return spectrum_; }

    /// @brief Real brick to spectrum().
    /// @param brick Owned brick values, x-outer / z-inner.
    void forward(const double* brick);
    /// @brief spectrum() to real brick; destroys spectrum().
    /// @param brick Owned brick values, x-outer / z-inner (output).
    void backward(double* brick);

    /// @brief Heap bytes held by the plans and work arrays.
    /// @return Bytes.
    std::size_t memory_bytes() const;

  private:
    int K_[3];
    std::array<int, 4> spectrum_box_{0, -1, 0, -1};
    std::size_t spectrum_size_ = 0;
    std::size_t real_size_ = 0; ///< doubles on the z pencils
    std::size_t half_size_ = 0; ///< complex values on the z pencils
    std::size_t mid_size_ = 0;  ///< complex values on the y pencils
    double* real_ = nullptr;
    std::complex<double>* half_ = nullptr;
    std::complex<double>* mid_ = nullptr;
    std::complex<double>* spectrum_ = nullptr;
    // FFTW plans (fftw_plan), null where this rank holds no pencil.
    void* plan_z_r2c_ = nullptr;
    void* plan_z_c2r_ = nullptr;
    void* plan_y_fwd_ = nullptr;
    void* plan_y_bwd_ = nullptr;
    void* plan_x_fwd_ = nullptr;
    void* plan_x_bwd_ = nullptr;
    std::unique_ptr<GridRemap> brick_to_z_;
    std::unique_ptr<GridRemap> z_to_y_;
    std::unique_ptr<GridRemap> y_to_x_;
    std::unique_ptr<GridRemap> x_to_y_;
    std::unique_ptr<GridRemap> y_to_z_;
    std::unique_ptr<GridRemap> z_to_brick_;
};

} // namespace ATV::longrange

#endif // ATV_HAS_MPI
