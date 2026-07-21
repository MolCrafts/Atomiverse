/// @file cpu/observer/trajectory_metrics.h
/// @brief Trajectory-analysis compute objects — thin C++ wrappers over the molrs
/// FFI compute types (molrs::{MsdCompute, DiffusionCompute, VacfCompute,
/// RdfCompute}) and their streaming accumulator counterparts
/// (molrs::{RdfAccumulator, MsdAccumulator, VacfAccumulator}).
///
/// The engine *measures*; molrs *analyzes*. Two calling conventions:
///
/// - **Batch** (`MSD`, `EinsteinDiffusion`, `VACF`, `RDF`): one-shot, stateless
///   `compute(...)` over a whole raw trajectory buffer — for callers that
///   already hold the full trajectory.
/// - **Streaming** (`RDFAccumulator`, `MSDAccumulator`, `VACFAccumulator`):
///   `accumulate(...)` one frame at a time, `finalize()` once. State is
///   O(bins / window·n_dof / resolution·n_dof) — never O(trajectory) — so the
///   observers stream arbitrarily long MD runs at bounded memory. Numerics
///   match the batch path (bit-for-bit for RDF; FFT round-off for VACF).
///
/// **No analysis math lives in C++** — see molrs/src/compute/.
///
/// CUDA-safe: this header exposes only std types (no `rust/cxx.h`); the molrs
/// objects live behind PIMPLs constructed inside the `.cpp`, so observer
/// headers that include it stay includable from `.cu`/`.cuh`.
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace ATV {
namespace cpu {
namespace compute {

/// @brief Mean squared displacement MSD(t) — wraps molrs `compute::MSD`.
///
/// Direct mode: first frame = reference (LAMMPS `compute msd`).
class MSD {
  public:
    MSD() = default;

    /// @brief Run MSD over a raw position trajectory.
    /// @param positions row-major `[n_frames, n_dof]`, per frame blocked `x|y|z`
    /// (`n_dof = 3 * n_atoms`). @param n_frames number of frames. @param n_dof
    /// degrees of freedom per frame. @return the mean-MSD curve (one value per
    /// frame, index 0 = 0); empty on fewer than two frames / bad shape.
    std::vector<double> compute(const std::vector<double>& positions,
                                int n_frames, int n_dof) const;
};

/// @brief Einstein self-diffusion coefficient D [length²/time] — wraps molrs
/// `EinsteinDiffusion` + `LinearFit`.
class EinsteinDiffusion {
  public:
    /// @param dt time between frames. @param dims spatial dimensionality.
    /// @param fit_lo @param fit_hi MSD-slope fit window (fractions of the last
    /// lag): `D = slope / (2·dims)`.
    EinsteinDiffusion(double dt, int dims, double fit_lo, double fit_hi);

    /// @brief Run the diffusion fit over a raw position trajectory.
    /// @param positions same `[n_frames, n_dof]` buffer as `MSD::compute`.
    /// @return D, or NaN on bad shape / fit error.
    double compute(const std::vector<double>& positions, int n_frames,
                   int n_dof) const;

  private:
    double dt_;
    int dims_;
    double fit_lo_;
    double fit_hi_;
};

/// @brief DOF-averaged velocity autocorrelation function — wraps molrs
/// `compute::VACF` (FFT-ACF).
class VACF {
  public:
    /// @param dt time between frames. @param resolution max lag (caps the curve
    /// length).
    VACF(double dt, int resolution);

    /// @brief Run VACF over a raw velocity trajectory.
    /// @param velocities row-major `[n_frames, n_dof]` (`n_dof = 3 * n_atoms`).
    /// @return the VACF curve (index 0 = zero lag); empty on fewer than two
    /// frames / shape mismatch.
    std::vector<double> compute(const std::vector<double>& velocities,
                                int n_frames, int n_dof) const;

  private:
    double dt_;
    int resolution_;
};

/// @brief Radial distribution function g(r) — wraps molrs `compute::RDF`
/// (per-frame LinkCell, ideal-gas shell-volume normalization).
class RDF {
  public:
    /// @param n_bins number of radial bins. @param r_max @param r_min bin range
    /// (Å). Argument order mirrors molrs `RDF::new(n_bins, r_max, r_min)`.
    RDF(int n_bins, double r_max, double r_min);

    /// @brief Run g(r) over a raw position trajectory + per-frame cells.
    /// @param positions row-major `[n_frames, 3*n_atoms]` (blocked `x|y|z`).
    /// @param boxes row-major `[n_frames, 9]` per-frame cell (supports NPT).
    /// @return the g(r) curve, one value per bin; bin-center radii are
    /// `r_min + (i + 0.5)·(r_max − r_min)/n_bins`. Empty on bad args / compute
    /// error.
    std::vector<double> compute(const std::vector<double>& positions,
                                const std::vector<double>& boxes, int n_frames,
                                int n_atoms) const;

  private:
    int n_bins_;
    double r_max_;
    double r_min_;
};

/// @brief Streaming g(r) accumulator — wraps molrs `RdfAccumulator`
/// (O(n_bins) state; identical numerics to `RDF::compute` over the same
/// frames).
class RDFAccumulator {
  public:
    /// @param n_bins number of radial bins (≥ 1). @param r_max upper edge of
    /// the last bin [Å] (> r_min). @param r_min lower edge of bin 0 [Å] (≥ 0).
    /// @throws std::invalid_argument on out-of-range parameters (no-fallback).
    RDFAccumulator(int n_bins, double r_max, double r_min);
    ~RDFAccumulator();
    RDFAccumulator(RDFAccumulator&&) noexcept;
    RDFAccumulator& operator=(RDFAccumulator&&) noexcept;
    RDFAccumulator(const RDFAccumulator&) = delete;
    RDFAccumulator& operator=(const RDFAccumulator&) = delete;

    /// @brief Fold one frame into the running histogram.
    /// @param positions blocked `x|y|z` [Å], length `3·n_atoms`.
    /// @param cell row-major 3×3 simulation cell [Å] (per-frame; supports NPT).
    /// @return false when molrs rejected the frame (bad shape / degenerate
    /// box) — state unchanged.
    bool accumulate(const std::vector<double>& positions,
                    const std::array<double, 9>& cell);

    /// @brief Number of accepted frames.
    int n_frames() const;

    /// @brief Normalized g(r) (one value per bin); empty before any accepted
    /// frame. The accumulator may keep accumulating afterwards.
    std::vector<double> finalize() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Streaming MSD accumulator — wraps molrs `MsdAccumulator` (Direct
/// curve, frame 0 = reference, plus windowed-MSD sums capped at `window`
/// lags over a ring buffer; O(window·n_dof) state).
class MSDAccumulator {
  public:
    /// @param window max windowed lag [frames] the Einstein-D fit resolves
    /// (≥ 0; 0 = Direct curve only, `diffusion()` then returns NaN).
    /// @throws std::invalid_argument when negative (no-fallback).
    explicit MSDAccumulator(int window);
    ~MSDAccumulator();
    MSDAccumulator(MSDAccumulator&&) noexcept;
    MSDAccumulator& operator=(MSDAccumulator&&) noexcept;
    MSDAccumulator(const MSDAccumulator&) = delete;
    MSDAccumulator& operator=(const MSDAccumulator&) = delete;

    /// @brief Fold one frame of positions (blocked `x|y|z` [Å]; the first
    /// frame latches the DOF count and becomes the Direct-mode reference).
    /// @return false on a shape/DOF mismatch — state unchanged.
    bool accumulate(const std::vector<double>& positions);

    /// @brief Number of accepted frames.
    int n_frames() const;

    /// @brief Direct-mode MSD(t) curve [Å²] (one value per accepted frame,
    /// index 0 = 0) — LAMMPS `compute msd` semantics.
    std::vector<double> direct_curve() const;

    /// @brief Einstein D = windowed-MSD slope / (2·dims) over the
    /// `[fit_lo, fit_hi]` fraction of the max resolved lag (≤ `window`).
    /// @param dt time between frames [fs]. @param dims spatial dimensionality.
    /// @return D [Å²/fs]; NaN on bad args, < 2 frames, or `window == 0`.
    double diffusion(double dt, int dims, double fit_lo, double fit_hi) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Streaming DOF-averaged velocity-ACF accumulator — wraps molrs
/// `VacfAccumulator` (lags `0..=resolution`; O(resolution·n_dof) state;
/// matches the batch FFT path to round-off).
class VACFAccumulator {
  public:
    /// @param resolution max lag [frames] (≥ 1).
    /// @throws std::invalid_argument when < 1 (no-fallback — a streaming
    /// accumulator cannot defer "full trajectory" resolution).
    explicit VACFAccumulator(int resolution);
    ~VACFAccumulator();
    VACFAccumulator(VACFAccumulator&&) noexcept;
    VACFAccumulator& operator=(VACFAccumulator&&) noexcept;
    VACFAccumulator(const VACFAccumulator&) = delete;
    VACFAccumulator& operator=(const VACFAccumulator&) = delete;

    /// @brief Fold one frame of velocities (blocked `vx|vy|vz` [Å/fs]; the
    /// first frame latches the DOF count).
    /// @return false on a DOF mismatch or empty frame — state unchanged.
    bool accumulate(const std::vector<double>& velocities);

    /// @brief Number of accepted frames.
    int n_frames() const;

    /// @brief Mean-subtracted, DOF-averaged ACF for lags
    /// `0..=min(resolution, n_frames − 1)` [(Å/fs)²]; empty with < 2 frames.
    std::vector<double> finalize() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace compute
} // namespace cpu
} // namespace ATV
