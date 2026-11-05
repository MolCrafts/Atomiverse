/// @file cpu/observe/trajectory_metrics.h
/// @brief Trajectory-analysis compute objects — thin C++ wrappers over the molpy
/// bridge analyses (molpy `compute::{Msd, Vacf, LinearFit, Rdf}`) and their
/// streaming accumulators (`CorrelatorAccumulator`, `RdfAccumulator`).
///
/// The engine *measures*; molpy *analyzes*. Each analysis takes its
/// configuration as constructor arguments; `accumulator()` hands out the molpy
/// streaming accumulator — `push` one frame at a time, `finish()` reads the
/// result so far without consuming the state. That is the only calling
/// convention (molpy' MSD / VACF / RDF are accumulator-only): a stored
/// trajectory is pushed frame by frame. Memory is O(max_lag · n_dof)
/// (MSD / VACF) or O(n_bins) (RDF) — never O(trajectory).
///
/// MSD and VACF return a `TimeCorrelation` (one curve, item-averaged). The
/// Einstein self-diffusion coefficient is not an analysis of its own: it is a
/// `LinearFit` of the MSD curve composed by the caller, `D = slope / (2·dims)`
/// (molpy `compute::Msd` docs).
///
/// Every fallible call throws — molpy never answers an error with an empty
/// curve, a NaN or `false`, and neither does this layer. Constructors throw
/// `std::invalid_argument` for a configuration molpy refuses; `push` and
/// `finish` throw `std::runtime_error` carrying the molpy message (a ragged
/// buffer, a singular cell, a wrapped trajectory, fewer frames than
/// `max_lag + 1`, …). A rejected `push` leaves the accumulator unchanged.
///
/// **No analysis math lives in C++** — see molpy crates/molrs/src/compute/.
///
/// CUDA-safe: this header exposes only std types (no `rust/cxx.h`); the molpy
/// objects live behind PIMPLs constructed inside the `.cpp`, so observer
/// headers that include it stay includable from `.cu`/`.cuh`.
#pragma once

#include <array>
#include <memory>
#include <vector>

namespace ATV {
namespace cpu {
namespace compute {

/// @brief A time-correlation curve — wraps molpy `compute::TimeCorrelation`
/// (the output of `MSD`, `VACF` and their `CorrelatorAccumulator`).
///
/// Move-only: it owns the molpy object so a `LinearFit` can read it without a
/// round trip through C++ buffers.
class TimeCorrelation {
  public:
    ~TimeCorrelation();
    TimeCorrelation(TimeCorrelation&&) noexcept;
    TimeCorrelation& operator=(TimeCorrelation&&) noexcept;
    TimeCorrelation(const TimeCorrelation&) = delete;
    TimeCorrelation& operator=(const TimeCorrelation&) = delete;

    /// @brief Lag of each point, `i · dt` in the unit of the `dt` passed to
    /// `push`. @return `max_lag + 1` lag times.
    std::vector<double> lag_times() const;

    /// @brief The item-averaged curve, indexed by lag (molpy channel 0, the
    /// only channel of the mean reduction these wrappers use).
    /// @return `max_lag + 1` values.
    std::vector<double> values() const;

  private:
    friend class CorrelatorAccumulator;
    friend class LinearFit;
    struct Impl;
    explicit TimeCorrelation(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

/// @brief Streaming accumulator of `MSD` / `VACF` — wraps molpy
/// `compute::CorrelatorAccumulator`. Obtain one from `MSD::accumulator()` or
/// `VACF::accumulator()`.
class CorrelatorAccumulator {
  public:
    ~CorrelatorAccumulator();
    CorrelatorAccumulator(CorrelatorAccumulator&&) noexcept;
    CorrelatorAccumulator& operator=(CorrelatorAccumulator&&) noexcept;
    CorrelatorAccumulator(const CorrelatorAccumulator&) = delete;
    CorrelatorAccumulator& operator=(const CorrelatorAccumulator&) = delete;

    /// @brief Fold one frame.
    /// @param frame one frame, blocked by component: positions `x|y|z` [Å]
    /// for an MSD accumulator, any fixed number of velocity DOF [Å/fs] for a
    /// VACF accumulator. The first frame latches the width.
    /// @param dt sample interval [fs]; must be identical on every frame.
    /// @throws std::runtime_error when molpy rejects the frame (width change,
    /// non-finite value, changed `dt`); state unchanged.
    void push(const std::vector<double>& frame, double dt);

    /// @brief Fold one MSD position frame and check it against its cell.
    /// @param frame blocked `x|y|z` unwrapped positions [Å].
    /// @param dt sample interval [fs]. @param cell row-major 3×3 cell matrix
    /// H [Å] of this frame.
    /// @throws std::runtime_error as `push(frame, dt)`, and when a step since
    /// the previous frame exceeds half the cell's shortest perpendicular
    /// height (a trajectory wrapped into its box — molpy
    /// `ComputeError::Wrapped`); state unchanged.
    void push(const std::vector<double>& frame, double dt,
              const std::array<double, 9>& cell);

    /// @brief Number of accepted frames.
    int n_frames() const;

    /// @brief The correlation over the frames so far (state is kept).
    /// @return the curve over lags `0..=max_lag`.
    /// @throws std::runtime_error with fewer than `max_lag + 1` frames (a lag
    /// with no origin is refused, never clamped).
    TimeCorrelation finish() const;

  private:
    friend class MSD;
    friend class VACF;
    struct Impl;
    explicit CorrelatorAccumulator(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

/// @brief Time origins of an `MSD`.
enum class MsdOrigins {
    All,   ///< average every origin τ — the windowed estimator (default)
    First, ///< measure from frame 0 only (LAMMPS `compute msd`)
};

/// @brief Mean squared displacement `MSD(t) = ⟨|r_i(τ+t) − r_i(τ)|²⟩` over
/// lags `0..=max_lag` frames, averaged over atoms — wraps molpy
/// `compute::Msd`. Positions must be continuous across periodic boundaries.
class MSD {
  public:
    /// @param max_lag largest lag [frames] (≥ 1).
    /// @param origins time origins (default: every frame).
    /// @param stride take every `stride`-th frame as an origin (≥ 1; must be 1
    /// with `MsdOrigins::First`).
    /// @throws std::invalid_argument on an out-of-range parameter.
    explicit MSD(int max_lag, MsdOrigins origins = MsdOrigins::All,
                 int stride = 1);

    /// @brief A fresh streaming accumulator with this configuration.
    /// @return an empty accumulator taking blocked `x|y|z` position frames.
    CorrelatorAccumulator accumulator() const;

  private:
    int max_lag_;
    MsdOrigins origins_;
    int stride_;
};

/// @brief Velocity autocorrelation over lags `0..=max_lag` frames, each
/// degree of freedom mean-subtracted, unbiased (`1/(T − t)`) and averaged over
/// degrees of freedom — wraps molpy `compute::Vacf` with its defaults.
class VACF {
  public:
    /// @param max_lag largest lag [frames] (≥ 1).
    /// @param stride take every `stride`-th frame as an origin (≥ 1).
    /// @throws std::invalid_argument on an out-of-range parameter.
    explicit VACF(int max_lag, int stride = 1);

    /// @brief A fresh streaming accumulator with this configuration.
    /// @return an empty accumulator taking one velocity frame per `push`.
    CorrelatorAccumulator accumulator() const;

  private:
    int max_lag_;
    int stride_;
};

/// @brief Result of a `LinearFit`; `fit_start..=fit_end` are lag indices.
struct LinearFitResult {
    double slope;     ///< [curve unit / dt unit]
    double intercept; ///< [curve unit]
    double r2;        ///< coefficient of determination
    int fit_start;    ///< first lag index in the fit
    int fit_end;      ///< last lag index in the fit
};

/// @brief Ordinary-least-squares line through a `TimeCorrelation` against its
/// lag times — wraps molpy `compute::LinearFit`. Of an MSD curve, the
/// Einstein self-diffusion coefficient is `slope / (2·dims)`.
class LinearFit {
  public:
    /// @param start @param end fit window as fractions of the last lag
    /// (`0 ≤ start < end ≤ 1`).
    /// @throws std::invalid_argument outside that range.
    LinearFit(double start, double end);

    /// @brief Fit `correlation`.
    /// @param correlation the curve to fit.
    /// @return slope, intercept, r² and the lag window used.
    /// @throws std::runtime_error on a molpy fit error.
    LinearFitResult compute(const TimeCorrelation& correlation) const;

  private:
    double start_;
    double end_;
};

/// @brief g(r) over one or more frames — molpy `compute::RdfResult`.
struct RDFResult {
    std::vector<double> bin_edges;   ///< `n_bins + 1` edges [Å]
    std::vector<double> bin_centers; ///< `n_bins` centers [Å]
    std::vector<double> rdf;         ///< normalized g(r), one per bin
    std::vector<double> n_r;         ///< raw pair counts, one per bin
    double volume = 0.0; ///< cell volume summed over the frames [Å³]
                         ///< (molpy `RdfResult::volume`; ÷ n_frames = mean)
    int n_frames = 0;    ///< frames folded
};

class RDFAccumulator;

/// @brief Radial distribution function g(r) over `[r_min, r_max)` Å in
/// `n_bins` bins — wraps molpy `compute::Rdf` (per-frame periodic cell-list
/// self-query, ideal-gas shell-volume normalization).
class RDF {
  public:
    /// @param n_bins number of radial bins (≥ 1). @param r_max upper edge of
    /// the last bin [Å] (> r_min). @param r_min lower edge of bin 0 [Å] (≥ 0).
    /// Argument order mirrors molpy `Rdf::new(n_bins, r_max, r_min)`.
    /// @throws std::invalid_argument on an out-of-range parameter.
    RDF(int n_bins, double r_max, double r_min);

    /// @brief A fresh streaming accumulator with this configuration.
    /// @return an empty accumulator (O(n_bins) state).
    RDFAccumulator accumulator() const;

  private:
    int n_bins_;
    double r_max_;
    double r_min_;
};

/// @brief Streaming g(r) accumulator — wraps molpy `compute::RdfAccumulator`,
/// the one form of the g(r). Obtain one from `RDF::accumulator()`.
class RDFAccumulator {
  public:
    ~RDFAccumulator();
    RDFAccumulator(RDFAccumulator&&) noexcept;
    RDFAccumulator& operator=(RDFAccumulator&&) noexcept;
    RDFAccumulator(const RDFAccumulator&) = delete;
    RDFAccumulator& operator=(const RDFAccumulator&) = delete;

    /// @brief Fold one frame into the running histogram.
    /// @param positions blocked `x|y|z` [Å], length `3·n_atoms`.
    /// @param cell row-major 3×3 cell matrix H [Å] of this frame.
    /// @throws std::runtime_error when molpy rejects the frame (bad shape,
    /// singular cell); state unchanged.
    void push(const std::vector<double>& positions,
              const std::array<double, 9>& cell);

    /// @brief Number of accepted frames.
    int n_frames() const;

    /// @brief Normalized g(r) over the frames so far (state is kept).
    /// @return the curve with its bins.
    /// @throws std::runtime_error before any accepted frame.
    RDFResult finish() const;

  private:
    friend class RDF;
    struct Impl;
    explicit RDFAccumulator(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace compute
} // namespace cpu
} // namespace ATV
