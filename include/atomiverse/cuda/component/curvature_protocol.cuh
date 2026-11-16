/// @file cuda/component/curvature_protocol.cuh
/// @brief CurvatureSource / ExternalDrive / CurvatureConsumer — the CUDA
///        twins of `cpu/component/curvature_protocol.h`: opt-in protocols
///        for the second derivatives of the interaction potential
///        (configurational temperature and its thermostat).
///
/// Every force term of a run with a `CurvatureConsumer` either supplies the
/// Hessian applied to a per-atom field, its Laplacian and the gradient of
/// the total Laplacian (`CurvatureSource`), or declares itself outside the
/// interaction energy (`ExternalDrive`). `cuda::Driver::build()` hands both
/// lists to the consumers and refuses a run in which a force term is
/// neither, so the temperature can never silently miss a term.
///
/// All rows are device columns indexed by frame row; every call only
/// enqueues kernels on the launch stream (no host read), so a consumer
/// stays graph-capturable.
#pragma once

#include <cstdint>
#include <vector>

namespace ATV {
namespace cuda {

struct LaunchContext;

/// @brief Device rows a `CurvatureSource` reads and accumulates into
///        (frame-row indexed, owned rows [0, `n`)).
struct CurvatureRows {
    int n = 0; ///< Owned rows.
    /// The vector field \(u\) the Hessian is applied to [eV/Å].
    const double* ux = nullptr;
    const double* uy = nullptr;
    const double* uz = nullptr;
    /// \(\mathrel{+}= (H u)_i = \sum_j \partial^2 U/\partial r_i \partial r_j
    /// \cdot u_j\) [eV/Å² × u].
    double* hx = nullptr;
    double* hy = nullptr;
    double* hz = nullptr;
    /// \(\mathrel{+}= \nabla_i^2 U\) [eV/Å²].
    double* lap = nullptr;
    /// \(\mathrel{+}= \nabla_i \sum_j \nabla_j^2 U\) [eV/Å³].
    double* gx = nullptr;
    double* gy = nullptr;
    double* gz = nullptr;
};

/// @brief A force term that supplies the second derivatives of its energy.
class CurvatureSource {
  public:
    virtual ~CurvatureSource() = default;

    /// @brief Enqueue this term's \(H u\), Laplacian and Laplacian gradient
    ///        at the frame's current positions.
    ///
    /// Called after the force phase of the same configuration (the
    /// published neighbor list and the term's own caches are current). Must
    /// walk exactly the interactions its force walk does.
    /// @param ctx  Launch context of the calling component.
    /// @param rows Accumulators (see `CurvatureRows`).
    virtual void add_curvature(const LaunchContext& ctx,
                               const CurvatureRows& rows) = 0;
};

/// @brief A force term that is not part of the interaction potential (an
///        applied field): excluded from the configurational temperature.
class ExternalDrive {
  public:
    virtual ~ExternalDrive() = default;

    /// @brief Enqueue the subtraction of this term's force from rows [0, n).
    /// @param ctx      Launch context (frame columns, stream).
    /// @param n        Owned rows.
    /// @param fx,fy,fz Device force rows to subtract from [eV/Å].
    virtual void subtract_drive_force(const LaunchContext& ctx, int n,
                                      double* fx, double* fy,
                                      double* fz) const = 0;

    /// @brief Changes whenever `subtract_drive_force` would enqueue different
    ///        kernel arguments; a consumer folds it into its `graph_epoch()`.
    virtual std::uint64_t drive_epoch() const { return 0; }
};

/// @brief A component that needs every curvature source and drive of its
///        run (configurational temperature / thermostat).
class CurvatureConsumer {
  public:
    virtual ~CurvatureConsumer() = default;

    /// @brief Receive the run's force terms, before any component build.
    /// @param sources Every force term's curvature (non-owning).
    /// @param drives  Every external drive (non-owning).
    virtual void
    set_curvature_terms(std::vector<CurvatureSource*> sources,
                        std::vector<const ExternalDrive*> drives) = 0;
};

} // namespace cuda
} // namespace ATV
