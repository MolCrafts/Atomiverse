/// @file cpu/component/curvature_protocol.h
/// @brief CurvatureSource / ExternalDrive / CurvatureConsumer — opt-in
///        protocols for the second derivatives of the interaction potential
///        (configurational temperature and its thermostat). CUDA-free.
///
/// The configurational temperature of Petravic & Delhommelle (JCP 118, 7477
/// (2003), eq. 7) and its gradient need, besides the forces, the Laplacian of
/// the potential energy \(U\), the Hessian \(H = \nabla\nabla U\) applied to a
/// per-atom vector field, and the gradient of the total Laplacian. Every
/// force term must either supply them (`CurvatureSource`) or declare itself
/// outside \(U\) (`ExternalDrive`: a field that drives the system and is not
/// part of the configurational distribution). The Driver hands both lists to
/// the consumers before any component builds and refuses a run in which a
/// force term is neither, so the temperature can never silently miss a term.
///
/// Same shape as `timestep_protocol.h`: a participant inherits the protocol
/// beside `Component` and overrides the matching `as_*` accessor to return
/// `this`.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {

struct LaunchContext;

/// @brief Per-atom rows a `CurvatureSource` reads and accumulates into.
///
/// Every array spans rows [0, `n_total`) of the frame (owned and ghost).
/// Sources add their pair or k-space contributions to whichever rows they
/// touch, ghosts included; the consumer folds ghost rows back onto their
/// owners afterwards (the force convention).
struct CurvatureRows {
    int n_total = 0;
    /// The vector field \(u\) the Hessian is applied to (owned and ghost
    /// rows current).
    const double* ux = nullptr;
    const double* uy = nullptr;
    const double* uz = nullptr;
    /// \(\mathrel{+}= (H u)_i = \sum_j \partial^2 U/\partial r_i \partial r_j
    /// \cdot u_j\) [energy/length² × u].
    double* hx = nullptr;
    double* hy = nullptr;
    double* hz = nullptr;
    /// \(\mathrel{+}= \nabla_i^2 U\), the trace of atom i's diagonal Hessian
    /// block [energy/length²].
    double* lap = nullptr;
    /// \(\mathrel{+}= \nabla_i \sum_j \nabla_j^2 U\) [energy/length³].
    double* gx = nullptr;
    double* gy = nullptr;
    double* gz = nullptr;
};

/// @brief A force term that supplies the second derivatives of its energy.
class CurvatureSource {
  public:
    virtual ~CurvatureSource() = default;

    /// @brief Add this term's \(H u\), Laplacian and Laplacian gradient at
    ///        the frame's current positions.
    ///
    /// Called after the force phase of the same configuration (published
    /// neighbor lists and caches are current). Must walk exactly the
    /// interactions its force walk does.
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

    /// @brief Subtract this term's force from owned rows [0, n).
    /// @param ctx        Launch context (frame columns, units).
    /// @param n          Owned rows.
    /// @param fx,fy,fz   Force rows to subtract from [force].
    virtual void subtract_drive_force(const LaunchContext& ctx, int n,
                                      double* fx, double* fy,
                                      double* fz) const = 0;
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

} // namespace cpu
} // namespace ATV
