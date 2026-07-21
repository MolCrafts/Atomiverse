/// @file cpu/replica/replica_expansion.h
/// @brief ReplicaExpansion — the strategy that turns ONE Frame into N replicas.
///
/// The Driver facade takes a single Frame (`run(frame, ...)`), and that is not an
/// accident to be worked around: Frame is the engine's sole input/output currency. A
/// category that advances N coupled replicas must therefore *generate* its replicas
/// rather than ask the user for them — which is also how the methods actually work:
///
/// | Method   | Expansion                     | Per-replica parameter |
/// |----------|-------------------------------|-----------------------|
/// | NEB      | interpolate between endpoints | path coordinate       |
/// | PIMD     | identity × P beads            | bead index            |
/// | REMD     | identity × N                  | temperature           |
/// | Umbrella | identity × N                  | bias centre           |
///
/// Nobody hands REMD sixteen different starting structures; they hand it one, plus a
/// temperature ladder. The ladder belongs to the *expansion*, not to the driver — which
/// is exactly what keeps the Path category from being welded to NEB.
#pragma once

#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

/// @brief One replica: a Frame plus the scalar that distinguishes it from its siblings.
struct Replica {
    cpu::Frame frame; ///< This replica's own copy of the system state.

    /// @brief The scalar that makes this replica different from the others.
    ///
    /// Its *meaning* belongs to the expansion that produced it — NEB writes the
    /// normalized path coordinate (0 at the initial state, 1 at the final state), REMD
    /// would write the temperature [K], PIMD the bead index. The coupling component
    /// reads it back and knows what it means because it was written for that component.
    ///
    /// A scalar covers every expansion listed in the file header. Umbrella sampling with
    /// a multi-dimensional bias centre would need more — at which point this becomes a
    /// small struct, not a double. It is not one today because nothing needs it to be.
    double parameter = 0.0;
};

/// @brief Abstract strategy: one Frame in, N replicas out (and back again).
class ReplicaExpansion {
  public:
    virtual ~ReplicaExpansion() = default;

    /// @brief How many replicas this expansion produces. Known before expanding.
    virtual int n_replicas() const = 0;

    /// @brief Expand the single input Frame into `n_replicas()` replicas.
    /// @param initial The Frame handed to `Driver::run()`.
    /// @return Exactly `n_replicas()` replicas, each a full independent Frame.
    virtual std::vector<Replica> expand(const cpu::Frame& initial) const = 0;

    /// @brief Fold the converged replicas back into the single output Frame.
    ///
    /// This is what keeps "Frame in, Frame out" true for a category that internally
    /// runs N of them. For CI-NEB the answer the user wants is one configuration — the
    /// saddle point — so the collapse selects the climbing image, not an average.
    ///
    /// @param replicas The advanced replica set.
    /// @param out The caller's Frame, overwritten with the result geometry.
    virtual void collapse(const std::vector<Replica>& replicas,
                          cpu::Frame& out) const = 0;
};

/// @brief Interpolate between the run's initial Frame and a final Frame (NEB).
///
/// The final state is this strategy's OWN configuration, not a second argument to
/// `run()`. That is the whole point: NEB's "two endpoints" would otherwise force the
/// Path driver to carry an NEB-shaped signature, and the next method to arrive (PIMD,
/// which has no endpoints at all) would not fit through it.
///
/// Interpolation is linear in Cartesian coordinates. That is the textbook starting band
/// and it is what the Müller-Brown regression uses. It is also known to be a poor guess
/// for real reaction paths, where atoms can be interpolated straight through each other
/// — IDPP (Smidstrup et al., J. Chem. Phys. 140, 214106 (2014)) is the fix, and it is a
/// second ReplicaExpansion, not a flag on this one.
class LinearInterpolation : public ReplicaExpansion {
  public:
    /// @param final_state Geometry of the product state. Must match the initial Frame's
    ///        atom count — a band between systems with different atoms is not a band.
    /// @param n_images Total images INCLUDING both endpoints. Must be >= 3: with 2 there
    ///        is nothing between the endpoints to relax, which is not a path calculation.
    LinearInterpolation(cpu::Frame final_state, int n_images);

    int n_replicas() const override { return n_images_; }
    std::vector<Replica> expand(const cpu::Frame& initial) const override;

    /// @brief Collapse to the highest-energy image — the saddle-point estimate.
    ///
    /// After CI-NEB has converged, the climbing image *is* the saddle, so "the highest
    /// energy replica" and "the climbing image" are the same row, and reading the energy
    /// keeps this correct even for a plain (non-climbing) band, where it degrades
    /// gracefully to the best available estimate rather than silently returning image 0.
    void collapse(const std::vector<Replica>& replicas,
                  cpu::Frame& out) const override;

  private:
    cpu::Frame final_;
    int n_images_;
};

/// @brief N identical copies of the input Frame, each tagged with its own parameter.
///
/// The shape PIMD (bead index), REMD (temperature) and umbrella sampling (bias centre)
/// all need. **Nothing in this repository uses it yet**, and it ships anyway: it is the
/// proof that `ReplicaExpansion` is a general strategy and not an NEB constructor
/// wearing a strategy's name. If the abstraction only ever admitted one implementation,
/// it would not have earned the indirection.
class IdentityReplication : public ReplicaExpansion {
  public:
    /// @param parameters One entry per replica — the temperature ladder, the bead
    ///        indices, the bias centres. Its length IS the replica count, so a caller
    ///        cannot ask for 8 replicas and supply 7 temperatures.
    explicit IdentityReplication(std::vector<double> parameters);

    int n_replicas() const override {
        return static_cast<int>(parameters_.size());
    }
    std::vector<Replica> expand(const cpu::Frame& initial) const override;

    /// @brief Collapse to replica 0.
    ///
    /// For REMD that is the physical (lowest-temperature) ensemble; for PIMD the
    /// centroid would be the right answer instead. Whichever it is, it belongs to the
    /// expansion — which is the argument this class exists to make.
    void collapse(const std::vector<Replica>& replicas,
                  cpu::Frame& out) const override;

  private:
    std::vector<double> parameters_;
};

} // namespace cpu
} // namespace ATV
