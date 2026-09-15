/// @file cpu/dynamics/fix_rattle.h
/// @brief cpu::md::FixRattle — iterative holonomic distance constraints.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/constraint.h"
#include "atomiverse/cpu/component/constraint_protocol.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief RATTLE: hold a set of interatomic distances rigid (Andersen 1983).
///
/// The engine's first constraint component, and the first user of
/// `ComponentRole::Constraint` and the `ConstraintProvider` protocol.
///
/// ## What it does
///
/// A rigid bond is a **holonomic constraint** `|r_i - r_j|^2 - d^2 = 0`. RATTLE
/// enforces it twice per step, because a constraint on the coordinates implies a
/// second one on the velocities:
///
/// - **Position stage** (`post_integrate`) — after the integrator has drifted the
///   atoms, project them back onto the constraint manifold: `sigma = |r|^2 - d^2 = 0`.
/// - **Velocity stage** (`final_integrate`) — after the second half-kick, remove the
///   velocity component along each bond: `d(sigma)/dt = 2 (r_i - r_j).(v_i - v_j) = 0`.
///   Skip this and the bond lengths stay right while the molecule silently absorbs
///   energy along its own bonds.
///
/// Both stages iterate to a tolerance (the constraints are coupled — correcting one
/// bond of a triangle perturbs the other two).
///
/// ## Why it is not tied to water
///
/// SETTLE solves the same problem *analytically*, but only for a three-atom rigid
/// body. RATTLE is iterative and general: it holds any set of distances rigid. A
/// rigid water is simply three distance constraints — the two O-H bonds plus the
/// H-H distance, which is what encodes the H-O-H angle. Hence
/// `constrain_bond()` + `constrain_angle()`: an angle constraint is a distance
/// constraint between the 1 and 3 atoms of the triplet.
///
/// ## Degrees of freedom
///
/// Each constraint removes one. Rigid water: 9 - 3 = 6 DOF per molecule. The
/// component reports this through `ConstraintProvider::n_constraints()`, and the
/// driver folds it into `ctx.md.n_dof_global` **before any thermostat builds** —
/// without which `T = 2 KE/(Ndof kB)` is computed against 9N/3 too many DOF and a
/// rigid-water run sits ~33 % below its setpoint while the thermostat happily drives
/// it there.
///
/// ## Ordering, and the shadow array
///
/// Three bindings:
///
/// | Phase | Order | What |
/// |---|---|---|
/// | `initial_integrate` | −500 | snapshot the (constraint-satisfying) reference positions |
/// | `post_integrate` | 0 | position stage |
/// | `final_integrate` | +200 | velocity stage |
///
/// The snapshot is not optional. SHAKE/RATTLE corrects along the **reference** bond
/// vector — the one from before the unconstrained move — and `VelocityVerlet` fuses
/// the half-kick and the drift into a single `launch()` and then wraps coordinates
/// into the box. By the time any `+order` component runs, the pre-move positions are
/// gone. So the component keeps its own shadow copy, taken at `initial_integrate`
/// @ −500, ahead of every integrator.
///
/// ## Multi-rank
///
/// Works, by the LAMMPS route: **every rank holding any atom of a cluster solves that
/// cluster redundantly** and applies the corrections to its own rows. Identical inputs
/// give identical Lagrange multipliers, so each owner ends up with exactly the
/// correction its owning rank computed, and no reverse exchange is needed.
///
/// Two things that route needs, and that the engine did not have:
///
/// 1. **A global constraint table.** A water's bonds and angle all live on the oxygen
///    (smallest tag / centre atom), so the rank owning the O discovers the cluster and
///    a rank owning only an H discovers *nothing* — yet that rank is the one
///    responsible for correcting that H. The table is therefore gathered once at
///    `build()`; it is tiny and static. Inverse masses ride along, because a ghost row
///    carries x/y/z/tag/charge only and its `mass` is never communicated.
///
/// 2. **Fresh ghost positions AND velocities at the two stages.** The engine's single
///    forward halo fires at `pre_neighbor` and carries positions only, so at
///    `post_integrate` the ghost data is a step stale and ghost velocities never exist
///    at all. After the integrator dirties owned rows this component posts
///    `ctx.md.comm->forward_comm(*this)`: XSHAKE at `post_integrate` (PBC shift
///    \(p_d L_d\) [Å]), VP at `final_integrate` (velocities [Å/fs], no shift).
class FixRattle : public cpu::Constraint,
                  public cpu::ConstraintProvider,
                  public cpu::ForwardPacker {
  public:
    /// @brief Smallest meaningful relative residual for the configured Frame
    /// storage precision. A tighter request cannot survive the write back to
    /// the position/velocity columns and is rejected instead of iterating to
    /// the cap forever.
    static constexpr double storage_tolerance_floor() noexcept {
        return 64.0
               * static_cast<double>(
                   std::numeric_limits<StorageReal>::epsilon());
    }

    /// @brief Precision-aware default: 1e-10 for double storage, the explicit
    /// representability floor for float storage.
    static constexpr double default_tolerance() noexcept {
        return storage_tolerance_floor() > 1.0e-10 ? storage_tolerance_floor()
                                                   : 1.0e-10;
    }

    /// @param dt Timestep [fs] — the position stage feeds its correction back into
    ///        the velocities as `dx/dt`, so it must match the integrator's.
    /// @param tolerance Relative tolerance on `|r|^2/d^2 - 1`. The default is
    ///        `default_tolerance()`; values below `storage_tolerance_floor()`
    ///        throw because the Frame columns cannot represent the requested result.
    /// @param max_iter Iteration cap; exceeding it throws rather than silently
    ///        returning an unconstrained molecule.
    explicit FixRattle(double dt, double tolerance = default_tolerance(),
                       int max_iter = 500);

    /// @brief Hold every bond of type `bond_type` at length `length`.
    /// @param bond_type Bond type index as stored in the frame's topology.
    /// @param length Constrained distance [Angstrom].
    void constrain_bond(int bond_type, double length);

    /// @brief Hold the 1-3 distance of every angle of type `angle_type`.
    ///
    /// This is how an angle is constrained: freeze the distance between the two
    /// outer atoms. For water, `d_13 = 2 r_OH sin(theta/2)`.
    /// @param angle_type Angle type index as stored in the frame's topology.
    /// @param distance_13 Constrained 1-3 distance [Angstrom].
    void constrain_angle(int angle_type, double distance_13);

    std::string name() const override { return "rattle"; }

    /// @brief Structural slot: a Constraint, NOT an Integrator. Tagging it as an
    /// Integrator would let a Plan with a constraint and no integrator pass
    /// `cpu::md::MD::validate_components()` and run with nothing propagating the motion.
    /* role via Constraint base */

    /// @brief Write-set: it projects both coordinates and velocities.

    const cpu::ConstraintProvider* as_constraint() const override {
        return this;
    }

    /// @brief Number of scalar constraints, globally (one per held distance).
    int n_constraints(const cpu::Context& ctx,
                      const cpu::Frame& frame) const override;

    std::vector<DefaultBinding> default_bindings() const override {
        return {
            ATV::make_binding(ATV::md::Phase::initial_integrate,
                              -500),                              // snapshot
            ATV::make_binding(ATV::md::Phase::post_integrate, 0), // positions
            ATV::make_binding(ATV::md::Phase::final_integrate,
                              +200), // velocities
        };
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief LAMMPS `FixRattle` packer mode (`fix_rattle.cpp:652-678`).
    enum class CommMode { XSHAKE, VP, V };

    /// @brief Select the packer mode (XSHAKE applies \(p_d L_d\) [Å]).
    /// @param mode Packer mode.
    void set_comm_mode(CommMode mode) { comm_mode_ = mode; }

    /// @brief Three doubles per atom (x/y/z [Å] or vx/vy/vz [Å/fs]).
    /// @return 3.
    int comm_forward() const override { return 3; }

    /// @brief Pack sendlist rows. XSHAKE applies \(p_d L_d\) [Å]; VP/V do not.
    /// @param n Number of local rows in `list` [dimensionless].
    /// @param list Local row indices to pack.
    /// @param buf Output buffer.
    /// @param pbc_flag 1 when this swap crosses a periodic image.
    /// @param pbc Per-axis image flags in `{-1,0,+1}`.
    /// @return Number of doubles written.
    int pack_forward_comm(int n, const int* list, double* buf, int pbc_flag,
                          const int* pbc) override;

    /// @brief Unpack received rows into ghost slots `[first, first+n)`.
    /// @param n Number of ghost rows [dimensionless].
    /// @param first First destination row index.
    /// @param buf Packed doubles.
    void unpack_forward_comm(int n, int first, const double* buf) override;

    /// @brief Iterations taken by the last position stage (diagnostics).
    int last_position_iters() const { return last_pos_iters_; }
    /// @brief Iterations taken by the last velocity stage (diagnostics).
    int last_velocity_iters() const { return last_vel_iters_; }

  private:
    /// A constraint as discovered from the topology: two global TAGS and a distance.
    struct Spec {
        TagInt ti = 0;
        TagInt tj = 0;
        double distance = 0.0;
    };

    /// One held distance, keyed by global tag.
    ///
    /// Tags, not rows: the table is GLOBAL (every rank holding any atom of a cluster
    /// needs the whole cluster), and rows are a per-rank, per-rebuild accident. The
    /// inverse masses are baked in because a ghost row carries x/y/z/tag/charge only —
    /// a ghost's `mass` is never communicated.
    struct Constraint {
        TagInt ti = 0;
        TagInt tj = 0;
        double d2 = 0.0;     ///< target squared distance [A^2]
        double inv_mi = 0.0; ///< 1/m_i [1/amu]
        double inv_mj = 0.0;
    };

    /// Rule: hold every bond/angle of this type at this distance.
    struct Rule {
        int type = 0;
        double distance = 0.0;
    };

    /// @brief Walk THIS RANK's owned topology and emit one Spec per matching
    ///        bond/angle. Two callers: `n_constraints()` and `build()`.
    std::vector<Spec> discover_local_(const cpu::Context& ctx,
                                      const cpu::Frame& frame) const;

    /// @brief Map every constraint's tags to current local rows (-1 = not here).
    void resolve_rows_(const cpu::LaunchContext& ctx);

    double dt_;
    double tol_;
    int max_iter_;
    std::vector<Rule> bond_rules_;
    std::vector<Rule> angle_rules_;
    std::vector<Constraint> constraints_; ///< global table, keyed by tag
    std::vector<int> rows_;               ///< resolved rows, 2 per constraint

    /// Reference (constraint-satisfying) positions, snapshotted before the
    /// integrator moves anything. Component-owned: the Frame atom Block is not a safe place
    /// to stash them, and by post_integrate the originals are gone.
    std::vector<double> ref_x_, ref_y_, ref_z_;

    int last_pos_iters_ = 0;
    int last_vel_iters_ = 0;
    CommMode comm_mode_ = CommMode::XSHAKE;
    cpu::Frame* frame_ = nullptr;
};

} // namespace md
} // namespace cpu
} // namespace ATV
