/// @file cpu/dynamics/fix_settle.h
/// @brief Analytic SETTLE constraint for rigid 3-site water (Miyamoto–Kollman).
///
/// Specialized alternative to general RATTLE for TIP3P-like triangles.
/// Ref: Miyamoto & Kollman, J. Comput. Chem. 13, 952 (1992).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/constraint.h"
#include "atomiverse/cpu/component/constraint_protocol.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief SETTLE: analytic rigid water constraint (one molecule or list).
///
/// Constructor takes OH and HH target distances [Å]. Add waters by O/H tags
/// via `add_water(tag_O, tag_H1, tag_H2)`.
///
/// ## What SETTLE actually is
///
/// SETTLE is the closed-form solution of the SHAKE equations for a three-site
/// rigid body. It is **not** "snap the triangle back to its ideal shape": the
/// answer depends on the *reference* (constraint-satisfying) triangle at time
/// `t` as well as on the unconstrained triangle at `t + dt`. Miyamoto &
/// Kollman rotate the reference triangle, about the unconstrained
/// configuration's centre of mass, onto the orientation SHAKE would have
/// converged to — three rotations (φ about the body x′ axis, ψ about y′, θ
/// about z′), each with an analytic sine.
///
/// Reconstructing the ideal triangle from the *unconstrained* geometry alone
/// reproduces the bond lengths and nothing else: with no memory of where the
/// molecule was, the reconstruction is free to mirror it, and the molecule
/// teleports every step while the geometry check stays perfectly green. Hence
/// the reference snapshot at `initial_integrate` @ −500, ahead of every
/// integrator — the same shadow-array discipline `FixRattle` uses, and for the
/// same reason.
///
/// ## Rows are not stable
///
/// Tags are the identity; rows are a per-rebuild accident (`atom_modify sort`
/// spatially reorders owned atoms at reneighbour). Rows are therefore
/// re-resolved from the Frame tag index at every `initial_integrate`, never
/// cached from `build()`.
///
/// ## Assumptions
///
/// Three sites, isosceles: the two hydrogens must carry the same mass, which
/// is what makes the closed form exist. A water whose H masses differ throws
/// at `build()` rather than being silently mis-constrained — use `FixRattle`,
/// which is iterative and general.
class FixSettle : public Constraint,
                  public ConstraintProvider,
                  public ForwardPacker {
  public:
    /// @param dt Timestep [fs] (for velocity stage).
    /// @param d_oh O–H distance [Å] (TIP3P default 0.9572).
    /// @param d_hh H–H distance [Å] (TIP3P default 1.5139).
    FixSettle(double dt, double d_oh = 0.9572, double d_hh = 1.5139);

    /// @brief Register one water by stable tags.
    void add_water(TagInt tag_O, TagInt tag_H1, TagInt tag_H2);

    std::string name() const override { return "settle"; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, -500),
                ATV::make_binding(ATV::md::Phase::post_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, +200)};
    }

    const ConstraintProvider* as_constraint() const override { return this; }

    /// @brief SETTLE names its waters by **tag** and hangs them on no topology,
    /// so a rigid-water box can legitimately carry no bonds at all. The Frame
    /// tag index is what turns those tags into rows, every step, after the
    /// spatial sort has renumbered them.
    bool requires_tag_index() const override { return true; }

    int n_constraints(const cpu::Context& /*ctx*/,
                      const cpu::Frame& /*frame*/) const override {
        return 3 * static_cast<int>(waters_.size());
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    /// @brief LAMMPS RATTLE/SHAKE packer mode. XSHAKE applies \(p_d L_d\) [Å].
    enum class CommMode { XSHAKE, VP, V };

    /// @brief Select the packer mode.
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

  private:
    struct Water {
        TagInt o, h1, h2;
        int io = -1, ih1 = -1, ih2 = -1;
    };

    bool try_resolve_rows_(const Frame& f);
    void resolve_rows_(Frame& f);
    void settle_positions_(Frame& f) const;
    void settle_velocities_(Frame& f) const;
    void snapshot_(const Frame& f);

    double dt_;
    double d_oh_, d_hh_;
    std::vector<Water> waters_;
    /// Reference (constraint-satisfying) triangle of every registered water,
    /// snapshotted before the integrator moves anything: 9 doubles per water
    /// (O, H1, H2 × x/y/z), minimum-image-unwrapped about the oxygen.
    std::vector<double> ref_;
    CommMode comm_mode_ = CommMode::XSHAKE;
    cpu::Frame* frame_ = nullptr;
};

} // namespace md
} // namespace cpu
} // namespace ATV
