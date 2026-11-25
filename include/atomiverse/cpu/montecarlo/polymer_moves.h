/// @file cpu/montecarlo/polymer_moves.h
/// @brief Geometry proposals for flexible, fixed-connectivity polymers.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/component/mover.h"

namespace ATV::cpu::mc {

/// Shared rotation kernel for the tag-selected polymer moves below.
class PolymerRotationMove : public Mover {
  public:
    std::string name() const override;
    std::vector<DefaultBinding> default_bindings() const override;
    bool requires_full_frame() const override { return true; }
    void launch(const LaunchContext& ctx) override;

  protected:
    enum class Kind { Torsion, EndMer, Libration, DimerFlip, Pivot };
    PolymerRotationMove(Kind kind, std::vector<std::vector<TagInt>> paths,
                        double max_angle);

  private:
    Kind kind_;
    std::vector<std::vector<TagInt>> paths_;
    double max_angle_;
};

/// For each (fixed, moving) bond, rotate the moving side around the bond.
class TorsionMove : public PolymerRotationMove {
  public:
    TorsionMove(const std::vector<std::array<TagInt, 2>>& bonds,
                double max_angle);
};

/// For each (fixed, axis, end) path, rotate a terminal bead about the first bond.
class EndMerRotationMove : public PolymerRotationMove {
  public:
    EndMerRotationMove(const std::vector<std::array<TagInt, 3>>& paths,
                       double max_angle);
};

/// Rotate the center and its attached branches about the line through its
/// two fixed neighbors. Requires flexible angles at the cuts.
class LibrationMove : public PolymerRotationMove {
  public:
    LibrationMove(const std::vector<std::array<TagInt, 3>>& paths,
                  double max_angle);
};

/// Rotate two adjacent beads and their branches about the line through the
/// two exterior anchors, keeping each of the three path bond lengths.
class DimerFlipMove : public PolymerRotationMove {
  public:
    DimerFlipMove(const std::vector<std::array<TagInt, 4>>& paths,
                  double max_angle);
};

/// Rotate the moving side of a cut bond about a random isotropic axis
/// through the fixed bead. Requires flexible angles at the pivot.
class PivotMove : public PolymerRotationMove {
  public:
    PivotMove(const std::vector<std::array<TagInt, 2>>& bonds,
              double max_angle);
};

/// Slithering snake on fully flexible linear homopolymers. A selected
/// ordered chain must be a whole bonded component. Coordinates shift by
/// one place and the new endpoint bond is drawn from a Cartesian Gaussian.
class ReptationMove : public Mover {
  public:
    ReptationMove(std::vector<std::vector<TagInt>> chains, double bond_sigma);
    std::string name() const override { return "reptation"; }
    std::vector<DefaultBinding> default_bindings() const override;
    bool requires_full_frame() const override { return true; }
    void launch(const LaunchContext& ctx) override;

  private:
    std::vector<std::vector<TagInt>> chains_;
    double bond_sigma_;
};

/// Transfer a terminal bead between two linear homopolymers with identical
/// bead and bond parameters. The ensemble allows chain lengths to fluctuate.
class GeneralizedReptationMove : public Mover {
  public:
    explicit GeneralizedReptationMove(double bond_sigma);
    std::string name() const override { return "generalized_reptation"; }
    std::vector<DefaultBinding> default_bindings() const override;
    bool requires_full_frame() const override { return true; }
    bool requires_tag_index() const override { return true; }
    bool changes_connectivity() const override { return true; }
    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

  private:
    double bond_sigma_;
};

/// Local reversible closure branch of a concerted or parallel rotation.
class PolymerClosureMove : public Mover {
  public:
    std::string name() const override;
    std::vector<DefaultBinding> default_bindings() const override;
    bool requires_full_frame() const override { return true; }
    void launch(const LaunchContext& ctx) override;

  protected:
    PolymerClosureMove(std::vector<std::vector<TagInt>> paths, double max_angle,
                       bool parallel);

  private:
    std::vector<std::vector<TagInt>> paths_;
    double max_angle_;
    bool parallel_;
};

/// Change one driver torsion and solve five dependent torsions. A path has
/// eight atoms. Its first two and last two atoms remain fixed.
class ConRotMove : public PolymerClosureMove {
  public:
    ConRotMove(const std::vector<std::array<TagInt, 8>>& paths,
               double max_angle);
};

/// Change one driver torsion and solve three dependent torsions. Each
/// six-atom path ends with two bonds fixing the moving tail's orientation.
class ParallelRotationMove : public PolymerClosureMove {
  public:
    ParallelRotationMove(const std::vector<std::array<TagInt, 6>>& paths,
                         double max_angle);
};

/// Exchange the spatial sites of two selected, unbonded atoms. Their tag,
/// force-field type, charge and mass travel together; composition is fixed.
class AtomIdentityExchangeMove : public Mover {
  public:
    explicit AtomIdentityExchangeMove(std::vector<std::array<TagInt, 2>> pairs);
    std::string name() const override { return "atom_identity_exchange"; }
    std::vector<DefaultBinding> default_bindings() const override;
    bool requires_full_frame() const override { return true; }
    void launch(const LaunchContext& ctx) override;

  private:
    std::vector<std::array<TagInt, 2>> pairs_;
};

/// Grow a terminal segment, one bead at a time, with configurational bias.
class CBMCMove : public Mover {
  public:
    CBMCMove(std::vector<std::vector<TagInt>> paths, int trials,
             double bond_sigma, double bias_beta);
    std::string name() const override;
    std::vector<DefaultBinding> default_bindings() const override;
    bool requires_full_frame() const override { return true; }
    void launch(const LaunchContext& ctx) override;

  protected:
    CBMCMove(std::vector<std::vector<TagInt>> paths, int trials,
             double bond_sigma, double bias_beta, bool fixed_endpoint);

  private:
    std::vector<std::vector<TagInt>> paths_;
    int trials_;
    double bond_sigma_;
    double bias_beta_;
    bool fixed_endpoint_;
};

/// Regrow the interior of a flexible path between two fixed endpoint beads.
/// Conditional Gaussian trials account for the fixed endpoint at every step.
class ExtendedCBMCMove : public CBMCMove {
  public:
    ExtendedCBMCMove(std::vector<std::vector<TagInt>> paths, int trials,
                     double bond_sigma, double bias_beta);
};

} // namespace ATV::cpu::mc
