/// @file cpu/longrange/fmm_tree.h
/// @brief Adaptive octree FMM for open-boundary point-charge Coulomb.
///
/// Backend-agnostic (no Frame / Component / MPI). Classical Greengard–Rokhlin
/// FMM: upward P2M+M2M, interaction-list M2L under the standard
/// well-separated criterion (≥ 1 cell width of separation), downward L2L,
/// leaf L2P + near-field direct sum. Open boundaries — no periodic images.
///
/// Kernel: dimensionless Σ_{i<j} q_i q_j / r_ij. Unit conversion to eV is
/// deferred to the HF/CFMM wiring (chain link 21).
///
/// References:
/// - Greengard & Rokhlin, *J. Comput. Phys.* **73**, 325 (1987),
///   DOI: 10.1016/0021-9991(87)90140-9
/// - White & Head-Gordon, *J. Chem. Phys.* **101**, 6593 (1994),
///   DOI: 10.1063/1.468354
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/longrange/fmm_operators.h"
#include "atomiverse/cpu/longrange/multipole_expansion.h"

namespace ATV {
namespace longrange {

/// @brief Adaptive octree FMM evaluator for open-boundary point charges.
///
/// Construct once with (order, max_per_leaf); each #energy / #potentials call
/// rebuilds the tree over the supplied particle set (private workspace —
/// nothing escapes onto a Frame).
class FmmTree {
  public:
    /// @param order        Multipole truncation order p ≥ 1.
    /// @param max_per_leaf Leaf capacity (≥ 1); split while a node holds more.
    /// @throws std::invalid_argument if order < 1 or max_per_leaf < 1.
    FmmTree(int order, int max_per_leaf);

    int order() const { return order_; }
    int max_per_leaf() const { return max_per_leaf_; }

    /// @brief Coulomb energy ½ Σ_i q_i Φ_i with self-terms excluded
    ///        [dimensionless; multiply by units::coulomb for eV].
    ///
    /// @param positions Particle positions [length], size N.
    /// @param charges   Particle charges [kernel charge], size N.
    /// @return Total interaction energy [dimensionless].
    AccReal energy(const std::vector<Vec3>& positions,
                   const std::vector<AccReal>& charges);

    /// @brief Per-particle potentials Φ_i = Σ_{j≠i} q_j / r_ij
    ///        [dimensionless].
    ///
    /// @param positions Particle positions [length], size N.
    /// @param charges   Particle charges [kernel charge], size N.
    /// @param[out] out  Potentials, resized to N.
    void potentials(const std::vector<Vec3>& positions,
                    const std::vector<AccReal>& charges,
                    std::vector<AccReal>& out);

    /// @brief Number of nodes in the last-built tree (0 before first call).
    int num_nodes() const { return static_cast<int>(nodes_.size()); }

    /// @brief Interaction-list pairs from the last evaluation (src, dst node
    ///        indices). Exposed for unit tests of the well-separated criterion.
    const std::vector<std::pair<int, int>>& interaction_list() const {
        return interactions_;
    }

    /// @brief Axis-aligned cell centre of node i (last tree).
    Vec3 node_center(int i) const;

    /// @brief Half-width of node i's cubic cell (last tree) [length].
    AccReal node_half_width(int i) const;

    /// @brief True if node i is a leaf.
    bool node_is_leaf(int i) const;

    /// @brief Particle count stored on leaf i (0 for internal nodes).
    int node_particle_count(int i) const;

  private:
    struct Node {
        Vec3 center;  ///< cell centre
        AccReal half; ///< half edge length of the cubic cell
        int parent = -1;
        int child[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        bool leaf = true;
        std::vector<int> particles; ///< indices into the particle arrays
        MultipoleExpansion multipole;
        LocalExpansion local;

        Node(int order, Vec3 c, AccReal h)
            : center(std::move(c)), half(h), multipole(order, center),
              local(order, center) {}
    };

    void build(const std::vector<Vec3>& positions,
               const std::vector<AccReal>& charges);
    void subdivide(int node_id, const std::vector<Vec3>& positions);
    void upward(const std::vector<Vec3>& positions,
                const std::vector<AccReal>& charges);
    void dual_tree(int a, int b);
    void downward();
    void evaluate_leaves(const std::vector<Vec3>& positions,
                         const std::vector<AccReal>& charges,
                         std::vector<AccReal>& phi) const;

    /// Neighbours: share a face/edge/corner or are the same cell (3³−1 = 26
    /// adjacent cells at the same level, plus self for near field).
    bool adjacent(int a, int b) const;
    bool well_separated(int a, int b) const;

    int order_;
    int max_per_leaf_;
    M2M m2m_;
    M2L m2l_;
    L2L l2l_;
    std::vector<Node> nodes_;
    std::vector<std::pair<int, int>> interactions_;
};

} // namespace longrange
} // namespace ATV
