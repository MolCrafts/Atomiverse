/// @file cpu/longrange/fmm_tree.cpp
/// @brief Adaptive octree FMM for open-boundary point charges.

#include "atomiverse/cpu/longrange/fmm_tree.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ATV {
namespace longrange {

FmmTree::FmmTree(int order, int max_per_leaf)
    : order_(order), max_per_leaf_(max_per_leaf), m2m_(order), m2l_(order),
      l2l_(order) {
    if (order < 1)
        throw std::invalid_argument("FmmTree: order must be >= 1");
    if (max_per_leaf < 1)
        throw std::invalid_argument("FmmTree: max_per_leaf must be >= 1");
}

Vec3 FmmTree::node_center(int i) const {
    return nodes_.at(static_cast<std::size_t>(i)).center;
}

AccReal FmmTree::node_half_width(int i) const {
    return nodes_.at(static_cast<std::size_t>(i)).half;
}

bool FmmTree::node_is_leaf(int i) const {
    return nodes_.at(static_cast<std::size_t>(i)).leaf;
}

int FmmTree::node_particle_count(int i) const {
    return static_cast<int>(
        nodes_.at(static_cast<std::size_t>(i)).particles.size());
}

bool FmmTree::well_separated(int a, int b) const {
    // Multipole acceptance: c = |cA−cB| / (rA+rB) ≥ 2, with r = half·√3 the
    // cell circumsphere. For equal cells this is centre distance ≥ 4 h √3 ≈
    // 3.46 cell half-widths ≈ 1.7 cell widths — the classical interaction-list
    // shell (Greengard–Rokhlin ≥1 empty cell between cubes). Pairs that fail
    // the MAC go to the near-field direct sum.
    if (a == b)
        return false;
    const auto& A = nodes_[static_cast<std::size_t>(a)];
    const auto& B = nodes_[static_cast<std::size_t>(b)];
    const AccReal ra = A.half * std::sqrt(AccReal(3));
    const AccReal rb = B.half * std::sqrt(AccReal(3));
    const AccReal dist = (A.center - B.center).norm();
    return dist >= AccReal(2) * (ra + rb);
}

bool FmmTree::adjacent(int a, int b) const {
    // "Near" for the direct-sum stage: anything that is not well-separated
    // (including self and the classical 26-neighbour block).
    return !well_separated(a, b);
}

void FmmTree::subdivide(int node_id, const std::vector<Vec3>& positions) {
    auto& node = nodes_[static_cast<std::size_t>(node_id)];
    if (static_cast<int>(node.particles.size()) <= max_per_leaf_)
        return;

    const AccReal h2 = 0.5 * node.half;
    const Vec3 c = node.center;
    // Create 8 children.
    int child_ids[8];
    for (int oct = 0; oct < 8; ++oct) {
        const AccReal sx = (oct & 1) ? h2 : -h2;
        const AccReal sy = (oct & 2) ? h2 : -h2;
        const AccReal sz = (oct & 4) ? h2 : -h2;
        Node ch(order_, Vec3{c.x() + sx, c.y() + sy, c.z() + sz}, h2);
        ch.parent = node_id;
        child_ids[oct] = static_cast<int>(nodes_.size());
        nodes_.push_back(std::move(ch));
        // re-fetch node after push_back (vector may reallocate)
        nodes_[static_cast<std::size_t>(node_id)].child[oct] = child_ids[oct];
    }
    // Bin particles (copy list first — node may be invalidated conceptually).
    std::vector<int> plist =
        nodes_[static_cast<std::size_t>(node_id)].particles;
    nodes_[static_cast<std::size_t>(node_id)].particles.clear();
    nodes_[static_cast<std::size_t>(node_id)].leaf = false;

    for (int pi : plist) {
        const Vec3& p = positions[static_cast<std::size_t>(pi)];
        int oct = 0;
        if (p.x() >= c.x())
            oct |= 1;
        if (p.y() >= c.y())
            oct |= 2;
        if (p.z() >= c.z())
            oct |= 4;
        nodes_[static_cast<std::size_t>(child_ids[oct])].particles.push_back(
            pi);
    }

    for (int oct = 0; oct < 8; ++oct)
        subdivide(child_ids[oct], positions);
}

void FmmTree::build(const std::vector<Vec3>& positions,
                    const std::vector<AccReal>& charges) {
    (void)charges;
    nodes_.clear();
    interactions_.clear();
    const std::size_t N = positions.size();
    if (N == 0)
        return;

    // Bounding cube.
    AccReal lo[3] = {positions[0].x(), positions[0].y(), positions[0].z()};
    AccReal hi[3] = {lo[0], lo[1], lo[2]};
    for (std::size_t i = 1; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], positions[i][d]);
            hi[d] = std::max(hi[d], positions[i][d]);
        }
    }
    AccReal extent = 0.0;
    Vec3 mid;
    for (int d = 0; d < 3; ++d) {
        mid[d] = 0.5 * (lo[d] + hi[d]);
        extent = std::max(extent, hi[d] - lo[d]);
    }
    // Inflate slightly so particles on the face bin stably into a child.
    if (extent <= 0.0)
        extent = 1.0;
    extent *= 1.0 + 1e-9;
    const AccReal half = 0.5 * extent;

    nodes_.emplace_back(order_, mid, half);
    nodes_[0].particles.resize(N);
    for (std::size_t i = 0; i < N; ++i)
        nodes_[0].particles[i] = static_cast<int>(i);

    subdivide(0, positions);
}

void FmmTree::upward(const std::vector<Vec3>& positions,
                     const std::vector<AccReal>& charges) {
    // Post-order: children first. nodes_ is stored root-first, so reverse.
    for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
        auto& node = nodes_[static_cast<std::size_t>(i)];
        node.multipole.center() = node.center;
        node.multipole.clear();
        node.local.center() = node.center;
        node.local.clear();

        if (node.leaf) {
            for (int pi : node.particles) {
                node.multipole.accumulate(
                    charges[static_cast<std::size_t>(pi)],
                    positions[static_cast<std::size_t>(pi)]);
            }
        } else {
            for (int oct = 0; oct < 8; ++oct) {
                const int c = node.child[oct];
                if (c < 0)
                    continue;
                m2m_.apply(nodes_[static_cast<std::size_t>(c)].multipole,
                           node.multipole);
            }
        }
    }
}

void FmmTree::dual_tree(int a, int b) {
    auto& A = nodes_[static_cast<std::size_t>(a)];
    auto& B = nodes_[static_cast<std::size_t>(b)];

    // Skip empty subtrees (no multipole mass → no contribution).
    if (A.leaf && A.particles.empty())
        return;
    if (B.leaf && B.particles.empty())
        return;

    // Same-node: expand to child pairs (including self-pairs). Never M2L.
    if (a == b) {
        if (A.leaf)
            return;
        int ch[8];
        int nch = 0;
        for (int oct = 0; oct < 8; ++oct)
            if (A.child[oct] >= 0)
                ch[nch++] = A.child[oct];
        for (int i = 0; i < nch; ++i)
            for (int j = i; j < nch; ++j)
                dual_tree(ch[i], ch[j]);
        return;
    }

    if (well_separated(a, b)) {
        // Each unordered pair (a,b) is visited once from the parent split.
        m2l_.apply(B.multipole, A.local);
        m2l_.apply(A.multipole, B.local);
        interactions_.emplace_back(b, a);
        interactions_.emplace_back(a, b);
        return;
    }

    // Near-field pair of distinct cells: open the larger (or both).
    if (A.leaf && B.leaf)
        return;

    if (A.leaf && !B.leaf) {
        for (int oct = 0; oct < 8; ++oct) {
            const int c = B.child[oct];
            if (c >= 0)
                dual_tree(a, c);
        }
        return;
    }
    if (B.leaf && !A.leaf) {
        for (int oct = 0; oct < 8; ++oct) {
            const int c = A.child[oct];
            if (c >= 0)
                dual_tree(c, b);
        }
        return;
    }
    // Both internal: split the larger cell; on tie split A.
    if (A.half >= B.half) {
        for (int oct = 0; oct < 8; ++oct) {
            const int c = A.child[oct];
            if (c >= 0)
                dual_tree(c, b);
        }
    } else {
        for (int oct = 0; oct < 8; ++oct) {
            const int c = B.child[oct];
            if (c >= 0)
                dual_tree(a, c);
        }
    }
}

void FmmTree::downward() {
    // Pre-order: parent local → children via L2L, then children process.
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        auto& node = nodes_[i];
        if (node.leaf)
            continue;
        for (int oct = 0; oct < 8; ++oct) {
            const int c = node.child[oct];
            if (c < 0)
                continue;
            l2l_.apply(node.local, nodes_[static_cast<std::size_t>(c)].local);
        }
    }
}

void FmmTree::evaluate_leaves(const std::vector<Vec3>& positions,
                              const std::vector<AccReal>& charges,
                              std::vector<AccReal>& phi) const {
    const std::size_t N = positions.size();
    phi.assign(N, 0.0);

    // Non-empty leaves only (empty cubes from octree splits are skipped).
    std::vector<int> leaves;
    for (std::size_t i = 0; i < nodes_.size(); ++i)
        if (nodes_[i].leaf && !nodes_[i].particles.empty())
            leaves.push_back(static_cast<int>(i));

    // Far field: L2P at every particle.
    for (int li : leaves) {
        const auto& leaf = nodes_[static_cast<std::size_t>(li)];
        for (int pi : leaf.particles) {
            phi[static_cast<std::size_t>(pi)] +=
                leaf.local.evaluate(positions[static_cast<std::size_t>(pi)]);
        }
    }

    // Near field: direct sum over non-WS leaf pairs (including self).
    for (std::size_t ia = 0; ia < leaves.size(); ++ia) {
        for (std::size_t ib = ia; ib < leaves.size(); ++ib) {
            const int a = leaves[ia];
            const int b = leaves[ib];
            if (!adjacent(a, b))
                continue;
            const auto& A = nodes_[static_cast<std::size_t>(a)];
            const auto& B = nodes_[static_cast<std::size_t>(b)];
            for (int pi : A.particles) {
                const Vec3& ri = positions[static_cast<std::size_t>(pi)];
                const AccReal qi = charges[static_cast<std::size_t>(pi)];
                for (int pj : B.particles) {
                    if (a == b && pj <= pi)
                        continue; // handle pair once; fill both potentials
                    const Vec3& rj = positions[static_cast<std::size_t>(pj)];
                    const AccReal qj = charges[static_cast<std::size_t>(pj)];
                    const Vec3 d = ri - rj;
                    const AccReal r = d.norm();
                    if (r == 0.0)
                        continue;
                    const AccReal inv = 1.0 / r;
                    phi[static_cast<std::size_t>(pi)] += qj * inv;
                    phi[static_cast<std::size_t>(pj)] += qi * inv;
                }
            }
        }
    }
}

void FmmTree::potentials(const std::vector<Vec3>& positions,
                         const std::vector<AccReal>& charges,
                         std::vector<AccReal>& out) {
    if (positions.size() != charges.size())
        throw std::invalid_argument("FmmTree: positions/charges size mismatch");

    const std::size_t N = positions.size();
    if (N == 0) {
        out.clear();
        return;
    }
    if (N == 1) {
        out.assign(1, 0.0);
        // Still build a trivial tree for diagnostics.
        build(positions, charges);
        upward(positions, charges);
        return;
    }

    build(positions, charges);
    upward(positions, charges);

    // Dual-tree walk starting at the root with itself.
    interactions_.clear();
    dual_tree(0, 0);
    downward();
    evaluate_leaves(positions, charges, out);
}

AccReal FmmTree::energy(const std::vector<Vec3>& positions,
                        const std::vector<AccReal>& charges) {
    std::vector<AccReal> phi;
    potentials(positions, charges, phi);
    AccReal e = 0.0;
    for (std::size_t i = 0; i < charges.size(); ++i)
        e += charges[i] * phi[i];
    return 0.5 * e;
}

} // namespace longrange
} // namespace ATV
