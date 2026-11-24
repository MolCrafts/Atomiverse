#include "atomiverse/cpu/montecarlo/polymer_moves.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

namespace ATV::cpu::mc {
namespace {

using Path = std::vector<TagInt>;
using Random = std::mt19937_64;

template <std::size_t N>
std::vector<Path> to_paths(const std::vector<std::array<TagInt, N>>& paths) {
    std::vector<Path> result;
    result.reserve(paths.size());
    for (const auto& path : paths)
        result.emplace_back(path.begin(), path.end());
    return result;
}

void validate_paths(const std::vector<Path>& paths, std::size_t minimum) {
    if (paths.empty())
        throw std::invalid_argument(
            "Polymer move requires at least one tag path");
    for (const auto& path : paths) {
        const std::unordered_set<TagInt> unique(path.begin(), path.end());
        if (path.size() < minimum || unique.size() != path.size())
            throw std::invalid_argument(
                "Polymer move requires distinct tags within each path");
    }
}

std::size_t pick(Random& rng, std::size_t n) {
    return std::uniform_int_distribution<std::size_t>(0, n - 1)(rng);
}

Vec3 mic(const Box& box, Vec3 delta) {
    box.min_image(delta.x(), delta.y(), delta.z());
    return delta;
}

bool same_image(const Box& box, const Vec3& delta) {
    const double tolerance = 64 * std::numeric_limits<double>::epsilon();
    return (mic(box, delta) - delta).norm() <= tolerance * (1 + delta.norm());
}

Vec3 rotate(const Vec3& v, const Vec3& axis, double cos_angle,
            double sin_angle) {
    return v * cos_angle + axis.cross(v) * sin_angle
           + axis * (axis.dot(v) * (1 - cos_angle));
}

Vec3 gaussian_vector(Random& rng, double sigma) {
    std::normal_distribution<double> normal(0.0, sigma);
    return {normal(rng), normal(rng), normal(rng)};
}

struct Topology {
    explicit Topology(Frame& frame)
        : neighbors(frame.nlocal()), points(frame.nlocal()) {
        const auto tag = frame.span(kAtom_tag);
        const auto xs = frame.span(kAtom_x);
        const auto ys = frame.span(kAtom_y);
        const auto zs = frame.span(kAtom_z);
        for (int i = 0; i < frame.nlocal(); ++i) {
            if (!rows.emplace(tag[i], i).second)
                throw std::runtime_error(
                    "Polymer moves require unique owned atom tags");
            points[i] = {xs[i], ys[i], zs[i]};
        }
        if (!frame.has_bonded())
            return;
        const auto count = frame.span(kAtom_num_bond);
        const auto partners = frame.span(kAtom_bond_atom);
        const auto types = frame.span(kAtom_bond_type);
        for (int i = 0; i < frame.nlocal(); ++i) {
            for (int j = 0; j < count[i]; ++j) {
                const auto partner = rows.find(partners(i, j));
                if (partner == rows.end())
                    throw std::runtime_error(
                        "Polymer moves require all bonded atoms on one rank");
                neighbors[i].emplace_back(partner->second, types(i, j));
                neighbors[partner->second].emplace_back(i, types(i, j));
            }
        }
    }

    int row_for(TagInt tag) const {
        const auto row = rows.find(tag);
        if (row == rows.end())
            throw std::invalid_argument(
                "Polymer move references an unknown atom tag");
        return row->second;
    }

    std::vector<int> bonded_path(const Path& path) const {
        std::vector<int> result;
        for (const auto tag : path) {
            const int row = row_for(tag);
            if (!result.empty() && bond(result.back(), row) == nullptr)
                throw std::invalid_argument(
                    "Polymer move path must follow existing bonds");
            result.push_back(row);
        }
        return result;
    }

    const std::pair<int, int>* bond(int a, int b) const {
        const auto& edges = neighbors[a];
        const auto it =
            std::find_if(edges.begin(), edges.end(),
                         [b](const auto& edge) { return edge.first == b; });
        return it == edges.end() ? nullptr : &*it;
    }

    std::vector<std::vector<std::pair<int, int>>> neighbors;
    std::vector<Vec3> points;
    std::unordered_map<TagInt, int> rows;
};

struct ParticleIdentity {
    explicit ParticleIdentity(Frame& frame)
        : types(frame.has("atoms", kAtom_type.key)
                    ? frame.span(kAtom_type).data()
                    : nullptr),
          elements(frame.has("atoms", kAtom_atomic_number.key)
                       ? frame.span(kAtom_atomic_number).data()
                       : nullptr),
          charges(frame.span(kAtom_charge).data()),
          masses(frame.span(kAtom_mass).data()) {}

    bool same(int a, int b) const {
        return (!types || types[a] == types[b])
               && (!elements || elements[a] == elements[b])
               && charges[a] == charges[b] && masses[a] == masses[b];
    }

    const int* types;
    const int* elements;
    const StorageReal* charges;
    const StorageReal* masses;
};

using Cut = std::pair<int, int>;

/// Unwrap one connected component with the named cut bonds removed.
/// Refuse cycles with nonzero periodic winding and cuts with alternate paths.
bool unwrap_fragment(const Topology& topology, const Box& box, int first,
                     const Vec3& first_position, const std::vector<Cut>& cuts,
                     std::vector<int>& selected, std::vector<Vec3>& positions) {
    std::vector<bool> visited(positions.size());
    selected.push_back(first);
    visited[first] = true;
    positions[first] = first_position;
    for (std::size_t cursor = 0; cursor < selected.size(); ++cursor) {
        const int a = selected[cursor];
        for (const auto& [b, type] : topology.neighbors[a]) {
            if (std::find(cuts.begin(), cuts.end(), Cut{a, b}) != cuts.end()
                || std::find(cuts.begin(), cuts.end(), Cut{b, a}) != cuts.end())
                continue;
            const Vec3 delta =
                mic(box, topology.points[b] - topology.points[a]);
            if (visited[b]) {
                if ((positions[b] - positions[a] - delta).norm()
                    > 1e-9 * (1 + positions[b].norm() + positions[a].norm()))
                    return false;
                continue;
            }
            visited[b] = true;
            positions[b] = positions[a] + delta;
            selected.push_back(b);
        }
    }
    return std::none_of(cuts.begin(), cuts.end(), [&](const Cut& cut) {
        return visited[cut.first] == visited[cut.second];
    });
}

void write_displacement(
    Frame& frame, const std::vector<int>& rows,
    const std::vector<Vec3>& displacements, const std::vector<Vec3>& original,
    const std::vector<std::array<ImageInt, 3>>& original_images) {
    const auto xs = frame.span(kAtom_x);
    const auto ys = frame.span(kAtom_y);
    const auto zs = frame.span(kAtom_z);
    auto* images = frame.span(kAtom_image).data();
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const int row = rows[i];
        const Vec3& delta = displacements[i];
        double x = original[i].x() + delta.x();
        double y = original[i].y() + delta.y();
        double z = original[i].z() + delta.z();
        auto* image = images + 3 * static_cast<std::size_t>(row);
        std::copy(original_images[i].begin(), original_images[i].end(), image);
        frame.box.wrap_into(x, y, z, image);
        xs[row] = static_cast<StorageReal>(x);
        ys[row] = static_cast<StorageReal>(y);
        zs[row] = static_cast<StorageReal>(z);
    }
}

struct Displacement {
    Displacement(Frame& frame, Transaction& txn, const std::vector<int>& rows,
                 const std::vector<Vec3>& original)
        : frame(frame), rows(rows) {
        txn.begin(Transaction::Verb::Displace, frame,
                  static_cast<int>(rows.size()));
        const auto image = frame.span(kAtom_image);
        for (const int row : rows) {
            txn.record_touch(frame, row);
            points.push_back(original[row]);
            images.push_back({image(row, 0), image(row, 1), image(row, 2)});
        }
    }

    void write(const std::vector<Vec3>& deltas) const {
        write_displacement(frame, rows, deltas, points, images);
    }

    Frame& frame;
    const std::vector<int>& rows;
    std::vector<Vec3> points;
    std::vector<std::array<ImageInt, 3>> images;
};

std::vector<DefaultBinding> proposal_binding() {
    return {make_binding(sampling::Phase::propose, 0)};
}

void require_mc(const LaunchContext& ctx) {
    if (!ctx.runtime.mc.rng || !ctx.runtime.mc.proposal)
        throw std::runtime_error("Polymer moves require an MC trial context");
}

} // namespace

PolymerRotationMove::PolymerRotationMove(Kind kind, std::vector<Path> paths,
                                         double max_angle)
    : kind_(kind), paths_(std::move(paths)), max_angle_(max_angle) {
    validate_paths(paths_, 2);
    if (!(max_angle > 0 && max_angle <= std::numbers::pi))
        throw std::invalid_argument(
            "Polymer move requires an angle in (0, pi] radians");
}

TorsionMove::TorsionMove(const std::vector<std::array<TagInt, 2>>& bonds,
                         double max_angle)
    : PolymerRotationMove(Kind::Torsion, to_paths(bonds), max_angle) {}

EndMerRotationMove::EndMerRotationMove(
    const std::vector<std::array<TagInt, 3>>& paths, double max_angle)
    : PolymerRotationMove(Kind::EndMer, to_paths(paths), max_angle) {}

LibrationMove::LibrationMove(const std::vector<std::array<TagInt, 3>>& paths,
                             double max_angle)
    : PolymerRotationMove(Kind::Libration, to_paths(paths), max_angle) {}

DimerFlipMove::DimerFlipMove(const std::vector<std::array<TagInt, 4>>& paths,
                             double max_angle)
    : PolymerRotationMove(Kind::DimerFlip, to_paths(paths), max_angle) {}

PivotMove::PivotMove(const std::vector<std::array<TagInt, 2>>& bonds,
                     double max_angle)
    : PolymerRotationMove(Kind::Pivot, to_paths(bonds), max_angle) {}

std::string PolymerRotationMove::name() const {
    switch (kind_) {
    case Kind::Torsion:
        return "torsion";
    case Kind::EndMer:
        return "end_mer_rotation";
    case Kind::Libration:
        return "libration";
    case Kind::DimerFlip:
        return "dimer_flip";
    case Kind::Pivot:
        return "pivot";
    }
    throw std::logic_error("Unknown polymer rotation move");
}

std::vector<DefaultBinding> PolymerRotationMove::default_bindings() const {
    return proposal_binding();
}

void PolymerRotationMove::launch(const LaunchContext& ctx) {
    if (ctx.step < 0)
        return;
    require_mc(ctx);
    auto& rng = *ctx.runtime.mc.rng;
    auto& frame = ctx.frame;
    const Topology topology(frame);
    const auto path = topology.bonded_path(paths_[pick(rng, paths_.size())]);
    if (kind_ == Kind::EndMer && topology.neighbors[path.back()].size() != 1)
        throw std::invalid_argument(
            "EndMerRotationMove requires a terminal bead");

    const bool interior = kind_ == Kind::Libration || kind_ == Kind::DimerFlip;
    const int first = kind_ == Kind::EndMer ? path[2] : path[1];
    const int fixed = kind_ == Kind::EndMer ? path[1] : path[0];
    std::vector<Cut> cuts{{fixed, first}};
    if (interior)
        cuts.emplace_back(path[path.size() - 2], path.back());

    std::vector<Vec3> backbone(path.size());
    backbone[0] = topology.points[path[0]];
    for (std::size_t i = 1; i < path.size(); ++i)
        backbone[i] = backbone[i - 1]
                      + mic(frame.box, topology.points[path[i]]
                                           - topology.points[path[i - 1]]);
    const Vec3 center = backbone[0];
    Vec3 axis = kind_ == Kind::Pivot
                    ? gaussian_vector(rng, 1)
                    : backbone[interior ? path.size() - 1 : 1] - center;
    if (!(axis.norm() > 0))
        return;
    axis = axis.normalized();

    std::vector<int> moving;
    std::vector<Vec3> old_positions(frame.nlocal());
    if (!unwrap_fragment(topology, frame.box, first,
                         backbone[kind_ == Kind::EndMer ? 2 : 1], cuts, moving,
                         old_positions))
        return;

    const double angle =
        std::uniform_real_distribution<double>(-max_angle_, max_angle_)(rng);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    std::vector<Vec3> updated = topology.points;
    for (const int row : moving)
        updated[row] = center + rotate(old_positions[row] - center, axis, c, s);
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (path[i] == fixed || (interior && i == path.size() - 1))
            updated[path[i]] = backbone[i];
    }

    // The reverse move unwraps the trial along the same edges. A bond that
    // would pick a different periodic image invalidates the proposal.
    std::vector<Vec3> displacements;
    displacements.reserve(moving.size());
    for (const int row : moving) {
        for (const auto& [neighbor, type] : topology.neighbors[row])
            if (!same_image(frame.box, updated[neighbor] - updated[row]))
                return;
        displacements.push_back(updated[row] - old_positions[row]);
    }
    auto& txn = *ctx.runtime.mc.proposal;
    Displacement(frame, txn, moving, topology.points).write(displacements);
    txn.log_proposal_ratio = 0;
}

ReptationMove::ReptationMove(std::vector<Path> chains, double bond_sigma)
    : chains_(std::move(chains)), bond_sigma_(bond_sigma) {
    validate_paths(chains_, 3);
    if (!(bond_sigma > 0 && std::isfinite(bond_sigma)))
        throw std::invalid_argument(
            "ReptationMove requires a finite positive bond sigma");
}

std::vector<DefaultBinding> ReptationMove::default_bindings() const {
    return proposal_binding();
}

void ReptationMove::launch(const LaunchContext& ctx) {
    if (ctx.step < 0)
        return;
    require_mc(ctx);
    auto& frame = ctx.frame;
    auto& rng = *ctx.runtime.mc.rng;
    const Topology topology(frame);
    auto chain = topology.bonded_path(chains_[pick(rng, chains_.size())]);
    if (std::bernoulli_distribution(0.5)(rng))
        std::reverse(chain.begin(), chain.end());

    const ParticleIdentity identity(frame);
    const int bond_type = topology.bond(chain[0], chain[1])->second;
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const int row = chain[i];
        if (topology.neighbors[row].size()
                != (i == 0 || i + 1 == chain.size() ? 1 : 2)
            || (i && topology.bond(chain[i - 1], row)->second != bond_type)
            || !identity.same(chain.front(), row))
            throw std::invalid_argument(
                "ReptationMove requires an entire linear homopolymer");
    }
    std::vector<Vec3> bonds;
    bonds.reserve(chain.size() - 1);
    for (std::size_t i = 1; i < chain.size(); ++i)
        bonds.push_back(mic(frame.box, topology.points[chain[i]]
                                           - topology.points[chain[i - 1]]));
    const Vec3 added = gaussian_vector(rng, bond_sigma_);
    if (!same_image(frame.box, added))
        return;

    // A cyclic coordinate shift plus an endpoint draw has unit Cartesian
    // Jacobian. The reverse direction draws back the discarded bond.
    const double ratio = 0.5
                         * ((added / bond_sigma_).norm_squared()
                            - (bonds.front() / bond_sigma_).norm_squared());
    if (!std::isfinite(ratio))
        return;
    bonds.push_back(added);
    auto& txn = *ctx.runtime.mc.proposal;
    Displacement(frame, txn, chain, topology.points).write(bonds);
    txn.log_proposal_ratio = ratio;
}

GeneralizedReptationMove::GeneralizedReptationMove(double bond_sigma)
    : bond_sigma_(bond_sigma) {
    if (!(bond_sigma > 0 && std::isfinite(bond_sigma)))
        throw std::invalid_argument(
            "GeneralizedReptationMove requires a finite positive bond sigma");
}

std::vector<DefaultBinding> GeneralizedReptationMove::default_bindings() const {
    return proposal_binding();
}

void GeneralizedReptationMove::build(const BuildContext& ctx) {
    auto& frame = ctx.frame;
    if (!frame.has_bonded())
        throw std::invalid_argument(
            "GeneralizedReptationMove requires a bonded Frame");
    const Topology topology(frame);
    const auto angles = frame.span(kAtom_num_angle);
    const auto dihedrals = frame.span(kAtom_num_dihedral);
    const auto impropers = frame.span(kAtom_num_improper);
    const ParticleIdentity identity(frame);
    std::optional<int> first;
    int kind = 0;
    for (int row = 0; row < frame.nlocal(); ++row) {
        if (angles[row] || dihedrals[row] || impropers[row])
            throw std::invalid_argument(
                "GeneralizedReptationMove supports a bond-only force field");
        const auto& bonds = topology.neighbors[row];
        if (bonds.empty())
            continue;
        if (!first) {
            first = row;
            kind = bonds.front().second;
        }
        if (bonds.size() > 2 || !identity.same(*first, row)
            || !std::ranges::all_of(
                bonds, [kind](const auto& b) { return b.second == kind; }))
            throw std::invalid_argument(
                "GeneralizedReptationMove requires identical beads and bonds "
                "on unbranched chains");
    }
    // A linear chain needs at most two owned bonds and six 1-2/1-3/1-4
    // neighbors. Fix the strides before the first transactional snapshot.
    frame.grow_bond_per_atom(2);
    frame.grow_maxspecial(6);
}

void GeneralizedReptationMove::launch(const LaunchContext& ctx) {
    if (ctx.step < 0)
        return;
    require_mc(ctx);
    auto& rng = *ctx.runtime.mc.rng;
    auto& frame = ctx.frame;
    auto& txn = *ctx.runtime.mc.proposal;
    const Topology topology(frame);
    std::vector<int> endpoints;
    for (int row = 0; row < frame.nlocal(); ++row)
        if (topology.neighbors[row].size() == 1)
            endpoints.push_back(row);
    if (endpoints.size() < 4)
        return;
    const auto donor_index = pick(rng, endpoints.size());
    auto recipient_index = pick(rng, endpoints.size() - 1);
    if (recipient_index >= donor_index)
        ++recipient_index;
    const int donor = endpoints[donor_index];
    const int recipient = endpoints[recipient_index];
    const auto [previous, kind] = topology.neighbors[donor].front();
    if (topology.neighbors[previous].size() != 2)
        return; // Keep at least two beads in each chain.

    int last = donor;
    int next = previous;
    while (topology.neighbors[next].size() == 2) {
        const auto& edges = topology.neighbors[next];
        const int onward = edges[edges[0].first == last ? 1 : 0].first;
        last = next;
        next = onward;
    }
    if (next == recipient)
        return;

    const Vec3 new_bond = gaussian_vector(rng, bond_sigma_);
    if (!same_image(frame.box, new_bond))
        return;
    const Vec3 old_bond =
        mic(frame.box, topology.points[donor] - topology.points[previous]);
    const double log_ratio = 0.5
                             * ((new_bond / bond_sigma_).norm_squared()
                                - (old_bond / bond_sigma_).norm_squared());
    if (!std::isfinite(log_ratio))
        return;

    // All rows' special-neighbor tables are recomputed for the proposed graph.
    // Snapshot them with the bonded rows before any mutation. Both directions
    // have the same number of ordered endpoint choices.
    txn.begin(Transaction::Verb::Displace, frame, frame.nlocal());
    for (int row = 0; row < frame.nlocal(); ++row)
        txn.record_touch(frame, row);
    txn.log_proposal_ratio = log_ratio;
    txn.connectivity_changed = true;
    const auto tags = frame.span(kAtom_tag);
    const TagInt donor_tag = tags[donor];
    const TagInt recipient_tag = tags[recipient];
    const bool donor_owns = donor_tag < tags[previous];
    const int owner = donor_owns ? donor : previous;
    const TagInt partner = donor_owns ? tags[previous] : donor_tag;
    auto counts = frame.span(kAtom_num_bond);
    auto partners = frame.span(kAtom_bond_atom);
    auto types = frame.span(kAtom_bond_type);
    for (int i = 0; i < counts[owner]; ++i) {
        if (partners(owner, i) != partner)
            continue;
        --counts[owner];
        partners(owner, i) = partners(owner, counts[owner]);
        types(owner, i) = types(owner, counts[owner]);
        break;
    }

    auto images = frame.span(kAtom_image);
    const auto recipient_image = std::array{
        images(recipient, 0), images(recipient, 1), images(recipient, 2)};
    write_displacement(frame, {donor}, {new_bond}, {topology.points[recipient]},
                       {recipient_image});
    frame.add_bond(donor_tag, recipient_tag, kind);
    frame.build_special_list();
}

namespace {

template <std::size_t N>
using ClosureVector = std::array<double, N>;

template <std::size_t N>
using ClosureMatrix = std::array<ClosureVector<N>, N>;

/// Pivoted elimination for the small closure equations. Return log|det A|;
/// a singular or badly conditioned configuration has no usable chart.
template <std::size_t N>
std::optional<double> solve(ClosureMatrix<N> matrix, ClosureVector<N>& rhs) {
    double log_det = 0;
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t pivot = i;
        for (std::size_t row = i + 1; row < N; ++row) {
            if (std::abs(matrix[row][i]) > std::abs(matrix[pivot][i]))
                pivot = row;
        }
        if (!(std::abs(matrix[pivot][i]) > 1e-10))
            return std::nullopt;
        std::swap(matrix[i], matrix[pivot]);
        std::swap(rhs[i], rhs[pivot]);
        log_det += std::log(std::abs(matrix[i][i]));
        for (std::size_t row = i + 1; row < N; ++row) {
            const double factor = matrix[row][i] / matrix[i][i];
            for (std::size_t col = i + 1; col < N; ++col)
                matrix[row][col] -= factor * matrix[i][col];
            rhs[row] -= factor * rhs[i];
        }
    }
    for (std::size_t i = N; i-- > 0;) {
        for (std::size_t col = i + 1; col < N; ++col)
            rhs[i] -= matrix[i][col] * rhs[col];
        rhs[i] /= matrix[i][i];
    }
    return log_det;
}

void apply_torsion(std::vector<Vec3>& points, std::size_t axis_index,
                   double angle) {
    const Vec3 origin = points[axis_index];
    const Vec3 axis = (points[axis_index + 1] - origin).normalized();
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    for (std::size_t i = axis_index + 2; i < points.size(); ++i)
        points[i] = origin + rotate(points[i] - origin, axis, c, s);
}

struct Closure {
    Closure(const std::vector<Vec3>& original, bool parallel)
        : parallel(parallel), anchor(original[original.size() - 2]),
          scale((original[1] - original[0]).norm()),
          direction(tail_direction(original)), normal(tail_normal(original)),
          tangent(direction.cross(normal)) {}

    Vec3 tail_direction(const std::vector<Vec3>& points) const {
        const auto i = points.size() - (parallel ? 3 : 2);
        return (points[i + 1] - points[i]).normalized();
    }

    Vec3 tail_normal(const std::vector<Vec3>& points) const {
        const auto i = points.size() - 3;
        const Vec3 a = points[i + 1] - points[i];
        const Vec3 b = points[i + 2] - points[i + 1];
        if (parallel)
            return a.cross(b).normalized();
        // ConRot only constrains one tail direction. The original bend
        // supplies a fixed tangent basis for the two scalar constraints.
        return b.cross(a).normalized();
    }

    template <std::size_t N>
    ClosureVector<N> residual(const std::vector<Vec3>& points) const {
        const Vec3 dr = tail_direction(points) - direction;
        if constexpr (N == 5) {
            const Vec3 dx = (points[points.size() - 2] - anchor) / scale;
            return {dx.x(), dx.y(), dx.z(), dr.dot(normal), dr.dot(tangent)};
        } else {
            const Vec3 dn = tail_normal(points) - normal;
            return {dr.dot(normal), dr.dot(tangent), dn.dot(tangent)};
        }
    }

    template <std::size_t N>
    ClosureMatrix<N> derivatives(const std::vector<Vec3>& points) const {
        ClosureMatrix<N> a{};
        const Vec3 dir = tail_direction(points);
        const std::size_t anchor_index = points.size() - 2;
        for (std::size_t col = 0; col < N; ++col) {
            const std::size_t k = col + 1;
            const Vec3 axis = (points[k + 1] - points[k]).normalized();
            const Vec3 dd = axis.cross(dir);
            if constexpr (N == 5) {
                const Vec3 dx =
                    axis.cross(points[anchor_index] - points[k]) / scale;
                a[0][col] = dx.x();
                a[1][col] = dx.y();
                a[2][col] = dx.z();
                a[3][col] = dd.dot(normal);
                a[4][col] = dd.dot(tangent);
            } else {
                a[0][col] = dd.dot(normal);
                a[1][col] = dd.dot(tangent);
                a[2][col] = axis.cross(tail_normal(points)).dot(tangent);
            }
        }
        return a;
    }

    bool closed(const std::vector<Vec3>& points) const {
        constexpr double tolerance = 1e-10;
        if (!((tail_direction(points) - direction).norm() < tolerance))
            return false;
        if (parallel)
            return (tail_normal(points) - normal).norm() < tolerance;
        return (points[points.size() - 2] - anchor).norm() < tolerance * scale;
    }

    bool parallel;
    Vec3 anchor;
    double scale;
    Vec3 direction;
    Vec3 normal;
    Vec3 tangent;
};

template <std::size_t N>
std::optional<double> close_backbone(const Closure& geometry,
                                     std::vector<Vec3>& points, double driver) {
    apply_torsion(points, 0, driver);
    for (int iteration = 0; iteration < 25; ++iteration) {
        auto correction = geometry.residual<N>(points);
        for (double& value : correction)
            value = -value;
        const auto log_det = solve(geometry.derivatives<N>(points), correction);
        if (!log_det
            || std::any_of(
                correction.begin(), correction.end(), [](double theta) {
                    return !std::isfinite(theta) || std::abs(theta) > 1;
                }))
            return std::nullopt;
        if (geometry.closed(points))
            return log_det;
        for (std::size_t j = 0; j < N; ++j)
            apply_torsion(points, j + 1, correction[j]);
    }
    return std::nullopt;
}

template <std::size_t N>
std::optional<double> reversible_closure(const std::vector<Vec3>& original,
                                         std::vector<Vec3>& current,
                                         double angle) {
    const Closure geometry(original, N == 3);
    ClosureVector<N> zero{};
    const auto old_log_det = solve(geometry.derivatives<N>(original), zero);
    const auto new_log_det = close_backbone<N>(geometry, current, angle);
    if (!old_log_det || !new_log_det)
        return std::nullopt;
    // Newton recovers one local branch. Reject jumps which cannot retrace
    // the same branch. There is no root-count factor for this deterministic
    // branch selector. The conditional torsion measure is 1/|det A|.
    auto reverse = current;
    if (!close_backbone<N>(Closure(current, N == 3), reverse, -angle))
        return std::nullopt;
    for (std::size_t i = 0; i < original.size(); ++i) {
        if (!((original[i] - reverse[i]).norm() < 1e-8 * geometry.scale))
            return std::nullopt;
    }
    return *old_log_det - *new_log_det;
}

} // namespace

PolymerClosureMove::PolymerClosureMove(std::vector<Path> paths,
                                       double max_angle, bool parallel)
    : paths_(std::move(paths)), max_angle_(max_angle), parallel_(parallel) {
    validate_paths(paths_, parallel ? 6 : 8);
    if (!(max_angle > 0 && max_angle <= std::numbers::pi))
        throw std::invalid_argument(
            "Polymer closure requires an angle in (0, pi] radians");
}

ConRotMove::ConRotMove(const std::vector<std::array<TagInt, 8>>& paths,
                       double max_angle)
    : PolymerClosureMove(to_paths(paths), max_angle, false) {}

ParallelRotationMove::ParallelRotationMove(
    const std::vector<std::array<TagInt, 6>>& paths, double max_angle)
    : PolymerClosureMove(to_paths(paths), max_angle, true) {}

std::string PolymerClosureMove::name() const {
    return parallel_ ? "parallel_rotation" : "conrot";
}

std::vector<DefaultBinding> PolymerClosureMove::default_bindings() const {
    return proposal_binding();
}

void PolymerClosureMove::launch(const LaunchContext& ctx) {
    if (ctx.step < 0)
        return;
    require_mc(ctx);
    auto& rng = *ctx.runtime.mc.rng;
    auto& frame = ctx.frame;
    const Topology topology(frame);
    const auto path = topology.bonded_path(paths_[pick(rng, paths_.size())]);

    const std::size_t interior_end = parallel_ ? 3 : 6;
    for (std::size_t i = 2; i < interior_end; ++i) {
        if (topology.neighbors[path[i]].size() != 2)
            throw std::invalid_argument(
                "Polymer closure requires a branch-free moving backbone");
    }
    std::vector<Vec3> original(path.size());
    original[0] = topology.points[path[0]];
    for (std::size_t i = 1; i < path.size(); ++i) {
        const Vec3 delta = mic(frame.box, topology.points[path[i]]
                                              - topology.points[path[i - 1]]);
        if (!(delta.norm() > 0))
            return;
        original[i] = original[i - 1] + delta;
    }

    std::vector<int> fragment;
    std::vector<Vec3> unwrapped(frame.nlocal());
    const std::vector<Cut> cuts{{path[1], path[2]}};
    if (parallel_
        && !unwrap_fragment(topology, frame.box, path[2], original[2], cuts,
                            fragment, unwrapped))
        return;
    if (parallel_
        && std::find(fragment.begin(), fragment.end(), path[0])
               != fragment.end())
        return;

    const double angle =
        std::uniform_real_distribution<double>(-max_angle_, max_angle_)(rng);
    auto rotated = original;
    const auto log_ratio =
        parallel_ ? reversible_closure<3>(original, rotated, angle)
                  : reversible_closure<5>(original, rotated, angle);
    if (!log_ratio)
        return;

    std::vector<int> moving;
    std::vector<Vec3> displacements;
    const auto check_bond = [&](std::size_t i) {
        return same_image(frame.box, rotated[i] - rotated[i - 1]);
    };
    for (std::size_t i = 2; i < interior_end; ++i) {
        if (!check_bond(i))
            return;
        moving.push_back(path[i]);
        displacements.push_back(rotated[i] - original[i]);
    }
    if (!check_bond(interior_end))
        return;
    if (parallel_) {
        // Both tail orientation constraints are satisfied. The component
        // beyond the moving atom is translated as a rigid body.
        const Vec3 translation = rotated[3] - original[3];
        for (const int row : fragment) {
            if (row == path[2])
                continue;
            moving.push_back(row);
            displacements.push_back(translation);
        }
    }
    auto& txn = *ctx.runtime.mc.proposal;
    Displacement(frame, txn, moving, topology.points).write(displacements);
    txn.log_proposal_ratio = *log_ratio;
}

AtomIdentityExchangeMove::AtomIdentityExchangeMove(
    std::vector<std::array<TagInt, 2>> pairs)
    : pairs_(std::move(pairs)) {
    validate_paths(to_paths(pairs_), 2);
}

std::vector<DefaultBinding> AtomIdentityExchangeMove::default_bindings() const {
    return proposal_binding();
}

void AtomIdentityExchangeMove::launch(const LaunchContext& ctx) {
    if (ctx.step < 0)
        return;
    require_mc(ctx);
    auto& frame = ctx.frame;
    auto& txn = *ctx.runtime.mc.proposal;
    const Topology topology(frame);
    const auto& tags = pairs_[pick(*ctx.runtime.mc.rng, pairs_.size())];
    const int a = topology.row_for(tags[0]);
    const int b = topology.row_for(tags[1]);
    if (!topology.neighbors[a].empty() || !topology.neighbors[b].empty())
        throw std::invalid_argument(
            "AtomIdentityExchangeMove requires two unbonded atoms");

    txn.begin(Transaction::Verb::Displace, frame);
    txn.record_touch(frame, a);
    txn.record_touch(frame, b);
    txn.log_proposal_ratio = 0;
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto images = frame.span(kAtom_image);
    std::swap(x[a], x[b]);
    std::swap(y[a], y[b]);
    std::swap(z[a], z[b]);
    for (int d = 0; d < 3; ++d)
        std::swap(images(a, d), images(b, d));
}

CBMCMove::CBMCMove(std::vector<Path> paths, int trials, double bond_sigma,
                   double bias_beta)
    : CBMCMove(std::move(paths), trials, bond_sigma, bias_beta, false) {}

CBMCMove::CBMCMove(std::vector<Path> paths, int trials, double bond_sigma,
                   double bias_beta, bool fixed_endpoint)
    : paths_(std::move(paths)), trials_(trials), bond_sigma_(bond_sigma),
      bias_beta_(bias_beta), fixed_endpoint_(fixed_endpoint) {
    validate_paths(paths_, fixed_endpoint ? 3 : 2);
    if (trials < 1 || !(bond_sigma > 0 && std::isfinite(bond_sigma))
        || !(bias_beta >= 0 && std::isfinite(bias_beta)))
        throw std::invalid_argument("CBMC requires trials >= 1, finite sigma > "
                                    "0 and finite bias_beta >= 0");
}

ExtendedCBMCMove::ExtendedCBMCMove(std::vector<Path> paths, int trials,
                                   double bond_sigma, double bias_beta)
    : CBMCMove(std::move(paths), trials, bond_sigma, bias_beta, true) {}

std::string CBMCMove::name() const {
    return fixed_endpoint_ ? "extended_cbmc" : "cbmc";
}

std::vector<DefaultBinding> CBMCMove::default_bindings() const {
    return proposal_binding();
}

void CBMCMove::launch(const LaunchContext& ctx) {
    if (ctx.step < 0)
        return;
    require_mc(ctx);
    const auto& evaluate = ctx.runtime.mc.evaluate_trial;
    if (!evaluate)
        throw std::logic_error("CBMC requires the MC energy evaluator");

    auto& rng = *ctx.runtime.mc.rng;
    auto& frame = ctx.frame;
    auto& txn = *ctx.runtime.mc.proposal;
    const Topology topology(frame);
    const auto path = topology.bonded_path(paths_[pick(rng, paths_.size())]);
    const std::size_t end = path.size() - (fixed_endpoint_ ? 1 : 0);
    const std::vector<int> moving(path.begin() + 1, path.begin() + end);
    for (std::size_t i = 1; i < end; ++i) {
        const std::size_t degree = i + 1 == path.size() ? 1 : 2;
        if (topology.neighbors[path[i]].size() != degree)
            throw std::invalid_argument(
                "CBMC requires a branch-free regrown segment");
    }

    std::vector<Vec3> old(path.size());
    old[0] = topology.points[path[0]];
    for (std::size_t i = 1; i < path.size(); ++i)
        old[i] = old[i - 1]
                 + mic(frame.box,
                       topology.points[path[i]] - topology.points[path[i - 1]]);

    const Displacement displacement(frame, txn, moving, topology.points);
    std::vector<Vec3> deltas(moving.size());
    const auto install = [&](const std::vector<Vec3>& points) {
        for (std::size_t j = 1; j < end; ++j)
            deltas[j - 1] = points[j] - old[j];
        // A score can resize the ghost halo and invalidate old column views.
        // Rebind once per candidate write under its own resolution window.
        struct RestoreWindow {
            Frame& frame;
            ResolutionWindow saved;
            ~RestoreWindow() { frame.restore_resolution_window(saved); }
        } restore{frame, frame.resolution_window()};
        frame.begin_resolution_window(kResolveBudgetPerLaunch);
        displacement.write(deltas);
        frame.end_resolution_window("CBMC trial coordinates");
    };

    const auto mean = [&](const Vec3& previous, std::size_t i) {
        if (!fixed_endpoint_)
            return previous;
        return previous
               + (old.back() - previous) / static_cast<double>(path.size() - i);
    };
    const auto sigma = [&](std::size_t i) {
        if (!fixed_endpoint_)
            return bond_sigma_;
        const double remaining = static_cast<double>(path.size() - i);
        return bond_sigma_ * std::sqrt((remaining - 1) / remaining);
    };

    std::vector<Vec3> suffix_noise(path.size());
    std::vector<Vec3> trial = old;
    constexpr double invalid = -std::numeric_limits<double>::infinity();
    const auto weight = [&](const std::vector<Vec3>& prefix, std::size_t i,
                            const Vec3& candidate) {
        std::copy_n(prefix.begin(), i, trial.begin());
        trial[i] = candidate;
        if (!same_image(frame.box, trial[i] - trial[i - 1]))
            return invalid;
        for (std::size_t j = i + 1;
             j < end && (bias_beta_ != 0 || fixed_endpoint_); ++j) {
            trial[j] = mean(trial[j - 1], j) + suffix_noise[j];
            if (!same_image(frame.box, trial[j] - trial[j - 1]))
                return invalid;
        }
        if (fixed_endpoint_
            && !same_image(frame.box, old.back() - trial[end - 1]))
            return invalid;
        if (bias_beta_ == 0)
            return 0.0;
        install(trial);
        const auto energy = (*evaluate)();
        if (std::isnan(energy) || energy == invalid)
            throw std::runtime_error("Invalid candidate energy in CBMC");
        const double log_weight = -bias_beta_ * energy;
        if (log_weight == -invalid)
            throw std::runtime_error("CBMC candidate weight overflow");
        return log_weight;
    };

    const auto reject = [&] {
        txn.rollback(frame);
        txn.begin(Transaction::Verb::None, frame);
    };
    std::vector<Vec3> grown = old;
    std::vector<Vec3> candidates(trials_);
    std::vector<double> forward(trials_);
    std::vector<double> reverse(trials_);
    const auto normalize = [](std::vector<double>& weights, double shift) {
        double total = 0;
        for (double& log_w : weights) {
            log_w = std::exp(log_w - shift);
            total += log_w;
        }
        return std::log(total);
    };

    double log_ratio = 0;
    for (std::size_t i = 1; i < end; ++i) {
        // The future beads in a candidate score are provisional. Share
        // independent suffix noise between the forward and reverse pools
        // so both use exactly the same conditional scoring function.
        if (bias_beta_ != 0 || fixed_endpoint_) {
            for (std::size_t j = i + 1; j < end; ++j)
                suffix_noise[j] = gaussian_vector(rng, sigma(j));
        }
        const Vec3 forward_mean = mean(grown[i - 1], i);
        const Vec3 reverse_mean = mean(old[i - 1], i);
        const double width = sigma(i);
        for (int k = 0; k < trials_; ++k) {
            candidates[k] = forward_mean + gaussian_vector(rng, width);
            forward[k] = weight(grown, i, candidates[k]);
        }
        const double maximum =
            *std::max_element(forward.begin(), forward.end());
        const double old_weight = weight(old, i, old[i]);
        if (maximum == invalid || old_weight == invalid) {
            reject();
            return;
        }
        const auto log_weights = forward;
        const double log_forward_sum = normalize(forward, maximum);
        const auto chosen = std::discrete_distribution<int>(forward.begin(),
                                                            forward.end())(rng);
        grown[i] = candidates[chosen];
        reverse[0] = old_weight;
        for (int k = 1; k < trials_; ++k)
            reverse[k] =
                weight(old, i, reverse_mean + gaussian_vector(rng, width));
        const double reverse_max =
            *std::max_element(reverse.begin(), reverse.end());
        const double log_reverse_sum = normalize(reverse, reverse_max);
        const double log_reference_ratio =
            0.5
            * (((grown[i] - forward_mean) / width).norm_squared()
               - ((old[i] - reverse_mean) / width).norm_squared());
        log_ratio += log_reference_ratio + (old_weight - reverse_max)
                     - (log_weights[chosen] - maximum) + log_forward_sum
                     - log_reverse_sum;
    }
    if (!std::isfinite(log_ratio)) {
        reject();
        return;
    }
    install(grown);
    txn.log_proposal_ratio = log_ratio;
}

} // namespace ATV::cpu::mc
