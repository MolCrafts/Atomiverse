/// @file cpu/dynamics/fix_rattle.cpp
/// @brief cpu::md::DistanceConstraints — shared table, clusters, halo packer
///        and the SHAKE/RATTLE Gauss–Seidel stages (LINCS: fix_lincs.cpp).

#include "atomiverse/core/timer.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/dynamics/distance_constraints.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

namespace {
// Scoped regions of the two stages roll up under Modify, like the component.
constexpr int kTimerCategory = static_cast<int>(StepTiming::Modify);

// The communicator a constraint stage must refresh ghosts through, or null.
// Constraint rows are resolved by tag: across ranks a partner may be a ghost
// row (dd->tag_to_local), so the stage re-sends what it reads. On one rank
// the tag index covers owned rows only and the minimum image supplies the
// periodic partner, so no ghost is ever read and the pass is skipped (LAMMPS
// FixRattle guards its own passes with nprocs > 1 for the same reason).
cpu::Comm* halo_comm_(const cpu::LaunchContext& ctx) {
    return cpu::multi_rank_dd(ctx.runtime) ? ctx.runtime.md.comm : nullptr;
}

#ifdef ATV_HAS_MPI
MPI_Datatype tag_mpi_type() {
    if constexpr (sizeof(TagInt) == sizeof(std::int64_t))
        return MPI_INT64_T;
    return MPI_INT32_T;
}

template <typename T>
std::vector<T>
allgatherv_native(const std::vector<T>& local, MPI_Datatype dtype,
                  const std::vector<int>& counts,
                  const std::vector<int>& displacements, MPI_Comm comm) {
    int total = 0;
    for (int count : counts)
        total += count;
    std::vector<T> global(static_cast<std::size_t>(total));
    MPI_Allgatherv(local.data(), static_cast<int>(local.size()), dtype,
                   global.data(), counts.data(), displacements.data(), dtype,
                   comm);
    return global;
}
#endif

} // namespace

DistanceConstraints::DistanceConstraints(double dt, double tolerance,
                                         int max_iter, Algorithm algorithm,
                                         int expansion_order)
    : algorithm_(algorithm),
      velocity_projection_(algorithm != Algorithm::Shake),
      expansion_order_(expansion_order), dt_(dt), tol_(tolerance),
      max_iter_(max_iter) {
    if (!std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument("DistanceConstraints: dt must be positive");
    if (!std::isfinite(tolerance) || tolerance <= 0.0)
        throw std::invalid_argument(
            "DistanceConstraints: tolerance must be positive");
    if (tolerance < storage_tolerance_floor())
        throw std::invalid_argument(
            "DistanceConstraints: tolerance is below the "
            "configured Frame storage precision "
            "floor");
    if (max_iter < 1)
        throw std::invalid_argument(
            "DistanceConstraints: max_iter must be >= 1");
    if (algorithm == Algorithm::Lincs && expansion_order < 1)
        throw std::invalid_argument(
            "DistanceConstraints: the LINCS expansion order must be >= 1");
}

void DistanceConstraints::constrain_bond(int bond_type, double length) {
    if (bond_type < 0 || !std::isfinite(length) || length <= 0.0)
        throw std::invalid_argument(
            "DistanceConstraints: constrained length must be positive");
    for (const auto& rule : bond_rules_)
        if (rule.type == bond_type)
            throw std::invalid_argument(
                "distance constraints: duplicate bond type rule");
    bond_rules_.push_back({bond_type, length});
}

void DistanceConstraints::constrain_angle(int angle_type, double distance_13) {
    if (angle_type < 0 || !std::isfinite(distance_13) || distance_13 <= 0.0)
        throw std::invalid_argument(
            "DistanceConstraints: constrained 1-3 distance must be "
            "positive");
    for (const auto& rule : angle_rules_)
        if (rule.type == angle_type)
            throw std::invalid_argument(
                "distance constraints: duplicate angle type rule");
    angle_rules_.push_back({angle_type, distance_13});
}

std::vector<DistanceConstraints::Spec>
DistanceConstraints::discover_local_(const cpu::Context& ctx,
                                     const cpu::Frame& frame) const {
    std::vector<Spec> out;
    const int n_own = frame.nlocal();
    const auto num_bond = frame.span(kAtom_num_bond);
    const auto bond_type = frame.span(kAtom_bond_type);
    const auto bond_atom = frame.span(kAtom_bond_atom);
    const auto num_angle = frame.span(kAtom_num_angle);
    const auto angle_type = frame.span(kAtom_angle_type);
    const auto angle_atom = frame.span(kAtom_angle_atom);
    const auto tag = frame.span(kAtom_tag);

    // Only OWNED rows are scanned, and that is exactly right: LAMMPS bond ownership
    // gives every bond to its smaller-tag atom, and every angle to its centre atom, so
    // summing the per-rank discoveries over all ranks yields each constraint exactly
    // once — no double counting, nothing missed.
    for (int i = 0; i < n_own; ++i) {
        for (int m = 0; m < num_bond[i]; ++m)
            for (const Rule& r : bond_rules_)
                if (bond_type(i, m) == r.type)
                    out.push_back({tag[i], bond_atom(i, m), r.distance});

        for (int m = 0; m < num_angle[i]; ++m)
            for (const Rule& r : angle_rules_)
                if (angle_type(i, m) == r.type)
                    // Atom i is the CENTRE; the outer two atoms of the triplet are the
                    // 1 and 3. Freezing their separation freezes the angle.
                    out.push_back({angle_atom(i, 2 * m + 0),
                                   angle_atom(i, 2 * m + 1), r.distance});
    }
    return out;
}

int DistanceConstraints::n_constraints(const cpu::Context& ctx,
                                       const cpu::Frame& frame) const {
    int n = static_cast<int>(discover_local_(ctx, frame).size());
#ifdef ATV_HAS_MPI
    // Ndof is GLOBAL, so the count must be too. Each constraint is discovered on
    // exactly one rank (the one owning its bond/angle), so a plain sum is correct.
    const auto* dd = ctx.md.dd;
    if (ctx.mpi_size > 1 && dd != nullptr && dd->cart_comm != MPI_COMM_NULL) {
        int local = n;
        MPI_Allreduce(&local, &n, 1, MPI_INT, MPI_SUM, dd->cart_comm);
    }
#endif
    return n;
}

void DistanceConstraints::count_constraints_by_type(
    const cpu::Context& /*ctx*/, const cpu::Frame& frame,
    std::vector<BigInt>& by_type) const {
    const int n_own = frame.nlocal();
    const auto num_bond = frame.span(kAtom_num_bond);
    const auto bond_type = frame.span(kAtom_bond_type);
    const auto num_angle = frame.span(kAtom_num_angle);
    const auto angle_type = frame.span(kAtom_angle_type);
    const auto type = frame.span(kAtom_type);
    // The same owned-row scan as `discover_local_`, so the per-type total
    // equals `n_constraints` once the ranks are summed.
    for (int i = 0; i < n_own; ++i) {
        BigInt n = 0;
        for (int m = 0; m < num_bond[i]; ++m)
            for (const Rule& r : bond_rules_)
                n += bond_type(i, m) == r.type;
        for (int m = 0; m < num_angle[i]; ++m)
            for (const Rule& r : angle_rules_)
                n += angle_type(i, m) == r.type;
        if (n > 0)
            cpu::tally_constraints(by_type, static_cast<int>(type[i]), n);
    }
}

void DistanceConstraints::build(const cpu::BuildContext& ctx) {
    const cpu::Context& rt = ctx.runtime;
    const cpu::Frame& frame = ctx.frame;
    const int n_own = frame.nlocal();
    const auto tag = frame.span(kAtom_tag);
    const auto mass = frame.span(kAtom_mass);

    // ── The constraint table must be GLOBAL, not this rank's share ──────
    //
    // A water's two bonds and its angle all live on the oxygen (smallest tag / centre
    // atom). So the rank owning the O discovers the cluster — and a rank that owns
    // only an H discovers NOTHING, yet it is the rank responsible for correcting that
    // H. Each rank must therefore know every cluster it holds any atom of, which means
    // the table has to be shared. It is tiny (three doubles per constraint) and static,
    // so it is gathered once, here.
    std::vector<Spec> local = discover_local_(rt, frame);

    // Masses, likewise. Ghost rows carry x/y/z/tag/charge only — a ghost's `mass` is
    // never communicated, so the inverse masses have to be baked into the table while
    // we can still see every atom on its owner.
    std::unordered_map<TagInt, double> mass_of;
    for (int i = 0; i < n_own; ++i)
        mass_of[tag[i]] = mass[i];

    std::vector<Spec> global = local;

#ifdef ATV_HAS_MPI
    const auto* dd = rt.md.dd;
    if (rt.mpi_size > 1 && dd != nullptr && dd->cart_comm != MPI_COMM_NULL) {
        const int nranks = rt.mpi_size;

        // Gather tags in their configured integer dtype.  In particular, a
        // bigbig tag above 2^53 must never pass through floating point.
        std::vector<TagInt> local_ti;
        std::vector<TagInt> local_tj;
        std::vector<double> local_distance;
        local_ti.reserve(local.size());
        local_tj.reserve(local.size());
        local_distance.reserve(local.size());
        for (const Spec& s : local) {
            local_ti.push_back(s.ti);
            local_tj.push_back(s.tj);
            local_distance.push_back(s.distance);
        }
        int my_n = static_cast<int>(local.size());
        std::vector<int> counts(static_cast<std::size_t>(nranks), 0);
        MPI_Allgather(&my_n, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      dd->cart_comm);
        std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
        int total = 0;
        for (int r = 0; r < nranks; ++r) {
            displs[static_cast<std::size_t>(r)] = total;
            total += counts[static_cast<std::size_t>(r)];
        }
        const auto all_ti = allgatherv_native(local_ti, tag_mpi_type(), counts,
                                              displs, dd->cart_comm);
        const auto all_tj = allgatherv_native(local_tj, tag_mpi_type(), counts,
                                              displs, dd->cart_comm);
        const auto all_distance = allgatherv_native(
            local_distance, MPI_DOUBLE, counts, displs, dd->cart_comm);

        global.clear();
        global.reserve(static_cast<std::size_t>(total));
        for (int k = 0; k < total; ++k)
            global.push_back({all_ti[static_cast<std::size_t>(k)],
                              all_tj[static_cast<std::size_t>(k)],
                              all_distance[static_cast<std::size_t>(k)]});

        // Same for (tag, mass), still as two native typed arrays.
        std::vector<TagInt> mass_tags;
        std::vector<double> masses;
        mass_tags.reserve(static_cast<std::size_t>(n_own));
        masses.reserve(static_cast<std::size_t>(n_own));
        for (int i = 0; i < n_own; ++i) {
            mass_tags.push_back(tag[i]);
            masses.push_back(mass[i]);
        }
        int my_m = n_own;
        MPI_Allgather(&my_m, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      dd->cart_comm);
        total = 0;
        for (int r = 0; r < nranks; ++r) {
            displs[static_cast<std::size_t>(r)] = total;
            total += counts[static_cast<std::size_t>(r)];
        }
        const auto all_mass_tags = allgatherv_native(
            mass_tags, tag_mpi_type(), counts, displs, dd->cart_comm);
        const auto all_masses = allgatherv_native(masses, MPI_DOUBLE, counts,
                                                  displs, dd->cart_comm);
        mass_of.clear();
        for (int k = 0; k < total; ++k)
            mass_of[all_mass_tags[static_cast<std::size_t>(k)]] =
                all_masses[static_cast<std::size_t>(k)];
    }
#endif

    if (global.empty())
        throw std::runtime_error(
            "DistanceConstraints: no constraints matched the topology. Did you "
            "call "
            "constrain_bond()/constrain_angle() with the right type indices?");

    std::vector<Constraint> table;
    table.reserve(global.size());
    for (const Spec& s : global) {
        const auto mi = mass_of.find(s.ti);
        const auto mj = mass_of.find(s.tj);
        if (mi == mass_of.end() || mj == mass_of.end())
            throw std::runtime_error(
                "DistanceConstraints: a constrained atom has no mass in the "
                "global table "
                "— the "
                "topology references a tag that no rank owns");
        Constraint c;
        c.ti = s.ti;
        c.tj = s.tj;
        c.d2 = s.distance * s.distance;
        c.inv_mi = 1.0 / mi->second;
        c.inv_mj = 1.0 / mj->second;
        table.push_back(c);
    }

    // ── Clusters: connected components over shared atoms (union-find) ────
    // Every rank sees the same global table in the same order, so every rank
    // derives the same clusters with the same ids.
    std::unordered_map<TagInt, int> node;
    std::vector<int> parent;
    auto node_of = [&](TagInt t) {
        const auto [it, fresh] =
            node.emplace(t, static_cast<int>(parent.size()));
        if (fresh)
            parent.push_back(it->second);
        return it->second;
    };
    auto find = [&](int v) {
        while (parent[static_cast<std::size_t>(v)] != v) {
            parent[static_cast<std::size_t>(v)] =
                parent[static_cast<std::size_t>(
                    parent[static_cast<std::size_t>(v)])];
            v = parent[static_cast<std::size_t>(v)];
        }
        return v;
    };
    for (const Constraint& c : table) {
        const int a = find(node_of(c.ti));
        const int b = find(node_of(c.tj));
        if (a != b)
            parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
    }
    std::unordered_map<int, int> cluster_id; // root -> dense id, table order
    std::vector<int> cluster_of_constraint(table.size());
    for (std::size_t k = 0; k < table.size(); ++k) {
        const int root = find(node_of(table[k].ti));
        const auto it =
            cluster_id.emplace(root, static_cast<int>(cluster_id.size())).first;
        cluster_of_constraint[k] = it->second;
    }
    const int n_clusters = static_cast<int>(cluster_id.size());
    cluster_offset_.assign(static_cast<std::size_t>(n_clusters) + 1, 0);
    for (int c : cluster_of_constraint)
        ++cluster_offset_[static_cast<std::size_t>(c) + 1];
    for (int c = 0; c < n_clusters; ++c)
        cluster_offset_[static_cast<std::size_t>(c) + 1] +=
            cluster_offset_[static_cast<std::size_t>(c)];
    constraints_.assign(table.size(), Constraint{});
    std::vector<int> fill(cluster_offset_.begin(), cluster_offset_.end() - 1);
    for (std::size_t k = 0; k < table.size(); ++k)
        constraints_[static_cast<std::size_t>(
            fill[static_cast<std::size_t>(cluster_of_constraint[k])]++)] =
            table[k];
    cluster_of_tag_.clear();
    cluster_of_tag_.reserve(node.size());
    for (const auto& [t, v] : node)
        cluster_of_tag_.emplace(t, cluster_id.at(find(v)));
    cluster_seen_.assign(static_cast<std::size_t>(n_clusters), 0);
    rows_cache_.clear();
}

void DistanceConstraints::refresh_local_(const cpu::LaunchContext& ctx) {
    if (!rows_cache_.stale(ctx))
        return;
    const cpu::Frame& f = ctx.frame;
    const auto* dd = ctx.runtime.md.dd;
    const bool multi = cpu::multi_rank_dd(ctx.runtime);
    const auto tag = f.span(kAtom_tag);
    auto row_of = [&](TagInt t) -> int {
        if (multi) {
            const auto it = dd->tag_to_local.find(static_cast<std::int64_t>(t));
            return it == dd->tag_to_local.end() ? -1 : it->second;
        }
        const auto& index = f.meta.get(kTagIndex);
        const auto it = index.find(t);
        return it == index.end() ? -1 : it->second;
    };

    // Clusters with an OWNED atom — the rows this rank is responsible for.
    // A cluster living only in ghost rows is solved by its owners; its ghost
    // copies here are refreshed by the next halo pass.
    std::vector<int> clusters;
    const int n_own = cpu::owned_count(ctx);
    for (int i = 0; i < n_own; ++i) {
        const auto it = cluster_of_tag_.find(tag[i]);
        if (it == cluster_of_tag_.end())
            continue;
        char& seen = cluster_seen_[static_cast<std::size_t>(it->second)];
        if (!seen) {
            seen = 1;
            clusters.push_back(it->second);
        }
    }
    std::sort(clusters.begin(), clusters.end());

    local_offset_.assign(1, 0);
    local_i_.clear();
    local_j_.clear();
    local_d2_.clear();
    local_imi_.clear();
    local_imj_.clear();
    for (int c : clusters) {
        cluster_seen_[static_cast<std::size_t>(c)] = 0;
        for (int k = cluster_offset_[static_cast<std::size_t>(c)];
             k < cluster_offset_[static_cast<std::size_t>(c) + 1]; ++k) {
            const Constraint& con = constraints_[static_cast<std::size_t>(k)];
            const int i = row_of(con.ti);
            const int j = row_of(con.tj);
            // One atom here and the other nowhere: the halo is too thin to
            // hold the whole cluster, so this rank cannot solve it. A dropped
            // constraint is a molecule that is not rigid: throw, never
            // approximate.
            if (i < 0 || j < 0)
                throw std::runtime_error(
                    "DistanceConstraints: a constrained pair spans ranks but "
                    "its "
                    "partner is not even a ghost here. The halo must cover "
                    "the whole constrained cluster — widen the cutoff/skin, "
                    "use fewer ranks, or use a bigger box.");
            local_i_.push_back(i);
            local_j_.push_back(j);
            local_d2_.push_back(con.d2);
            local_imi_.push_back(con.inv_mi);
            local_imj_.push_back(con.inv_mj);
        }
        local_offset_.push_back(static_cast<int>(local_i_.size()));
    }
    const std::size_t n = local_i_.size();
    ref_sx_.assign(n, 0.0);
    ref_sy_.assign(n, 0.0);
    ref_sz_.assign(n, 0.0);
    if (algorithm_ == Algorithm::Lincs)
        build_lincs_coupling_();
    rows_cache_.mark_built(ctx);
}

int DistanceConstraints::pack_forward_comm(int n, const int* list, double* buf,
                                           int pbc_flag, const int* pbc) {
    if (!frame_ || n <= 0)
        return 0;
    // Columns were resolved once in launch(); a forward pass calls this per
    // swap, and per-swap lookups would count against the resolution budget.
    int m = 0;
    if (comm_mode_ == CommMode::XSHAKE || comm_mode_ == CommMode::PREDICTED) {
        // The image translation is the lattice combination H·pbc, which
        // has off-axis components in a tilted cell.
        double dx = 0.0, dy = 0.0, dz = 0.0;
        if (pbc_flag != 0) {
            const Vec3 shift = frame_->box.to_cartesian(
                {static_cast<double>(pbc[0]), static_cast<double>(pbc[1]),
                 static_cast<double>(pbc[2])});
            dx = shift[0];
            dy = shift[1];
            dz = shift[2];
        }
        if (comm_mode_ == CommMode::PREDICTED) {
            // LAMMPS FixShake::pack_forward_comm: the owner's xshake.
            for (int i = 0; i < n; ++i) {
                const auto& p =
                    shake_predicted_[static_cast<std::size_t>(list[i])];
                buf[m++] = p[0] + dx;
                buf[m++] = p[1] + dy;
                buf[m++] = p[2] + dz;
            }
            return m;
        }
        for (int i = 0; i < n; ++i) {
            const int j = list[i];
            buf[m++] = static_cast<double>(col_[0][j]) + dx;
            buf[m++] = static_cast<double>(col_[1][j]) + dy;
            buf[m++] = static_cast<double>(col_[2][j]) + dz;
        }
        return m;
    }
    if (comm_mode_ == CommMode::F) {
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < 3; ++d)
                buf[m++] = force_col_[d][list[i]];
        return m;
    }
    const int base = 3;
    for (int i = 0; i < n; ++i) {
        const int j = list[i];
        buf[m++] = static_cast<double>(col_[base][j]);
        buf[m++] = static_cast<double>(col_[base + 1][j]);
        buf[m++] = static_cast<double>(col_[base + 2][j]);
    }
    return m;
}

void DistanceConstraints::unpack_forward_comm(int n, int first,
                                              const double* buf) {
    if (!frame_ || n <= 0)
        return;
    if (comm_mode_ == CommMode::F) {
        int m = 0;
        for (int i = first; i < first + n; ++i)
            for (int d = 0; d < 3; ++d)
                force_col_[d][i] = buf[m++];
        return;
    }
    if (comm_mode_ == CommMode::PREDICTED) {
        int m = 0;
        for (int i = first; i < first + n; ++i)
            for (int d = 0; d < 3; ++d)
                shake_predicted_[static_cast<std::size_t>(i)]
                                [static_cast<std::size_t>(d)] = buf[m++];
        return;
    }
    const int base = comm_mode_ == CommMode::XSHAKE ? 0 : 3;
    int m = 0;
    for (int i = first; i < first + n; ++i) {
        col_[base][i] = static_cast<StorageReal>(buf[m++]);
        col_[base + 1][i] = static_cast<StorageReal>(buf[m++]);
        col_[base + 2][i] = static_cast<StorageReal>(buf[m++]);
    }
}

void DistanceConstraints::snapshot_(const cpu::LaunchContext& ctx) {
    const cpu::Frame& f = ctx.frame;
    const auto x = f.span(kAtom_x);
    const auto y = f.span(kAtom_y);
    const auto z = f.span(kAtom_z);
    const cpu::Box& box = f.box;
    const int n = static_cast<int>(local_i_.size());
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) {
        const int i = local_i_[static_cast<std::size_t>(k)];
        const int j = local_j_[static_cast<std::size_t>(k)];
        double sx = x[i] - x[j], sy = y[i] - y[j], sz = z[i] - z[j];
        box.min_image(sx, sy, sz);
        ref_sx_[static_cast<std::size_t>(k)] = sx;
        ref_sy_[static_cast<std::size_t>(k)] = sy;
        ref_sz_[static_cast<std::size_t>(k)] = sz;
    }
}

void DistanceConstraints::solve_positions_(const cpu::LaunchContext& ctx) {
    cpu::Frame& f = ctx.frame;
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    const cpu::Box& box = f.box;
    const double inv_dt = 1.0 / dt_;
    const double tol = tol_;
    const int max_iter = max_iter_;
    const int n_clusters = static_cast<int>(local_offset_.size()) - 1;
    const double virial_factor =
        (velocity_projection_ ? 2.0 : 1.0)
        / ctx.runtime.units.velocity_impulse(dt_ * dt_);
    const int n_owned = f.nlocal();
    double w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0;
    int most_iters = 0;
    double failed_residual = 0.0;

#pragma omp parallel for schedule(dynamic, 64)                                 \
    reduction(max : most_iters, failed_residual)                               \
    reduction(+ : w0, w1, w2, w3, w4, w5)
    for (int c = 0; c < n_clusters; ++c) {
        const int k0 = local_offset_[static_cast<std::size_t>(c)];
        const int k1 = local_offset_[static_cast<std::size_t>(c) + 1];
        int iter = 0;
        double worst = 0.0;
        for (; iter < max_iter; ++iter) {
            worst = 0.0;
            for (int k = k0; k < k1; ++k) {
                const std::size_t q = static_cast<std::size_t>(k);
                const int i = local_i_[q], j = local_j_[q];
                double rx = x[i] - x[j], ry = y[i] - y[j], rz = z[i] - z[j];
                box.min_image(rx, ry, rz);
                const double sigma = rx * rx + ry * ry + rz * rz - local_d2_[q];
                const double err = std::fabs(sigma) / local_d2_[q];
                worst = std::max(worst, err);
                if (err <= tol)
                    continue;
                const double rs =
                    rx * ref_sx_[q] + ry * ref_sy_[q] + rz * ref_sz_[q];
                const double denom = 2.0 * (local_imi_[q] + local_imj_[q]) * rs;
                if (std::fabs(denom) < 1e-12)
                    continue;
                const double g = -sigma / denom;
                const double owner_weight =
                    0.5 * (int(i < n_owned) + int(j < n_owned));
                const double lambda = owner_weight * virial_factor * g;
                const double sx = ref_sx_[q], sy = ref_sy_[q], sz = ref_sz_[q];
                w0 += lambda * sx * sx;
                w1 += lambda * sy * sy;
                w2 += lambda * sz * sz;
                w3 += lambda * sx * sy;
                w4 += lambda * sx * sz;
                w5 += lambda * sy * sz;
                const double gi = g * local_imi_[q], gj = g * local_imj_[q];
                x[i] += gi * ref_sx_[q];
                y[i] += gi * ref_sy_[q];
                z[i] += gi * ref_sz_[q];
                x[j] -= gj * ref_sx_[q];
                y[j] -= gj * ref_sy_[q];
                z[j] -= gj * ref_sz_[q];
                if (velocity_projection_) {
                    // The constraint force also changed the momentum.
                    vx[i] += gi * ref_sx_[q] * inv_dt;
                    vy[i] += gi * ref_sy_[q] * inv_dt;
                    vz[i] += gi * ref_sz_[q] * inv_dt;
                    vx[j] -= gj * ref_sx_[q] * inv_dt;
                    vy[j] -= gj * ref_sy_[q] * inv_dt;
                    vz[j] -= gj * ref_sz_[q] * inv_dt;
                }
            }
            if (worst <= tol)
                break;
        }
        most_iters = std::max(most_iters, iter);
        if (iter >= max_iter)
            failed_residual = std::max(failed_residual, worst);
    }
    if (failed_residual > 0.0)
        throw std::runtime_error(
            "DistanceConstraints: the position stage did not converge in "
            + std::to_string(max_iter_) + " iterations (last relative residual "
            + std::to_string(failed_residual)
            + "). An unconverged constraint is a corrupted molecule, not an "
              "approximation.");
    constraint_virial_ = {w0, w1, w2, w3, w4, w5};
    last_pos_iters_ = most_iters;
}

void DistanceConstraints::solve_velocities_(const cpu::LaunchContext& ctx) {
    cpu::Frame& f = ctx.frame;
    const auto x = f.span(kAtom_x);
    const auto y = f.span(kAtom_y);
    const auto z = f.span(kAtom_z);
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    const cpu::Box& box = f.box;
    const double tol = tol_;
    const int max_iter = max_iter_;
    const int n_clusters = static_cast<int>(local_offset_.size()) - 1;
    int most_iters = 0;
    double failed_residual = 0.0;

#pragma omp parallel for schedule(dynamic, 64)                                 \
    reduction(max : most_iters, failed_residual)
    for (int c = 0; c < n_clusters; ++c) {
        const int k0 = local_offset_[static_cast<std::size_t>(c)];
        const int k1 = local_offset_[static_cast<std::size_t>(c) + 1];
        int iter = 0;
        double worst = 0.0;
        for (; iter < max_iter; ++iter) {
            worst = 0.0;
            for (int k = k0; k < k1; ++k) {
                const std::size_t q = static_cast<std::size_t>(k);
                const int i = local_i_[q], j = local_j_[q];
                double rx = x[i] - x[j], ry = y[i] - y[j], rz = z[i] - z[j];
                box.min_image(rx, ry, rz);
                const double rv = rx * (vx[i] - vx[j]) + ry * (vy[i] - vy[j])
                                  + rz * (vz[i] - vz[j]);
                const double err = std::fabs(rv) / local_d2_[q];
                worst = std::max(worst, err);
                if (err <= tol)
                    continue;
                const double k_lag =
                    -rv / ((local_imi_[q] + local_imj_[q]) * local_d2_[q]);
                const double ki = k_lag * local_imi_[q];
                const double kj = k_lag * local_imj_[q];
                vx[i] += ki * rx;
                vy[i] += ki * ry;
                vz[i] += ki * rz;
                vx[j] -= kj * rx;
                vy[j] -= kj * ry;
                vz[j] -= kj * rz;
            }
            if (worst <= tol)
                break;
        }
        most_iters = std::max(most_iters, iter);
        if (iter >= max_iter)
            failed_residual = std::max(failed_residual, worst);
    }
    if (failed_residual > 0.0)
        throw std::runtime_error(
            "DistanceConstraints: the velocity stage did not converge in "
            + std::to_string(max_iter_) + " iterations (last relative residual "
            + std::to_string(failed_residual) + ").");
    last_vel_iters_ = most_iters;
}

void DistanceConstraints::initial_virial_(const cpu::LaunchContext& ctx) {
    // At run(0) there is no position impulse yet. Solve the differentiated
    // constraints r.a_rel + v_rel² = 0 without changing x, v or the force.
    // This supplies the step-zero barostat with the reaction virial too.
    const int n = ctx.frame.n_total(), own = ctx.frame.nlocal();
    std::vector<std::array<double, 3>> acceleration(n);
    std::vector<double> lambda(local_i_.size(), 0.0);
    for (std::size_t k = 0; k < local_i_.size(); ++k) {
        for (auto [i, im] : {std::pair{local_i_[k], local_imi_[k]},
                             std::pair{local_j_[k], local_imj_[k]}})
            for (int d = 0; d < 3; ++d)
                acceleration[i][d] = force_col_[d][i] * im;
    }
    constraint_virial_.fill(0.0);
    for (std::size_t c = 0; c + 1 < local_offset_.size(); ++c) {
        const int begin = local_offset_[c], end = local_offset_[c + 1];
        int iteration = 0;
        for (; iteration < max_iter_; ++iteration) {
            double worst = 0;
            for (int k = begin; k < end; ++k) {
                const int i = local_i_[k], j = local_j_[k];
                double r[3] = {col_[0][i] - col_[0][j], col_[1][i] - col_[1][j],
                               col_[2][i] - col_[2][j]};
                ctx.frame.box.min_image(r[0], r[1], r[2]);
                double rv2 = 0, ra = 0, r2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const double dv = col_[3 + d][i] - col_[3 + d][j];
                    rv2 += dv * dv;
                    ra += r[d] * (acceleration[i][d] - acceleration[j][d]);
                    r2 += r[d] * r[d];
                }
                if (!(r2 > 0))
                    throw std::runtime_error(
                        "distance constraints: coincident constrained atoms");
                const double target =
                    ctx.runtime.units.velocity_impulse.inverse(rv2);
                const double residual = ra + target;
                worst = std::max(worst, std::abs(residual)
                                            / std::max(1.0, std::abs(target)));
                const double dl =
                    -residual / ((local_imi_[k] + local_imj_[k]) * r2);
                lambda[k] += dl;
                for (int d = 0; d < 3; ++d) {
                    acceleration[i][d] += dl * local_imi_[k] * r[d];
                    acceleration[j][d] -= dl * local_imj_[k] * r[d];
                }
            }
            if (worst <= tol_)
                break;
        }
        if (iteration == max_iter_)
            throw std::runtime_error("distance constraints: initial reaction "
                                     "solve did not converge");
        for (int k = begin; k < end; ++k) {
            const int i = local_i_[k], j = local_j_[k];
            double x = col_[0][i] - col_[0][j], y = col_[1][i] - col_[1][j],
                   z = col_[2][i] - col_[2][j];
            ctx.frame.box.min_image(x, y, z);
            const double l = lambda[k] * 0.5 * (int(i < own) + int(j < own));
            const double w[6] = {x * x, y * y, z * z, x * y, x * z, y * z};
            for (int d = 0; d < 6; ++d)
                constraint_virial_[d] += l * w[d];
        }
    }
}

void DistanceConstraints::solve_shake_forces_(const cpu::LaunchContext& ctx,
                                              bool initial) {
    auto& f = ctx.frame;
    const ColKey<StorageReal> keys[6] = {kAtom_x,  kAtom_y,  kAtom_z,
                                         kAtom_vx, kAtom_vy, kAtom_vz};
    for (int d = 0; d < 6; ++d)
        col_[d] = f.span(keys[d]).data();
    force_col_[0] = f.span(kAtom_fx).data();
    force_col_[1] = f.span(kAtom_fy).data();
    force_col_[2] = f.span(kAtom_fz).data();
    refresh_local_(ctx);
    // Reference bond vectors from the current positions: ghosts hold the
    // x the force pass was just evaluated with.
    snapshot_(ctx);
    // At startup v is a full-step velocity; during post_force it is the
    // half-step velocity. Predict the next drift, as LAMMPS fix shake does.
    const double dtfsq =
        (initial ? 0.5 : 1.0) * ctx.runtime.units.velocity_impulse(dt_ * dt_);
    // Members, so a step reuses the capacity instead of allocating. Only
    // the rows of constrained atoms are written and read.
    auto& predicted = shake_predicted_;
    auto& lambda = shake_lambda_;
    predicted.resize(static_cast<std::size_t>(f.n_total()));
    lambda.assign(local_i_.size(), 0.0);
    // LAMMPS FixShake::unconstrained_update + forward_comm: each owner
    // predicts its own atoms and one pass carries the result to the ghosts.
    // A ghost's v and f are never read, so they are not communicated.
    const int n_own = f.nlocal();
    for (std::size_t k = 0; k < local_i_.size(); ++k)
        for (auto [i, im] : {std::pair{local_i_[k], local_imi_[k]},
                             std::pair{local_j_[k], local_imj_[k]}})
            if (i < n_own)
                for (int d = 0; d < 3; ++d)
                    predicted[i][d] = col_[d][i] + dt_ * col_[3 + d][i]
                                      + dtfsq * im * force_col_[d][i];
    if (cpu::Comm* comm = halo_comm_(ctx)) {
        ATV_PROFILE_SCOPE(ctx, "constraint/halo", ctx.stage, kTimerCategory);
        comm_mode_ = CommMode::PREDICTED;
        comm->forward_comm(*this);
    }
    ATV_PROFILE_SCOPE(ctx, "constraint/solve", ctx.stage, kTimerCategory);
    for (std::size_t c = 0; c + 1 < local_offset_.size(); ++c) {
        const int begin = local_offset_[c], end = local_offset_[c + 1];
        int iteration = 0;
        for (; iteration < max_iter_; ++iteration) {
            double worst = 0;
            for (int k = begin; k < end; ++k) {
                const int i = local_i_[k], j = local_j_[k];
                double r[3] = {predicted[i][0] - predicted[j][0],
                               predicted[i][1] - predicted[j][1],
                               predicted[i][2] - predicted[j][2]};
                f.box.min_image(r[0], r[1], r[2]);
                const double sigma =
                    r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - local_d2_[k];
                const double error = std::abs(sigma) / local_d2_[k];
                worst = std::max(worst, error);
                if (error <= tol_)
                    continue;
                const double ref[3] = {ref_sx_[k], ref_sy_[k], ref_sz_[k]};
                const double dot =
                    r[0] * ref[0] + r[1] * ref[1] + r[2] * ref[2];
                const double denom = 2 * (local_imi_[k] + local_imj_[k]) * dot;
                if (std::abs(denom) < 1e-12)
                    continue;
                const double g = -sigma / denom;
                lambda[k] += g;
                for (int d = 0; d < 3; ++d) {
                    predicted[i][d] += g * local_imi_[k] * ref[d];
                    predicted[j][d] -= g * local_imj_[k] * ref[d];
                }
            }
            if (worst <= tol_)
                break;
        }
        if (iteration == max_iter_)
            throw std::runtime_error(
                "SHAKE: next-position force solve did not converge");
    }
    constraint_virial_.fill(0);
    const int own = f.nlocal();
    for (std::size_t k = 0; k < local_i_.size(); ++k) {
        const int i = local_i_[k], j = local_j_[k];
        const double l = lambda[k] / dtfsq;
        const double r[3] = {ref_sx_[k], ref_sy_[k], ref_sz_[k]};
        for (int d = 0; d < 3; ++d) {
            if (i < own)
                force_col_[d][i] += l * r[d];
            if (j < own)
                force_col_[d][j] -= l * r[d];
        }
        const double weight = 0.5 * (int(i < own) + int(j < own));
        const double w[6] = {r[0] * r[0], r[1] * r[1], r[2] * r[2],
                             r[0] * r[1], r[0] * r[2], r[1] * r[2]};
        for (int d = 0; d < 6; ++d)
            constraint_virial_[d] += weight * l * w[d];
    }
}

void DistanceConstraints::initialize_forces(const cpu::LaunchContext& ctx) {
    if (!ctx.at(ATV::md::Phase::post_force))
        return;
    if (!velocity_projection_) {
        frame_ = &ctx.frame;
        solve_shake_forces_(ctx, true);
        if (ctx.stress_requested) {
            auto& stress = ctx.frame.meta.get(kStress);
            for (int d = 0; d < 6; ++d)
                stress[d] += constraint_virial_[d];
        }
        return;
    }
    if (!ctx.stress_requested)
        return;
    frame_ = &ctx.frame;
    const ColKey<StorageReal> keys[6] = {kAtom_x,  kAtom_y,  kAtom_z,
                                         kAtom_vx, kAtom_vy, kAtom_vz};
    for (int d = 0; d < 6; ++d)
        col_[d] = ctx.frame.span(keys[d]).data();
    force_col_[0] = ctx.frame.span(kAtom_fx).data();
    force_col_[1] = ctx.frame.span(kAtom_fy).data();
    force_col_[2] = ctx.frame.span(kAtom_fz).data();
    refresh_local_(ctx);
    if (cpu::Comm* comm = ctx.runtime.md.comm) {
        for (auto mode : {CommMode::XSHAKE, CommMode::VP, CommMode::F}) {
            comm_mode_ = mode;
            comm->forward_comm(*this);
        }
    }
    initial_virial_(ctx);
    auto& stress = ctx.frame.meta.get(kStress);
    for (int d = 0; d < 6; ++d)
        stress[d] += constraint_virial_[d];
}

void DistanceConstraints::launch(const cpu::LaunchContext& ctx) {
    frame_ = &ctx.frame;
    {
        const ColKey<StorageReal> keys[6] = {kAtom_x,  kAtom_y,  kAtom_z,
                                             kAtom_vx, kAtom_vy, kAtom_vz};
        for (int k = 0; k < 6; ++k)
            col_[k] = ctx.frame.span(keys[k]).data();
    }

    // ── initial_integrate @ -500: reference bond vectors ─────────────────
    // Taken BEFORE any integrator moves anything. SHAKE/RATTLE corrects along
    // the reference bond vector, and VelocityVerlet fuses kick+drift and then
    // wraps, so by post_integrate the pre-move positions are unrecoverable.
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        refresh_local_(ctx);
        snapshot_(ctx);
        constraint_virial_.fill(0.0);
        return;
    }

    // Every rank holding an owned atom of a cluster solves that cluster
    // REDUNDANTLY, from identical inputs (XSHAKE forward_comm has just re-sent
    // the ghosts' x), cluster by cluster, so it takes identical iterations and
    // arrives at identical corrections. Ghost-row writes are local scratch,
    // discarded at the next refresh, while each owner ends up with exactly the
    // correction its owning rank computed — LAMMPS's replicated-cluster route.
    if (ctx.at(ATV::md::Phase::post_integrate)) {
        comm_mode_ = CommMode::XSHAKE;
        if (cpu::Comm* comm = halo_comm_(ctx)) {
            ATV_PROFILE_SCOPE(ctx, "constraint/halo", ctx.stage,
                              kTimerCategory);
            comm->forward_comm(*this);
        }
        ATV_PROFILE_SCOPE(ctx, "constraint/solve", ctx.stage, kTimerCategory);
        if (algorithm_ == Algorithm::Lincs)
            solve_lincs_positions_(ctx);
        else
            solve_positions_(ctx);
        return;
    }

    if (ctx.at(ATV::md::Phase::post_force)) {
        if (!velocity_projection_)
            solve_shake_forces_(ctx, false);
        if (ctx.stress_requested) {
            auto& stress = ctx.frame.meta.get(kStress);
            for (int a = 0; a < 6; ++a)
                stress[a] += constraint_virial_[a];
        }
        return;
    }

    // ── final_integrate @ +200: velocity stage ───────────────────────────
    //   d(sigma)/dt = 2 (r_i - r_j).(v_i - v_j) = 0
    // `CommBrick` runs at pre_neighbor, between the position stage and this
    // one, and a full rebuild reshuffles rows — refresh_local_ sees that
    // through the row-identity cache and re-resolves before solving.
    if (ctx.at(ATV::md::Phase::final_integrate)) {
        refresh_local_(ctx);
        comm_mode_ = CommMode::VP;
        if (cpu::Comm* comm = halo_comm_(ctx)) {
            ATV_PROFILE_SCOPE(ctx, "constraint/halo", ctx.stage,
                              kTimerCategory);
            comm->forward_comm(*this);
        }
        ATV_PROFILE_SCOPE(ctx, "constraint/solve", ctx.stage, kTimerCategory);
        if (algorithm_ == Algorithm::Lincs)
            solve_lincs_velocities_(ctx);
        else
            solve_velocities_(ctx);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
