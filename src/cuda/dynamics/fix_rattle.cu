/// @file src/cuda/dynamics/fix_rattle.cu
/// @brief SHAKE/RATTLE clusters on the GPU — the table, clustering and
///        Gauss–Seidel sweeps of `src/cpu/dynamics/fix_rattle.cpp`, one
///        thread per cluster.

#include "atomiverse/cuda/dynamics/fix_rattle.cuh"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/frame/precision.cuh"

namespace ATV {
namespace cuda {
namespace md {

struct FixRattle::State {
    int n_constraints = 0;
    int n_clusters = 0;
    int* offset = nullptr;  ///< CSR over constraints, n_clusters + 1.
    int2* pair = nullptr;   ///< (i, j) rows per constraint.
    mixed_t* d2 = nullptr;  ///< Target squared distance [Å²].
    mixed_t* imi = nullptr; ///< 1/m_i.
    mixed_t* imj = nullptr; ///< 1/m_j.
    mixed_t* ref = nullptr; ///< 3 per constraint: reference r_i − r_j [Å].
};

namespace {

struct Positions {
    pos_t *x, *y, *z, *xc, *yc, *zc;
};

__global__ void rattle_snapshot(Positions p, const int2* __restrict__ pair,
                                mixed_t* __restrict__ ref, int n, cuda::Box box,
                                const ForceFault* fault) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n || state_frozen(fault))
        return;
    box = box.current();
    const int2 ij = pair[k];
    mixed_t sx =
        load_position(p.x, p.xc, ij.x) - load_position(p.x, p.xc, ij.y);
    mixed_t sy =
        load_position(p.y, p.yc, ij.x) - load_position(p.y, p.yc, ij.y);
    mixed_t sz =
        load_position(p.z, p.zc, ij.x) - load_position(p.z, p.zc, ij.y);
    box.min_image(sx, sy, sz);
    ref[3 * k + 0] = sx;
    ref[3 * k + 1] = sy;
    ref[3 * k + 2] = sz;
}

__device__ __forceinline__ void add_position(pos_t* primary, pos_t* correction,
                                             int i, mixed_t delta) {
    store_position(primary, correction, i,
                   load_position(primary, correction, i) + delta);
}

__global__ void rattle_positions(
    Positions p, mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
    mixed_t* __restrict__ vz, const int* __restrict__ offset,
    const int2* __restrict__ pair, const mixed_t* __restrict__ d2,
    const mixed_t* __restrict__ imi, const mixed_t* __restrict__ imj,
    const mixed_t* __restrict__ ref, int n_clusters, mixed_t inv_dt,
    mixed_t tol, int max_iter, cuda::Box box, ForceFault* fault) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_clusters || state_frozen(fault))
        return;
    box = box.current();
    const int k0 = offset[c], k1 = offset[c + 1];
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        mixed_t worst = 0;
        for (int k = k0; k < k1; ++k) {
            const int i = pair[k].x, j = pair[k].y;
            mixed_t rx =
                load_position(p.x, p.xc, i) - load_position(p.x, p.xc, j);
            mixed_t ry =
                load_position(p.y, p.yc, i) - load_position(p.y, p.yc, j);
            mixed_t rz =
                load_position(p.z, p.zc, i) - load_position(p.z, p.zc, j);
            box.min_image(rx, ry, rz);
            const mixed_t sigma = rx * rx + ry * ry + rz * rz - d2[k];
            const mixed_t err = fabs(sigma) / d2[k];
            worst = fmax(worst, err);
            if (err <= tol)
                continue;
            const mixed_t sx = ref[3 * k], sy = ref[3 * k + 1],
                          sz = ref[3 * k + 2];
            const mixed_t rs = rx * sx + ry * sy + rz * sz;
            const mixed_t denom = mixed_t(2) * (imi[k] + imj[k]) * rs;
            if (fabs(denom) < mixed_t(1e-12))
                continue;
            const mixed_t g = -sigma / denom;
            const mixed_t gi = g * imi[k], gj = g * imj[k];
            add_position(p.x, p.xc, i, gi * sx);
            add_position(p.y, p.yc, i, gi * sy);
            add_position(p.z, p.zc, i, gi * sz);
            add_position(p.x, p.xc, j, -gj * sx);
            add_position(p.y, p.yc, j, -gj * sy);
            add_position(p.z, p.zc, j, -gj * sz);
            // The constraint force also changed the momentum.
            vx[i] += gi * sx * inv_dt;
            vy[i] += gi * sy * inv_dt;
            vz[i] += gi * sz * inv_dt;
            vx[j] -= gj * sx * inv_dt;
            vy[j] -= gj * sy * inv_dt;
            vz[j] -= gj * sz * inv_dt;
        }
        if (worst <= tol)
            break;
    }
    if (iter >= max_iter)
        raise_force_fault(fault, ForceFault::kConstraint, pair[k0].x);
}

__global__ void
rattle_velocities(Positions p, mixed_t* __restrict__ vx,
                  mixed_t* __restrict__ vy, mixed_t* __restrict__ vz,
                  const int* __restrict__ offset, const int2* __restrict__ pair,
                  const mixed_t* __restrict__ d2,
                  const mixed_t* __restrict__ imi,
                  const mixed_t* __restrict__ imj, int n_clusters, mixed_t tol,
                  int max_iter, cuda::Box box, ForceFault* fault) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_clusters || state_frozen(fault))
        return;
    box = box.current();
    const int k0 = offset[c], k1 = offset[c + 1];
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        mixed_t worst = 0;
        for (int k = k0; k < k1; ++k) {
            const int i = pair[k].x, j = pair[k].y;
            mixed_t rx =
                load_position(p.x, p.xc, i) - load_position(p.x, p.xc, j);
            mixed_t ry =
                load_position(p.y, p.yc, i) - load_position(p.y, p.yc, j);
            mixed_t rz =
                load_position(p.z, p.zc, i) - load_position(p.z, p.zc, j);
            box.min_image(rx, ry, rz);
            const mixed_t rv = rx * (vx[i] - vx[j]) + ry * (vy[i] - vy[j])
                               + rz * (vz[i] - vz[j]);
            const mixed_t err = fabs(rv) / d2[k];
            worst = fmax(worst, err);
            if (err <= tol)
                continue;
            const mixed_t k_lag = -rv / ((imi[k] + imj[k]) * d2[k]);
            const mixed_t ki = k_lag * imi[k], kj = k_lag * imj[k];
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
    if (iter >= max_iter)
        raise_force_fault(fault, ForceFault::kConstraint, pair[k0].x);
}

template <typename T>
std::vector<T> download(const T* device, int n) {
    std::vector<T> host(static_cast<std::size_t>(std::max(n, 0)));
    if (n > 0)
        ATV_CUDA_CHECK(cudaMemcpy(host.data(), device, n * sizeof(T),
                                  cudaMemcpyDeviceToHost));
    return host;
}

template <typename T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    ATV_CUDA_CHECK(
        cudaMalloc(&device, std::max<std::size_t>(1, host.size()) * sizeof(T)));
    if (!host.empty())
        ATV_CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
    return device;
}

} // namespace

FixRattle::FixRattle(double dt, double tolerance, int max_iter)
    : dt_(dt), tol_(tolerance), max_iter_(max_iter),
      state_(std::make_unique<State>()) {
    if (!(dt > 0.0))
        throw std::invalid_argument("cuda::md::FixRattle: dt must be positive");
    if (!(tolerance > 0.0))
        throw std::invalid_argument(
            "cuda::md::FixRattle: tolerance must be positive");
    if (tolerance < precision_tolerance_floor())
        throw std::invalid_argument(
            "cuda::md::FixRattle: tolerance is below the integration "
            "precision floor");
    if (max_iter < 1)
        throw std::invalid_argument(
            "cuda::md::FixRattle: max_iter must be >= 1");
}

FixRattle::~FixRattle() = default;
FixRattle::FixRattle(FixRattle&&) noexcept = default;
FixRattle& FixRattle::operator=(FixRattle&&) noexcept = default;

void FixRattle::set_timestep(double dt) {
    if (!std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument(
            "cuda::md::FixRattle: timestep must be finite and positive");
    dt_ = dt;
}

void FixRattle::constrain_bond(int bond_type, double length) {
    if (!(length > 0.0))
        throw std::invalid_argument(
            "cuda::md::FixRattle: constrained length must be positive");
    bond_rules_.push_back({bond_type, length});
}

void FixRattle::constrain_angle(int angle_type, double distance_13) {
    if (!(distance_13 > 0.0))
        throw std::invalid_argument(
            "cuda::md::FixRattle: constrained 1-3 distance must be positive");
    angle_rules_.push_back({angle_type, distance_13});
}

BigInt FixRattle::n_constraints(const cpu::Frame& frame) const {
    if (!frame.has_bonded())
        return 0;
    const auto num_bond = frame.span(kAtom_num_bond);
    const auto bond_type = frame.span(kAtom_bond_type);
    const auto num_angle = frame.span(kAtom_num_angle);
    const auto angle_type = frame.span(kAtom_angle_type);
    BigInt n = 0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        for (int m = 0; m < num_bond[i]; ++m)
            for (const Rule& r : bond_rules_)
                n += bond_type(i, m) == r.type;
        for (int m = 0; m < num_angle[i]; ++m)
            for (const Rule& r : angle_rules_)
                n += angle_type(i, m) == r.type;
    }
    return n;
}

void FixRattle::build(const BuildContext& ctx) {
    const cuda::Frame& f = ctx.frame;
    if (f.n_ghost() != 0 || f.n_local() != f.n_atoms())
        throw std::runtime_error(
            "cuda::md::FixRattle: single-GPU frames only (no halo rows)");
    const auto mass = download(f.atoms.mass, f.n_atoms());
    const auto bi = download(f.bonds.atomi, f.n_bonds());
    const auto bj = download(f.bonds.atomj, f.n_bonds());
    const auto bt = download(f.bonds.type, f.n_bonds());
    const auto ai = download(f.angles.atomi, f.n_angles());
    const auto aj = download(f.angles.atomj, f.n_angles());
    const auto ak = download(f.angles.atomk, f.n_angles());
    const auto at = download(f.angles.type, f.n_angles());

    // The CPU table order: per owner atom (bond atom i, angle centre j) its
    // bonds, then its angles, each against every rule in registration order.
    struct Entry {
        int owner, kind, index;
    };
    std::vector<Entry> entries;
    entries.reserve(bi.size() + ai.size());
    for (int k = 0; k < f.n_bonds(); ++k)
        entries.push_back({bi[static_cast<std::size_t>(k)], 0, k});
    for (int k = 0; k < f.n_angles(); ++k)
        entries.push_back({aj[static_cast<std::size_t>(k)], 1, k});
    std::stable_sort(
        entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return a.owner != b.owner ? a.owner < b.owner : a.kind < b.kind;
        });
    struct Row {
        int i, j;
        double distance;
    };
    std::vector<Row> table;
    for (const Entry& e : entries) {
        const auto q = static_cast<std::size_t>(e.index);
        if (e.kind == 0) {
            for (const Rule& r : bond_rules_)
                if (bt[q] == r.type)
                    table.push_back({bi[q], bj[q], r.distance});
        } else {
            for (const Rule& r : angle_rules_)
                if (at[q] == r.type)
                    table.push_back({ai[q], ak[q], r.distance});
        }
    }
    if (table.empty())
        throw std::runtime_error(
            "cuda::md::FixRattle: no constraints matched the topology. Did "
            "you call constrain_bond()/constrain_angle() with the right type "
            "indices?");
    const int n_atoms = f.n_atoms();
    for (const Row& r : table)
        if (r.i < 0 || r.i >= n_atoms || r.j < 0 || r.j >= n_atoms
            || r.i == r.j)
            throw std::runtime_error(
                "cuda::md::FixRattle: a constrained pair names a row outside "
                "the frame");

    // Clusters: connected components over shared atoms, ids by first
    // appearance in table order, constraints in table order within each.
    std::vector<int> parent(static_cast<std::size_t>(n_atoms));
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int v) {
        while (parent[static_cast<std::size_t>(v)] != v) {
            auto& p = parent[static_cast<std::size_t>(v)];
            p = parent[static_cast<std::size_t>(p)];
            v = p;
        }
        return v;
    };
    for (const Row& r : table) {
        const int a = find(r.i), b = find(r.j);
        if (a != b)
            parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
    }
    std::unordered_map<int, int> cluster_id;
    std::vector<int> cluster_of(table.size());
    for (std::size_t k = 0; k < table.size(); ++k)
        cluster_of[k] =
            cluster_id
                .emplace(find(table[k].i), static_cast<int>(cluster_id.size()))
                .first->second;
    const int n_clusters = static_cast<int>(cluster_id.size());
    std::vector<int> offset(static_cast<std::size_t>(n_clusters) + 1, 0);
    for (int c : cluster_of)
        ++offset[static_cast<std::size_t>(c) + 1];
    std::partial_sum(offset.begin(), offset.end(), offset.begin());
    std::vector<int> fill(offset.begin(), offset.end() - 1);
    std::vector<int2> pair(table.size());
    std::vector<mixed_t> d2(table.size()), imi(table.size()), imj(table.size());
    for (std::size_t k = 0; k < table.size(); ++k) {
        const auto q = static_cast<std::size_t>(
            fill[static_cast<std::size_t>(cluster_of[k])]++);
        const Row& r = table[k];
        const double mi = mass[static_cast<std::size_t>(r.i)],
                     mj = mass[static_cast<std::size_t>(r.j)];
        if (!(mi > 0.0) || !(mj > 0.0))
            throw std::runtime_error(
                "cuda::md::FixRattle: a constrained atom has no positive mass");
        pair[q] = {r.i, r.j};
        d2[q] = static_cast<mixed_t>(r.distance * r.distance);
        imi[q] = static_cast<mixed_t>(1.0 / mi);
        imj[q] = static_cast<mixed_t>(1.0 / mj);
    }

    destroy({ctx.runtime, ctx.frame});
    state_->n_constraints = static_cast<int>(table.size());
    state_->n_clusters = n_clusters;
    state_->offset = upload(offset);
    state_->pair = upload(pair);
    state_->d2 = upload(d2);
    state_->imi = upload(imi);
    state_->imj = upload(imj);
    state_->ref = upload(std::vector<mixed_t>(3 * table.size(), mixed_t(0)));
}

void FixRattle::launch(const LaunchContext& ctx) {
    const State& s = *state_;
    if (s.n_clusters == 0)
        return;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    const auto& a = f.atoms;
    const Positions pos{a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr};
    const int block = cuda::kDefaultBlock;
    if (ctx.at(ATV::md::Phase::initial_integrate))
        rattle_snapshot<<<cuda::grid_size(s.n_constraints, block), block, 0,
                          stream>>>(pos, s.pair, s.ref, s.n_constraints,
                                    ctx.runtime.host_box, f.force_fault);
    else if (ctx.at(ATV::md::Phase::post_integrate))
        rattle_positions<<<cuda::grid_size(s.n_clusters, block), block, 0,
                           stream>>>(
            pos, a.vx, a.vy, a.vz, s.offset, s.pair, s.d2, s.imi, s.imj, s.ref,
            s.n_clusters, mixed_t(1.0 / dt_), mixed_t(tol_), max_iter_,
            ctx.runtime.host_box, f.force_fault);
    else if (ctx.at(ATV::md::Phase::final_integrate))
        rattle_velocities<<<cuda::grid_size(s.n_clusters, block), block, 0,
                            stream>>>(pos, a.vx, a.vy, a.vz, s.offset, s.pair,
                                      s.d2, s.imi, s.imj, s.n_clusters,
                                      mixed_t(tol_), max_iter_,
                                      ctx.runtime.host_box, f.force_fault);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void FixRattle::destroy(const DestroyContext&) {
    cudaFree(state_->offset);
    cudaFree(state_->pair);
    cudaFree(state_->d2);
    cudaFree(state_->imi);
    cudaFree(state_->imj);
    cudaFree(state_->ref);
    *state_ = {};
}

std::size_t FixRattle::memory_bytes() const {
    return static_cast<std::size_t>(state_->n_clusters + 1) * sizeof(int)
           + static_cast<std::size_t>(state_->n_constraints)
                 * (sizeof(int2) + 6 * sizeof(mixed_t));
}

int FixRattle::n_clusters() const { return state_->n_clusters; }

} // namespace md
} // namespace cuda
} // namespace ATV
