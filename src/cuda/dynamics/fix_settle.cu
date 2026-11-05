/// @file src/cuda/dynamics/fix_settle.cu
/// @brief Analytic SETTLE on the GPU — the arithmetic of
///        `src/cpu/dynamics/fix_settle.cpp`, one thread per water.
///
/// Ref: S. Miyamoto and P. A. Kollman, J. Comput. Chem. 13, 952 (1992),
/// Appendix (positions) and §II.B (velocities).

#include "atomiverse/cuda/dynamics/fix_settle.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace md {

struct FixSettle::State {
    int n_waters = 0;
    int3* rows = nullptr;   ///< (O, H1, H2) rows per water [device].
    mixed_t* ref = nullptr; ///< 9 per water: O, H1, H2 reference [Å].
};

namespace {

struct V3 {
    mixed_t x, y, z;
};

__device__ __forceinline__ V3 cross(const V3& a, const V3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

__device__ __forceinline__ mixed_t dot(const V3& a, const V3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// False (and `out` untouched) for a zero vector: the body frame is undefined.
__device__ __forceinline__ bool normalized(const V3& v, V3& out) {
    const mixed_t n = sqrt(dot(v, v));
    if (!(n > mixed_t(0)))
        return false;
    out = {v.x / n, v.y / n, v.z / n};
    return true;
}

__device__ __forceinline__ mixed_t safe_sqrt(mixed_t v) {
    return sqrt(v > mixed_t(0) ? v : mixed_t(0));
}

struct Positions {
    pos_t *x, *y, *z, *xc, *yc, *zc;
    __device__ __forceinline__ V3 load(int i) const {
        return {load_position(x, xc, i), load_position(y, yc, i),
                load_position(z, zc, i)};
    }
    __device__ __forceinline__ void store(int i, const V3& p) const {
        store_position(x, xc, i, p.x);
        store_position(y, yc, i, p.y);
        store_position(z, zc, i, p.z);
    }
};

__global__ void settle_snapshot(Positions pos, const int3* __restrict__ rows,
                                mixed_t* __restrict__ ref, int n_waters,
                                cuda::Box box, const ForceFault* fault) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_waters || state_frozen(fault))
        return;
    box = box.current();
    const int3 w = rows[k];
    const V3 o = pos.load(w.x), a = pos.load(w.y), b = pos.load(w.z);
    mixed_t h1x = a.x - o.x, h1y = a.y - o.y, h1z = a.z - o.z;
    mixed_t h2x = b.x - o.x, h2y = b.y - o.y, h2z = b.z - o.z;
    box.min_image(h1x, h1y, h1z);
    box.min_image(h2x, h2y, h2z);
    mixed_t* r = ref + 9 * k;
    r[0] = o.x;
    r[1] = o.y;
    r[2] = o.z;
    r[3] = r[0] + h1x;
    r[4] = r[1] + h1y;
    r[5] = r[2] + h1z;
    r[6] = r[0] + h2x;
    r[7] = r[1] + h2y;
    r[8] = r[2] + h2z;
}

__global__ void settle_positions(Positions pos,
                                 const mixed_t* __restrict__ mass,
                                 const int3* __restrict__ rows,
                                 const mixed_t* __restrict__ ref, int n_waters,
                                 mixed_t d_oh, mixed_t d_hh, cuda::Box box,
                                 ForceFault* fault) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_waters || state_frozen(fault))
        return;
    box = box.current();
    const int3 w = rows[k];
    const int o = w.x, a = w.y, b = w.z;
    // Canonical triangle in its centre-of-mass frame (eq. A1).
    const mixed_t rc = mixed_t(0.5) * d_hh;
    const mixed_t height = sqrt(d_oh * d_oh - rc * rc);
    const mixed_t m_o = mass[o], m_h = mass[a];
    const mixed_t m_total = m_o + mixed_t(2) * m_h;
    const mixed_t ra = mixed_t(2) * m_h * height / m_total;
    const mixed_t rb = height - ra;

    const V3 o1 = pos.load(o), pa = pos.load(a), pb = pos.load(b);
    mixed_t b1x = pa.x - o1.x, b1y = pa.y - o1.y, b1z = pa.z - o1.z;
    mixed_t c1x = pb.x - o1.x, c1y = pb.y - o1.y, c1z = pb.z - o1.z;
    box.min_image(b1x, b1y, b1z);
    box.min_image(c1x, c1y, c1z);

    const V3 com{o1.x + m_h * (b1x + c1x) / m_total,
                 o1.y + m_h * (b1y + c1y) / m_total,
                 o1.z + m_h * (b1z + c1z) / m_total};
    const V3 a1{o1.x - com.x, o1.y - com.y, o1.z - com.z};
    const V3 b1{o1.x + b1x - com.x, o1.y + b1y - com.y, o1.z + b1z - com.z};
    const V3 c1{o1.x + c1x - com.x, o1.y + c1y - com.y, o1.z + c1z - com.z};

    const mixed_t* r = ref + 9 * k;
    const V3 b0{r[3] - r[0], r[4] - r[1], r[5] - r[2]};
    const V3 c0{r[6] - r[0], r[7] - r[1], r[8] - r[2]};

    V3 ez, ex;
    if (!normalized(cross(b0, c0), ez) || !normalized(cross(a1, ez), ex)) {
        raise_force_fault(fault, ForceFault::kConstraint, o);
        return;
    }
    const V3 ey = cross(ez, ex);

    const mixed_t xb0d = dot(b0, ex), yb0d = dot(b0, ey);
    const mixed_t xc0d = dot(c0, ex), yc0d = dot(c0, ey);
    const mixed_t za1d = dot(a1, ez);
    const mixed_t xb1d = dot(b1, ex), yb1d = dot(b1, ey), zb1d = dot(b1, ez);
    const mixed_t xc1d = dot(c1, ex), yc1d = dot(c1, ey), zc1d = dot(c1, ez);

    // φ about x′ and ψ about y′ (eq. A6/A7).
    const mixed_t sinphi = za1d / ra;
    const mixed_t cosphi = safe_sqrt(mixed_t(1) - sinphi * sinphi);
    if (!(cosphi > mixed_t(0))) { // oxygen left the plane
        raise_force_fault(fault, ForceFault::kConstraint, o);
        return;
    }
    const mixed_t sinpsi = (zb1d - zc1d) / (mixed_t(2) * rc * cosphi);
    const mixed_t cospsi = safe_sqrt(mixed_t(1) - sinpsi * sinpsi);

    const mixed_t ya2d = ra * cosphi;
    const mixed_t xb2d = -rc * cospsi;
    const mixed_t t1 = -rb * cosphi;
    const mixed_t t2 = rc * sinpsi * sinphi;
    const mixed_t yb2d = t1 - t2;
    const mixed_t yc2d = t1 + t2;

    // θ about z′ (eq. A11), fixed by the reference orientation.
    const mixed_t alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d;
    const mixed_t beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d;
    const mixed_t gamma = xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d;
    const mixed_t al2be2 = alpha * alpha + beta * beta;
    if (!(al2be2 > mixed_t(0))) { // degenerate reference triangle
        raise_force_fault(fault, ForceFault::kConstraint, o);
        return;
    }
    const mixed_t sinthe =
        (alpha * gamma - beta * safe_sqrt(al2be2 - gamma * gamma)) / al2be2;
    const mixed_t costhe = safe_sqrt(mixed_t(1) - sinthe * sinthe);

    // Constrained positions in the body frame (eq. A12).
    const mixed_t xa3d = -ya2d * sinthe;
    const mixed_t ya3d = ya2d * costhe;
    const mixed_t za3d = za1d;
    const mixed_t xb3d = xb2d * costhe - yb2d * sinthe;
    const mixed_t yb3d = xb2d * sinthe + yb2d * costhe;
    const mixed_t zb3d = zb1d;
    const mixed_t xc3d = -xb2d * costhe - yc2d * sinthe;
    const mixed_t yc3d = -xb2d * sinthe + yc2d * costhe;
    const mixed_t zc3d = zc1d;

    auto place = [&](mixed_t xd, mixed_t yd, mixed_t zd, int row) {
        pos.store(row, {com.x + ex.x * xd + ey.x * yd + ez.x * zd,
                        com.y + ex.y * xd + ey.y * yd + ez.y * zd,
                        com.z + ex.z * xd + ey.z * yd + ez.z * zd});
    };
    place(xa3d, ya3d, za3d, o);
    place(xb3d, yb3d, zb3d, a);
    place(xc3d, yc3d, zc3d, b);
}

__global__ void settle_velocities(Positions pos, mixed_t* __restrict__ vx,
                                  mixed_t* __restrict__ vy,
                                  mixed_t* __restrict__ vz,
                                  const mixed_t* __restrict__ mass,
                                  const int3* __restrict__ rows, int n_waters,
                                  cuda::Box box, ForceFault* fault) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_waters || state_frozen(fault))
        return;
    box = box.current();
    const int3 w = rows[k];
    const int atom[3] = {w.x, w.y, w.z};
    // Constraints as (plus, minus) local indices: O–H1, O–H2, H1–H2.
    constexpr int pc[3] = {0, 0, 1}, qc[3] = {1, 2, 2};
    mixed_t inv_m[3];
    V3 p[3];
    for (int a = 0; a < 3; ++a) {
        inv_m[a] = mixed_t(1) / mass[atom[a]];
        p[a] = pos.load(atom[a]);
    }
    mixed_t e[3][3], v[3];
    for (int c = 0; c < 3; ++c) {
        const int i = atom[pc[c]], j = atom[qc[c]];
        mixed_t dx = p[pc[c]].x - p[qc[c]].x, dy = p[pc[c]].y - p[qc[c]].y,
                dz = p[pc[c]].z - p[qc[c]].z;
        box.min_image(dx, dy, dz);
        const mixed_t inv_r = mixed_t(1) / sqrt(dx * dx + dy * dy + dz * dz);
        e[c][0] = dx * inv_r;
        e[c][1] = dy * inv_r;
        e[c][2] = dz * inv_r;
        v[c] = e[c][0] * (vx[i] - vx[j]) + e[c][1] * (vy[i] - vy[j])
               + e[c][2] * (vz[i] - vz[j]);
    }
    mixed_t A[3][3];
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d) {
            const mixed_t ee =
                e[c][0] * e[d][0] + e[c][1] * e[d][1] + e[c][2] * e[d][2];
            const mixed_t sp =
                mixed_t((pc[c] == pc[d]) - (pc[c] == qc[d])) * inv_m[pc[c]];
            const mixed_t sq =
                mixed_t((qc[c] == pc[d]) - (qc[c] == qc[d])) * inv_m[qc[c]];
            A[c][d] = ee * (sp - sq);
        }
    // Cramer on the symmetric positive-definite 3×3.
    const mixed_t c00 = A[1][1] * A[2][2] - A[1][2] * A[2][1];
    const mixed_t c01 = A[1][2] * A[2][0] - A[1][0] * A[2][2];
    const mixed_t c02 = A[1][0] * A[2][1] - A[1][1] * A[2][0];
    const mixed_t det = A[0][0] * c00 + A[0][1] * c01 + A[0][2] * c02;
    if (!(fabs(det) > mixed_t(1e-14) * A[0][0] * A[1][1] * A[2][2])) {
        raise_force_fault(fault, ForceFault::kConstraint, atom[0]);
        return;
    }
    const mixed_t inv_det = mixed_t(1) / det;
    const mixed_t b0 = -v[0], b1 = -v[1], b2 = -v[2];
    const mixed_t tau[3] = {
        (b0 * c00 + b1 * (A[0][2] * A[2][1] - A[0][1] * A[2][2])
         + b2 * (A[0][1] * A[1][2] - A[0][2] * A[1][1]))
            * inv_det,
        (b0 * c01 + b1 * (A[0][0] * A[2][2] - A[0][2] * A[2][0])
         + b2 * (A[0][2] * A[1][0] - A[0][0] * A[1][2]))
            * inv_det,
        (b0 * c02 + b1 * (A[0][1] * A[2][0] - A[0][0] * A[2][1])
         + b2 * (A[0][0] * A[1][1] - A[0][1] * A[1][0]))
            * inv_det};
    mixed_t dv[3][3] = {};
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d) {
            dv[pc[c]][d] += tau[c] * e[c][d] * inv_m[pc[c]];
            dv[qc[c]][d] -= tau[c] * e[c][d] * inv_m[qc[c]];
        }
    mixed_t vnew[3][3];
    for (int a = 0; a < 3; ++a) {
        const int r = atom[a];
        vx[r] += dv[a][0];
        vy[r] += dv[a][1];
        vz[r] += dv[a][2];
        vnew[a][0] = vx[r];
        vnew[a][1] = vy[r];
        vnew[a][2] = vz[r];
    }
    // The solve is exact up to round-off; a residual above it is a
    // corrupted (NaN / degenerate) molecule, never an approximation.
    mixed_t worst = 0;
    for (int c = 0; c < 3; ++c) {
        mixed_t rel[3], norm2 = 0;
        for (int d = 0; d < 3; ++d) {
            rel[d] = vnew[pc[c]][d] - vnew[qc[c]][d];
            norm2 += rel[d] * rel[d];
        }
        const mixed_t along =
            e[c][0] * rel[0] + e[c][1] * rel[1] + e[c][2] * rel[2];
        const mixed_t r = fabs(along) / sqrt(norm2 + mixed_t(1e-300));
        worst = fmax(worst, isfinite(r) ? r : mixed_t(1e300));
    }
    if (!(worst < mixed_t(1e-8)))
        raise_force_fault(fault, ForceFault::kConstraint, atom[0]);
}

} // namespace

FixSettle::FixSettle(double dt, double d_oh, double d_hh)
    : dt_(dt), d_oh_(d_oh), d_hh_(d_hh), state_(std::make_unique<State>()) {
    if (!(dt > 0.0))
        throw std::invalid_argument("cuda::md::FixSettle: dt must be > 0");
    if (!(d_oh > 0.0) || !(d_hh > 0.0))
        throw std::invalid_argument(
            "cuda::md::FixSettle: distances must be > 0");
    if (d_hh >= 2.0 * d_oh)
        throw std::invalid_argument(
            "cuda::md::FixSettle: triangle inequality violated");
}

FixSettle::~FixSettle() = default;
FixSettle::FixSettle(FixSettle&&) noexcept = default;
FixSettle& FixSettle::operator=(FixSettle&&) noexcept = default;

void FixSettle::set_timestep(double dt) {
    if (!std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument(
            "cuda::md::FixSettle: timestep must be finite and positive");
    dt_ = dt;
}

void FixSettle::add_water(TagInt tag_O, TagInt tag_H1, TagInt tag_H2) {
    if (tag_O == tag_H1 || tag_O == tag_H2 || tag_H1 == tag_H2)
        throw std::invalid_argument(
            "cuda::md::FixSettle: water tags must be distinct");
    waters_.push_back({tag_O, tag_H1, tag_H2});
}

void FixSettle::build(const BuildContext& ctx) {
    if (waters_.empty())
        throw std::runtime_error("cuda::md::FixSettle: no waters registered");
    const cuda::Frame& f = ctx.frame;
    if (f.n_ghost() != 0 || f.n_local() != f.n_atoms())
        throw std::runtime_error(
            "cuda::md::FixSettle: single-GPU frames only (no halo rows)");
    // Waters are solved concurrently; an atom in two of them would be
    // written by two threads and constrained twice.
    {
        std::unordered_set<TagInt> seen;
        seen.reserve(3 * waters_.size());
        for (const auto& w : waters_)
            for (const TagInt t : {w.o, w.h1, w.h2})
                if (!seen.insert(t).second)
                    throw std::invalid_argument(
                        "cuda::md::FixSettle: atom tag "
                        + std::to_string(static_cast<long long>(t))
                        + " belongs to more than one water");
    }
    const int n = f.n_atoms();
    std::vector<std::int64_t> tag(static_cast<std::size_t>(n));
    std::vector<mixed_t> mass(static_cast<std::size_t>(n));
    ATV_CUDA_CHECK(cudaMemcpy(tag.data(), f.atoms.tag, n * sizeof(std::int64_t),
                              cudaMemcpyDeviceToHost));
    ATV_CUDA_CHECK(cudaMemcpy(mass.data(), f.atoms.mass, n * sizeof(mixed_t),
                              cudaMemcpyDeviceToHost));
    std::unordered_map<std::int64_t, int> row_of;
    row_of.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        row_of.emplace(tag[static_cast<std::size_t>(i)], i);
    std::vector<int3> rows;
    rows.reserve(waters_.size());
    for (const auto& w : waters_) {
        const auto io = row_of.find(w.o), ia = row_of.find(w.h1),
                   ib = row_of.find(w.h2);
        if (io == row_of.end() || ia == row_of.end() || ib == row_of.end())
            throw std::runtime_error(
                "cuda::md::FixSettle: water ("
                + std::to_string(static_cast<long long>(w.o)) + ", "
                + std::to_string(static_cast<long long>(w.h1)) + ", "
                + std::to_string(static_cast<long long>(w.h2))
                + ") names a tag no row of this frame carries");
        const double ma = mass[static_cast<std::size_t>(ia->second)],
                     mb = mass[static_cast<std::size_t>(ib->second)];
        if (!(ma > 0.0) || !(mass[static_cast<std::size_t>(io->second)] > 0.0))
            throw std::runtime_error(
                "cuda::md::FixSettle: water masses must be > 0");
        if (std::fabs(ma - mb) > 1e-12 * std::max(ma, mb))
            throw std::runtime_error(
                "cuda::md::FixSettle: the two hydrogens of a water must have "
                "equal mass — SETTLE's closed form assumes an isosceles "
                "triangle. Use FixRattle for an asymmetric rigid triple.");
        rows.push_back({io->second, ia->second, ib->second});
    }
    destroy({ctx.runtime, ctx.frame});
    state_->n_waters = static_cast<int>(rows.size());
    ATV_CUDA_CHECK(cudaMalloc(&state_->rows, rows.size() * sizeof(int3)));
    ATV_CUDA_CHECK(cudaMemcpy(state_->rows, rows.data(),
                              rows.size() * sizeof(int3),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMalloc(&state_->ref, 9 * rows.size() * sizeof(mixed_t)));
    ATV_CUDA_CHECK(
        cudaMemset(state_->ref, 0, 9 * rows.size() * sizeof(mixed_t)));
}

void FixSettle::launch(const LaunchContext& ctx) {
    const int nw = state_->n_waters;
    if (nw == 0)
        return;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    const auto& a = f.atoms;
    const Positions pos{a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr};
    const int block = cuda::kDefaultBlock;
    const int grid = cuda::grid_size(nw, block);
    if (ctx.at(ATV::md::Phase::initial_integrate))
        settle_snapshot<<<grid, block, 0, stream>>>(
            pos, state_->rows, state_->ref, nw, ctx.runtime.host_box,
            f.force_fault);
    else if (ctx.at(ATV::md::Phase::post_integrate))
        settle_positions<<<grid, block, 0, stream>>>(
            pos, a.mass, state_->rows, state_->ref, nw, mixed_t(d_oh_),
            mixed_t(d_hh_), ctx.runtime.host_box, f.force_fault);
    else if (ctx.at(ATV::md::Phase::final_integrate))
        settle_velocities<<<grid, block, 0, stream>>>(
            pos, a.vx, a.vy, a.vz, a.mass, state_->rows, nw,
            ctx.runtime.host_box, f.force_fault);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void FixSettle::destroy(const DestroyContext&) {
    cudaFree(state_->rows);
    cudaFree(state_->ref);
    *state_ = {};
}

std::size_t FixSettle::memory_bytes() const {
    return static_cast<std::size_t>(state_->n_waters)
           * (sizeof(int3) + 9 * sizeof(mixed_t));
}

} // namespace md
} // namespace cuda
} // namespace ATV
