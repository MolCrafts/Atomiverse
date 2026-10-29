/// @file cuda/gpw/internal/gpw_forces.cu
/// @brief Device grid force kernels for CUDA GPW (Γ RKS first slice).
///
/// All scientific grid loops run on device — no host loop over grid points
/// in the force hot path. Formulas match CPU `assemble_gpw_forces` grid
/// terms (Quickstep GPW forces).
#include "atomiverse/cuda/gpw/internal/gpw_forces.cuh"

#include <cmath>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

constexpr int kBlock = 256;
constexpr double kPi = 3.14159265358979323846;

__device__ __forceinline__ void grid_r(long idx, int n0, int n1, int n2,
                                       double L0, double L1, double L2,
                                       double o0, double o1, double o2,
                                       double r[3]) {
    const int k = static_cast<int>(idx % n2);
    const long t = idx / n2;
    const int j = static_cast<int>(t % n1);
    const int i = static_cast<int>(t / n1);
    r[0] = o0 + L0 * static_cast<double>(i) / static_cast<double>(n0);
    r[1] = o1 + L1 * static_cast<double>(j) / static_cast<double>(n1);
    r[2] = o2 + L2 * static_cast<double>(k) / static_cast<double>(n2);
}

/// One cbgd active point per thread: accumulate V_eff grid Pulay.
__global__ void grid_veff_force_kernel(
    int n_active_d, double w, int nb, const long* __restrict__ cbgd_gidx,
    const std::size_t* __restrict__ cbgd_off, const int* __restrict__ cbgd_fu,
    const double* __restrict__ cbgd_fv, const std::size_t* __restrict__ cbg_off,
    const int* __restrict__ cbg_fu, const double* __restrict__ cbg_fv,
    const int* __restrict__ g2a, const double* __restrict__ veff,
    const double* __restrict__ P, const int* __restrict__ owner,
    double* __restrict__ forces) {
    const int ad = static_cast<int>(blockIdx.x) * blockDim.x
                   + static_cast<int>(threadIdx.x);
    if (ad >= n_active_d)
        return;
    const long g = cbgd_gidx[ad];
    const int a = g2a[g];
    if (a < 0)
        return;
    const std::size_t cs = cbg_off[a];
    const std::size_t ce = cbg_off[a + 1];
    const std::size_t ds = cbgd_off[ad];
    const std::size_t de = cbgd_off[ad + 1];
    const double vg = veff[g];
    const double scale = 2.0 * w * vg;
    for (std::size_t id = ds; id < de; ++id) {
        const int fdu = cbgd_fu[id];
        const int u = fdu / 3;
        const int k = fdu % 3;
        const double chi_dmu = cbgd_fv[id];
        const double* Pu = P + static_cast<std::size_t>(u) * nb;
        double pchi = 0.0;
        for (std::size_t ib = cs; ib < ce; ++ib)
            pchi += Pu[cbg_fu[ib]] * cbg_fv[ib];
        const double f = -scale * chi_dmu * pchi;
        atomicAdd(forces + 3 * owner[u] + k, f);
    }
}

/// One (atom, grid-chunk) via grid-stride: ionic HF.
__global__ void ionic_hf_kernel(int n_atoms, long npts, double w, int n0,
                                int n1, int n2, double L0, double L1, double L2,
                                double o0, double o1, double o2, bool periodic,
                                const DeviceForceAtom* __restrict__ atoms,
                                const double* __restrict__ v_es,
                                double* __restrict__ forces) {
    const long ntot = npts;
    for (int A = 0; A < n_atoms; ++A) {
        const DeviceForceAtom a = atoms[A];
        if (!(a.z_val != 0.0) || !(a.rloc > 0.0))
            continue;
        const double s2 = a.rloc * a.rloc;
        const double norm = a.z_val * pow(2.0 * kPi * s2, -1.5);
        const double inv_s2 = 1.0 / s2;
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (long g = static_cast<long>(blockIdx.x) * blockDim.x
                      + static_cast<long>(threadIdx.x);
             g < ntot; g += static_cast<long>(blockDim.x) * gridDim.x) {
            double r[3];
            grid_r(g, n0, n1, n2, L0, L1, L2, o0, o1, o2, r);
            double d[3] = {r[0] - a.pos[0], r[1] - a.pos[1], r[2] - a.pos[2]};
            if (periodic) {
                d[0] -= L0 * nearbyint(d[0] / L0);
                d[1] -= L1 * nearbyint(d[1] / L1);
                d[2] -= L2 * nearbyint(d[2] / L2);
            }
            const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            const double gauss = norm * exp(-r2 * 0.5 * inv_s2);
            const double vg = v_es[g] * gauss * inv_s2;
            sx += vg * d[0];
            sy += vg * d[1];
            sz += vg * d[2];
        }
        // Block reduce then atomic to force (one block partial per atom).
        __shared__ double s[3][256];
        s[0][threadIdx.x] = sx;
        s[1][threadIdx.x] = sy;
        s[2][threadIdx.x] = sz;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s[0][threadIdx.x] += s[0][threadIdx.x + stride];
                s[1][threadIdx.x] += s[1][threadIdx.x + stride];
                s[2][threadIdx.x] += s[2][threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(forces + 3 * A + 0, w * s[0][0]);
            atomicAdd(forces + 3 * A + 1, w * s[1][0]);
            atomicAdd(forces + 3 * A + 2, w * s[2][0]);
        }
        __syncthreads();
    }
}

/// Local-PP HF: one atom outer, grid-stride inner (like CPU).
__global__ void local_pp_hf_kernel(int n_atoms, long npts, double w, int n0,
                                   int n1, int n2, double L0, double L1,
                                   double L2, double o0, double o1, double o2,
                                   bool periodic,
                                   const DeviceForceAtom* __restrict__ atoms,
                                   const double* __restrict__ rho,
                                   double* __restrict__ forces) {
    for (int A = 0; A < n_atoms; ++A) {
        const DeviceForceAtom a = atoms[A];
        if (a.n_cloc <= 0 || !(a.rloc > 0.0))
            continue;
        const double inv_s2 = 1.0 / (a.rloc * a.rloc);
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (long g = static_cast<long>(blockIdx.x) * blockDim.x
                      + static_cast<long>(threadIdx.x);
             g < npts; g += static_cast<long>(blockDim.x) * gridDim.x) {
            const double rg = rho[g];
            if (rg == 0.0)
                continue;
            double r[3];
            grid_r(g, n0, n1, n2, L0, L1, L2, o0, o1, o2, r);
            double d[3] = {r[0] - a.pos[0], r[1] - a.pos[1], r[2] - a.pos[2]};
            if (periodic) {
                d[0] -= L0 * nearbyint(d[0] / L0);
                d[1] -= L1 * nearbyint(d[1] / L1);
                d[2] -= L2 * nearbyint(d[2] / L2);
            }
            const double t = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * inv_s2;
            double poly = 0.0, dpoly = 0.0;
            double tp = 1.0, tpm1 = 0.0;
            for (int k = 0; k < a.n_cloc; ++k) {
                poly += a.cloc[k] * tp;
                if (k > 0)
                    dpoly += static_cast<double>(k) * a.cloc[k] * tpm1;
                tpm1 = tp;
                tp *= t;
            }
            const double dvdt = exp(-0.5 * t) * (dpoly - 0.5 * poly);
            const double cc = rg * dvdt * 2.0 * inv_s2;
            sx += cc * d[0];
            sy += cc * d[1];
            sz += cc * d[2];
        }
        __shared__ double s[3][256];
        s[0][threadIdx.x] = sx;
        s[1][threadIdx.x] = sy;
        s[2][threadIdx.x] = sz;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s[0][threadIdx.x] += s[0][threadIdx.x + stride];
                s[1][threadIdx.x] += s[1][threadIdx.x + stride];
                s[2][threadIdx.x] += s[2][threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(forces + 3 * A + 0, w * s[0][0]);
            atomicAdd(forces + 3 * A + 1, w * s[1][0]);
            atomicAdd(forces + 3 * A + 2, w * s[2][0]);
        }
        __syncthreads();
    }
}

} // namespace

void accumulate_grid_veff_forces_device(const CollocateGridDesc& grid,
                                        const DeviceCbgView& cbg,
                                        const DeviceCbgView& cbgd,
                                        const double* d_veff, const double* d_P,
                                        int nb, const int* d_owner,
                                        double* d_forces, cudaStream_t stream) {
    if (cbgd.n_active <= 0 || nb <= 0)
        return;
    const double w = grid.weight();
    const int grid_dim = (cbgd.n_active + kBlock - 1) / kBlock;
    grid_veff_force_kernel<<<grid_dim, kBlock, 0, stream>>>(
        cbgd.n_active, w, nb, cbgd.gidx, cbgd.off, cbgd.fu, cbgd.fv, cbg.off,
        cbg.fu, cbg.fv, cbg.g2a, d_veff, d_P, d_owner, d_forces);
}

void accumulate_ionic_hf_forces_device(const CollocateGridDesc& grid,
                                       const DeviceForceAtom* d_atoms,
                                       int n_atoms, const double* d_v_es,
                                       bool periodic, double* d_forces,
                                       cudaStream_t stream) {
    if (n_atoms <= 0)
        return;
    const long npts = grid.npts();
    if (npts <= 0)
        return;
    const double w = grid.weight();
    // Enough blocks to cover the grid once per atom loop (grid-stride).
    int grid_dim =
        static_cast<int>((npts + kBlock - 1) / static_cast<long>(kBlock));
    if (grid_dim > 1024)
        grid_dim = 1024;
    ionic_hf_kernel<<<grid_dim, kBlock, 0, stream>>>(
        n_atoms, npts, w, grid.n[0], grid.n[1], grid.n[2], grid.L[0], grid.L[1],
        grid.L[2], grid.origin[0], grid.origin[1], grid.origin[2], periodic,
        d_atoms, d_v_es, d_forces);
}

void accumulate_local_pp_hf_forces_device(const CollocateGridDesc& grid,
                                          const DeviceForceAtom* d_atoms,
                                          int n_atoms, const double* d_rho,
                                          bool periodic, double* d_forces,
                                          cudaStream_t stream) {
    if (n_atoms <= 0)
        return;
    const long npts = grid.npts();
    if (npts <= 0)
        return;
    const double w = grid.weight();
    int grid_dim =
        static_cast<int>((npts + kBlock - 1) / static_cast<long>(kBlock));
    if (grid_dim > 1024)
        grid_dim = 1024;
    local_pp_hf_kernel<<<grid_dim, kBlock, 0, stream>>>(
        n_atoms, npts, w, grid.n[0], grid.n[1], grid.n[2], grid.L[0], grid.L[1],
        grid.L[2], grid.origin[0], grid.origin[1], grid.origin[2], periodic,
        d_atoms, d_rho, d_forces);
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV

#include <algorithm>
#include <array>
#include <climits>
#include <stdexcept>
#include <string>

namespace ATV::cuda::gpw::internal {
namespace {
void force_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("GpwForceWorkspace: ") + operation
                                 + ": " + cudaGetErrorString(error));
}
// One thread owns one sparse grid point. Reproduce the legacy compact-basis
// triangular contraction, including its exact screened chi support and order.
__global__ void compact_density_kernel(DeviceCbgView cbg, int nb,
                                       const double* P, double* rho) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= cbg.n_active)
        return;
    const auto start = cbg.off[a], end = cbg.off[a + 1];
    double total = 0;
    for (auto i = start; i < end; ++i) {
        const int u = cbg.fu[i];
        const double value = cbg.fv[i];
        const double* row = P + static_cast<std::size_t>(u) * nb;
        total += value * value * row[u];
        double upper = 0;
        for (auto j = i + 1; j < end; ++j)
            upper += row[cbg.fu[j]] * cbg.fv[j];
        total += 2 * value * upper;
    }
    rho[cbg.gidx[a]] = total;
}
// Independent immutable entries stream through contiguous memory. Each thread
// contracts one entry with device P/W; atom-lane sums use native FP64 atomics.
// Grid stride covers long sparse lists with fixed launch dimensions and tails.
__global__ void ao_force_entries_kernel(const AoForceEntry* entries,
                                        std::size_t count, const double* P,
                                        const double* W, double* forces) {
    for (std::size_t i =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const auto entry = entries[i];
        atomicAdd(forces + entry.force_lane,
                  entry.overlap * W[entry.matrix_index]
                      + entry.one_body * P[entry.matrix_index]);
    }
}
// One thread owns a canonical pair and computes the radial force on device.
// Self-image separations change with strain but not with an atom translation:
// their energy/virial are retained by the shared model, their atomic force is 0.
__global__ void core_overlap_force_kernel(const gaussian_core::Pair* pairs,
                                          std::size_t count, double* forces) {
    for (std::size_t i =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const auto pair = pairs[i];
        if (pair.atom_a == pair.atom_b)
            continue;
        const auto value = gaussian_core::evaluate_pair(pair);
        for (int k = 0; k < 3; ++k) {
            const double f = value.force_over_r * pair.displacement[k];
            atomicAdd(forces + 3 * pair.atom_a + k, f);
            atomicAdd(forces + 3 * pair.atom_b + k, -f);
        }
    }
}
} // namespace

struct GpwForceWorkspace::Impl {
    cudaStream_t stream = nullptr;
    CollocateGridDesc grid;
    DeviceCbgView cbg{}, cbgd{};
    int nb = 0, na = 0;
    int* owner = nullptr;
    DeviceForceAtom* atoms = nullptr;
    AoForceEntry* ao = nullptr;
    std::size_t ao_count = 0;
    gaussian_core::Pair* core_pairs = nullptr;
    std::size_t core_count = 0;
    double* force = nullptr;
    double* rho = nullptr;
    std::array<void*, 20> allocations{};
    std::size_t count = 0, fixed_bytes = 0, work_bytes = 0;
    ~Impl() {
        cudaStreamSynchronize(stream);
        while (count)
            cudaFree(allocations[--count]);
    }
    template <class T>
    T* allocate(std::size_t n, bool scratch = false) {
        if (!n)
            return nullptr;
        if (count == allocations.size())
            throw std::logic_error(
                "GPW force geometry scratch capacity exceeded");
        if (n > static_cast<std::size_t>(-1) / sizeof(T))
            throw std::overflow_error("GPW force geometry allocation overflow");
        T* pointer = nullptr;
        force_check(cudaMalloc(&pointer, n * sizeof(T)),
                    "allocate cold force table");
        allocations[count++] = pointer;
        (scratch ? work_bytes : fixed_bytes) += n * sizeof(T);
        return pointer;
    }
    template <class T>
    T* upload(const std::vector<T>& values) {
        auto* pointer = allocate<T>(values.size());
        if (pointer)
            force_check(cudaMemcpyAsync(pointer, values.data(),
                                        values.size() * sizeof(T),
                                        cudaMemcpyHostToDevice, stream),
                        "upload cold force table");
        return pointer;
    }
};
GpwForceWorkspace::GpwForceWorkspace() = default;
GpwForceWorkspace::~GpwForceWorkspace() = default;
void GpwForceWorkspace::build(
    const CollocateGridDesc& grid, const cpu::gpw::CompactBasisGrid& cbg,
    const cpu::gpw::CompactBasisGrid& cbgd, const std::vector<int>& owner,
    const std::vector<DeviceForceAtom>& atoms,
    const std::vector<AoForceEntry>& ao_entries,
    const std::vector<gaussian_core::Pair>& core_pairs, cudaStream_t stream) {
    if (owner.empty() || atoms.empty() || owner.size() > 46340
        || atoms.size() > static_cast<std::size_t>(INT_MAX / 3))
        throw std::invalid_argument(
            "GPW force geometry has invalid AO/atom dimensions");
    const long npts = static_cast<long>(grid.n[0]) * grid.n[1] * grid.n[2];
    if (npts <= 0 || cbg.npts != npts || cbgd.npts != npts
        || cbg.off.size() != cbg.gidx.size() + 1
        || cbgd.off.size() != cbgd.gidx.size() + 1
        || cbg.fu.size() != cbg.fv.size() || cbgd.fu.size() != cbgd.fv.size()
        || cbg.gidx.size() > INT_MAX || cbgd.gidx.size() > INT_MAX)
        throw std::invalid_argument(
            "GPW force geometry has inconsistent sparse grids");
    for (int atom : owner)
        if (atom < 0 || atom >= static_cast<int>(atoms.size()))
            throw std::invalid_argument("GPW force AO owner out of range");
    for (const auto& entry : ao_entries)
        if (entry.force_lane < 0
            || entry.force_lane >= 3 * static_cast<int>(atoms.size())
            || entry.matrix_index < 0
            || entry.matrix_index
                   >= static_cast<int>(owner.size() * owner.size()))
            throw std::invalid_argument("GPW force AO entry out of range");
    for (const auto& pair : core_pairs)
        if (pair.atom_a < 0 || pair.atom_b < 0
            || pair.atom_a >= static_cast<int>(atoms.size())
            || pair.atom_b >= static_cast<int>(atoms.size()))
            throw std::invalid_argument(
                "GPW force core pair atom out of range");
    // Complete validation/host allocations before the first asynchronous copy.
    std::vector<int> g2a(static_cast<std::size_t>(npts), -1);
    for (std::size_t a = 0; a < cbg.gidx.size(); ++a) {
        if (cbg.gidx[a] < 0 || cbg.gidx[a] >= npts)
            throw std::invalid_argument("GPW force sparse point out of range");
        g2a[static_cast<std::size_t>(cbg.gidx[a])] = static_cast<int>(a);
    }
    auto candidate = std::make_unique<Impl>();
    candidate->stream = stream;
    candidate->grid = grid;
    candidate->nb = static_cast<int>(owner.size());
    candidate->na = static_cast<int>(atoms.size());
    auto upload_grid = [&](const cpu::gpw::CompactBasisGrid& input,
                           DeviceCbgView& output) {
        output.npts = npts;
        output.n_active = static_cast<int>(input.gidx.size());
        output.gidx = candidate->upload(input.gidx);
        output.off = candidate->upload(input.off);
        output.fu = candidate->upload(input.fu);
        output.fv = candidate->upload(input.fv);
    };
    upload_grid(cbg, candidate->cbg);
    upload_grid(cbgd, candidate->cbgd);
    candidate->cbg.g2a = candidate->upload(g2a);
    candidate->owner = candidate->upload(owner);
    candidate->atoms = candidate->upload(atoms);
    candidate->ao = candidate->upload(ao_entries);
    candidate->ao_count = ao_entries.size();
    candidate->core_pairs = candidate->upload(core_pairs);
    candidate->core_count = core_pairs.size();
    candidate->force = candidate->allocate<double>(3 * atoms.size(), true);
    candidate->rho =
        candidate->allocate<double>(static_cast<std::size_t>(npts), true);
    force_check(cudaStreamSynchronize(stream), "cold force-table upload fence");
    impl_ = std::move(candidate);
}
void GpwForceWorkspace::evaluate(const double* P, const double* W,
                                 const double* veff, const double* v_es,
                                 bool local_pp_on_grid) {
    if (!impl_ || !P || !veff || !v_es || (impl_->ao_count && !W))
        throw std::invalid_argument(
            "GPW force body needs complete resident physical state");
    auto& state = *impl_;
    force_check(cudaMemsetAsync(state.force, 0, 3 * state.na * sizeof(double),
                                state.stream),
                "initialize resident forces");
    if (state.ao_count) {
        const auto blocks = static_cast<unsigned>(
            std::min<std::size_t>((state.ao_count + 255) / 256, 65535));
        ao_force_entries_kernel<<<blocks, 256, 0, state.stream>>>(
            state.ao, state.ao_count, P, W, state.force);
        force_check(cudaGetLastError(), "resident AO force contraction");
    }
    accumulate_grid_veff_forces_device(state.grid, state.cbg, state.cbgd, veff,
                                       P, state.nb, state.owner, state.force,
                                       state.stream);
    accumulate_ionic_hf_forces_device(state.grid, state.atoms, state.na, v_es,
                                      state.grid.periodic, state.force,
                                      state.stream);
    if (local_pp_on_grid) {
        force_check(cudaMemsetAsync(state.rho, 0,
                                    state.cbg.npts * sizeof(double),
                                    state.stream),
                    "initialize compact density");
        if (state.cbg.n_active) {
            compact_density_kernel<<<(state.cbg.n_active + 255) / 256, 256, 0,
                                     state.stream>>>(state.cbg, state.nb, P,
                                                     state.rho);
            force_check(cudaGetLastError(),
                        "resident compact density contraction");
        }
        accumulate_local_pp_hf_forces_device(state.grid, state.atoms, state.na,
                                             state.rho, state.grid.periodic,
                                             state.force, state.stream);
    }
    force_check(cudaGetLastError(), "resident grid-force launches");
    if (state.core_count) {
        const auto blocks = static_cast<unsigned>(
            std::min<std::size_t>((state.core_count + 255) / 256, 65535));
        core_overlap_force_kernel<<<blocks, 256, 0, state.stream>>>(
            state.core_pairs, state.core_count, state.force);
        force_check(cudaGetLastError(), "resident Gaussian-core pair forces");
    }
}
const double* GpwForceWorkspace::d_forces() const {
    return impl_ ? impl_->force : nullptr;
}
const double* GpwForceWorkspace::d_rho() const {
    return impl_ ? impl_->rho : nullptr;
}
std::size_t GpwForceWorkspace::geometry_bytes() const {
    return impl_ ? impl_->fixed_bytes : 0;
}
std::size_t GpwForceWorkspace::scratch_bytes() const {
    return impl_ ? impl_->work_bytes : 0;
}
std::size_t GpwForceWorkspace::ao_entries() const {
    return impl_ ? impl_->ao_count : 0;
}
} // namespace ATV::cuda::gpw::internal
