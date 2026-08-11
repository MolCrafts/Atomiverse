/// @file cuda/gpw/internal/product_weight_pack.cu
/// @brief Device kernels for GPW product density weight + V_AO scatter.
///
/// Twins of host `product_density_weight` / `scatter_product_acc` in
/// `gpw_solver.cu`. All product loops run on device; P is column-major
/// matching `cpu::linalg::RealMatrix` / `AoDiisEigenDevice::d_P()`.
#include "atomiverse/cuda/gpw/internal/product_weight_pack.cuh"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {
namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ProductWeightPackDevice: ") + what
                                 + ": " + cudaGetErrorString(e));
}

/// @brief Density weight for one product from column-major P.
///
/// Host twin uses row-major mat_to_vec: P_row[μ·nb+ν] = P(μ,ν).
/// Column-major: P_col[μ + ν·nb] = P(μ,ν). Same value.
__device__ __forceinline__ double
product_weight_device(const DeviceProductGeom& g,
                      const DevicePrimRef* __restrict__ prims,
                      const DevicePrimContrib* __restrict__ contribs, int nb,
                      const double* __restrict__ P) {
    const DevicePrimRef A = prims[g.ua];
    const DevicePrimRef B = prims[g.ub];
    double s = 0.0;
    for (int i = 0; i < A.cn; ++i) {
        const DevicePrimContrib ca = contribs[A.cbeg + i];
        // row of column-major P starts at ca.mu (stride nb for columns).
        double rs = 0.0;
        for (int j = 0; j < B.cn; ++j) {
            const DevicePrimContrib cb = contribs[B.cbeg + j];
            rs += P[ca.mu + cb.mu * nb] * cb.coeff;
        }
        s += ca.coeff * rs;
    }
    return s * g.k;
}

__global__ void pack_weights_kernel(const double* __restrict__ P,
                                    const DeviceProductGeom* __restrict__ geom,
                                    const DevicePrimRef* __restrict__ prims,
                                    const DevicePrimContrib* __restrict__ contribs,
                                    DeviceProduct* __restrict__ prods, int nprod,
                                    int nb, double eps_drop) {
    const int ip = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (ip >= nprod)
        return;
    const DeviceProductGeom g = geom[ip];
    double w = product_weight_device(g, prims, contribs, nb, P);
    if (fabs(w) < eps_drop)
        w = 0.0;
    DeviceProduct q;
    q.p = g.p;
    q.weight = w;
    // Per-monom default; apply_group_max_mag lifts mag + L_win to group max.
    q.mag = fabs(w);
    q.L_win = g.la[0] + g.la[1] + g.la[2] + g.lb[0] + g.lb[1] + g.lb[2];
    for (int d = 0; d < 3; ++d) {
        q.pc[d] = g.pc[d];
        q.ra[d] = g.ra[d];
        q.rb[d] = g.rb[d];
        q.la[d] = g.la[d];
        q.lb[d] = g.lb[d];
    }
    prods[ip] = q;
}

/// @brief Contiguous geometry-group max\|weight\| + max Ltot → mag/L_win.
///
/// Products on each level are sorted by collint geometry; a task [beg,end)
/// shares one pol table. One block per task (choose-launch-configuration —
/// few tasks, sequential monom walk per block is fine).
__global__ void group_max_mag_kernel(DeviceProduct* __restrict__ prods,
                                     const int* __restrict__ task_beg,
                                     const int* __restrict__ task_end,
                                     int ntask) {
    const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= ntask)
        return;
    const int beg = task_beg[t];
    const int end = task_end[t];
    if (end <= beg)
        return;
    double m = 0.0;
    int Lmax = 0;
    for (int i = beg; i < end; ++i) {
        m = fmax(m, fabs(prods[i].weight));
        const int Lm = prods[i].la[0] + prods[i].la[1] + prods[i].la[2]
                       + prods[i].lb[0] + prods[i].lb[1] + prods[i].lb[2];
        if (Lm > Lmax)
            Lmax = Lm;
        if (prods[i].L_win > Lmax)
            Lmax = prods[i].L_win;
    }
    for (int i = beg; i < end; ++i) {
        prods[i].mag = m;
        prods[i].L_win = Lmax;
    }
}

/// @brief Sequential product loop matching host scatter order.
///
/// One thread walks all products so V accumulation is bit-identical to the
/// host twin (atomicAdd over shared (μ,ν) is non-associative and can thrash
/// dual-channel O₂ UKS DIIS). Throughput is fine for Γ product lists.
__global__ void scatter_acc_kernel(const DeviceProductGeom* __restrict__ geom,
                                   const DevicePrimRef* __restrict__ prims,
                                   const DevicePrimContrib* __restrict__ contribs,
                                   const double* __restrict__ acc, int nprod,
                                   int nb, double w_grid,
                                   double* __restrict__ V) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    for (int ip = 0; ip < nprod; ++ip) {
        const double ig = acc[ip];
        if (ig == 0.0)
            continue;
        const DeviceProductGeom g = geom[ip];
        const DevicePrimRef A = prims[g.ua];
        const DevicePrimRef B = prims[g.ub];
        const double sw = g.k * w_grid * ig;
        for (int i = 0; i < A.cn; ++i) {
            const DevicePrimContrib ca = contribs[A.cbeg + i];
            const double cas = ca.coeff * sw;
            for (int j = 0; j < B.cn; ++j) {
                const DevicePrimContrib cb = contribs[B.cbeg + j];
                // V(μ,ν) column-major index μ + ν·nb
                V[ca.mu + cb.mu * nb] += cas * cb.coeff;
            }
        }
    }
}

/// @brief Block-reduce Σ P_i M_i into d_out (atomic).
__global__ void frobenius_kernel(const double* __restrict__ P,
                                 const double* __restrict__ M, long n,
                                 double* __restrict__ out) {
    __shared__ double s[256];
    const long i =
        static_cast<long>(blockIdx.x) * blockDim.x + static_cast<long>(threadIdx.x);
    double v = 0.0;
    if (i < n)
        v = P[i] * M[i];
    s[threadIdx.x] = v;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out, s[0]);
}

constexpr int kBlock = 256;

} // namespace

void ProductWeightPackDevice::build(cudaStream_t stream, int nb, int nprim,
                                    int ncontrib) {
    destroy();
    if (nb < 1)
        throw std::invalid_argument(
            "ProductWeightPackDevice::build: nb must be >= 1");
    if (nprim < 1 || ncontrib < 1)
        throw std::invalid_argument(
            "ProductWeightPackDevice::build: empty prim/contrib tables");
    stream_ = stream;
    nb_ = nb;
    nprim_ = nprim;
    ncontrib_ = ncontrib;
    check_cuda(cudaMalloc(&d_prims_,
                          static_cast<size_t>(nprim) * sizeof(DevicePrimRef)),
               "malloc d_prims");
    check_cuda(cudaMalloc(&d_contribs_, static_cast<size_t>(ncontrib)
                                            * sizeof(DevicePrimContrib)),
               "malloc d_contribs");
    check_cuda(cudaMalloc(&d_V_,
                          static_cast<size_t>(nb) * nb * sizeof(double)),
               "malloc d_V");
    check_cuda(cudaMalloc(&d_reduce_, sizeof(double)), "malloc d_reduce");
    built_ = true;
}

void ProductWeightPackDevice::upload_tables(
    const DevicePrimRef* prims_host, int nprim,
    const DevicePrimContrib* contribs_host, int ncontrib) {
    ensure_built_();
    if (nprim != nprim_ || ncontrib != ncontrib_)
        throw std::invalid_argument(
            "ProductWeightPackDevice::upload_tables: size mismatch");
    check_cuda(cudaMemcpyAsync(d_prims_, prims_host,
                               static_cast<size_t>(nprim) * sizeof(DevicePrimRef),
                               cudaMemcpyHostToDevice, stream_),
               "H2D prims");
    check_cuda(cudaMemcpyAsync(d_contribs_, contribs_host,
                               static_cast<size_t>(ncontrib)
                                   * sizeof(DevicePrimContrib),
                               cudaMemcpyHostToDevice, stream_),
               "H2D contribs");
}

void ProductWeightPackDevice::pack_weights(const double* d_P,
                                           const DeviceProductGeom* d_geom,
                                           DeviceProduct* d_prods, int nprod,
                                           double eps_drop) {
    ensure_built_();
    if (nprod <= 0 || d_P == nullptr || d_geom == nullptr || d_prods == nullptr)
        return;
    const int blocks = (nprod + kBlock - 1) / kBlock;
    pack_weights_kernel<<<blocks, kBlock, 0, stream_>>>(
        d_P, d_geom, d_prims_, d_contribs_, d_prods, nprod, nb_, eps_drop);
    check_cuda(cudaGetLastError(), "pack_weights_kernel");
}

void ProductWeightPackDevice::apply_group_max_mag(DeviceProduct* d_prods,
                                                  const int* d_task_beg,
                                                  const int* d_task_end,
                                                  int ntask) {
    ensure_built_();
    if (ntask <= 0 || d_prods == nullptr || d_task_beg == nullptr
        || d_task_end == nullptr)
        return;
    const int blocks = (ntask + kBlock - 1) / kBlock;
    group_max_mag_kernel<<<blocks, kBlock, 0, stream_>>>(
        d_prods, d_task_beg, d_task_end, ntask);
    check_cuda(cudaGetLastError(), "group_max_mag_kernel");
}

void ProductWeightPackDevice::scatter_acc_(const DeviceProductGeom* d_geom,
                                           const double* d_acc, int nprod,
                                           double w_grid, bool zero_first) {
    ensure_built_();
    if (nprod <= 0)
        return;
    const size_t vbytes =
        static_cast<size_t>(nb_) * static_cast<size_t>(nb_) * sizeof(double);
    if (zero_first)
        check_cuda(cudaMemsetAsync(d_V_, 0, vbytes, stream_), "zero d_V");
    // Single-thread sequential twin of host scatter (see kernel comment).
    scatter_acc_kernel<<<1, 1, 0, stream_>>>(
        d_geom, d_prims_, d_contribs_, d_acc, nprod, nb_, w_grid, d_V_);
    check_cuda(cudaGetLastError(), "scatter_acc_kernel");
}

void ProductWeightPackDevice::scatter_acc(const DeviceProductGeom* d_geom,
                                          const double* d_acc, int nprod,
                                          double w_grid) {
    scatter_acc_(d_geom, d_acc, nprod, w_grid, /*zero_first=*/true);
}

void ProductWeightPackDevice::scatter_acc_add(const DeviceProductGeom* d_geom,
                                              const double* d_acc, int nprod,
                                              double w_grid) {
    scatter_acc_(d_geom, d_acc, nprod, w_grid, /*zero_first=*/false);
}

void ProductWeightPackDevice::download_V(double* V_host) const {
    ensure_built_();
    check_cuda(cudaMemcpyAsync(V_host, d_V_,
                               static_cast<size_t>(nb_) * nb_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H V");
    check_cuda(cudaStreamSynchronize(stream_), "sync D2H V");
}

double ProductWeightPackDevice::frobenius(const double* d_P,
                                          const double* d_M) const {
    ensure_built_();
    if (d_P == nullptr || d_M == nullptr)
        return 0.0;
    const long n =
        static_cast<long>(nb_) * static_cast<long>(nb_);
    check_cuda(cudaMemsetAsync(d_reduce_, 0, sizeof(double), stream_),
               "zero d_reduce");
    const int blocks =
        static_cast<int>((n + kBlock - 1) / kBlock);
    frobenius_kernel<<<blocks, kBlock, 0, stream_>>>(d_P, d_M, n, d_reduce_);
    check_cuda(cudaGetLastError(), "frobenius_kernel");
    double h = 0.0;
    check_cuda(cudaMemcpyAsync(&h, d_reduce_, sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H frobenius");
    check_cuda(cudaStreamSynchronize(stream_), "sync frobenius");
    return h;
}

void ProductWeightPackDevice::upload_matrix(const double* M_host,
                                            double* d_M) const {
    ensure_built_();
    check_cuda(cudaMemcpyAsync(d_M, M_host,
                               static_cast<size_t>(nb_) * nb_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D matrix");
}

void ProductWeightPackDevice::destroy() {
    if (d_prims_) {
        cudaFree(d_prims_);
        d_prims_ = nullptr;
    }
    if (d_contribs_) {
        cudaFree(d_contribs_);
        d_contribs_ = nullptr;
    }
    if (d_V_) {
        cudaFree(d_V_);
        d_V_ = nullptr;
    }
    if (d_reduce_) {
        cudaFree(d_reduce_);
        d_reduce_ = nullptr;
    }
    stream_ = nullptr;
    nb_ = 0;
    nprim_ = 0;
    ncontrib_ = 0;
    built_ = false;
}

void ProductWeightPackDevice::ensure_built_() const {
    if (!built_)
        throw std::runtime_error(
            "ProductWeightPackDevice: build() not called");
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
