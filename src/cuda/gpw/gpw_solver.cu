/// @file cuda/gpw/gpw_solver.cu
/// @brief `cuda::gpw::GpwDft` — product slice (Γ RKS/UKS + multi-k RKS DIIS).
///
/// Hybrid composition (stated):
///   * host — Γ STVnl (S/T + KB Vnl via add_nonlocal_kb_k@Γ) / product-list
///     build; multi-k χ^k build (once);
///     multi-k host residual: energy ctrace, payload diag, cold-start;
///     force AO Pulay on Γ free-fn path
///   * device — product collocate + KS + product integrate (mid-SCF):
///     single-grid RKS device product KS (`KsChainDevice` + PBC min-image
///     collocate/integrate — cuda-gpw-product-collocate-pbc) + device
///     H=H0+V / DIIS; multigrid RKS mid-SCF uses **host collint product
///     multigrid** (`multigrid_collocate_rho` + `ks_grid_build_rks` +
///     `multigrid_integrate_veff`) then device H=H0+V / DIIS — Si
///     monom-device multi-level under-corrects vs collint by ~0.4 mHa
///     (cuda-gpw-multigrid-si-energy-path; residual true device monom MG
///     → deeper slug); collinear UKS `KsChainDevice::run_uks`;
///     multi-k **device phased STVnl** via `KpointStvnlDevice`
///     (cuda-gpw-kpoint-device-stvnl; S/T device; Vnl zero or host-KB H2D);
///     multi-k **device χ^k density collocate** via
///     `internal::KpointCollocateDevice` (cuda-gpw-kpoint-device-collocate)
///     then `KsChainDevice::run_from_rho` (ρ resident; no mid-SCF H2D ρ) +
///     **device χ^k V→H_k integrate** via `KpointIntegrateDevice`
///     (cuda-gpw-kpoint-device-integrate; V_eff stays resident mid-SCF) +
///     **device H_k = H0_k + Vmat fuse** (cuda-gpw-kpoint-h-assemble;
///     H0_k = T_k+Vnl_k uploaded once; mid-SCF no host T+Vnl+Vg) +
///     **device H residency into DIIS residual** (cuda-gpw-kpoint-h-residency;
///     mid-SCF no full \(H_k\) D2H→re-H2D for residual; D2D RM→CM ingest) +
///     **multi-k mid-SCF Pulay DIIS + cuSOLVER zheevd + density** via
///     `internal::KpointDiisEigenDevice` (cuda-gpw-kpoint-device-diis);
///     **Γ RKS mid-SCF H=H0+V on device** (cuda-gpw-device-h-assemble; no
///     mid-iter V_AO D2H) + Pulay DIIS + cuSOLVER eigen + density via
///     `internal::AoDiisEigenDevice` (cuda-gpw-device-diis-eigen);
///     **Γ RKS final payload C/ε from device eigen** (cuda-gpw-device-payload;
///     no host re-diag of H at the SCF boundary; D2H C/eps/P/H once);
///     UKS dual α/β still residual host H path (cuda-gpw-uks-device-h);
///     (cuda-gpw-uks-device-diis);
///     **device P→weight + acc→V_AO scatter** via
///     `internal::ProductWeightPackDevice` (cuda-gpw-device-weight-pack);
///     force grid product (V_eff·dχ·Pχ + ionic HF + local-PP HF) on Γ RKS
///     single-grid free-fn path.
/// Final KS still D2Hs V_eff/V_es for force reuse on the free-fn path.
/// Mid-SCF product work runs on a non-default `cudaStream_t` owned for
/// the duration of `run()` (cuda-gpw-nondefault-stream).

#include "atomiverse/cuda/gpw/gpw_solver.cuh"

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/diis.h"
#include "atomiverse/cpu/gpw/internal/gth_integrals.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cpu/linalg/density.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cuda/gpw/internal/ao_diis_eigen.cuh"
#include "atomiverse/cuda/gpw/internal/gpw_forces.cuh"
#include "atomiverse/cuda/gpw/internal/kpoint_collocate.cuh"
#include "atomiverse/cuda/gpw/internal/kpoint_diis_eigen.cuh"
#include "atomiverse/cuda/gpw/internal/kpoint_integrate.cuh"
#include "atomiverse/cuda/gpw/internal/kpoint_stvnl.cuh"
#include "atomiverse/cuda/gpw/internal/ks_chain.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"
#include "atomiverse/cuda/gpw/internal/product_weight_pack.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {
namespace {

using cpu::gpw::CompactBasisGrid;
using cpu::gpw::GridBasisFunction;
using cpu::gpw::GridProduct;
using cpu::gpw::MultiGrid;
using cpu::gpw::Prim;
using cpu::gpw::PrimContrib;
using cpu::gpw::PrimRef;
using cpu::gpw::ProductSet;
using cpu::gpw::UniformGrid;
using cpu::gpw::basis_function_radius;
using cpu::gpw::build_compact_basis_grid;
using cpu::gpw::collocate_density;
using cpu::gpw::integrate_potential;
using cpu::gpw::build_products;
using cpu::gpw::contracted_ST_periodic;
using cpu::gpw::gth_nonlocal;
using cpu::gpw::local_pp_grid;
using cpu::gpw::make_multigrid;
using cpu::gpw::make_uniform_grid;
using cpu::gpw::multigrid_collocate_rho;
using cpu::gpw::multigrid_integrate_veff;
using cpu::gpw::pair_image_counts;
using cpu::linalg::CanonicalOrth;
using cpu::linalg::RealMatrix;
using cpu::linalg::build_density_matrix;
using cpu::linalg::diagonalize_fock;
using internal::AoDiisEigenDevice;
using internal::CollocateGridDesc;
using internal::collocate_integrate_rho;
using internal::DevicePrimContrib;
using internal::DevicePrimRef;
using internal::DeviceProduct;
using internal::DeviceProductGeom;
using internal::KpointCollocateDevice;
using internal::KpointDiisEigenDevice;
using internal::KpointIntegrateDevice;
using internal::KpointStvnlDevice;
using internal::KsChainDevice;
using internal::KsChainEnergies;
using internal::KsMultiGridChainDevice;
using internal::KsXcKind;
using internal::MultigridDesc;
using internal::ProductWeightPackDevice;

constexpr double kBuildScreen = 1.0e-14;
constexpr double kDiisErrTol = 1.0e-5;
/// DIIS subspace reset when energy rises by more than this [Ha]. Device KS
/// residual (cuFFT / tree reductions) can make pure Pulay non-stationary on
/// multi-center systems (water class); clearing history restarts the warm-up
/// path. H₂-class dual-gates stay within stated tols.
constexpr double kDiisAscentResetHa = 1.0e-3;
/// Matches CPU product mass-margin drop (multigrid.cpp); drop monoms below
/// `eps * kProductMassMargin` when packing live density weights.
constexpr double kProductMassMargin = 1.0e-4;

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda::gpw::GpwDft: ") + what
                                 + ": " + cudaGetErrorString(e));
}

/// @brief c[i] = a[i] + b[i] (device; used for H0=T+Vpp and H=H0+V).
__global__ void add_mats_kernel(const double* __restrict__ a,
                                const double* __restrict__ b,
                                double* __restrict__ c, int n) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    if (i < n)
        c[i] = a[i] + b[i];
}

void launch_add_mats(const double* a, const double* b, double* c, int n,
                     cudaStream_t stream) {
    if (n <= 0)
        return;
    constexpr int kB = 256;
    const int grid = (n + kB - 1) / kB;
    add_mats_kernel<<<grid, kB, 0, stream>>>(a, b, c, n);
    check_cuda(cudaGetLastError(), "add_mats_kernel");
}

/// @brief Device Re ⟨P,M⟩ for complex CM: sum_i (Pr Mr + Pi Mi); scalar D2H.
__global__ void re_dot_cm_kernel(const double* __restrict__ P,
                                 const double* __restrict__ M, int n2,
                                 double* __restrict__ out) {
    // P,M as interleaved re/im pairs: length 2*n_complex = 2*(nb*nb).
    // n2 = nb*nb complex elements; storage is 2*n2 doubles.
    __shared__ double s[256];
    double local = 0.0;
    for (int i = static_cast<int>(threadIdx.x); i < n2;
         i += static_cast<int>(blockDim.x)) {
        const double pr = P[2 * i];
        const double pi = P[2 * i + 1];
        const double mr = M[2 * i];
        const double mi = M[2 * i + 1];
        local += pr * mr + pi * mi;
    }
    s[threadIdx.x] = local;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0;
         stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *out = s[0];
}

void launch_re_dot_cm(const double* d_P, const double* d_M, int n_complex,
                      double* d_out, cudaStream_t stream) {
    if (n_complex <= 0) {
        check_cuda(cudaMemsetAsync(d_out, 0, sizeof(double), stream),
                   "zero re_dot");
        return;
    }
    re_dot_cm_kernel<<<1, 256, 0, stream>>>(d_P, d_M, n_complex, d_out);
    check_cuda(cudaGetLastError(), "re_dot_cm_kernel");
}

/// Non-default SCF stream for product mid-SCF (cuda-gpw-nondefault-stream).
/// Destroyed after the last device use in `run` (fences this stream only).
struct ScfStream {
    cudaStream_t s = nullptr;
    ScfStream() {
        check_cuda(cudaStreamCreate(&s), "cudaStreamCreate SCF");
    }
    ~ScfStream() {
        if (s) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
            s = nullptr;
        }
    }
    ScfStream(const ScfStream&) = delete;
    ScfStream& operator=(const ScfStream&) = delete;
    operator cudaStream_t() const { return s; }
};


std::vector<GridBasisFunction> flatten_basis(const std::vector<GpwAtom>& atoms) {
    std::vector<GridBasisFunction> basis;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            basis.push_back(bf);
    return basis;
}

double pair_overlap_bound(const GridBasisFunction& a,
                          const GridBasisFunction& b) {
    double best = 0.0;
    for (const auto& pa : a) {
        for (const auto& pb : b) {
            const double p = pa.alpha + pb.alpha;
            if (!(p > 0.0))
                continue;
            double dx = pa.center[0] - pb.center[0];
            double dy = pa.center[1] - pb.center[1];
            double dz = pa.center[2] - pb.center[2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double mu = pa.alpha * pb.alpha / p;
            const double mag =
                std::abs(pa.coeff * pb.coeff) * std::exp(-mu * r2);
            best = std::max(best, mag);
        }
    }
    return best;
}

// Analytic core self-energy of one normalized Gaussian core: z²/(2 rloc √π).
double core_self_energy(const GpwAtom& a) {
    return a.z_val * a.z_val / (2.0 * a.rloc * std::sqrt(M_PI));
}

// Ionic core density n_ion(r) = Σ_A z_A (2π rloc²)^{-3/2} exp(-|r-R|²/(2 rloc²)).
std::vector<double> ionic_density(const UniformGrid& grid,
                                  const std::vector<GpwAtom>& atoms,
                                  bool periodic) {
    const long ng = grid.npts();
    std::vector<double> n(static_cast<std::size_t>(ng), 0.0);
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        double s = 0.0;
        for (const auto& a : atoms) {
            const double s2 = a.rloc * a.rloc;
            const double norm = a.z_val * std::pow(2.0 * M_PI * s2, -1.5);
            double dx = r[0] - a.pos[0], dy = r[1] - a.pos[1],
                   dz = r[2] - a.pos[2];
            if (periodic) {
                dx -= grid.L[0] * std::nearbyint(dx / grid.L[0]);
                dy -= grid.L[1] * std::nearbyint(dy / grid.L[1]);
                dz -= grid.L[2] * std::nearbyint(dz / grid.L[2]);
            }
            s += norm * std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * s2));
        }
        n[static_cast<std::size_t>(g)] = s;
    }
    return n;
}

// Γ nonlocal GTH onto Vpp when local_pp_on_grid (local SR is on the grid).
//
// Uses the same separable Kleinman–Bylander assembly as CPU Γ
// (`add_nonlocal_kb`) via the shared multi-k path at κ=0. The previous
// per-pair `gth_nonlocal` form only lattice-imaged the ket and missed
// bra-image / Bloch projector sums — H₂ (empty NL) and large-cell water
// hid it; Si-class GTH nonlocal first-iter E was ~2× CPU (tip GH200
// 1144059 / 1144870; primary residual for product-si-gate +
// multigrid-si-energy-path).
void add_nonlocal_contracted(RealMatrix& Vpp,
                             const std::vector<GridBasisFunction>& basis,
                             const std::vector<GpwAtom>& atoms,
                             const std::vector<double>& radii,
                             const double cell[3], const int mimg[3]) {
    bool any_nl = false;
    for (const auto& at : atoms)
        if (!at.nonlocal.empty()) {
            any_nl = true;
            break;
        }
    if (!any_nl)
        return;

    const int nb = static_cast<int>(basis.size());
    cpu::gpw::internal::KPoint g{};
    g.frac[0] = g.frac[1] = g.frac[2] = 0.0;
    g.weight = 1.0;
    const std::vector<cpu::gpw::internal::KPoint> kpts{g};
    using Cplx = std::complex<double>;
    using CplxMatrix = cpu::gpw::internal::CplxMatrix;
    std::vector<CplxMatrix> V_k(1, xt::zeros<Cplx>({nb, nb}));
    cpu::gpw::internal::add_nonlocal_kb_k(V_k, basis, atoms, radii, cell, mimg,
                                          kBuildScreen, kpts);
    const CplxMatrix& Vk = V_k[0];
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < nb; ++j) {
            // Γ: Hermitian real; imag part is numerical noise only.
            Vpp(i, j) += Vk(i, j).real();
        }
    }
}

void build_STVpp(const std::vector<GpwAtom>& atoms,
                 const std::vector<GridBasisFunction>& basis,
                 const std::vector<double>& radii, const double cell[3],
                 bool periodic, bool grid_loc, const int mimg[3], RealMatrix& S,
                 RealMatrix& T, RealMatrix& Vpp) {
    const int nb = static_cast<int>(basis.size());
    S = xt::zeros<double>({nb, nb});
    T = xt::zeros<double>({nb, nb});
    Vpp = xt::zeros<double>({nb, nb});
    for (int i = 0; i < nb; ++i) {
        if (basis[static_cast<std::size_t>(i)].empty())
            continue;
        for (int j = i; j < nb; ++j) {
            if (basis[static_cast<std::size_t>(j)].empty())
                continue;
            if (pair_overlap_bound(basis[static_cast<std::size_t>(i)],
                                   basis[static_cast<std::size_t>(j)])
                < kBuildScreen)
                continue;
            const double reach = radii[static_cast<std::size_t>(i)]
                                 + radii[static_cast<std::size_t>(j)];
            double s_ij = 0.0, t_ij = 0.0;
            contracted_ST_periodic(basis[static_cast<std::size_t>(i)],
                                   basis[static_cast<std::size_t>(j)], cell,
                                   mimg, reach, s_ij, t_ij);
            S(i, j) = s_ij;
            T(i, j) = t_ij;
            if (i != j) {
                S(j, i) = s_ij;
                T(j, i) = t_ij;
            }
        }
    }
    if (grid_loc)
        add_nonlocal_contracted(Vpp, basis, atoms, radii, cell, mimg);
    // Analytic local SR when !grid_loc is not offered on the device first
    // slice (fail-closed in validate); local is always on the grid.
    (void)grid_loc;
    (void)periodic;
}

std::vector<double> mat_to_vec(const RealMatrix& m, int n) {
    std::vector<double> v(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            v[static_cast<std::size_t>(i) * n + j] = m(i, j);
    return v;
}

double trace_product(const RealMatrix& P, const RealMatrix& M, int n) {
    double t = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            t += P(i, j) * M(i, j);
    return t;
}

// Γ RKS final payload uses AoDiisEigenDevice::eigen_from_device (device-payload).
// Host re-diag store_converged_state removed for the RKS product path.

void store_converged_state_uks(GpwResult& res, const RealMatrix& Ha,
                               const RealMatrix& Hb, const RealMatrix& X,
                               const RealMatrix& S, int n_alpha, int n_beta,
                               int nb) {
    auto [eiga, Ca] = diagonalize_fock(Ha, X);
    auto [eigb, Cb] = diagonalize_fock(Hb, X);
    res.C = Ca;
    res.C_beta = Cb;
    res.eps.assign(eiga.begin(), eiga.end());
    res.eps_beta.assign(eigb.begin(), eigb.end());
    res.P = build_density_matrix(Ca, n_alpha, nb, 1.0);
    res.P_beta = build_density_matrix(Cb, n_beta, nb, 1.0);
    res.H_ks = Ha;
    res.H_ks_beta = Hb;

    const double sz = 0.5 * static_cast<double>(n_alpha - n_beta);
    double ov = 0.0;
    if (n_alpha > 0 && n_beta > 0) {
        const RealMatrix Ca_occ =
            xt::view(Ca, xt::all(), xt::range(0, n_alpha));
        const RealMatrix Cb_occ = xt::view(Cb, xt::all(), xt::range(0, n_beta));
        const RealMatrix Sab =
            xt::linalg::dot(xt::transpose(Ca_occ), xt::linalg::dot(S, Cb_occ));
        for (int i = 0; i < n_alpha; ++i)
            for (int j = 0; j < n_beta; ++j) {
                const double s = Sab(i, j);
                ov += s * s;
            }
    }
    res.s2 = sz * (sz + 1.0) + static_cast<double>(n_beta) - ov;
}

CollocateGridDesc to_coll(const UniformGrid& g) {
    CollocateGridDesc d;
    for (int i = 0; i < 3; ++i) {
        d.n[i] = g.n[i];
        d.L[i] = g.L[i];
        d.origin[i] = g.origin[i];
    }
    // CUDA GPW product rejects nonperiodic; collocate/integrate always
    // sphere-window PBC (cuda-gpw-product-collocate-pbc).
    d.periodic = true;
    return d;
}

KsXcKind to_ks_xc(XcKind xc) {
    if (xc == XcKind::PADE)
        return KsXcKind::PADE;
    if (xc == XcKind::PBE)
        return KsXcKind::PBE;
    throw std::invalid_argument(
        "cuda::gpw::GpwDft: xc has no device path — only XcKind::PADE and "
        "XcKind::PBE. Use cpu::gpw::GpwDft for other functionals.");
}

struct DeviceKsBuffers {
    double* d_n_ion = nullptr;
    double* d_v_loc = nullptr;
    DeviceProduct* d_prods = nullptr;   ///< RKS products / UKS α
    DeviceProduct* d_prods_b = nullptr; ///< UKS β products
    double* d_acc = nullptr;            ///< RKS acc / UKS α
    double* d_acc_b = nullptr;          ///< UKS β acc
    int nprod_cap = 0;
    long npts = 0;

    void alloc_grids(long n) {
        free_grids();
        npts = n;
        const size_t bytes = static_cast<size_t>(n) * sizeof(double);
        check_cuda(cudaMalloc(&d_n_ion, bytes), "malloc d_n_ion");
        check_cuda(cudaMalloc(&d_v_loc, bytes), "malloc d_v_loc");
    }
    void ensure_prods(int nprod, bool uks = false) {
        if (nprod <= nprod_cap && (!uks || d_prods_b != nullptr))
            return;
        free_prods();
        nprod_cap = nprod;
        if (nprod <= 0)
            return;
        check_cuda(cudaMalloc(&d_prods,
                              static_cast<size_t>(nprod) * sizeof(DeviceProduct)),
                   "malloc d_prods");
        check_cuda(cudaMalloc(&d_acc,
                              static_cast<size_t>(nprod) * sizeof(double)),
                   "malloc d_acc");
        if (uks) {
            check_cuda(cudaMalloc(&d_prods_b,
                                  static_cast<size_t>(nprod)
                                      * sizeof(DeviceProduct)),
                       "malloc d_prods_b");
            check_cuda(cudaMalloc(&d_acc_b,
                                  static_cast<size_t>(nprod) * sizeof(double)),
                       "malloc d_acc_b");
        }
    }
    void free_grids() {
        if (d_n_ion)
            cudaFree(d_n_ion);
        if (d_v_loc)
            cudaFree(d_v_loc);
        d_n_ion = d_v_loc = nullptr;
        npts = 0;
    }
    void free_prods() {
        if (d_prods)
            cudaFree(d_prods);
        if (d_prods_b)
            cudaFree(d_prods_b);
        if (d_acc)
            cudaFree(d_acc);
        if (d_acc_b)
            cudaFree(d_acc_b);
        d_prods = d_prods_b = nullptr;
        d_acc = d_acc_b = nullptr;
        nprod_cap = 0;
    }
    void free() {
        free_grids();
        free_prods();
    }
    ~DeviceKsBuffers() { free(); }
};

/// Per-level product / integrate buffers for multigrid product energy.
struct DeviceMgProdBuffers {
    std::vector<DeviceProduct*> d_prods;
    std::vector<double*> d_acc;
    std::vector<int> nprod_cap;

    void resize_levels(int nlev) {
        free_all();
        d_prods.assign(static_cast<std::size_t>(nlev), nullptr);
        d_acc.assign(static_cast<std::size_t>(nlev), nullptr);
        nprod_cap.assign(static_cast<std::size_t>(nlev), 0);
    }
    void ensure_level(int l, int nprod) {
        const std::size_t il = static_cast<std::size_t>(l);
        if (nprod <= nprod_cap[il])
            return;
        if (d_prods[il])
            cudaFree(d_prods[il]);
        if (d_acc[il])
            cudaFree(d_acc[il]);
        d_prods[il] = nullptr;
        d_acc[il] = nullptr;
        nprod_cap[il] = nprod;
        if (nprod <= 0)
            return;
        check_cuda(cudaMalloc(&d_prods[il],
                              static_cast<size_t>(nprod) * sizeof(DeviceProduct)),
                   "malloc d_prods_level");
        check_cuda(cudaMalloc(&d_acc[il],
                              static_cast<size_t>(nprod) * sizeof(double)),
                   "malloc d_acc_level");
    }
    void free_all() {
        for (DeviceProduct* p : d_prods)
            if (p)
                cudaFree(p);
        for (double* p : d_acc)
            if (p)
                cudaFree(p);
        d_prods.clear();
        d_acc.clear();
        nprod_cap.clear();
    }
    ~DeviceMgProdBuffers() { free_all(); }
};

/// Host GridProduct → device geometry meta (weight filled each SCF step).
DeviceProductGeom to_device_geom(const GridProduct& prod) {
    DeviceProductGeom g;
    g.p = prod.p;
    g.ua = prod.ua;
    g.ub = prod.ub;
    g.k = prod.k;
    for (int d = 0; d < 3; ++d) {
        g.pc[d] = prod.pc[d];
        g.ra[d] = prod.ra[d];
        g.rb[d] = prod.rb[d];
        g.la[d] = prod.la[d];
        g.lb[d] = prod.lb[d];
    }
    return g;
}

/// RAII device buffer (move-only; typed via as<T>()).
struct DeviceBuf {
    void* p = nullptr;
    DeviceBuf() = default;
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    DeviceBuf(DeviceBuf&& o) noexcept : p(o.p) { o.p = nullptr; }
    DeviceBuf& operator=(DeviceBuf&& o) noexcept {
        if (this != &o) {
            if (p)
                cudaFree(p);
            p = o.p;
            o.p = nullptr;
        }
        return *this;
    }
    ~DeviceBuf() {
        if (p)
            cudaFree(p);
    }
    template <typename T>
    T* as() {
        return static_cast<T*>(p);
    }
    template <typename T>
    const T* as() const {
        return static_cast<const T*>(p);
    }
};

// ── Multi-k RKS DIIS (STVnl + collocate + integrate + H assemble + DIIS) ──
// Device phased STVnl once via KpointStvnlDevice
// (cuda-gpw-kpoint-device-stvnl); host χ^k build once; device mid-SCF density
// collocate via KpointCollocateDevice; device Poisson+XC via
// KsChainDevice::run_from_rho; device V→Vmat + H_k=H0+Vmat fuse via
// KpointIntegrateDevice (cuda-gpw-kpoint-h-assemble); device H residual path
// (no mid-SCF full H_k D2H) via ingest_H_k_device_rm + residual_from_device
// (cuda-gpw-kpoint-h-residency); device multi-k Pulay DIIS + zheevd + density
// via KpointDiisEigenDevice (cuda-gpw-kpoint-device-diis). Fixed occupations,
// single-grid, no smear / OT / MG / UKS.

using cpu::gpw::internal::CplxMatrix;
using cpu::gpw::internal::KOrth;
using cpu::gpw::internal::KPoint;
using cpu::gpw::internal::KpointBasisGrid;
using Cplx = std::complex<double>;

std::vector<KPoint> sampled_kpoints_cuda(const GpwDft& cfg) {
    std::vector<KPoint> kpts;
    if (!cfg.k_points.empty()) {
        for (std::size_t i = 0; i < cfg.k_points.size(); ++i) {
            KPoint kp;
            kp.frac[0] = cfg.k_points[i][0];
            kp.frac[1] = cfg.k_points[i][1];
            kp.frac[2] = cfg.k_points[i][2];
            kp.weight = cfg.k_point_weights[i];
            kpts.push_back(kp);
        }
        return kpts;
    }
    return cpu::gpw::internal::monkhorst_pack(cfg.k_mesh, cfg.k_time_reversal);
}

CplxMatrix density_from_fixed_k(const CplxMatrix& C, int n_occ, int nb) {
    CplxMatrix P = xt::zeros<Cplx>({nb, nb});
    const int m = static_cast<int>(C.shape(1));
    const int nfill = std::min(m, n_occ);
    for (int o = 0; o < nfill; ++o) {
        for (int i = 0; i < nb; ++i) {
            const Cplx ci = 2.0 * C(i, o);
            for (int j = 0; j < nb; ++j)
                P(i, j) += ci * std::conj(C(j, o));
        }
    }
    return P;
}

double ctrace_k(const CplxMatrix& P, const CplxMatrix& M, int nb) {
    double s = 0.0;
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < nb; ++j)
            s += (P(i, j) * M(j, i)).real();
    return s;
}

/// Multi-k closed-shell RKS DIIS energy (device ρ collocate + KS + V→H_k).
GpwResult run_cuda_kpoints_rks(const std::vector<GpwAtom>& atoms,
                               const GpwDft& cfg) {
    const std::vector<GridBasisFunction> basis = flatten_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    if (nb < 1)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft::run: empty AO basis (multi-k)");

    std::vector<double> radii(static_cast<std::size_t>(nb), 0.0);
    double rmax = 0.0;
    for (int i = 0; i < nb; ++i) {
        radii[static_cast<std::size_t>(i)] =
            basis_function_radius(basis[static_cast<std::size_t>(i)],
                                  cfg.eps_default);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(i)]);
    }
    int mimg[3] = {0, 0, 0};
    pair_image_counts(cfg.cell, /*periodic=*/true, 2.0 * rmax, mimg);

    const std::vector<KPoint> kpts = sampled_kpoints_cuda(cfg);
    const int nk = static_cast<int>(kpts.size());
    if (nk < 1)
        throw std::runtime_error(
            "cuda::gpw::GpwDft: multi-k sampling produced zero k-points");

    std::vector<double> weights(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik)
        weights[static_cast<std::size_t>(ik)] =
            kpts[static_cast<std::size_t>(ik)].weight;

    // Phased analytic STVnl on device (cuda-gpw-kpoint-device-stvnl). D2H for
    // host Löwdin orth / energy ctrace residual; H0 uploaded to integrate.
    std::vector<double> k_frac_flat(static_cast<std::size_t>(nk) * 3);
    for (int ik = 0; ik < nk; ++ik) {
        k_frac_flat[static_cast<std::size_t>(ik) * 3 + 0] =
            kpts[static_cast<std::size_t>(ik)].frac[0];
        k_frac_flat[static_cast<std::size_t>(ik) * 3 + 1] =
            kpts[static_cast<std::size_t>(ik)].frac[1];
        k_frac_flat[static_cast<std::size_t>(ik) * 3 + 2] =
            kpts[static_cast<std::size_t>(ik)].frac[2];
    }
    KpointStvnlDevice stvnl;
    // Stream not yet created; null stream for setup (geometry-fixed once).
    stvnl.compute(/*stream=*/nullptr, basis, radii, atoms, cfg.cell, mimg,
                  kBuildScreen, k_frac_flat.data(), nk);
    std::vector<Cplx> S_flat(static_cast<std::size_t>(nk)
                             * static_cast<std::size_t>(nb)
                             * static_cast<std::size_t>(nb));
    std::vector<Cplx> T_flat(S_flat.size());
    std::vector<Cplx> Vnl_flat(S_flat.size());
    stvnl.download_S(S_flat.data());
    stvnl.download_T(T_flat.data());
    stvnl.download_Vnl(Vnl_flat.data());
    std::vector<CplxMatrix> S_k(static_cast<std::size_t>(nk));
    std::vector<CplxMatrix> T_k(static_cast<std::size_t>(nk));
    std::vector<CplxMatrix> Vnl_k(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik) {
        S_k[static_cast<std::size_t>(ik)] = xt::zeros<Cplx>({nb, nb});
        T_k[static_cast<std::size_t>(ik)] = xt::zeros<Cplx>({nb, nb});
        Vnl_k[static_cast<std::size_t>(ik)] = xt::zeros<Cplx>({nb, nb});
        for (int u = 0; u < nb; ++u)
            for (int t = 0; t < nb; ++t) {
                const std::size_t idx =
                    static_cast<std::size_t>(ik) * nb * nb
                    + static_cast<std::size_t>(u) * nb
                    + static_cast<std::size_t>(t);
                S_k[static_cast<std::size_t>(ik)](u, t) = S_flat[idx];
                T_k[static_cast<std::size_t>(ik)](u, t) = T_flat[idx];
                Vnl_k[static_cast<std::size_t>(ik)](u, t) = Vnl_flat[idx];
            }
    }

    std::vector<KOrth> orth;
    orth.reserve(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik) {
        orth.emplace_back(S_k[static_cast<std::size_t>(ik)],
                          cfg.lindep_threshold);
        const int m = static_cast<int>(orth.back().X.shape(1));
        if (cfg.n_occ > m)
            throw std::runtime_error(
                "cuda::gpw::GpwDft multi-k: n_occ=" + std::to_string(cfg.n_occ)
                + " exceeds the " + std::to_string(m)
                + " independent basis directions at k #" + std::to_string(ik));
    }

    // Single-grid density grid + ion / local-PP (same Γ setup).
    const MultiGrid mg = make_multigrid(cfg.cell, cfg.cutoff_ry, /*nlev=*/1,
                                        cfg.cutoff_ry);
    if (mg.levels.empty())
        throw std::runtime_error(
            "cuda::gpw::GpwDft multi-k: make_multigrid returned empty levels");
    const UniformGrid& grid = mg.levels[0];
    for (int d = 0; d < 3; ++d)
        if (grid.n[d] % 2 != 0)
            throw std::runtime_error(
                "cuda::gpw::GpwDft multi-k: grid axis " + std::to_string(d)
                + " has odd n=" + std::to_string(grid.n[d])
                + " (device Poisson requires even n)");

    const std::vector<double> n_ion =
        ionic_density(grid, atoms, /*periodic=*/true);
    double e_self = 0.0;
    for (const auto& a : atoms)
        e_self += core_self_energy(a);
    const std::vector<double> v_locsr =
        local_pp_grid(grid, atoms, /*periodic=*/true);

    const KpointBasisGrid kbg = cpu::gpw::internal::build_kpoint_basis_grid(
        grid, basis, kpts, cfg.eps_default);

    // Cold-start densities from H0_k = T_k + Vnl_k.
    std::vector<CplxMatrix> P_k(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik) {
        const CplxMatrix H0k = T_k[static_cast<std::size_t>(ik)]
                               + Vnl_k[static_cast<std::size_t>(ik)];
        auto [eps0, C0k] = cpu::gpw::internal::diagonalize_k(
            H0k, orth[static_cast<std::size_t>(ik)]);
        (void)eps0;
        P_k[static_cast<std::size_t>(ik)] =
            density_from_fixed_k(C0k, cfg.n_occ, nb);
    }

    ScfStream scf_stream;
    const cudaStream_t stream = scf_stream;
    const long npts = grid.npts();
    const size_t bytes = static_cast<size_t>(npts) * sizeof(double);
    DeviceKsBuffers dev;
    dev.alloc_grids(npts);
    check_cuda(cudaMemcpyAsync(dev.d_n_ion, n_ion.data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "H2D n_ion multi-k");
    check_cuda(cudaMemcpyAsync(dev.d_v_loc, v_locsr.data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "H2D v_loc multi-k");
    double* d_rho = nullptr;
    check_cuda(cudaMalloc(&d_rho, bytes), "malloc d_rho multi-k");
    // RAII for d_rho
    struct RhoGuard {
        double* p = nullptr;
        ~RhoGuard() {
            if (p)
                cudaFree(p);
        }
    } rho_guard{d_rho};

    // Device multi-k density collocate (cuda-gpw-kpoint-device-collocate):
    // χ^k uploaded once; each SCF iter only H2Ds small P_k and accumulates ρ
    // on device — no host collocate_density_k / mid-SCF H2D of full ρ.
    KpointCollocateDevice kcoll;
    kcoll.build(stream, nk, nb, npts);
    for (int ik = 0; ik < nk; ++ik) {
        const auto& chi = kbg.chi[static_cast<std::size_t>(ik)];
        if (static_cast<long>(chi.size()) != npts * nb)
            throw std::runtime_error(
                "cuda::gpw::GpwDft multi-k: chi size mismatch at k #"
                + std::to_string(ik));
        kcoll.upload_chi(ik, chi.data());
    }

    KsChainDevice ks(to_coll(grid), stream);
    const KsXcKind kxc = to_ks_xc(cfg.xc);
    const bool use_diis = cfg.diis_depth > 0;

    // Device χ^k integrate + H_k assemble: χ and H0_k = T_k+Vnl_k uploaded
    // once; mid-SCF keeps V_eff resident and fuses H_k on device
    // (cuda-gpw-kpoint-h-assemble). Residual path keeps H device-resident
    // (cuda-gpw-kpoint-h-residency); payload still D2Hs H for host re-diag.
    KpointIntegrateDevice kint;
    kint.build(stream, nb, npts, nk);
    for (int ik = 0; ik < nk; ++ik) {
        kint.upload_chi(ik, kbg.chi[static_cast<std::size_t>(ik)].data());
        const CplxMatrix H0k = T_k[static_cast<std::size_t>(ik)]
                               + Vnl_k[static_cast<std::size_t>(ik)];
        kint.upload_H0(ik, H0k.data());
    }
    check_cuda(cudaStreamSynchronize(stream), "sync multi-k chi/H0 H2D");

    // Multi-k device Pulay DIIS + cuSOLVER zheevd + density
    // (cuda-gpw-kpoint-device-diis). S/X uploaded once; P refreshed each iter
    // (P residency for collocate is a deeper slug).
    int max_m = 0;
    for (int ik = 0; ik < nk; ++ik)
        max_m = std::max(max_m,
                         static_cast<int>(orth[static_cast<std::size_t>(ik)]
                                              .X.shape(1)));
    KpointDiisEigenDevice kdiis;
    kdiis.build(stream, nk, nb, max_m, use_diis ? cfg.diis_depth : 0,
                cfg.n_occ, /*occ_factor=*/2.0, weights);
    kdiis.reset();
    for (int ik = 0; ik < nk; ++ik) {
        const int mk =
            static_cast<int>(orth[static_cast<std::size_t>(ik)].X.shape(1));
        kdiis.upload_SX_k(ik, S_k[static_cast<std::size_t>(ik)].data(),
                          orth[static_cast<std::size_t>(ik)].X.data(), mk);
        kdiis.upload_P_k(ik, P_k[static_cast<std::size_t>(ik)].data());
    }

    // Mid-SCF: integrate + fuse H on device, D2D-ingest into DIIS (no H D2H).
    auto assemble_Hk_into_diis = [&](int ik) {
        kint.integrate(ks.d_veff(), kbg.weight, ik);
        kint.assemble_H(ik);
        kdiis.ingest_H_k_device_rm(ik, kint.d_H());
    };
    // Payload / host re-diag: same fuse then D2H assembled H (child
    // kpoint-payload-device drops this path later).
    auto assemble_Hk_host = [&](int ik) -> CplxMatrix {
        kint.integrate(ks.d_veff(), kbg.weight, ik);
        kint.assemble_H(ik);
        CplxMatrix Hk = xt::zeros<Cplx>({nb, nb});
        kint.download_H(Hk.data());
        return Hk;
    };

    GpwResult res;
    res.n_kpts = nk;
    res.n_kpts_local = nk;
    res.scf_kbg = std::make_shared<KpointBasisGrid>(kbg);
    res.scf_cbg_valid = false;

    std::vector<double> last_veff;
    std::vector<double> last_v_es;


    // Geometry-fixed T_k / Vnl_k on device (CM) for mid-SCF Re⟨P,M⟩ (no P D2H).
    std::vector<DeviceBuf> t_k_bufs(static_cast<std::size_t>(nk));
    std::vector<DeviceBuf> vnl_k_bufs(static_cast<std::size_t>(nk));
    std::vector<double*> d_T_k(static_cast<std::size_t>(nk), nullptr);
    std::vector<double*> d_Vnl_k(static_cast<std::size_t>(nk), nullptr);
    DeviceBuf re_dot_buf;
    check_cuda(cudaMalloc(&re_dot_buf.p, sizeof(double)), "malloc re_dot");
    double* d_re_dot = re_dot_buf.as<double>();
    auto upload_rm_to_cm = [&](const CplxMatrix& M, double* d_cm) {
        std::vector<Cplx> cm(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb));
        for (int j = 0; j < nb; ++j)
            for (int i = 0; i < nb; ++i)
                cm[static_cast<std::size_t>(j) * nb + static_cast<std::size_t>(i)] =
                    M(i, j);
        check_cuda(cudaMemcpyAsync(d_cm, cm.data(),
                                   static_cast<size_t>(nb) * nb * sizeof(Cplx),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D T/Vnl cm setup");
    };
    for (int ik = 0; ik < nk; ++ik) {
        check_cuda(cudaMalloc(&t_k_bufs[static_cast<std::size_t>(ik)].p,
                              static_cast<size_t>(nb) * nb * sizeof(Cplx)),
                   "malloc d_T_k");
        check_cuda(cudaMalloc(&vnl_k_bufs[static_cast<std::size_t>(ik)].p,
                              static_cast<size_t>(nb) * nb * sizeof(Cplx)),
                   "malloc d_Vnl_k");
        d_T_k[static_cast<std::size_t>(ik)] =
            t_k_bufs[static_cast<std::size_t>(ik)].as<double>();
        d_Vnl_k[static_cast<std::size_t>(ik)] =
            vnl_k_bufs[static_cast<std::size_t>(ik)].as<double>();
        upload_rm_to_cm(T_k[static_cast<std::size_t>(ik)],
                        d_T_k[static_cast<std::size_t>(ik)]);
        upload_rm_to_cm(Vnl_k[static_cast<std::size_t>(ik)],
                        d_Vnl_k[static_cast<std::size_t>(ik)]);
    }
    check_cuda(cudaStreamSynchronize(stream), "sync T/Vnl k setup H2D");

    auto re_frobenius_cm = [&](const void* d_P_cm, const double* d_M_cm) -> double {
        // Device Re sum_i (P_i conj? ) — Hermitian: sum (Pr*Mr + Pi*Mi).
        launch_re_dot_cm(static_cast<const double*>(d_P_cm), d_M_cm, nb * nb,
                         d_re_dot, stream);
        double h = 0.0;
        check_cuda(cudaMemcpyAsync(&h, d_re_dot, sizeof(double),
                                   cudaMemcpyDeviceToHost, stream),
                   "D2H re_dot scalar");
        check_cuda(cudaStreamSynchronize(stream), "sync re_dot");
        return h;
    };

    auto build_k_energy = [&](bool store_grids) -> double {
        kcoll.zero_rho(d_rho);
        for (int ik = 0; ik < nk; ++ik) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            // CUDA-only: collocate from device CM P (no mid-SCF H2D of P).
            kcoll.collocate_k_device_cm(ik, kdiis.d_P(ik), wk, d_rho);
        }
        const KsChainEnergies e = ks.run_from_rho(
            d_rho, dev.d_n_ion, dev.d_v_loc, /*d_prods=*/nullptr,
            /*nprod=*/0, /*d_acc=*/nullptr, kxc);

        if (store_grids) {
            std::vector<double> veff(static_cast<std::size_t>(npts));
            std::vector<double> v_es(static_cast<std::size_t>(npts));
            check_cuda(cudaMemcpyAsync(veff.data(), ks.d_veff(), bytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "D2H veff multi-k boundary");
            check_cuda(cudaMemcpyAsync(v_es.data(), ks.d_v_es(), bytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "D2H v_es multi-k boundary");
            check_cuda(cudaStreamSynchronize(stream), "sync multi-k final grids");
            last_veff = std::move(veff);
            last_v_es = std::move(v_es);
        }

        double e_kin = 0.0, e_nl = 0.0;
        for (int ik = 0; ik < nk; ++ik) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            assemble_Hk_into_diis(ik);
            e_kin += wk * re_frobenius_cm(kdiis.d_P(ik),
                                          d_T_k[static_cast<std::size_t>(ik)]);
            e_nl += wk * re_frobenius_cm(kdiis.d_P(ik),
                                         d_Vnl_k[static_cast<std::size_t>(ik)]);
        }
        res.e_kinetic = e_kin;
        res.e_nonlocal = 0.0;
        res.e_loc_sr = e_nl + e.e_locgrid;
        res.e_es = e.e_hartree - e_self;
        res.e_xc = e.e_xc;
        if (store_grids) {
            res.veff = last_veff;
            res.v_es = last_v_es;
        }
        return res.e_kinetic + res.e_loc_sr + res.e_es + res.e_xc;
    };

    auto store_payload = [&]() {
        res.k_frac.clear();
        res.k_weights.clear();
        res.eps_k.clear();
        res.C_k.clear();
        res.C_occ_k.clear();
        res.P_k.clear();
        res.C_k.reserve(static_cast<std::size_t>(nk));
        res.C_occ_k.resize(static_cast<std::size_t>(nk));
        res.P_k.resize(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const auto& kp = kpts[static_cast<std::size_t>(ik)];
            res.k_frac.push_back({kp.frac[0], kp.frac[1], kp.frac[2]});
            res.k_weights.push_back(kp.weight);
            const CplxMatrix Hfull = assemble_Hk_host(ik);
            auto [eps, C] = cpu::gpw::internal::diagonalize_k(
                Hfull, orth[static_cast<std::size_t>(ik)]);
            res.eps_k.push_back(eps);
            res.C_k.push_back(cpu::linalg::CplxMatrix(C));
            std::vector<Cplx> occ(static_cast<std::size_t>(nb)
                                  * static_cast<std::size_t>(cfg.n_occ));
            for (int mu = 0; mu < nb; ++mu)
                for (int o = 0; o < cfg.n_occ; ++o)
                    occ[static_cast<std::size_t>(mu) * cfg.n_occ
                        + static_cast<std::size_t>(o)] = C(mu, o);
            res.C_occ_k[static_cast<std::size_t>(ik)] = std::move(occ);
            res.P_k[static_cast<std::size_t>(ik)] =
                density_from_fixed_k(C, cfg.n_occ, nb);
        }
        res.n_elec_occ = 2.0 * static_cast<double>(cfg.n_occ);
    };

    if (cfg.max_iter == 0)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft multi-k: max_iter==0 fixed-density path is not "
            "on the first multi-k product slice — set max_iter>0");
    if (!use_diis)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft multi-k: diis_depth>0 required for CUDA-only "
            "mid-path (linear mix needs host P blend)");

    double prev_energy = std::numeric_limits<double>::quiet_NaN();
    bool conv = false;
    int it = 0;
    for (it = 0; it < cfg.max_iter; ++it) {
        const double e_total = build_k_energy(/*store_grids=*/false);
        res.e_total = e_total;

        const double err_norm = kdiis.residual_from_device();
        const bool has_prev = !std::isnan(prev_energy);
        const double dE = has_prev ? (e_total - prev_energy)
                                   : std::numeric_limits<double>::infinity();
        prev_energy = e_total;
        if (has_prev && std::abs(dE) < cfg.tol && err_norm < kDiisErrTol) {
            conv = true;
            break;
        }

        // Device DIIS + zheevd + density — P stays on device (no mid-SCF D2H).
        kdiis.advance();
    }



    // Final KS at the last density (energy pieces + V_eff for force follow-on).
    {
        const double e_final = build_k_energy(/*store_grids=*/true);
        res.e_total = e_final;
        store_payload();
        res.converged = conv;
        res.iters = conv ? (it + 1) : cfg.max_iter;
    }
    return res;
}

} // namespace

void GpwDft::validate_slice_() const {
    if (is_uks()) {
        if (n_alpha < 0 || n_beta < 0)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: n_alpha and n_beta must be >= 0");
        if (n_alpha + n_beta < 1)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: UKS requires n_alpha+n_beta >= 1");
        // First UKS product cut: Padé LSD only (PBE-UKS → uks-pbe-gate).
        if (xc != XcKind::PADE)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: collinear UKS first cut is XcKind::PADE "
                "only (Padé LSD). Set xc=PADE or use cpu::gpw::GpwDft; "
                "PBE-UKS is a deeper slug (cuda-gpw-uks-pbe-gate).");
        // UKS multigrid is not on the product path (single-grid run_uks only).
        if (ngrids > 1)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: collinear UKS first cut is single-grid "
                "only (ngrids=1). Multigrid UKS is a deeper slug; set "
                "ngrids=1 or use cpu::gpw::GpwDft.");
    } else if (n_occ < 1) {
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: n_occ must be >= 1 for closed-shell RKS (got "
            + std::to_string(n_occ)
            + "), or set n_alpha/n_beta for collinear UKS");
    }
    if (nonperiodic)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: nonperiodic (Martyna–Tuckerman) Poisson has no "
            "device path yet — set nonperiodic=false or use cpu::gpw::GpwDft.");
    if (use_ot)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: OT minimizer is not on the first CUDA product "
            "slice — set use_ot=false (DIIS) or use cpu::gpw::GpwDft.");
    if (ngrids < 1)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: ngrids must be >= 1 (got "
            + std::to_string(ngrids) + ")");
    // True multigrid product energy requires REL_CUTOFF calibration (CPU twin:
    // use_mg = ngrids>1 && rel_cutoff>0). ngrids>1 with rel_cutoff<=0 would
    // silently collapse to a single fine grid in make_multigrid — fail closed.
    if (ngrids > 1 && !(rel_cutoff > 0.0))
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: multigrid product energy requires "
            "rel_cutoff > 0 when ngrids > 1 (got ngrids="
            + std::to_string(ngrids) + ", rel_cutoff="
            + std::to_string(rel_cutoff)
            + "). Set rel_cutoff (CP2K REL_CUTOFF [Ry]) or ngrids=1.");
    // Multi-k RKS DIIS (cuda-gpw-kpoints): single-grid closed-shell only.
    // Joint paths (UKS×k, MG×k) are deeper slugs.
    if (k_points.size() != k_point_weights.size())
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: k_points and k_point_weights must have the "
            "same length (got " + std::to_string(k_points.size()) + " vs "
            + std::to_string(k_point_weights.size()) + ")");
    for (int d = 0; d < 3; ++d)
        if (k_mesh[d] < 1)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: k_mesh subdivision must be >= 1 (axis "
                + std::to_string(d) + " got " + std::to_string(k_mesh[d])
                + ")");
    if (uses_kpath()) {
        if (is_uks())
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: multi-k UKS is not on the device product "
                "path yet — keep k_mesh={1,1,1} / empty k_points for UKS, or "
                "use cpu::gpw::GpwDft (deeper slug cuda-gpw-uks-kpoints).");
        if (ngrids > 1)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: multi-k multigrid is not on the device "
                "product path yet — set ngrids=1 for multi-k RKS, or use "
                "cpu::gpw::GpwDft (deeper slug cuda-gpw-multigrid-kpoints).");
    }
    if (xc != XcKind::PADE && xc != XcKind::PBE)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: only XcKind::PADE and XcKind::PBE have device "
            "KS paths. Use cpu::gpw::GpwDft for other functionals.");
    if (!local_pp_on_grid)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: local_pp_on_grid=false (analytic local SR) is "
            "not on the first device slice — set local_pp_on_grid=true.");
    if (!(cutoff_ry > 0.0))
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: cutoff_ry must be > 0 [Ry]");
    if (!(tol > 0.0))
        throw std::invalid_argument("cuda::gpw::GpwDft: tol must be > 0");
    if (max_iter < 0)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: max_iter must be >= 0");
    if (diis_depth < 0)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: diis_depth must be >= 0");
    for (int d = 0; d < 3; ++d)
        if (!(cell[d] > 0.0))
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: periodic cell edge " + std::to_string(d)
                + " must be > 0 Bohr (got " + std::to_string(cell[d]) + ")");
}

void GpwDft::build(const std::vector<GpwAtom>& reference) {
    if (reference.empty())
        throw std::invalid_argument(
            "cuda::gpw::GpwDft::build: empty atom set (nothing to solve)");
    validate_slice_();
    built_ = true;
}

GpwResult& GpwDft::run(const std::vector<GpwAtom>& atoms,
                      const RealMatrix* guess_C, const RealMatrix* guess_P,
                      const RealMatrix* guess_P_beta) {
    validate_slice_();
    if (!built_)
        build(atoms);
    if (atoms.empty())
        throw std::invalid_argument(
            "cuda::gpw::GpwDft::run: empty atom set");

    // Multi-k RKS DIIS branch (cuda-gpw-kpoints): host χ^k + device ρ→V_eff.
    // Γ product path (guess_C/P) is not used here — multi-k cold-starts H0.
    if (uses_kpath()) {
        if (guess_C != nullptr || guess_P != nullptr || guess_P_beta != nullptr)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft::run: multi-k path ignores real Γ "
                "guess_C/guess_P; pass nullptr (complex P_k warm-start is a "
                "deeper slug)");
        result_ = run_cuda_kpoints_rks(atoms, *this);
        return result_;
    }

    result_ = GpwResult{};
    const bool periodic = true;
    const double eps = eps_default;
    const bool uks = is_uks();

    const std::vector<GridBasisFunction> basis = flatten_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    if (nb < 1)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft::run: empty AO basis");

    std::vector<double> radii(static_cast<std::size_t>(nb), 0.0);
    double rmax = 0.0;
    for (int i = 0; i < nb; ++i) {
        radii[static_cast<std::size_t>(i)] =
            basis_function_radius(basis[static_cast<std::size_t>(i)], eps);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(i)]);
    }
    int mimg[3] = {0, 0, 0};
    pair_image_counts(cell, periodic, 2.0 * rmax, mimg);

    RealMatrix S, T, Vpp;
    build_STVpp(atoms, basis, radii, cell, periodic, local_pp_on_grid, mimg, S,
                T, Vpp);
    const RealMatrix H0 = T + Vpp;

    // Multigrid product energy when ngrids>1 && rel_cutoff>0 (CPU twin gate).
    // ngrids==1 keeps the single-grid KsChainDevice path (RKS + UKS).
    // UKS + multigrid is rejected in validate_slice_.
    const bool use_mg = ngrids > 1 && rel_cutoff > 0.0;
    const int nlev = use_mg ? ngrids : 1;
    const double rel_for_build = use_mg ? rel_cutoff : cutoff_ry;
    const MultiGrid mg =
        make_multigrid(cell, cutoff_ry, nlev, rel_for_build);
    if (mg.levels.empty())
        throw std::runtime_error(
            "cuda::gpw::GpwDft::run: make_multigrid returned empty levels");
    for (std::size_t l = 0; l < mg.levels.size(); ++l) {
        const UniformGrid& gl = mg.levels[l];
        for (int d = 0; d < 3; ++d)
            if (gl.n[d] % 2 != 0)
                throw std::runtime_error(
                    "cuda::gpw::GpwDft::run: multigrid level "
                    + std::to_string(l) + " axis " + std::to_string(d)
                    + " has odd n=" + std::to_string(gl.n[d])
                    + " (device Poisson / spectral transfer require even n)");
    }
    const UniformGrid& grid = mg.levels[0]; // fine reference for Poisson/XC

    const std::vector<double> n_ion = ionic_density(grid, atoms, periodic);
    double e_self = 0.0;
    for (const auto& a : atoms)
        e_self += core_self_energy(a);
    const std::vector<double> v_locsr =
        local_pp_on_grid ? local_pp_grid(grid, atoms, periodic)
                         : std::vector<double>{};

    // Compact AO grid for force assembly — single-grid only (CPU MG skips cbg;
    // fine-grid cbg at production cutoffs is prohibitive).
    if (!use_mg) {
        CompactBasisGrid cbg =
            build_compact_basis_grid(grid, basis, periodic, eps);
        result_.scf_cbg = std::move(cbg);
        result_.scf_cbg_valid = true;
    } else {
        result_.scf_cbg_valid = false;
    }

    // Geometry-fixed product list (per level under MG). Device collocate /
    // integrate consume DeviceProduct views; P enters only as weights.
    const ProductSet pset =
        build_products(mg, basis, cell, rel_for_build, eps, periodic);
    const int nlevels = static_cast<int>(pset.by_level.size());
    if (nlevels < 1)
        throw std::runtime_error(
            "cuda::gpw::GpwDft::run: empty ProductSet::by_level");
    std::vector<int> nprod_by_level(static_cast<std::size_t>(nlevels), 0);
    int nprod_tot = 0;
    for (int l = 0; l < nlevels; ++l) {
        nprod_by_level[static_cast<std::size_t>(l)] =
            static_cast<int>(pset.by_level[static_cast<std::size_t>(l)].size());
        nprod_tot += nprod_by_level[static_cast<std::size_t>(l)];
    }
    if (nprod_tot < 1)
        throw std::runtime_error(
            "cuda::gpw::GpwDft::run: empty product list after build_products "
            "(basis/radius screening dropped every pair)");
    // Single-grid product count (UKS + RKS ngrids==1); geometry on device.
    const int nprod_geom = nprod_by_level[0];

    CanonicalOrth orth(S, lindep_threshold);
    if (orth.s_min < -1e-8)
        throw std::runtime_error(
            "cuda::gpw::GpwDft: overlap not positive definite (s_min="
            + std::to_string(orth.s_min) + ")");
    const int n_need =
        uks ? std::max(n_alpha, n_beta) : n_occ;
    if (n_need > static_cast<int>(orth.X.shape(1)))
        throw std::runtime_error(
            "cuda::gpw::GpwDft: occupied count exceeds linearly independent "
            "directions");

    RealMatrix C0;
    if (guess_C != nullptr && static_cast<int>(guess_C->shape(0)) == nb
        && static_cast<int>(guess_C->shape(1)) == nb) {
        C0 = *guess_C;
    } else {
        C0 = std::get<1>(diagonalize_fock(H0, orth.X));
    }

    // Non-default stream for product SCF: grid upload + mid-SCF H2D/D2H + KS.
    ScfStream scf_stream;
    const cudaStream_t stream = scf_stream;

    // Device KS workspace: n_ion / v_loc on fine grid once (stream-ordered).
    DeviceKsBuffers dev;
    DeviceMgProdBuffers mg_prods;
    const long npts = grid.npts();
    const double w_fine = grid.weight();
    const double w_grid = w_fine; // UKS single-grid alias
    dev.alloc_grids(npts);
    const size_t bytes = static_cast<size_t>(npts) * sizeof(double);
    check_cuda(cudaMemcpyAsync(dev.d_n_ion, n_ion.data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "H2D n_ion");
    if (!v_locsr.empty())
        check_cuda(cudaMemcpyAsync(dev.d_v_loc, v_locsr.data(), bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "H2D v_loc");
    else
        check_cuda(cudaMemsetAsync(dev.d_v_loc, 0, bytes, stream),
                   "zero v_loc");

    const KsXcKind kxc = to_ks_xc(xc);
    const double* d_vloc_arg = v_locsr.empty() ? nullptr : dev.d_v_loc;
    const double eps_drop = eps * kProductMassMargin;
    const bool use_diis = diis_depth > 0;

    // Single-grid chain (RKS/UKS) vs multi-grid chain (RKS only).
    std::unique_ptr<KsChainDevice> ks_sg;
    std::unique_ptr<KsMultiGridChainDevice> ks_mg;
    MultigridDesc mg_desc;
    if (use_mg) {
        mg_desc.levels.resize(static_cast<std::size_t>(nlevels));
        for (int l = 0; l < nlevels; ++l)
            mg_desc.levels[static_cast<std::size_t>(l)] =
                to_coll(mg.levels[static_cast<std::size_t>(l)]);
        ks_mg = std::make_unique<KsMultiGridChainDevice>(mg_desc, stream);
        mg_prods.resize_levels(nlevels);
        for (int l = 0; l < nlevels; ++l)
            mg_prods.ensure_level(l, nprod_by_level[static_cast<std::size_t>(l)]);
    } else {
        ks_sg = std::make_unique<KsChainDevice>(to_coll(grid), stream);
        dev.ensure_prods(nprod_geom, uks);
    }

    // ── Device weight pack tables (cuda-gpw-device-weight-pack) ───────────
    // Prim/contrib + per-level product geometry stay resident for the run;
    // each SCF iter only refreshes DeviceProduct.weight from d_P and scatters
    // integrate acc → V_AO on device.
    const int nprim = static_cast<int>(pset.prims.size());
    const int ncontrib = static_cast<int>(pset.contribs.size());
    std::vector<DevicePrimRef> h_prims(static_cast<std::size_t>(nprim));
    for (int i = 0; i < nprim; ++i) {
        h_prims[static_cast<std::size_t>(i)].cbeg = pset.prims[static_cast<std::size_t>(i)].cbeg;
        h_prims[static_cast<std::size_t>(i)].cn = pset.prims[static_cast<std::size_t>(i)].cn;
    }
    std::vector<DevicePrimContrib> h_contribs(
        static_cast<std::size_t>(ncontrib));
    for (int i = 0; i < ncontrib; ++i) {
        h_contribs[static_cast<std::size_t>(i)].mu =
            pset.contribs[static_cast<std::size_t>(i)].mu;
        h_contribs[static_cast<std::size_t>(i)].coeff =
            pset.contribs[static_cast<std::size_t>(i)].coeff;
    }

    ProductWeightPackDevice wpack;
    wpack.build(stream, nb, nprim, ncontrib);
    wpack.upload_tables(h_prims.data(), nprim, h_contribs.data(), ncontrib);

    // Per-level geometry on device (once).
    std::vector<DeviceBuf> geom_bufs(static_cast<std::size_t>(nlevels));
    std::vector<DeviceProductGeom*> d_geom_by_level(
        static_cast<std::size_t>(nlevels), nullptr);
    for (int l = 0; l < nlevels; ++l) {
        const int np = nprod_by_level[static_cast<std::size_t>(l)];
        if (np <= 0)
            continue;
        const std::vector<GridProduct>& prods_l =
            pset.by_level[static_cast<std::size_t>(l)];
        std::vector<DeviceProductGeom> h_geom(static_cast<std::size_t>(np));
        for (int ip = 0; ip < np; ++ip)
            h_geom[static_cast<std::size_t>(ip)] =
                to_device_geom(prods_l[static_cast<std::size_t>(ip)]);
        check_cuda(cudaMalloc(&geom_bufs[static_cast<std::size_t>(l)].p,
                              static_cast<size_t>(np) * sizeof(DeviceProductGeom)),
                   "malloc d_geom");
        d_geom_by_level[static_cast<std::size_t>(l)] =
            geom_bufs[static_cast<std::size_t>(l)].as<DeviceProductGeom>();
        check_cuda(cudaMemcpyAsync(d_geom_by_level[static_cast<std::size_t>(l)],
                                   h_geom.data(),
                                   static_cast<size_t>(np)
                                       * sizeof(DeviceProductGeom),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D product geom");
    }
    DeviceProductGeom* d_geom0 = d_geom_by_level[0];

    // T / Vpp on device for mid-SCF energy pieces without downloading P.
    DeviceBuf t_buf;
    DeviceBuf vpp_buf;
    check_cuda(cudaMalloc(&t_buf.p,
                          static_cast<size_t>(nb) * nb * sizeof(double)),
               "malloc d_T");
    check_cuda(cudaMalloc(&vpp_buf.p,
                          static_cast<size_t>(nb) * nb * sizeof(double)),
               "malloc d_Vpp");
    double* d_T = t_buf.as<double>();
    double* d_Vpp = vpp_buf.as<double>();
    wpack.upload_matrix(T.data(), d_T);
    wpack.upload_matrix(Vpp.data(), d_Vpp);

    // H0 = T + Vpp on device once; mid-SCF H = H0 + V without V_AO D2H
    // (cuda-gpw-device-h-assemble; CUDA-only iron law).
    DeviceBuf h0_buf;
    DeviceBuf h_work_buf;
    const size_t mat_bytes =
        static_cast<size_t>(nb) * static_cast<size_t>(nb) * sizeof(double);
    check_cuda(cudaMalloc(&h0_buf.p, mat_bytes), "malloc d_H0");
    check_cuda(cudaMalloc(&h_work_buf.p, mat_bytes), "malloc d_H_work");
    double* d_H0 = h0_buf.as<double>();
    double* d_H_work = h_work_buf.as<double>();
    launch_add_mats(d_T, d_Vpp, d_H0, nb * nb, stream);

    // UKS: host Pa/Pb uploaded each KS build (device DIIS may D2H P).
    DeviceBuf pa_buf;
    DeviceBuf pb_buf;
    double* d_Pa = nullptr;
    double* d_Pb = nullptr;
    if (uks) {
        const size_t nn = static_cast<size_t>(nb) * nb * sizeof(double);
        check_cuda(cudaMalloc(&pa_buf.p, nn), "malloc d_Pa");
        check_cuda(cudaMalloc(&pb_buf.p, nn), "malloc d_Pb");
        d_Pa = pa_buf.as<double>();
        d_Pb = pb_buf.as<double>();
    }

    // Boundary-only V download (final payload / max_iter==0). Not mid-SCF.
    auto download_V_host = [&]() -> RealMatrix {
        RealMatrix V = xt::zeros<double>({nb, nb});
        wpack.download_V(V.data());
        return V;
    };

    // ── Collinear UKS (Padé LSD + dual device DIIS; single-grid only) ─────
    if (uks) {
        // use_mg is impossible after validate_slice_; ks_sg is live.
        RealMatrix Pa = build_density_matrix(C0, n_alpha, nb, 1.0);
        RealMatrix Pb = build_density_matrix(C0, n_beta, nb, 1.0);
        if (guess_P != nullptr) {
            if (static_cast<int>(guess_P->shape(0)) != nb
                || static_cast<int>(guess_P->shape(1)) != nb)
                throw std::invalid_argument(
                    "cuda::gpw::GpwDft::run: guess_P shape does not match nb");
            if (guess_P_beta == nullptr)
                throw std::invalid_argument(
                    "cuda::gpw::GpwDft::run: UKS guess_P requires guess_P_beta");
            if (static_cast<int>(guess_P_beta->shape(0)) != nb
                || static_cast<int>(guess_P_beta->shape(1)) != nb)
                throw std::invalid_argument(
                    "cuda::gpw::GpwDft::run: guess_P_beta shape does not match "
                    "nb");
            Pa = *guess_P;
            Pb = *guess_P_beta;
        } else if (guess_P_beta != nullptr) {
            throw std::invalid_argument(
                "cuda::gpw::GpwDft::run: guess_P_beta without guess_P");
        }

        // Dual-spin device Pulay DIIS + cuSOLVER eigen
        // (cuda-gpw-uks-device-diis). S/X shared; one channel per spin.
        const int n_mo_uks = static_cast<int>(orth.X.shape(1));
        AoDiisEigenDevice ao_a;
        AoDiisEigenDevice ao_b;
        ao_a.build(stream, nb, n_mo_uks, use_diis ? diis_depth : 0, n_alpha,
                   /*occ_factor=*/1.0, level_shift);
        ao_b.build(stream, nb, n_mo_uks, use_diis ? diis_depth : 0, n_beta,
                   /*occ_factor=*/1.0, level_shift);
        ao_a.reset();
        ao_b.reset();
        ao_a.upload_SX(S.data(), orth.X.data());
        ao_b.upload_SX(S.data(), orth.X.data());
        ao_a.upload_P(Pa.data());
        ao_b.upload_P(Pb.data());
        // Seed C for level-shift projector (CPU starts C_prev from C0).
        if (static_cast<int>(C0.shape(0)) == nb
            && static_cast<int>(C0.shape(1)) >= n_mo_uks) {
            ao_a.upload_C(C0.data());
            ao_b.upload_C(C0.data());
        }

        // Device H workspaces for α/β (cuda-gpw-uks-device-h).
        DeviceBuf ha_buf;
        DeviceBuf hb_buf;
        check_cuda(cudaMalloc(&ha_buf.p, mat_bytes), "malloc d_Ha");
        check_cuda(cudaMalloc(&hb_buf.p, mat_bytes), "malloc d_Hb");
        double* d_Ha = ha_buf.as<double>();
        double* d_Hb = hb_buf.as<double>();

        /// Mid-SCF: pack from device P, KS, scatter, H=H0+V on device.
        auto build_uks_energy_device =
            [&](const double* d_Pa_use, const double* d_Pb_use,
                bool store_grids) -> double {
            wpack.pack_weights(d_Pa_use, d_geom0, dev.d_prods, nprod_geom,
                               eps_drop);
            wpack.pack_weights(d_Pb_use, d_geom0, dev.d_prods_b, nprod_geom,
                               eps_drop);

            const KsChainEnergies e = ks_sg->run_uks(
                dev.d_prods, nprod_geom, dev.d_prods_b, nprod_geom, dev.d_n_ion,
                d_vloc_arg, dev.d_acc, dev.d_acc_b, kxc);

            wpack.scatter_acc(d_geom0, dev.d_acc, nprod_geom, w_grid);
            launch_add_mats(d_H0, wpack.d_V(), d_Ha, nb * nb, stream);
            wpack.scatter_acc(d_geom0, dev.d_acc_b, nprod_geom, w_grid);
            launch_add_mats(d_H0, wpack.d_V(), d_Hb, nb * nb, stream);

            if (store_grids) {
                std::vector<double> veff_a(static_cast<std::size_t>(npts));
                std::vector<double> veff_b(static_cast<std::size_t>(npts));
                std::vector<double> v_es(static_cast<std::size_t>(npts));
                check_cuda(cudaMemcpyAsync(veff_a.data(), ks_sg->d_veff(), bytes,
                                           cudaMemcpyDeviceToHost, stream),
                           "D2H veff_a boundary");
                check_cuda(cudaMemcpyAsync(veff_b.data(), ks_sg->d_veff_b(),
                                           bytes, cudaMemcpyDeviceToHost,
                                           stream),
                           "D2H veff_b boundary");
                check_cuda(cudaMemcpyAsync(v_es.data(), ks_sg->d_v_es(), bytes,
                                           cudaMemcpyDeviceToHost, stream),
                           "D2H v_es boundary");
                check_cuda(cudaStreamSynchronize(stream), "sync uks veff D2H");
                result_.veff_a = std::move(veff_a);
                result_.veff_b = std::move(veff_b);
                result_.v_es = std::move(v_es);
                result_.veff.clear();
            }

            result_.e_kinetic =
                wpack.frobenius(d_Pa_use, d_T) + wpack.frobenius(d_Pb_use, d_T);
            result_.e_nonlocal = 0.0;
            result_.e_loc_sr = wpack.frobenius(d_Pa_use, d_Vpp)
                               + wpack.frobenius(d_Pb_use, d_Vpp) + e.e_locgrid;
            result_.e_es = e.e_hartree - e_self;
            result_.e_xc = e.e_xc;
            return result_.e_kinetic + result_.e_loc_sr + result_.e_es
                   + result_.e_xc;
        };

        if (max_iter == 0) {
            if (guess_P == nullptr || guess_P_beta == nullptr)
                throw std::invalid_argument(
                    "cuda::gpw::GpwDft: max_iter==0 UKS requires guess_P and "
                    "guess_P_beta");
            wpack.upload_matrix(Pa.data(), d_Pa);
            wpack.upload_matrix(Pb.data(), d_Pb);
            const double e_total =
                build_uks_energy_device(d_Pa, d_Pb, /*store_grids=*/true);
            result_.e_total = e_total;
            result_.P = Pa;
            result_.P_beta = Pb;
            // Boundary host H.
            const RealMatrix Vb = download_V_host(); // last scatter was beta
            // Rebuild Va: re-scatter alpha for host H (boundary only).
            wpack.scatter_acc(d_geom0, dev.d_acc, nprod_geom, w_grid);
            const RealMatrix Va = download_V_host();
            result_.H_ks = H0 + Va;
            result_.H_ks_beta = H0 + Vb;
            result_.S = S;
            result_.converged = true;
            result_.iters = 0;
            return result_;
        }

        if (!use_diis)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft UKS: diis_depth>0 required for CUDA-only "
                "mid-path");

        double prev_energy = std::numeric_limits<double>::quiet_NaN();
        RealMatrix Ha_raw, Hb_raw;
        bool conv = false;
        int it = 0;
        const bool scf_trace_uks =
            std::getenv("ATV_GPW_CUDA_TRACE") != nullptr;
        for (it = 0; it < max_iter; ++it) {
            const double e_total = build_uks_energy_device(
                ao_a.d_P(), ao_b.d_P(), /*store_grids=*/false);
            result_.e_total = e_total;
            const double err_a = ao_a.residual_device(d_Ha);
            const double err_b = ao_b.residual_device(d_Hb);
            const double err_norm = std::max(err_a, err_b);
            const bool has_prev = !std::isnan(prev_energy);
            const double dE =
                has_prev ? (e_total - prev_energy)
                         : std::numeric_limits<double>::infinity();
            if (scf_trace_uks)
                std::fprintf(stderr,
                             "[cuda gpw uks %3d] E=%.10f  dE=%.3e  "
                             "err_a=%.3e err_b=%.3e\n",
                             it, e_total, has_prev ? dE : 0.0, err_a, err_b);
            prev_energy = e_total;
            if (has_prev && std::abs(dE) < tol && err_norm < kDiisErrTol) {
                conv = true;
                break;
            }
            ao_a.advance();
            ao_b.advance();
            // P stays on device under DIIS (no mid-SCF download_P).
        }
        {
            // Boundary: payload D2H of P and H.
            ao_a.download_P(Pa.data());
            ao_b.download_P(Pb.data());
            const double e_final = build_uks_energy_device(
                ao_a.d_P(), ao_b.d_P(), /*store_grids=*/true);
            result_.e_total = e_final;
            wpack.scatter_acc(d_geom0, dev.d_acc, nprod_geom, w_grid);
            Ha_raw = H0 + download_V_host();
            wpack.scatter_acc(d_geom0, dev.d_acc_b, nprod_geom, w_grid);
            Hb_raw = H0 + download_V_host();
            store_converged_state_uks(result_, Ha_raw, Hb_raw, orth.X, S,
                                      n_alpha, n_beta, nb);
            result_.P = Pa;
            result_.P_beta = Pb;
            result_.S = S;
            result_.converged = conv;
            result_.iters = conv ? (it + 1) : max_iter;
        }
        return result_;
    }

    // ── Closed-shell RKS (single-grid or multigrid) ───────────────────────
    RealMatrix P = build_density_matrix(C0, n_occ, nb, 2.0);
    if (guess_P != nullptr) {
        if (static_cast<int>(guess_P->shape(0)) != nb
            || static_cast<int>(guess_P->shape(1)) != nb)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft::run: guess_P shape does not match nb");
        P = *guess_P;
    }
    if (guess_P_beta != nullptr)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft::run: guess_P_beta is UKS-only (set n_alpha/"
            "n_beta)");

    // Device Pulay DIIS + cuSOLVER eigen for RKS mid-SCF
    // (cuda-gpw-device-diis-eigen). S/X fixed for the geometry; P refreshed.
    const int n_mo = static_cast<int>(orth.X.shape(1));
    AoDiisEigenDevice ao_scf;
    ao_scf.build(stream, nb, n_mo, use_diis ? diis_depth : 0, n_occ,
                 /*occ_factor=*/2.0, level_shift);
    ao_scf.reset(); // clear DIIS history; does not touch S/X/P
    ao_scf.upload_SX(S.data(), orth.X.data());
    ao_scf.upload_P(P.data());
    // Seed C for level-shift projector (CPU starts C_prev from C0).
    if (static_cast<int>(C0.shape(0)) == nb
        && static_cast<int>(C0.shape(1)) >= n_mo) {
        ao_scf.upload_C(C0.data());
    }

    /// Device-P mid-SCF KS: pack → KS chain → scatter V → H=H0+V on device.
    /// Mid-path does **not** D2H V_AO (cuda-only iron law). Boundary-only
    /// downloads when `store_grids` (final payload / max_iter==0).
    auto build_energy_device =
        [&](const double* d_P_use, bool store_grids) -> double {
        KsChainEnergies e{};

        if (!use_mg) {
            // Single-grid device product KS (cuda-gpw-product-collocate-pbc):
            // pack P→weights → sphere-window PBC collocate → Poisson/XC →
            // integrate → scatter V_AO. No mid-SCF P/ρ/V_AO DH (iron law).
            // CompactBasisGrid retained only for force free-fn / payload.
            wpack.pack_weights(d_P_use, d_geom0, dev.d_prods, nprod_geom,
                               eps_drop);
            e = ks_sg->run(dev.d_prods, nprod_geom, dev.d_n_ion, d_vloc_arg,
                           dev.d_acc, kxc);
            if (std::getenv("ATV_GPW_CUDA_TRACE") != nullptr) {
                // Diagnostic: ∫ρ after device product collocate (PBC check).
                DeviceBuf chg;
                check_cuda(cudaMalloc(&chg.p, sizeof(double)), "malloc chg");
                collocate_integrate_rho(ks_sg->d_rho(), npts, w_grid,
                                        chg.as<double>(), stream);
                double q_el = 0.0;
                check_cuda(cudaMemcpyAsync(&q_el, chg.as<double>(),
                                           sizeof(double),
                                           cudaMemcpyDeviceToHost, stream),
                           "D2H charge diag");
                check_cuda(cudaStreamSynchronize(stream), "sync chg");
                std::fprintf(stderr,
                             "[cuda gpw sg] nprod=%d  Q_el=%.6f  "
                             "eH=%.6f eXC=%.6f\n",
                             nprod_geom, q_el, e.e_hartree, e.e_xc);
            }
            wpack.scatter_acc(d_geom0, dev.d_acc, nprod_geom, w_grid);
            if (store_grids) {
                std::vector<double> veff(static_cast<std::size_t>(npts));
                std::vector<double> v_es(static_cast<std::size_t>(npts));
                check_cuda(cudaMemcpyAsync(veff.data(), ks_sg->d_veff(), bytes,
                                           cudaMemcpyDeviceToHost, stream),
                           "D2H veff SG boundary");
                check_cuda(cudaMemcpyAsync(v_es.data(), ks_sg->d_v_es(), bytes,
                                           cudaMemcpyDeviceToHost, stream),
                           "D2H v_es SG boundary");
                check_cuda(cudaStreamSynchronize(stream), "sync SG veff D2H");
                result_.veff = std::move(veff);
                result_.v_es = std::move(v_es);
            }
        } else {
            // Multigrid product KS: host collint multi-level twin (CPU MAIN
            // path: multigrid_collocate_rho + ks_grid_build_rks +
            // multigrid_integrate_veff). Device monom pack→KsMultiGridChain
            // under-corrects Si multi-level by ~4e-4 Ha vs collint (GH200
            // 1180025: CUDA MG ≈ SG energy while CPU MG shifts −15.1523 →
            // −15.1519). H₂ monom MG stays green because s-only products.
            // Mid-path P D2H + V H2D here is residual (iron-law P2) until a
            // collint-faithful device multi-level restore — same pattern as
            // product-si host cbg before product-collocate-pbc.
            // Optional ATV_GPW_CUDA_DEVICE_MG=1 forces monom device path.
            const bool force_device_mg =
                std::getenv("ATV_GPW_CUDA_DEVICE_MG") != nullptr;
            if (!force_device_mg) {
                RealMatrix P_host = xt::zeros<double>({nb, nb});
                // d_P is column-major (xtensor RealMatrix / AoDiis).
                check_cuda(cudaMemcpyAsync(P_host.data(), d_P_use, mat_bytes,
                                           cudaMemcpyDeviceToHost, stream),
                           "D2H P for host MG KS");
                check_cuda(cudaStreamSynchronize(stream), "sync P D2H MG");
                // Row-major Pvec for CPU product_density_weight.
                std::vector<double> Pvec(static_cast<std::size_t>(nb) * nb);
                for (int i = 0; i < nb; ++i)
                    for (int j = 0; j < nb; ++j)
                        Pvec[static_cast<std::size_t>(i) * nb + j] =
                            P_host(i, j);
                std::vector<double> rho = multigrid_collocate_rho(
                    mg, pset, nb, Pvec, eps);
                const cpu::gpw::internal::KsGridRks ks =
                    cpu::gpw::internal::ks_grid_build_rks(
                        grid, rho, n_ion, v_locsr, xc, /*nonperiodic=*/false,
                        false, false, false);
                e.e_hartree = ks.e_hartree;
                e.e_xc = ks.e_xc;
                e.e_locgrid = ks.e_locgrid;
                const std::vector<double> Vflat = multigrid_integrate_veff(
                    mg, pset, nb, ks.veff, eps);
                RealMatrix Vgrid = xt::zeros<double>({nb, nb});
                for (int i = 0; i < nb; ++i)
                    for (int j = 0; j < nb; ++j)
                        Vgrid(i, j) =
                            Vflat[static_cast<std::size_t>(i) * nb + j];
                wpack.upload_matrix(Vgrid.data(), wpack.d_V());
                if (store_grids) {
                    result_.veff = ks.veff;
                    result_.v_es = ks.v_es;
                }
                if (std::getenv("ATV_GPW_CUDA_TRACE") != nullptr) {
                    double q_el = 0.0;
                    for (double v : rho)
                        q_el += v;
                    q_el *= w_fine;
                    int nprod_sum = 0;
                    for (int l = 0; l < nlevels; ++l)
                        nprod_sum +=
                            nprod_by_level[static_cast<std::size_t>(l)];
                    std::fprintf(stderr,
                                 "[cuda gpw mg host-collint] nlev=%d "
                                 "nprod=%d  Q_el=%.6f  eH=%.6f eXC=%.6f\n",
                                 nlevels, nprod_sum, q_el, e.e_hartree,
                                 e.e_xc);
                }
            } else {
                std::vector<DeviceProduct*> d_prods_ptr(
                    static_cast<std::size_t>(nlevels), nullptr);
                std::vector<double*> d_acc_ptr(
                    static_cast<std::size_t>(nlevels), nullptr);
                for (int l = 0; l < nlevels; ++l) {
                    const int np = nprod_by_level[static_cast<std::size_t>(l)];
                    if (np <= 0)
                        continue;
                    DeviceProduct* d_prods_l =
                        mg_prods.d_prods[static_cast<std::size_t>(l)];
                    wpack.pack_weights(
                        d_P_use, d_geom_by_level[static_cast<std::size_t>(l)],
                        d_prods_l, np, eps_drop);
                    d_prods_ptr[static_cast<std::size_t>(l)] = d_prods_l;
                    d_acc_ptr[static_cast<std::size_t>(l)] =
                        mg_prods.d_acc[static_cast<std::size_t>(l)];
                }
                e = ks_mg->run(d_prods_ptr.data(), nprod_by_level.data(),
                               dev.d_n_ion, d_vloc_arg, d_acc_ptr.data(),
                               kxc);
                if (std::getenv("ATV_GPW_CUDA_TRACE") != nullptr) {
                    DeviceBuf chg;
                    check_cuda(cudaMalloc(&chg.p, sizeof(double)),
                               "malloc mg chg");
                    collocate_integrate_rho(ks_mg->fine_chain().d_rho(), npts,
                                            w_fine, chg.as<double>(), stream);
                    double q_el = 0.0;
                    check_cuda(cudaMemcpyAsync(&q_el, chg.as<double>(),
                                               sizeof(double),
                                               cudaMemcpyDeviceToHost, stream),
                               "D2H mg charge diag");
                    check_cuda(cudaStreamSynchronize(stream), "sync mg chg");
                    int nprod_sum = 0;
                    for (int l = 0; l < nlevels; ++l)
                        nprod_sum +=
                            nprod_by_level[static_cast<std::size_t>(l)];
                    std::fprintf(stderr,
                                 "[cuda gpw mg device-monom] nlev=%d "
                                 "nprod=%d  Q_el=%.6f  eH=%.6f eXC=%.6f\n",
                                 nlevels, nprod_sum, q_el, e.e_hartree,
                                 e.e_xc);
                }
                bool first = true;
                for (int l = 0; l < nlevels; ++l) {
                    const int np = nprod_by_level[static_cast<std::size_t>(l)];
                    if (np <= 0)
                        continue;
                    const double w_l =
                        mg.levels[static_cast<std::size_t>(l)].weight();
                    if (first) {
                        wpack.scatter_acc(
                            d_geom_by_level[static_cast<std::size_t>(l)],
                            mg_prods.d_acc[static_cast<std::size_t>(l)], np,
                            w_l);
                        first = false;
                    } else {
                        wpack.scatter_acc_add(
                            d_geom_by_level[static_cast<std::size_t>(l)],
                            mg_prods.d_acc[static_cast<std::size_t>(l)], np,
                            w_l);
                    }
                }
                if (first) {
                    check_cuda(
                        cudaMemsetAsync(wpack.d_V(), 0, mat_bytes, stream),
                        "zero empty V");
                }
                if (store_grids) {
                    std::vector<double> veff(static_cast<std::size_t>(npts));
                    std::vector<double> v_es(static_cast<std::size_t>(npts));
                    check_cuda(cudaMemcpyAsync(veff.data(),
                                               ks_mg->fine_chain().d_veff(),
                                               bytes, cudaMemcpyDeviceToHost,
                                               stream),
                               "D2H veff fine (boundary)");
                    check_cuda(cudaMemcpyAsync(v_es.data(),
                                               ks_mg->fine_chain().d_v_es(),
                                               bytes, cudaMemcpyDeviceToHost,
                                               stream),
                               "D2H v_es fine (boundary)");
                    check_cuda(cudaStreamSynchronize(stream),
                               "sync mg veff D2H");
                    result_.veff = std::move(veff);
                    result_.v_es = std::move(v_es);
                }
            }
        }

        // H = H0 + V on device (no V_AO D2H).
        launch_add_mats(d_H0, wpack.d_V(), d_H_work, nb * nb, stream);

        result_.e_kinetic = wpack.frobenius(d_P_use, d_T);
        result_.e_nonlocal = 0.0;
        result_.e_loc_sr = wpack.frobenius(d_P_use, d_Vpp) + e.e_locgrid;
        result_.e_es = e.e_hartree - e_self;
        result_.e_xc = e.e_xc;
        return result_.e_kinetic + result_.e_loc_sr + result_.e_es
               + result_.e_xc;
    };

    // Fixed-density path (Car–Parrinello-class single KS eval).
    if (max_iter == 0) {
        if (guess_P == nullptr)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: max_iter==0 requires guess_P");
        ao_scf.upload_P(P.data());
        const double e_total =
            build_energy_device(ao_scf.d_P(), /*store_grids=*/true);
        result_.e_total = e_total;
        result_.P = P;
        // Boundary: host H for last_result.
        const RealMatrix Vgrid = download_V_host();
        result_.H_ks = H0 + Vgrid;
        result_.S = S;
        result_.converged = true;
        result_.iters = 0;
        return result_;
    }

    if (!use_diis)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: linear density mixing mid-SCF still requires "
            "host P blend (DH). Set diis_depth>0 for CUDA-only mid-path "
            "(default), or use cpu::gpw::GpwDft.");

    double prev_energy = std::numeric_limits<double>::quiet_NaN();
    bool conv = false;
    int it = 0;
    const bool scf_trace = std::getenv("ATV_GPW_CUDA_TRACE") != nullptr;
    for (it = 0; it < max_iter; ++it) {
        // Mid-SCF: device pack + KS + H assemble; P stays on device (DIIS).
        const double e_total =
            build_energy_device(ao_scf.d_P(), /*store_grids=*/false);
        result_.e_total = e_total;
        // Residual from device H (no H2D of KS matrix).
        const double err_norm = ao_scf.residual_device(d_H_work);
        const bool has_prev = !std::isnan(prev_energy);
        const double dE =
            has_prev ? (e_total - prev_energy)
                     : std::numeric_limits<double>::infinity();
        if (use_diis && has_prev && dE > kDiisAscentResetHa)
            ao_scf.reset();
        if (scf_trace)
            std::fprintf(stderr,
                         "[cuda gpw scf %3d] E=%.10f  dE=%.3e  err=%.3e\n", it,
                         e_total, has_prev ? dE : 0.0, err_norm);
        prev_energy = e_total;
        if (has_prev && std::abs(dE) < tol && err_norm < kDiisErrTol) {
            conv = true;
            break;
        }
        ao_scf.advance();
    }
    // Final KS at the last density: energy pieces + V_eff/V_es for forces.
    // Boundary D2H only (payload). Device eigen of assembled H replaces host
    // store_converged_state → diagonalize_fock (cuda-gpw-device-payload).
    {
        ao_scf.download_P(P.data()); // DIIS density used for final energy
        const double e_final =
            build_energy_device(ao_scf.d_P(), /*store_grids=*/true);
        result_.e_total = e_final;
        // Device eigen of d_H_work = H0 + V (already assembled mid final KS).
        // Boundary D2H of C/ε; densified d_P_ is not the SCF energy density.
        result_.C = xt::zeros<double>({nb, n_mo});
        result_.eps.assign(static_cast<std::size_t>(n_mo), 0.0);
        ao_scf.eigen_from_device(d_H_work, result_.eps.data(), result_.C.data());
        // H_ks: boundary D2H of assembled device H (no host H0+V rebuild).
        result_.H_ks = xt::zeros<double>({nb, nb});
        check_cuda(cudaMemcpyAsync(result_.H_ks.data(), d_H_work,
                                   mat_bytes, cudaMemcpyDeviceToHost, stream),
                   "D2H H_ks payload");
        check_cuda(cudaStreamSynchronize(stream), "sync H_ks payload");
        result_.P = P;
        result_.S = S;
        result_.converged = conv;
        if (conv)
            result_.iters = it + 1;
        else
            result_.iters = max_iter;
    }
    return result_;
}

namespace {

// ── First-slice Γ RKS force assembly (host AO Pulay + device grid product) ──

using cpu::gpw::GridPrimitive;
using cpu::gpw::Prim;

GridBasisFunction deriv_bf(const GridBasisFunction& bf, int k) {
    GridBasisFunction d;
    d.reserve(bf.size() * 2);
    for (const auto& p : bf) {
        GridPrimitive up = p;
        up.l[k] += 1;
        up.coeff = p.coeff * 2.0 * p.alpha;
        d.push_back(up);
        if (p.l[k] > 0) {
            GridPrimitive dn = p;
            dn.coeff = p.coeff * (-static_cast<double>(p.l[k]));
            dn.l[k] -= 1;
            d.push_back(dn);
        }
    }
    return d;
}

Prim to_prim(const GridPrimitive& p) {
    Prim q;
    q.alpha = p.alpha;
    q.center[0] = p.center[0];
    q.center[1] = p.center[1];
    q.center[2] = p.center[2];
    q.l[0] = p.l[0];
    q.l[1] = p.l[1];
    q.l[2] = p.l[2];
    return q;
}

// Nonlocal (and optional analytic local) V_pp image sum — twin of CPU
// gpw_forces.cpp::vpp_images. First-slice always has local on grid, so
// with_local=false and only nonlocal channels contribute (often empty for H).
double vpp_images(const GridBasisFunction& a, const GridBasisFunction& b,
                  const std::vector<const GpwAtom*>& centers, const double L[3],
                  const int m[3], double reach, bool with_local) {
    if (a.empty() || b.empty())
        return 0.0;
    const double* ac = a[0].center;
    const double* bc = b[0].center;
    const double reach2 = reach * reach;
    const bool per = m[0] || m[1] || m[2];
    double v = 0.0;
    for (const auto& p : a) {
        const Prim pp = to_prim(p);
        for (const auto& q0 : b) {
            const double cc = p.coeff * q0.coeff;
            if (cc == 0.0)
                continue;
            const double aa = p.alpha;
            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        const double sx = ac[0] - bc[0] - tx * L[0];
                        const double sy = ac[1] - bc[1] - ty * L[1];
                        const double sz = ac[2] - bc[2] - tz * L[2];
                        if (sx * sx + sy * sy + sz * sz > reach2)
                            continue;
                        GridPrimitive q = q0;
                        q.center[0] += tx * L[0];
                        q.center[1] += ty * L[1];
                        q.center[2] += tz * L[2];
                        const double bb = q.alpha, g = aa + bb;
                        double P[3];
                        for (int d = 0; d < 3; ++d)
                            P[d] = (aa * p.center[d] + bb * q.center[d]) / g;
                        const Prim qq = to_prim(q);
                        for (const GpwAtom* Cp : centers) {
                            const GpwAtom& C = *Cp;
                            double apos[3];
                            for (int d = 0; d < 3; ++d) {
                                apos[d] = C.pos[d];
                                if (per && L[d] > 0.0)
                                    apos[d] -= L[d]
                                               * std::nearbyint(
                                                   (C.pos[d] - P[d]) / L[d]);
                            }
                            if (with_local && !C.cloc.empty())
                                v += cc
                                     * cpu::gpw::gth_local_sr(pp, qq, apos,
                                                              C.rloc, C.cloc);
                            for (const auto& ch : C.nonlocal)
                                v += cc
                                     * cpu::gpw::gth_nonlocal(pp, qq, apos, ch);
                        }
                    }
        }
    }
    return v;
}

RealMatrix energy_weighted(const RealMatrix& C, const std::vector<double>& eps,
                           int n_occ, double occ) {
    const int nb = static_cast<int>(C.shape(0));
    RealMatrix W = xt::zeros<double>({nb, nb});
    for (int i = 0; i < n_occ; ++i) {
        const double w = occ * eps[static_cast<std::size_t>(i)];
        for (int mu = 0; mu < nb; ++mu) {
            const double cmu = w * C(mu, i);
            for (int nu = 0; nu < nb; ++nu)
                W(mu, nu) += cmu * C(nu, i);
        }
    }
    return W;
}

template <typename T>
T* d_alloc(std::size_t n) {
    T* p = nullptr;
    check_cuda(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc force");
    return p;
}

} // namespace

std::vector<std::array<double, 3>>
gpw_forces_from_scf(const std::vector<GpwAtom>& atoms, const GpwDft& method,
                    const GpwResult& scf) {
    if (!scf.converged)
        throw std::runtime_error(
            "cuda::gpw::gpw_forces_from_scf: SCF did not converge");
    // First-slice guards: RKS Γ single-grid local-on-grid device product.
    // UKS / multigrid / multi-k free-fn forces are deeper slugs.
    if (method.is_uks())
        throw std::invalid_argument(
            "cuda::gpw::gpw_forces_from_scf: UKS forces not on device first "
            "slice — use cpu::gpw::gpw_forces_from_scf");
    if (method.uses_kpath())
        throw std::invalid_argument(
            "cuda::gpw::gpw_forces_from_scf: multi-k forces not on device "
            "first slice — keep k_mesh={1,1,1} or use "
            "cpu::gpw::gpw_forces_from_scf (deeper slug cuda-gpw-kpoint-forces)");
    if (method.ngrids != 1)
        throw std::invalid_argument(
            "cuda::gpw::gpw_forces_from_scf: multigrid forces not on device "
            "first slice — set ngrids=1 or use cpu::gpw::gpw_forces_from_scf");
    if (method.n_occ < 1)
        throw std::invalid_argument(
            "cuda::gpw::gpw_forces_from_scf: n_occ must be >= 1");
    if (scf.P.size() == 0 || scf.C.size() == 0 || scf.eps.empty())
        throw std::runtime_error(
            "cuda::gpw::gpw_forces_from_scf: converged result missing P/C/eps");
    if (scf.veff.empty() || scf.v_es.empty())
        throw std::runtime_error(
            "cuda::gpw::gpw_forces_from_scf: missing SCF V_eff/V_es "
            "(need store_grids from final KS)");

    const int na = static_cast<int>(atoms.size());
    const std::vector<GridBasisFunction> basis = flatten_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    if (static_cast<int>(scf.P.shape(0)) != nb
        || static_cast<int>(scf.P.shape(1)) != nb)
        throw std::runtime_error(
            "cuda::gpw::gpw_forces_from_scf: P shape does not match basis");

    const bool periodic = true;
    const double eps = method.eps_default;
    const bool analytic_loc = !method.local_pp_on_grid; // first slice: false

    // Map atoms ↔ AO blocks.
    std::vector<int> atom_off(static_cast<std::size_t>(na) + 1, 0);
    for (int a = 0; a < na; ++a)
        atom_off[static_cast<std::size_t>(a) + 1] =
            atom_off[static_cast<std::size_t>(a)]
            + static_cast<int>(atoms[static_cast<std::size_t>(a)].basis.size());
    std::vector<int> owner(static_cast<std::size_t>(nb), 0);
    for (int a = 0; a < na; ++a)
        for (int u = atom_off[static_cast<std::size_t>(a)];
             u < atom_off[static_cast<std::size_t>(a) + 1]; ++u)
            owner[static_cast<std::size_t>(u)] = a;

    std::vector<double> radii(static_cast<std::size_t>(nb));
    double rmax = 0.0;
    for (int u = 0; u < nb; ++u) {
        radii[static_cast<std::size_t>(u)] =
            basis_function_radius(basis[static_cast<std::size_t>(u)], eps);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(u)]);
    }
    std::vector<std::array<GridBasisFunction, 3>> dphi(
        static_cast<std::size_t>(nb));
    std::vector<std::array<double, 3>> drad(static_cast<std::size_t>(nb));
    double drmax = 0.0;
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k) {
            dphi[static_cast<std::size_t>(u)][k] =
                deriv_bf(basis[static_cast<std::size_t>(u)], k);
            drad[static_cast<std::size_t>(u)][k] = basis_function_radius(
                dphi[static_cast<std::size_t>(u)][k], eps);
            drmax = std::max(drmax, drad[static_cast<std::size_t>(u)][k]);
        }
    int mimg[3];
    pair_image_counts(method.cell, periodic, std::max(2.0 * rmax, drmax + rmax),
                      mimg);

    // Energy-weighted density W = Σ_i 2 ε_i c_i c_iᵀ (closed-shell).
    const RealMatrix& P = scf.P;
    RealMatrix W = energy_weighted(scf.C, scf.eps, method.n_occ, 2.0);

    std::vector<std::array<double, 3>> forces(
        static_cast<std::size_t>(na), {0.0, 0.0, 0.0});

    // ── Host AO Pulay: +2 W dS − 2 P dT − 2 P dV_pp(basis) + op-center ──
    std::vector<const GpwAtom*> all(static_cast<std::size_t>(na));
    for (int a = 0; a < na; ++a)
        all[static_cast<std::size_t>(a)] = &atoms[static_cast<std::size_t>(a)];
    for (int A = 0; A < na; ++A) {
        const std::vector<const GpwAtom*> onlyA = {
            &atoms[static_cast<std::size_t>(A)]};
        for (int k = 0; k < 3; ++k) {
            double f = 0.0;
            for (int mu = atom_off[static_cast<std::size_t>(A)];
                 mu < atom_off[static_cast<std::size_t>(A) + 1]; ++mu) {
                const GridBasisFunction& dmu =
                    dphi[static_cast<std::size_t>(mu)][k];
                const double rdmu = drad[static_cast<std::size_t>(mu)][k];
                for (int nu = 0; nu < nb; ++nu) {
                    const double reach =
                        rdmu + radii[static_cast<std::size_t>(nu)];
                    double s = 0.0, t = 0.0;
                    contracted_ST_periodic(dmu, basis[static_cast<std::size_t>(nu)],
                                           method.cell, mimg, reach, s, t);
                    f += 2.0 * W(mu, nu) * s;
                    f -= 2.0 * P(mu, nu) * t;
                    f -= 2.0 * P(mu, nu)
                         * vpp_images(dmu, basis[static_cast<std::size_t>(nu)],
                                      all, method.cell, mimg, reach,
                                      analytic_loc);
                }
            }
            for (int mu = 0; mu < nb; ++mu) {
                const GridBasisFunction& dmu =
                    dphi[static_cast<std::size_t>(mu)][k];
                const double rdmu = drad[static_cast<std::size_t>(mu)][k];
                for (int nu = 0; nu < nb; ++nu) {
                    const double reach =
                        rdmu + radii[static_cast<std::size_t>(nu)];
                    f += 2.0 * P(mu, nu)
                         * vpp_images(dmu, basis[static_cast<std::size_t>(nu)],
                                      onlyA, method.cell, mimg, reach,
                                      analytic_loc);
                }
            }
            forces[static_cast<std::size_t>(A)][k] = f;
        }
    }

    // ── Device grid product: V_eff grid + ionic HF + local-PP HF ──────────
    // Force free-fn is a **post-SCF boundary**: one setup H2D of SCF payload
    // + tables, then device kernels on a non-default stream (no default-stream
    // cudaDeviceSynchronize). Host AO Pulay above remains hybrid residual.
    UniformGrid grid = make_uniform_grid(method.cell, method.cutoff_ry);
    const long ng = grid.npts();
    if (static_cast<long>(scf.veff.size()) != ng
        || static_cast<long>(scf.v_es.size()) != ng)
        throw std::runtime_error(
            "cuda::gpw::gpw_forces_from_scf: V_eff/V_es size != grid npts");

    CompactBasisGrid cbg_local;
    const CompactBasisGrid* cbg_ptr = nullptr;
    if (scf.scf_cbg_valid && scf.scf_cbg.npts == ng) {
        cbg_ptr = &scf.scf_cbg;
    } else {
        cbg_local = build_compact_basis_grid(grid, basis, periodic, eps);
        cbg_ptr = &cbg_local;
    }
    const CompactBasisGrid& cbg = *cbg_ptr;

    std::vector<GridBasisFunction> dflat(static_cast<std::size_t>(nb) * 3);
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k)
            dflat[static_cast<std::size_t>(u) * 3 + k] =
                dphi[static_cast<std::size_t>(u)][k];
    CompactBasisGrid cbgd_local;
    const CompactBasisGrid* cbgd_ptr = nullptr;
    if (scf.force_cbgd_valid && scf.force_cbgd.npts == ng) {
        cbgd_ptr = &scf.force_cbgd;
    } else {
        GpwResult& scf_mut = const_cast<GpwResult&>(scf);
        scf_mut.force_cbgd =
            build_compact_basis_grid(grid, dflat, periodic, eps);
        scf_mut.force_cbgd_valid = true;
        cbgd_ptr = &scf_mut.force_cbgd;
    }
    const CompactBasisGrid& cbgd = *cbgd_ptr;

    const std::vector<double> Pvec = mat_to_vec(P, nb);
    // Host densify is force-setup (geometry CSR + P); device kernels own
    // the O(N_grid) HF. Prefer scf.rho if ever cached (not yet).
    const std::vector<double> rho = collocate_density(cbg, nb, Pvec);

    std::vector<int> g2a(static_cast<std::size_t>(ng), -1);
    for (std::size_t a = 0; a < cbg.gidx.size(); ++a)
        g2a[static_cast<std::size_t>(cbg.gidx[a])] = static_cast<int>(a);

    const int n_act = static_cast<int>(cbg.gidx.size());
    const int n_act_d = static_cast<int>(cbgd.gidx.size());
    const std::size_t nnz = cbg.fu.size();
    const std::size_t nnz_d = cbgd.fu.size();

    ScfStream force_stream;
    const cudaStream_t stream = force_stream;

    double* d_veff = d_alloc<double>(static_cast<std::size_t>(ng));
    double* d_v_es = d_alloc<double>(static_cast<std::size_t>(ng));
    double* d_rho = d_alloc<double>(static_cast<std::size_t>(ng));
    double* d_P = d_alloc<double>(Pvec.size());
    double* d_forces = d_alloc<double>(static_cast<std::size_t>(3 * na));
    int* d_owner = d_alloc<int>(static_cast<std::size_t>(nb));
    long* d_cbg_gidx = d_alloc<long>(static_cast<std::size_t>(n_act));
    std::size_t* d_cbg_off =
        d_alloc<std::size_t>(static_cast<std::size_t>(n_act) + 1);
    int* d_cbg_fu = d_alloc<int>(nnz);
    double* d_cbg_fv = d_alloc<double>(nnz);
    long* d_cbgd_gidx = d_alloc<long>(static_cast<std::size_t>(n_act_d));
    std::size_t* d_cbgd_off =
        d_alloc<std::size_t>(static_cast<std::size_t>(n_act_d) + 1);
    int* d_cbgd_fu = d_alloc<int>(nnz_d);
    double* d_cbgd_fv = d_alloc<double>(nnz_d);
    int* d_g2a = d_alloc<int>(static_cast<std::size_t>(ng));
    internal::DeviceForceAtom* d_atoms =
        d_alloc<internal::DeviceForceAtom>(static_cast<std::size_t>(na));

    auto free_all = [&]() {
        cudaFree(d_veff);
        cudaFree(d_v_es);
        cudaFree(d_rho);
        cudaFree(d_P);
        cudaFree(d_forces);
        cudaFree(d_owner);
        cudaFree(d_cbg_gidx);
        cudaFree(d_cbg_off);
        cudaFree(d_cbg_fu);
        cudaFree(d_cbg_fv);
        cudaFree(d_cbgd_gidx);
        cudaFree(d_cbgd_off);
        cudaFree(d_cbgd_fu);
        cudaFree(d_cbgd_fv);
        cudaFree(d_g2a);
        cudaFree(d_atoms);
    };

    try {
        const size_t bytes_g = static_cast<size_t>(ng) * sizeof(double);
        // Force-setup H2D (boundary for free-fn), all async on force stream.
        check_cuda(cudaMemcpyAsync(d_veff, scf.veff.data(), bytes_g,
                                   cudaMemcpyHostToDevice, stream),
                   "H2D veff force-setup");
        check_cuda(cudaMemcpyAsync(d_v_es, scf.v_es.data(), bytes_g,
                                   cudaMemcpyHostToDevice, stream),
                   "H2D v_es force-setup");
        check_cuda(cudaMemcpyAsync(d_rho, rho.data(), bytes_g,
                                   cudaMemcpyHostToDevice, stream),
                   "H2D rho force-setup");
        check_cuda(cudaMemcpyAsync(d_P, Pvec.data(),
                                   Pvec.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D P force-setup");
        check_cuda(cudaMemsetAsync(
                       d_forces, 0,
                       static_cast<size_t>(3 * na) * sizeof(double), stream),
                   "zero forces");
        check_cuda(cudaMemcpyAsync(d_owner, owner.data(),
                                   static_cast<size_t>(nb) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D owner");
        check_cuda(cudaMemcpyAsync(d_cbg_gidx, cbg.gidx.data(),
                                   static_cast<size_t>(n_act) * sizeof(long),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D cbg.gidx");
        check_cuda(cudaMemcpyAsync(
                       d_cbg_off, cbg.off.data(),
                       (static_cast<size_t>(n_act) + 1) * sizeof(std::size_t),
                       cudaMemcpyHostToDevice, stream),
                   "H2D cbg.off");
        check_cuda(cudaMemcpyAsync(d_cbg_fu, cbg.fu.data(), nnz * sizeof(int),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D cbg.fu");
        check_cuda(cudaMemcpyAsync(d_cbg_fv, cbg.fv.data(),
                                   nnz * sizeof(double), cudaMemcpyHostToDevice,
                                   stream),
                   "H2D cbg.fv");
        check_cuda(cudaMemcpyAsync(d_cbgd_gidx, cbgd.gidx.data(),
                                   static_cast<size_t>(n_act_d) * sizeof(long),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D cbgd.gidx");
        check_cuda(cudaMemcpyAsync(
                       d_cbgd_off, cbgd.off.data(),
                       (static_cast<size_t>(n_act_d) + 1) * sizeof(std::size_t),
                       cudaMemcpyHostToDevice, stream),
                   "H2D cbgd.off");
        check_cuda(cudaMemcpyAsync(d_cbgd_fu, cbgd.fu.data(),
                                   nnz_d * sizeof(int), cudaMemcpyHostToDevice,
                                   stream),
                   "H2D cbgd.fu");
        check_cuda(cudaMemcpyAsync(d_cbgd_fv, cbgd.fv.data(),
                                   nnz_d * sizeof(double),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D cbgd.fv");
        check_cuda(cudaMemcpyAsync(d_g2a, g2a.data(),
                                   static_cast<size_t>(ng) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D g2a");

        std::vector<internal::DeviceForceAtom> h_atoms(
            static_cast<std::size_t>(na));
        for (int a = 0; a < na; ++a) {
            const GpwAtom& A = atoms[static_cast<std::size_t>(a)];
            internal::DeviceForceAtom& da = h_atoms[static_cast<std::size_t>(a)];
            da.pos[0] = A.pos[0];
            da.pos[1] = A.pos[1];
            da.pos[2] = A.pos[2];
            da.z_val = A.z_val;
            da.rloc = A.rloc;
            da.n_cloc = static_cast<int>(
                std::min(A.cloc.size(),
                         static_cast<std::size_t>(internal::kMaxCloc)));
            for (int k = 0; k < da.n_cloc; ++k)
                da.cloc[k] = A.cloc[static_cast<std::size_t>(k)];
        }
        check_cuda(cudaMemcpyAsync(
                       d_atoms, h_atoms.data(),
                       static_cast<size_t>(na)
                           * sizeof(internal::DeviceForceAtom),
                       cudaMemcpyHostToDevice, stream),
                   "H2D atoms");

        internal::DeviceCbgView cbg_view;
        cbg_view.gidx = d_cbg_gidx;
        cbg_view.off = d_cbg_off;
        cbg_view.fu = d_cbg_fu;
        cbg_view.fv = d_cbg_fv;
        cbg_view.g2a = d_g2a;
        cbg_view.n_active = n_act;
        cbg_view.npts = ng;

        internal::DeviceCbgView cbgd_view;
        cbgd_view.gidx = d_cbgd_gidx;
        cbgd_view.off = d_cbgd_off;
        cbgd_view.fu = d_cbgd_fu;
        cbgd_view.fv = d_cbgd_fv;
        cbgd_view.g2a = nullptr;
        cbgd_view.n_active = n_act_d;
        cbgd_view.npts = ng;

        const CollocateGridDesc coll = to_coll(grid);
        internal::accumulate_grid_veff_forces_device(
            coll, cbg_view, cbgd_view, d_veff, d_P, nb, d_owner, d_forces,
            stream);
        internal::accumulate_ionic_hf_forces_device(
            coll, d_atoms, na, d_v_es, periodic, d_forces, stream);
        if (method.local_pp_on_grid)
            internal::accumulate_local_pp_hf_forces_device(
                coll, d_atoms, na, d_rho, periodic, d_forces, stream);

        // Boundary D2H of force vector only (not full grids).
        std::vector<double> h_f(static_cast<std::size_t>(3 * na), 0.0);
        check_cuda(cudaMemcpyAsync(h_f.data(), d_forces,
                                   static_cast<size_t>(3 * na) * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream),
                   "D2H forces boundary");
        check_cuda(cudaStreamSynchronize(stream), "force stream fence");
        for (int a = 0; a < na; ++a)
            for (int k = 0; k < 3; ++k)
                forces[static_cast<std::size_t>(a)][k] +=
                    h_f[static_cast<std::size_t>(3 * a + k)];
    } catch (...) {
        free_all();
        throw;
    }
    free_all();
    return forces;
}

std::vector<std::array<double, 3>>
gpw_forces_analytic(const std::vector<GpwAtom>& atoms, GpwDft& method) {
    const GpwResult& scf = method.run(atoms);
    if (!scf.converged)
        throw std::runtime_error(
            "cuda::gpw::gpw_forces_analytic: SCF did not converge (iters="
            + std::to_string(scf.iters) + ")");
    return gpw_forces_from_scf(atoms, method, scf);
}


namespace {

/// Map CUDA product members onto a CPU shell for fixed-P stress assembly.
cpu::gpw::GpwDft cpu_shell_from_cuda(const GpwDft& m) {
    cpu::gpw::GpwDft c;
    for (int d = 0; d < 3; ++d)
        c.cell[d] = m.cell[d];
    c.cutoff_ry = m.cutoff_ry;
    c.ngrids = m.ngrids;
    c.rel_cutoff = m.rel_cutoff;
    c.n_occ = m.n_occ;
    c.n_alpha = m.n_alpha;
    c.n_beta = m.n_beta;
    c.max_iter = m.max_iter;
    c.tol = m.tol;
    c.eps_default = m.eps_default;
    c.mix = m.mix;
    c.diis_depth = m.diis_depth;
    c.lindep_threshold = m.lindep_threshold;
    c.level_shift = m.level_shift;
    c.xc = m.xc;
    c.nonperiodic = m.nonperiodic;
    c.local_pp_on_grid = m.local_pp_on_grid;
    c.use_ot = m.use_ot;
    for (int d = 0; d < 3; ++d)
        c.k_mesh[d] = m.k_mesh[d];
    return c;
}

void validate_stress_slice(const GpwDft& method) {
    if (method.uses_kpath())
        throw std::invalid_argument(
            "cuda::gpw::gpw_stress: multi-k stress has no device product path "
            "yet — keep k_mesh={1,1,1} / empty k_points or use "
            "cpu::gpw::gpw_stress (deeper slug cuda-gpw-kpoint-stress)");
}

} // namespace

std::array<double, 6>
gpw_stress(const std::vector<GpwAtom>& atoms, GpwDft& method, double h,
           const std::array<bool, 3>& axes) {
    validate_stress_slice(method);
    const GpwResult& scf = method.run(atoms);
    if (!scf.converged)
        throw std::runtime_error(
            "cuda::gpw::gpw_stress: SCF did not converge (iters="
            + std::to_string(scf.iters) + ")");
    return gpw_stress(atoms, method, scf, h, axes);
}

std::array<double, 6>
gpw_stress(const std::vector<GpwAtom>& atoms, const GpwDft& method,
           const GpwResult& scf, double h, const std::array<bool, 3>& axes) {
    validate_stress_slice(method);
    if (!scf.converged)
        throw std::invalid_argument(
            "cuda::gpw::gpw_stress: supplied state is not converged");
    // First cut: host fixed-P analytic assembly at the CUDA density payload
    // (mirrors early free-fn force hybrid — density from device SCF, stress
    // assembly reuses CPU product twin). No silent re-SCF on CPU.
    cpu::gpw::GpwDft shell = cpu_shell_from_cuda(method);
    return cpu::gpw::gpw_stress(atoms, shell, scf, h, axes);
}

} // namespace gpw
} // namespace cuda
} // namespace ATV
