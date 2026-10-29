/// @file cuda/gpw/gpw_solver.cu
/// @brief `cuda::gpw::GpwDft` — product slice (Γ RKS/UKS + multi-k RKS DIIS).
///
/// Hybrid composition (stated):
///   * host — Γ STVnl (S/T + KB Vnl via add_nonlocal_kb_k@Γ) / product-list
///     build; multi-k χ^k build (once);
///     multi-k payload diag / cold-start; Gamma single-grid sparse AO force
///     coefficient setup once per geometry; host force assembly for UKS,
///     multigrid and multi-k, and fixed-P stress assembly
///   * device — product collocate + KS + product integrate (mid-SCF):
///     single-grid RKS device product KS (`KsChainDevice` + PBC min-image
///     collocate/integrate — cuda-gpw-product-collocate-pbc) + device
///     H=H0+V / DIIS; multigrid RKS mid-SCF defaults to **device monom**
///     pack→`KsMultiGridChainDevice`→scatter with collint group max\|w\| +
///     group Ltot window (cuda-gpw-multigrid-device-ks; host collint opt-in
///     `ATV_GPW_CUDA_HOST_MG=1`); collinear UKS
///     `KsChainDevice::run_uks`;
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
///     **Γ UKS final dual payload C/ε from dual device eigen**
///     (cuda-gpw-uks-payload-device; no host dual diagonalize_fock);
///     UKS dual α/β mid-SCF H on device (cuda-gpw-uks-h-assemble) + dual
///     DIIS (cuda-gpw-uks-device-diis);
///     **device P→weight + acc→V_AO scatter** via
///     `internal::ProductWeightPackDevice` (cuda-gpw-device-weight-pack);
///     Gamma RKS single-grid forces consume accepted P, raw physical-H W,
///     resident V_eff/V_es and cold sparse AO/grid tables on the same stream.
/// Gamma single-grid final scientific payload is retrieved only after its
/// force contractions complete. Other paths retain their stated host boundary.
/// Mid-SCF product work borrows the execution owner's non-default stream.
/// Standalone calls own a Backend; nested calls use their parent Backend.

#include "atomiverse/cuda/gpw/gpw_solver.cuh"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/frame/precision.cuh"

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/core_overlap.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/gpw_stress.h"
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
#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
// Minimal CUDA runtime packages omit the optional profiler header. These
// stable C runtime entry points are exported by the already-linked cudart.
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif
#include <cuda_runtime.h>
#include <exception>
#include <functional>
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

using GpwAtom = cpu::gpw::internal::GpwAtom;
using cpu::gpw::basis_function_radius;
using cpu::gpw::build_compact_basis_grid;
using cpu::gpw::build_products;
using cpu::gpw::CompactBasisGrid;
using cpu::gpw::contracted_ST_periodic;
using cpu::gpw::GridBasisFunction;
using cpu::gpw::GridProduct;
using cpu::gpw::gth_nonlocal;
using cpu::gpw::integrate_potential;
using cpu::gpw::local_pp_grid;
using cpu::gpw::make_multigrid;
using cpu::gpw::make_uniform_grid;
using cpu::gpw::MultiGrid;
using cpu::gpw::multigrid_collocate_rho;
using cpu::gpw::multigrid_integrate_veff;
using cpu::gpw::pair_image_counts;
using cpu::gpw::Prim;
using cpu::gpw::PrimContrib;
using cpu::gpw::PrimRef;
using cpu::gpw::ProductSet;
using cpu::gpw::UniformGrid;
using cpu::linalg::build_density_matrix;
using cpu::linalg::CanonicalOrth;
using cpu::linalg::diagonalize_fock;
using cpu::linalg::RealMatrix;
using internal::AoDiisEigenDevice;
using internal::collocate_integrate_rho;
using internal::CollocateGridDesc;
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
using internal::MultigridDesc;
using internal::ProductWeightPackDevice;

constexpr double kBuildScreen = 1.0e-14;
/// DIIS subspace reset when energy rises by more than this [Ha]. Device KS
/// residual (cuFFT / tree reductions) can make pure Pulay non-stationary on
/// multi-center systems (water class); clearing history restarts the warm-up
/// path. H₂-class dual-gates stay within stated tols.
constexpr double kDiisAscentResetHa = 1.0e-3;
/// Matches CPU product mass-margin drop (multigrid.cpp); drop monoms below
/// `eps * kProductMassMargin` when packing live density weights.
constexpr double kProductMassMargin = 1.0e-4;
/// Fine-level monom sphere eps factor (product-collocate-pbc Si charge hold).
constexpr double kSphereRadiusEpsFine = kProductMassMargin;
/// Coarse-level monom sphere eps factor. Matches CPU collint `kProductWindowEps`
/// (multigrid.cpp) — monom continuous lacks cmax refine, so fine stays
/// mass-margin (product-si); coarse uses collint-class 0.02 with group mag +
/// group Ltot (L_win). Not an eps-fit scan: same constant as host collint.
constexpr double kSphereRadiusEpsCoarse = 2.0e-2;

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

std::vector<GridBasisFunction>
flatten_basis(const std::vector<GpwAtom>& atoms) {
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

// Γ RKS final payload: AoDiisEigenDevice::eigen_from_device (device-payload).
// Γ UKS final payload: dual eigen_from_device (uks-payload-device) — no host
// diagonalize_fock re-diag of Ha/Hb.

/// @brief ⟨S²⟩ from boundary-downloaded α/β MO coefficients (payload only).
double spin_contamination_s2(const RealMatrix& Ca, const RealMatrix& Cb,
                             const RealMatrix& S, int n_alpha, int n_beta) {
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
    return sz * (sz + 1.0) + static_cast<double>(n_beta) - ov;
}

CollocateGridDesc to_coll(const UniformGrid& g, double eps_win = 1.0e-14) {
    CollocateGridDesc d;
    for (int i = 0; i < 3; ++i) {
        d.n[i] = g.n[i];
        d.L[i] = g.L[i];
        d.origin[i] = g.origin[i];
    }
    // CUDA GPW product rejects nonperiodic; collocate/integrate always
    // sphere-window PBC (cuda-gpw-product-collocate-pbc).
    d.periodic = true;
    d.eps_win = eps_win;
    return d;
}

cpu::xc::XCType to_ks_xc(cpu::xc::XCType xc) {
    if (xc == cpu::xc::XCType::PADE || cpu::xc::is_pbe_family(xc))
        return xc;
    throw std::invalid_argument(
        "cuda::gpw::GpwDft: xc has no device path — only cpu::xc::XCType::PADE "
        "and the PBE family. Use cpu::gpw::GpwDft for other "
        "functionals.");
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
        check_cuda(cudaMalloc(&d_prods, static_cast<size_t>(nprod)
                                            * sizeof(DeviceProduct)),
                   "malloc d_prods");
        check_cuda(
            cudaMalloc(&d_acc, static_cast<size_t>(nprod) * sizeof(double)),
            "malloc d_acc");
        if (uks) {
            check_cuda(cudaMalloc(&d_prods_b, static_cast<size_t>(nprod)
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
        check_cuda(cudaMalloc(&d_prods[il], static_cast<size_t>(nprod)
                                                * sizeof(DeviceProduct)),
                   "malloc d_prods_level");
        check_cuda(
            cudaMalloc(&d_acc[il], static_cast<size_t>(nprod) * sizeof(double)),
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

} // namespace

using GpwStep = std::function<void(int)>;
namespace {
class GpwStage final : public cuda::Component {
    GpwStep* step_;
    ATV::gpw::Phase phase_;
    const char* name_;

  public:
    GpwStage(GpwStep& step, ATV::gpw::Phase phase, const char* name)
        : step_(&step), phase_(phase), name_(name) {}
    std::string name() const override { return name_; }
    void launch(const cuda::LaunchContext& context) override {
        if (context.at(phase_) && *step_)
            (*step_)(static_cast<int>(context.step));
    }
};
class GpwConverged final : public cuda::Criterion {
    const bool* gate_;

  public:
    explicit GpwConverged(const bool& gate) : gate_(&gate) {}
    std::string name() const override { return "gpw_converged"; }
    bool contributes_to_convergence() const override { return true; }
    Verdict verdict(const cuda::LoopState&) override {
        return *gate_ ? Verdict::Converged : Verdict::Continue;
    }
};

// A cold geometry boundary is required by the existing host AO/grid setup.
// Scientific density/orbital arrays stay under the device SCF's ownership.
cpu::Frame host_geometry(const cuda::Frame& device, cudaStream_t stream) {
    const int n = device.n_atoms();
    if (n <= 0 || !device.atoms.x || !device.atoms.y || !device.atoms.z
        || !device.atoms.type || !device.box)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: incomplete parent geometry");
    const pos_t* primary[3] = {device.atoms.x, device.atoms.y, device.atoms.z};
    const pos_t* correction[3] = {device.atoms.x_corr, device.atoms.y_corr,
                                  device.atoms.z_corr};
    if constexpr (kPositionCorrection)
        for (const auto* column : correction)
            if (!column)
                throw std::invalid_argument(
                    "cuda::gpw::GpwDft: missing position correction");
    cpu::Frame host(n);
    std::array<std::vector<pos_t>, 3> hi, lo;
    for (int d = 0; d < 3; ++d) {
        hi[d].resize(n);
        lo[d].assign(n, pos_t(0));
    }
    std::vector<int> types(n);
    geom_t cell[9];
    struct Transfers {
        cudaStream_t stream;
        bool pending = true;
        ~Transfers() {
            if (pending)
                cudaStreamSynchronize(stream);
        }
    } transfers{stream};
    for (int d = 0; d < 3; ++d) {
        check_cuda(cudaMemcpyAsync(hi[d].data(), primary[d], n * sizeof(pos_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "D2H parent geometry");
        if constexpr (kPositionCorrection)
            check_cuda(cudaMemcpyAsync(lo[d].data(), correction[d],
                                       n * sizeof(pos_t),
                                       cudaMemcpyDeviceToHost, stream),
                       "D2H parent geometry correction");
    }
    check_cuda(cudaMemcpyAsync(types.data(), device.atoms.type, n * sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               "D2H parent species");
    check_cuda(cudaMemcpyAsync(cell, device.box, sizeof(cell),
                               cudaMemcpyDeviceToHost, stream),
               "D2H parent cell");
    check_cuda(cudaStreamSynchronize(stream), "parent geometry fence");
    transfers.pending = false;
    auto x = host.span(kAtom_x), y = host.span(kAtom_y), z = host.span(kAtom_z);
    auto type = host.span(kAtom_type);
    for (int i = 0; i < n; ++i) {
        x[i] = load_position(hi[0].data(), lo[0].data(), i);
        y[i] = load_position(hi[1].data(), lo[1].data(), i);
        z[i] = load_position(hi[2].data(), lo[2].data(), i);
        type[i] = types[i];
    }
    Mat3 matrix;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            matrix(i, j) = cell[3 * i + j];
    host.box.set_cell(matrix);
    return host;
}
class ForceProfileRange {
    cudaStream_t stream_;
    bool active_ = false;

  public:
    explicit ForceProfileRange(cudaStream_t stream) : stream_(stream) {
        if (std::getenv("ATV_GPW_PROFILE_FORCE")) {
            check_cuda(cudaStreamSynchronize(stream_),
                       "force capture initial fence");
            check_cuda(cudaProfilerStart(), "start accepted force capture");
            active_ = true;
        }
    }
    ~ForceProfileRange() {
        if (active_) {
            cudaStreamSynchronize(stream_);
            cudaProfilerStop();
        }
    }
};

struct ScalarPublication {
    double energy;
    double virial[6];
};
__global__ void accumulate_scalars(SystemScalars* out, ScalarPublication value,
                                   bool stress) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out->energy += value.energy;
        if (stress)
            for (int k = 0; k < 6; ++k)
                out->stress[k] += value.virial[k];
    }
}
__global__ void publish_force(cuda::Frame target, const double* force,
                              const double* host_part, int n, double scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    add_force(target.atoms.fx, i,
              (force[3 * i] + (host_part ? host_part[3 * i] : 0)) * scale,
              target.force_fault);
    add_force(target.atoms.fy, i,
              (force[3 * i + 1] + (host_part ? host_part[3 * i + 1] : 0))
                  * scale,
              target.force_fault);
    add_force(target.atoms.fz, i,
              (force[3 * i + 2] + (host_part ? host_part[3 * i + 2] : 0))
                  * scale,
              target.force_fault);
}
void accumulate_host_forces(const std::vector<std::array<double, 3>>& force,
                            cuda::Frame& target, cudaStream_t stream) {
    std::vector<double> packed(3 * force.size());
    for (std::size_t i = 0; i < force.size(); ++i)
        for (int k = 0; k < 3; ++k)
            packed[3 * i + k] = force[i][k];
    struct Buffer {
        double* p = nullptr;
        cudaStream_t stream;
        ~Buffer() {
            cudaStreamSynchronize(stream);
            if (p)
                cudaFree(p);
        }
    } buffer{nullptr, stream};
    check_cuda(cudaMalloc(&buffer.p, packed.size() * sizeof(double)),
               "allocate host force publication");
    check_cuda(cudaMemcpyAsync(buffer.p, packed.data(),
                               packed.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "upload host force contribution");
    publish_force<<<(force.size() + 255) / 256, 256, 0, stream>>>(
        target, buffer.p, nullptr, static_cast<int>(force.size()),
        units::kHartreePerBohrInEvPerAngstrom);
    check_cuda(cudaGetLastError(), "publish host force contribution");
}
} // namespace

struct GpwLoopState {
    GpwStep ks_build, advance;
    bool gate = false;
    GpwStage ks_stage{ks_build, ATV::gpw::Phase::ks_build,
                      "gpw_ks_build_stage"};
    // The existing device operation fuses DIIS, eigen and density update.
    GpwStage eigen_stage{advance, ATV::gpw::Phase::diagonalize,
                         "gpw_diis_eigen_density_stage"};
    GpwConverged criterion{gate};
    void clear() {
        ks_build = {};
        advance = {};
    }
};

GpwDft::GpwDft(std::vector<cpu::gpw::Kind> species, double cutoff,
               cpu::xc::XCType functional)
    : kinds(std::move(species)), cutoff_ry(cutoff), xc(functional),
      loop_(std::make_unique<GpwLoopState>()) {
    this->on(ATV::gpw::Phase::ks_build).add_ref(loop_->ks_stage);
    this->on(ATV::gpw::Phase::diagonalize).add_ref(loop_->eigen_stage);
    this->add_criterion_ref(loop_->criterion);
    cuda::Driver<ATV::gpw::Phase>::build();
}
GpwDft::~GpwDft() {
    try {
        release_backend_();
    } catch (...) {
    }
}
GpwDft::GpwDft(GpwDft&& other)
    : cuda::Driver<ATV::gpw::Phase>(std::move(other)),
      kinds(std::move(other.kinds)), cutoff_ry(other.cutoff_ry), xc(other.xc),
      ngrids(other.ngrids), rel_cutoff(other.rel_cutoff), n_occ(other.n_occ),
      n_alpha(other.n_alpha), n_beta(other.n_beta), max_iter(other.max_iter),
      tol(other.tol), residual_tol(other.residual_tol),
      eps_default(other.eps_default), mix(other.mix),
      diis_depth(other.diis_depth), lindep_threshold(other.lindep_threshold),
      level_shift(other.level_shift), nonperiodic(other.nonperiodic),
      local_pp_on_grid(other.local_pp_on_grid), use_ot(other.use_ot),
      k_mesh{other.k_mesh[0], other.k_mesh[1], other.k_mesh[2]},
      k_points(std::move(other.k_points)),
      k_point_weights(std::move(other.k_point_weights)),
      k_time_reversal(other.k_time_reversal),
      stress_strain_h(other.stress_strain_h), stress_axes(other.stress_axes),
      cell_{other.cell_[0], other.cell_[1], other.cell_[2]},
      result_(std::move(other.result_)), built_(other.built_),
      loop_(std::move(other.loop_)) {
    other.built_ = false;
    other.result_ = GpwResult{};
}
void GpwDft::release_backend_() {
    if (!this->backend_)
        return;
    std::exception_ptr failure;
    try {
        this->backend_->sync();
        this->backend_->destroy_components();
    } catch (...) {
        failure = std::current_exception();
    }
    try {
        this->backend_->shutdown();
    } catch (...) {
        if (!failure)
            failure = std::current_exception();
    }
    this->backend_.reset();
    this->built_plugins_.clear();
    if (failure)
        std::rethrow_exception(failure);
}
void GpwDft::drive_stages_(cuda::Context& ctx, cuda::Frame& frame, int budget) {
    struct Restore {
        cuda::Context& ctx;
        cuda::LoopServices saved;
        GpwLoopState& loop;
        ~Restore() {
            ctx.loop = saved;
            loop.clear();
        }
    } restore{ctx, ctx.loop, *loop_};
    status_.reset();
    metrics_ = ConvergenceMetrics{};
    loop_->gate = false;
    ctx.loop.status = &status_;
    ctx.loop.metrics = &metrics_;
    const int cap = budget < 0 ? max_iter : std::min(budget, max_iter);
    double previous = 0;
    for (int iter = 0; iter < cap; ++iter) {
        mutable_outer_status().reset();
        result_.iters = iter + 1;
        if (this->backend_ && &this->backend_->context() == &ctx
            && &this->backend_->frame() == &frame)
            this->step_once(iter);
        else
            this->step_borrowed_(ctx, frame, iter);
        metrics_.energy = result_.e_total * units::kHartreeInEv;
        metrics_.energy_change = iter ? std::abs(metrics_.energy - previous)
                                      : std::numeric_limits<double>::infinity();
        previous = metrics_.energy;
        if (this->poll_criteria_(iter, frame, ctx) != Verdict::Continue)
            break;
    }
    result_.converged = this->converged();
}

GpwResult& GpwDft::solve_kpoints_(const std::vector<GpwAtom>& atoms,
                                  cuda::Context& ctx, cuda::Frame& frame,
                                  int budget) {
    const auto& cfg = *this;
    const auto stream = static_cast<cudaStream_t>(ctx.stream);
    const std::vector<GridBasisFunction> basis = flatten_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    if (nb < 1)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft::run: empty AO basis (multi-k)");

    std::vector<double> radii(static_cast<std::size_t>(nb), 0.0);
    double rmax = 0.0;
    for (int i = 0; i < nb; ++i) {
        radii[static_cast<std::size_t>(i)] = basis_function_radius(
            basis[static_cast<std::size_t>(i)], cfg.eps_default);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(i)]);
    }
    int mimg[3] = {0, 0, 0};
    pair_image_counts(cfg.cell_, /*periodic=*/true, 2.0 * rmax, mimg);

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
    // Geometry setup shares the execution owner's stream.
    stvnl.compute(stream, basis, radii, atoms, cfg.cell_, mimg, kBuildScreen,
                  k_frac_flat.data(), nk);
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
                const std::size_t idx = static_cast<std::size_t>(ik) * nb * nb
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
    const MultiGrid mg =
        make_multigrid(cfg.cell_, cfg.cutoff_ry, /*nlev=*/1, cfg.cutoff_ry);
    if (mg.levels.empty())
        throw std::runtime_error(
            "cuda::gpw::GpwDft multi-k: make_multigrid returned empty levels");
    const UniformGrid& grid = mg.levels[0];
    for (int d = 0; d < 3; ++d)
        if (grid.n[d] % 2 != 0)
            throw std::runtime_error("cuda::gpw::GpwDft multi-k: grid axis "
                                     + std::to_string(d)
                                     + " has odd n=" + std::to_string(grid.n[d])
                                     + " (device Poisson requires even n)");

    const std::vector<double> n_ion =
        ionic_density(grid, atoms, /*periodic=*/true);
    double e_self = 0.0;
    for (const auto& a : atoms)
        e_self += core_self_energy(a);
    const double e_overlap = cpu::gpw::internal::core_overlap_energy(
        atoms, cfg.cell_, /*periodic=*/true);
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
    const cpu::xc::XCType kxc = to_ks_xc(cfg.xc);
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
        max_m = std::max(
            max_m,
            static_cast<int>(orth[static_cast<std::size_t>(ik)].X.shape(1)));
    KpointDiisEigenDevice kdiis;
    kdiis.build(stream, nk, nb, max_m, use_diis ? cfg.diis_depth : 0, cfg.n_occ,
                /*occ_factor=*/2.0, weights);
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

    GpwResult& res = result_;
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
        std::vector<Cplx> cm(static_cast<std::size_t>(nb)
                             * static_cast<std::size_t>(nb));
        for (int j = 0; j < nb; ++j)
            for (int i = 0; i < nb; ++i)
                cm[static_cast<std::size_t>(j) * nb
                   + static_cast<std::size_t>(i)] = M(i, j);
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

    auto re_frobenius_cm = [&](const void* d_P_cm,
                               const double* d_M_cm) -> double {
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
            check_cuda(cudaStreamSynchronize(stream),
                       "sync multi-k final grids");
            last_veff = std::move(veff);
            last_v_es = std::move(v_es);
        }

        double e_kin = 0.0, e_nl = 0.0;
        for (int ik = 0; ik < nk; ++ik) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            assemble_Hk_into_diis(ik);
            e_kin += wk
                     * re_frobenius_cm(kdiis.d_P(ik),
                                       d_T_k[static_cast<std::size_t>(ik)]);
            e_nl += wk
                    * re_frobenius_cm(kdiis.d_P(ik),
                                      d_Vnl_k[static_cast<std::size_t>(ik)]);
        }
        res.e_kinetic = e_kin;
        res.e_nonlocal = 0.0;
        res.e_loc_sr = e_nl + e.e_locgrid;
        res.e_es = e.e_hartree - e_self + e_overlap;
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
    loop_->ks_build = [&](int it) {
        const double e_total = build_k_energy(/*store_grids=*/false);
        res.e_total = e_total;
        const double err_norm = kdiis.residual_from_device();
        const bool has_prev = !std::isnan(prev_energy);
        const double dE = has_prev ? (e_total - prev_energy)
                                   : std::numeric_limits<double>::infinity();
        prev_energy = e_total;
        metrics_.residual = err_norm;
        loop_->gate =
            has_prev && std::abs(dE) < cfg.tol && err_norm < cfg.residual_tol;
    };
    loop_->advance = [&](int) {
        if (!loop_->gate)
            kdiis.advance();
    };
    drive_stages_(ctx, frame, budget);

    // Final KS at the last density (energy pieces + V_eff for force follow-on).
    {
        const double e_final = build_k_energy(/*store_grids=*/true);
        res.e_total = e_final;
        store_payload();
    }
    return res;
}

void GpwDft::validate_slice_() const {
    if (!loop_)
        throw std::logic_error(
            "cuda::gpw::GpwDft: moved-from method has no loop");
    if (!(residual_tol > 0) || !std::isfinite(residual_tol))
        throw std::invalid_argument("cuda::gpw::GpwDft: residual_tol must be "
                                    "finite and positive [Hartree]");
    cpu::gpw::require_gpw_xc(xc);
#if !defined(ATV_HAS_PW)
    if (cpu::xc::is_pbe_family(xc))
        throw std::invalid_argument("cuda::gpw::GpwDft: PBE-family XC requires "
                                    "ATV_BUILD_PW (shared device XC)");
#endif
    if (is_uks()) {
        if (n_alpha < 0 || n_beta < 0)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: n_alpha and n_beta must be >= 0");
        if (n_alpha + n_beta < 1)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: UKS requires n_alpha+n_beta >= 1");
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
            + std::to_string(ngrids)
            + ", rel_cutoff=" + std::to_string(rel_cutoff)
            + "). Set rel_cutoff (CP2K REL_CUTOFF [Ry]) or ngrids=1.");
    // Multi-k RKS DIIS (cuda-gpw-kpoints): single-grid closed-shell only.
    // Joint paths (UKS×k, MG×k) are deeper slugs.
    if (k_points.size() != k_point_weights.size())
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: k_points and k_point_weights must have the "
            "same length (got "
            + std::to_string(k_points.size()) + " vs "
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
    if (xc != cpu::xc::XCType::PADE && !cpu::xc::is_pbe_family(xc))
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: only Padé LDA and the PBE family have "
            "device "
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
        throw std::invalid_argument("cuda::gpw::GpwDft: max_iter must be >= 0");
    if (diis_depth < 0)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: diis_depth must be >= 0");
    for (int d = 0; d < 3; ++d)
        if (!(cell_[d] > 0.0))
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: periodic cell edge " + std::to_string(d)
                + " must be > 0 Bohr (got " + std::to_string(cell_[d]) + ")");
}

namespace {

std::vector<GpwAtom> atoms_from_frame(const GpwDft& m,
                                      const cpu::Frame& frame) {
    const int n = frame.nlocal();
    if (n < 1)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: empty Frame (nothing to solve) — fill "
            "kAtom_x/y/z and kAtom_type (Z) first");
    const int* sp_type = frame.span(kAtom_type);
    const StorageReal* sp_x = frame.span(kAtom_x);
    const StorageReal* sp_y = frame.span(kAtom_y);
    const StorageReal* sp_z = frame.span(kAtom_z);
    std::vector<GpwAtom> atoms;
    atoms.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int z = sp_type[i];
        const cpu::gpw::Kind* kind = nullptr;
        for (const cpu::gpw::Kind& k : m.kinds)
            if (static_cast<int>(k.element) == z) {
                kind = &k;
                break;
            }
        if (kind == nullptr)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: atom " + std::to_string(i)
                + " has Z=" + std::to_string(z)
                + " but no Kind is configured — add "
                  "kinds.push_back({element, GaussianBasis(...), "
                  "GthPotential(...)})");
        const double pos[3] = {
            static_cast<double>(sp_x[i]) * units::kAngstromInBohr,
            static_cast<double>(sp_y[i]) * units::kAngstromInBohr,
            static_cast<double>(sp_z[i]) * units::kAngstromInBohr};
        atoms.emplace_back(pos, kind->basis, kind->pot, z);
    }
    return atoms;
}

void fill_closed_shell_occ(GpwDft& m, const std::vector<GpwAtom>& atoms) {
    if (m.is_uks() || m.n_occ != 0)
        return;
    double z_val = 0.0;
    for (const GpwAtom& a : atoms)
        z_val += a.z_val;
    const long n_elec = std::lround(z_val);
    if (n_elec % 2 != 0)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft::build: closed-shell RKS needs an even valence "
            "electron count, but the Frame carries "
            + std::to_string(n_elec)
            + " — set n_alpha/n_beta for UKS, or n_occ explicitly");
    m.n_occ = static_cast<int>(n_elec / 2);
}

} // namespace

std::vector<GpwDft::GpwAtom> GpwDft::prepare_(const cpu::Frame& frame) {
    const auto edges = cpu::gpw::internal::cell_from_box(frame.box);
    for (int d = 0; d < 3; ++d)
        cell_[d] = edges[d];
    auto atoms = atoms_from_frame(*this, frame);
    fill_closed_shell_occ(*this, atoms);
    validate_slice_();
    return atoms;
}
void GpwDft::build(const cpu::Frame& reference) {
    prepare_(reference);
    built_ = true;
}

GpwResult& GpwDft::standalone_(const cpu::Frame& geometry,
                               const RealMatrix* guess_C,
                               const RealMatrix* guess_P,
                               const RealMatrix* guess_P_beta,
                               std::vector<std::array<double, 3>>* force,
                               std::array<double, 6>* stress) {
    if (stress)
        cpu::gpw::internal::validate_stress_probe(stress_strain_h, stress_axes,
                                                  "cuda::gpw::GpwDft::stress");
    if (!built_)
        throw std::logic_error(
            "cuda::gpw::GpwDft::run called before build(frame) — validate the "
            "configuration first");
    if (this->borrowed_backend_)
        throw std::logic_error(
            "cuda::gpw::GpwDft: standalone entry while mounted on a parent");
    auto atoms = prepare_(geometry);
    cuda::Driver<ATV::gpw::Phase>::build();
    cpu::Frame scratch(geometry);
    try {
        this->setup_pass(scratch);
        auto& ctx = this->backend_->context();
        const bool resident =
            force && !is_uks() && ngrids == 1 && !uses_kpath();
        solve_(atoms, guess_C, guess_P, guess_P_beta, ctx,
               this->backend_->frame(), max_iter, resident ? force : nullptr);
        if ((force || stress) && !result_.converged)
            throw std::runtime_error(
                "cuda::gpw::GpwDft: derivative requires a converged SCF — "
                "raise max_iter or adjust solver settings");
        if (force && !resident)
            *force = forces_from_state_(atoms, result_,
                                        static_cast<cudaStream_t>(ctx.stream));
        if (stress) {
            *stress = stress_from_state_(atoms, result_);
            for (auto& entry : *stress)
                entry *= units::kHartreePerBohr3InEvPerAngstrom3;
        }
        release_backend_();
    } catch (...) {
        loop_->clear();
        try {
            release_backend_();
        } catch (...) {
        }
        throw;
    }
    return result_;
}
GpwResult& GpwDft::run(const cpu::Frame& frame, const RealMatrix* guess_C,
                       const RealMatrix* guess_P,
                       const RealMatrix* guess_P_beta) {
    return standalone_(frame, guess_C, guess_P, guess_P_beta, nullptr, nullptr);
}
GpwResult& GpwDft::run(cpu::Frame& frame, const RealMatrix* guess_C,
                       const RealMatrix* guess_P,
                       const RealMatrix* guess_P_beta) {
    standalone_(frame, guess_C, guess_P, guess_P_beta, nullptr, nullptr);
    frame.meta.get(kEnergy) = result_.e_total * units::kHartreeInEv;
    return result_;
}
GpwResult& GpwDft::forces(cpu::Frame& frame) {
    std::vector<std::array<double, 3>> force;
    standalone_(frame, nullptr, nullptr, nullptr, &force, nullptr);
    auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
         fz = frame.span(kAtom_fz);
    for (int i = 0; i < frame.nlocal(); ++i) {
        fx[i] = force[i][0] * units::kHartreePerBohrInEvPerAngstrom;
        fy[i] = force[i][1] * units::kHartreePerBohrInEvPerAngstrom;
        fz[i] = force[i][2] * units::kHartreePerBohrInEvPerAngstrom;
    }
    frame.meta.get(kEnergy) = result_.e_total * units::kHartreeInEv;
    return result_;
}
std::array<double, 6> GpwDft::stress(cpu::Frame& frame) {
    std::array<double, 6> sigma{};
    standalone_(frame, nullptr, nullptr, nullptr, nullptr, &sigma);
    frame.meta.get(kEnergy) = result_.e_total * units::kHartreeInEv;
    auto& virial = frame.meta.get(kStress);
    for (std::size_t k = 0; k < 6; ++k)
        virial[k] = sigma[k] * frame.box.volume();
    return sigma;
}
void GpwDft::build_on(cuda::Context& ctx, cuda::Frame& frame) {
    if (!loop_)
        throw std::logic_error(
            "cuda::gpw::GpwDft: moved-from method has no loop");
    this->build_borrowed_(ctx, frame);
}
void GpwDft::destroy_on(cuda::Context& ctx, cuda::Frame& frame) {
    if (loop_)
        loop_->clear();
    this->destroy_borrowed_(ctx, frame);
}
int GpwDft::run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                              int budget) {
    if (!loop_ || !this->borrowed_backend_
        || ctx.exec.backend != this->borrowed_backend_
        || this->borrowed_frame_ != &frame)
        throw std::logic_error(
            "cuda::gpw::GpwDft: nested entry requires a matching parent");
    if (!frame.scalars
        || (ctx.exec.forces_requested
            && (!frame.atoms.fx || !frame.atoms.fy || !frame.atoms.fz)))
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: incomplete parent outputs");
    if (ctx.exec.stress_requested)
        cpu::gpw::internal::validate_stress_probe(stress_strain_h, stress_axes,
                                                  "cuda::gpw::GpwDft::stress");
    const auto stream = static_cast<cudaStream_t>(ctx.stream);
    const auto geometry = host_geometry(frame, stream);
    auto atoms = prepare_(geometry);
    built_ = true;
    const bool resident =
        ctx.exec.forces_requested && !is_uks() && ngrids == 1 && !uses_kpath();
    std::array<double, 6> sigma{};
    solve_(atoms, nullptr, nullptr, nullptr, ctx, frame, budget, nullptr,
           resident ? &frame : nullptr,
           resident && ctx.exec.stress_requested ? &sigma : nullptr);
    if ((ctx.exec.forces_requested || ctx.exec.stress_requested)
        && !result_.converged)
        throw std::runtime_error(
            "cuda::gpw::GpwDft: nested derivatives require a converged SCF");
    if (ctx.exec.stress_requested && !resident)
        sigma = stress_from_state_(atoms, result_);
    if (ctx.exec.forces_requested && !resident)
        forces_from_state_(atoms, result_, stream, &frame);
    ScalarPublication value{};
    value.energy = result_.e_total * units::kHartreeInEv;
    for (int k = 0; k < 6; ++k)
        value.virial[k] = sigma[k] * units::kHartreePerBohr3InEvPerAngstrom3
                          * geometry.box.volume();
    accumulate_scalars<<<1, 1, 0, stream>>>(frame.scalars, value,
                                            ctx.exec.stress_requested);
    check_cuda(cudaGetLastError(), "publish nested GPW scalars");
    return result_.iters;
}

GpwResult& GpwDft::solve_(const std::vector<GpwAtom>& atoms,
                          const RealMatrix* guess_C, const RealMatrix* guess_P,
                          const RealMatrix* guess_P_beta, cuda::Context& ctx,
                          cuda::Frame& frame, int budget,
                          std::vector<std::array<double, 3>>* resident_force,
                          cuda::Frame* resident_target,
                          std::array<double, 6>* resident_stress) {
    struct ClearCallbacks {
        GpwLoopState& loop;
        ~ClearCallbacks() { loop.clear(); }
    } clear{*loop_};
    const auto stream = static_cast<cudaStream_t>(ctx.stream);
    result_ = GpwResult{};
    // Multi-k RKS DIIS branch (cuda-gpw-kpoints): host χ^k + device ρ→V_eff.
    // Γ product path (guess_C/P) is not used here — multi-k cold-starts H0.
    if (uses_kpath()) {
        if (guess_C != nullptr || guess_P != nullptr || guess_P_beta != nullptr)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft::run: multi-k path ignores real Γ "
                "guess_C/guess_P; pass nullptr (complex P_k warm-start is a "
                "deeper slug)");
        return solve_kpoints_(atoms, ctx, frame, budget);
    }

    result_ = GpwResult{};
    const bool periodic = true;
    const double eps = eps_default;
    // Fine monom: mass-margin class (Si SG charge). Coarse monom: collint
    // window class so multi-level residual matches CPU collint MG.
    const double eps_win_fine = eps * kSphereRadiusEpsFine;
    const double eps_win_coarse = eps * kSphereRadiusEpsCoarse;
    const bool uks = is_uks();

    const std::vector<GridBasisFunction> basis = flatten_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    if (nb < 1)
        throw std::invalid_argument("cuda::gpw::GpwDft::run: empty AO basis");

    std::vector<double> radii(static_cast<std::size_t>(nb), 0.0);
    double rmax = 0.0;
    for (int i = 0; i < nb; ++i) {
        radii[static_cast<std::size_t>(i)] =
            basis_function_radius(basis[static_cast<std::size_t>(i)], eps);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(i)]);
    }
    int mimg[3] = {0, 0, 0};
    pair_image_counts(cell_, periodic, 2.0 * rmax, mimg);

    RealMatrix S, T, Vpp;
    build_STVpp(atoms, basis, radii, cell_, periodic, local_pp_on_grid, mimg, S,
                T, Vpp);
    const RealMatrix H0 = T + Vpp;

    // Multigrid product energy when ngrids>1 && rel_cutoff>0 (CPU twin gate).
    // ngrids==1 keeps the single-grid KsChainDevice path (RKS + UKS).
    // UKS + multigrid is rejected in validate_slice_.
    const bool use_mg = ngrids > 1 && rel_cutoff > 0.0;
    const int nlev = use_mg ? ngrids : 1;
    const double rel_for_build = use_mg ? rel_cutoff : cutoff_ry;
    const MultiGrid mg = make_multigrid(cell_, cutoff_ry, nlev, rel_for_build);
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
    const auto core_overlap =
        cpu::gpw::internal::make_core_overlap_model(atoms, cell_, periodic);
    const double e_overlap =
        cpu::gpw::internal::core_overlap_energy(core_overlap);
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

    const bool want_resident_force = resident_force || resident_target;
    std::unique_ptr<internal::GpwForceWorkspace> force_workspace;
    if (want_resident_force) {
        if (uks || use_mg)
            throw std::logic_error(
                "resident GPW force handoff requires Gamma RKS single grid");
        force_workspace = prepare_force_workspace_(atoms, core_overlap, stream);
    }

    // Geometry-fixed product list (per level under MG). Device collocate /
    // integrate consume DeviceProduct views; P enters only as weights.
    const ProductSet pset =
        build_products(mg, basis, cell_, rel_for_build, eps, periodic);
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
    const int n_need = uks ? std::max(n_alpha, n_beta) : n_occ;
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

    const cpu::xc::XCType kxc = to_ks_xc(xc);
    const double* d_vloc_arg = v_locsr.empty() ? nullptr : dev.d_v_loc;
    const double eps_drop = eps * kProductMassMargin;
    const bool use_diis = diis_depth > 0;

    // Single-grid chain (RKS/UKS) vs multi-grid chain (RKS only).
    std::unique_ptr<KsChainDevice> ks_sg;
    std::unique_ptr<KsMultiGridChainDevice> ks_mg;
    MultigridDesc mg_desc;
    if (use_mg) {
        mg_desc.levels.resize(static_cast<std::size_t>(nlevels));
        for (int l = 0; l < nlevels; ++l) {
            // l==0 finest: large monom windows; l>0: collint-class windows.
            const double ew = (l == 0) ? eps_win_fine : eps_win_coarse;
            mg_desc.levels[static_cast<std::size_t>(l)] =
                to_coll(mg.levels[static_cast<std::size_t>(l)], ew);
        }
        ks_mg = std::make_unique<KsMultiGridChainDevice>(mg_desc, stream);
        mg_prods.resize_levels(nlevels);
        for (int l = 0; l < nlevels; ++l)
            mg_prods.ensure_level(l,
                                  nprod_by_level[static_cast<std::size_t>(l)]);
    } else {
        ks_sg = std::make_unique<KsChainDevice>(to_coll(grid, eps_win_fine),
                                                stream);
        dev.ensure_prods(nprod_geom, uks);
    }

    // Collint task ranges (geometry groups) → device for group-max window mag.
    // Built once from ProductSet::tasks_by_level (same order as by_level monoms).
    std::vector<DeviceBuf> task_beg_bufs(static_cast<std::size_t>(nlevels));
    std::vector<DeviceBuf> task_end_bufs(static_cast<std::size_t>(nlevels));
    std::vector<int*> d_task_beg(static_cast<std::size_t>(nlevels), nullptr);
    std::vector<int*> d_task_end(static_cast<std::size_t>(nlevels), nullptr);
    std::vector<int> ntask_by_level(static_cast<std::size_t>(nlevels), 0);
    for (int l = 0; l < nlevels; ++l) {
        const auto& tasks = pset.tasks_by_level[static_cast<std::size_t>(l)];
        const int nt = static_cast<int>(tasks.size());
        ntask_by_level[static_cast<std::size_t>(l)] = nt;
        if (nt <= 0)
            continue;
        std::vector<int> h_beg(static_cast<std::size_t>(nt));
        std::vector<int> h_end(static_cast<std::size_t>(nt));
        for (int t = 0; t < nt; ++t) {
            h_beg[static_cast<std::size_t>(t)] =
                tasks[static_cast<std::size_t>(t)].beg;
            h_end[static_cast<std::size_t>(t)] =
                tasks[static_cast<std::size_t>(t)].end;
        }
        check_cuda(cudaMalloc(&task_beg_bufs[static_cast<std::size_t>(l)].p,
                              static_cast<size_t>(nt) * sizeof(int)),
                   "malloc task_beg");
        check_cuda(cudaMalloc(&task_end_bufs[static_cast<std::size_t>(l)].p,
                              static_cast<size_t>(nt) * sizeof(int)),
                   "malloc task_end");
        d_task_beg[static_cast<std::size_t>(l)] =
            task_beg_bufs[static_cast<std::size_t>(l)].as<int>();
        d_task_end[static_cast<std::size_t>(l)] =
            task_end_bufs[static_cast<std::size_t>(l)].as<int>();
        check_cuda(cudaMemcpyAsync(d_task_beg[static_cast<std::size_t>(l)],
                                   h_beg.data(),
                                   static_cast<size_t>(nt) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D task_beg");
        check_cuda(cudaMemcpyAsync(d_task_end[static_cast<std::size_t>(l)],
                                   h_end.data(),
                                   static_cast<size_t>(nt) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D task_end");
    }

    // ── Device weight pack tables (cuda-gpw-device-weight-pack) ───────────
    // Prim/contrib + per-level product geometry stay resident for the run;
    // each SCF iter only refreshes DeviceProduct.weight from d_P and scatters
    // integrate acc → V_AO on device.
    const int nprim = static_cast<int>(pset.prims.size());
    const int ncontrib = static_cast<int>(pset.contribs.size());
    std::vector<DevicePrimRef> h_prims(static_cast<std::size_t>(nprim));
    for (int i = 0; i < nprim; ++i) {
        h_prims[static_cast<std::size_t>(i)].cbeg =
            pset.prims[static_cast<std::size_t>(i)].cbeg;
        h_prims[static_cast<std::size_t>(i)].cn =
            pset.prims[static_cast<std::size_t>(i)].cn;
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
        check_cuda(
            cudaMalloc(&geom_bufs[static_cast<std::size_t>(l)].p,
                       static_cast<size_t>(np) * sizeof(DeviceProductGeom)),
            "malloc d_geom");
        d_geom_by_level[static_cast<std::size_t>(l)] =
            geom_bufs[static_cast<std::size_t>(l)].as<DeviceProductGeom>();
        check_cuda(
            cudaMemcpyAsync(d_geom_by_level[static_cast<std::size_t>(l)],
                            h_geom.data(),
                            static_cast<size_t>(np) * sizeof(DeviceProductGeom),
                            cudaMemcpyHostToDevice, stream),
            "H2D product geom");
    }
    DeviceProductGeom* d_geom0 = d_geom_by_level[0];

    // T / Vpp on device for mid-SCF energy pieces without downloading P.
    DeviceBuf t_buf;
    DeviceBuf vpp_buf;
    check_cuda(
        cudaMalloc(&t_buf.p, static_cast<size_t>(nb) * nb * sizeof(double)),
        "malloc d_T");
    check_cuda(
        cudaMalloc(&vpp_buf.p, static_cast<size_t>(nb) * nb * sizeof(double)),
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

    // ── Collinear UKS (Padé LSD / spin GGA; dual device DIIS, single grid) ──
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
                throw std::invalid_argument("cuda::gpw::GpwDft::run: UKS "
                                            "guess_P requires guess_P_beta");
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
        auto build_uks_energy_device = [&](const double* d_Pa_use,
                                           const double* d_Pb_use,
                                           bool store_grids) -> double {
            wpack.pack_weights(d_Pa_use, d_geom0, dev.d_prods, nprod_geom,
                               eps_drop);
            wpack.pack_weights(d_Pb_use, d_geom0, dev.d_prods_b, nprod_geom,
                               eps_drop);
            if (ntask_by_level[0] > 0) {
                wpack.apply_group_max_mag(dev.d_prods, d_task_beg[0],
                                          d_task_end[0], ntask_by_level[0]);
                wpack.apply_group_max_mag(dev.d_prods_b, d_task_beg[0],
                                          d_task_end[0], ntask_by_level[0]);
            }

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
                check_cuda(cudaMemcpyAsync(veff_a.data(), ks_sg->d_veff(),
                                           bytes, cudaMemcpyDeviceToHost,
                                           stream),
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
            result_.e_es = e.e_hartree - e_self + e_overlap;
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
        const bool scf_trace_uks = std::getenv("ATV_GPW_CUDA_TRACE") != nullptr;
        loop_->ks_build = [&](int it) {
            const double e_total = build_uks_energy_device(
                ao_a.d_P(), ao_b.d_P(), /*store_grids=*/false);
            result_.e_total = e_total;
            const double err_a = ao_a.residual_device(d_Ha);
            const double err_b = ao_b.residual_device(d_Hb);
            const double err_norm = std::max(err_a, err_b);
            const bool has_prev = !std::isnan(prev_energy);
            const double dE = has_prev
                                  ? (e_total - prev_energy)
                                  : std::numeric_limits<double>::infinity();
            if (scf_trace_uks)
                std::fprintf(stderr,
                             "[cuda gpw uks %3d] E=%.10f  dE=%.3e  "
                             "err_a=%.3e err_b=%.3e\n",
                             it, e_total, has_prev ? dE : 0.0, err_a, err_b);
            prev_energy = e_total;
            metrics_.residual = err_norm;
            loop_->gate =
                has_prev && std::abs(dE) < tol && err_norm < residual_tol;
        };
        loop_->advance = [&](int) {
            if (!loop_->gate) {
                // Both spin equations form one coupled fixed point, matching
                // the CPU dual-spin DIISMixer's shared Pulay coefficients.
                ao_a.advance_paired(ao_b);
            }
        };
        drive_stages_(ctx, frame, budget);
        // Final KS at last density: energy + veff boundary + dual device eigen
        // payload. Device eigen of d_Ha/d_Hb replaces host store_converged_state_uks
        // → dual diagonalize_fock (cuda-gpw-uks-payload-device).
        {
            ao_a.download_P(Pa.data());
            ao_b.download_P(Pb.data());
            const double e_final = build_uks_energy_device(
                ao_a.d_P(), ao_b.d_P(), /*store_grids=*/true);
            result_.e_total = e_final;
            // Device eigen of assembled H = H0+V (already on d_Ha/d_Hb).
            // Boundary D2H of dual C/ε; densified d_P is not SCF energy density.
            result_.C = xt::zeros<double>({nb, n_mo_uks});
            result_.C_beta = xt::zeros<double>({nb, n_mo_uks});
            result_.eps.assign(static_cast<std::size_t>(n_mo_uks), 0.0);
            result_.eps_beta.assign(static_cast<std::size_t>(n_mo_uks), 0.0);
            ao_a.eigen_from_device(d_Ha, result_.eps.data(), result_.C.data());
            ao_b.eigen_from_device(d_Hb, result_.eps_beta.data(),
                                   result_.C_beta.data());
            // H_ks: boundary D2H of assembled device H (no host H0+V rebuild).
            result_.H_ks = xt::zeros<double>({nb, nb});
            result_.H_ks_beta = xt::zeros<double>({nb, nb});
            check_cuda(cudaMemcpyAsync(result_.H_ks.data(), d_Ha, mat_bytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "D2H H_ks alpha payload");
            check_cuda(cudaMemcpyAsync(result_.H_ks_beta.data(), d_Hb,
                                       mat_bytes, cudaMemcpyDeviceToHost,
                                       stream),
                       "D2H H_ks beta payload");
            check_cuda(cudaStreamSynchronize(stream), "sync UKS H_ks payload");
            result_.s2 = spin_contamination_s2(result_.C, result_.C_beta, S,
                                               n_alpha, n_beta);
            result_.P = Pa;
            result_.P_beta = Pb;
            result_.S = S;
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
    if (want_resident_force)
        ao_scf.reserve_force_state();
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
    auto build_energy_device = [&](const double* d_P_use,
                                   bool store_grids) -> double {
        KsChainEnergies e{};

        if (!use_mg) {
            // Single-grid device product KS (cuda-gpw-product-collocate-pbc):
            // pack P→weights → sphere-window PBC collocate → Poisson/XC →
            // integrate → scatter V_AO. No mid-SCF P/ρ/V_AO DH (iron law).
            // Immutable compact-basis geometry is retained for resident forces / payload.
            wpack.pack_weights(d_P_use, d_geom0, dev.d_prods, nprod_geom,
                               eps_drop);
            if (ntask_by_level[0] > 0)
                wpack.apply_group_max_mag(dev.d_prods, d_task_beg[0],
                                          d_task_end[0], ntask_by_level[0]);
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
            // Multigrid product KS: default device monom multi-level
            // pack→KsMultiGridChain→scatter with collint group max|w| + group
            // Ltot window sizing (cuda-gpw-multigrid-device-ks). Host collint
            // mid-path (P D2H + V H2D) is opt-in debug: ATV_GPW_CUDA_HOST_MG=1.
            const bool force_host_mg =
                std::getenv("ATV_GPW_CUDA_HOST_MG") != nullptr;
            if (force_host_mg) {
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
                std::vector<double> rho =
                    multigrid_collocate_rho(mg, pset, nb, Pvec, eps);
                const cpu::gpw::internal::KsGridRks ks =
                    cpu::gpw::internal::ks_grid_build_rks(
                        grid, rho, n_ion, v_locsr, xc, /*nonperiodic=*/false,
                        false, false, false);
                e.e_hartree = ks.e_hartree;
                e.e_xc = ks.e_xc;
                e.e_locgrid = ks.e_locgrid;
                const std::vector<double> Vflat =
                    multigrid_integrate_veff(mg, pset, nb, ks.veff, eps);
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
                                 nlevels, nprod_sum, q_el, e.e_hartree, e.e_xc);
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
                    if (ntask_by_level[static_cast<std::size_t>(l)] > 0)
                        wpack.apply_group_max_mag(
                            d_prods_l, d_task_beg[static_cast<std::size_t>(l)],
                            d_task_end[static_cast<std::size_t>(l)],
                            ntask_by_level[static_cast<std::size_t>(l)]);
                    d_prods_ptr[static_cast<std::size_t>(l)] = d_prods_l;
                    d_acc_ptr[static_cast<std::size_t>(l)] =
                        mg_prods.d_acc[static_cast<std::size_t>(l)];
                }
                e = ks_mg->run(d_prods_ptr.data(), nprod_by_level.data(),
                               dev.d_n_ion, d_vloc_arg, d_acc_ptr.data(), kxc);
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
                                 nlevels, nprod_sum, q_el, e.e_hartree, e.e_xc);
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
                    check_cuda(cudaMemcpyAsync(
                                   veff.data(), ks_mg->fine_chain().d_veff(),
                                   bytes, cudaMemcpyDeviceToHost, stream),
                               "D2H veff fine (boundary)");
                    check_cuda(cudaMemcpyAsync(
                                   v_es.data(), ks_mg->fine_chain().d_v_es(),
                                   bytes, cudaMemcpyDeviceToHost, stream),
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
        result_.e_es = e.e_hartree - e_self + e_overlap;
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
    const bool scf_trace = std::getenv("ATV_GPW_CUDA_TRACE") != nullptr;
    loop_->ks_build = [&](int it) {
        const double e_total =
            build_energy_device(ao_scf.d_P(), /*store_grids=*/false);
        result_.e_total = e_total;
        const double err_norm = ao_scf.residual_device(d_H_work);
        const bool has_prev = !std::isnan(prev_energy);
        const double dE = has_prev ? (e_total - prev_energy)
                                   : std::numeric_limits<double>::infinity();
        if (use_diis && has_prev && dE > kDiisAscentResetHa)
            ao_scf.reset();
        if (scf_trace)
            std::fprintf(stderr,
                         "[cuda gpw scf %3d] E=%.10f  dE=%.3e  err=%.3e\n", it,
                         e_total, has_prev ? dE : 0.0, err_norm);
        prev_energy = e_total;
        metrics_.residual = err_norm;
        loop_->gate = has_prev && std::abs(dE) < tol && err_norm < residual_tol;
    };
    loop_->advance = [&](int) {
        if (!loop_->gate)
            ao_scf.advance();
    };
    drive_stages_(ctx, frame, budget);
    // Final physical KS at accepted P. A force request consumes that device
    // state before any scientific payload is retrieved or any workspace dies.
    {
        const bool live_force = want_resident_force && result_.converged;
        const double e_final =
            build_energy_device(ao_scf.d_P(), /*store_grids=*/!live_force);
        result_.e_total = e_final;
        if (live_force) {
            {
                ForceProfileRange profile(stream);
                ao_scf.finalize_physical_force_state(d_H_work);
                force_workspace->evaluate(ao_scf.d_P(), ao_scf.d_W(),
                                          ks_sg->d_veff(), ks_sg->d_v_es(),
                                          local_pp_on_grid);
            } // End capture before all final Frame / last_result retrieval.
            result_.veff.resize(static_cast<std::size_t>(npts));
            result_.v_es.resize(static_cast<std::size_t>(npts));
            check_cuda(cudaMemcpyAsync(result_.veff.data(), ks_sg->d_veff(),
                                       bytes, cudaMemcpyDeviceToHost, stream),
                       "final veff payload");
            check_cuda(cudaMemcpyAsync(result_.v_es.data(), ks_sg->d_v_es(),
                                       bytes, cudaMemcpyDeviceToHost, stream),
                       "final v_es payload");
        }
        // Accepted P is retained independently of the final raw-H eigenspace.
        ao_scf.download_P(P.data());
        result_.C = xt::zeros<double>({nb, n_mo});
        result_.eps.assign(static_cast<std::size_t>(n_mo), 0.0);
        if (live_force) {
            ao_scf.download_eig(result_.eps.data());
            ao_scf.download_C(result_.C.data());
        } else {
            ao_scf.eigen_from_device(d_H_work, result_.eps.data(),
                                     result_.C.data());
        }
        result_.H_ks = xt::zeros<double>({nb, nb});
        check_cuda(cudaMemcpyAsync(result_.H_ks.data(), d_H_work, mat_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "final H_ks payload");
        check_cuda(cudaStreamSynchronize(stream),
                   "final accepted-state payload fence");
        result_.P = P;
        result_.S = S;
        if (live_force) {
            // Match the original transaction: a fallible requested stress
            // assembly must succeed before parent force columns are touched.
            if (resident_stress)
                *resident_stress = stress_from_state_(atoms, result_);
            if (resident_target) {
                const int na = static_cast<int>(atoms.size());
                publish_force<<<(na + 255) / 256, 256, 0, stream>>>(
                    *resident_target, force_workspace->d_forces(), nullptr, na,
                    units::kHartreePerBohrInEvPerAngstrom);
                check_cuda(cudaGetLastError(), "publish resident GPW forces");
            }
            if (resident_force) {
                std::vector<double> final_force(3 * atoms.size());
                struct ReadbackFence {
                    cudaStream_t stream;
                    ~ReadbackFence() { cudaStreamSynchronize(stream); }
                } fence{stream};
                check_cuda(cudaMemcpyAsync(final_force.data(),
                                           force_workspace->d_forces(),
                                           final_force.size() * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream),
                           "final Frame force boundary");
                check_cuda(cudaStreamSynchronize(stream),
                           "final Frame force fence");
                resident_force->resize(atoms.size());
                for (std::size_t a = 0; a < atoms.size(); ++a)
                    for (int k = 0; k < 3; ++k)
                        (*resident_force)[a][k] = final_force[3 * a + k];
            }
        }
    }
    return result_;
}

namespace {

// ── Cold Gamma RKS force geometry and legacy hybrid-state admission ──

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

} // namespace

std::unique_ptr<internal::GpwForceWorkspace>
GpwDft::prepare_force_workspace_(const std::vector<GpwAtom>& atoms,
                                 const gaussian_core::Model& core_overlap,
                                 cudaStream_t stream) {
    // Geometry-only work, performed before iteration zero. The coefficients
    // below exactly match the old AO Pulay expression and image convention;
    // neither density P nor energy-weighted density W exists on the host here.
    const auto basis = flatten_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    const int na = static_cast<int>(atoms.size());
    const bool periodic = !nonperiodic, analytic_loc = !local_pp_on_grid;
    const auto grid = make_uniform_grid(cell_, cutoff_ry);
    if (!result_.scf_cbg_valid || result_.scf_cbg.npts != grid.npts())
        throw std::logic_error(
            "resident GPW forces require the cold compact basis grid");
    std::vector<int> offsets(atoms.size() + 1, 0),
        owner(static_cast<std::size_t>(nb), 0);
    for (int a = 0; a < na; ++a)
        offsets[a + 1] = offsets[a] + static_cast<int>(atoms[a].basis.size());
    for (int a = 0; a < na; ++a)
        for (int mu = offsets[a]; mu < offsets[a + 1]; ++mu)
            owner[mu] = a;
    std::vector<double> radii(static_cast<std::size_t>(nb));
    std::vector<std::array<GridBasisFunction, 3>> derivative(
        static_cast<std::size_t>(nb));
    std::vector<std::array<double, 3>> derivative_radius(
        static_cast<std::size_t>(nb));
    double rmax = 0, drmax = 0;
    for (int mu = 0; mu < nb; ++mu) {
        radii[mu] = basis_function_radius(basis[mu], eps_default);
        rmax = std::max(rmax, radii[mu]);
        for (int k = 0; k < 3; ++k) {
            derivative[mu][k] = deriv_bf(basis[mu], k);
            derivative_radius[mu][k] =
                basis_function_radius(derivative[mu][k], eps_default);
            drmax = std::max(drmax, derivative_radius[mu][k]);
        }
    }
    int images[3];
    pair_image_counts(cell_, periodic, std::max(2 * rmax, drmax + rmax),
                      images);
    std::vector<const GpwAtom*> all;
    all.reserve(atoms.size());
    bool any_nonlocal = false;
    for (const auto& atom : atoms) {
        all.push_back(&atom);
        any_nonlocal |= !atom.nonlocal.empty();
    }
    std::vector<internal::AoForceEntry> entries;
    // Reserve a small streaming prefix, never a Nat*nb*nb tensor.
    entries.reserve(
        std::min<std::size_t>(3 * basis.size() * basis.size(), 65536));
    for (int a = 0; a < na; ++a)
        for (int k = 0; k < 3; ++k) {
            for (int mu = offsets[a]; mu < offsets[a + 1]; ++mu)
                for (int nu = 0; nu < nb; ++nu) {
                    double overlap = 0, kinetic = 0;
                    const double reach = derivative_radius[mu][k] + radii[nu];
                    contracted_ST_periodic(derivative[mu][k], basis[nu], cell_,
                                           images, reach, overlap, kinetic);
                    const double v =
                        (any_nonlocal || analytic_loc)
                            ? vpp_images(derivative[mu][k], basis[nu], all,
                                         cell_, images, reach, analytic_loc)
                            : 0;
                    const double s = 2 * overlap, p = -2 * kinetic - 2 * v;
                    if (s != 0 || p != 0)
                        entries.push_back({3 * a + k, mu + nu * nb, s, p});
                }
            // Exact-zero absence of a GTH operator is not numerical screening.
            if (atoms[a].nonlocal.empty()
                && (!analytic_loc || atoms[a].cloc.empty()))
                continue;
            const std::vector<const GpwAtom*> only = {&atoms[a]};
            for (int mu = 0; mu < nb; ++mu)
                for (int nu = 0; nu < nb; ++nu) {
                    const double reach = derivative_radius[mu][k] + radii[nu];
                    const double p =
                        2
                        * vpp_images(derivative[mu][k], basis[nu], only, cell_,
                                     images, reach, analytic_loc);
                    if (p != 0)
                        entries.push_back({3 * a + k, mu + nu * nb, 0, p});
                }
        }
    std::vector<GridBasisFunction> derivative_flat(3 * basis.size());
    for (int mu = 0; mu < nb; ++mu)
        for (int k = 0; k < 3; ++k)
            derivative_flat[3 * mu + k] = std::move(derivative[mu][k]);
    result_.force_cbgd =
        build_compact_basis_grid(grid, derivative_flat, periodic, eps_default);
    result_.force_cbgd_valid = true;
    std::vector<internal::DeviceForceAtom> force_atoms(atoms.size());
    for (int a = 0; a < na; ++a) {
        auto& out = force_atoms[a];
        const auto& atom = atoms[a];
        for (int k = 0; k < 3; ++k)
            out.pos[k] = atom.pos[k];
        out.z_val = atom.z_val;
        out.rloc = atom.rloc;
        out.n_cloc = static_cast<int>(std::min(
            atom.cloc.size(), static_cast<std::size_t>(internal::kMaxCloc)));
        for (int k = 0; k < out.n_cloc; ++k)
            out.cloc[k] = atom.cloc[k];
    }
    auto workspace = std::make_unique<internal::GpwForceWorkspace>();
    workspace->build(to_coll(grid), result_.scf_cbg, result_.force_cbgd, owner,
                     force_atoms, entries, core_overlap.pairs(), stream);
    if (std::getenv("ATV_GPW_CUDA_TRACE"))
        std::fprintf(stderr,
                     "[cuda gpw force cold] ao_entries=%zu geometry_bytes=%zu "
                     "scratch_bytes=%zu W_P_bytes=%zu\n",
                     workspace->ao_entries(), workspace->geometry_bytes(),
                     workspace->scratch_bytes(),
                     2 * basis.size() * basis.size() * sizeof(double));
    return workspace;
}

cpu::gpw::GpwDft GpwDft::cpu_shell_() const {
    const auto& m = *this;
    cpu::gpw::GpwDft c(m.kinds, m.cutoff_ry, m.xc);
    cpu::gpw::internal::GpwDftAccess::set_cell(
        c, {m.cell_[0], m.cell_[1], m.cell_[2]});
    c.ngrids = m.ngrids;
    c.rel_cutoff = m.rel_cutoff;
    c.n_occ = m.n_occ;
    c.n_alpha = m.n_alpha;
    c.n_beta = m.n_beta;
    c.max_iter = m.max_iter;
    c.tol = m.tol;
    c.residual_tol = m.residual_tol;
    c.eps_default = m.eps_default;
    c.mix = m.mix;
    c.diis_depth = m.diis_depth;
    c.lindep_threshold = m.lindep_threshold;
    c.level_shift = m.level_shift;
    c.nonperiodic = m.nonperiodic;
    c.local_pp_on_grid = m.local_pp_on_grid;
    c.use_ot = m.use_ot;
    c.k_time_reversal = m.k_time_reversal;
    c.stress_strain_h = m.stress_strain_h;
    c.stress_axes = m.stress_axes;
    c.k_points = m.k_points;
    c.k_point_weights = m.k_point_weights;
    for (int d = 0; d < 3; ++d)
        c.k_mesh[d] = m.k_mesh[d];
    return c;
}

std::vector<std::array<double, 3>>
GpwDft::forces_from_state_(const std::vector<GpwAtom>& atoms, GpwResult& scf,
                           cudaStream_t stream, cuda::Frame* accumulate_into) {
    if (!scf.converged)
        throw std::runtime_error(
            "cuda::gpw::GpwDft::forces: SCF did not converge");
    if (!is_uks() && !uses_kpath() && ngrids == 1) {
        if (scf.P.size() == 0 || scf.C.size() == 0 || scf.eps.empty())
            throw std::runtime_error(
                "cuda::gpw::GpwDft::forces: converged result missing P/C/eps");
        if (scf.veff.empty() || scf.v_es.empty())
            throw std::runtime_error(
                "cuda::gpw::GpwDft::forces: missing SCF grids");
        throw std::runtime_error("cuda::gpw::GpwDft::forces: Gamma single-grid "
                                 "force requires live device SCF state");
    }
    // These explicitly remaining paths consume the original host payload.
    // They do not claim the Gamma single-grid accepted-state residency repair.
    auto shell = cpu_shell_();
    auto force = cpu::gpw::internal::gpw_forces_from_scf(atoms, shell, scf);
    if (accumulate_into) {
        accumulate_host_forces(force, *accumulate_into, stream);
        return {};
    }
    return force;
}

std::array<double, 6>
GpwDft::stress_from_state_(const std::vector<GpwAtom>& atoms,
                           const GpwResult& scf) const {
    if (!scf.converged)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft::stress: supplied state is not converged");
    auto shell = cpu_shell_();
    return cpu::gpw::internal::gpw_stress_at_state_impl(
        atoms, shell, scf, stress_strain_h, stress_axes);
}

} // namespace gpw
} // namespace cuda
} // namespace ATV
