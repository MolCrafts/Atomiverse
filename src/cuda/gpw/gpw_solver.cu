/// @file cuda/gpw/gpw_solver.cu
/// @brief `cuda::gpw::GpwDft` — product slice (Γ RKS/UKS DIIS + device KS).
///
/// Lever (merged multigrid-run + uks-gamma): host STVnl / DIIS / diagonalize
/// compose with geometry-fixed product list + device collocate / KS / integrate.
/// Single-grid RKS: `KsChainDevice::run`. Multigrid RKS (`ngrids>1 &&
/// rel_cutoff>0`): `KsMultiGridChainDevice::run`. Collinear UKS (single-grid
/// Padé LSD): `KsChainDevice::run_uks` + host dual DIIS/eigen. Mid-SCF drops
/// host collocate/integrate and H2D ρ / D2H V_eff for H. Forces free functions
/// stay CPU-backed analytic.

#include "atomiverse/cuda/gpw/gpw_solver.cuh"

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/diis.h"
#include "atomiverse/cpu/gpw/internal/gth_integrals.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cpu/linalg/density.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cuda/gpw/internal/ks_chain.cuh"
#include "atomiverse/cuda/gpw/internal/product_collocate.cuh"

#include <algorithm>
#include <cmath>
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
using cpu::gpw::build_products;
using cpu::gpw::contracted_ST_periodic;
using cpu::gpw::gth_nonlocal;
using cpu::gpw::local_pp_grid;
using cpu::gpw::make_multigrid;
using cpu::gpw::make_uniform_grid;
using cpu::gpw::pair_image_counts;
using cpu::linalg::CanonicalOrth;
using cpu::linalg::RealMatrix;
using cpu::linalg::RealVector;
using cpu::linalg::build_density_matrix;
using cpu::linalg::diagonalize_fock;
using internal::CollocateGridDesc;
using internal::DeviceProduct;
using internal::KsChainDevice;
using internal::KsChainEnergies;
using internal::KsMultiGridChainDevice;
using internal::KsXcKind;
using internal::MultigridDesc;

constexpr double kBuildScreen = 1.0e-14;
constexpr double kDiisErrTol = 1.0e-5;
/// Matches CPU product mass-margin drop (multigrid.cpp); drop monoms below
/// `eps * kProductMassMargin` when packing live density weights.
constexpr double kProductMassMargin = 1.0e-4;

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda::gpw::GpwDft: ") + what
                                 + ": " + cudaGetErrorString(e));
}

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

// Contracted nonlocal GTH (local_pp_on_grid ⇒ local SR is on the grid, not here).
// First-slice H GTH has empty nonlocal — this is a no-op then. When channels
// exist, sums lattice images of the ket against each projector channel.
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
            double v = 0.0;
            for (const auto& at : atoms) {
                if (at.nonlocal.empty())
                    continue;
                for (const auto& pa : basis[static_cast<std::size_t>(i)]) {
                    Prim A;
                    A.alpha = pa.alpha;
                    A.center[0] = pa.center[0];
                    A.center[1] = pa.center[1];
                    A.center[2] = pa.center[2];
                    A.l[0] = pa.l[0];
                    A.l[1] = pa.l[1];
                    A.l[2] = pa.l[2];
                    for (const auto& pb : basis[static_cast<std::size_t>(j)]) {
                        for (int tx = -mimg[0]; tx <= mimg[0]; ++tx)
                            for (int ty = -mimg[1]; ty <= mimg[1]; ++ty)
                                for (int tz = -mimg[2]; tz <= mimg[2]; ++tz) {
                                    Prim B;
                                    B.alpha = pb.alpha;
                                    B.center[0] = pb.center[0] + tx * cell[0];
                                    B.center[1] = pb.center[1] + ty * cell[1];
                                    B.center[2] = pb.center[2] + tz * cell[2];
                                    B.l[0] = pb.l[0];
                                    B.l[1] = pb.l[1];
                                    B.l[2] = pb.l[2];
                                    const double dx = A.center[0] - B.center[0];
                                    const double dy = A.center[1] - B.center[1];
                                    const double dz = A.center[2] - B.center[2];
                                    const double r2 =
                                        dx * dx + dy * dy + dz * dz;
                                    if (r2 > reach * reach)
                                        continue;
                                    for (const auto& ch : at.nonlocal)
                                        v += pa.coeff * pb.coeff
                                             * gth_nonlocal(A, B, at.pos, ch);
                                }
                    }
                }
            }
            Vpp(i, j) += v;
            if (i != j)
                Vpp(j, i) += v;
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

std::pair<RealVector, RealMatrix>
diagonalize_fock_shifted(const RealMatrix& F, const RealMatrix& X,
                         const RealMatrix& S, const RealMatrix& C_prev,
                         int n_occ, double shift) {
    if (shift <= 0.0 || C_prev.size() == 0 || n_occ <= 0)
        return diagonalize_fock(F, X);

    const int m = static_cast<int>(X.shape(1));
    RealMatrix Fp = xt::linalg::dot(xt::transpose(X), xt::linalg::dot(F, X));
    const RealMatrix C_occ = xt::view(C_prev, xt::all(), xt::range(0, n_occ));
    const RealMatrix Cp_occ =
        xt::linalg::dot(xt::transpose(X), xt::linalg::dot(S, C_occ));
    const RealMatrix Pocc = xt::linalg::dot(Cp_occ, xt::transpose(Cp_occ));
    for (int i = 0; i < m; ++i) {
        Fp(i, i) += shift;
        for (int j = 0; j < m; ++j)
            Fp(i, j) -= shift * Pocc(i, j);
    }
    auto [eig, Cp] = xt::linalg::eigh(Fp);
    RealMatrix C = xt::linalg::dot(X, Cp);
    for (int i = n_occ; i < m; ++i)
        eig(i) -= shift;
    return {eig, C};
}

void store_converged_state(GpwResult& res, const RealMatrix& H,
                           const RealMatrix& X, int n_occ, int nb) {
    auto [eig, C] = diagonalize_fock(H, X);
    res.C = std::move(C);
    res.eps.assign(eig.begin(), eig.end());
    res.P = build_density_matrix(res.C, n_occ, nb, 2.0);
    res.H_ks = H;
}

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

// Map CUDA method members onto a CPU GpwDft for force assembly only.
cpu::gpw::GpwDft to_cpu_cfg(const GpwDft& m) {
    cpu::gpw::GpwDft c;
    for (int d = 0; d < 3; ++d) {
        c.cell[d] = m.cell[d];
        c.k_mesh[d] = m.k_mesh[d];
    }
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
    return c;
}

CollocateGridDesc to_coll(const UniformGrid& g) {
    CollocateGridDesc d;
    for (int i = 0; i < 3; ++i) {
        d.n[i] = g.n[i];
        d.L[i] = g.L[i];
        d.origin[i] = g.origin[i];
    }
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

/// Density weight w = k · Σ P[μ,ν] coeff_μ coeff_ν (CPU product path twin).
double product_density_weight(const GridProduct& prod,
                              const std::vector<PrimRef>& prims,
                              const std::vector<PrimContrib>& contribs, int nb,
                              const std::vector<double>& P) {
    const PrimRef& A = prims[static_cast<std::size_t>(prod.ua)];
    const PrimRef& B = prims[static_cast<std::size_t>(prod.ub)];
    double s = 0.0;
    for (int i = 0; i < A.cn; ++i) {
        const PrimContrib& ca = contribs[static_cast<std::size_t>(A.cbeg + i)];
        const std::size_t row = static_cast<std::size_t>(ca.mu) * nb;
        double rs = 0.0;
        for (int j = 0; j < B.cn; ++j) {
            const PrimContrib& cb =
                contribs[static_cast<std::size_t>(B.cbeg + j)];
            rs += P[row + static_cast<std::size_t>(cb.mu)] * cb.coeff;
        }
        s += ca.coeff * rs;
    }
    return s * prod.k;
}

DeviceProduct to_device_product(const GridProduct& prod, double weight) {
    DeviceProduct q;
    q.p = prod.p;
    q.weight = weight;
    for (int d = 0; d < 3; ++d) {
        q.pc[d] = prod.pc[d];
        q.ra[d] = prod.ra[d];
        q.rb[d] = prod.rb[d];
        q.la[d] = prod.la[d];
        q.lb[d] = prod.lb[d];
    }
    return q;
}

/// Scatter bare product integrals (Σ_g V·shape) into V_μν with k · w · coeffs.
RealMatrix scatter_product_acc(const std::vector<GridProduct>& prods,
                               const std::vector<PrimRef>& prims,
                               const std::vector<PrimContrib>& contribs, int nb,
                               const std::vector<double>& acc, double w_grid) {
    RealMatrix V = xt::zeros<double>({nb, nb});
    const int nprod = static_cast<int>(prods.size());
    for (int ip = 0; ip < nprod; ++ip) {
        const double ig = acc[static_cast<std::size_t>(ip)];
        if (ig == 0.0)
            continue;
        const GridProduct& prod = prods[static_cast<std::size_t>(ip)];
        const PrimRef& A = prims[static_cast<std::size_t>(prod.ua)];
        const PrimRef& B = prims[static_cast<std::size_t>(prod.ub)];
        const double sw = prod.k * w_grid * ig;
        for (int i = 0; i < A.cn; ++i) {
            const PrimContrib& ca =
                contribs[static_cast<std::size_t>(A.cbeg + i)];
            const double cas = ca.coeff * sw;
            for (int j = 0; j < B.cn; ++j) {
                const PrimContrib& cb =
                    contribs[static_cast<std::size_t>(B.cbeg + j)];
                V(ca.mu, cb.mu) += cas * cb.coeff;
            }
        }
    }
    return V;
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
    if (k_mesh[0] != 1 || k_mesh[1] != 1 || k_mesh[2] != 1)
        throw std::invalid_argument(
            "cuda::gpw::GpwDft: k-point sampling has no device product path "
            "yet — keep k_mesh={1,1,1} (Γ) or use cpu::gpw::GpwDft.");
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
    // Single-grid product list alias (UKS + RKS ngrids==1).
    const std::vector<GridProduct>& prods = pset.by_level[0];
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

    // Device KS workspace: n_ion / v_loc on fine grid once.
    DeviceKsBuffers dev;
    DeviceMgProdBuffers mg_prods;
    const long npts = grid.npts();
    const double w_fine = grid.weight();
    const double w_grid = w_fine; // UKS single-grid alias
    dev.alloc_grids(npts);
    const size_t bytes = static_cast<size_t>(npts) * sizeof(double);
    check_cuda(cudaMemcpy(dev.d_n_ion, n_ion.data(), bytes,
                          cudaMemcpyHostToDevice),
               "H2D n_ion");
    if (!v_locsr.empty())
        check_cuda(cudaMemcpy(dev.d_v_loc, v_locsr.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "H2D v_loc");
    else
        check_cuda(cudaMemset(dev.d_v_loc, 0, bytes), "zero v_loc");

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
        ks_mg = std::make_unique<KsMultiGridChainDevice>(mg_desc,
                                                         /*stream=*/nullptr);
        mg_prods.resize_levels(nlevels);
        for (int l = 0; l < nlevels; ++l)
            mg_prods.ensure_level(l, nprod_by_level[static_cast<std::size_t>(l)]);
    } else {
        ks_sg = std::make_unique<KsChainDevice>(to_coll(grid),
                                                /*stream=*/nullptr);
        dev.ensure_prods(nprod_geom, uks);
    }

    // Host scratch for packing products / reading integrate acc each SCF step.
    std::vector<std::vector<DeviceProduct>> h_prods_by_level(
        static_cast<std::size_t>(nlevels));
    std::vector<std::vector<double>> h_acc_by_level(
        static_cast<std::size_t>(nlevels));
    for (int l = 0; l < nlevels; ++l) {
        const int np = nprod_by_level[static_cast<std::size_t>(l)];
        h_prods_by_level[static_cast<std::size_t>(l)].resize(
            static_cast<std::size_t>(np));
        h_acc_by_level[static_cast<std::size_t>(l)].assign(
            static_cast<std::size_t>(np), 0.0);
    }
    // UKS β scratch (single-grid only; same length as level-0).
    std::vector<DeviceProduct> h_prods_b(static_cast<std::size_t>(nprod_geom));
    std::vector<double> h_acc_b(static_cast<std::size_t>(nprod_geom), 0.0);
    auto& h_prods = h_prods_by_level[0];
    auto& h_acc = h_acc_by_level[0];

    auto pack_products = [&](const RealMatrix& Pin,
                             std::vector<DeviceProduct>& out) {
        const std::vector<double> Pvec = mat_to_vec(Pin, nb);
        for (int ip = 0; ip < nprod_geom; ++ip) {
            const GridProduct& gp = prods[static_cast<std::size_t>(ip)];
            double w = product_density_weight(gp, pset.prims, pset.contribs, nb,
                                              Pvec);
            if (std::abs(w) < eps_drop)
                w = 0.0;
            out[static_cast<std::size_t>(ip)] = to_device_product(gp, w);
        }
    };

    // ── Collinear UKS (Padé LSD + dual DIIS; single-grid only) ────────────
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
        RealMatrix Ca_prev = C0;
        RealMatrix Cb_prev = C0;

        cpu::gpw::internal::GpwDiis diis_a(use_diis ? diis_depth : 1);
        cpu::gpw::internal::GpwDiis diis_b(use_diis ? diis_depth : 1);

        auto build_uks_H =
            [&](const RealMatrix& Pa_in, const RealMatrix& Pb_in,
                bool store_grids)
            -> std::tuple<RealMatrix, RealMatrix, double> {
            pack_products(Pa_in, h_prods);
            pack_products(Pb_in, h_prods_b);
            check_cuda(cudaMemcpy(dev.d_prods, h_prods.data(),
                                  static_cast<size_t>(nprod_geom)
                                      * sizeof(DeviceProduct),
                                  cudaMemcpyHostToDevice),
                       "H2D products_a");
            check_cuda(cudaMemcpy(dev.d_prods_b, h_prods_b.data(),
                                  static_cast<size_t>(nprod_geom)
                                      * sizeof(DeviceProduct),
                                  cudaMemcpyHostToDevice),
                       "H2D products_b");

            const KsChainEnergies e = ks_sg->run_uks(
                dev.d_prods, nprod_geom, dev.d_prods_b, nprod_geom, dev.d_n_ion,
                d_vloc_arg, dev.d_acc, dev.d_acc_b, kxc);

            check_cuda(cudaMemcpy(h_acc.data(), dev.d_acc,
                                  static_cast<size_t>(nprod_geom)
                                      * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "D2H product acc_a");
            check_cuda(cudaMemcpy(h_acc_b.data(), dev.d_acc_b,
                                  static_cast<size_t>(nprod_geom)
                                      * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "D2H product acc_b");
            const RealMatrix Va = scatter_product_acc(
                prods, pset.prims, pset.contribs, nb, h_acc, w_grid);
            const RealMatrix Vb = scatter_product_acc(
                prods, pset.prims, pset.contribs, nb, h_acc_b, w_grid);
            const RealMatrix Ha = H0 + Va;
            const RealMatrix Hb = H0 + Vb;

            if (store_grids) {
                std::vector<double> veff_a(static_cast<std::size_t>(npts));
                std::vector<double> veff_b(static_cast<std::size_t>(npts));
                std::vector<double> v_es(static_cast<std::size_t>(npts));
                check_cuda(cudaMemcpy(veff_a.data(), ks_sg->d_veff(), bytes,
                                      cudaMemcpyDeviceToHost),
                           "D2H veff_a");
                check_cuda(cudaMemcpy(veff_b.data(), ks_sg->d_veff_b(), bytes,
                                      cudaMemcpyDeviceToHost),
                           "D2H veff_b");
                check_cuda(cudaMemcpy(v_es.data(), ks_sg->d_v_es(), bytes,
                                      cudaMemcpyDeviceToHost),
                           "D2H v_es");
                result_.veff_a = std::move(veff_a);
                result_.veff_b = std::move(veff_b);
                result_.v_es = std::move(v_es);
                // RKS veff slot unused for UKS force path (CPU uses veff_a/b).
                result_.veff.clear();
            }

            const RealMatrix Ptot = Pa_in + Pb_in;
            result_.e_kinetic = trace_product(Ptot, T, nb);
            result_.e_nonlocal = 0.0;
            result_.e_loc_sr = trace_product(Ptot, Vpp, nb) + e.e_locgrid;
            result_.e_es = e.e_hartree - e_self;
            result_.e_xc = e.e_xc;
            const double e_total = result_.e_kinetic + result_.e_loc_sr
                                   + result_.e_es + result_.e_xc;
            return {Ha, Hb, e_total};
        };

        if (max_iter == 0) {
            if (guess_P == nullptr || guess_P_beta == nullptr)
                throw std::invalid_argument(
                    "cuda::gpw::GpwDft: max_iter==0 UKS requires guess_P and "
                    "guess_P_beta");
            auto [Ha, Hb, e_total] =
                build_uks_H(Pa, Pb, /*store_grids=*/true);
            result_.e_total = e_total;
            result_.P = Pa;
            result_.P_beta = Pb;
            result_.H_ks = std::move(Ha);
            result_.H_ks_beta = std::move(Hb);
            result_.S = S;
            result_.converged = true;
            result_.iters = 0;
            return result_;
        }

        double prev_energy = std::numeric_limits<double>::quiet_NaN();
        RealMatrix Ha_raw, Hb_raw;
        bool conv = false;
        int it = 0;
        for (it = 0; it < max_iter; ++it) {
            auto [Ha, Hb, e_total] =
                build_uks_H(Pa, Pb, /*store_grids=*/false);
            Ha_raw = Ha;
            Hb_raw = Hb;
            result_.e_total = e_total;
            const RealMatrix err_a =
                cpu::gpw::internal::gpw_diis_error(Ha, Pa, S, orth.X);
            const RealMatrix err_b =
                cpu::gpw::internal::gpw_diis_error(Hb, Pb, S, orth.X);
            double err_norm = 0.0;
            for (const double v : err_a)
                err_norm = std::max(err_norm, std::abs(v));
            for (const double v : err_b)
                err_norm = std::max(err_norm, std::abs(v));
            const bool has_prev = !std::isnan(prev_energy);
            const double dE =
                has_prev ? (e_total - prev_energy)
                         : std::numeric_limits<double>::infinity();
            prev_energy = e_total;
            if (has_prev && std::abs(dE) < tol && err_norm < kDiisErrTol) {
                conv = true;
                break;
            }
            RealMatrix Ha_use, Hb_use;
            if (use_diis) {
                Ha_use = diis_a.step(Ha, err_a);
                Hb_use = diis_b.step(Hb, err_b);
            } else {
                Ha_use = Ha;
                Hb_use = Hb;
            }
            auto [eiga, Ca] = diagonalize_fock_shifted(
                Ha_use, orth.X, S, Ca_prev, n_alpha, level_shift);
            auto [eigb, Cb] = diagonalize_fock_shifted(
                Hb_use, orth.X, S, Cb_prev, n_beta, level_shift);
            (void)eiga;
            (void)eigb;
            Ca_prev = std::move(Ca);
            Cb_prev = std::move(Cb);
            RealMatrix Pa_new = build_density_matrix(Ca_prev, n_alpha, nb, 1.0);
            RealMatrix Pb_new = build_density_matrix(Cb_prev, n_beta, nb, 1.0);
            if (use_diis) {
                Pa = std::move(Pa_new);
                Pb = std::move(Pb_new);
            } else {
                Pa = (1.0 - mix) * Pa + mix * Pa_new;
                Pb = (1.0 - mix) * Pb + mix * Pb_new;
            }
        }
        {
            auto [Ha_final, Hb_final, e_final] =
                build_uks_H(Pa, Pb, /*store_grids=*/true);
            result_.e_total = e_final;
            Ha_raw = std::move(Ha_final);
            Hb_raw = std::move(Hb_final);
            store_converged_state_uks(result_, Ha_raw, Hb_raw, orth.X, S,
                                      n_alpha, n_beta, nb);
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
    RealMatrix C_prev = C0;

    cpu::gpw::internal::GpwDiis diis(use_diis ? diis_depth : 1);

    auto build_H_and_energy =
        [&](const RealMatrix& Pin,
            bool store_grids) -> std::pair<RealMatrix, double> {
        const std::vector<double> Pvec = mat_to_vec(Pin, nb);
        KsChainEnergies e{};
        RealMatrix Vgrid = xt::zeros<double>({nb, nb});

        if (!use_mg) {
            // ── Single-grid device product path ──────────────────────────
            for (int ip = 0; ip < nprod_geom; ++ip) {
                const GridProduct& gp = prods[static_cast<std::size_t>(ip)];
                double w = product_density_weight(gp, pset.prims, pset.contribs,
                                                  nb, Pvec);
                if (std::abs(w) < eps_drop)
                    w = 0.0;
                h_prods[static_cast<std::size_t>(ip)] = to_device_product(gp, w);
            }
            check_cuda(cudaMemcpy(dev.d_prods, h_prods.data(),
                                  static_cast<size_t>(nprod_geom)
                                      * sizeof(DeviceProduct),
                                  cudaMemcpyHostToDevice),
                       "H2D products");
            e = ks_sg->run(dev.d_prods, nprod_geom, dev.d_n_ion, d_vloc_arg,
                           dev.d_acc, kxc);
            check_cuda(cudaMemcpy(h_acc.data(), dev.d_acc,
                                  static_cast<size_t>(nprod_geom)
                                      * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "D2H product acc");
            Vgrid = scatter_product_acc(prods, pset.prims, pset.contribs, nb,
                                        h_acc, w_fine);
            if (store_grids) {
                std::vector<double> veff(static_cast<std::size_t>(npts));
                std::vector<double> v_es(static_cast<std::size_t>(npts));
                check_cuda(cudaMemcpy(veff.data(), ks_sg->d_veff(), bytes,
                                      cudaMemcpyDeviceToHost),
                           "D2H veff");
                check_cuda(cudaMemcpy(v_es.data(), ks_sg->d_v_es(), bytes,
                                      cudaMemcpyDeviceToHost),
                           "D2H v_es");
                result_.veff = std::move(veff);
                result_.v_es = std::move(v_es);
            }
        } else {
            // ── Multigrid: per-level pack → KsMultiGridChainDevice ───────
            std::vector<DeviceProduct*> d_prods_ptr(
                static_cast<std::size_t>(nlevels), nullptr);
            std::vector<double*> d_acc_ptr(
                static_cast<std::size_t>(nlevels), nullptr);
            for (int l = 0; l < nlevels; ++l) {
                const int np = nprod_by_level[static_cast<std::size_t>(l)];
                const std::vector<GridProduct>& prods_l =
                    pset.by_level[static_cast<std::size_t>(l)];
                auto& h_prods_l = h_prods_by_level[static_cast<std::size_t>(l)];
                for (int ip = 0; ip < np; ++ip) {
                    const GridProduct& gp =
                        prods_l[static_cast<std::size_t>(ip)];
                    double w = product_density_weight(
                        gp, pset.prims, pset.contribs, nb, Pvec);
                    if (std::abs(w) < eps_drop)
                        w = 0.0;
                    h_prods_l[static_cast<std::size_t>(ip)] =
                        to_device_product(gp, w);
                }
                if (np > 0) {
                    check_cuda(
                        cudaMemcpy(
                            mg_prods.d_prods[static_cast<std::size_t>(l)],
                            h_prods_l.data(),
                            static_cast<size_t>(np) * sizeof(DeviceProduct),
                            cudaMemcpyHostToDevice),
                        "H2D products level");
                    d_prods_ptr[static_cast<std::size_t>(l)] =
                        mg_prods.d_prods[static_cast<std::size_t>(l)];
                    d_acc_ptr[static_cast<std::size_t>(l)] =
                        mg_prods.d_acc[static_cast<std::size_t>(l)];
                }
            }
            e = ks_mg->run(d_prods_ptr.data(), nprod_by_level.data(),
                           dev.d_n_ion, d_vloc_arg, d_acc_ptr.data(), kxc);
            for (int l = 0; l < nlevels; ++l) {
                const int np = nprod_by_level[static_cast<std::size_t>(l)];
                if (np <= 0)
                    continue;
                auto& h_acc_l = h_acc_by_level[static_cast<std::size_t>(l)];
                check_cuda(
                    cudaMemcpy(h_acc_l.data(),
                               mg_prods.d_acc[static_cast<std::size_t>(l)],
                               static_cast<size_t>(np) * sizeof(double),
                               cudaMemcpyDeviceToHost),
                    "D2H product acc level");
                const double w_l =
                    mg.levels[static_cast<std::size_t>(l)].weight();
                const RealMatrix Vl = scatter_product_acc(
                    pset.by_level[static_cast<std::size_t>(l)], pset.prims,
                    pset.contribs, nb, h_acc_l, w_l);
                Vgrid = Vgrid + Vl;
            }
            if (store_grids) {
                std::vector<double> veff(static_cast<std::size_t>(npts));
                std::vector<double> v_es(static_cast<std::size_t>(npts));
                check_cuda(cudaMemcpy(veff.data(),
                                      ks_mg->fine_chain().d_veff(), bytes,
                                      cudaMemcpyDeviceToHost),
                           "D2H veff fine");
                check_cuda(cudaMemcpy(v_es.data(),
                                      ks_mg->fine_chain().d_v_es(), bytes,
                                      cudaMemcpyDeviceToHost),
                           "D2H v_es fine");
                result_.veff = std::move(veff);
                result_.v_es = std::move(v_es);
            }
        }

        const RealMatrix H = H0 + Vgrid;
        result_.e_kinetic = trace_product(Pin, T, nb);
        result_.e_nonlocal = 0.0;
        result_.e_loc_sr = trace_product(Pin, Vpp, nb) + e.e_locgrid;
        result_.e_es = e.e_hartree - e_self;
        result_.e_xc = e.e_xc;
        const double e_total =
            result_.e_kinetic + result_.e_loc_sr + result_.e_es + result_.e_xc;
        return {H, e_total};
    };

    // Fixed-density path (Car–Parrinello-class single KS eval).
    if (max_iter == 0) {
        if (guess_P == nullptr)
            throw std::invalid_argument(
                "cuda::gpw::GpwDft: max_iter==0 requires guess_P");
        auto [H, e_total] = build_H_and_energy(P, /*store_grids=*/true);
        result_.e_total = e_total;
        result_.P = P;
        result_.H_ks = std::move(H);
        result_.S = S;
        result_.converged = true;
        result_.iters = 0;
        return result_;
    }

    double prev_energy = std::numeric_limits<double>::quiet_NaN();
    RealMatrix H_raw;
    bool conv = false;
    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        // Mid-SCF: device collocate/integrate only — skip D2H V_eff/V_es.
        auto [H, e_total] = build_H_and_energy(P, /*store_grids=*/false);
        H_raw = H;
        result_.e_total = e_total;
        const RealMatrix err =
            cpu::gpw::internal::gpw_diis_error(H, P, S, orth.X);
        double err_norm = 0.0;
        for (const double v : err)
            err_norm = std::max(err_norm, std::abs(v));
        const bool has_prev = !std::isnan(prev_energy);
        const double dE =
            has_prev ? (e_total - prev_energy)
                     : std::numeric_limits<double>::infinity();
        prev_energy = e_total;
        if (has_prev && std::abs(dE) < tol && err_norm < kDiisErrTol) {
            conv = true;
            break;
        }
        RealMatrix H_ext;
        if (use_diis)
            H_ext = diis.step(H, err);
        else
            H_ext = H;
        auto [eig, C] = diagonalize_fock_shifted(H_ext, orth.X, S, C_prev,
                                                 n_occ, level_shift);
        (void)eig;
        C_prev = C;
        RealMatrix P_new = build_density_matrix(C, n_occ, nb, 2.0);
        if (use_diis)
            P = std::move(P_new);
        else
            P = (1.0 - mix) * P + mix * P_new;
    }
    // Final KS at the last density: energy pieces + V_eff/V_es for forces.
    {
        auto [H_final, e_final] = build_H_and_energy(P, /*store_grids=*/true);
        result_.e_total = e_final;
        H_raw = H_final;
        store_converged_state(result_, H_raw, orth.X, n_occ, nb);
        result_.S = S;
        result_.converged = conv;
        if (conv)
            result_.iters = it + 1;
        else
            result_.iters = max_iter;
    }
    return result_;
}

std::vector<std::array<double, 3>>
gpw_forces_from_scf(const std::vector<GpwAtom>& atoms, const GpwDft& method,
                    const GpwResult& scf) {
    cpu::gpw::GpwDft cfg = to_cpu_cfg(method);
    // Force assembly is the public CPU free function — same Hamiltonian
    // pieces as the CUDA SCF when members match.
    return cpu::gpw::gpw_forces_from_scf(atoms, cfg, scf);
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

} // namespace gpw
} // namespace cuda
} // namespace ATV
