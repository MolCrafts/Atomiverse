/// @file cpu/scf/xc_grid_engine.cpp
/// @brief Blocked molecular XC integration (see xc_grid_engine.h).

#include "atomiverse/cpu/scf/xc_grid_engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "atomiverse/cpu/linalg/internal/blas.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace scf {

namespace {
using linalg::internal::gemm;

int thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int thread_index() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/// Per-thread block scratch, grown on demand and reused across blocks.
struct Scratch {
    std::vector<double> P[2];     // gathered m × m spin densities
    std::vector<double> T[2];     // φ P_sub, ng × m
    std::vector<double> X;        // ½ w v_ρ φ + w g·∇φ, ng × m
    std::vector<double> V;        // φᵀ X, m × m
    std::vector<double> rho[2];   // ng
    std::vector<double> grad[2];  // 3 × ng, direction-major
    std::vector<double> sigma[3]; // aa, ab, bb
    std::vector<double> f, vrho[2], vsigma[3];
    std::vector<double> wt;        // quadrature weights of the block
    std::vector<double> grad_coef; // w g_d, 3 × ng
};

void grow(std::vector<double>& v, std::size_t n) {
    if (v.size() < n)
        v.resize(n);
}
} // namespace

void XCGridBasis::build(const MolecularGrid& grid,
                        const std::vector<native::Shell>& shells, int n_ao,
                        bool is_gga) {
    const int G = static_cast<int>(grid.points.size());
    n_basis = n_ao;
    gga = is_gga;
    const int nblk = (G + kBlockSize - 1) / kBlockSize;
    blocks.assign(static_cast<std::size_t>(nblk), Block{});
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        // Full-width staging of one block, then compaction to the AOs whose
        // value or gradient exceeds the cutoff anywhere on the block.
        std::vector<double> v, gx, gy, gz;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int b = 0; b < nblk; ++b) {
            Block& blk = blocks[static_cast<std::size_t>(b)];
            blk.g0 = b * kBlockSize;
            blk.ng = std::min(kBlockSize, G - blk.g0);
            const std::size_t ng = static_cast<std::size_t>(blk.ng);
            const std::size_t full = ng * static_cast<std::size_t>(n_ao);
            v.assign(full, 0.0);
            if (is_gga) {
                gx.assign(full, 0.0);
                gy.assign(full, 0.0);
                gz.assign(full, 0.0);
            }
            for (int mu = 0; mu < n_ao; ++mu) {
                const auto& sh = shells[static_cast<std::size_t>(mu)];
                for (std::size_t g = 0; g < ng; ++g) {
                    const auto& pt = grid.points[blk.g0 + g];
                    const std::size_t at = mu * ng + g;
                    if (is_gga)
                        eval_basis_val_grad(sh, pt.x, pt.y, pt.z, v[at], gx[at],
                                            gy[at], gz[at]);
                    else
                        v[at] = eval_basis(sh, pt.x, pt.y, pt.z);
                }
            }
            blk.ao.clear();
            for (int mu = 0; mu < n_ao; ++mu) {
                double m = 0.0;
                for (std::size_t g = 0; g < ng; ++g) {
                    const std::size_t at = mu * ng + g;
                    m = std::max(m, std::abs(v[at]));
                    if (is_gga)
                        m = std::max({m, std::abs(gx[at]), std::abs(gy[at]),
                                      std::abs(gz[at])});
                }
                if (m > kAoCutoff)
                    blk.ao.push_back(mu);
            }
            const std::size_t m = blk.ao.size();
            const auto compact = [&](const std::vector<double>& src,
                                     RealMatrix& dst) {
                dst = RealMatrix::from_shape({ng, m});
                for (std::size_t k = 0; k < m; ++k)
                    std::copy_n(src.data() + blk.ao[k] * ng, ng,
                                dst.data() + k * ng);
            };
            compact(v, blk.phi);
            if (is_gga) {
                compact(gx, blk.dx);
                compact(gy, blk.dy);
                compact(gz, blk.dz);
            } else {
                blk.dx = blk.dy = blk.dz = RealMatrix{};
            }
        }
    }
    built = true;
}

XCGridResult
evaluate_xc_batched(const XCGridBasis& aog, const MolecularGrid& grid,
                    const xt::xtensor<double, 2>& P_alpha,
                    const xt::xtensor<double, 2>& P_beta,
                    const ATV::electronic::XcFunctional& functional,
                    ATV::electronic::XcSpin spin) {
    const auto mask = static_cast<unsigned>(functional.ingredients());
    if ((mask & 1u) == 0 || (mask & ~3u) != 0)
        throw std::invalid_argument(
            "Molecular XC: only rho and sigma channels are implemented");
    const bool is_gga = (mask & 2u) != 0;
    if (is_gga && !aog.gga)
        throw std::invalid_argument(
            "Molecular XC: GGA functional on an LDA-only AO basis");
    const int N = aog.n_basis;
    const std::size_t Ns = static_cast<std::size_t>(N);
    if (P_alpha.shape()[0] != Ns || P_alpha.shape()[1] != Ns
        || P_beta.shape()[0] != Ns || P_beta.shape()[1] != Ns)
        throw std::invalid_argument(
            "Molecular XC: density / AO basis dimension mismatch");
    const bool restricted = spin == ATV::electronic::XcSpin::Restricted;
    const int nspin = restricted ? 1 : 2;
    const xt::xtensor<double, 2>* P[2] = {&P_alpha, &P_beta};

    const int nt = thread_count();
    std::vector<std::vector<double>> Vt[2];
    for (int s = 0; s < nspin; ++s)
        Vt[s].assign(static_cast<std::size_t>(nt), {});
    std::vector<double> e_t(nt, 0.0), na_t(nt, 0.0), nb_t(nt, 0.0);
    const int nblk = static_cast<int>(aog.blocks.size());

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        const int tid = thread_index();
        for (int s = 0; s < nspin; ++s)
            Vt[s][tid].assign(Ns * Ns, 0.0);
        Scratch w;
#ifdef _OPENMP
        // Static block ownership + fixed-order thread reduction: deterministic
        // for a given thread count.
#pragma omp for schedule(static)
#endif
        for (int b = 0; b < nblk; ++b) {
            const auto& blk = aog.blocks[static_cast<std::size_t>(b)];
            const int m = static_cast<int>(blk.ao.size());
            const int ng = blk.ng;
            if (m == 0 || ng == 0)
                continue;
            const std::size_t gm = static_cast<std::size_t>(ng) * m;
            const double* phi = blk.phi.data();
            const double* dphi[3] = {blk.dx.data(), blk.dy.data(),
                                     blk.dz.data()};
            grow(w.wt, ng);
            for (int g = 0; g < ng; ++g)
                w.wt[g] = grid.points[blk.g0 + g].weight;
            const double* wt = w.wt.data();

            // 1. Densities (and gradients) of each spin channel.
            for (int s = 0; s < nspin; ++s) {
                grow(w.P[s], static_cast<std::size_t>(m) * m);
                grow(w.T[s], gm);
                grow(w.rho[s], ng);
                for (int j = 0; j < m; ++j)
                    for (int i = 0; i < m; ++i)
                        w.P[s][i + static_cast<std::size_t>(j) * m] =
                            (*P[s])(blk.ao[i], blk.ao[j]);
                gemm('N', 'N', ng, m, m, 1.0, phi, ng, w.P[s].data(), m, 0.0,
                     w.T[s].data(), ng);
                const double* T = w.T[s].data();
                double* rho = w.rho[s].data();
                std::fill_n(rho, ng, 0.0);
                for (int k = 0; k < m; ++k)
                    for (int g = 0; g < ng; ++g)
                        rho[g] += T[k * ng + g] * phi[k * ng + g];
                for (int g = 0; g < ng; ++g)
                    rho[g] = std::max(rho[g], 0.0);
                if (is_gga) {
                    grow(w.grad[s], 3 * static_cast<std::size_t>(ng));
                    double* gr = w.grad[s].data();
                    std::fill_n(gr, 3 * ng, 0.0);
                    for (int d = 0; d < 3; ++d)
                        for (int k = 0; k < m; ++k)
                            for (int g = 0; g < ng; ++g)
                                gr[d * ng + g] +=
                                    2.0 * T[k * ng + g] * dphi[d][k * ng + g];
                }
            }
            const double* ra = w.rho[0].data();
            const double* rb = w.rho[restricted ? 0 : 1].data();
            const double* ga = is_gga ? w.grad[0].data() : nullptr;
            const double* gb =
                is_gga ? w.grad[restricted ? 0 : 1].data() : nullptr;

            // 2. One batched functional evaluation.
            for (auto* v : {&w.f, &w.vrho[0], &w.vrho[1]})
                grow(*v, ng);
            ATV::electronic::XcBatch batch;
            batch.spin = spin;
            batch.rho_alpha = {ra, static_cast<std::size_t>(ng)};
            batch.rho_beta = {rb, static_cast<std::size_t>(ng)};
            ATV::electronic::XcOutput out;
            out.energy_density = {w.f.data(), static_cast<std::size_t>(ng)};
            out.vrho_alpha = {w.vrho[0].data(), static_cast<std::size_t>(ng)};
            out.vrho_beta = {w.vrho[1].data(), static_cast<std::size_t>(ng)};
            if (is_gga) {
                for (int c = 0; c < 3; ++c) {
                    grow(w.sigma[c], ng);
                    grow(w.vsigma[c], ng);
                }
                for (int g = 0; g < ng; ++g) {
                    double aa = 0, ab = 0, bb = 0;
                    for (int d = 0; d < 3; ++d) {
                        const double x = ga[d * ng + g], y = gb[d * ng + g];
                        aa += x * x;
                        ab += x * y;
                        bb += y * y;
                    }
                    w.sigma[0][g] = aa;
                    w.sigma[1][g] = ab;
                    w.sigma[2][g] = bb;
                }
                const std::size_t n = static_cast<std::size_t>(ng);
                batch.sigma_aa = {w.sigma[0].data(), n};
                batch.sigma_ab = {w.sigma[1].data(), n};
                batch.sigma_bb = {w.sigma[2].data(), n};
                out.vsigma_aa = {w.vsigma[0].data(), n};
                out.vsigma_ab = {w.vsigma[1].data(), n};
                out.vsigma_bb = {w.vsigma[2].data(), n};
            }
            functional.evaluate(batch, ATV::electronic::XcOrder::First, out);

            for (int g = 0; g < ng; ++g) {
                e_t[tid] += wt[g] * w.f[g];
                na_t[tid] += wt[g] * ra[g];
                nb_t[tid] += wt[g] * rb[g];
            }

            // 3. V_sub = φᵀ X + Xᵀ φ, X = ½ w v_ρ φ + Σ_d w g_d ∂_d φ with
            //    g_α = 2 v_σαα ∇ρ_α + v_σαβ ∇ρ_β (and α ↔ β).
            grow(w.X, gm);
            grow(w.V, static_cast<std::size_t>(m) * m);
            for (int s = 0; s < nspin; ++s) {
                double* X = w.X.data();
                const double* vr = w.vrho[s].data();
                for (int k = 0; k < m; ++k)
                    for (int g = 0; g < ng; ++g)
                        X[k * ng + g] = 0.5 * wt[g] * vr[g] * phi[k * ng + g];
                if (is_gga) {
                    const double* own = s == 0 ? ga : gb;
                    const double* other = s == 0 ? gb : ga;
                    const double* vss = w.vsigma[s == 0 ? 0 : 2].data();
                    const double* vab = w.vsigma[1].data();
                    grow(w.grad_coef, 3 * static_cast<std::size_t>(ng));
                    double* coef = w.grad_coef.data();
                    for (int d = 0; d < 3; ++d)
                        for (int g = 0; g < ng; ++g)
                            coef[d * ng + g] = wt[g]
                                               * (2.0 * vss[g] * own[d * ng + g]
                                                  + vab[g] * other[d * ng + g]);
                    for (int d = 0; d < 3; ++d)
                        for (int k = 0; k < m; ++k)
                            for (int g = 0; g < ng; ++g)
                                X[k * ng + g] +=
                                    coef[d * ng + g] * dphi[d][k * ng + g];
                }
                gemm('T', 'N', m, m, ng, 1.0, phi, ng, X, ng, 0.0, w.V.data(),
                     m);
                double* Vacc = Vt[s][tid].data();
                for (int j = 0; j < m; ++j)
                    for (int i = 0; i < m; ++i)
                        Vacc[static_cast<std::size_t>(blk.ao[i]) * Ns
                             + blk.ao[j]] +=
                            w.V[i + static_cast<std::size_t>(j) * m]
                            + w.V[j + static_cast<std::size_t>(i) * m];
            }
        }
    }

    XCGridResult result;
    result.E_xc = 0.0;
    result.n_alpha_integrated = 0.0;
    result.n_beta_integrated = 0.0;
    for (int t = 0; t < nt; ++t) {
        result.E_xc += e_t[t];
        result.n_alpha_integrated += na_t[t];
        result.n_beta_integrated += nb_t[t];
    }
    for (int s = 0; s < nspin; ++s) {
        xt::xtensor<double, 2> V = xt::zeros<double>({Ns, Ns});
        for (int t = 0; t < nt; ++t)
            for (std::size_t i = 0; i < Ns * Ns; ++i)
                V.data()[i] += Vt[s][t][i];
        (s == 0 ? result.Vxc_alpha : result.Vxc_beta) = std::move(V);
    }
    if (restricted)
        result.Vxc_beta = result.Vxc_alpha;
    return result;
}

XCGridResult
evaluate_xc_batched(const MolecularGrid& grid,
                    const std::vector<native::Shell>& shells, int n_basis,
                    const xt::xtensor<double, 2>& P_alpha,
                    const xt::xtensor<double, 2>& P_beta,
                    const ATV::electronic::XcFunctional& functional) {
    XCGridBasis aog;
    aog.build(grid, shells, n_basis,
              (static_cast<unsigned>(functional.ingredients()) & 2u) != 0);
    return evaluate_xc_batched(aog, grid, P_alpha, P_beta, functional);
}

} // namespace scf
} // namespace cpu
} // namespace ATV
