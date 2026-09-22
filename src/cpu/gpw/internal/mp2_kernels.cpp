/// @file src/cpu/gpw/internal/mp2_kernels.cpp
/// @brief GPW MP2 kernels — MO-on-grid, (ia|jb) via per-pair Poisson or
///        RI-GPW G-space Coulomb intermediates, and the closed-shell
///        doubles sum. See mp2_kernels.h.

#include "atomiverse/cpu/gpw/internal/mp2_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/gpw/internal/mt_poisson.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"

// solve_poisson_q is used by multi-k RPA finite-q pair densities.

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

std::vector<double> mo_on_grid(const CompactBasisGrid& cbg,
                               const linalg::RealMatrix& C, int p0, int np) {
    const int n_mo = static_cast<int>(C.shape(1));
    if (p0 < 0 || np < 0 || p0 + np > n_mo)
        throw std::invalid_argument(
            "gpw mo_on_grid: MO range [" + std::to_string(p0) + ", "
            + std::to_string(p0 + np) + ") outside 0.." + std::to_string(n_mo));
    const long npts = cbg.npts;
    std::vector<double> psi(
        static_cast<std::size_t>(np) * static_cast<std::size_t>(npts), 0.0);
    const std::size_t na = cbg.gidx.size();
    for (std::size_t k = 0; k < na; ++k) {
        const long g = cbg.gidx[k];
        for (std::size_t nz = cbg.off[k]; nz < cbg.off[k + 1]; ++nz) {
            const int u = cbg.fu[nz];
            const double val = cbg.fv[nz];
            for (int p = 0; p < np; ++p)
                psi[static_cast<std::size_t>(p) * static_cast<std::size_t>(npts)
                    + static_cast<std::size_t>(g)] += C(u, p0 + p) * val;
        }
    }
    return psi;
}

std::vector<double> mo_eri_ovov(const UniformGrid& grid,
                                const std::vector<double>& psi_occ, int n_occ,
                                const std::vector<double>& psi_virt, int n_virt,
                                bool nonperiodic) {
    const long npts = grid.npts();
    const double w = grid.weight();
    if (psi_occ.size()
            != static_cast<std::size_t>(n_occ) * static_cast<std::size_t>(npts)
        || psi_virt.size()
               != static_cast<std::size_t>(n_virt)
                      * static_cast<std::size_t>(npts))
        throw std::invalid_argument(
            "gpw mo_eri_ovov: MO grid arrays do not match n_occ/n_virt/npts");
    const int nov = n_occ * n_virt;
    std::vector<double> v(static_cast<std::size_t>(nov)
                          * static_cast<std::size_t>(nov));
    std::vector<double> rho(static_cast<std::size_t>(npts));
    for (int ia = 0; ia < nov; ++ia) {
        const double* pi = psi_occ.data()
                           + static_cast<std::size_t>(ia / n_virt)
                                 * static_cast<std::size_t>(npts);
        const double* pa = psi_virt.data()
                           + static_cast<std::size_t>(ia % n_virt)
                                 * static_cast<std::size_t>(npts);
        for (long g = 0; g < npts; ++g)
            rho[static_cast<std::size_t>(g)] = pi[g] * pa[g];
        const PoissonResult pr = nonperiodic ? solve_poisson_mt(grid, rho)
                                             : solve_poisson(grid, rho);
        // (ia|jb) = (jb|ia): fill the upper triangle jb >= ia and mirror.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int jb = ia; jb < nov; ++jb) {
            const double* pj = psi_occ.data()
                               + static_cast<std::size_t>(jb / n_virt)
                                     * static_cast<std::size_t>(npts);
            const double* pb = psi_virt.data()
                               + static_cast<std::size_t>(jb % n_virt)
                                     * static_cast<std::size_t>(npts);
            double s = 0.0;
            for (long g = 0; g < npts; ++g)
                s += pr.v_r[static_cast<std::size_t>(g)] * pj[g] * pb[g];
            const double val = w * s;
            v[static_cast<std::size_t>(ia) * static_cast<std::size_t>(nov)
              + static_cast<std::size_t>(jb)] = val;
            v[static_cast<std::size_t>(jb) * static_cast<std::size_t>(nov)
              + static_cast<std::size_t>(ia)] = val;
        }
    }
    return v;
}

std::vector<double>
mo_eri_ovov_ri(const UniformGrid& grid, const std::vector<double>& psi_occ,
               int n_occ, const std::vector<double>& psi_virt, int n_virt) {
    // Product RI-GPW path (Del Ben 2012 / CP2K MP2_GPW): form the Coulomb
    // potential of every pair density once (G-space 4π/G² intermediate via
    // the shared FFT Poisson), store the real-space V_ia intermediates
    // (three-index product structure, memory O(n_ov · N_g)), then contract
    // (ia|jb) = Σ_g w V_ia(g) ρ_jb(g). Numerically identical to
    // mo_eri_ovov(..., nonperiodic=false) — the Poisson kernel is the PW
    // resolution of the identity for the pair density on this grid.
    const long npts = grid.npts();
    const double w = grid.weight();
    if (psi_occ.size()
            != static_cast<std::size_t>(n_occ) * static_cast<std::size_t>(npts)
        || psi_virt.size()
               != static_cast<std::size_t>(n_virt)
                      * static_cast<std::size_t>(npts))
        throw std::invalid_argument("gpw mo_eri_ovov_ri: MO grid arrays do not "
                                    "match n_occ/n_virt/npts");
    if (n_occ < 0 || n_virt < 0)
        throw std::invalid_argument(
            "gpw mo_eri_ovov_ri: negative orbital count");

    const int nov = n_occ * n_virt;
    std::vector<double> V(static_cast<std::size_t>(nov)
                          * static_cast<std::size_t>(npts));
    std::vector<double> rho(static_cast<std::size_t>(npts));
    for (int ia = 0; ia < nov; ++ia) {
        const double* pi = psi_occ.data()
                           + static_cast<std::size_t>(ia / n_virt)
                                 * static_cast<std::size_t>(npts);
        const double* pa = psi_virt.data()
                           + static_cast<std::size_t>(ia % n_virt)
                                 * static_cast<std::size_t>(npts);
        for (long g = 0; g < npts; ++g)
            rho[static_cast<std::size_t>(g)] = pi[g] * pa[g];
        // FFT → 4π/G² → IFFT (G=0 dropped); owns the RI-GPW intermediate.
        const PoissonResult pr = solve_poisson(grid, rho);
        double* Via =
            V.data()
            + static_cast<std::size_t>(ia) * static_cast<std::size_t>(npts);
        for (long g = 0; g < npts; ++g)
            Via[g] = pr.v_r[static_cast<std::size_t>(g)];
    }

    std::vector<double> v(static_cast<std::size_t>(nov)
                          * static_cast<std::size_t>(nov));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int ia = 0; ia < nov; ++ia) {
        const double* Via =
            V.data()
            + static_cast<std::size_t>(ia) * static_cast<std::size_t>(npts);
        for (int jb = ia; jb < nov; ++jb) {
            const double* pj = psi_occ.data()
                               + static_cast<std::size_t>(jb / n_virt)
                                     * static_cast<std::size_t>(npts);
            const double* pb = psi_virt.data()
                               + static_cast<std::size_t>(jb % n_virt)
                                     * static_cast<std::size_t>(npts);
            double s = 0.0;
            for (long g = 0; g < npts; ++g)
                s += Via[g] * pj[g] * pb[g];
            const double val = w * s;
            v[static_cast<std::size_t>(ia) * static_cast<std::size_t>(nov)
              + static_cast<std::size_t>(jb)] = val;
            v[static_cast<std::size_t>(jb) * static_cast<std::size_t>(nov)
              + static_cast<std::size_t>(ia)] = val;
        }
    }
    return v;
}

void mp2_rhf_energy(const std::vector<double>& v_ovov,
                    const std::vector<double>& eps_occ,
                    const std::vector<double>& eps_virt, double& e_os,
                    double& e_ss) {
    const int no = static_cast<int>(eps_occ.size());
    const int nv = static_cast<int>(eps_virt.size());
    const std::size_t nov =
        static_cast<std::size_t>(no) * static_cast<std::size_t>(nv);
    if (v_ovov.size() != nov * nov)
        throw std::invalid_argument(
            "gpw mp2_rhf_energy: tensor size does not match eps_occ/eps_virt");
    e_os = 0.0;
    e_ss = 0.0;
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b) {
                    const double v_iajb =
                        v_ovov[(static_cast<std::size_t>(i) * nv + a) * nov
                               + static_cast<std::size_t>(j) * nv + b];
                    const double v_ibja =
                        v_ovov[(static_cast<std::size_t>(i) * nv + b) * nov
                               + static_cast<std::size_t>(j) * nv + a];
                    const double denom =
                        eps_occ[static_cast<std::size_t>(i)]
                        + eps_occ[static_cast<std::size_t>(j)]
                        - eps_virt[static_cast<std::size_t>(a)]
                        - eps_virt[static_cast<std::size_t>(b)];
                    e_os += v_iajb * v_iajb / denom;
                    e_ss += v_iajb * (v_iajb - v_ibja) / denom;
                }
}

std::vector<double>
mo_eri_ovov_ab(const UniformGrid& grid, const std::vector<double>& psi_occ_a,
               int n_occ_a, const std::vector<double>& psi_virt_a, int n_virt_a,
               const std::vector<double>& psi_occ_b, int n_occ_b,
               const std::vector<double>& psi_virt_b, int n_virt_b,
               bool nonperiodic) {
    const long npts = grid.npts();
    const double w = grid.weight();
    if (n_occ_a < 0 || n_virt_a < 0 || n_occ_b < 0 || n_virt_b < 0)
        throw std::invalid_argument(
            "gpw mo_eri_ovov_ab: negative orbital count");
    if (psi_occ_a.size()
            != static_cast<std::size_t>(n_occ_a)
                   * static_cast<std::size_t>(npts)
        || psi_virt_a.size()
               != static_cast<std::size_t>(n_virt_a)
                      * static_cast<std::size_t>(npts)
        || psi_occ_b.size()
               != static_cast<std::size_t>(n_occ_b)
                      * static_cast<std::size_t>(npts)
        || psi_virt_b.size()
               != static_cast<std::size_t>(n_virt_b)
                      * static_cast<std::size_t>(npts))
        throw std::invalid_argument(
            "gpw mo_eri_ovov_ab: MO grid arrays do not match counts/npts");
    const int nov_a = n_occ_a * n_virt_a;
    const int nov_b = n_occ_b * n_virt_b;
    if (nov_a == 0 || nov_b == 0)
        return {};
    std::vector<double> v(static_cast<std::size_t>(nov_a)
                          * static_cast<std::size_t>(nov_b));
    std::vector<double> rho(static_cast<std::size_t>(npts));
    for (int ia = 0; ia < nov_a; ++ia) {
        const double* pi = psi_occ_a.data()
                           + static_cast<std::size_t>(ia / n_virt_a)
                                 * static_cast<std::size_t>(npts);
        const double* pa = psi_virt_a.data()
                           + static_cast<std::size_t>(ia % n_virt_a)
                                 * static_cast<std::size_t>(npts);
        for (long g = 0; g < npts; ++g)
            rho[static_cast<std::size_t>(g)] = pi[g] * pa[g];
        const PoissonResult pr = nonperiodic ? solve_poisson_mt(grid, rho)
                                             : solve_poisson(grid, rho);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int jb = 0; jb < nov_b; ++jb) {
            const double* pj = psi_occ_b.data()
                               + static_cast<std::size_t>(jb / n_virt_b)
                                     * static_cast<std::size_t>(npts);
            const double* pb = psi_virt_b.data()
                               + static_cast<std::size_t>(jb % n_virt_b)
                                     * static_cast<std::size_t>(npts);
            double s = 0.0;
            for (long g = 0; g < npts; ++g)
                s += pr.v_r[static_cast<std::size_t>(g)] * pj[g] * pb[g];
            v[static_cast<std::size_t>(ia) * static_cast<std::size_t>(nov_b)
              + static_cast<std::size_t>(jb)] = w * s;
        }
    }
    return v;
}

std::vector<double>
mo_eri_ovov_ab_ri(const UniformGrid& grid, const std::vector<double>& psi_occ_a,
                  int n_occ_a, const std::vector<double>& psi_virt_a,
                  int n_virt_a, const std::vector<double>& psi_occ_b,
                  int n_occ_b, const std::vector<double>& psi_virt_b,
                  int n_virt_b) {
    const long npts = grid.npts();
    const double w = grid.weight();
    if (n_occ_a < 0 || n_virt_a < 0 || n_occ_b < 0 || n_virt_b < 0)
        throw std::invalid_argument(
            "gpw mo_eri_ovov_ab_ri: negative orbital count");
    if (psi_occ_a.size()
            != static_cast<std::size_t>(n_occ_a)
                   * static_cast<std::size_t>(npts)
        || psi_virt_a.size()
               != static_cast<std::size_t>(n_virt_a)
                      * static_cast<std::size_t>(npts)
        || psi_occ_b.size()
               != static_cast<std::size_t>(n_occ_b)
                      * static_cast<std::size_t>(npts)
        || psi_virt_b.size()
               != static_cast<std::size_t>(n_virt_b)
                      * static_cast<std::size_t>(npts))
        throw std::invalid_argument(
            "gpw mo_eri_ovov_ab_ri: MO grid arrays do not match counts/npts");
    const int nov_a = n_occ_a * n_virt_a;
    const int nov_b = n_occ_b * n_virt_b;
    if (nov_a == 0 || nov_b == 0)
        return {};

    std::vector<double> V(static_cast<std::size_t>(nov_a)
                          * static_cast<std::size_t>(npts));
    std::vector<double> rho(static_cast<std::size_t>(npts));
    for (int ia = 0; ia < nov_a; ++ia) {
        const double* pi = psi_occ_a.data()
                           + static_cast<std::size_t>(ia / n_virt_a)
                                 * static_cast<std::size_t>(npts);
        const double* pa = psi_virt_a.data()
                           + static_cast<std::size_t>(ia % n_virt_a)
                                 * static_cast<std::size_t>(npts);
        for (long g = 0; g < npts; ++g)
            rho[static_cast<std::size_t>(g)] = pi[g] * pa[g];
        const PoissonResult pr = solve_poisson(grid, rho);
        double* Via =
            V.data()
            + static_cast<std::size_t>(ia) * static_cast<std::size_t>(npts);
        for (long g = 0; g < npts; ++g)
            Via[g] = pr.v_r[static_cast<std::size_t>(g)];
    }

    std::vector<double> v(static_cast<std::size_t>(nov_a)
                          * static_cast<std::size_t>(nov_b));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int ia = 0; ia < nov_a; ++ia) {
        const double* Via =
            V.data()
            + static_cast<std::size_t>(ia) * static_cast<std::size_t>(npts);
        for (int jb = 0; jb < nov_b; ++jb) {
            const double* pj = psi_occ_b.data()
                               + static_cast<std::size_t>(jb / n_virt_b)
                                     * static_cast<std::size_t>(npts);
            const double* pb = psi_virt_b.data()
                               + static_cast<std::size_t>(jb % n_virt_b)
                                     * static_cast<std::size_t>(npts);
            double s = 0.0;
            for (long g = 0; g < npts; ++g)
                s += Via[g] * pj[g] * pb[g];
            v[static_cast<std::size_t>(ia) * static_cast<std::size_t>(nov_b)
              + static_cast<std::size_t>(jb)] = w * s;
        }
    }
    return v;
}

namespace {

// Same-spin UMP2 channel: (1/4) Σ [(ia|jb)−(ib|ja)]² / Δ  (PySCF convention).
double same_spin_ump2_(const std::vector<double>& v_ovov,
                       const std::vector<double>& eps_occ,
                       const std::vector<double>& eps_virt) {
    const int no = static_cast<int>(eps_occ.size());
    const int nv = static_cast<int>(eps_virt.size());
    if (no < 2 || nv < 2)
        return 0.0;
    const std::size_t nov =
        static_cast<std::size_t>(no) * static_cast<std::size_t>(nv);
    if (v_ovov.size() != nov * nov)
        throw std::invalid_argument(
            "gpw mp2_uhf_energy: same-spin tensor size mismatch");
    double e = 0.0;
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b) {
                    const double v_iajb =
                        v_ovov[(static_cast<std::size_t>(i) * nv + a) * nov
                               + static_cast<std::size_t>(j) * nv + b];
                    const double v_ibja =
                        v_ovov[(static_cast<std::size_t>(i) * nv + b) * nov
                               + static_cast<std::size_t>(j) * nv + a];
                    const double t = v_iajb - v_ibja;
                    const double denom =
                        eps_occ[static_cast<std::size_t>(i)]
                        + eps_occ[static_cast<std::size_t>(j)]
                        - eps_virt[static_cast<std::size_t>(a)]
                        - eps_virt[static_cast<std::size_t>(b)];
                    e += 0.25 * t * t / denom;
                }
    return e;
}

} // namespace

void mp2_uhf_energy(
    const std::vector<double>& v_aa, const std::vector<double>& v_bb,
    const std::vector<double>& v_ab, const std::vector<double>& eps_occ_a,
    const std::vector<double>& eps_virt_a, const std::vector<double>& eps_occ_b,
    const std::vector<double>& eps_virt_b, double& e_os, double& e_ss) {
    const int no_a = static_cast<int>(eps_occ_a.size());
    const int nv_a = static_cast<int>(eps_virt_a.size());
    const int no_b = static_cast<int>(eps_occ_b.size());
    const int nv_b = static_cast<int>(eps_virt_b.size());
    const std::size_t nov_a =
        static_cast<std::size_t>(no_a) * static_cast<std::size_t>(nv_a);
    const std::size_t nov_b =
        static_cast<std::size_t>(no_b) * static_cast<std::size_t>(nv_b);

    e_ss = same_spin_ump2_(v_aa, eps_occ_a, eps_virt_a)
           + same_spin_ump2_(v_bb, eps_occ_b, eps_virt_b);

    e_os = 0.0;
    if (no_a > 0 && nv_a > 0 && no_b > 0 && nv_b > 0) {
        if (v_ab.size() != nov_a * nov_b)
            throw std::invalid_argument(
                "gpw mp2_uhf_energy: cross-spin tensor size mismatch");
        for (int i = 0; i < no_a; ++i)
            for (int j = 0; j < no_b; ++j)
                for (int a = 0; a < nv_a; ++a)
                    for (int b = 0; b < nv_b; ++b) {
                        const std::size_t ia =
                            static_cast<std::size_t>(i) * nv_a
                            + static_cast<std::size_t>(a);
                        const std::size_t jb =
                            static_cast<std::size_t>(j) * nv_b
                            + static_cast<std::size_t>(b);
                        const double v = v_ab[ia * nov_b + jb];
                        const double denom =
                            eps_occ_a[static_cast<std::size_t>(i)]
                            + eps_occ_b[static_cast<std::size_t>(j)]
                            - eps_virt_a[static_cast<std::size_t>(a)]
                            - eps_virt_b[static_cast<std::size_t>(b)];
                        e_os += v * v / denom;
                    }
    } else if (!v_ab.empty()) {
        throw std::invalid_argument(
            "gpw mp2_uhf_energy: cross-spin tensor present with empty channel");
    }
}

std::vector<Cplx> mo_on_grid_k(const KpointBasisGrid& kbg, int ik,
                               const linalg::CplxMatrix& C, int p0, int np) {
    const int n_mo = static_cast<int>(C.shape(1));
    const int nb = static_cast<int>(C.shape(0));
    if (p0 < 0 || np < 0 || p0 + np > n_mo)
        throw std::invalid_argument(
            "gpw mo_on_grid_k: MO range [" + std::to_string(p0) + ", "
            + std::to_string(p0 + np) + ") outside 0.." + std::to_string(n_mo));
    if (ik < 0 || ik >= static_cast<int>(kbg.chi.size()))
        throw std::invalid_argument("gpw mo_on_grid_k: k index out of range");
    if (nb != kbg.nb)
        throw std::invalid_argument(
            "gpw mo_on_grid_k: C.nb does not match KpointBasisGrid.nb");
    const long npts = kbg.npts;
    std::vector<Cplx> psi(static_cast<std::size_t>(np)
                              * static_cast<std::size_t>(npts),
                          Cplx{0.0, 0.0});
    // KpointBasisGrid::chi[k] is dense over all npts: chi[g * nb + u].
    const auto& chi_k = kbg.chi[static_cast<std::size_t>(ik)];
    if (chi_k.size()
        != static_cast<std::size_t>(npts) * static_cast<std::size_t>(nb))
        throw std::logic_error(
            "gpw mo_on_grid_k: chi_k size is not npts * nb (dense layout)");
    for (long g = 0; g < npts; ++g) {
        const Cplx* chi =
            chi_k.data()
            + static_cast<std::size_t>(g) * static_cast<std::size_t>(nb);
        for (int p = 0; p < np; ++p) {
            Cplx s{0.0, 0.0};
            for (int u = 0; u < nb; ++u)
                s += C(u, p0 + p) * chi[u];
            psi[static_cast<std::size_t>(p) * static_cast<std::size_t>(npts)
                + static_cast<std::size_t>(g)] = s;
        }
    }
    return psi;
}

namespace {

// Wrap fractional κ into (−1/2, 1/2].
double wrap_frac(double x) {
    x -= std::floor(x + 0.5);
    if (x <= -0.5)
        x += 1.0;
    if (x > 0.5)
        x -= 1.0;
    return x;
}

int find_k_index(const std::vector<std::array<double, 3>>& k_frac, double fx,
                 double fy, double fz, double tol = 1e-8) {
    const double tx = wrap_frac(fx), ty = wrap_frac(fy), tz = wrap_frac(fz);
    for (std::size_t i = 0; i < k_frac.size(); ++i) {
        const double dx = wrap_frac(k_frac[i][0] - tx);
        const double dy = wrap_frac(k_frac[i][1] - ty);
        const double dz = wrap_frac(k_frac[i][2] - tz);
        if (std::abs(dx) < tol && std::abs(dy) < tol && std::abs(dz) < tol)
            return static_cast<int>(i);
    }
    return -1;
}

// Coulomb potential of a complex pair density via linear real Poisson.
// Optional crystal-momentum transfer q_frac (nullptr ⇒ q = 0, G=0 dropped).
void poisson_complex(const UniformGrid& grid, const Cplx* rho,
                     std::vector<Cplx>& V, const double* q_frac = nullptr) {
    const long npts = grid.npts();
    std::vector<double> re(static_cast<std::size_t>(npts));
    std::vector<double> im(static_cast<std::size_t>(npts));
    for (long g = 0; g < npts; ++g) {
        re[static_cast<std::size_t>(g)] = rho[g].real();
        im[static_cast<std::size_t>(g)] = rho[g].imag();
    }
    const double q0[3] = {0.0, 0.0, 0.0};
    const double* q = q_frac ? q_frac : q0;
    const PoissonResult vr = solve_poisson_q(grid, re, q);
    const PoissonResult vi = solve_poisson_q(grid, im, q);
    V.resize(static_cast<std::size_t>(npts));
    for (long g = 0; g < npts; ++g)
        V[static_cast<std::size_t>(g)] =
            Cplx{vr.v_r[static_cast<std::size_t>(g)],
                 vi.v_r[static_cast<std::size_t>(g)]};
}

} // namespace

void mp2_rhf_energy_kpoints(const UniformGrid& grid,
                            const std::vector<std::vector<Cplx>>& psi_occ_k,
                            const std::vector<std::vector<Cplx>>& psi_virt_k,
                            const std::vector<std::vector<double>>& eps_occ_k,
                            const std::vector<std::vector<double>>& eps_virt_k,
                            const std::vector<std::array<double, 3>>& k_frac,
                            const std::vector<double>& k_weights, double& e_os,
                            double& e_ss) {
    const int nk = static_cast<int>(k_frac.size());
    if (nk < 1)
        throw std::invalid_argument("gpw mp2_rhf_energy_kpoints: empty k mesh");
    if (static_cast<int>(psi_occ_k.size()) != nk
        || static_cast<int>(psi_virt_k.size()) != nk
        || static_cast<int>(eps_occ_k.size()) != nk
        || static_cast<int>(eps_virt_k.size()) != nk
        || static_cast<int>(k_weights.size()) != nk)
        throw std::invalid_argument(
            "gpw mp2_rhf_energy_kpoints: per-k array length mismatch");

    const int n_occ = static_cast<int>(eps_occ_k[0].size());
    const int n_virt = static_cast<int>(eps_virt_k[0].size());
    if (n_occ < 1 || n_virt < 1)
        throw std::invalid_argument(
            "gpw mp2_rhf_energy_kpoints: need n_occ>=1 and n_virt>=1");
    const long npts = grid.npts();
    const double w = grid.weight();
    const int nov = n_occ * n_virt;

    for (int ik = 0; ik < nk; ++ik) {
        if (static_cast<int>(eps_occ_k[static_cast<std::size_t>(ik)].size())
                != n_occ
            || static_cast<int>(eps_virt_k[static_cast<std::size_t>(ik)].size())
                   != n_virt)
            throw std::invalid_argument("gpw mp2_rhf_energy_kpoints: "
                                        "inconsistent n_occ/n_virt across k");
        const std::size_t need_o =
            static_cast<std::size_t>(n_occ) * static_cast<std::size_t>(npts);
        const std::size_t need_v =
            static_cast<std::size_t>(n_virt) * static_cast<std::size_t>(npts);
        if (psi_occ_k[static_cast<std::size_t>(ik)].size() != need_o
            || psi_virt_k[static_cast<std::size_t>(ik)].size() != need_v)
            throw std::invalid_argument(
                "gpw mp2_rhf_energy_kpoints: MO grid size mismatch");
    }

    // Equal-weight factor: E = (1/Nk) Σ_{ki,kj,ka} ... when w_k = 1/Nk.
    // General equal-ish: use Nk * w_ki * w_kj * Nk * w_ka… → for equal w:
    // weight_triple = Nk * w_i * w_j  (kb fixed; one free sum over ka with
    // w absorbed as 1 when all w=1/Nk ⇒ weight = 1/Nk).
    // Use: weight = nk * w[ki] * w[kj] * nk * w[ka] / (nk * w[kb]) simplified
    // for equal w to 1/nk. For robustness with equal weights only:
    double wsum = 0.0;
    for (double wk : k_weights)
        wsum += wk;
    if (std::abs(wsum - 1.0) > 1e-8)
        throw std::invalid_argument(
            "gpw mp2_rhf_energy_kpoints: k_weights must sum to 1");

    // Precompute V_ia for every (ki, ka, i, a): complex Coulomb potential of
    // ρ_ia = conj(ψ_i^{ki}) ψ_a^{ka}. Memory O(nk² · nov · npts).
    // Layout: Via_cache[ki * nk + ka][ia * npts + g]
    std::vector<std::vector<Cplx>> Via_cache(static_cast<std::size_t>(nk)
                                             * static_cast<std::size_t>(nk));
    std::vector<Cplx> rho(static_cast<std::size_t>(npts));
    for (int ki = 0; ki < nk; ++ki) {
        for (int ka = 0; ka < nk; ++ka) {
            auto& block = Via_cache[static_cast<std::size_t>(ki) * nk
                                    + static_cast<std::size_t>(ka)];
            block.resize(static_cast<std::size_t>(nov)
                         * static_cast<std::size_t>(npts));
            const auto& po = psi_occ_k[static_cast<std::size_t>(ki)];
            const auto& pv = psi_virt_k[static_cast<std::size_t>(ka)];
            for (int i = 0; i < n_occ; ++i) {
                for (int a = 0; a < n_virt; ++a) {
                    const int ia = i * n_virt + a;
                    const Cplx* pi = po.data()
                                     + static_cast<std::size_t>(i)
                                           * static_cast<std::size_t>(npts);
                    const Cplx* pa = pv.data()
                                     + static_cast<std::size_t>(a)
                                           * static_cast<std::size_t>(npts);
                    for (long g = 0; g < npts; ++g)
                        rho[static_cast<std::size_t>(g)] =
                            std::conj(pi[g]) * pa[g];
                    std::vector<Cplx> V;
                    poisson_complex(grid, rho.data(), V);
                    Cplx* dest = block.data()
                                 + static_cast<std::size_t>(ia)
                                       * static_cast<std::size_t>(npts);
                    for (long g = 0; g < npts; ++g)
                        dest[g] = V[static_cast<std::size_t>(g)];
                }
            }
        }
    }

    auto contract = [&](int ki, int ka, int i, int a, int kj, int kb, int j,
                        int b) -> Cplx {
        // (ia|jb) = Σ_g w V_ia(g) ρ_jb(g), ρ_jb = conj(ψ_j) ψ_b
        const auto& Vblock = Via_cache[static_cast<std::size_t>(ki) * nk
                                       + static_cast<std::size_t>(ka)];
        const int ia = i * n_virt + a;
        const Cplx* Via =
            Vblock.data()
            + static_cast<std::size_t>(ia) * static_cast<std::size_t>(npts);
        const Cplx* pj =
            psi_occ_k[static_cast<std::size_t>(kj)].data()
            + static_cast<std::size_t>(j) * static_cast<std::size_t>(npts);
        const Cplx* pb =
            psi_virt_k[static_cast<std::size_t>(kb)].data()
            + static_cast<std::size_t>(b) * static_cast<std::size_t>(npts);
        Cplx s{0.0, 0.0};
        for (long g = 0; g < npts; ++g)
            s += Via[g] * (std::conj(pj[g]) * pb[g]);
        return w * s;
    };

    e_os = 0.0;
    e_ss = 0.0;
    // weight_triple = 1/Nk for equal weights (w_k = 1/Nk).
    // Generalization used: Nk * w[ki] * w[kj]  (matches 1/Nk when equal;
    // free sum over ka with kb determined — each (ki,kj,ka) counted once).
    for (int ki = 0; ki < nk; ++ki) {
        for (int kj = 0; kj < nk; ++kj) {
            const double w_ij = static_cast<double>(nk)
                                * k_weights[static_cast<std::size_t>(ki)]
                                * k_weights[static_cast<std::size_t>(kj)];
            for (int ka = 0; ka < nk; ++ka) {
                const double fbx = k_frac[static_cast<std::size_t>(ki)][0]
                                   + k_frac[static_cast<std::size_t>(kj)][0]
                                   - k_frac[static_cast<std::size_t>(ka)][0];
                const double fby = k_frac[static_cast<std::size_t>(ki)][1]
                                   + k_frac[static_cast<std::size_t>(kj)][1]
                                   - k_frac[static_cast<std::size_t>(ka)][1];
                const double fbz = k_frac[static_cast<std::size_t>(ki)][2]
                                   + k_frac[static_cast<std::size_t>(kj)][2]
                                   - k_frac[static_cast<std::size_t>(ka)][2];
                const int kb = find_k_index(k_frac, fbx, fby, fbz);
                if (kb < 0)
                    throw std::runtime_error(
                        "gpw mp2_rhf_energy_kpoints: momentum-conserving kb "
                        "not in the sampled mesh — use a closed (uniform) "
                        "k-mesh");
                for (int i = 0; i < n_occ; ++i)
                    for (int j = 0; j < n_occ; ++j)
                        for (int a = 0; a < n_virt; ++a)
                            for (int b = 0; b < n_virt; ++b) {
                                const Cplx v_iajb =
                                    contract(ki, ka, i, a, kj, kb, j, b);
                                const Cplx v_ibja =
                                    contract(ki, kb, i, b, kj, ka, j, a);
                                const double denom =
                                    eps_occ_k[static_cast<std::size_t>(ki)]
                                             [static_cast<std::size_t>(i)]
                                    + eps_occ_k[static_cast<std::size_t>(kj)]
                                               [static_cast<std::size_t>(j)]
                                    - eps_virt_k[static_cast<std::size_t>(ka)]
                                                [static_cast<std::size_t>(a)]
                                    - eps_virt_k[static_cast<std::size_t>(kb)]
                                                [static_cast<std::size_t>(b)];
                                if (std::abs(denom) < 1e-12)
                                    continue;
                                const double inv = w_ij / denom;
                                e_os += std::norm(v_iajb) * inv;
                                e_ss +=
                                    (v_iajb * std::conj(v_iajb - v_ibja)).real()
                                    * inv;
                            }
            }
        }
    }
}

namespace {

// Build Via_cache[ki*nk+ka][ia*npts+g] for one spin channel.
std::vector<std::vector<Cplx>> build_via_cache_k_(
    const UniformGrid& grid, const std::vector<std::vector<Cplx>>& psi_occ_k,
    const std::vector<std::vector<Cplx>>& psi_virt_k, int n_occ, int n_virt) {
    const int nk = static_cast<int>(psi_occ_k.size());
    const long npts = grid.npts();
    const int nov = n_occ * n_virt;
    std::vector<std::vector<Cplx>> Via_cache(static_cast<std::size_t>(nk)
                                             * static_cast<std::size_t>(nk));
    std::vector<Cplx> rho(static_cast<std::size_t>(npts));
    for (int ki = 0; ki < nk; ++ki) {
        for (int ka = 0; ka < nk; ++ka) {
            auto& block = Via_cache[static_cast<std::size_t>(ki) * nk
                                    + static_cast<std::size_t>(ka)];
            block.resize(static_cast<std::size_t>(nov)
                         * static_cast<std::size_t>(npts));
            const auto& po = psi_occ_k[static_cast<std::size_t>(ki)];
            const auto& pv = psi_virt_k[static_cast<std::size_t>(ka)];
            for (int i = 0; i < n_occ; ++i) {
                for (int a = 0; a < n_virt; ++a) {
                    const int ia = i * n_virt + a;
                    const Cplx* pi = po.data()
                                     + static_cast<std::size_t>(i)
                                           * static_cast<std::size_t>(npts);
                    const Cplx* pa = pv.data()
                                     + static_cast<std::size_t>(a)
                                           * static_cast<std::size_t>(npts);
                    for (long g = 0; g < npts; ++g)
                        rho[static_cast<std::size_t>(g)] =
                            std::conj(pi[g]) * pa[g];
                    std::vector<Cplx> V;
                    poisson_complex(grid, rho.data(), V);
                    Cplx* dest = block.data()
                                 + static_cast<std::size_t>(ia)
                                       * static_cast<std::size_t>(npts);
                    for (long g = 0; g < npts; ++g)
                        dest[g] = V[static_cast<std::size_t>(g)];
                }
            }
        }
    }
    return Via_cache;
}

Cplx contract_ovov_k_(const std::vector<std::vector<Cplx>>& Via_cache,
                      const std::vector<std::vector<Cplx>>& psi_occ_k,
                      const std::vector<std::vector<Cplx>>& psi_virt_k, int nk,
                      int n_virt, long npts, double w_grid, int ki, int ka,
                      int i, int a, int kj, int kb, int j, int b) {
    const auto& Vblock = Via_cache[static_cast<std::size_t>(ki) * nk
                                   + static_cast<std::size_t>(ka)];
    const int ia = i * n_virt + a;
    const Cplx* Via =
        Vblock.data()
        + static_cast<std::size_t>(ia) * static_cast<std::size_t>(npts);
    const Cplx* pj =
        psi_occ_k[static_cast<std::size_t>(kj)].data()
        + static_cast<std::size_t>(j) * static_cast<std::size_t>(npts);
    const Cplx* pb =
        psi_virt_k[static_cast<std::size_t>(kb)].data()
        + static_cast<std::size_t>(b) * static_cast<std::size_t>(npts);
    Cplx s{0.0, 0.0};
    for (long g = 0; g < npts; ++g)
        s += Via[g] * (std::conj(pj[g]) * pb[g]);
    return w_grid * s;
}

// UMP2 same-spin multi-k: (1/4) Σ |v_iajb − v_ibja|² / Δ
double same_spin_ump2_k_(const UniformGrid& grid,
                         const std::vector<std::vector<Cplx>>& psi_occ_k,
                         const std::vector<std::vector<Cplx>>& psi_virt_k,
                         const std::vector<std::vector<double>>& eps_occ_k,
                         const std::vector<std::vector<double>>& eps_virt_k,
                         const std::vector<std::array<double, 3>>& k_frac,
                         const std::vector<double>& k_weights) {
    const int nk = static_cast<int>(k_frac.size());
    const int n_occ = static_cast<int>(eps_occ_k[0].size());
    const int n_virt = static_cast<int>(eps_virt_k[0].size());
    if (n_occ < 2 || n_virt < 2)
        return 0.0;
    const long npts = grid.npts();
    const double w = grid.weight();
    const auto Via =
        build_via_cache_k_(grid, psi_occ_k, psi_virt_k, n_occ, n_virt);
    double e = 0.0;
    for (int ki = 0; ki < nk; ++ki) {
        for (int kj = 0; kj < nk; ++kj) {
            const double w_ij = static_cast<double>(nk)
                                * k_weights[static_cast<std::size_t>(ki)]
                                * k_weights[static_cast<std::size_t>(kj)];
            for (int ka = 0; ka < nk; ++ka) {
                const double fbx = k_frac[static_cast<std::size_t>(ki)][0]
                                   + k_frac[static_cast<std::size_t>(kj)][0]
                                   - k_frac[static_cast<std::size_t>(ka)][0];
                const double fby = k_frac[static_cast<std::size_t>(ki)][1]
                                   + k_frac[static_cast<std::size_t>(kj)][1]
                                   - k_frac[static_cast<std::size_t>(ka)][1];
                const double fbz = k_frac[static_cast<std::size_t>(ki)][2]
                                   + k_frac[static_cast<std::size_t>(kj)][2]
                                   - k_frac[static_cast<std::size_t>(ka)][2];
                const int kb = find_k_index(k_frac, fbx, fby, fbz);
                if (kb < 0)
                    throw std::runtime_error(
                        "gpw mp2_uhf_energy_kpoints: momentum-conserving kb "
                        "not in the sampled mesh — use a closed (uniform) "
                        "k-mesh");
                for (int i = 0; i < n_occ; ++i)
                    for (int j = 0; j < n_occ; ++j)
                        for (int a = 0; a < n_virt; ++a)
                            for (int b = 0; b < n_virt; ++b) {
                                const Cplx v_iajb = contract_ovov_k_(
                                    Via, psi_occ_k, psi_virt_k, nk, n_virt,
                                    npts, w, ki, ka, i, a, kj, kb, j, b);
                                const Cplx v_ibja = contract_ovov_k_(
                                    Via, psi_occ_k, psi_virt_k, nk, n_virt,
                                    npts, w, ki, kb, i, b, kj, ka, j, a);
                                const double denom =
                                    eps_occ_k[static_cast<std::size_t>(ki)]
                                             [static_cast<std::size_t>(i)]
                                    + eps_occ_k[static_cast<std::size_t>(kj)]
                                               [static_cast<std::size_t>(j)]
                                    - eps_virt_k[static_cast<std::size_t>(ka)]
                                                [static_cast<std::size_t>(a)]
                                    - eps_virt_k[static_cast<std::size_t>(kb)]
                                                [static_cast<std::size_t>(b)];
                                if (std::abs(denom) < 1e-12)
                                    continue;
                                e += 0.25 * std::norm(v_iajb - v_ibja) * w_ij
                                     / denom;
                            }
            }
        }
    }
    return e;
}

} // namespace

void mp2_uhf_energy_kpoints(
    const UniformGrid& grid, const std::vector<std::vector<Cplx>>& psi_occ_a_k,
    const std::vector<std::vector<Cplx>>& psi_virt_a_k,
    const std::vector<std::vector<Cplx>>& psi_occ_b_k,
    const std::vector<std::vector<Cplx>>& psi_virt_b_k,
    const std::vector<std::vector<double>>& eps_occ_a_k,
    const std::vector<std::vector<double>>& eps_virt_a_k,
    const std::vector<std::vector<double>>& eps_occ_b_k,
    const std::vector<std::vector<double>>& eps_virt_b_k,
    const std::vector<std::array<double, 3>>& k_frac,
    const std::vector<double>& k_weights, double& e_os, double& e_ss) {
    const int nk = static_cast<int>(k_frac.size());
    if (nk < 1)
        throw std::invalid_argument("gpw mp2_uhf_energy_kpoints: empty k mesh");
    if (static_cast<int>(psi_occ_a_k.size()) != nk
        || static_cast<int>(psi_virt_a_k.size()) != nk
        || static_cast<int>(psi_occ_b_k.size()) != nk
        || static_cast<int>(psi_virt_b_k.size()) != nk
        || static_cast<int>(eps_occ_a_k.size()) != nk
        || static_cast<int>(eps_virt_a_k.size()) != nk
        || static_cast<int>(eps_occ_b_k.size()) != nk
        || static_cast<int>(eps_virt_b_k.size()) != nk
        || static_cast<int>(k_weights.size()) != nk)
        throw std::invalid_argument(
            "gpw mp2_uhf_energy_kpoints: per-k array length mismatch");

    const int n_oa = static_cast<int>(eps_occ_a_k[0].size());
    const int n_va = static_cast<int>(eps_virt_a_k[0].size());
    const int n_ob = static_cast<int>(eps_occ_b_k[0].size());
    const int n_vb = static_cast<int>(eps_virt_b_k[0].size());
    const long npts = grid.npts();
    const double w = grid.weight();

    double wsum = 0.0;
    for (double wk : k_weights)
        wsum += wk;
    if (std::abs(wsum - 1.0) > 1e-8)
        throw std::invalid_argument(
            "gpw mp2_uhf_energy_kpoints: k_weights must sum to 1");

    for (int ik = 0; ik < nk; ++ik) {
        if (static_cast<int>(eps_occ_a_k[static_cast<std::size_t>(ik)].size())
                != n_oa
            || static_cast<int>(
                   eps_virt_a_k[static_cast<std::size_t>(ik)].size())
                   != n_va
            || static_cast<int>(
                   eps_occ_b_k[static_cast<std::size_t>(ik)].size())
                   != n_ob
            || static_cast<int>(
                   eps_virt_b_k[static_cast<std::size_t>(ik)].size())
                   != n_vb)
            throw std::invalid_argument("gpw mp2_uhf_energy_kpoints: "
                                        "inconsistent n_occ/n_virt across k");
        const std::size_t need_oa =
            static_cast<std::size_t>(n_oa) * static_cast<std::size_t>(npts);
        const std::size_t need_va =
            static_cast<std::size_t>(n_va) * static_cast<std::size_t>(npts);
        const std::size_t need_ob =
            static_cast<std::size_t>(n_ob) * static_cast<std::size_t>(npts);
        const std::size_t need_vb =
            static_cast<std::size_t>(n_vb) * static_cast<std::size_t>(npts);
        if (psi_occ_a_k[static_cast<std::size_t>(ik)].size() != need_oa
            || psi_virt_a_k[static_cast<std::size_t>(ik)].size() != need_va
            || psi_occ_b_k[static_cast<std::size_t>(ik)].size() != need_ob
            || psi_virt_b_k[static_cast<std::size_t>(ik)].size() != need_vb)
            throw std::invalid_argument(
                "gpw mp2_uhf_energy_kpoints: MO grid size mismatch");
    }

    e_ss = same_spin_ump2_k_(grid, psi_occ_a_k, psi_virt_a_k, eps_occ_a_k,
                             eps_virt_a_k, k_frac, k_weights)
           + same_spin_ump2_k_(grid, psi_occ_b_k, psi_virt_b_k, eps_occ_b_k,
                               eps_virt_b_k, k_frac, k_weights);

    e_os = 0.0;
    if (n_oa < 1 || n_va < 1 || n_ob < 1 || n_vb < 1)
        return;

    // Cross-spin: V_ia from α, contract with ρ_jb from β.
    const auto Via_a =
        build_via_cache_k_(grid, psi_occ_a_k, psi_virt_a_k, n_oa, n_va);
    for (int ki = 0; ki < nk; ++ki) {
        for (int kj = 0; kj < nk; ++kj) {
            const double w_ij = static_cast<double>(nk)
                                * k_weights[static_cast<std::size_t>(ki)]
                                * k_weights[static_cast<std::size_t>(kj)];
            for (int ka = 0; ka < nk; ++ka) {
                const double fbx = k_frac[static_cast<std::size_t>(ki)][0]
                                   + k_frac[static_cast<std::size_t>(kj)][0]
                                   - k_frac[static_cast<std::size_t>(ka)][0];
                const double fby = k_frac[static_cast<std::size_t>(ki)][1]
                                   + k_frac[static_cast<std::size_t>(kj)][1]
                                   - k_frac[static_cast<std::size_t>(ka)][1];
                const double fbz = k_frac[static_cast<std::size_t>(ki)][2]
                                   + k_frac[static_cast<std::size_t>(kj)][2]
                                   - k_frac[static_cast<std::size_t>(ka)][2];
                const int kb = find_k_index(k_frac, fbx, fby, fbz);
                if (kb < 0)
                    throw std::runtime_error(
                        "gpw mp2_uhf_energy_kpoints: momentum-conserving kb "
                        "not in the sampled mesh — use a closed (uniform) "
                        "k-mesh");
                for (int i = 0; i < n_oa; ++i)
                    for (int j = 0; j < n_ob; ++j)
                        for (int a = 0; a < n_va; ++a)
                            for (int b = 0; b < n_vb; ++b) {
                                // Via_a indexed with n_va; ρ_jb from β grids.
                                const Cplx v = contract_ovov_k_(
                                    Via_a, psi_occ_b_k, psi_virt_b_k, nk, n_va,
                                    npts, w, ki, ka, i, a, kj, kb, j, b);
                                const double denom =
                                    eps_occ_a_k[static_cast<std::size_t>(ki)]
                                               [static_cast<std::size_t>(i)]
                                    + eps_occ_b_k[static_cast<std::size_t>(kj)]
                                                 [static_cast<std::size_t>(j)]
                                    - eps_virt_a_k[static_cast<std::size_t>(ka)]
                                                  [static_cast<std::size_t>(a)]
                                    - eps_virt_b_k[static_cast<std::size_t>(kb)]
                                                  [static_cast<std::size_t>(b)];
                                if (std::abs(denom) < 1e-12)
                                    continue;
                                e_os += std::norm(v) * w_ij / denom;
                            }
            }
        }
    }
}

void direct_rpa_rhf(const std::vector<double>& v_ovov,
                    const std::vector<double>& eps_occ,
                    const std::vector<double>& eps_virt,
                    DirectRpaAmplitudes& out) {
    using linalg::RealMatrix;
    using linalg::sym_power;

    const int no = static_cast<int>(eps_occ.size());
    const int nv = static_cast<int>(eps_virt.size());
    if (no < 1 || nv < 1)
        throw std::invalid_argument(
            "gpw direct_rpa_rhf: need n_occ >= 1 and n_virt >= 1");
    const std::size_t nov =
        static_cast<std::size_t>(no) * static_cast<std::size_t>(nv);
    if (v_ovov.size() != nov * nov)
        throw std::invalid_argument(
            "gpw direct_rpa_rhf: tensor size does not match eps_occ/eps_virt");

    // Singlet dRPA (α_hf = 0): A = Δε δ + 2 V, B = 2 V, index p = i*nv + a.
    RealMatrix A =
        xt::zeros<double>({static_cast<int>(nov), static_cast<int>(nov)});
    RealMatrix B =
        xt::zeros<double>({static_cast<int>(nov), static_cast<int>(nov)});
    out.tr_A = 0.0;
    for (int i = 0; i < no; ++i)
        for (int a = 0; a < nv; ++a) {
            const int p = i * nv + a;
            const double de = eps_virt[static_cast<std::size_t>(a)]
                              - eps_occ[static_cast<std::size_t>(i)];
            for (int j = 0; j < no; ++j)
                for (int b = 0; b < nv; ++b) {
                    const int q = j * nv + b;
                    const double v = v_ovov[static_cast<std::size_t>(p) * nov
                                            + static_cast<std::size_t>(q)];
                    const double two_v = 2.0 * v;
                    double Aval = two_v;
                    if (p == q)
                        Aval += de;
                    A(p, q) = Aval;
                    B(p, q) = two_v;
                }
            out.tr_A += A(p, p);
        }

    constexpr double kAmBFloor = 1e-10;
    constexpr double kOmega2Tol = 1e-10;
    const RealMatrix Mp = A + B;
    const RealMatrix Mm = A - B;
    {
        auto [s, U] = xt::linalg::eigh(Mm);
        (void)U;
        if (s(0) <= kAmBFloor)
            throw std::runtime_error(
                "gpw direct_rpa_rhf: (A - B) is not positive definite "
                "(lowest eigenvalue = "
                + std::to_string(s(0))
                + " Ha) — dRPA reduction undefined (no fallback)");
    }
    const RealMatrix T = sym_power(Mm, 0.5, kAmBFloor, "gpw direct_rpa_rhf");
    const RealMatrix Ti = sym_power(Mm, -0.5, kAmBFloor, "gpw direct_rpa_rhf");
    RealMatrix M = xt::linalg::dot(T, RealMatrix(xt::linalg::dot(Mp, T)));
    for (std::size_t r = 0; r < nov; ++r)
        for (std::size_t c = r + 1; c < nov; ++c) {
            const double v = 0.5 * (M(r, c) + M(c, r));
            M(r, c) = v;
            M(c, r) = v;
        }
    auto [w2, Z] = xt::linalg::eigh(M);

    out.omega.assign(nov, 0.0);
    out.X.assign(nov * nov, 0.0);
    out.Y.assign(nov * nov, 0.0);
    double sum_om = 0.0;
    for (std::size_t s = 0; s < nov; ++s) {
        if (w2(s) < -kOmega2Tol)
            throw std::runtime_error(
                "gpw direct_rpa_rhf: dRPA omega^2 = " + std::to_string(w2(s))
                + " Ha^2 (root " + std::to_string(s)
                + ") is negative — imaginary excitation (no fallback)");
        if (w2(s) <= kOmega2Tol)
            throw std::runtime_error("gpw direct_rpa_rhf: dRPA omega^2 = "
                                     + std::to_string(w2(s)) + " Ha^2 (root "
                                     + std::to_string(s)
                                     + ") is numerically zero (no fallback)");
        const double w = std::sqrt(w2(s));
        out.omega[s] = w;
        sum_om += w;
        // (X+Y) = ω^{-1/2} T z, (X−Y) = ω^{1/2} T^{-1} z
        for (std::size_t r = 0; r < nov; ++r) {
            double xpy = 0.0, xmy = 0.0;
            for (std::size_t c = 0; c < nov; ++c) {
                xpy += T(static_cast<int>(r), static_cast<int>(c))
                       * Z(static_cast<int>(c), static_cast<int>(s));
                xmy += Ti(static_cast<int>(r), static_cast<int>(c))
                       * Z(static_cast<int>(c), static_cast<int>(s));
            }
            xpy /= std::sqrt(w);
            xmy *= std::sqrt(w);
            out.X[s * nov + r] = 0.5 * (xpy + xmy);
            out.Y[s * nov + r] = 0.5 * (xpy - xmy);
        }
    }
    out.e_corr = 0.5 * (sum_om - out.tr_A);
}

void direct_rpa_rhf_energy(const std::vector<double>& v_ovov,
                           const std::vector<double>& eps_occ,
                           const std::vector<double>& eps_virt, double& e_corr,
                           double& tr_A, std::vector<double>& omega) {
    DirectRpaAmplitudes full;
    direct_rpa_rhf(v_ovov, eps_occ, eps_virt, full);
    e_corr = full.e_corr;
    tr_A = full.tr_A;
    omega = std::move(full.omega);
}

void direct_rpa_uhf(
    const std::vector<double>& v_aa, const std::vector<double>& v_bb,
    const std::vector<double>& v_ab, const std::vector<double>& eps_occ_a,
    const std::vector<double>& eps_virt_a, const std::vector<double>& eps_occ_b,
    const std::vector<double>& eps_virt_b, DirectRpaAmplitudes& out) {
    using linalg::RealMatrix;
    using linalg::sym_power;

    const int noa = static_cast<int>(eps_occ_a.size());
    const int nva = static_cast<int>(eps_virt_a.size());
    const int nob = static_cast<int>(eps_occ_b.size());
    const int nvb = static_cast<int>(eps_virt_b.size());
    const int nov_a = noa * nva;
    const int nov_b = nob * nvb;
    const int nov = nov_a + nov_b;
    if (nov < 1)
        throw std::invalid_argument(
            "gpw direct_rpa_uhf: need at least one non-empty spin channel "
            "(n_ov_α + n_ov_β >= 1)");

    const std::size_t nov_a_sz = static_cast<std::size_t>(nov_a);
    const std::size_t nov_b_sz = static_cast<std::size_t>(nov_b);
    const std::size_t nov_sz = static_cast<std::size_t>(nov);

    if (nov_a > 0 && v_aa.size() != nov_a_sz * nov_a_sz)
        throw std::invalid_argument(
            "gpw direct_rpa_uhf: α same-spin tensor size mismatch");
    if (nov_b > 0 && v_bb.size() != nov_b_sz * nov_b_sz)
        throw std::invalid_argument(
            "gpw direct_rpa_uhf: β same-spin tensor size mismatch");
    if (nov_a > 0 && nov_b > 0 && v_ab.size() != nov_a_sz * nov_b_sz)
        throw std::invalid_argument(
            "gpw direct_rpa_uhf: cross-spin tensor size mismatch");
    if ((nov_a == 0 || nov_b == 0) && !v_ab.empty())
        throw std::invalid_argument(
            "gpw direct_rpa_uhf: cross-spin tensor present with empty channel");

    // Spin-conserving dRPA: A = Δε δ + V, B = V (spins explicit; no ×2).
    RealMatrix A = xt::zeros<double>({nov, nov});
    RealMatrix B = xt::zeros<double>({nov, nov});
    out.tr_A = 0.0;

    auto set_V = [&](int p, int q, double v) {
        A(p, q) += v;
        B(p, q) = v;
    };

    // αα block
    for (int i = 0; i < noa; ++i)
        for (int a = 0; a < nva; ++a) {
            const int p = i * nva + a;
            const double de = eps_virt_a[static_cast<std::size_t>(a)]
                              - eps_occ_a[static_cast<std::size_t>(i)];
            A(p, p) += de;
            for (int j = 0; j < noa; ++j)
                for (int b = 0; b < nva; ++b) {
                    const int q = j * nva + b;
                    const double v = v_aa[static_cast<std::size_t>(p) * nov_a_sz
                                          + static_cast<std::size_t>(q)];
                    set_V(p, q, v);
                }
            out.tr_A += A(p, p);
        }
    // ββ block
    for (int i = 0; i < nob; ++i)
        for (int a = 0; a < nvb; ++a) {
            const int p = nov_a + i * nvb + a;
            const double de = eps_virt_b[static_cast<std::size_t>(a)]
                              - eps_occ_b[static_cast<std::size_t>(i)];
            A(p, p) += de;
            for (int j = 0; j < nob; ++j)
                for (int b = 0; b < nvb; ++b) {
                    const int q = nov_a + j * nvb + b;
                    const int p0 = i * nvb + a;
                    const int q0 = j * nvb + b;
                    const double v =
                        v_bb[static_cast<std::size_t>(p0) * nov_b_sz
                             + static_cast<std::size_t>(q0)];
                    set_V(p, q, v);
                }
            out.tr_A += A(p, p);
        }
    // αβ / βα cross blocks
    if (nov_a > 0 && nov_b > 0) {
        for (int p0 = 0; p0 < nov_a; ++p0)
            for (int q0 = 0; q0 < nov_b; ++q0) {
                const double v = v_ab[static_cast<std::size_t>(p0) * nov_b_sz
                                      + static_cast<std::size_t>(q0)];
                set_V(p0, nov_a + q0, v);
                set_V(nov_a + q0, p0, v);
            }
    }

    constexpr double kAmBFloor = 1e-10;
    constexpr double kOmega2Tol = 1e-10;
    const RealMatrix Mp = A + B;
    const RealMatrix Mm = A - B;
    {
        auto [s, U] = xt::linalg::eigh(Mm);
        (void)U;
        if (s(0) <= kAmBFloor)
            throw std::runtime_error(
                "gpw direct_rpa_uhf: (A - B) is not positive definite "
                "(lowest eigenvalue = "
                + std::to_string(s(0))
                + " Ha) — dRPA reduction undefined (no fallback)");
    }
    const RealMatrix T = sym_power(Mm, 0.5, kAmBFloor, "gpw direct_rpa_uhf");
    const RealMatrix Ti = sym_power(Mm, -0.5, kAmBFloor, "gpw direct_rpa_uhf");
    RealMatrix M = xt::linalg::dot(T, RealMatrix(xt::linalg::dot(Mp, T)));
    for (std::size_t r = 0; r < nov_sz; ++r)
        for (std::size_t c = r + 1; c < nov_sz; ++c) {
            const double v = 0.5 * (M(r, c) + M(c, r));
            M(r, c) = v;
            M(c, r) = v;
        }
    auto [w2, Z] = xt::linalg::eigh(M);

    out.omega.assign(nov_sz, 0.0);
    out.X.assign(nov_sz * nov_sz, 0.0);
    out.Y.assign(nov_sz * nov_sz, 0.0);
    double sum_om = 0.0;
    for (std::size_t s = 0; s < nov_sz; ++s) {
        if (w2(s) < -kOmega2Tol)
            throw std::runtime_error(
                "gpw direct_rpa_uhf: dRPA omega^2 = " + std::to_string(w2(s))
                + " Ha^2 (root " + std::to_string(s)
                + ") is negative — imaginary excitation (no fallback)");
        if (w2(s) <= kOmega2Tol)
            throw std::runtime_error("gpw direct_rpa_uhf: dRPA omega^2 = "
                                     + std::to_string(w2(s)) + " Ha^2 (root "
                                     + std::to_string(s)
                                     + ") is numerically zero (no fallback)");
        const double w = std::sqrt(w2(s));
        out.omega[s] = w;
        sum_om += w;
        for (std::size_t r = 0; r < nov_sz; ++r) {
            double xpy = 0.0, xmy = 0.0;
            for (std::size_t c = 0; c < nov_sz; ++c) {
                xpy += T(static_cast<int>(r), static_cast<int>(c))
                       * Z(static_cast<int>(c), static_cast<int>(s));
                xmy += Ti(static_cast<int>(r), static_cast<int>(c))
                       * Z(static_cast<int>(c), static_cast<int>(s));
            }
            xpy /= std::sqrt(w);
            xmy *= std::sqrt(w);
            out.X[s * nov_sz + r] = 0.5 * (xpy + xmy);
            out.Y[s * nov_sz + r] = 0.5 * (xpy - xmy);
        }
    }
    out.e_corr = 0.5 * (sum_om - out.tr_A);
}

std::vector<double> mo_pair_potentials_ov(const UniformGrid& grid,
                                          const std::vector<double>& psi_occ,
                                          int n_occ,
                                          const std::vector<double>& psi_virt,
                                          int n_virt) {
    const long npts = grid.npts();
    if (psi_occ.size()
            != static_cast<std::size_t>(n_occ) * static_cast<std::size_t>(npts)
        || psi_virt.size()
               != static_cast<std::size_t>(n_virt)
                      * static_cast<std::size_t>(npts))
        throw std::invalid_argument(
            "gpw mo_pair_potentials_ov: MO grid arrays do not match "
            "n_occ/n_virt/npts");
    if (n_occ < 0 || n_virt < 0)
        throw std::invalid_argument(
            "gpw mo_pair_potentials_ov: negative orbital count");

    const int nov = n_occ * n_virt;
    std::vector<double> V(static_cast<std::size_t>(nov)
                          * static_cast<std::size_t>(npts));
    std::vector<double> rho(static_cast<std::size_t>(npts));
    for (int ia = 0; ia < nov; ++ia) {
        const double* pi = psi_occ.data()
                           + static_cast<std::size_t>(ia / n_virt)
                                 * static_cast<std::size_t>(npts);
        const double* pa = psi_virt.data()
                           + static_cast<std::size_t>(ia % n_virt)
                                 * static_cast<std::size_t>(npts);
        for (long g = 0; g < npts; ++g)
            rho[static_cast<std::size_t>(g)] = pi[g] * pa[g];
        const PoissonResult pr = solve_poisson(grid, rho);
        double* Via =
            V.data()
            + static_cast<std::size_t>(ia) * static_cast<std::size_t>(npts);
        for (long g = 0; g < npts; ++g)
            Via[g] = pr.v_r[static_cast<std::size_t>(g)];
    }
    return V;
}

std::vector<double> exchange_diag_mt(const UniformGrid& grid,
                                     const std::vector<double>& psi_mo,
                                     int n_mo, int n_occ) {
    const long npts = grid.npts();
    const double w = grid.weight();
    if (n_mo < 1 || n_occ < 1 || n_occ > n_mo)
        throw std::invalid_argument("gpw exchange_diag_mt: bad n_mo/n_occ");
    if (psi_mo.size()
        != static_cast<std::size_t>(n_mo) * static_cast<std::size_t>(npts))
        throw std::invalid_argument(
            "gpw exchange_diag_mt: MO grid size mismatch");

    std::vector<double> sx(static_cast<std::size_t>(n_mo), 0.0);
    std::vector<double> rho(static_cast<std::size_t>(npts));
    for (int p = 0; p < n_mo; ++p) {
        const double* pp =
            psi_mo.data()
            + static_cast<std::size_t>(p) * static_cast<std::size_t>(npts);
        double sum = 0.0;
        for (int i = 0; i < n_occ; ++i) {
            const double* pi =
                psi_mo.data()
                + static_cast<std::size_t>(i) * static_cast<std::size_t>(npts);
            for (long g = 0; g < npts; ++g)
                rho[static_cast<std::size_t>(g)] = pp[g] * pi[g];
            // MT truncated Coulomb — G=0-safe for charged (ii|ii) pairs.
            const PoissonResult pr = solve_poisson_mt(grid, rho);
            double s = 0.0;
            for (long g = 0; g < npts; ++g)
                s += pr.v_r[static_cast<std::size_t>(g)]
                     * rho[static_cast<std::size_t>(g)];
            sum -= w * s; // Σ_x,p = −Σ_i (pi|pi)
        }
        sx[static_cast<std::size_t>(p)] = sum;
    }
    return sx;
}

void direct_rpa_rhf_energy_kpoints(
    const UniformGrid& grid, const std::vector<std::vector<Cplx>>& psi_occ_k,
    const std::vector<std::vector<Cplx>>& psi_virt_k,
    const std::vector<std::vector<double>>& eps_occ_k,
    const std::vector<std::vector<double>>& eps_virt_k,
    const std::vector<std::array<double, 3>>& k_frac,
    const std::vector<double>& k_weights, double& e_corr, double& tr_A,
    std::vector<double>& omega) {
    using linalg::CplxMatrix;

    const int nk = static_cast<int>(k_frac.size());
    if (nk < 1)
        throw std::invalid_argument(
            "gpw direct_rpa_rhf_energy_kpoints: empty k mesh");
    if (static_cast<int>(psi_occ_k.size()) != nk
        || static_cast<int>(psi_virt_k.size()) != nk
        || static_cast<int>(eps_occ_k.size()) != nk
        || static_cast<int>(eps_virt_k.size()) != nk
        || static_cast<int>(k_weights.size()) != nk)
        throw std::invalid_argument(
            "gpw direct_rpa_rhf_energy_kpoints: per-k array length mismatch");

    const int n_occ = static_cast<int>(eps_occ_k[0].size());
    const int n_virt = static_cast<int>(eps_virt_k[0].size());
    if (n_occ < 1 || n_virt < 1)
        throw std::invalid_argument(
            "gpw direct_rpa_rhf_energy_kpoints: need n_occ>=1 and n_virt>=1");
    const long npts = grid.npts();
    const double w_grid = grid.weight();
    const int nov = n_occ * n_virt;

    for (int ik = 0; ik < nk; ++ik) {
        if (static_cast<int>(eps_occ_k[static_cast<std::size_t>(ik)].size())
                != n_occ
            || static_cast<int>(eps_virt_k[static_cast<std::size_t>(ik)].size())
                   != n_virt)
            throw std::invalid_argument(
                "gpw direct_rpa_rhf_energy_kpoints: inconsistent n_occ/n_virt "
                "across k");
        const std::size_t need_o =
            static_cast<std::size_t>(n_occ) * static_cast<std::size_t>(npts);
        const std::size_t need_v =
            static_cast<std::size_t>(n_virt) * static_cast<std::size_t>(npts);
        if (psi_occ_k[static_cast<std::size_t>(ik)].size() != need_o
            || psi_virt_k[static_cast<std::size_t>(ik)].size() != need_v)
            throw std::invalid_argument(
                "gpw direct_rpa_rhf_energy_kpoints: MO grid size mismatch");
    }
    double wsum = 0.0;
    for (double wk : k_weights)
        wsum += wk;
    if (std::abs(wsum - 1.0) > 1e-8)
        throw std::invalid_argument(
            "gpw direct_rpa_rhf_energy_kpoints: k_weights must sum to 1");

    // Product multi-k dRPA: finite-q inter-k dielectric (Furche plasmon).
    // q-points are unique wrapped differences ka − ki over the sampled k mesh
    // (Γ-avoiding MP meshes are not groups under addition, so q ≠ k in
    // general). For each transfer q, Casida from transitions i_k → a_{k+q}
    // for every ik with k+q on the mesh, Coulomb at q (RI-GPW Poisson),
    // Furche-averaged
    //   E_c = Σ_q w_q · ½(Σ_s Ω_{q,s} − Tr A_q),
    // with w_q = n_src(q) / N_k² * N_k = n_src(q)/N_k  so Σ_q w_q = 1 when
    // every ordered pair (ki,ka) contributes once (n_src summed over q = N_k).
    // Recovers the real Γ formula when N_k = 1 and orbitals are real.
    constexpr double kAmBFloor = 1e-10;
    constexpr double kOmega2Tol = 1e-10;
    constexpr double kQTol = 1e-10;

    e_corr = 0.0;
    tr_A = 0.0;
    omega.clear();
    std::vector<Cplx> rho(static_cast<std::size_t>(npts));

    // Unique q = wrap(ka − ki) over all ordered pairs.
    std::vector<std::array<double, 3>> q_list;
    auto q_index_of = [&](double qx, double qy, double qz) -> int {
        const double tx = wrap_frac(qx), ty = wrap_frac(qy), tz = wrap_frac(qz);
        for (std::size_t iq = 0; iq < q_list.size(); ++iq) {
            const double dx = wrap_frac(q_list[iq][0] - tx);
            const double dy = wrap_frac(q_list[iq][1] - ty);
            const double dz = wrap_frac(q_list[iq][2] - tz);
            if (std::abs(dx) < kQTol && std::abs(dy) < kQTol
                && std::abs(dz) < kQTol)
                return static_cast<int>(iq);
        }
        return -1;
    };
    for (int ik = 0; ik < nk; ++ik) {
        for (int ka = 0; ka < nk; ++ka) {
            const double qx = k_frac[static_cast<std::size_t>(ka)][0]
                              - k_frac[static_cast<std::size_t>(ik)][0];
            const double qy = k_frac[static_cast<std::size_t>(ka)][1]
                              - k_frac[static_cast<std::size_t>(ik)][1];
            const double qz = k_frac[static_cast<std::size_t>(ka)][2]
                              - k_frac[static_cast<std::size_t>(ik)][2];
            if (q_index_of(qx, qy, qz) < 0)
                q_list.push_back(
                    {{wrap_frac(qx), wrap_frac(qy), wrap_frac(qz)}});
        }
    }
    const int nq = static_cast<int>(q_list.size());
    if (nq < 1)
        throw std::runtime_error(
            "gpw direct_rpa_rhf_energy_kpoints: empty q mesh from k "
            "differences");

    for (int iq = 0; iq < nq; ++iq) {
        const double q_frac[3] = {q_list[static_cast<std::size_t>(iq)][0],
                                  q_list[static_cast<std::size_t>(iq)][1],
                                  q_list[static_cast<std::size_t>(iq)][2]};
        (void)q_frac; // diagnostics; Bloch ρ already carries e^{iq·r}

        // Sources ik with ka = k[ik] + q on the mesh.
        std::vector<int> src_k;
        std::vector<int> tgt_k;
        src_k.reserve(static_cast<std::size_t>(nk));
        tgt_k.reserve(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const double fx =
                k_frac[static_cast<std::size_t>(ik)][0] + q_frac[0];
            const double fy =
                k_frac[static_cast<std::size_t>(ik)][1] + q_frac[1];
            const double fz =
                k_frac[static_cast<std::size_t>(ik)][2] + q_frac[2];
            const int ka = find_k_index(k_frac, fx, fy, fz);
            if (ka >= 0) {
                src_k.push_back(ik);
                tgt_k.push_back(ka);
            }
        }
        const int n_src = static_cast<int>(src_k.size());
        if (n_src < 1)
            continue;
        // Equal-pair weight: each ordered (ki,ka) contributes 1/N_k once
        // when summed over q of n_src(q)/N_k  (Σ_q n_src = N_k when every
        // ki has exactly one ka per… wait: Σ_q n_src(q) = N_k² ordered
        // pairs that land on mesh = N_k² always since every (ki,ka) defines
        // a q). So Σ_q w_q = Σ_q n_src/N_k = N_k. Need /N_k more → w_q =
        // n_src / N_k².
        const double wq = static_cast<double>(n_src)
                          / (static_cast<double>(nk) * static_cast<double>(nk));

        const int n_block = n_src * nov;
        std::vector<std::vector<Cplx>> Via(static_cast<std::size_t>(n_block));
        std::vector<double> de(static_cast<std::size_t>(n_block));
        std::vector<std::vector<Cplx>> rho_alpha(
            static_cast<std::size_t>(n_block));

        for (int is = 0; is < n_src; ++is) {
            const int ik = src_k[static_cast<std::size_t>(is)];
            const int ka = tgt_k[static_cast<std::size_t>(is)];
            const auto& po = psi_occ_k[static_cast<std::size_t>(ik)];
            const auto& pv = psi_virt_k[static_cast<std::size_t>(ka)];
            const auto& eo = eps_occ_k[static_cast<std::size_t>(ik)];
            const auto& ev = eps_virt_k[static_cast<std::size_t>(ka)];

            for (int i = 0; i < n_occ; ++i) {
                for (int a = 0; a < n_virt; ++a) {
                    const int ia = i * n_virt + a;
                    const int alpha = is * nov + ia;
                    const Cplx* pi = po.data()
                                     + static_cast<std::size_t>(i)
                                           * static_cast<std::size_t>(npts);
                    const Cplx* pa = pv.data()
                                     + static_cast<std::size_t>(a)
                                           * static_cast<std::size_t>(npts);
                    rho_alpha[static_cast<std::size_t>(alpha)].resize(
                        static_cast<std::size_t>(npts));
                    for (long g = 0; g < npts; ++g) {
                        const Cplx r = std::conj(pi[g]) * pa[g];
                        rho[static_cast<std::size_t>(g)] = r;
                        rho_alpha[static_cast<std::size_t>(alpha)]
                                 [static_cast<std::size_t>(g)] = r;
                    }
                    // Full Bloch product ρ = ψ_i^{k*} ψ_a^{k+q} already
                    // carries the e^{iq·r} envelope from the AO Bloch sum —
                    // use plain Poisson (G=0 dropped; ρ is charge-neutral by
                    // k-orthogonality). Do NOT also shift by q in G-space
                    // (that double-counts the transfer; same as multi-k MP2).
                    poisson_complex(grid, rho.data(),
                                    Via[static_cast<std::size_t>(alpha)]);

                    de[static_cast<std::size_t>(alpha)] =
                        ev[static_cast<std::size_t>(a)]
                        - eo[static_cast<std::size_t>(i)];
                    if (de[static_cast<std::size_t>(alpha)] <= kAmBFloor)
                        throw std::runtime_error(
                            "gpw direct_rpa_rhf_energy_kpoints: non-positive "
                            "Δε at a (q,k,i,a) transition — dRPA (A−B) not PD");
                }
            }
        }

        // Hermitian Coulomb form (ρ_p | ρ_q) = ∫ V[ρ_p]* ρ_q  (complex PD).
        CplxMatrix Vmat = xt::zeros<Cplx>({n_block, n_block});
        for (int p = 0; p < n_block; ++p) {
            const Cplx* Vp = Via[static_cast<std::size_t>(p)].data();
            for (int qq = 0; qq < n_block; ++qq) {
                const Cplx* rq = rho_alpha[static_cast<std::size_t>(qq)].data();
                Cplx s{0.0, 0.0};
                for (long g = 0; g < npts; ++g)
                    s += std::conj(Vp[g]) * rq[g];
                Vmat(p, qq) = w_grid * s;
            }
        }
        for (int p = 0; p < n_block; ++p) {
            Vmat(p, p) = Cplx{Vmat(p, p).real(), 0.0};
            for (int qq = p + 1; qq < n_block; ++qq) {
                const Cplx avg = 0.5 * (Vmat(p, qq) + std::conj(Vmat(qq, p)));
                Vmat(p, qq) = avg;
                Vmat(qq, p) = std::conj(avg);
            }
        }

        CplxMatrix M = xt::zeros<Cplx>({n_block, n_block});
        double tr_Aq = 0.0;
        for (int p = 0; p < n_block; ++p) {
            const double dep = de[static_cast<std::size_t>(p)];
            const double sp = std::sqrt(dep);
            tr_Aq += dep + 2.0 * Vmat(p, p).real();
            for (int qq = 0; qq < n_block; ++qq) {
                const double sq = std::sqrt(de[static_cast<std::size_t>(qq)]);
                Cplx apb = 4.0 * Vmat(p, qq);
                if (p == qq)
                    apb += dep;
                M(p, qq) = sp * apb * sq;
            }
        }
        for (int p = 0; p < n_block; ++p) {
            M(p, p) = Cplx{M(p, p).real(), 0.0};
            for (int qq = p + 1; qq < n_block; ++qq) {
                const Cplx avg = 0.5 * (M(p, qq) + std::conj(M(qq, p)));
                M(p, qq) = avg;
                M(qq, p) = std::conj(avg);
            }
        }

        std::vector<double> w2;
        CplxMatrix Z;
        linalg::eigh_hermitian(M, w2, Z);
        (void)Z;

        double sum_om = 0.0;
        for (int s = 0; s < n_block; ++s) {
            double lam = w2[static_cast<std::size_t>(s)];
            if (lam < -kOmega2Tol)
                throw std::runtime_error(
                    "gpw direct_rpa_rhf_energy_kpoints: dRPA omega^2 = "
                    + std::to_string(lam) + " Ha^2 (q=" + std::to_string(iq)
                    + " root " + std::to_string(s)
                    + ") is negative — imaginary excitation (no fallback)");
            if (lam <= kOmega2Tol)
                throw std::runtime_error(
                    "gpw direct_rpa_rhf_energy_kpoints: dRPA omega^2 = "
                    + std::to_string(lam) + " Ha^2 (q=" + std::to_string(iq)
                    + " root " + std::to_string(s)
                    + ") is numerically zero (no fallback)");
            const double om = std::sqrt(lam);
            omega.push_back(om);
            sum_om += om;
        }
        e_corr += wq * 0.5 * (sum_om - tr_Aq);
        tr_A += wq * tr_Aq;
    }
    std::sort(omega.begin(), omega.end());
}

void direct_rpa_uhf_energy_kpoints(
    const UniformGrid& grid, const std::vector<std::vector<Cplx>>& psi_occ_a_k,
    const std::vector<std::vector<Cplx>>& psi_virt_a_k,
    const std::vector<std::vector<Cplx>>& psi_occ_b_k,
    const std::vector<std::vector<Cplx>>& psi_virt_b_k,
    const std::vector<std::vector<double>>& eps_occ_a_k,
    const std::vector<std::vector<double>>& eps_virt_a_k,
    const std::vector<std::vector<double>>& eps_occ_b_k,
    const std::vector<std::vector<double>>& eps_virt_b_k,
    const std::vector<std::array<double, 3>>& k_frac,
    const std::vector<double>& k_weights, double& e_corr, double& tr_A,
    std::vector<double>& omega) {
    using linalg::CplxMatrix;

    const int nk = static_cast<int>(k_frac.size());
    if (nk < 1)
        throw std::invalid_argument(
            "gpw direct_rpa_uhf_energy_kpoints: empty k mesh");
    if (static_cast<int>(psi_occ_a_k.size()) != nk
        || static_cast<int>(psi_virt_a_k.size()) != nk
        || static_cast<int>(psi_occ_b_k.size()) != nk
        || static_cast<int>(psi_virt_b_k.size()) != nk
        || static_cast<int>(eps_occ_a_k.size()) != nk
        || static_cast<int>(eps_virt_a_k.size()) != nk
        || static_cast<int>(eps_occ_b_k.size()) != nk
        || static_cast<int>(eps_virt_b_k.size()) != nk
        || static_cast<int>(k_weights.size()) != nk)
        throw std::invalid_argument(
            "gpw direct_rpa_uhf_energy_kpoints: per-k array length mismatch");

    const int n_oa = static_cast<int>(eps_occ_a_k[0].size());
    const int n_va = static_cast<int>(eps_virt_a_k[0].size());
    const int n_ob = static_cast<int>(eps_occ_b_k[0].size());
    const int n_vb = static_cast<int>(eps_virt_b_k[0].size());
    const int nov_a = n_oa * n_va;
    const int nov_b = n_ob * n_vb;
    const int nov = nov_a + nov_b;
    if (nov < 1)
        throw std::invalid_argument(
            "gpw direct_rpa_uhf_energy_kpoints: need at least one non-empty "
            "spin channel (n_ov_α + n_ov_β >= 1)");
    if ((n_oa == 0) != (n_va == 0) || (n_ob == 0) != (n_vb == 0))
        throw std::invalid_argument(
            "gpw direct_rpa_uhf_energy_kpoints: each spin channel must be "
            "empty on both occ and virt or neither");

    const long npts = grid.npts();
    const double w_grid = grid.weight();

    for (int ik = 0; ik < nk; ++ik) {
        if (static_cast<int>(eps_occ_a_k[static_cast<std::size_t>(ik)].size())
                != n_oa
            || static_cast<int>(
                   eps_virt_a_k[static_cast<std::size_t>(ik)].size())
                   != n_va
            || static_cast<int>(
                   eps_occ_b_k[static_cast<std::size_t>(ik)].size())
                   != n_ob
            || static_cast<int>(
                   eps_virt_b_k[static_cast<std::size_t>(ik)].size())
                   != n_vb)
            throw std::invalid_argument(
                "gpw direct_rpa_uhf_energy_kpoints: inconsistent n_occ/n_virt "
                "across k");
        const std::size_t need_oa =
            static_cast<std::size_t>(n_oa) * static_cast<std::size_t>(npts);
        const std::size_t need_va =
            static_cast<std::size_t>(n_va) * static_cast<std::size_t>(npts);
        const std::size_t need_ob =
            static_cast<std::size_t>(n_ob) * static_cast<std::size_t>(npts);
        const std::size_t need_vb =
            static_cast<std::size_t>(n_vb) * static_cast<std::size_t>(npts);
        if (psi_occ_a_k[static_cast<std::size_t>(ik)].size() != need_oa
            || psi_virt_a_k[static_cast<std::size_t>(ik)].size() != need_va
            || psi_occ_b_k[static_cast<std::size_t>(ik)].size() != need_ob
            || psi_virt_b_k[static_cast<std::size_t>(ik)].size() != need_vb)
            throw std::invalid_argument(
                "gpw direct_rpa_uhf_energy_kpoints: MO grid size mismatch");
    }
    double wsum = 0.0;
    for (double wk : k_weights)
        wsum += wk;
    if (std::abs(wsum - 1.0) > 1e-8)
        throw std::invalid_argument(
            "gpw direct_rpa_uhf_energy_kpoints: k_weights must sum to 1");

    // Product multi-k URPA: finite-q inter-k dielectric with spins explicit
    // (same q construction as closed-shell multi-k dRPA; Casida factor 2
    // not 4 — matches spectral_g0w0_uhf_kpoints / Γ direct_rpa_uhf).
    constexpr double kAmBFloor = 1e-10;
    constexpr double kOmega2Tol = 1e-10;
    constexpr double kQTol = 1e-10;

    e_corr = 0.0;
    tr_A = 0.0;
    omega.clear();
    std::vector<Cplx> rho(static_cast<std::size_t>(npts));

    std::vector<std::array<double, 3>> q_list;
    auto q_index_of = [&](double qx, double qy, double qz) -> int {
        const double tx = wrap_frac(qx), ty = wrap_frac(qy), tz = wrap_frac(qz);
        for (std::size_t iq = 0; iq < q_list.size(); ++iq) {
            const double dx = wrap_frac(q_list[iq][0] - tx);
            const double dy = wrap_frac(q_list[iq][1] - ty);
            const double dz = wrap_frac(q_list[iq][2] - tz);
            if (std::abs(dx) < kQTol && std::abs(dy) < kQTol
                && std::abs(dz) < kQTol)
                return static_cast<int>(iq);
        }
        return -1;
    };
    for (int ik = 0; ik < nk; ++ik) {
        for (int ka = 0; ka < nk; ++ka) {
            const double qx = k_frac[static_cast<std::size_t>(ka)][0]
                              - k_frac[static_cast<std::size_t>(ik)][0];
            const double qy = k_frac[static_cast<std::size_t>(ka)][1]
                              - k_frac[static_cast<std::size_t>(ik)][1];
            const double qz = k_frac[static_cast<std::size_t>(ka)][2]
                              - k_frac[static_cast<std::size_t>(ik)][2];
            if (q_index_of(qx, qy, qz) < 0)
                q_list.push_back(
                    {{wrap_frac(qx), wrap_frac(qy), wrap_frac(qz)}});
        }
    }
    const int nq = static_cast<int>(q_list.size());
    if (nq < 1)
        throw std::runtime_error(
            "gpw direct_rpa_uhf_energy_kpoints: empty q mesh from k "
            "differences");

    for (int iq = 0; iq < nq; ++iq) {
        const double q_frac[3] = {q_list[static_cast<std::size_t>(iq)][0],
                                  q_list[static_cast<std::size_t>(iq)][1],
                                  q_list[static_cast<std::size_t>(iq)][2]};

        std::vector<int> src_k;
        std::vector<int> tgt_k;
        src_k.reserve(static_cast<std::size_t>(nk));
        tgt_k.reserve(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const double fx =
                k_frac[static_cast<std::size_t>(ik)][0] + q_frac[0];
            const double fy =
                k_frac[static_cast<std::size_t>(ik)][1] + q_frac[1];
            const double fz =
                k_frac[static_cast<std::size_t>(ik)][2] + q_frac[2];
            const int ka = find_k_index(k_frac, fx, fy, fz);
            if (ka >= 0) {
                src_k.push_back(ik);
                tgt_k.push_back(ka);
            }
        }
        const int n_src = static_cast<int>(src_k.size());
        if (n_src < 1)
            continue;
        const double wq = static_cast<double>(n_src)
                          / (static_cast<double>(nk) * static_cast<double>(nk));

        const int n_block = n_src * nov;
        std::vector<std::vector<Cplx>> Via(static_cast<std::size_t>(n_block));
        std::vector<double> de(static_cast<std::size_t>(n_block));
        std::vector<std::vector<Cplx>> rho_alpha(
            static_cast<std::size_t>(n_block));

        auto fill_spin = [&](int is, int nov_off, int n_o, int n_v,
                             const std::vector<std::vector<Cplx>>& psi_o_k,
                             const std::vector<std::vector<Cplx>>& psi_v_k,
                             const std::vector<std::vector<double>>& eps_o_k,
                             const std::vector<std::vector<double>>& eps_v_k) {
            const int ik = src_k[static_cast<std::size_t>(is)];
            const int ka = tgt_k[static_cast<std::size_t>(is)];
            const auto& po = psi_o_k[static_cast<std::size_t>(ik)];
            const auto& pv = psi_v_k[static_cast<std::size_t>(ka)];
            const auto& eo = eps_o_k[static_cast<std::size_t>(ik)];
            const auto& ev = eps_v_k[static_cast<std::size_t>(ka)];
            for (int i = 0; i < n_o; ++i) {
                for (int a = 0; a < n_v; ++a) {
                    const int ia = i * n_v + a;
                    const int alpha = is * nov + nov_off + ia;
                    const Cplx* pi = po.data()
                                     + static_cast<std::size_t>(i)
                                           * static_cast<std::size_t>(npts);
                    const Cplx* pa = pv.data()
                                     + static_cast<std::size_t>(a)
                                           * static_cast<std::size_t>(npts);
                    rho_alpha[static_cast<std::size_t>(alpha)].resize(
                        static_cast<std::size_t>(npts));
                    for (long g = 0; g < npts; ++g) {
                        const Cplx r = std::conj(pi[g]) * pa[g];
                        rho[static_cast<std::size_t>(g)] = r;
                        rho_alpha[static_cast<std::size_t>(alpha)]
                                 [static_cast<std::size_t>(g)] = r;
                    }
                    poisson_complex(grid, rho.data(),
                                    Via[static_cast<std::size_t>(alpha)]);
                    de[static_cast<std::size_t>(alpha)] =
                        ev[static_cast<std::size_t>(a)]
                        - eo[static_cast<std::size_t>(i)];
                    if (de[static_cast<std::size_t>(alpha)] <= kAmBFloor)
                        throw std::runtime_error(
                            "gpw direct_rpa_uhf_energy_kpoints: "
                            "non-positive Δε at a (q,k,i,a,σ) transition "
                            "— dRPA (A−B) not PD");
                }
            }
        };

        for (int is = 0; is < n_src; ++is) {
            if (nov_a > 0)
                fill_spin(is, 0, n_oa, n_va, psi_occ_a_k, psi_virt_a_k,
                          eps_occ_a_k, eps_virt_a_k);
            if (nov_b > 0)
                fill_spin(is, nov_a, n_ob, n_vb, psi_occ_b_k, psi_virt_b_k,
                          eps_occ_b_k, eps_virt_b_k);
        }

        CplxMatrix Vmat = xt::zeros<Cplx>({n_block, n_block});
        for (int p = 0; p < n_block; ++p) {
            const Cplx* Vp = Via[static_cast<std::size_t>(p)].data();
            for (int qq = 0; qq < n_block; ++qq) {
                const Cplx* rq = rho_alpha[static_cast<std::size_t>(qq)].data();
                Cplx s{0.0, 0.0};
                for (long g = 0; g < npts; ++g)
                    s += std::conj(Vp[g]) * rq[g];
                Vmat(p, qq) = w_grid * s;
            }
        }
        for (int p = 0; p < n_block; ++p) {
            Vmat(p, p) = Cplx{Vmat(p, p).real(), 0.0};
            for (int qq = p + 1; qq < n_block; ++qq) {
                const Cplx avg = 0.5 * (Vmat(p, qq) + std::conj(Vmat(qq, p)));
                Vmat(p, qq) = avg;
                Vmat(qq, p) = std::conj(avg);
            }
        }

        // UHF Casida: A−B = diag(Δε), A+B = diag(Δε) + 2V (spins explicit).
        CplxMatrix M = xt::zeros<Cplx>({n_block, n_block});
        double tr_Aq = 0.0;
        for (int p = 0; p < n_block; ++p) {
            const double dep = de[static_cast<std::size_t>(p)];
            const double sp = std::sqrt(dep);
            tr_Aq += dep + Vmat(p, p).real();
            for (int qq = 0; qq < n_block; ++qq) {
                const double sq = std::sqrt(de[static_cast<std::size_t>(qq)]);
                Cplx apb = 2.0 * Vmat(p, qq);
                if (p == qq)
                    apb += dep;
                M(p, qq) = sp * apb * sq;
            }
        }
        for (int p = 0; p < n_block; ++p) {
            M(p, p) = Cplx{M(p, p).real(), 0.0};
            for (int qq = p + 1; qq < n_block; ++qq) {
                const Cplx avg = 0.5 * (M(p, qq) + std::conj(M(qq, p)));
                M(p, qq) = avg;
                M(qq, p) = std::conj(avg);
            }
        }

        std::vector<double> w2;
        CplxMatrix Z;
        linalg::eigh_hermitian(M, w2, Z);
        (void)Z;

        double sum_om = 0.0;
        for (int s = 0; s < n_block; ++s) {
            double lam = w2[static_cast<std::size_t>(s)];
            if (lam < -kOmega2Tol)
                throw std::runtime_error(
                    "gpw direct_rpa_uhf_energy_kpoints: dRPA omega^2 = "
                    + std::to_string(lam) + " Ha^2 (q=" + std::to_string(iq)
                    + " root " + std::to_string(s)
                    + ") is negative — imaginary excitation (no fallback)");
            if (lam <= kOmega2Tol)
                throw std::runtime_error(
                    "gpw direct_rpa_uhf_energy_kpoints: dRPA omega^2 = "
                    + std::to_string(lam) + " Ha^2 (q=" + std::to_string(iq)
                    + " root " + std::to_string(s)
                    + ") is numerically zero (no fallback)");
            const double om = std::sqrt(lam);
            omega.push_back(om);
            sum_om += om;
        }
        e_corr += wq * 0.5 * (sum_om - tr_Aq);
        tr_A += wq * tr_Aq;
    }
    std::sort(omega.begin(), omega.end());
}

namespace {

constexpr double kGwEta = 1e-8;
constexpr double kGwQpeTol = 1e-8;
constexpr int kGwQpeMaxIter = 100;

/// One spectral Σ_c contribution: pole Ω, intermediate ε_m, scaled |t|².
struct ScTerm {
    double Omega = 0.0;
    double eps_m = 0.0;
    double t2 = 0.0; ///< |t|² × q-weight
    bool m_occ = true;
};

/// Diagonal Re Σ_c,p(ω) from a flat list of finite-q spectral terms.
double sigma_c_from_terms(double omega, const std::vector<ScTerm>& terms) {
    double re = 0.0;
    for (const ScTerm& t : terms) {
        const double den_r =
            t.m_occ ? (omega + t.Omega - t.eps_m) : (omega - t.Omega - t.eps_m);
        const double den2 = den_r * den_r + kGwEta * kGwEta;
        re += t.t2 * den_r / den2;
    }
    return re;
}

bool solve_qp_terms(double eps_p, double vx_m_vxc,
                    const std::vector<ScTerm>& terms, double& omega_out) {
    double w = eps_p;
    constexpr double kH = 1e-6;
    for (int it = 0; it < kGwQpeMaxIter; ++it) {
        const double sig = sigma_c_from_terms(w, terms);
        const double f = w - eps_p - (sig + vx_m_vxc);
        if (std::abs(f) < kGwQpeTol) {
            omega_out = w;
            return true;
        }
        const double sig_p = sigma_c_from_terms(w + kH, terms);
        const double dsig = (sig_p - sig) / kH;
        const double df = 1.0 - dsig;
        if (!(std::abs(df) > 1e-14))
            return false;
        w -= f / df;
    }
    omega_out = w;
    return std::abs(w - eps_p - (sigma_c_from_terms(w, terms) + vx_m_vxc))
           < 10.0 * kGwQpeTol;
}

/// MT Poisson on a complex density via independent real/imag solves.
void poisson_mt_complex(const UniformGrid& grid, const Cplx* rho,
                        std::vector<Cplx>& V) {
    const long npts = grid.npts();
    std::vector<double> re(static_cast<std::size_t>(npts));
    std::vector<double> im(static_cast<std::size_t>(npts));
    for (long g = 0; g < npts; ++g) {
        re[static_cast<std::size_t>(g)] = rho[g].real();
        im[static_cast<std::size_t>(g)] = rho[g].imag();
    }
    const PoissonResult vr = solve_poisson_mt(grid, re);
    const PoissonResult vi = solve_poisson_mt(grid, im);
    V.resize(static_cast<std::size_t>(npts));
    for (long g = 0; g < npts; ++g)
        V[static_cast<std::size_t>(g)] =
            Cplx{vr.v_r[static_cast<std::size_t>(g)],
                 vi.v_r[static_cast<std::size_t>(g)]};
}

} // namespace

void spectral_g0w0_rhf_kpoints(const UniformGrid& grid,
                               const std::vector<std::vector<Cplx>>& psi_mo_k,
                               const std::vector<std::vector<double>>& eps_k,
                               const std::vector<std::array<double, 3>>& k_frac,
                               const std::vector<double>& k_weights, int n_occ,
                               int n_virt, const std::vector<double>& vxc_r,
                               std::vector<std::vector<double>>& qp_energy_k,
                               std::vector<std::vector<double>>& sigma_x_k,
                               std::vector<std::vector<double>>& vxc_diag_k,
                               bool& all_converged) {
    using linalg::CplxMatrix;

    const int nk = static_cast<int>(psi_mo_k.size());
    if (nk < 1)
        throw std::invalid_argument(
            "gpw spectral_g0w0_rhf_kpoints: empty k mesh");
    if (static_cast<int>(eps_k.size()) != nk
        || static_cast<int>(k_weights.size()) != nk
        || static_cast<int>(k_frac.size()) != nk)
        throw std::invalid_argument(
            "gpw spectral_g0w0_rhf_kpoints: per-k array length mismatch");
    if (n_occ < 1 || n_virt < 1)
        throw std::invalid_argument(
            "gpw spectral_g0w0_rhf_kpoints: need n_occ>=1 and n_virt>=1");

    const long npts = grid.npts();
    const double w_grid = grid.weight();
    if (static_cast<int>(vxc_r.size()) != static_cast<int>(npts))
        throw std::invalid_argument(
            "gpw spectral_g0w0_rhf_kpoints: vxc_r size != npts");

    const int n_mo = static_cast<int>(eps_k[0].size());
    if (n_mo < n_occ + n_virt)
        throw std::invalid_argument(
            "gpw spectral_g0w0_rhf_kpoints: n_mo < n_occ + n_virt");
    const int nov = n_occ * n_virt;
    const std::size_t need_mo =
        static_cast<std::size_t>(n_mo) * static_cast<std::size_t>(npts);

    for (int ik = 0; ik < nk; ++ik) {
        if (static_cast<int>(eps_k[static_cast<std::size_t>(ik)].size())
            != n_mo)
            throw std::invalid_argument(
                "gpw spectral_g0w0_rhf_kpoints: inconsistent n_mo across k");
        if (psi_mo_k[static_cast<std::size_t>(ik)].size() != need_mo)
            throw std::invalid_argument(
                "gpw spectral_g0w0_rhf_kpoints: MO grid size mismatch");
    }
    double wsum = 0.0;
    for (double wk : k_weights)
        wsum += wk;
    if (std::abs(wsum - 1.0) > 1e-8)
        throw std::invalid_argument(
            "gpw spectral_g0w0_rhf_kpoints: k_weights must sum to 1");

    constexpr double kAmBFloor = 1e-10;
    constexpr double kOmega2Tol = 1e-10;
    constexpr double kQTol = 1e-10;
    const double sqrt2 = std::sqrt(2.0);
    // Equal-mesh spectral weight: each q contributes 1/N_k (N_k=1 → 1).
    const double w_q_se = 1.0 / static_cast<double>(nk);

    qp_energy_k.assign(
        static_cast<std::size_t>(nk),
        std::vector<double>(static_cast<std::size_t>(n_mo), 0.0));
    sigma_x_k.assign(static_cast<std::size_t>(nk),
                     std::vector<double>(static_cast<std::size_t>(n_mo), 0.0));
    vxc_diag_k.assign(static_cast<std::size_t>(nk),
                      std::vector<double>(static_cast<std::size_t>(n_mo), 0.0));
    all_converged = true;

    // Per-(k,p) spectral Σ_c terms accumulated over q.
    std::vector<std::vector<std::vector<ScTerm>>> sc_terms(
        static_cast<std::size_t>(nk),
        std::vector<std::vector<ScTerm>>(static_cast<std::size_t>(n_mo)));

    std::vector<Cplx> rho(static_cast<std::size_t>(npts));

    // ── Unique q = wrap(ka − ki) ──────────────────────────────────────
    std::vector<std::array<double, 3>> q_list;
    auto q_index_of = [&](double qx, double qy, double qz) -> int {
        const double tx = wrap_frac(qx), ty = wrap_frac(qy), tz = wrap_frac(qz);
        for (std::size_t iq = 0; iq < q_list.size(); ++iq) {
            const double dx = wrap_frac(q_list[iq][0] - tx);
            const double dy = wrap_frac(q_list[iq][1] - ty);
            const double dz = wrap_frac(q_list[iq][2] - tz);
            if (std::abs(dx) < kQTol && std::abs(dy) < kQTol
                && std::abs(dz) < kQTol)
                return static_cast<int>(iq);
        }
        return -1;
    };
    for (int ik = 0; ik < nk; ++ik) {
        for (int ka = 0; ka < nk; ++ka) {
            const double qx = k_frac[static_cast<std::size_t>(ka)][0]
                              - k_frac[static_cast<std::size_t>(ik)][0];
            const double qy = k_frac[static_cast<std::size_t>(ka)][1]
                              - k_frac[static_cast<std::size_t>(ik)][1];
            const double qz = k_frac[static_cast<std::size_t>(ka)][2]
                              - k_frac[static_cast<std::size_t>(ik)][2];
            if (q_index_of(qx, qy, qz) < 0)
                q_list.push_back(
                    {{wrap_frac(qx), wrap_frac(qy), wrap_frac(qz)}});
        }
    }
    const int nq = static_cast<int>(q_list.size());
    if (nq < 1)
        throw std::runtime_error(
            "gpw spectral_g0w0_rhf_kpoints: empty q mesh from k differences");

    for (int iq = 0; iq < nq; ++iq) {
        const double q_frac[3] = {q_list[static_cast<std::size_t>(iq)][0],
                                  q_list[static_cast<std::size_t>(iq)][1],
                                  q_list[static_cast<std::size_t>(iq)][2]};

        // Sources ik with ka = k[ik] + q on the mesh (RPA dielectric block).
        std::vector<int> src_k;
        std::vector<int> tgt_k;
        src_k.reserve(static_cast<std::size_t>(nk));
        tgt_k.reserve(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const double fx =
                k_frac[static_cast<std::size_t>(ik)][0] + q_frac[0];
            const double fy =
                k_frac[static_cast<std::size_t>(ik)][1] + q_frac[1];
            const double fz =
                k_frac[static_cast<std::size_t>(ik)][2] + q_frac[2];
            const int ka = find_k_index(k_frac, fx, fy, fz);
            if (ka >= 0) {
                src_k.push_back(ik);
                tgt_k.push_back(ka);
            }
        }
        const int n_src = static_cast<int>(src_k.size());
        if (n_src < 1)
            continue;

        const int n_block = n_src * nov;
        std::vector<std::vector<Cplx>> Via(static_cast<std::size_t>(n_block));
        std::vector<double> de(static_cast<std::size_t>(n_block));
        std::vector<std::vector<Cplx>> rho_alpha(
            static_cast<std::size_t>(n_block));

        for (int is = 0; is < n_src; ++is) {
            const int ik = src_k[static_cast<std::size_t>(is)];
            const int ka = tgt_k[static_cast<std::size_t>(is)];
            const auto& psi_o = psi_mo_k[static_cast<std::size_t>(ik)];
            const auto& psi_v = psi_mo_k[static_cast<std::size_t>(ka)];
            const auto& eps_o = eps_k[static_cast<std::size_t>(ik)];
            const auto& eps_v = eps_k[static_cast<std::size_t>(ka)];

            for (int i = 0; i < n_occ; ++i) {
                for (int a = 0; a < n_virt; ++a) {
                    const int ia = i * n_virt + a;
                    const int alpha = is * nov + ia;
                    const Cplx* pi = psi_o.data()
                                     + static_cast<std::size_t>(i)
                                           * static_cast<std::size_t>(npts);
                    const Cplx* pa = psi_v.data()
                                     + static_cast<std::size_t>(n_occ + a)
                                           * static_cast<std::size_t>(npts);
                    rho_alpha[static_cast<std::size_t>(alpha)].resize(
                        static_cast<std::size_t>(npts));
                    for (long g = 0; g < npts; ++g) {
                        const Cplx r = std::conj(pi[g]) * pa[g];
                        rho[static_cast<std::size_t>(g)] = r;
                        rho_alpha[static_cast<std::size_t>(alpha)]
                                 [static_cast<std::size_t>(g)] = r;
                    }
                    // Bloch product already carries e^{iq·r}; plain Poisson.
                    poisson_complex(grid, rho.data(),
                                    Via[static_cast<std::size_t>(alpha)]);

                    de[static_cast<std::size_t>(alpha)] =
                        eps_v[static_cast<std::size_t>(n_occ + a)]
                        - eps_o[static_cast<std::size_t>(i)];
                    if (de[static_cast<std::size_t>(alpha)] <= kAmBFloor)
                        throw std::runtime_error(
                            "gpw spectral_g0w0_rhf_kpoints: non-positive "
                            "Δε at a (q,k,i,a) transition — dRPA (A−B) not "
                            "PD");
                }
            }
        }

        // Hermitian Coulomb form (ρ_p | ρ_q).
        CplxMatrix Vmat = xt::zeros<Cplx>({n_block, n_block});
        for (int p = 0; p < n_block; ++p) {
            const Cplx* Vp = Via[static_cast<std::size_t>(p)].data();
            for (int qq = 0; qq < n_block; ++qq) {
                const Cplx* rq = rho_alpha[static_cast<std::size_t>(qq)].data();
                Cplx s{0.0, 0.0};
                for (long g = 0; g < npts; ++g)
                    s += std::conj(Vp[g]) * rq[g];
                Vmat(p, qq) = w_grid * s;
            }
        }
        for (int p = 0; p < n_block; ++p) {
            Vmat(p, p) = Cplx{Vmat(p, p).real(), 0.0};
            for (int qq = p + 1; qq < n_block; ++qq) {
                const Cplx avg = 0.5 * (Vmat(p, qq) + std::conj(Vmat(qq, p)));
                Vmat(p, qq) = avg;
                Vmat(qq, p) = std::conj(avg);
            }
        }

        // M = (A−B)^{1/2}(A+B)(A−B)^{1/2}, A−B = diag(Δε), A+B = diag+4V.
        CplxMatrix M = xt::zeros<Cplx>({n_block, n_block});
        for (int p = 0; p < n_block; ++p) {
            const double dep = de[static_cast<std::size_t>(p)];
            const double sp = std::sqrt(dep);
            for (int qq = 0; qq < n_block; ++qq) {
                const double sq = std::sqrt(de[static_cast<std::size_t>(qq)]);
                Cplx apb = 4.0 * Vmat(p, qq);
                if (p == qq)
                    apb += dep;
                M(p, qq) = sp * apb * sq;
            }
        }
        for (int p = 0; p < n_block; ++p) {
            M(p, p) = Cplx{M(p, p).real(), 0.0};
            for (int qq = p + 1; qq < n_block; ++qq) {
                const Cplx avg = 0.5 * (M(p, qq) + std::conj(M(qq, p)));
                M(p, qq) = avg;
                M(qq, p) = std::conj(avg);
            }
        }

        std::vector<double> w2;
        CplxMatrix Z;
        linalg::eigh_hermitian(M, w2, Z);

        std::vector<double> Om(static_cast<std::size_t>(n_block));
        std::vector<Cplx> Xpy(static_cast<std::size_t>(n_block)
                              * static_cast<std::size_t>(n_block));
        for (int s = 0; s < n_block; ++s) {
            double lam = w2[static_cast<std::size_t>(s)];
            if (lam < -kOmega2Tol)
                throw std::runtime_error(
                    "gpw spectral_g0w0_rhf_kpoints: dRPA omega^2 = "
                    + std::to_string(lam) + " Ha^2 (q=" + std::to_string(iq)
                    + " root " + std::to_string(s)
                    + ") is negative — imaginary excitation (no fallback)");
            if (lam <= kOmega2Tol)
                throw std::runtime_error(
                    "gpw spectral_g0w0_rhf_kpoints: dRPA omega^2 = "
                    + std::to_string(lam) + " Ha^2 (q=" + std::to_string(iq)
                    + " root " + std::to_string(s)
                    + ") is numerically zero (no fallback)");
            const double om = std::sqrt(lam);
            Om[static_cast<std::size_t>(s)] = om;
            const double inv_sqrt_om = 1.0 / std::sqrt(om);
            for (int p = 0; p < n_block; ++p) {
                const double sp = std::sqrt(de[static_cast<std::size_t>(p)]);
                Xpy[static_cast<std::size_t>(s)
                        * static_cast<std::size_t>(n_block)
                    + static_cast<std::size_t>(p)] = inv_sqrt_om * sp * Z(p, s);
            }
        }

        // W_s(r) and matrix elements for every target kp with km = kp − q.
        std::vector<Cplx> Ws(static_cast<std::size_t>(npts));
        for (int s = 0; s < n_block; ++s) {
            std::fill(Ws.begin(), Ws.end(), Cplx{0.0, 0.0});
            for (int alpha = 0; alpha < n_block; ++alpha) {
                const Cplx z = sqrt2
                               * Xpy[static_cast<std::size_t>(s)
                                         * static_cast<std::size_t>(n_block)
                                     + static_cast<std::size_t>(alpha)];
                if (z.real() == 0.0 && z.imag() == 0.0)
                    continue;
                const Cplx* Via_g = Via[static_cast<std::size_t>(alpha)].data();
                for (long g = 0; g < npts; ++g)
                    Ws[static_cast<std::size_t>(g)] += z * Via_g[g];
            }

            const double Os = Om[static_cast<std::size_t>(s)];
            for (int kp = 0; kp < nk; ++kp) {
                // Intermediate crystal momentum km = kp − q.
                const double fmx =
                    k_frac[static_cast<std::size_t>(kp)][0] - q_frac[0];
                const double fmy =
                    k_frac[static_cast<std::size_t>(kp)][1] - q_frac[1];
                const double fmz =
                    k_frac[static_cast<std::size_t>(kp)][2] - q_frac[2];
                const int km = find_k_index(k_frac, fmx, fmy, fmz);
                if (km < 0)
                    continue;

                const auto& psi_p = psi_mo_k[static_cast<std::size_t>(kp)];
                const auto& psi_m = psi_mo_k[static_cast<std::size_t>(km)];
                const auto& eps_m = eps_k[static_cast<std::size_t>(km)];

                for (int m = 0; m < n_mo; ++m) {
                    const Cplx* pm = psi_m.data()
                                     + static_cast<std::size_t>(m)
                                           * static_cast<std::size_t>(npts);
                    for (int p = 0; p < n_mo; ++p) {
                        const Cplx* pp = psi_p.data()
                                         + static_cast<std::size_t>(p)
                                               * static_cast<std::size_t>(npts);
                        Cplx acc{0.0, 0.0};
                        for (long g = 0; g < npts; ++g)
                            acc += Ws[static_cast<std::size_t>(g)]
                                   * (std::conj(pm[g]) * pp[g]);
                        acc *= w_grid;
                        const double t2 = w_q_se * std::norm(acc);
                        if (t2 == 0.0)
                            continue;
                        ScTerm term;
                        term.Omega = Os;
                        term.eps_m = eps_m[static_cast<std::size_t>(m)];
                        term.t2 = t2;
                        term.m_occ = (m < n_occ);
                        sc_terms[static_cast<std::size_t>(kp)]
                                [static_cast<std::size_t>(p)]
                                    .push_back(term);
                    }
                }
            }
        }
    }

    // ── Per-k: MT Σ_x, v_xc diagonals, Newton QP ─────────────────────
    for (int ik = 0; ik < nk; ++ik) {
        const auto& psi = psi_mo_k[static_cast<std::size_t>(ik)];
        const auto& eps = eps_k[static_cast<std::size_t>(ik)];

        auto& sx = sigma_x_k[static_cast<std::size_t>(ik)];
        std::vector<Cplx> Vmt;
        for (int p = 0; p < n_mo; ++p) {
            const Cplx* pp =
                psi.data()
                + static_cast<std::size_t>(p) * static_cast<std::size_t>(npts);
            double sum = 0.0;
            for (int i = 0; i < n_occ; ++i) {
                const Cplx* pi = psi.data()
                                 + static_cast<std::size_t>(i)
                                       * static_cast<std::size_t>(npts);
                for (long g = 0; g < npts; ++g)
                    rho[static_cast<std::size_t>(g)] = std::conj(pp[g]) * pi[g];
                poisson_mt_complex(grid, rho.data(), Vmt);
                Cplx s{0.0, 0.0};
                for (long g = 0; g < npts; ++g)
                    s += Vmt[static_cast<std::size_t>(g)]
                         * rho[static_cast<std::size_t>(g)];
                sum -= w_grid * s.real();
            }
            sx[static_cast<std::size_t>(p)] = sum;
        }

        auto& vxc_d = vxc_diag_k[static_cast<std::size_t>(ik)];
        for (int p = 0; p < n_mo; ++p) {
            const Cplx* pp =
                psi.data()
                + static_cast<std::size_t>(p) * static_cast<std::size_t>(npts);
            double acc = 0.0;
            for (long g = 0; g < npts; ++g)
                acc += std::norm(pp[g]) * vxc_r[static_cast<std::size_t>(g)];
            vxc_d[static_cast<std::size_t>(p)] = w_grid * acc;
        }

        auto& qp = qp_energy_k[static_cast<std::size_t>(ik)];
        for (int p = 0; p < n_mo; ++p) {
            const double vx_m_vxc = sx[static_cast<std::size_t>(p)]
                                    - vxc_d[static_cast<std::size_t>(p)];
            double w = eps[static_cast<std::size_t>(p)];
            const bool ok =
                solve_qp_terms(eps[static_cast<std::size_t>(p)], vx_m_vxc,
                               sc_terms[static_cast<std::size_t>(ik)]
                                       [static_cast<std::size_t>(p)],
                               w);
            if (!ok)
                all_converged = false;
            qp[static_cast<std::size_t>(p)] = w;
        }
    }
}

void spectral_g0w0_uhf_kpoints(const UniformGrid& grid,
                               const std::vector<std::vector<Cplx>>& psi_mo_a_k,
                               const std::vector<std::vector<Cplx>>& psi_mo_b_k,
                               const std::vector<std::vector<double>>& eps_a_k,
                               const std::vector<std::vector<double>>& eps_b_k,
                               const std::vector<std::array<double, 3>>& k_frac,
                               const std::vector<double>& k_weights, int n_oa,
                               int n_va, int n_ob, int n_vb,
                               const std::vector<double>& vxc_a_r,
                               std::vector<std::vector<double>>& qp_energy_k,
                               std::vector<std::vector<double>>& sigma_x_k,
                               std::vector<std::vector<double>>& vxc_diag_k,
                               bool& all_converged) {
    using linalg::CplxMatrix;

    const int nk = static_cast<int>(psi_mo_a_k.size());
    if (nk < 1)
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: empty k mesh");
    if (static_cast<int>(psi_mo_b_k.size()) != nk
        || static_cast<int>(eps_a_k.size()) != nk
        || static_cast<int>(eps_b_k.size()) != nk
        || static_cast<int>(k_weights.size()) != nk
        || static_cast<int>(k_frac.size()) != nk)
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: per-k array length mismatch");
    if (n_oa < 1 || n_va < 1)
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: need n_oa>=1 and n_va>=1 "
            "(α primary report)");
    if (n_ob < 0 || n_vb < 0)
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: negative β orbital counts");
    if ((n_ob == 0) != (n_vb == 0))
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: β channel must be empty on both "
            "occ and virt or neither");

    const long npts = grid.npts();
    const double w_grid = grid.weight();
    if (static_cast<int>(vxc_a_r.size()) != static_cast<int>(npts))
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: vxc_a_r size != npts");

    const int n_mo_a = static_cast<int>(eps_a_k[0].size());
    const int n_mo_b =
        (n_ob > 0 || n_vb > 0) ? static_cast<int>(eps_b_k[0].size()) : 0;
    if (n_mo_a < n_oa + n_va)
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: n_mo_a < n_oa + n_va");
    if (n_ob > 0 && n_mo_b < n_ob + n_vb)
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: n_mo_b < n_ob + n_vb");

    const int nov_a = n_oa * n_va;
    const int nov_b = n_ob * n_vb;
    const int nov = nov_a + nov_b;
    if (nov < 1)
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: empty composite OV space");

    const std::size_t need_a =
        static_cast<std::size_t>(n_mo_a) * static_cast<std::size_t>(npts);
    const std::size_t need_b = static_cast<std::size_t>(std::max(n_mo_b, 0))
                               * static_cast<std::size_t>(npts);

    for (int ik = 0; ik < nk; ++ik) {
        if (static_cast<int>(eps_a_k[static_cast<std::size_t>(ik)].size())
            != n_mo_a)
            throw std::invalid_argument(
                "gpw spectral_g0w0_uhf_kpoints: inconsistent n_mo_a across k");
        if (psi_mo_a_k[static_cast<std::size_t>(ik)].size() != need_a)
            throw std::invalid_argument(
                "gpw spectral_g0w0_uhf_kpoints: α MO grid size mismatch");
        if (n_ob > 0) {
            if (static_cast<int>(eps_b_k[static_cast<std::size_t>(ik)].size())
                != n_mo_b)
                throw std::invalid_argument(
                    "gpw spectral_g0w0_uhf_kpoints: inconsistent n_mo_b "
                    "across k");
            if (psi_mo_b_k[static_cast<std::size_t>(ik)].size() != need_b)
                throw std::invalid_argument(
                    "gpw spectral_g0w0_uhf_kpoints: β MO grid size mismatch");
        }
    }
    double wsum = 0.0;
    for (double wk : k_weights)
        wsum += wk;
    if (std::abs(wsum - 1.0) > 1e-8)
        throw std::invalid_argument(
            "gpw spectral_g0w0_uhf_kpoints: k_weights must sum to 1");

    constexpr double kAmBFloor = 1e-10;
    constexpr double kOmega2Tol = 1e-10;
    constexpr double kQTol = 1e-10;
    // Equal-mesh spectral weight: each q contributes 1/N_k.
    const double w_q_se = 1.0 / static_cast<double>(nk);

    qp_energy_k.assign(
        static_cast<std::size_t>(nk),
        std::vector<double>(static_cast<std::size_t>(n_mo_a), 0.0));
    sigma_x_k.assign(
        static_cast<std::size_t>(nk),
        std::vector<double>(static_cast<std::size_t>(n_mo_a), 0.0));
    vxc_diag_k.assign(
        static_cast<std::size_t>(nk),
        std::vector<double>(static_cast<std::size_t>(n_mo_a), 0.0));
    all_converged = true;

    // Per-(k,p_α) spectral Σ_c terms accumulated over q (same-spin G_α W).
    std::vector<std::vector<std::vector<ScTerm>>> sc_terms(
        static_cast<std::size_t>(nk),
        std::vector<std::vector<ScTerm>>(static_cast<std::size_t>(n_mo_a)));

    std::vector<Cplx> rho(static_cast<std::size_t>(npts));

    // ── Unique q = wrap(ka − ki) ──────────────────────────────────────
    std::vector<std::array<double, 3>> q_list;
    auto q_index_of = [&](double qx, double qy, double qz) -> int {
        const double tx = wrap_frac(qx), ty = wrap_frac(qy), tz = wrap_frac(qz);
        for (std::size_t iq = 0; iq < q_list.size(); ++iq) {
            const double dx = wrap_frac(q_list[iq][0] - tx);
            const double dy = wrap_frac(q_list[iq][1] - ty);
            const double dz = wrap_frac(q_list[iq][2] - tz);
            if (std::abs(dx) < kQTol && std::abs(dy) < kQTol
                && std::abs(dz) < kQTol)
                return static_cast<int>(iq);
        }
        return -1;
    };
    for (int ik = 0; ik < nk; ++ik) {
        for (int ka = 0; ka < nk; ++ka) {
            const double qx = k_frac[static_cast<std::size_t>(ka)][0]
                              - k_frac[static_cast<std::size_t>(ik)][0];
            const double qy = k_frac[static_cast<std::size_t>(ka)][1]
                              - k_frac[static_cast<std::size_t>(ik)][1];
            const double qz = k_frac[static_cast<std::size_t>(ka)][2]
                              - k_frac[static_cast<std::size_t>(ik)][2];
            if (q_index_of(qx, qy, qz) < 0)
                q_list.push_back(
                    {{wrap_frac(qx), wrap_frac(qy), wrap_frac(qz)}});
        }
    }
    const int nq = static_cast<int>(q_list.size());
    if (nq < 1)
        throw std::runtime_error(
            "gpw spectral_g0w0_uhf_kpoints: empty q mesh from k differences");

    for (int iq = 0; iq < nq; ++iq) {
        const double q_frac[3] = {q_list[static_cast<std::size_t>(iq)][0],
                                  q_list[static_cast<std::size_t>(iq)][1],
                                  q_list[static_cast<std::size_t>(iq)][2]};

        std::vector<int> src_k;
        std::vector<int> tgt_k;
        src_k.reserve(static_cast<std::size_t>(nk));
        tgt_k.reserve(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const double fx =
                k_frac[static_cast<std::size_t>(ik)][0] + q_frac[0];
            const double fy =
                k_frac[static_cast<std::size_t>(ik)][1] + q_frac[1];
            const double fz =
                k_frac[static_cast<std::size_t>(ik)][2] + q_frac[2];
            const int ka = find_k_index(k_frac, fx, fy, fz);
            if (ka >= 0) {
                src_k.push_back(ik);
                tgt_k.push_back(ka);
            }
        }
        const int n_src = static_cast<int>(src_k.size());
        if (n_src < 1)
            continue;

        // Composite OV per source k: α then β (matches direct_rpa_uhf layout).
        const int n_block = n_src * nov;
        std::vector<std::vector<Cplx>> Via(static_cast<std::size_t>(n_block));
        std::vector<double> de(static_cast<std::size_t>(n_block));
        std::vector<std::vector<Cplx>> rho_alpha(
            static_cast<std::size_t>(n_block));

        auto fill_spin = [&](int is, int nov_off, int n_o, int n_v,
                             const std::vector<std::vector<Cplx>>& psi_mo_k,
                             const std::vector<std::vector<double>>& eps_k) {
            const int ik = src_k[static_cast<std::size_t>(is)];
            const int ka = tgt_k[static_cast<std::size_t>(is)];
            const auto& psi_o = psi_mo_k[static_cast<std::size_t>(ik)];
            const auto& psi_v = psi_mo_k[static_cast<std::size_t>(ka)];
            const auto& eps_o = eps_k[static_cast<std::size_t>(ik)];
            const auto& eps_v = eps_k[static_cast<std::size_t>(ka)];
            for (int i = 0; i < n_o; ++i) {
                for (int a = 0; a < n_v; ++a) {
                    const int ia = i * n_v + a;
                    const int alpha = is * nov + nov_off + ia;
                    const Cplx* pi = psi_o.data()
                                     + static_cast<std::size_t>(i)
                                           * static_cast<std::size_t>(npts);
                    const Cplx* pa = psi_v.data()
                                     + static_cast<std::size_t>(n_o + a)
                                           * static_cast<std::size_t>(npts);
                    rho_alpha[static_cast<std::size_t>(alpha)].resize(
                        static_cast<std::size_t>(npts));
                    for (long g = 0; g < npts; ++g) {
                        const Cplx r = std::conj(pi[g]) * pa[g];
                        rho[static_cast<std::size_t>(g)] = r;
                        rho_alpha[static_cast<std::size_t>(alpha)]
                                 [static_cast<std::size_t>(g)] = r;
                    }
                    poisson_complex(grid, rho.data(),
                                    Via[static_cast<std::size_t>(alpha)]);
                    de[static_cast<std::size_t>(alpha)] =
                        eps_v[static_cast<std::size_t>(n_o + a)]
                        - eps_o[static_cast<std::size_t>(i)];
                    if (de[static_cast<std::size_t>(alpha)] <= kAmBFloor)
                        throw std::runtime_error(
                            "gpw spectral_g0w0_uhf_kpoints: non-positive "
                            "Δε at a (q,k,i,a,σ) transition — dRPA (A−B) not "
                            "PD");
                }
            }
        };

        for (int is = 0; is < n_src; ++is) {
            fill_spin(is, 0, n_oa, n_va, psi_mo_a_k, eps_a_k);
            if (nov_b > 0)
                fill_spin(is, nov_a, n_ob, n_vb, psi_mo_b_k, eps_b_k);
        }

        // Hermitian Coulomb form (ρ_p | ρ_q) — same- and cross-spin.
        CplxMatrix Vmat = xt::zeros<Cplx>({n_block, n_block});
        for (int p = 0; p < n_block; ++p) {
            const Cplx* Vp = Via[static_cast<std::size_t>(p)].data();
            for (int qq = 0; qq < n_block; ++qq) {
                const Cplx* rq = rho_alpha[static_cast<std::size_t>(qq)].data();
                Cplx s{0.0, 0.0};
                for (long g = 0; g < npts; ++g)
                    s += std::conj(Vp[g]) * rq[g];
                Vmat(p, qq) = w_grid * s;
            }
        }
        for (int p = 0; p < n_block; ++p) {
            Vmat(p, p) = Cplx{Vmat(p, p).real(), 0.0};
            for (int qq = p + 1; qq < n_block; ++qq) {
                const Cplx avg = 0.5 * (Vmat(p, qq) + std::conj(Vmat(qq, p)));
                Vmat(p, qq) = avg;
                Vmat(qq, p) = std::conj(avg);
            }
        }

        // UHF Casida: A−B = diag(Δε), A+B = diag(Δε) + 2V (spins explicit).
        CplxMatrix M = xt::zeros<Cplx>({n_block, n_block});
        for (int p = 0; p < n_block; ++p) {
            const double dep = de[static_cast<std::size_t>(p)];
            const double sp = std::sqrt(dep);
            for (int qq = 0; qq < n_block; ++qq) {
                const double sq = std::sqrt(de[static_cast<std::size_t>(qq)]);
                Cplx apb = 2.0 * Vmat(p, qq);
                if (p == qq)
                    apb += dep;
                M(p, qq) = sp * apb * sq;
            }
        }
        for (int p = 0; p < n_block; ++p) {
            M(p, p) = Cplx{M(p, p).real(), 0.0};
            for (int qq = p + 1; qq < n_block; ++qq) {
                const Cplx avg = 0.5 * (M(p, qq) + std::conj(M(qq, p)));
                M(p, qq) = avg;
                M(qq, p) = std::conj(avg);
            }
        }

        std::vector<double> w2;
        CplxMatrix Z;
        linalg::eigh_hermitian(M, w2, Z);

        std::vector<double> Om(static_cast<std::size_t>(n_block));
        std::vector<Cplx> Xpy(static_cast<std::size_t>(n_block)
                              * static_cast<std::size_t>(n_block));
        for (int s = 0; s < n_block; ++s) {
            double lam = w2[static_cast<std::size_t>(s)];
            if (lam < -kOmega2Tol)
                throw std::runtime_error(
                    "gpw spectral_g0w0_uhf_kpoints: dRPA omega^2 = "
                    + std::to_string(lam) + " Ha^2 (q=" + std::to_string(iq)
                    + " root " + std::to_string(s)
                    + ") is negative — imaginary excitation (no fallback)");
            if (lam <= kOmega2Tol)
                throw std::runtime_error(
                    "gpw spectral_g0w0_uhf_kpoints: dRPA omega^2 = "
                    + std::to_string(lam) + " Ha^2 (q=" + std::to_string(iq)
                    + " root " + std::to_string(s)
                    + ") is numerically zero (no fallback)");
            const double om = std::sqrt(lam);
            Om[static_cast<std::size_t>(s)] = om;
            const double inv_sqrt_om = 1.0 / std::sqrt(om);
            for (int p = 0; p < n_block; ++p) {
                const double sp = std::sqrt(de[static_cast<std::size_t>(p)]);
                Xpy[static_cast<std::size_t>(s)
                        * static_cast<std::size_t>(n_block)
                    + static_cast<std::size_t>(p)] = inv_sqrt_om * sp * Z(p, s);
            }
        }

        // W_s(r) = Σ_iaσ (X+Y) V_iaσ  (no √2 — spins explicit, Γ UGW form).
        std::vector<Cplx> Ws(static_cast<std::size_t>(npts));
        for (int s = 0; s < n_block; ++s) {
            std::fill(Ws.begin(), Ws.end(), Cplx{0.0, 0.0});
            for (int alpha = 0; alpha < n_block; ++alpha) {
                const Cplx z = Xpy[static_cast<std::size_t>(s)
                                       * static_cast<std::size_t>(n_block)
                                   + static_cast<std::size_t>(alpha)];
                if (z.real() == 0.0 && z.imag() == 0.0)
                    continue;
                const Cplx* Via_g = Via[static_cast<std::size_t>(alpha)].data();
                for (long g = 0; g < npts; ++g)
                    Ws[static_cast<std::size_t>(g)] += z * Via_g[g];
            }

            const double Os = Om[static_cast<std::size_t>(s)];
            for (int kp = 0; kp < nk; ++kp) {
                const double fmx =
                    k_frac[static_cast<std::size_t>(kp)][0] - q_frac[0];
                const double fmy =
                    k_frac[static_cast<std::size_t>(kp)][1] - q_frac[1];
                const double fmz =
                    k_frac[static_cast<std::size_t>(kp)][2] - q_frac[2];
                const int km = find_k_index(k_frac, fmx, fmy, fmz);
                if (km < 0)
                    continue;

                // Same-spin G_α: intermediates and external legs are α.
                const auto& psi_p = psi_mo_a_k[static_cast<std::size_t>(kp)];
                const auto& psi_m = psi_mo_a_k[static_cast<std::size_t>(km)];
                const auto& eps_m = eps_a_k[static_cast<std::size_t>(km)];

                for (int m = 0; m < n_mo_a; ++m) {
                    const Cplx* pm = psi_m.data()
                                     + static_cast<std::size_t>(m)
                                           * static_cast<std::size_t>(npts);
                    for (int p = 0; p < n_mo_a; ++p) {
                        const Cplx* pp = psi_p.data()
                                         + static_cast<std::size_t>(p)
                                               * static_cast<std::size_t>(npts);
                        Cplx acc{0.0, 0.0};
                        for (long g = 0; g < npts; ++g)
                            acc += Ws[static_cast<std::size_t>(g)]
                                   * (std::conj(pm[g]) * pp[g]);
                        acc *= w_grid;
                        const double t2 = w_q_se * std::norm(acc);
                        if (t2 == 0.0)
                            continue;
                        ScTerm term;
                        term.Omega = Os;
                        term.eps_m = eps_m[static_cast<std::size_t>(m)];
                        term.t2 = t2;
                        term.m_occ = (m < n_oa);
                        sc_terms[static_cast<std::size_t>(kp)]
                                [static_cast<std::size_t>(p)]
                                    .push_back(term);
                    }
                }
            }
        }
    }

    // ── Per-k α: MT Σ_x, v_xc^α diagonals, Newton QP ──────────────────
    for (int ik = 0; ik < nk; ++ik) {
        const auto& psi = psi_mo_a_k[static_cast<std::size_t>(ik)];
        const auto& eps = eps_a_k[static_cast<std::size_t>(ik)];

        auto& sx = sigma_x_k[static_cast<std::size_t>(ik)];
        std::vector<Cplx> Vmt;
        for (int p = 0; p < n_mo_a; ++p) {
            const Cplx* pp =
                psi.data()
                + static_cast<std::size_t>(p) * static_cast<std::size_t>(npts);
            double sum = 0.0;
            for (int i = 0; i < n_oa; ++i) {
                const Cplx* pi = psi.data()
                                 + static_cast<std::size_t>(i)
                                       * static_cast<std::size_t>(npts);
                for (long g = 0; g < npts; ++g)
                    rho[static_cast<std::size_t>(g)] = std::conj(pp[g]) * pi[g];
                poisson_mt_complex(grid, rho.data(), Vmt);
                Cplx s{0.0, 0.0};
                for (long g = 0; g < npts; ++g)
                    s += Vmt[static_cast<std::size_t>(g)]
                         * rho[static_cast<std::size_t>(g)];
                sum -= w_grid * s.real();
            }
            sx[static_cast<std::size_t>(p)] = sum;
        }

        auto& vxc_d = vxc_diag_k[static_cast<std::size_t>(ik)];
        for (int p = 0; p < n_mo_a; ++p) {
            const Cplx* pp =
                psi.data()
                + static_cast<std::size_t>(p) * static_cast<std::size_t>(npts);
            double acc = 0.0;
            for (long g = 0; g < npts; ++g)
                acc += std::norm(pp[g]) * vxc_a_r[static_cast<std::size_t>(g)];
            vxc_d[static_cast<std::size_t>(p)] = w_grid * acc;
        }

        auto& qp = qp_energy_k[static_cast<std::size_t>(ik)];
        for (int p = 0; p < n_mo_a; ++p) {
            const double vx_m_vxc = sx[static_cast<std::size_t>(p)]
                                    - vxc_d[static_cast<std::size_t>(p)];
            double w = eps[static_cast<std::size_t>(p)];
            const bool ok =
                solve_qp_terms(eps[static_cast<std::size_t>(p)], vx_m_vxc,
                               sc_terms[static_cast<std::size_t>(ik)]
                                       [static_cast<std::size_t>(p)],
                               w);
            if (!ok)
                all_converged = false;
            qp[static_cast<std::size_t>(p)] = w;
        }
    }
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
