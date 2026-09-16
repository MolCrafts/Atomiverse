/// @file src/cpu/multiscale/qmmm_polarizable.cpp
#include "atomiverse/cpu/multiscale/qmmm_polarizable.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace {

inline double thole_u(double r, double ai, double aj, double a) {
    if (!(a > 0.0) || !(ai > 0.0) || !(aj > 0.0))
        return 0.0;
    const double aij = std::pow(ai * aj, 1.0 / 6.0);
    return a * r / aij;
}

void accumulate_T_mu(int n, const double* pos, const double* alpha,
                     const double* mu, double thole_a, double* e_out) {
    for (int i = 0; i < n; ++i) {
        const double* ri = pos + 3 * i;
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            const double* rj = pos + 3 * j;
            const double* muj = mu + 3 * j;
            double dx = ri[0] - rj[0], dy = ri[1] - rj[1], dz = ri[2] - rj[2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0))
                continue;
            const double r = std::sqrt(r2);
            const double r3 = r2 * r;
            const double r5 = r3 * r2;
            double f3 = 1.0 / r3;
            double f5 = 1.0 / r5;
            if (thole_a > 0.0) {
                const double u = thole_u(r, alpha[i], alpha[j], thole_a);
                const double u3 = u * u * u;
                const double expu = std::exp(-u3);
                f3 *= (1.0 - expu);
                f5 *= (1.0 - (1.0 + u3) * expu);
            }
            const double mdotr = muj[0] * dx + muj[1] * dy + muj[2] * dz;
            e_out[3 * i + 0] += 3.0 * f5 * mdotr * dx - f3 * muj[0];
            e_out[3 * i + 1] += 3.0 * f5 * mdotr * dy - f3 * muj[1];
            e_out[3 * i + 2] += 3.0 * f5 * mdotr * dz - f3 * muj[2];
        }
    }
}

} // namespace

double solve_induced_dipoles(int n, const double* pos, const double* alpha,
                             const double* e_perm, double thole_a, double tol,
                             int max_iter, std::vector<double>& mu,
                             int* iters_used) {
    if (n < 0)
        throw std::invalid_argument("solve_induced_dipoles: n < 0");
    if (n == 0) {
        mu.clear();
        if (iters_used)
            *iters_used = 0;
        return 0.0;
    }
    if (!pos || !alpha || !e_perm)
        throw std::invalid_argument("solve_induced_dipoles: null pointer");
    if (!(tol > 0.0) || max_iter < 1)
        throw std::invalid_argument("solve_induced_dipoles: bad tol/max_iter");

    mu.assign(static_cast<std::size_t>(3 * n), 0.0);
    std::vector<double> e_tot(static_cast<std::size_t>(3 * n), 0.0);
    std::vector<double> mu_new(static_cast<std::size_t>(3 * n), 0.0);

    int it = 0;
    bool ok = false;
    for (; it < max_iter; ++it) {
        std::fill(e_tot.begin(), e_tot.end(), 0.0);
        for (int k = 0; k < 3 * n; ++k)
            e_tot[static_cast<std::size_t>(k)] =
                e_perm[static_cast<std::size_t>(k)];
        accumulate_T_mu(n, pos, alpha, mu.data(), thole_a, e_tot.data());

        double dmax = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < 3; ++c) {
                const double m =
                    alpha[i] * e_tot[static_cast<std::size_t>(3 * i + c)];
                mu_new[static_cast<std::size_t>(3 * i + c)] = m;
                dmax = std::max(
                    dmax,
                    std::fabs(m - mu[static_cast<std::size_t>(3 * i + c)]));
            }
        }
        mu.swap(mu_new);
        if (dmax < tol) {
            ok = true;
            ++it;
            break;
        }
    }
    if (iters_used)
        *iters_used = it;
    if (!ok)
        throw std::runtime_error(
            "solve_induced_dipoles: SCC did not converge within max_iter");

    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        e -= 0.5
             * (mu[static_cast<std::size_t>(3 * i + 0)]
                    * e_perm[static_cast<std::size_t>(3 * i + 0)]
                + mu[static_cast<std::size_t>(3 * i + 1)]
                      * e_perm[static_cast<std::size_t>(3 * i + 1)]
                + mu[static_cast<std::size_t>(3 * i + 2)]
                      * e_perm[static_cast<std::size_t>(3 * i + 2)]);
    }
    return units::coulomb * e;
}

QmmmPolarizable::QmmmPolarizable(std::vector<LJTypeCoeff> cross_lj,
                                 double lj_cutoff, double embed_cutoff)
    : vdw_(std::move(cross_lj), lj_cutoff) {
    sub_.embed_cutoff() = embed_cutoff;
}

void QmmmPolarizable::build(const BuildContext& ctx) {
    if (!(alpha >= 0.0))
        throw std::invalid_argument("QmmmPolarizable: alpha must be >= 0");
    if (!(thole_a >= 0.0))
        throw std::invalid_argument("QmmmPolarizable: thole_a must be >= 0");
    if (!(pol_tol > 0.0) || pol_max_iter < 1)
        throw std::invalid_argument("QmmmPolarizable: invalid pol SCC params");
    sub_.build(ctx.frame, ctx.runtime.mpi_size, /*require_embedding=*/true);
}

void QmmmPolarizable::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    // Embedding QM evaluate (writes Mulliken charges onto private frame).
    sub_.launch(f, ctx.forces_requested, e_qm_);

    const auto& qm_rows = sub_.qm_rows();
    const Frame* fqm = sub_.private_frame();
    if (!fqm)
        throw std::runtime_error("QmmmPolarizable: missing private QM frame");

    // Density-derived permanent field: engine-written kAtom_charge on private
    // QM frame.
    const auto q_qm = fqm->span(kAtom_charge);
    double qmax = 0.0;
    for (std::size_t k = 0; k < qm_rows.size(); ++k)
        qmax = std::max(qmax, std::abs(static_cast<double>(q_qm[k])));
    if (qmax < 1e-14)
        throw std::runtime_error(
            "QmmmPolarizable: engine left private kAtom_charge all-zero — "
            "permanent field requires density-derived charges");

    const int n = static_cast<int>(f.nlocal());
    std::unordered_set<int> qm_set(qm_rows.begin(), qm_rows.end());
    std::vector<int> mm_rows;
    for (int i = 0; i < n; ++i)
        if (!qm_set.count(i))
            mm_rows.push_back(i);

    const StorageReal* x = f.span(kAtom_x);
    const StorageReal* y = f.span(kAtom_y);
    const StorageReal* z = f.span(kAtom_z);
    const auto q_user = f.span(kAtom_charge);
    const StorageReal* qx = fqm->span(kAtom_x);
    const StorageReal* qy = fqm->span(kAtom_y);
    const StorageReal* qz = fqm->span(kAtom_z);

    const int n_mm = static_cast<int>(mm_rows.size());
    std::vector<double> pos(static_cast<std::size_t>(3 * n_mm), 0.0);
    std::vector<double> alpha_v(static_cast<std::size_t>(n_mm), alpha);
    std::vector<double> e_perm(static_cast<std::size_t>(3 * n_mm), 0.0);
    for (int a = 0; a < n_mm; ++a) {
        const int i = mm_rows[static_cast<std::size_t>(a)];
        pos[static_cast<std::size_t>(3 * a + 0)] = x[i];
        pos[static_cast<std::size_t>(3 * a + 1)] = y[i];
        pos[static_cast<std::size_t>(3 * a + 2)] = z[i];
    }
    for (int a = 0; a < n_mm; ++a) {
        double ex = 0, ey = 0, ez = 0;
        const double xi = pos[static_cast<std::size_t>(3 * a + 0)];
        const double yi = pos[static_cast<std::size_t>(3 * a + 1)];
        const double zi = pos[static_cast<std::size_t>(3 * a + 2)];
        // Field from QM density charges (private frame).
        for (std::size_t k = 0; k < qm_rows.size(); ++k) {
            const double q = static_cast<double>(q_qm[k]);
            double dx = xi - qx[k], dy = yi - qy[k], dz = zi - qz[k];
            f.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0))
                continue;
            const double r3 = r2 * std::sqrt(r2);
            const double fac = q / r3;
            ex += fac * dx;
            ey += fac * dy;
            ez += fac * dz;
        }
        // Field from other permanent MM charges
        for (int b = 0; b < n_mm; ++b) {
            if (a == b)
                continue;
            const int jb = mm_rows[static_cast<std::size_t>(b)];
            double dx = xi - pos[static_cast<std::size_t>(3 * b + 0)];
            double dy = yi - pos[static_cast<std::size_t>(3 * b + 1)];
            double dz = zi - pos[static_cast<std::size_t>(3 * b + 2)];
            f.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0))
                continue;
            const double r3 = r2 * std::sqrt(r2);
            const double fac = static_cast<double>(q_user[jb]) / r3;
            ex += fac * dx;
            ey += fac * dy;
            ez += fac * dz;
        }
        e_perm[static_cast<std::size_t>(3 * a + 0)] = ex;
        e_perm[static_cast<std::size_t>(3 * a + 1)] = ey;
        e_perm[static_cast<std::size_t>(3 * a + 2)] = ez;
    }

    e_pol_ = 0.0;
    last_pol_iters_ = 0;
    if (alpha > 0.0 && n_mm > 0) {
        std::vector<double> mu;
        e_pol_ = solve_induced_dipoles(n_mm, pos.data(), alpha_v.data(),
                                       e_perm.data(), thole_a, pol_tol,
                                       pol_max_iter, mu, &last_pol_iters_);
        f.meta.get(kEnergy) += e_pol_;
    }

    e_vdw_ = vdw_.launch(f, qm_rows, ctx.forces_requested);
    f.meta.get(kEnergy) += e_vdw_;
}

} // namespace cpu
} // namespace ATV
