/// @file cpu/dynamics/fix_lincs.cpp
/// @brief cpu::md::DistanceConstraints — the LINCS position and velocity
///        stages (Hess, Bekker, Berendsen, Fraaije, J. Comput. Chem. 18,
///        1463 (1997)), written from the paper.
///
/// The constraint table, the clusters, the reference snapshot, the halo
/// packer and the redundant per-rank cluster solve are the shared
/// DistanceConstraints machinery in fix_rattle.cpp; only the per-cluster
/// linear algebra differs.

#include "atomiverse/cpu/dynamics/distance_constraints.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

void DistanceConstraints::build_lincs_coupling_() {
    const std::size_t n = local_i_.size();
    lincs_s_.resize(n);
    for (std::size_t k = 0; k < n; ++k)
        lincs_s_[k] = 1.0 / std::sqrt(local_imi_[k] + local_imj_[k]);

    // (row, constraint) incidences grouped by row: two constraints couple
    // exactly when they share a row. Clusters share no row, so one global
    // sort keeps every coupling inside its cluster.
    std::vector<std::pair<int, int>> incidence;
    incidence.reserve(2 * n);
    for (std::size_t k = 0; k < n; ++k) {
        incidence.emplace_back(local_i_[k], static_cast<int>(k));
        incidence.emplace_back(local_j_[k], static_cast<int>(k));
    }
    std::sort(incidence.begin(), incidence.end());

    // A_kn = -S_k S_n c_k c_n / m_shared * (B_k.B_n), c = +1 on the i end of
    // a constraint and -1 on its j end.
    std::vector<std::tuple<int, int, double>> entries;
    for (std::size_t a = 0; a < incidence.size();) {
        std::size_t b = a;
        while (b < incidence.size() && incidence[b].first == incidence[a].first)
            ++b;
        const int row = incidence[a].first;
        for (std::size_t p = a; p < b; ++p)
            for (std::size_t q = a; q < b; ++q) {
                const std::size_t k =
                    static_cast<std::size_t>(incidence[p].second);
                const std::size_t m =
                    static_cast<std::size_t>(incidence[q].second);
                if (k == m)
                    continue;
                const bool k_is_i = local_i_[k] == row;
                const bool m_is_i = local_i_[m] == row;
                const double inv_mass = k_is_i ? local_imi_[k] : local_imj_[k];
                const double sign = k_is_i == m_is_i ? -1.0 : 1.0;
                entries.emplace_back(static_cast<int>(k), static_cast<int>(m),
                                     sign * lincs_s_[k] * lincs_s_[m]
                                         * inv_mass);
            }
        a = b;
    }
    // Ordered by (constraint, neighbour): the order inside a cluster is the
    // global table's, so every rank holding the cluster sums in one order.
    std::sort(entries.begin(), entries.end(), [](const auto& l, const auto& r) {
        return std::pair{std::get<0>(l), std::get<1>(l)}
               < std::pair{std::get<0>(r), std::get<1>(r)};
    });

    lincs_offset_.assign(n + 1, 0);
    lincs_neighbor_.clear();
    lincs_static_.clear();
    int last_k = -1, last_m = -1;
    for (const auto& [k, m, value] : entries) {
        if (k == last_k && m == last_m) {
            // Two constraints sharing both atoms: one summed entry.
            lincs_static_.back() += value;
            continue;
        }
        lincs_neighbor_.push_back(m);
        lincs_static_.push_back(value);
        ++lincs_offset_[static_cast<std::size_t>(k) + 1];
        last_k = k;
        last_m = m;
    }
    for (std::size_t k = 0; k < n; ++k)
        lincs_offset_[k + 1] += lincs_offset_[k];

    lincs_coef_.assign(lincs_static_.size(), 0.0);
    for (auto* scratch : {&lincs_bx_, &lincs_by_, &lincs_bz_, &lincs_rhs_,
                          &lincs_tmp_, &lincs_sol_, &lincs_lambda_})
        scratch->assign(n, 0.0);
}

void DistanceConstraints::lincs_expand_(int k0, int k1) {
    // sol = sum_{n <= order} A^n rhs.
    double* rhs = lincs_rhs_.data();
    double* next = lincs_tmp_.data();
    for (int k = k0; k < k1; ++k)
        lincs_sol_[static_cast<std::size_t>(k)] = rhs[k];
    for (int term = 0; term < expansion_order_; ++term) {
        for (int k = k0; k < k1; ++k) {
            double sum = 0.0;
            for (int e = lincs_offset_[static_cast<std::size_t>(k)];
                 e < lincs_offset_[static_cast<std::size_t>(k) + 1]; ++e)
                sum += lincs_coef_[static_cast<std::size_t>(e)]
                       * rhs[lincs_neighbor_[static_cast<std::size_t>(e)]];
            next[k] = sum;
            lincs_sol_[static_cast<std::size_t>(k)] += sum;
        }
        std::swap(rhs, next);
    }
}

void DistanceConstraints::solve_lincs_positions_(
    const cpu::LaunchContext& ctx) {
    cpu::Frame& f = ctx.frame;
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    const cpu::Box& box = f.box;
    const double inv_dt = 1.0 / dt_;
    const double tol = tol_;
    const int max_iter = max_iter_;
    const int n_clusters = static_cast<int>(local_offset_.size()) - 1;
    // The same half-step impulse convention as the RATTLE position stage.
    const double virial_factor =
        2.0 / ctx.runtime.units.velocity_impulse(dt_ * dt_);
    const int n_owned = f.nlocal();
    double w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0;
    int most_iters = 0;
    int degenerate = 0;
    double failed_residual = 0.0;

#pragma omp parallel for schedule(dynamic, 64)                                 \
    reduction(max : most_iters, failed_residual, degenerate)                   \
    reduction(+ : w0, w1, w2, w3, w4, w5)
    for (int c = 0; c < n_clusters; ++c) {
        const int k0 = local_offset_[static_cast<std::size_t>(c)];
        const int k1 = local_offset_[static_cast<std::size_t>(c) + 1];

        // Unit reference directions B and the coupling coefficients.
        bool coincident = false;
        for (int k = k0; k < k1; ++k) {
            const std::size_t q = static_cast<std::size_t>(k);
            const double len =
                std::sqrt(ref_sx_[q] * ref_sx_[q] + ref_sy_[q] * ref_sy_[q]
                          + ref_sz_[q] * ref_sz_[q]);
            if (!(len > 0.0)) {
                coincident = true;
                break;
            }
            lincs_bx_[q] = ref_sx_[q] / len;
            lincs_by_[q] = ref_sy_[q] / len;
            lincs_bz_[q] = ref_sz_[q] / len;
            lincs_lambda_[q] = 0.0;
        }
        if (coincident) {
            degenerate = 1;
            continue;
        }
        for (int k = k0; k < k1; ++k) {
            const std::size_t q = static_cast<std::size_t>(k);
            for (int e = lincs_offset_[q]; e < lincs_offset_[q + 1]; ++e) {
                const std::size_t ee = static_cast<std::size_t>(e);
                const std::size_t m =
                    static_cast<std::size_t>(lincs_neighbor_[ee]);
                lincs_coef_[ee] =
                    lincs_static_[ee]
                    * (lincs_bx_[q] * lincs_bx_[m] + lincs_by_[q] * lincs_by_[m]
                       + lincs_bz_[q] * lincs_bz_[m]);
            }
        }

        // x -= M^-1 B^T S sol, with the matching momentum change.
        const auto apply = [&](int a, int b) {
            for (int k = a; k < b; ++k) {
                const std::size_t q = static_cast<std::size_t>(k);
                const int i = local_i_[q], j = local_j_[q];
                const double mu = lincs_s_[q] * lincs_sol_[q];
                lincs_lambda_[q] += mu;
                const double gi = mu * local_imi_[q], gj = mu * local_imj_[q];
                x[i] -= gi * lincs_bx_[q];
                y[i] -= gi * lincs_by_[q];
                z[i] -= gi * lincs_bz_[q];
                x[j] += gj * lincs_bx_[q];
                y[j] += gj * lincs_by_[q];
                z[j] += gj * lincs_bz_[q];
                vx[i] -= gi * lincs_bx_[q] * inv_dt;
                vy[i] -= gi * lincs_by_[q] * inv_dt;
                vz[i] -= gi * lincs_bz_[q] * inv_dt;
                vx[j] += gj * lincs_bx_[q] * inv_dt;
                vy[j] += gj * lincs_by_[q] * inv_dt;
                vz[j] += gj * lincs_bz_[q] * inv_dt;
            }
        };

        // Projection: the new bonds' components along B become d.
        for (int k = k0; k < k1; ++k) {
            const std::size_t q = static_cast<std::size_t>(k);
            const int i = local_i_[q], j = local_j_[q];
            double rx = x[i] - x[j], ry = y[i] - y[j], rz = z[i] - z[j];
            box.min_image(rx, ry, rz);
            lincs_rhs_[q] = lincs_s_[q]
                            * (lincs_bx_[q] * rx + lincs_by_[q] * ry
                               + lincs_bz_[q] * rz - std::sqrt(local_d2_[q]));
        }
        lincs_expand_(k0, k1);
        apply(k0, k1);

        // Length correction: the projection leaves each bond rotated and
        // therefore too long; shorten its B component to
        // p = sqrt(2 d^2 - l^2), and repeat until every length is met.
        int iter = 0;
        for (;; ++iter) {
            double worst = 0.0;
            for (int k = k0; k < k1; ++k) {
                const std::size_t q = static_cast<std::size_t>(k);
                const int i = local_i_[q], j = local_j_[q];
                double rx = x[i] - x[j], ry = y[i] - y[j], rz = z[i] - z[j];
                box.min_image(rx, ry, rz);
                const double l2 = rx * rx + ry * ry + rz * rz;
                worst = std::max(worst,
                                 std::fabs(l2 - local_d2_[q]) / local_d2_[q]);
                const double p2 = 2.0 * local_d2_[q] - l2;
                const double p = p2 > 0.0 ? std::sqrt(p2) : 0.0;
                lincs_rhs_[q] = lincs_s_[q] * (std::sqrt(local_d2_[q]) - p);
            }
            if (worst <= tol)
                break;
            if (iter >= max_iter || !std::isfinite(worst)) {
                failed_residual =
                    std::max(failed_residual,
                             std::isfinite(worst)
                                 ? worst
                                 : std::numeric_limits<double>::infinity());
                break;
            }
            lincs_expand_(k0, k1);
            apply(k0, k1);
        }
        most_iters = std::max(most_iters, iter);

        for (int k = k0; k < k1; ++k) {
            const std::size_t q = static_cast<std::size_t>(k);
            const double sx = ref_sx_[q], sy = ref_sy_[q], sz = ref_sz_[q];
            const double len = std::sqrt(sx * sx + sy * sy + sz * sz);
            const double owner_weight =
                0.5 * (int(local_i_[q] < n_owned) + int(local_j_[q] < n_owned));
            // The displacement of atom i is g s / m_i with s the reference
            // bond vector, exactly the RATTLE multiplier.
            const double lambda =
                -owner_weight * virial_factor * lincs_lambda_[q] / len;
            w0 += lambda * sx * sx;
            w1 += lambda * sy * sy;
            w2 += lambda * sz * sz;
            w3 += lambda * sx * sy;
            w4 += lambda * sx * sz;
            w5 += lambda * sy * sz;
        }
    }
    if (degenerate)
        throw std::runtime_error(
            "LINCS: coincident constrained atoms in the reference geometry");
    if (failed_residual > 0.0)
        throw std::runtime_error(
            "LINCS: the position stage did not converge in "
            + std::to_string(max_iter_)
            + " correction sweeps (last relative residual "
            + std::to_string(failed_residual)
            + "). The matrix expansion diverges for strongly coupled "
              "constraints (coupled angle constraints, rigid triangles); use "
              "FixRattle or FixSettle for those, or raise the expansion "
              "order.");
    constraint_virial_ = {w0, w1, w2, w3, w4, w5};
    last_pos_iters_ = most_iters;
}

void DistanceConstraints::solve_lincs_velocities_(
    const cpu::LaunchContext& ctx) {
    cpu::Frame& f = ctx.frame;
    const auto x = f.span(kAtom_x);
    const auto y = f.span(kAtom_y);
    const auto z = f.span(kAtom_z);
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    const cpu::Box& box = f.box;
    const double tol = tol_;
    const int max_iter = max_iter_;
    const int n_clusters = static_cast<int>(local_offset_.size()) - 1;
    int most_iters = 0;
    int degenerate = 0;
    double failed_residual = 0.0;

#pragma omp parallel for schedule(dynamic, 64)                                 \
    reduction(max : most_iters, failed_residual, degenerate)
    for (int c = 0; c < n_clusters; ++c) {
        const int k0 = local_offset_[static_cast<std::size_t>(c)];
        const int k1 = local_offset_[static_cast<std::size_t>(c) + 1];

        // Current bond directions; lincs_lambda_ holds the bond lengths so
        // the residual is RATTLE's |r.v|/d^2.
        bool coincident = false;
        for (int k = k0; k < k1; ++k) {
            const std::size_t q = static_cast<std::size_t>(k);
            const int i = local_i_[q], j = local_j_[q];
            double rx = x[i] - x[j], ry = y[i] - y[j], rz = z[i] - z[j];
            box.min_image(rx, ry, rz);
            const double len = std::sqrt(rx * rx + ry * ry + rz * rz);
            if (!(len > 0.0)) {
                coincident = true;
                break;
            }
            lincs_bx_[q] = rx / len;
            lincs_by_[q] = ry / len;
            lincs_bz_[q] = rz / len;
            lincs_lambda_[q] = len;
        }
        if (coincident) {
            degenerate = 1;
            continue;
        }
        for (int k = k0; k < k1; ++k) {
            const std::size_t q = static_cast<std::size_t>(k);
            for (int e = lincs_offset_[q]; e < lincs_offset_[q + 1]; ++e) {
                const std::size_t ee = static_cast<std::size_t>(e);
                const std::size_t m =
                    static_cast<std::size_t>(lincs_neighbor_[ee]);
                lincs_coef_[ee] =
                    lincs_static_[ee]
                    * (lincs_bx_[q] * lincs_bx_[m] + lincs_by_[q] * lincs_by_[m]
                       + lincs_bz_[q] * lincs_bz_[m]);
            }
        }

        // The constraints are linear in v, so each sweep is one step of
        // iterative refinement on the truncated expansion.
        int iter = 0;
        for (;; ++iter) {
            double worst = 0.0;
            for (int k = k0; k < k1; ++k) {
                const std::size_t q = static_cast<std::size_t>(k);
                const int i = local_i_[q], j = local_j_[q];
                const double bv = lincs_bx_[q] * (vx[i] - vx[j])
                                  + lincs_by_[q] * (vy[i] - vy[j])
                                  + lincs_bz_[q] * (vz[i] - vz[j]);
                worst = std::max(worst, std::fabs(bv) * lincs_lambda_[q]
                                            / local_d2_[q]);
                lincs_rhs_[q] = lincs_s_[q] * bv;
            }
            if (worst <= tol)
                break;
            if (iter >= max_iter || !std::isfinite(worst)) {
                failed_residual =
                    std::max(failed_residual,
                             std::isfinite(worst)
                                 ? worst
                                 : std::numeric_limits<double>::infinity());
                break;
            }
            lincs_expand_(k0, k1);
            for (int k = k0; k < k1; ++k) {
                const std::size_t q = static_cast<std::size_t>(k);
                const int i = local_i_[q], j = local_j_[q];
                const double mu = lincs_s_[q] * lincs_sol_[q];
                const double gi = mu * local_imi_[q], gj = mu * local_imj_[q];
                vx[i] -= gi * lincs_bx_[q];
                vy[i] -= gi * lincs_by_[q];
                vz[i] -= gi * lincs_bz_[q];
                vx[j] += gj * lincs_bx_[q];
                vy[j] += gj * lincs_by_[q];
                vz[j] += gj * lincs_bz_[q];
            }
        }
        most_iters = std::max(most_iters, iter);
    }
    if (degenerate)
        throw std::runtime_error("LINCS: coincident constrained atoms");
    if (failed_residual > 0.0)
        throw std::runtime_error(
            "LINCS: the velocity stage did not converge in "
            + std::to_string(max_iter_) + " sweeps (last relative residual "
            + std::to_string(failed_residual)
            + "). The matrix expansion diverges for strongly coupled "
              "constraints; use FixRattle or FixSettle for those, or raise "
              "the expansion order.");
    last_vel_iters_ = most_iters;
}

} // namespace md
} // namespace cpu
} // namespace ATV
