/// @file cpu/dynamics/const_pot_cg.cpp
/// @brief CG constant-potential electrode charge solve.

#include "atomiverse/cpu/dynamics/const_pot_cg.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/dynamics/charge_packer.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ATV {
namespace cpu {
namespace md {

namespace {

constexpr double kSqrt2 = 1.41421356237309504880;
constexpr double kSqrt2OverPi = 0.79788456080286535588; // sqrt(2/π)

double min_image_r(const cpu::Box& box, double dx, double dy, double dz) {
    double rx = dx, ry = dy, rz = dz;
    box.min_image(rx, ry, rz);
    return std::sqrt(rx * rx + ry * ry + rz * rz);
}

} // namespace

ConstPotCG::ConstPotCG(double eta_in) : eta(eta_in) {
    if (!(eta > 0.0))
        throw std::invalid_argument("ConstPotCG: eta must be > 0 [1/Å]");
}

ConstPotCG& ConstPotCG::add_electrode(int type, double psi) & {
    groups_.push_back(ElectrodeGroup{type, psi});
    return *this;
}

ConstPotCG ConstPotCG::add_electrode(int type, double psi) && {
    groups_.push_back(ElectrodeGroup{type, psi});
    return std::move(*this);
}

void ConstPotCG::build(const BuildContext& ctx) {
    if (groups_.empty())
        throw std::runtime_error("ConstPotCG: add at least one electrode via "
                                 "add_electrode(type, Psi)");
    collect_electrodes_(ctx.frame);
    if (elec_atoms_.empty())
        throw std::runtime_error(
            "ConstPotCG: no atoms match registered electrode types");
    build_matrix_(ctx.frame);
    matrix_ready_ = true;
    total_Q_.assign(groups_.size(), 0.0);
}

void ConstPotCG::launch(const LaunchContext& ctx) {
    if (!matrix_ready_ || rebuild_matrix_every_step) {
        collect_electrodes_(ctx.frame);
        build_matrix_(ctx.frame);
        matrix_ready_ = true;
    }
    build_rhs_(ctx.frame);
    solve_();
    write_charges_(ctx.frame);
    ChargePacker::forward(ctx);
}

void ConstPotCG::destroy(const DestroyContext&) {
    matrix_ready_ = false;
    elec_atoms_.clear();
    A_.clear();
    Q_.clear();
}

double ConstPotCG::total_charge(std::size_t ie) const {
    if (ie >= total_Q_.size())
        throw std::out_of_range("ConstPotCG::total_charge");
    return total_Q_[ie];
}

double ConstPotCG::delta_psi() const {
    if (groups_.size() < 2)
        return 0.0;
    return groups_.back().psi - groups_.front().psi;
}

double ConstPotCG::neutrality_residual() const {
    double s = 0.0;
    for (double q : Q_)
        s += q;
    return std::abs(s);
}

void ConstPotCG::collect_electrodes_(const cpu::Frame& frame) {
    elec_atoms_.clear();
    elec_group_.clear();
    psi_vec_.clear();
    const int n = frame.nlocal();
    auto type = frame.span(kAtom_type);
    for (int i = 0; i < n; ++i) {
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            if (static_cast<int>(type[i]) == groups_[g].type) {
                elec_atoms_.push_back(i);
                elec_group_.push_back(static_cast<int>(g));
                psi_vec_.push_back(groups_[g].psi);
                break;
            }
        }
    }
    const int m = static_cast<int>(elec_atoms_.size());
    Q_.assign(static_cast<std::size_t>(m), 0.0);
    b_.assign(static_cast<std::size_t>(m), 0.0);
    A_.assign(static_cast<std::size_t>(m) * m, 0.0);
}

double ConstPotCG::pair_A_(double r, double eta_i, double eta_j) const {
    if (r < 1e-14)
        return units::coulomb * eta_i * kSqrt2OverPi;
    const double etaij =
        eta_i * eta_j / std::sqrt(eta_i * eta_i + eta_j * eta_j);
    return units::coulomb * std::erf(etaij * r) / r;
}

double ConstPotCG::pair_b_(double r, double q, double eta_i) const {
    if (r < 1e-14)
        return 0.0;
    return units::coulomb * q * std::erf(eta_i * r) / r;
}

void ConstPotCG::build_matrix_(const cpu::Frame& frame) {
    const int m = static_cast<int>(elec_atoms_.size());
    if (m == 0)
        return;
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    const cpu::Box& box = frame.box;
    const double Lx = box.length(0);
    const double Ly = box.length(1);
    const int R = xy_image_range < 0 ? 0 : xy_image_range;

    std::fill(A_.begin(), A_.end(), 0.0);
    for (int a = 0; a < m; ++a) {
        const int ia = elec_atoms_[static_cast<std::size_t>(a)];
        A_[static_cast<std::size_t>(a) * m + a] =
            units::coulomb * eta * kSqrt2OverPi;

        for (int b = a + 1; b < m; ++b) {
            const int ib = elec_atoms_[static_cast<std::size_t>(b)];
            double aij = 0.0;
            for (int nx = -R; nx <= R; ++nx) {
                for (int ny = -R; ny <= R; ++ny) {
                    double r;
                    if (R == 0) {
                        r = min_image_r(box, x[ia] - x[ib], y[ia] - y[ib],
                                        z[ia] - z[ib]);
                    } else {
                        const double dx = x[ia] - x[ib] + nx * Lx;
                        const double dy = y[ia] - y[ib] + ny * Ly;
                        const double dz = z[ia] - z[ib];
                        r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    }
                    if (r < 1e-14)
                        continue;
                    aij += pair_A_(r, eta, eta);
                    if (R == 0)
                        break;
                }
                if (R == 0)
                    break;
            }
            A_[static_cast<std::size_t>(a) * m + b] = aij;
            A_[static_cast<std::size_t>(b) * m + a] = aij;
        }

        if (R > 0) {
            double self_img = 0.0;
            for (int nx = -R; nx <= R; ++nx) {
                for (int ny = -R; ny <= R; ++ny) {
                    if (nx == 0 && ny == 0)
                        continue;
                    const double r = std::sqrt((nx * Lx) * (nx * Lx)
                                               + (ny * Ly) * (ny * Ly));
                    self_img +=
                        units::coulomb * std::erf((eta / kSqrt2) * r) / r;
                }
            }
            A_[static_cast<std::size_t>(a) * m + a] += self_img;
        }
    }
}

void ConstPotCG::build_rhs_(const cpu::Frame& frame) {
    const int m = static_cast<int>(elec_atoms_.size());
    const int n = frame.nlocal();
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto q = frame.span(kAtom_charge);
    const cpu::Box& box = frame.box;
    const double Lx = box.length(0);
    const double Ly = box.length(1);
    const int R = xy_image_range < 0 ? 0 : xy_image_range;

    std::vector<char> is_elec(static_cast<std::size_t>(n), 0);
    for (int ia : elec_atoms_)
        is_elec[static_cast<std::size_t>(ia)] = 1;

    std::fill(b_.begin(), b_.end(), 0.0);
    for (int a = 0; a < m; ++a) {
        const int ia = elec_atoms_[static_cast<std::size_t>(a)];
        double bi = 0.0;
        for (int j = 0; j < n; ++j) {
            if (is_elec[static_cast<std::size_t>(j)])
                continue;
            const double qj = static_cast<double>(q[j]);
            if (qj == 0.0)
                continue;
            for (int nx = -R; nx <= R; ++nx) {
                for (int ny = -R; ny <= R; ++ny) {
                    double r;
                    if (R == 0) {
                        r = min_image_r(box, x[ia] - x[j], y[ia] - y[j],
                                        z[ia] - z[j]);
                    } else {
                        const double dx = x[ia] - x[j] + nx * Lx;
                        const double dy = y[ia] - y[j] + ny * Ly;
                        const double dz = z[ia] - z[j];
                        r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    }
                    bi += pair_b_(r, qj, eta);
                    if (R == 0)
                        break;
                }
                if (R == 0)
                    break;
            }
        }
        b_[static_cast<std::size_t>(a)] = bi;
    }
}

int ConstPotCG::pcg_solve_(const std::vector<double>& rhs,
                           std::vector<double>& x) const {
    const int m = static_cast<int>(elec_atoms_.size());
    if (m == 0)
        return 0;

    // Jacobi preconditioner: M = diag(A)
    std::vector<double> inv_diag(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        const double d = A_[static_cast<std::size_t>(i) * m + i];
        if (!(d > 0.0))
            throw std::runtime_error(
                "ConstPotCG: non-positive diagonal in elastance matrix");
        inv_diag[static_cast<std::size_t>(i)] = 1.0 / d;
    }

    x.assign(static_cast<std::size_t>(m), 0.0);
    std::vector<double> r = rhs; // r = b - A x0, x0 = 0
    std::vector<double> z(static_cast<std::size_t>(m));
    std::vector<double> p(static_cast<std::size_t>(m));
    std::vector<double> Ap(static_cast<std::size_t>(m));

    auto apply_M_inv = [&](const std::vector<double>& v,
                           std::vector<double>& out) {
        for (int i = 0; i < m; ++i)
            out[static_cast<std::size_t>(i)] =
                inv_diag[static_cast<std::size_t>(i)]
                * v[static_cast<std::size_t>(i)];
    };
    auto matvec = [&](const std::vector<double>& v, std::vector<double>& out) {
        for (int i = 0; i < m; ++i) {
            double s = 0.0;
            for (int j = 0; j < m; ++j)
                s += A_[static_cast<std::size_t>(i) * m + j]
                     * v[static_cast<std::size_t>(j)];
            out[static_cast<std::size_t>(i)] = s;
        }
    };
    auto dot = [&](const std::vector<double>& a,
                   const std::vector<double>& bvec) {
        double s = 0.0;
        for (int i = 0; i < m; ++i)
            s += a[static_cast<std::size_t>(i)]
                 * bvec[static_cast<std::size_t>(i)];
        return s;
    };

    double rhs_norm = std::sqrt(dot(rhs, rhs));
    if (rhs_norm < 1e-30) {
        // Already zero solution.
        return 0;
    }

    apply_M_inv(r, z);
    p = z;
    double rz_old = dot(r, z);
    const double tol_abs = cg_tol * rhs_norm;
    const int maxit = max_cg_iter > 0 ? max_cg_iter : m * 4;

    for (int it = 0; it < maxit; ++it) {
        matvec(p, Ap);
        const double pAp = dot(p, Ap);
        if (!(std::abs(pAp) > 1e-300))
            throw std::runtime_error("ConstPotCG: CG breakdown (p·Ap ≈ 0)");
        const double alpha = rz_old / pAp;
        for (int i = 0; i < m; ++i) {
            x[static_cast<std::size_t>(i)] +=
                alpha * p[static_cast<std::size_t>(i)];
            r[static_cast<std::size_t>(i)] -=
                alpha * Ap[static_cast<std::size_t>(i)];
        }
        const double rnorm = std::sqrt(dot(r, r));
        if (rnorm <= tol_abs)
            return it + 1;

        apply_M_inv(r, z);
        const double rz_new = dot(r, z);
        const double beta = rz_new / rz_old;
        for (int i = 0; i < m; ++i)
            p[static_cast<std::size_t>(i)] =
                z[static_cast<std::size_t>(i)]
                + beta * p[static_cast<std::size_t>(i)];
        rz_old = rz_new;
    }
    throw std::runtime_error(
        "ConstPotCG: CG did not converge within max_cg_iter");
}

void ConstPotCG::solve_() {
    const int m = static_cast<int>(elec_atoms_.size());
    if (m == 0)
        return;

    std::vector<double> rhs(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i)
        rhs[static_cast<std::size_t>(i)] = psi_vec_[static_cast<std::size_t>(i)]
                                           - b_[static_cast<std::size_t>(i)];

    if (electroneutrality && m >= 1) {
        // Q = v - (e·v)/(e·u) u  with A v = rhs, A u = e  (same projector as BO).
        std::vector<double> v(static_cast<std::size_t>(m));
        std::vector<double> u(static_cast<std::size_t>(m));
        std::vector<double> e(static_cast<std::size_t>(m), 1.0);
        cg_iters_primary_ = pcg_solve_(rhs, v);
        (void)pcg_solve_(e, u);
        double eT_v = 0.0;
        double eT_u = 0.0;
        for (int i = 0; i < m; ++i) {
            eT_v += v[static_cast<std::size_t>(i)];
            eT_u += u[static_cast<std::size_t>(i)];
        }
        if (std::abs(eT_u) < 1e-30)
            throw std::runtime_error(
                "ConstPotCG: singular neutrality projector");
        const double scale = eT_v / eT_u;
        for (int i = 0; i < m; ++i)
            Q_[static_cast<std::size_t>(i)] =
                v[static_cast<std::size_t>(i)]
                - scale * u[static_cast<std::size_t>(i)];
    } else {
        cg_iters_primary_ = pcg_solve_(rhs, Q_);
    }

    residual_ = 0.0;
    for (int i = 0; i < m; ++i) {
        double Ax = 0.0;
        for (int j = 0; j < m; ++j)
            Ax += A_[static_cast<std::size_t>(i) * m + j]
                  * Q_[static_cast<std::size_t>(j)];
        residual_ = std::max(residual_,
                             std::abs(Ax - rhs[static_cast<std::size_t>(i)]));
    }

    total_Q_.assign(groups_.size(), 0.0);
    for (int a = 0; a < m; ++a) {
        const int g = elec_group_[static_cast<std::size_t>(a)];
        total_Q_[static_cast<std::size_t>(g)] +=
            Q_[static_cast<std::size_t>(a)];
    }
}

void ConstPotCG::write_charges_(cpu::Frame& frame) const {
    auto q = frame.span(kAtom_charge);
    for (std::size_t a = 0; a < elec_atoms_.size(); ++a)
        q[elec_atoms_[a]] = static_cast<StorageReal>(Q_[a]);
}

} // namespace md
} // namespace cpu
} // namespace ATV
