/// @file cpu/dynamics/dipole_electrode_joint.cpp
/// @brief Coupled PIM ⊕ ConstPot electrode joint solve (MetalWalls E5).

#include "atomiverse/cpu/dynamics/dipole_electrode_joint.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/dynamics/charge_packer.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ATV {
namespace cpu {
namespace md {
namespace {

constexpr double kSqrt2 = 1.41421356237309504880;
constexpr double kSqrt2OverPi = 0.79788456080286535588; // sqrt(2/π)

void cholesky_solve(int n, std::vector<double>& A, std::vector<double>& x) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[static_cast<std::size_t>(i) * n + j];
            for (int k = 0; k < j; ++k)
                s -= A[static_cast<std::size_t>(i) * n + k]
                     * A[static_cast<std::size_t>(j) * n + k];
            if (i == j) {
                if (s <= 0.0)
                    throw std::runtime_error(
                        "DipoleElectrodeJoint: elastance not SPD");
                A[static_cast<std::size_t>(i) * n + j] = std::sqrt(s);
            } else {
                A[static_cast<std::size_t>(i) * n + j] =
                    s / A[static_cast<std::size_t>(j) * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        double s = x[static_cast<std::size_t>(i)];
        for (int k = 0; k < i; ++k)
            s -= A[static_cast<std::size_t>(i) * n + k]
                 * x[static_cast<std::size_t>(k)];
        x[static_cast<std::size_t>(i)] =
            s / A[static_cast<std::size_t>(i) * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = x[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < n; ++k)
            s -= A[static_cast<std::size_t>(k) * n + i]
                 * x[static_cast<std::size_t>(k)];
        x[static_cast<std::size_t>(i)] =
            s / A[static_cast<std::size_t>(i) * n + i];
    }
}

void invert_spd(int n, const std::vector<double>& A_in,
                std::vector<double>& inv) {
    inv.assign(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> Awork;
    std::vector<double> col(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        Awork = A_in;
        std::fill(col.begin(), col.end(), 0.0);
        col[static_cast<std::size_t>(j)] = 1.0;
        cholesky_solve(n, Awork, col);
        for (int i = 0; i < n; ++i)
            inv[static_cast<std::size_t>(i) * n + j] =
                col[static_cast<std::size_t>(i)];
    }
}

double min_image_r(const cpu::Box& box, double dx, double dy, double dz) {
    double rx = dx, ry = dy, rz = dz;
    box.min_image(rx, ry, rz);
    return std::sqrt(rx * rx + ry * ry + rz * rz);
}

inline double thole_u(double r, double ai, double aj, double a) {
    if (!(a > 0.0) || !(ai > 0.0) || !(aj > 0.0) || !(r > 0.0))
        return 0.0;
    return a * r / std::pow(ai * aj, 1.0 / 6.0);
}

inline void thole_lambda(double u, double& l3, double& l5, double& l7) {
    if (!(u > 0.0)) {
        l3 = l5 = l7 = 1.0;
        return;
    }
    const double u3 = u * u * u;
    const double u6 = u3 * u3;
    const double expu = std::exp(-u3);
    l3 = 1.0 - expu;
    l5 = 1.0 - (1.0 + u3) * expu;
    l7 = 1.0 - (1.0 + u3 + 0.6 * u6) * expu;
}

} // namespace

DipoleElectrodeJoint::DipoleElectrodeJoint(double eta_in,
                                           std::vector<double> alpha_by_type_in)
    : eta(eta_in), alpha_by_type(std::move(alpha_by_type_in)) {
    if (!(eta > 0.0))
        throw std::invalid_argument(
            "DipoleElectrodeJoint: eta must be > 0 [1/Å]");
    for (double a : alpha_by_type) {
        if (!(a >= 0.0) || !std::isfinite(a))
            throw std::invalid_argument(
                "DipoleElectrodeJoint: alpha_by_type entries must be >= 0");
    }
    if (!(thole_a >= 0.0) || !(pol_tol > 0.0) || pol_max_iter < 1
        || !(joint_tol > 0.0) || joint_max_iter < 1)
        throw std::invalid_argument(
            "DipoleElectrodeJoint: invalid SCC / joint parameters");
}

DipoleElectrodeJoint& DipoleElectrodeJoint::add_electrode(int type,
                                                          double psi) & {
    groups_.push_back(ElectrodeGroup{type, psi});
    return *this;
}

DipoleElectrodeJoint DipoleElectrodeJoint::add_electrode(int type,
                                                         double psi) && {
    groups_.push_back(ElectrodeGroup{type, psi});
    return std::move(*this);
}

double DipoleElectrodeJoint::alpha_of_(int type) const {
    if (type < 0 || static_cast<std::size_t>(type) >= alpha_by_type.size())
        return 0.0;
    return alpha_by_type[static_cast<std::size_t>(type)];
}

double DipoleElectrodeJoint::pair_A_(double r, double eta_i,
                                     double eta_j) const {
    if (r < 1e-14)
        return units::coulomb * eta_i * kSqrt2OverPi;
    const double etaij =
        eta_i * eta_j / std::sqrt(eta_i * eta_i + eta_j * eta_j);
    return units::coulomb * std::erf(etaij * r) / r;
}

double DipoleElectrodeJoint::pair_b_(double r, double q, double eta_i) const {
    if (r < 1e-14)
        return 0.0;
    return units::coulomb * q * std::erf(eta_i * r) / r;
}

void DipoleElectrodeJoint::collect_electrodes_(const cpu::Frame& frame) {
    elec_atoms_.clear();
    elec_group_.clear();
    psi_vec_.clear();
    const int n = frame.nlocal();
    auto type = frame.span(kAtom_type);
    is_elec_.assign(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            if (static_cast<int>(type[i]) == groups_[g].type) {
                elec_atoms_.push_back(i);
                elec_group_.push_back(static_cast<int>(g));
                psi_vec_.push_back(groups_[g].psi);
                is_elec_[static_cast<std::size_t>(i)] = 1;
                break;
            }
        }
    }
    const int m = static_cast<int>(elec_atoms_.size());
    Q_.assign(static_cast<std::size_t>(m), 0.0);
    b_.assign(static_cast<std::size_t>(m), 0.0);
    A_.assign(static_cast<std::size_t>(m) * m, 0.0);
    last_mu_.assign(static_cast<std::size_t>(3 * n), 0.0);
}

void DipoleElectrodeJoint::build_matrix_(const cpu::Frame& frame) {
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

void DipoleElectrodeJoint::build_rhs_(const Spans& s,
                                      const std::vector<double>& mu) {
    const int m = static_cast<int>(elec_atoms_.size());
    const int n = s.n;
    const StorageReal* x = s.x;
    const StorageReal* y = s.y;
    const StorageReal* z = s.z;
    const StorageReal* q = s.charge;
    const int* type = s.type;
    const cpu::Box& box = *s.box;
    const double Lx = box.length(0);
    const double Ly = box.length(1);
    const int R = xy_image_range < 0 ? 0 : xy_image_range;
    const double k = units::coulomb;

    std::fill(b_.begin(), b_.end(), 0.0);
    for (int a = 0; a < m; ++a) {
        const int ia = elec_atoms_[static_cast<std::size_t>(a)];
        double bi = 0.0;

        // Permanent non-electrode charges (Gaussian–point).
        for (int j = 0; j < n; ++j) {
            if (is_elec_[static_cast<std::size_t>(j)])
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

        // Dipole potential at electrode (α_elec = 0 → undamped λ3=1).
        for (int j = 0; j < n; ++j) {
            if (is_elec_[static_cast<std::size_t>(j)])
                continue;
            const double aj = alpha_of_(type[j]);
            if (!(aj > 0.0))
                continue;
            const double* muj = mu.data() + 3 * j;
            if (muj[0] == 0.0 && muj[1] == 0.0 && muj[2] == 0.0)
                continue;
            for (int nx = -R; nx <= R; ++nx) {
                for (int ny = -R; ny <= R; ++ny) {
                    double dx, dy, dz, r2;
                    if (R == 0) {
                        dx = x[ia] - x[j];
                        dy = y[ia] - y[j];
                        dz = z[ia] - z[j];
                        box.min_image(dx, dy, dz);
                        r2 = dx * dx + dy * dy + dz * dz;
                    } else {
                        dx = x[ia] - x[j] + nx * Lx;
                        dy = y[ia] - y[j] + ny * Ly;
                        dz = z[ia] - z[j];
                        r2 = dx * dx + dy * dy + dz * dz;
                    }
                    if (!(r2 > 1e-28)) {
                        if (R == 0)
                            break;
                        continue;
                    }
                    const double r = std::sqrt(r2);
                    const double mdotr =
                        muj[0] * dx + muj[1] * dy + muj[2] * dz;
                    bi += k * mdotr / (r2 * r);
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

void DipoleElectrodeJoint::solve_Q_() {
    const int m = static_cast<int>(elec_atoms_.size());
    std::vector<double> rhs(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i)
        rhs[static_cast<std::size_t>(i)] = psi_vec_[static_cast<std::size_t>(i)]
                                           - b_[static_cast<std::size_t>(i)];

    A_work_ = A_;
    std::vector<double> Q_unconst = rhs;
    cholesky_solve(m, A_work_, Q_unconst);

    if (electroneutrality && m >= 1) {
        std::vector<double> inv;
        invert_spd(m, A_, inv);
        std::vector<double> u(static_cast<std::size_t>(m), 0.0);
        for (int i = 0; i < m; ++i) {
            double s = 0.0;
            for (int j = 0; j < m; ++j)
                s += inv[static_cast<std::size_t>(i) * m + j];
            u[static_cast<std::size_t>(i)] = s;
        }
        double eT_u = 0.0, eT_Q = 0.0;
        for (int i = 0; i < m; ++i) {
            eT_u += u[static_cast<std::size_t>(i)];
            eT_Q += Q_unconst[static_cast<std::size_t>(i)];
        }
        if (std::abs(eT_u) < 1e-30)
            throw std::runtime_error(
                "DipoleElectrodeJoint: singular neutrality projector");
        const double scale = eT_Q / eT_u;
        for (int i = 0; i < m; ++i)
            Q_[static_cast<std::size_t>(i)] =
                Q_unconst[static_cast<std::size_t>(i)]
                - scale * u[static_cast<std::size_t>(i)];
    } else {
        Q_ = Q_unconst;
    }

    // Unconstrained residual of AQ − (Ψ − b) after neutrality projection
    // still reports the physical equation residual (with projected Q).
    charge_residual_ = 0.0;
    for (int i = 0; i < m; ++i) {
        double Ax = 0.0;
        for (int j = 0; j < m; ++j)
            Ax += A_[static_cast<std::size_t>(i) * m + j]
                  * Q_[static_cast<std::size_t>(j)];
        charge_residual_ = std::max(
            charge_residual_, std::abs(Ax - rhs[static_cast<std::size_t>(i)]));
    }

    total_Q_.assign(groups_.size(), 0.0);
    for (int a = 0; a < m; ++a) {
        const int g = elec_group_[static_cast<std::size_t>(a)];
        total_Q_[static_cast<std::size_t>(g)] +=
            Q_[static_cast<std::size_t>(a)];
    }
}

void DipoleElectrodeJoint::write_charges_(const Spans& s) const {
    for (std::size_t a = 0; a < elec_atoms_.size(); ++a)
        s.charge[elec_atoms_[a]] = static_cast<StorageReal>(Q_[a]);
}

double DipoleElectrodeJoint::scc_mu_(const Spans& s) {
    const int n = s.n;
    const StorageReal* x = s.x;
    const StorageReal* y = s.y;
    const StorageReal* z = s.z;
    const StorageReal* charge = s.charge;
    const int* type = s.type;
    const cpu::Box& box = *s.box;

    std::vector<double> alpha(static_cast<std::size_t>(n), 0.0);
    bool any_pol = false;
    for (int i = 0; i < n; ++i) {
        // Electrode sites are never polarizable in the joint foothold.
        if (is_elec_[static_cast<std::size_t>(i)]) {
            alpha[static_cast<std::size_t>(i)] = 0.0;
            continue;
        }
        alpha[static_cast<std::size_t>(i)] = alpha_of_(type[i]);
        if (alpha[static_cast<std::size_t>(i)] > 0.0)
            any_pol = true;
    }

    if (!any_pol) {
        std::fill(last_mu_.begin(), last_mu_.end(), 0.0);
        dipole_residual_ = 0.0;
        return 0.0;
    }

    // Permanent field from all charges (point multipole, Thole when both α>0).
    std::vector<double> e_perm(static_cast<std::size_t>(3 * n), 0.0);
    for (int i = 0; i < n; ++i) {
        const double ai = alpha[static_cast<std::size_t>(i)];
        if (!(ai > 0.0))
            continue;
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            const double qj = static_cast<double>(charge[j]);
            if (qj == 0.0)
                continue;
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0))
                continue;
            const double r = std::sqrt(r2);
            const double aj = alpha[static_cast<std::size_t>(j)];
            double l3 = 1.0, l5 = 1.0, l7 = 1.0;
            if (thole_a > 0.0 && ai > 0.0 && aj > 0.0)
                thole_lambda(thole_u(r, ai, aj, thole_a), l3, l5, l7);
            (void)l5;
            (void)l7;
            const double fac = l3 * qj / (r2 * r);
            e_perm[static_cast<std::size_t>(3 * i + 0)] += fac * dx;
            e_perm[static_cast<std::size_t>(3 * i + 1)] += fac * dy;
            e_perm[static_cast<std::size_t>(3 * i + 2)] += fac * dz;
        }
    }

    std::vector<double> mu = last_mu_; // warm-start outer loop
    if (static_cast<int>(mu.size()) != 3 * n)
        mu.assign(static_cast<std::size_t>(3 * n), 0.0);
    // Zero electrode dipoles
    for (int i = 0; i < n; ++i) {
        if (is_elec_[static_cast<std::size_t>(i)]
            || !(alpha[static_cast<std::size_t>(i)] > 0.0)) {
            mu[static_cast<std::size_t>(3 * i + 0)] = 0.0;
            mu[static_cast<std::size_t>(3 * i + 1)] = 0.0;
            mu[static_cast<std::size_t>(3 * i + 2)] = 0.0;
        }
    }

    std::vector<double> mu_new(static_cast<std::size_t>(3 * n), 0.0);
    std::vector<double> e_tot(static_cast<std::size_t>(3 * n), 0.0);
    double dmax = 0.0;
    bool ok = false;
    for (int it = 0; it < pol_max_iter; ++it) {
        std::copy(e_perm.begin(), e_perm.end(), e_tot.begin());
        for (int i = 0; i < n; ++i) {
            const double ai = alpha[static_cast<std::size_t>(i)];
            if (!(ai > 0.0))
                continue;
            for (int j = 0; j < n; ++j) {
                if (i == j)
                    continue;
                const double aj = alpha[static_cast<std::size_t>(j)];
                if (!(aj > 0.0))
                    continue;
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                double dz = z[i] - z[j];
                box.min_image(dx, dy, dz);
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (!(r2 > 0.0))
                    continue;
                const double r = std::sqrt(r2);
                const double r3 = r2 * r;
                const double r5 = r3 * r2;
                double l3 = 1.0, l5 = 1.0, l7 = 1.0;
                if (thole_a > 0.0)
                    thole_lambda(thole_u(r, ai, aj, thole_a), l3, l5, l7);
                (void)l7;
                const double f3 = l3 / r3;
                const double f5 = l5 / r5;
                const double* muj = mu.data() + 3 * j;
                const double mdotr = muj[0] * dx + muj[1] * dy + muj[2] * dz;
                e_tot[static_cast<std::size_t>(3 * i + 0)] +=
                    3.0 * f5 * mdotr * dx - f3 * muj[0];
                e_tot[static_cast<std::size_t>(3 * i + 1)] +=
                    3.0 * f5 * mdotr * dy - f3 * muj[1];
                e_tot[static_cast<std::size_t>(3 * i + 2)] +=
                    3.0 * f5 * mdotr * dz - f3 * muj[2];
            }
        }

        dmax = 0.0;
        for (int i = 0; i < n; ++i) {
            const double ai = alpha[static_cast<std::size_t>(i)];
            for (int c = 0; c < 3; ++c) {
                const double m =
                    ai * e_tot[static_cast<std::size_t>(3 * i + c)];
                mu_new[static_cast<std::size_t>(3 * i + c)] = m;
                dmax = std::max(
                    dmax,
                    std::fabs(m - mu[static_cast<std::size_t>(3 * i + c)]));
            }
        }
        mu.swap(mu_new);
        if (dmax < pol_tol) {
            ok = true;
            break;
        }
    }
    if (!ok)
        throw std::runtime_error(
            "DipoleElectrodeJoint: inner PIM SCC did not converge");

    double d_outer = 0.0;
    for (std::size_t k = 0; k < mu.size(); ++k)
        d_outer = std::max(d_outer, std::fabs(mu[k] - last_mu_[k]));
    last_mu_ = mu;
    dipole_residual_ = d_outer;
    return d_outer;
}

void DipoleElectrodeJoint::build(const BuildContext& ctx) {
    if (groups_.empty())
        throw std::runtime_error(
            "DipoleElectrodeJoint: add at least one electrode via "
            "add_electrode(type, Psi)");
    if (!(joint_tol > 0.0) || joint_max_iter < 1)
        throw std::invalid_argument(
            "DipoleElectrodeJoint: invalid joint_tol / joint_max_iter");
    collect_electrodes_(ctx.frame);
    if (elec_atoms_.empty())
        throw std::runtime_error(
            "DipoleElectrodeJoint: no atoms match registered electrode types");
    build_matrix_(ctx.frame);
    matrix_ready_ = true;
    total_Q_.assign(groups_.size(), 0.0);
    last_joint_iters_ = 0;
    last_joint_converged_ = false;
}

void DipoleElectrodeJoint::launch(const LaunchContext& ctx) {
    if (!matrix_ready_ || rebuild_matrix_every_step) {
        collect_electrodes_(ctx.frame);
        build_matrix_(ctx.frame);
        matrix_ready_ = true;
    }

    // Resolve Frame columns once per launch (resolution budget).
    Spans s;
    s.n = ctx.frame.nlocal();
    s.x = ctx.frame.span(kAtom_x);
    s.y = ctx.frame.span(kAtom_y);
    s.z = ctx.frame.span(kAtom_z);
    s.charge = ctx.frame.span(kAtom_charge);
    s.type = ctx.frame.span(kAtom_type);
    s.box = &ctx.frame.box;

    const int m = static_cast<int>(elec_atoms_.size());
    if (static_cast<int>(last_mu_.size()) != 3 * s.n)
        last_mu_.assign(static_cast<std::size_t>(3 * s.n), 0.0);

    std::vector<double> Q_prev(static_cast<std::size_t>(m), 0.0);
    last_joint_converged_ = false;
    last_joint_iters_ = 0;

    for (int outer = 0; outer < joint_max_iter; ++outer) {
        build_rhs_(s, last_mu_);
        Q_prev = Q_;
        solve_Q_();
        write_charges_(s);

        double dQ = 0.0;
        for (int a = 0; a < m; ++a)
            dQ = std::max(dQ, std::abs(Q_[static_cast<std::size_t>(a)]
                                       - Q_prev[static_cast<std::size_t>(a)]));

        const double dmu = scc_mu_(s);
        last_joint_iters_ = outer + 1;

        if (dQ < joint_tol && dmu < joint_tol) {
            last_joint_converged_ = true;
            break;
        }
    }
    if (!last_joint_converged_)
        throw std::runtime_error(
            "DipoleElectrodeJoint: joint Q–μ loop did not converge");
    ChargePacker::forward(ctx);
}

void DipoleElectrodeJoint::destroy(const DestroyContext&) {
    matrix_ready_ = false;
    elec_atoms_.clear();
    A_.clear();
    Q_.clear();
    last_mu_.clear();
    last_joint_iters_ = 0;
    last_joint_converged_ = false;
}

double DipoleElectrodeJoint::total_charge(std::size_t ie) const {
    if (ie >= total_Q_.size())
        throw std::out_of_range("DipoleElectrodeJoint::total_charge");
    return total_Q_[ie];
}

double DipoleElectrodeJoint::neutrality_residual() const {
    double s = 0.0;
    for (double q : Q_)
        s += q;
    return std::abs(s);
}

} // namespace md
} // namespace cpu
} // namespace ATV
