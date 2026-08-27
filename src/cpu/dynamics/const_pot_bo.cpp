/// @file cpu/dynamics/const_pot_bo.cpp
/// @brief Dense / CG BO constant-potential electrode charge solve + ASPC.

#include "atomiverse/cpu/dynamics/const_pot_bo.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/dynamics/charge_packer.h"
#include "atomiverse/cpu/electrodes/charge_aspc.h"
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

/// Dense symmetric positive-definite solve via Cholesky (row-major).
void cholesky_solve(int n, std::vector<double>& A, std::vector<double>& x) {
    // A is overwritten with L (lower); x holds RHS then solution.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[static_cast<std::size_t>(i) * n + j];
            for (int k = 0; k < j; ++k)
                s -= A[static_cast<std::size_t>(i) * n + k]
                     * A[static_cast<std::size_t>(j) * n + k];
            if (i == j) {
                if (s <= 0.0)
                    throw std::runtime_error(
                        "ConstPotBO: elastance matrix not SPD (Cholesky)");
                A[static_cast<std::size_t>(i) * n + j] = std::sqrt(s);
            } else {
                A[static_cast<std::size_t>(i) * n + j] =
                    s / A[static_cast<std::size_t>(j) * n + j];
            }
        }
    }
    // Forward L y = b
    for (int i = 0; i < n; ++i) {
        double s = x[static_cast<std::size_t>(i)];
        for (int k = 0; k < i; ++k)
            s -= A[static_cast<std::size_t>(i) * n + k]
                 * x[static_cast<std::size_t>(k)];
        x[static_cast<std::size_t>(i)] =
            s / A[static_cast<std::size_t>(i) * n + i];
    }
    // Back L^T x = y
    for (int i = n - 1; i >= 0; --i) {
        double s = x[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < n; ++k)
            s -= A[static_cast<std::size_t>(k) * n + i]
                 * x[static_cast<std::size_t>(k)];
        x[static_cast<std::size_t>(i)] =
            s / A[static_cast<std::size_t>(i) * n + i];
    }
}

/// Invert SPD A (row-major) into inv (row-major) via n solves.
void invert_spd(int n, const std::vector<double>& A_in,
                std::vector<double>& inv) {
    inv.assign(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> Awork(A_in);
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

void matvec(const std::vector<double>& A, int n, const std::vector<double>& x,
            std::vector<double>& y) {
    y.assign(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j)
            s += A[static_cast<std::size_t>(i) * n + j]
                 * x[static_cast<std::size_t>(j)];
        y[static_cast<std::size_t>(i)] = s;
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

} // namespace

ConstPotBO::ConstPotBO(double eta_in) : eta(eta_in) {
    if (!(eta > 0.0))
        throw std::invalid_argument("ConstPotBO: eta must be > 0 [1/Å]");
}

double ConstPotBO::capacitance() const {
    const double dpsi = std::abs(delta_psi());
    if (dpsi < 1e-30 || groups_.size() < 2)
        return 0.0;
    // |Q| on the last electrode group (opposite of first under neutrality).
    return std::abs(total_charge(groups_.size() - 1)) / dpsi;
}

ConstPotBO& ConstPotBO::add_electrode(int type, double psi) & {
    groups_.push_back(ElectrodeGroup{type, psi});
    return *this;
}

ConstPotBO ConstPotBO::add_electrode(int type, double psi) && {
    groups_.push_back(ElectrodeGroup{type, psi});
    return std::move(*this);
}

ConstPotBO& ConstPotBO::add_electrode(int type, double psi, double eta_e) & {
    if (!(eta_e > 0.0))
        throw std::invalid_argument(
            "ConstPotBO: per-electrode eta must be > 0 [1/Å]");
    groups_.push_back(ElectrodeGroup{type, psi, eta_e});
    return *this;
}

ConstPotBO ConstPotBO::add_electrode(int type, double psi, double eta_e) && {
    add_electrode(type, psi, eta_e); // lvalue overload validates + pushes
    return std::move(*this);
}

ConstPotBO& ConstPotBO::add_electrode_charge(int type, double q_total) & {
    groups_.push_back(ElectrodeGroup{type, /*psi=*/0.0, /*eta=*/0.0,
                                     /*const_charge=*/true, q_total});
    return *this;
}

ConstPotBO ConstPotBO::add_electrode_charge(int type, double q_total) && {
    groups_.push_back(ElectrodeGroup{type, /*psi=*/0.0, /*eta=*/0.0,
                                     /*const_charge=*/true, q_total});
    return std::move(*this);
}

bool ConstPotBO::has_const_charge_() const {
    for (const auto& g : groups_)
        if (g.const_charge)
            return true;
    return false;
}

double ConstPotBO::electrode_potential(std::size_t ie) const {
    if (ie >= groups_.size())
        throw std::out_of_range("ConstPotBO::electrode_potential");
    if (ie < solved_psi_.size())
        return solved_psi_[ie];
    return groups_[ie].psi;
}

void ConstPotBO::ensure_aspc_() {
    if (predictor_order <= 0) {
        aspc_.reset();
        return;
    }
    if (!aspc_ || aspc_->order() != predictor_order)
        aspc_ = std::make_unique<electrodes::ChargeASPC>(predictor_order);
}

void ConstPotBO::record_aspc_() {
    if (!aspc_ || aspc_record_.empty())
        return;
    // Record the unconstrained CG/dense solution of A Q = (Ψ − b), which is
    // what the next warm-start targets — not the neutrality-projected Q.
    aspc_->record(aspc_record_);
}

void ConstPotBO::build(const BuildContext& ctx) {
    if (groups_.empty())
        throw std::runtime_error(
            "ConstPotBO: add at least one electrode via add_electrode(type, "
            "Psi) or add_electrode_charge(type, Q)");
    if (!(eta > 0.0))
        throw std::invalid_argument("ConstPotBO: eta must be > 0 [1/Å]");
    if (l_tf < 0.0)
        throw std::invalid_argument("ConstPotBO: l_tf must be >= 0 [Å]");
    if (l_tf > 0.0 && (site_density > 0.0) == !site_volume.empty())
        throw std::invalid_argument(
            "ConstPotBO: exactly one of site_density [1/Å³] or per-site "
            "site_volume [Å³] must be set when l_tf > 0");
    collect_electrodes_(ctx.frame);
    if (elec_atoms_.empty())
        throw std::runtime_error(
            "ConstPotBO: no atoms match registered electrode types");
    build_matrix_(ctx.frame);
    matrix_ready_ = true;
    total_Q_.assign(groups_.size(), 0.0);
    ensure_aspc_();
    last_iters_ = 0;
    last_used_predictor_ = false;
}

void ConstPotBO::launch(const LaunchContext& ctx) {
    if (!matrix_ready_ || rebuild_matrix_every_step) {
        collect_electrodes_(ctx.frame);
        build_matrix_(ctx.frame);
        matrix_ready_ = true;
    }
    ensure_aspc_();
    build_rhs_(ctx.frame);
    solve_();
    write_charges_(ctx.frame);
    ChargePacker::forward(ctx);
    if (write_electrode_forces)
        write_electrode_forces_(ctx.frame);
    record_aspc_();
}

void ConstPotBO::destroy(const DestroyContext&) {
    matrix_ready_ = false;
    elec_atoms_.clear();
    A_.clear();
    Q_.clear();
    solved_psi_.clear();
    if (aspc_)
        aspc_->reset();
    last_iters_ = 0;
    last_used_predictor_ = false;
}

double ConstPotBO::total_charge(std::size_t ie) const {
    if (ie >= total_Q_.size())
        throw std::out_of_range("ConstPotBO::total_charge");
    return total_Q_[ie];
}

double ConstPotBO::delta_psi() const {
    if (groups_.size() < 2)
        return 0.0;
    return electrode_potential(groups_.size() - 1) - electrode_potential(0);
}

double ConstPotBO::neutrality_residual() const {
    double s = 0.0;
    for (double q : Q_)
        s += q;
    return std::abs(s);
}

void ConstPotBO::collect_electrodes_(const cpu::Frame& frame) {
    elec_atoms_.clear();
    elec_group_.clear();
    psi_vec_.clear();
    eta_vec_.clear();
    const int n = frame.nlocal();
    auto type = frame.span(kAtom_type);
    for (int i = 0; i < n; ++i) {
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            if (static_cast<int>(type[i]) == groups_[g].type) {
                elec_atoms_.push_back(i);
                elec_group_.push_back(static_cast<int>(g));
                psi_vec_.push_back(groups_[g].const_charge ? 0.0
                                                           : groups_[g].psi);
                // Per-group η_e when registered; shared member otherwise.
                eta_vec_.push_back(groups_[g].eta > 0.0 ? groups_[g].eta : eta);
                break;
            }
        }
    }
    const int m = static_cast<int>(elec_atoms_.size());
    Q_.assign(static_cast<std::size_t>(m), 0.0);
    b_.assign(static_cast<std::size_t>(m), 0.0);
    A_.assign(static_cast<std::size_t>(m) * m, 0.0);
}

double ConstPotBO::pair_A_(double r, double eta_i, double eta_j) const {
    if (r < 1e-14) {
        // Should not be called for i==j self; self uses eta * sqrt(2/π).
        return units::coulomb * eta_i * kSqrt2OverPi;
    }
    const double etaij =
        eta_i * eta_j / std::sqrt(eta_i * eta_i + eta_j * eta_j);
    return units::coulomb * std::erf(etaij * r) / r;
}

double ConstPotBO::pair_b_(double r, double q, double eta_i) const {
    if (r < 1e-14)
        return 0.0;
    return units::coulomb * q * std::erf(eta_i * r) / r;
}

void ConstPotBO::build_matrix_(const cpu::Frame& frame) {
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
        const double eta_a = eta_vec_[static_cast<std::size_t>(a)];
        // Self: k * η_a * sqrt(2/π)
        A_[static_cast<std::size_t>(a) * m + a] =
            units::coulomb * eta_a * kSqrt2OverPi;

        for (int b = a + 1; b < m; ++b) {
            const int ib = elec_atoms_[static_cast<std::size_t>(b)];
            const double eta_b = eta_vec_[static_cast<std::size_t>(b)];
            double aij = 0.0;
            for (int nx = -R; nx <= R; ++nx) {
                for (int ny = -R; ny <= R; ++ny) {
                    // Primary cell: include once; for a==b images only nx,ny≠0
                    // handled in self loop below for a==b.
                    double dx = x[ia] - x[ib] + nx * Lx;
                    double dy = y[ia] - y[ib] + ny * Ly;
                    double dz = z[ia] - z[ib];
                    // When R==0 use min-image for general orthorhombic cells.
                    double r;
                    if (R == 0) {
                        r = min_image_r(box, x[ia] - x[ib], y[ia] - y[ib],
                                        z[ia] - z[ib]);
                    } else {
                        r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    }
                    if (r < 1e-14)
                        continue;
                    aij += pair_A_(r, eta_a, eta_b);
                    if (R == 0)
                        break; // single min-image contrib
                }
                if (R == 0)
                    break;
            }
            A_[static_cast<std::size_t>(a) * m + b] = aij;
            A_[static_cast<std::size_t>(b) * m + a] = aij;
        }

        // Self images (nx,ny) ≠ (0,0) for 2D lattice
        if (R > 0) {
            double self_img = 0.0;
            for (int nx = -R; nx <= R; ++nx) {
                for (int ny = -R; ny <= R; ++ny) {
                    if (nx == 0 && ny == 0)
                        continue;
                    const double r = std::sqrt((nx * Lx) * (nx * Lx)
                                               + (ny * Ly) * (ny * Ly));
                    // Same-site images use η_a/√2 (plate_cap / Gaussian product)
                    self_img +=
                        units::coulomb * std::erf((eta_a / kSqrt2) * r) / r;
                }
            }
            A_[static_cast<std::size_t>(a) * m + a] += self_img;
        }
    }

    // Thomas–Fermi metallicity (Scalfi et al. JCP 153, 174704 (2020)):
    // U_TF^(2) = Σ_i (l_TF² / (2 ε0 v_i)) Q_i²  ⇒  A_ii += l_TF² / (ε0 v_i)
    // with 1/ε0 = 4π k_e and k_e = units::coulomb [eV·Å/e²]. The fixed-
    // density foothold uses v_i = 1/site_density for every site; the
    // MetalWalls `voronoi_volume` companion supplies per-site v_i.
    if (l_tf > 0.0) {
        constexpr double kPi = 3.14159265358979323846;
        const double pref = 4.0 * kPi * units::coulomb * l_tf * l_tf;
        if (!site_volume.empty()) {
            if (site_volume.size() != static_cast<std::size_t>(frame.nlocal()))
                throw std::invalid_argument(
                    "ConstPotBO: site_volume size must equal the frame's "
                    "nlocal (per-site volumes are indexed by frame row)");
            for (int a = 0; a < m; ++a) {
                const double v = site_volume[static_cast<std::size_t>(
                    elec_atoms_[static_cast<std::size_t>(a)])];
                if (!(v > 0.0))
                    throw std::invalid_argument(
                        "ConstPotBO: site_volume must be > 0 [Å³] on every "
                        "electrode row");
                A_[static_cast<std::size_t>(a) * m + a] += pref / v;
            }
        } else {
            const double hardness = pref * site_density;
            for (int a = 0; a < m; ++a)
                A_[static_cast<std::size_t>(a) * m + a] += hardness;
        }
    }
}

void ConstPotBO::build_rhs_(const cpu::Frame& frame) {
    const int m = static_cast<int>(elec_atoms_.size());
    // Reset per-atom base potentials every launch: const-Q entries were
    // overwritten with the solved response after the previous solve.
    for (int a = 0; a < m; ++a) {
        const auto& g = groups_[static_cast<std::size_t>(
            elec_group_[static_cast<std::size_t>(a)])];
        psi_vec_[static_cast<std::size_t>(a)] = g.const_charge ? 0.0 : g.psi;
    }
    const int n = frame.nlocal();
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto q = frame.span(kAtom_charge);
    auto type = frame.span(kAtom_type);
    const cpu::Box& box = frame.box;
    const double Lx = box.length(0);
    const double Ly = box.length(1);
    const int R = xy_image_range < 0 ? 0 : xy_image_range;

    // Mark electrode rows for skip
    std::vector<char> is_elec(static_cast<std::size_t>(n), 0);
    for (int ia : elec_atoms_)
        is_elec[static_cast<std::size_t>(ia)] = 1;

    // Global neutrality target: sum_Q + sum_q_elyt = 0.
    neutrality_target_ = 0.0;
    if (electroneutrality && global_electroneutrality) {
        double q_elyt = 0.0;
        for (int j = 0; j < n; ++j) {
            if (is_elec[static_cast<std::size_t>(j)])
                continue;
            q_elyt += static_cast<double>(q[j]);
        }
        neutrality_target_ = -q_elyt;
    }

    // b stores Φ_electrolyte − E·r so that RHS = Ψ − b = Ψ − Φ_elyt + E·r.
    // Φ_ext = −E·r (U = q Φ_ext = −q E·r). Ex=Ey=Ez=0 → E1 path unchanged.
    const double ex = Ex;
    const double ey = Ey;
    const double ez = Ez;

    std::fill(b_.begin(), b_.end(), 0.0);
    for (int a = 0; a < m; ++a) {
        const int ia = elec_atoms_[static_cast<std::size_t>(a)];
        const double eta_a = eta_vec_[static_cast<std::size_t>(a)];
        double bi = 0.0;
        for (int j = 0; j < n; ++j) {
            if (is_elec[static_cast<std::size_t>(j)])
                continue;
            (void)type; // electrolyte = non-electrode
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
                    bi += pair_b_(r, qj, eta_a);
                    if (R == 0)
                        break;
                }
                if (R == 0)
                    break;
            }
        }
        // Finite-field term: b ← Φ_elyt − E·r  (primary-cell r)
        bi -= ex * static_cast<double>(x[ia]) + ey * static_cast<double>(y[ia])
              + ez * static_cast<double>(z[ia]);
        b_[static_cast<std::size_t>(a)] = bi;
    }
}

int ConstPotBO::cg_solve_(const std::vector<double>& A, int n,
                          const std::vector<double>& b,
                          std::vector<double>& x) const {
    // Classical CG for SPD A. x is the initial guess (in/out).
    std::vector<double> Ax(static_cast<std::size_t>(n));
    std::vector<double> r(static_cast<std::size_t>(n));
    std::vector<double> p(static_cast<std::size_t>(n));
    std::vector<double> Ap(static_cast<std::size_t>(n));

    matvec(A, n, x, Ax);
    for (int i = 0; i < n; ++i)
        r[static_cast<std::size_t>(i)] =
            b[static_cast<std::size_t>(i)] - Ax[static_cast<std::size_t>(i)];
    p = r;
    double rsold = dot(r, r);
    const double bnorm = std::sqrt(std::max(dot(b, b), 1e-300));
    const double tol2 = (cg_tol * bnorm) * (cg_tol * bnorm);

    if (rsold <= tol2)
        return 0;

    const int cap = max_cg_iters > 0 ? max_cg_iters : n * 4;
    for (int it = 0; it < cap; ++it) {
        matvec(A, n, p, Ap);
        const double pAp = dot(p, Ap);
        if (!(std::abs(pAp) > 1e-300))
            throw std::runtime_error("ConstPotBO: CG breakdown (p·Ap ≈ 0)");
        const double alpha = rsold / pAp;
        for (int i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] +=
                alpha * p[static_cast<std::size_t>(i)];
            r[static_cast<std::size_t>(i)] -=
                alpha * Ap[static_cast<std::size_t>(i)];
        }
        const double rsnew = dot(r, r);
        if (rsnew <= tol2)
            return it + 1;
        const double beta = rsnew / rsold;
        for (int i = 0; i < n; ++i)
            p[static_cast<std::size_t>(i)] =
                r[static_cast<std::size_t>(i)]
                + beta * p[static_cast<std::size_t>(i)];
        rsold = rsnew;
    }
    throw std::runtime_error(
        "ConstPotBO: CG failed to converge within max_cg_iters");
}

void ConstPotBO::apply_inverse_(const std::vector<double>& v,
                                std::vector<double>& y) {
    const int m = static_cast<int>(elec_atoms_.size());
    if (use_cg) {
        y.assign(static_cast<std::size_t>(m), 0.0);
        (void)cg_solve_(A_, m, v, y);
    } else {
        A_work_ = A_;
        y = v;
        cholesky_solve(m, A_work_, y);
    }
}

void ConstPotBO::constrain_const_q_(std::vector<double>& Q0) {
    // LAMMPS `electrode/conq`: unknown uniform group potentials ψ_e so that
    // d_e·Q = Q_e with Q = Q0 + Σ_e ψ_e u_e,  u_e = A^{-1} d_e.
    const int m = static_cast<int>(elec_atoms_.size());
    std::vector<int> cq_groups;
    for (std::size_t g = 0; g < groups_.size(); ++g)
        if (groups_[g].const_charge)
            cq_groups.push_back(static_cast<int>(g));
    const int nq = static_cast<int>(cq_groups.size());
    if (nq == 0)
        return;

    // u_e = A^{-1} d_e per const-Q group.
    std::vector<std::vector<double>> u(static_cast<std::size_t>(nq));
    std::vector<double> d(static_cast<std::size_t>(m));
    for (int e = 0; e < nq; ++e) {
        std::fill(d.begin(), d.end(), 0.0);
        for (int a = 0; a < m; ++a)
            if (elec_group_[static_cast<std::size_t>(a)]
                == cq_groups[static_cast<std::size_t>(e)])
                d[static_cast<std::size_t>(a)] = 1.0;
        apply_inverse_(d, u[static_cast<std::size_t>(e)]);
    }

    // Small SPD system M ψ = r:  M_ef = d_e·u_f,  r_e = Q_e − d_e·Q0.
    std::vector<double> M(static_cast<std::size_t>(nq) * nq, 0.0);
    std::vector<double> psi(static_cast<std::size_t>(nq), 0.0);
    for (int e = 0; e < nq; ++e) {
        const int ge = cq_groups[static_cast<std::size_t>(e)];
        double dQ0 = 0.0;
        for (int a = 0; a < m; ++a)
            if (elec_group_[static_cast<std::size_t>(a)] == ge)
                dQ0 += Q0[static_cast<std::size_t>(a)];
        psi[static_cast<std::size_t>(e)] =
            groups_[static_cast<std::size_t>(ge)].q_target - dQ0;
        for (int f = 0; f < nq; ++f) {
            double s = 0.0;
            for (int a = 0; a < m; ++a)
                if (elec_group_[static_cast<std::size_t>(a)] == ge)
                    s += u[static_cast<std::size_t>(f)]
                          [static_cast<std::size_t>(a)];
            M[static_cast<std::size_t>(e) * nq + f] = s;
        }
    }
    cholesky_solve(nq, M, psi); // M is a Gram matrix of A^{-1} — SPD

    // Q = Q0 + Σ ψ_e u_e; record solved potentials for readout + residual.
    for (int e = 0; e < nq; ++e) {
        const int ge = cq_groups[static_cast<std::size_t>(e)];
        const double pe = psi[static_cast<std::size_t>(e)];
        for (int a = 0; a < m; ++a)
            Q0[static_cast<std::size_t>(a)] +=
                pe
                * u[static_cast<std::size_t>(e)][static_cast<std::size_t>(a)];
        solved_psi_[static_cast<std::size_t>(ge)] = pe;
        for (int a = 0; a < m; ++a)
            if (elec_group_[static_cast<std::size_t>(a)] == ge)
                psi_vec_[static_cast<std::size_t>(a)] = pe;
    }
}

void ConstPotBO::solve_dense_() {
    const int m = static_cast<int>(elec_atoms_.size());
    // RHS: Ψ - b  with b = Φ_elyt − E·r  (field folded in build_rhs_)
    std::vector<double> rhs(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i)
        rhs[static_cast<std::size_t>(i)] = psi_vec_[static_cast<std::size_t>(i)]
                                           - b_[static_cast<std::size_t>(i)];

    // Unconstrained A Q = rhs (also the ASPC record target).
    A_work_ = A_;
    std::vector<double> Q_unconst = rhs;
    cholesky_solve(m, A_work_, Q_unconst);
    aspc_record_ = Q_unconst;

    if (has_const_charge_()) {
        // Constant-charge control: group totals are prescribed; the
        // electroneutrality projection is superseded by the constraints.
        constrain_const_q_(Q_unconst);
        Q_ = Q_unconst;
        last_iters_ = 0;
        last_used_predictor_ = false;
        return;
    }

    if (electroneutrality && m >= 1) {
        // Q = Q_unconst − λ A^{-1} e so that e·Q = target
        // (target 0 electrode-only; −Q_elyt for global neutrality).
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
                "ConstPotBO: singular neutrality projector");
        const double scale = (eT_Q - neutrality_target_) / eT_u;
        for (int i = 0; i < m; ++i)
            Q_[static_cast<std::size_t>(i)] =
                Q_unconst[static_cast<std::size_t>(i)]
                - scale * u[static_cast<std::size_t>(i)];
    } else {
        Q_ = Q_unconst;
    }
    last_iters_ = 0;
    last_used_predictor_ = false;
}

void ConstPotBO::solve_cg_() {
    const int m = static_cast<int>(elec_atoms_.size());
    std::vector<double> rhs(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i)
        rhs[static_cast<std::size_t>(i)] = psi_vec_[static_cast<std::size_t>(i)]
                                           - b_[static_cast<std::size_t>(i)];

    // Warm-start unconstrained CG from ASPC (or zero).
    std::vector<double> Q0(static_cast<std::size_t>(m), 0.0);
    last_used_predictor_ = false;
    if (aspc_) {
        std::vector<double> Qpred;
        if (aspc_->predict(Qpred) && static_cast<int>(Qpred.size()) == m) {
            Q0 = Qpred;
            last_used_predictor_ = true;
        }
    }

    const int iters_main = cg_solve_(A_, m, rhs, Q0);
    aspc_record_ = Q0; // unconstrained solution for next ASPC record

    if (has_const_charge_()) {
        constrain_const_q_(Q0);
        Q_ = Q0;
        last_iters_ = iters_main;
        return;
    }

    if (electroneutrality && m >= 1) {
        // A u = 1,  λ = (1·Q0 − target)/(1·u),  Q = Q0 − λ u
        std::vector<double> ones(static_cast<std::size_t>(m), 1.0);
        std::vector<double> u(static_cast<std::size_t>(m), 0.0);
        (void)cg_solve_(A_, m, ones, u);
        double one_dot_Q0 = 0.0, one_dot_u = 0.0;
        for (int i = 0; i < m; ++i) {
            one_dot_Q0 += Q0[static_cast<std::size_t>(i)];
            one_dot_u += u[static_cast<std::size_t>(i)];
        }
        if (!(std::abs(one_dot_u) > 1e-30))
            throw std::runtime_error(
                "ConstPotBO: singular neutrality row (CG)");
        const double lambda = (one_dot_Q0 - neutrality_target_) / one_dot_u;
        for (int i = 0; i < m; ++i)
            Q_[static_cast<std::size_t>(i)] =
                Q0[static_cast<std::size_t>(i)]
                - lambda * u[static_cast<std::size_t>(i)];
    } else {
        Q_ = Q0;
    }
    last_iters_ = iters_main;
}

void ConstPotBO::solve_() {
    const int m = static_cast<int>(elec_atoms_.size());
    if (m == 0)
        return;

    // Per-group potential readout: applied for const-Ψ; const-Q entries are
    // overwritten with the solved response inside constrain_const_q_.
    solved_psi_.assign(groups_.size(), 0.0);
    for (std::size_t g = 0; g < groups_.size(); ++g)
        solved_psi_[g] = groups_[g].const_charge ? 0.0 : groups_[g].psi;

    // neutrality_target_ set in build_rhs_ (0 or −Q_elyt).

    if (use_cg)
        solve_cg_();
    else
        solve_dense_();

    // Residual of unconstrained equation AQ − (Ψ − b)
    std::vector<double> rhs(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i)
        rhs[static_cast<std::size_t>(i)] = psi_vec_[static_cast<std::size_t>(i)]
                                           - b_[static_cast<std::size_t>(i)];
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

void ConstPotBO::write_charges_(cpu::Frame& frame) const {
    auto q = frame.span(kAtom_charge);
    // Zero electrode charges first (only electrode sites we own).
    for (std::size_t a = 0; a < elec_atoms_.size(); ++a)
        q[elec_atoms_[a]] = static_cast<StorageReal>(Q_[a]);
}

double ConstPotBO::electrolyte_charge_(const cpu::Frame& frame) const {
    const int n = frame.nlocal();
    auto q = frame.span(kAtom_charge);
    std::vector<char> is_elec(static_cast<std::size_t>(n), 0);
    for (int ia : elec_atoms_)
        is_elec[static_cast<std::size_t>(ia)] = 1;
    double s = 0.0;
    for (int j = 0; j < n; ++j) {
        if (is_elec[static_cast<std::size_t>(j)])
            continue;
        s += static_cast<double>(q[j]);
    }
    return s;
}

void ConstPotBO::write_electrode_forces_(cpu::Frame& frame) const {
    // Fixed-Q Hellmann–Feynman: F_i = −Q_i ∇_i Φ_elyt from point electrolyte.
    // Φ uses the same Gaussian–point kernel as pair_b_ (∂/∂r of erfc-style
    // interaction). Electrode–electrode A-geometry force deferred (frozen
    // walls / rebuild_matrix_every_step path).
    const int m = static_cast<int>(elec_atoms_.size());
    if (m == 0)
        return;
    const int n = frame.nlocal();
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto q = frame.span(kAtom_charge);
    auto fx = frame.span(kAtom_fx);
    auto fy = frame.span(kAtom_fy);
    auto fz = frame.span(kAtom_fz);
    const cpu::Box& box = frame.box;

    std::vector<char> is_elec(static_cast<std::size_t>(n), 0);
    for (int ia : elec_atoms_)
        is_elec[static_cast<std::size_t>(ia)] = 1;

    // d(pair_b)/dr of the SAME kernel the RHS used:
    // pair_b_ = coulomb * q * erf(α r) / r with α = η_i (Gaussian–point).
    // dΦ/dr = coulomb * q * [ (2α/√π) exp(-(αr)²)/r − erf(αr)/r² ]
    // (Was α = η/√2 — the gradient of a Gaussian–Gaussian kernel the solve
    // never used; fixed to match pair_b_ per iron law, per-atom η_i.)
    constexpr double kSqrtPi = 1.7724538509055160273;

    for (int a = 0; a < m; ++a) {
        const int ia = elec_atoms_[static_cast<std::size_t>(a)];
        const double alpha = eta_vec_[static_cast<std::size_t>(a)];
        const double Qi = Q_[static_cast<std::size_t>(a)];
        if (Qi == 0.0)
            continue;
        double Ex = 0.0, Ey = 0.0, Ez = 0.0; // E = −∇Φ so F = Q E
        for (int j = 0; j < n; ++j) {
            if (is_elec[static_cast<std::size_t>(j)])
                continue;
            const double qj = static_cast<double>(q[j]);
            if (qj == 0.0)
                continue;
            double dx = static_cast<double>(x[ia] - x[j]);
            double dy = static_cast<double>(y[ia] - y[j]);
            double dz = static_cast<double>(z[ia] - z[j]);
            box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-30)
                continue;
            const double r = std::sqrt(r2);
            const double ar = alpha * r;
            const double erf_ar = std::erf(ar);
            // dΦ/dr with Φ = k q erf(αr)/r
            const double dPhi_dr =
                units::coulomb * qj
                * (2.0 * alpha / kSqrtPi * std::exp(-ar * ar) / r
                   - erf_ar / (r * r));
            // E = −∇Φ = −(dΦ/dr)(r_hat); force on electrode F = Q E
            // r_hat points from j to i; dPhi/dr along that direction
            const double inv_r = 1.0 / r;
            Ex -= dPhi_dr * dx * inv_r;
            Ey -= dPhi_dr * dy * inv_r;
            Ez -= dPhi_dr * dz * inv_r;
        }
        fx[ia] += static_cast<AccReal>(Qi * Ex);
        fy[ia] += static_cast<AccReal>(Qi * Ey);
        fz[ia] += static_cast<AccReal>(Qi * Ez);
    }
}

double ConstPotBO::matrix_condition() const {
    const int m = static_cast<int>(elec_atoms_.size());
    if (m <= 0 || A_.empty())
        return 0.0;
    // Power iteration for λ_max, inverse-free estimate of λ_min via
    // Rayleigh on residual of Cholesky — use Gershgorin fallback for
    // foothold: κ ≈ max_i A_ii / min_i A_ii for diagonally dominant A.
    double amin = A_[0], amax = A_[0];
    for (int i = 0; i < m; ++i) {
        const double aii = A_[static_cast<std::size_t>(i) * m + i];
        amin = std::min(amin, aii);
        amax = std::max(amax, aii);
    }
    if (!(amin > 0.0))
        return 1e300;
    return amax / amin;
}

} // namespace md
} // namespace cpu
} // namespace ATV
