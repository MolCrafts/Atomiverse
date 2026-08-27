/// @file cpu/dynamics/qeq.cpp
/// @brief QEq / fluctuating-charge equilibration (LAMMPS `fix qeq/point` twin).

#include "atomiverse/cpu/dynamics/qeq.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/dynamics/charge_packer.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ATV {
namespace cpu {
namespace md {

namespace {

/// Dense symmetric positive-definite solve via Cholesky (row-major), the
/// same private per-TU pattern as the const-potential electrode solvers.
/// A is overwritten with L (lower); x holds RHS then solution.
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
                        "QEq: hardness matrix not SPD (Cholesky); eta too "
                        "small vs 1/r coupling");
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

/// Forward/back substitution reusing an already-computed factor L in A.
void cholesky_substitute(int n, const std::vector<double>& L,
                         std::vector<double>& x) {
    for (int i = 0; i < n; ++i) {
        double s = x[static_cast<std::size_t>(i)];
        for (int k = 0; k < i; ++k)
            s -= L[static_cast<std::size_t>(i) * n + k]
                 * x[static_cast<std::size_t>(k)];
        x[static_cast<std::size_t>(i)] =
            s / L[static_cast<std::size_t>(i) * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = x[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < n; ++k)
            s -= L[static_cast<std::size_t>(k) * n + i]
                 * x[static_cast<std::size_t>(k)];
        x[static_cast<std::size_t>(i)] =
            s / L[static_cast<std::size_t>(i) * n + i];
    }
}

} // namespace

QEq::QEq(double cutoff_in) : cutoff(cutoff_in) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("QEq: cutoff must be > 0 [Å]");
}

QEq& QEq::set_type(int type, double chi, double eta) & {
    if (!(eta > 0.0))
        throw std::invalid_argument("QEq: eta must be > 0 [eV/e²]");
    for (auto& p : params_) {
        if (p.type == type) {
            p.chi = chi;
            p.eta = eta;
            return *this;
        }
    }
    params_.push_back(TypeParam{type, chi, eta});
    return *this;
}

QEq QEq::set_type(int type, double chi, double eta) && {
    set_type(type, chi, eta);
    return std::move(*this);
}

void QEq::build(const BuildContext& ctx) {
    if (params_.empty())
        throw std::runtime_error(
            "QEq: register per-type parameters via set_type(type, chi, eta)");
    if (!(cutoff > 0.0))
        throw std::invalid_argument("QEq: cutoff must be > 0 [Å]");
    collect_params_(ctx.frame);
}

void QEq::launch(const LaunchContext& ctx) {
    collect_params_(ctx.frame);
    build_matrix_(ctx.frame);
    solve_and_write_(ctx.frame);
    ChargePacker::forward(ctx);
}

void QEq::destroy(const DestroyContext&) {
    chi_i_.clear();
    eta_i_.clear();
    H_.clear();
    q_.clear();
    energy_ = 0.0;
}

double QEq::net_charge() const {
    double s = 0.0;
    for (double qi : q_)
        s += qi;
    return s;
}

void QEq::collect_params_(const cpu::Frame& frame) {
    const int n = frame.nlocal();
    auto type = frame.span(kAtom_type);
    chi_i_.assign(static_cast<std::size_t>(n), 0.0);
    eta_i_.assign(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        const int ti = static_cast<int>(type[i]);
        const TypeParam* found = nullptr;
        for (const auto& p : params_) {
            if (p.type == ti) {
                found = &p;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(
                "QEq: no set_type parameters for atom type "
                + std::to_string(ti) + " (no fallback)");
        chi_i_[static_cast<std::size_t>(i)] = found->chi;
        eta_i_[static_cast<std::size_t>(i)] = found->eta;
    }
}

void QEq::build_matrix_(const cpu::Frame& frame) {
    const int n = frame.nlocal();
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    const cpu::Box& box = frame.box;
    const double rc_sq = cutoff * cutoff;

    H_.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        H_[static_cast<std::size_t>(i) * n + i] =
            eta_i_[static_cast<std::size_t>(i)];
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = static_cast<double>(x[j]) - static_cast<double>(x[i]);
            double dy = static_cast<double>(y[j]) - static_cast<double>(y[i]);
            double dz = static_cast<double>(z[j]) - static_cast<double>(z[i]);
            box.min_image(dx, dy, dz);
            const double r_sq = dx * dx + dy * dy + dz * dz;
            if (r_sq > rc_sq)
                continue;
            // LAMMPS fix qeq/point kernel: bare 1/r, no qqrd2e prefactor.
            const double inv_r = 1.0 / std::sqrt(r_sq);
            H_[static_cast<std::size_t>(i) * n + j] = inv_r;
            H_[static_cast<std::size_t>(j) * n + i] = inv_r;
        }
    }
}

void QEq::solve_and_write_(cpu::Frame& frame) {
    const int n = frame.nlocal();
    q_.assign(static_cast<std::size_t>(n), 0.0);
    if (n == 0) {
        energy_ = 0.0;
        return;
    }

    // Factor once (H overwritten with L), substitute the two RHS:
    //   H s = -chi,  H t = -1  →  q = s − ((Σs − qtot)/Σt)·t.
    std::vector<double> L(H_);
    std::vector<double> s(chi_i_);
    for (double& v : s)
        v = -v;
    cholesky_solve(n, L, s); // L now holds the factor.
    std::vector<double> t(static_cast<std::size_t>(n), -1.0);
    cholesky_substitute(n, L, t);

    double s_sum = 0.0;
    double t_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        s_sum += s[static_cast<std::size_t>(i)];
        t_sum += t[static_cast<std::size_t>(i)];
    }
    const double u = (s_sum - qtot) / t_sum;
    for (int i = 0; i < n; ++i)
        q_[static_cast<std::size_t>(i)] =
            s[static_cast<std::size_t>(i)] - u * t[static_cast<std::size_t>(i)];

    // QEq objective at the solution: Σχq + ½ qᵀHq (bare-1/r kernel).
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        double hq = 0.0;
        for (int j = 0; j < n; ++j)
            hq += H_[static_cast<std::size_t>(i) * n + j]
                  * q_[static_cast<std::size_t>(j)];
        e += chi_i_[static_cast<std::size_t>(i)]
                 * q_[static_cast<std::size_t>(i)]
             + 0.5 * q_[static_cast<std::size_t>(i)] * hq;
    }
    energy_ = e;

    auto charge = frame.span(kAtom_charge);
    for (int i = 0; i < n; ++i)
        charge[i] = q_[static_cast<std::size_t>(i)];
}

} // namespace md
} // namespace cpu
} // namespace ATV
