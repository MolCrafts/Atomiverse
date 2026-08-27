/// @file cpu/dynamics/const_potential.cpp
/// @brief ConstPotentialBO — dense BO electrode charge solve with neutrality.

#include "atomiverse/cpu/dynamics/const_potential.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/dynamics/charge_packer.h"
#include "atomiverse/cpu/frame/box.h"

namespace ATV {
namespace cpu {
namespace {

/// Dense lower Cholesky of SPD matrix `a` (n×n row-major). Returns false on fail.
bool cholesky_lower(std::vector<double>& a, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = a[static_cast<std::size_t>(i * n + j)];
            for (int k = 0; k < j; ++k)
                s -= a[static_cast<std::size_t>(i * n + k)]
                     * a[static_cast<std::size_t>(j * n + k)];
            if (i == j) {
                if (s <= 0.0)
                    return false;
                a[static_cast<std::size_t>(i * n + j)] = std::sqrt(s);
            } else {
                a[static_cast<std::size_t>(i * n + j)] =
                    s / a[static_cast<std::size_t>(j * n + j)];
            }
        }
        for (int j = i + 1; j < n; ++j)
            a[static_cast<std::size_t>(i * n + j)] = 0.0;
    }
    return true;
}

void chol_solve(const std::vector<double>& L, int n,
                const std::vector<double>& b, std::vector<double>& x) {
    // Ly = b
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        double s = b[static_cast<std::size_t>(i)];
        for (int k = 0; k < i; ++k)
            s -= L[static_cast<std::size_t>(i * n + k)]
                 * y[static_cast<std::size_t>(k)];
        y[static_cast<std::size_t>(i)] =
            s / L[static_cast<std::size_t>(i * n + i)];
    }
    // L^T x = y
    x.resize(static_cast<std::size_t>(n));
    for (int i = n - 1; i >= 0; --i) {
        double s = y[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < n; ++k)
            s -= L[static_cast<std::size_t>(k * n + i)]
                 * x[static_cast<std::size_t>(k)];
        x[static_cast<std::size_t>(i)] =
            s / L[static_cast<std::size_t>(i * n + i)];
    }
}

inline double erf_over_r(double r, double w) {
    // erf(r / (w √2)) / r  — continuous at r→0 → √(2/π)/w
    constexpr double inv_sqrt2 = 0.7071067811865476;
    if (r < 1e-14)
        return std::sqrt(2.0 / 3.14159265358979323846) / w;
    return std::erf(r * inv_sqrt2 / w) / r;
}

} // namespace

ConstPotentialBO::ConstPotentialBO(std::vector<int> electrode_types,
                                   std::vector<double> potentials_V,
                                   double gaussian_width_A)
    : electrode_types_(std::move(electrode_types)),
      potentials_(std::move(potentials_V)), w_(gaussian_width_A) {
    if (electrode_types_.empty())
        throw std::invalid_argument("ConstPotentialBO: need ≥1 electrode type");
    if (electrode_types_.size() != potentials_.size())
        throw std::invalid_argument(
            "ConstPotentialBO: types and potentials size mismatch");
    if (!(w_ > 0.0))
        throw std::invalid_argument(
            "ConstPotentialBO: gaussian width must be > 0");
    Q_per_electrode_.assign(electrode_types_.size(), 0.0);
}

void ConstPotentialBO::build(const BuildContext& ctx) {
    const int n = owned_count(ctx);
    const auto type = ctx.frame.span(kAtom_type);

    elec_idx_.clear();
    elec_id_.clear();
    for (int i = 0; i < n; ++i) {
        for (int e = 0; e < static_cast<int>(electrode_types_.size()); ++e) {
            if (type[i] == electrode_types_[static_cast<std::size_t>(e)]) {
                elec_idx_.push_back(i);
                elec_id_.push_back(e);
                break;
            }
        }
    }
    n_elec_ = static_cast<int>(elec_idx_.size());
    if (n_elec_ == 0)
        throw std::runtime_error("ConstPotentialBO: no electrode atoms found");

    const std::size_t nn =
        static_cast<std::size_t>(n_elec_) * static_cast<std::size_t>(n_elec_);
    A_.assign(nn, 0.0);
    L_.assign(nn, 0.0);
    b_.assign(static_cast<std::size_t>(n_elec_), 0.0);
    q_.assign(static_cast<std::size_t>(n_elec_), 0.0);

    const int naug = n_elec_ + 1;
    Aaug_.assign(static_cast<std::size_t>(naug * naug), 0.0);
    Laug_.assign(static_cast<std::size_t>(naug * naug), 0.0);
    baug_.assign(static_cast<std::size_t>(naug), 0.0);
    xaug_.assign(static_cast<std::size_t>(naug), 0.0);

    rebuild_A_(ctx.frame);
    a_built_ = true;

    // Seed electrode charges for the first force evaluation before launch.
    solve_(ctx.frame);
}

void ConstPotentialBO::rebuild_A_(const Frame& frame) {
    const auto x = frame.span(kAtom_x);
    const auto y = frame.span(kAtom_y);
    const auto z = frame.span(kAtom_z);
    const double ke = units::coulomb;
    const double aii = ke * std::sqrt(2.0 / 3.14159265358979323846) / w_;

    for (int i = 0; i < n_elec_; ++i) {
        const int ii = elec_idx_[static_cast<std::size_t>(i)];
        A_[static_cast<std::size_t>(i * n_elec_ + i)] = aii;
        for (int j = i + 1; j < n_elec_; ++j) {
            const int jj = elec_idx_[static_cast<std::size_t>(j)];
            double dx = static_cast<double>(x[ii] - x[jj]);
            double dy = static_cast<double>(y[ii] - y[jj]);
            double dz = static_cast<double>(z[ii] - z[jj]);
            frame.box.min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double aij = ke * erf_over_r(r, w_);
            A_[static_cast<std::size_t>(i * n_elec_ + j)] = aij;
            A_[static_cast<std::size_t>(j * n_elec_ + i)] = aij;
        }
    }
}

void ConstPotentialBO::solve_(Frame& frame) {
    if (n_elec_ == 0)
        return;

    // Electrode positions are frozen in the E1 product, so A is fixed after build.
    // (Recompute if a future piston path moves electrodes.)
    if (!a_built_)
        rebuild_A_(frame);

    const int n_own = frame.nlocal();
    const auto type = frame.span(kAtom_type);
    const auto qcol = frame.span(kAtom_charge);
    const auto x = frame.span(kAtom_x);
    const auto y = frame.span(kAtom_y);
    const auto z = frame.span(kAtom_z);
    const double ke = units::coulomb;

    // Mark electrode rows for Φ skip.
    std::vector<char> is_elec(static_cast<std::size_t>(n_own), 0);
    for (int k = 0; k < n_elec_; ++k)
        is_elec[static_cast<std::size_t>(
            elec_idx_[static_cast<std::size_t>(k)])] = 1;

    // Electrolyte net charge (for global neutrality constraint).
    double q_electrolyte = 0.0;
    for (int j = 0; j < n_own; ++j) {
        if (!is_elec[static_cast<std::size_t>(j)])
            q_electrolyte += static_cast<double>(qcol[j]);
    }

    // b_i = Ψ_{e(i)} − Φ_i^{electrolyte}
    for (int i = 0; i < n_elec_; ++i) {
        const int ii = elec_idx_[static_cast<std::size_t>(i)];
        const int e = elec_id_[static_cast<std::size_t>(i)];
        double phi = 0.0;
        for (int j = 0; j < n_own; ++j) {
            if (is_elec[static_cast<std::size_t>(j)])
                continue;
            const double qj = static_cast<double>(qcol[j]);
            if (qj == 0.0)
                continue;
            double dx = static_cast<double>(x[ii] - x[j]);
            double dy = static_cast<double>(y[ii] - y[j]);
            double dz = static_cast<double>(z[ii] - z[j]);
            frame.box.min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12)
                continue;
            // Point charge electrolyte → Gaussian electrode: erf form
            phi += ke * qj * erf_over_r(r, w_);
        }
        b_[static_cast<std::size_t>(i)] =
            potentials_[static_cast<std::size_t>(e)] - phi;
    }

    // Augmented system for neutrality: Σ Q = −q_electrolyte
    // [ A  1 ] [Q]   [b]
    // [ 1ᵀ 0 ] [λ] = [−q_elec]
    //
    // A is SPD but the saddle-point matrix is indefinite — use a bordered
    // solve: A Q0 = b, A u = 1, then λ = (1ᵀ Q0 + q_elec) / (1ᵀ u), Q = Q0 − λ u.
    std::vector<double> ones(static_cast<std::size_t>(n_elec_), 1.0);
    L_ = A_;
    if (!cholesky_lower(L_, n_elec_))
        throw std::runtime_error(
            "ConstPotentialBO: A not SPD (Cholesky failed)");

    std::vector<double> Q0, u;
    chol_solve(L_, n_elec_, b_, Q0);
    chol_solve(L_, n_elec_, ones, u);

    double one_dot_Q0 = 0.0, one_dot_u = 0.0;
    for (int i = 0; i < n_elec_; ++i) {
        one_dot_Q0 += Q0[static_cast<std::size_t>(i)];
        one_dot_u += u[static_cast<std::size_t>(i)];
    }
    if (!(std::fabs(one_dot_u) > 1e-30))
        throw std::runtime_error("ConstPotentialBO: singular neutrality row");

    const double lambda = (one_dot_Q0 + q_electrolyte) / one_dot_u;
    double sum_q = 0.0;
    for (int i = 0; i < n_elec_; ++i) {
        q_[static_cast<std::size_t>(i)] =
            Q0[static_cast<std::size_t>(i)]
            - lambda * u[static_cast<std::size_t>(i)];
        sum_q += q_[static_cast<std::size_t>(i)];
    }
    neutrality_residual_ = std::fabs(sum_q + q_electrolyte);

    // Write charges + per-electrode totals.
    Q_per_electrode_.assign(electrode_types_.size(), 0.0);
    auto qwrite = frame.span(kAtom_charge);
    for (int i = 0; i < n_elec_; ++i) {
        const int ii = elec_idx_[static_cast<std::size_t>(i)];
        const int e = elec_id_[static_cast<std::size_t>(i)];
        qwrite[ii] = static_cast<StorageReal>(q_[static_cast<std::size_t>(i)]);
        Q_per_electrode_[static_cast<std::size_t>(e)] +=
            q_[static_cast<std::size_t>(i)];
    }
}

void ConstPotentialBO::launch(const LaunchContext& ctx) {
    if (!ctx.at(ATV::md::Phase::pre_force))
        return;
    solve_(ctx.frame);
    ChargePacker::forward(ctx);
}

void ConstPotentialBO::destroy(const DestroyContext& /*ctx*/) {
    a_built_ = false;
    n_elec_ = 0;
    elec_idx_.clear();
    elec_id_.clear();
}

double ConstPotentialBO::total_charge(int electrode) const {
    if (electrode < 0 || electrode >= static_cast<int>(Q_per_electrode_.size()))
        throw std::out_of_range("ConstPotentialBO::total_charge");
    return Q_per_electrode_[static_cast<std::size_t>(electrode)];
}

} // namespace cpu
} // namespace ATV
