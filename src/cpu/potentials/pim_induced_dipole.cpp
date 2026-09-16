/// @file src/cpu/potentials/pim_induced_dipole.cpp
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace {

/// Thole scale u = a r / (α_i α_j)^{1/6}.
inline double thole_u(double r, double ai, double aj, double a) {
    if (!(a > 0.0) || !(ai > 0.0) || !(aj > 0.0) || !(r > 0.0))
        return 0.0;
    return a * r / std::pow(ai * aj, 1.0 / 6.0);
}

/// Exponential Thole screening factors λ3, λ5, λ7 (van Duijnen / Swart).
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

/// Tang–Toennies f_n(x) = 1 − e^{−x} Σ_{k=0}^n x^k / k!
inline double tang_toennies(int n, double x) {
    if (!(x > 0.0))
        return 1.0;
    double term = 1.0;
    double sum = 1.0;
    for (int k = 1; k <= n; ++k) {
        term *= x / static_cast<double>(k);
        sum += term;
    }
    return 1.0 - std::exp(-x) * sum;
}

/// Combined multipole screening: Thole λ and optional TT f_n.
struct Damp {
    double l3 = 1, l5 = 1, l7 = 1;
    double f1 = 1, f3 = 1, f5 = 1, f7 = 1;
};

inline Damp make_damp(double r, double ai, double aj, double thole_a,
                      double tt_b) {
    Damp d;
    if (thole_a > 0.0 && ai > 0.0 && aj > 0.0) {
        thole_lambda(thole_u(r, ai, aj, thole_a), d.l3, d.l5, d.l7);
    }
    if (tt_b > 0.0) {
        const double x = tt_b * r;
        d.f1 = tang_toennies(1, x);
        d.f3 = tang_toennies(3, x);
        d.f5 = tang_toennies(5, x);
        d.f7 = tang_toennies(7, x);
    }
    return d;
}

inline double s3(const Damp& d) { return d.l3 * d.f3; }
inline double s5(const Damp& d) { return d.l5 * d.f5; }
inline double s7(const Damp& d) { return d.l7 * d.f7; }

} // namespace

PimInducedDipole::PimInducedDipole(double cutoff_,
                                   std::vector<double> alpha_by_type_,
                                   double thole_a_, double pol_tol_,
                                   int pol_max_iter_)
    : cutoff(cutoff_), thole_a(thole_a_), pol_tol(pol_tol_),
      pol_max_iter(pol_max_iter_), alpha_by_type(std::move(alpha_by_type_)) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PimInducedDipole: cutoff must be > 0");
    if (!(thole_a >= 0.0))
        throw std::invalid_argument("PimInducedDipole: thole_a must be >= 0");
    if (!(pol_tol > 0.0) || pol_max_iter < 1)
        throw std::invalid_argument("PimInducedDipole: invalid SCC params");
    for (double a : alpha_by_type) {
        if (!(a >= 0.0) || !std::isfinite(a))
            throw std::invalid_argument(
                "PimInducedDipole: alpha_by_type entries must be >= 0");
    }
}

double PimInducedDipole::alpha_of(int type) const {
    if (type < 0 || static_cast<std::size_t>(type) >= alpha_by_type.size())
        return 0.0;
    return alpha_by_type[static_cast<std::size_t>(type)];
}

void PimInducedDipole::build(const BuildContext&) {}

void PimInducedDipole::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = f.nlocal();
    if (n <= 0) {
        e_pol_ = 0;
        last_mu_.clear();
        mu_guess.clear();
        last_pol_iters_ = 0;
        last_pol_converged_ = true;
        return;
    }

    const StorageReal* x = f.span(kAtom_x);
    const StorageReal* y = f.span(kAtom_y);
    const StorageReal* z = f.span(kAtom_z);
    const auto charge = f.span(kAtom_charge);
    const auto type = f.span(kAtom_type);
    AccReal* fx = f.span(kAtom_fx);
    AccReal* fy = f.span(kAtom_fy);
    AccReal* fz = f.span(kAtom_fz);

    std::vector<double> alpha(static_cast<std::size_t>(n), 0.0);
    bool any_pol = false;
    for (int i = 0; i < n; ++i) {
        alpha[static_cast<std::size_t>(i)] =
            alpha_of(static_cast<int>(type[i]));
        if (alpha[static_cast<std::size_t>(i)] > 0.0)
            any_pol = true;
    }

    if (!any_pol) {
        e_pol_ = 0;
        last_mu_.assign(static_cast<std::size_t>(3 * n), 0.0);
        mu_guess.clear();
        last_pol_iters_ = 0;
        last_pol_converged_ = true;
        return;
    }

    const double rc2 = cutoff * cutoff;
    const double k = units::coulomb;

    // Permanent field E^perm [e/Å²] (Gaussian electrostatics, damped).
    std::vector<double> e_perm(static_cast<std::size_t>(3 * n), 0.0);
    for (int i = 0; i < n; ++i) {
        const double ai = alpha[static_cast<std::size_t>(i)];
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            f.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc2 || !(r2 > 0.0))
                continue;
            const double r = std::sqrt(r2);
            const double aj = alpha[static_cast<std::size_t>(j)];
            const Damp d = make_damp(r, ai, aj, thole_a, tt_b);
            const double fac = s3(d) * charge[j] / (r2 * r);
            e_perm[static_cast<std::size_t>(3 * i + 0)] += fac * dx;
            e_perm[static_cast<std::size_t>(3 * i + 1)] += fac * dy;
            e_perm[static_cast<std::size_t>(3 * i + 2)] += fac * dz;
        }
    }

    // SCC: μ = α (E^perm + T·μ) — warm-started from mu_guess when supplied
    // (extended-Lagrangian / predictor owners), cold from μ = 0 otherwise.
    if (mu_guess.size() == static_cast<std::size_t>(3 * n)) {
        last_mu_ = mu_guess;
    } else {
        last_mu_.assign(static_cast<std::size_t>(3 * n), 0.0);
    }
    mu_guess.clear();
    std::vector<double> mu_new(static_cast<std::size_t>(3 * n), 0.0);
    std::vector<double> e_tot(static_cast<std::size_t>(3 * n), 0.0);

    int it = 0;
    bool ok = false;
    for (; it < pol_max_iter; ++it) {
        std::copy(e_perm.begin(), e_perm.end(), e_tot.begin());

        // Accumulate T·μ into e_tot
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
                f.box.min_image(dx, dy, dz);
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc2 || !(r2 > 0.0))
                    continue;
                const double r = std::sqrt(r2);
                const double r3 = r2 * r;
                const double r5 = r3 * r2;
                const Damp d = make_damp(r, ai, aj, thole_a, tt_b);
                const double f3 = s3(d) / r3;
                const double f5 = s5(d) / r5;
                const double* muj = last_mu_.data() + 3 * j;
                const double mdotr = muj[0] * dx + muj[1] * dy + muj[2] * dz;
                e_tot[static_cast<std::size_t>(3 * i + 0)] +=
                    3.0 * f5 * mdotr * dx - f3 * muj[0];
                e_tot[static_cast<std::size_t>(3 * i + 1)] +=
                    3.0 * f5 * mdotr * dy - f3 * muj[1];
                e_tot[static_cast<std::size_t>(3 * i + 2)] +=
                    3.0 * f5 * mdotr * dz - f3 * muj[2];
            }
        }

        double dmax = 0.0;
        for (int i = 0; i < n; ++i) {
            const double ai = alpha[static_cast<std::size_t>(i)];
            for (int c = 0; c < 3; ++c) {
                const double m =
                    ai * e_tot[static_cast<std::size_t>(3 * i + c)];
                mu_new[static_cast<std::size_t>(3 * i + c)] = m;
                dmax = std::max(
                    dmax,
                    std::fabs(m
                              - last_mu_[static_cast<std::size_t>(3 * i + c)]));
            }
        }
        last_mu_.swap(mu_new);
        if (dmax < pol_tol) {
            ok = true;
            ++it;
            break;
        }
    }
    last_pol_iters_ = it;
    last_pol_converged_ = ok;
    if (!ok)
        throw std::runtime_error(
            "PimInducedDipole: induced-dipole SCC did not converge");

    // E_pol = −½ Σ μ · E^perm  [eV]
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        e -= 0.5
             * (last_mu_[static_cast<std::size_t>(3 * i + 0)]
                    * e_perm[static_cast<std::size_t>(3 * i + 0)]
                + last_mu_[static_cast<std::size_t>(3 * i + 1)]
                      * e_perm[static_cast<std::size_t>(3 * i + 1)]
                + last_mu_[static_cast<std::size_t>(3 * i + 2)]
                      * e_perm[static_cast<std::size_t>(3 * i + 2)]);
    }
    e_pol_ = k * e;
    f.meta.get(kEnergy) += e_pol_;

    // Hellmann–Feynman forces at fixed μ:
    //   F_i = q_i E_i^μ + (μ_i · ∇) E_i^{q+μ}
    // excluding permanent charge–charge (q_i E_i^q).
    // Undamped multipole gradients; Thole/TT scale the multipole kernels.
    //
    // Configurational virial (Newton-pair form over the same per-pair
    // forces): W = ½ Σ_{i≠j} r_ij ⊗ f_ij, accumulated in Voigt order
    // xx yy zz xy xz yz [eV] into frame.meta kStress — LAMMPS sign
    // (positive trace ⇔ compressive), same convention as PairCoulCut/EAM.
    // Barostats (NoseHooverNPT, PR-NPT, Berendsen, NPH) read this trace,
    // so PIM under NPT barostats on the full polarization pressure.
    double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;
    for (int i = 0; i < n; ++i) {
        const double qi = charge[i];
        const double ai = alpha[static_cast<std::size_t>(i)];
        const double* mui = last_mu_.data() + 3 * i;
        double fix = 0, fiy = 0, fiz = 0;

        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            const double qj = charge[j];
            const double aj = alpha[static_cast<std::size_t>(j)];
            const double* muj = last_mu_.data() + 3 * j;

            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            f.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc2 || !(r2 > 0.0))
                continue;
            const double r = std::sqrt(r2);
            const double r3 = r2 * r;
            const double r5 = r3 * r2;
            const double r7 = r5 * r2;

            // α=0 ⇒ Thole u=0 ⇒ undamped λ=1 (charge-only or mixed pairs).
            const Damp dmp = make_damp(r, ai, aj, thole_a, tt_b);
            const double L3 = s3(dmp);
            const double L5 = s5(dmp);
            const double L7 = s7(dmp);

            // Field at i from dipole μ_j: E^μ = L5*3(μ·r)r/r^5 − L3 μ/r^3
            const double mjr = muj[0] * dx + muj[1] * dy + muj[2] * dz;
            const double Emx = L5 * 3.0 * mjr * dx / r5 - L3 * muj[0] / r3;
            const double Emy = L5 * 3.0 * mjr * dy / r5 - L3 * muj[1] / r3;
            const double Emz = L5 * 3.0 * mjr * dz / r5 - L3 * muj[2] / r3;

            // Per-pair force f_ij on i due to j (Gaussian e²/Å² until the
            // final ×k). Newton's third law holds term-by-term (each kernel
            // is odd under r → −r), so the pair virial is ½ r_ij ⊗ f_ij.
            double fpx = 0, fpy = 0, fpz = 0;

            // q_i × E_i^μ  (induced field force on permanent charge)
            fpx += qi * Emx;
            fpy += qi * Emy;
            fpz += qi * Emz;

            // Field gradient of charge q_j at i (for μ_i · G^q):
            // G_ab = q_j [ L3 δ_ab/r^3 − L5 * 3 r_a r_b / r^5 ]
            // (μ · G)_a = q_j [ L3 μ_a/r^3 − L5 * 3 (μ·r) r_a / r^5 ]
            const double mir = mui[0] * dx + mui[1] * dy + mui[2] * dz;
            fpx += qj * (L3 * mui[0] / r3 - L5 * 3.0 * mir * dx / r5);
            fpy += qj * (L3 * mui[1] / r3 - L5 * 3.0 * mir * dy / r5);
            fpz += qj * (L3 * mui[2] / r3 - L5 * 3.0 * mir * dz / r5);

            // Dipole–dipole: V = μ_i T μ_j = 3(μi·r)(μj·r)/r^5 − (μi·μj)/r^3
            // U_pair = −k V  (from U_T = −½ Σ_{a≠b} = −Σ_{a<b}).
            // ∇_i V = 3/r^5 [(μi·μj)r + (μj·r)μi + (μi·r)μj]
            //         − 15/r^7 (μi·r)(μj·r) r
            // F_i = −∇_i U_pair = k ∇_i V (Thole: L5/L7 scale the kernels).
            const double mir_mj =
                mui[0] * muj[0] + mui[1] * muj[1] + mui[2] * muj[2];
            fpx += L5 * 3.0 / r5 * (mir_mj * dx + mjr * mui[0] + mir * muj[0])
                   - L7 * 15.0 / r7 * mir * mjr * dx;
            fpy += L5 * 3.0 / r5 * (mir_mj * dy + mjr * mui[1] + mir * muj[1])
                   - L7 * 15.0 / r7 * mir * mjr * dy;
            fpz += L5 * 3.0 / r5 * (mir_mj * dz + mjr * mui[2] + mir * muj[2])
                   - L7 * 15.0 / r7 * mir * mjr * dz;

            fix += fpx;
            fiy += fpy;
            fiz += fpz;

            // ½ r_ij ⊗ f_ij per ordered-pair visit (each unordered pair is
            // visited twice with equal contribution).
            v_xx += 0.5 * dx * fpx;
            v_yy += 0.5 * dy * fpy;
            v_zz += 0.5 * dz * fpz;
            v_xy += 0.5 * dx * fpy;
            v_xz += 0.5 * dx * fpz;
            v_yz += 0.5 * dy * fpz;
        }

        // Convert Gaussian force (e²/Å²) → eV/Å
        fx[i] += k * fix;
        fy[i] += k * fiy;
        fz[i] += k * fiz;
    }

    // Convert Gaussian virial (e²/Å) → eV, publish to the shared stress meta.
    f.meta.get(kStress)[0] += k * v_xx;
    f.meta.get(kStress)[1] += k * v_yy;
    f.meta.get(kStress)[2] += k * v_zz;
    f.meta.get(kStress)[3] += k * v_xy;
    f.meta.get(kStress)[4] += k * v_xz;
    f.meta.get(kStress)[5] += k * v_yz;
}

} // namespace cpu
} // namespace ATV
