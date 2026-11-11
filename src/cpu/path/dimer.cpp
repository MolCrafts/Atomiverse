/// @file src/cpu/path/dimer.cpp
#include "atomiverse/cpu/path/dimer.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace path {

Dimer::Dimer(double delta_, double dR_, int max_rotations_, double rot_tol_,
             double f_tol_, std::uint64_t seed_)
    : delta(delta_), dR(dR_), max_rotations(max_rotations_), rot_tol(rot_tol_),
      f_tol(f_tol_), seed(seed_) {
    if (!(delta > 0.0))
        throw std::invalid_argument("Dimer: delta must be > 0");
    if (!(dR > 0.0))
        throw std::invalid_argument("Dimer: dR must be > 0");
    if (max_rotations < 1)
        throw std::invalid_argument("Dimer: max_rotations must be ≥ 1");
}

void Dimer::set_force_fn(ForceFn fn) {
    force_fn_ = std::move(fn);
    use_test_pot_ = false;
}

void Dimer::use_inverted_harmonic(double kx, double ky) {
    if (!(kx > 0.0) || !(ky > 0.0))
        throw std::invalid_argument("Dimer: kx, ky must be > 0");
    kx_ = kx;
    ky_ = ky;
    use_test_pot_ = true;
    force_fn_ = nullptr;
    // Unstable mode is y (negative curvature of V = ½kx x² − ½ky y²).
    nx_ = 0.0;
    ny_ = 1.0;
    nz_ = 0.0;
}

void Dimer::normalize3_(double& x, double& y, double& z) const {
    const double n = std::sqrt(x * x + y * y + z * z);
    if (!(n > 1e-30)) {
        x = 1;
        y = 0;
        z = 0;
        return;
    }
    x /= n;
    y /= n;
    z /= n;
}

void Dimer::random_mode_(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    nx_ = g(rng);
    ny_ = g(rng);
    nz_ = g(rng);
    normalize3_(nx_, ny_, nz_);
}

void Dimer::eval_force_(cpu::Frame& f) {
    if (use_test_pot_) {
        const int n = f.nlocal();
        auto x = f.span(kAtom_x);
        auto y = f.span(kAtom_y);
        auto z = f.span(kAtom_z);
        auto fx = f.span(kAtom_fx);
        auto fy = f.span(kAtom_fy);
        auto fz = f.span(kAtom_fz);
        double e = 0;
        for (int i = 0; i < n; ++i) {
            // V = ½ kx x² − ½ ky y² ; F = (−kx x, +ky y, 0)
            fx[i] = -kx_ * x[i];
            fy[i] = ky_ * y[i];
            fz[i] = 0.0;
            e += 0.5 * kx_ * x[i] * x[i] - 0.5 * ky_ * y[i] * y[i];
            (void)z;
        }
        f.meta.get(kEnergy) = e;
        return;
    }
    if (!force_fn_)
        throw std::runtime_error(
            "Dimer: set_force_fn or use_inverted_harmonic before run");
    force_fn_(f);
}

void Dimer::end_force_difference_(const cpu::Frame& mid, cpu::Frame& f1,
                                  cpu::Frame& f2, double nx, double ny,
                                  double nz, double& dfx, double& dfy,
                                  double& dfz) {
    const int n = mid.nlocal();
    for (int i = 0; i < n; ++i) {
        f1.span(kAtom_x)[i] = mid.span(kAtom_x)[i] + delta * nx;
        f1.span(kAtom_y)[i] = mid.span(kAtom_y)[i] + delta * ny;
        f1.span(kAtom_z)[i] = mid.span(kAtom_z)[i] + delta * nz;
        f2.span(kAtom_x)[i] = mid.span(kAtom_x)[i] - delta * nx;
        f2.span(kAtom_y)[i] = mid.span(kAtom_y)[i] - delta * ny;
        f2.span(kAtom_z)[i] = mid.span(kAtom_z)[i] - delta * nz;
        f1.span(kAtom_fx)[i] = f1.span(kAtom_fy)[i] = f1.span(kAtom_fz)[i] = 0;
        f2.span(kAtom_fx)[i] = f2.span(kAtom_fy)[i] = f2.span(kAtom_fz)[i] = 0;
    }
    eval_force_(f1);
    eval_force_(f2);
    // Skeleton: one 3D mode on atom 0 (single particle / uniform N).
    dfx = f1.span(kAtom_fx)[0] - f2.span(kAtom_fx)[0];
    dfy = f1.span(kAtom_fy)[0] - f2.span(kAtom_fy)[0];
    dfz = f1.span(kAtom_fz)[0] - f2.span(kAtom_fz)[0];
}

int Dimer::run(cpu::Frame& frame, int max_iter) {
    if (max_iter < 1)
        throw std::invalid_argument("Dimer::run: max_iter must be ≥ 1");
    if (frame.nlocal() < 1)
        throw std::invalid_argument("Dimer::run: empty frame");

    random_mode_(seed);
    converged_ = false;
    f_mod_ = 1e30;

    const int n = frame.nlocal();

    // Working copies for the two dimer ends.
    cpu::Frame f1 = frame;
    cpu::Frame f2 = frame;

    for (int it = 0; it < max_iter; ++it) {
        // Midpoint forces.
        for (int i = 0; i < n; ++i) {
            frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
                frame.span(kAtom_fz)[i] = 0;
        }
        eval_force_(frame);

        // --- Rotate N toward the lowest-curvature mode ---
        // Henkelman & Jónsson, J. Chem. Phys. 111, 7010 (1999), with the
        // trial-rotation line search of Kästner & Sherwood, J. Chem. Phys.
        // 128, 014106 (2008), eqs. 5-8. The rotational force F_perp descends
        // the curvature C(N) = N·H·N; following the raw finite-difference
        // force instead converges on the largest-|curvature| mode, which is
        // the stiff one whenever the start leans toward it.
        for (int rot = 0; rot < max_rotations; ++rot) {
            double dfx, dfy, dfz; // F1 − F2 at N [eV/Å]
            end_force_difference_(frame, f1, f2, nx_, ny_, nz_, dfx, dfy, dfz);
            const double df_n = dfx * nx_ + dfy * ny_ + dfz * nz_;
            const double c0 = -df_n / (2.0 * delta); // curvature [eV/Å²]
            double px = dfx - df_n * nx_;
            double py = dfy - df_n * ny_;
            double pz = dfz - df_n * nz_;
            const double f_perp = std::sqrt(px * px + py * py + pz * pz);
            if (f_perp < rot_tol)
                break;
            // Θ: unit rotation direction; dC/dφ at φ = 0 is −2|F_perp|/(2δ).
            px /= f_perp;
            py /= f_perp;
            pz /= f_perp;
            const double dc = -f_perp / delta;
            const double phi1 = -0.5 * std::atan(dc / (2.0 * std::fabs(c0)));
            // A trial angle this small cannot resolve a1 (0/0 below).
            if (!(1.0 - std::cos(2.0 * phi1) > 1e-14))
                break;
            const double c1x = std::cos(phi1) * nx_ + std::sin(phi1) * px;
            const double c1y = std::cos(phi1) * ny_ + std::sin(phi1) * py;
            const double c1z = std::cos(phi1) * nz_ + std::sin(phi1) * pz;
            double d1x, d1y, d1z;
            end_force_difference_(frame, f1, f2, c1x, c1y, c1z, d1x, d1y, d1z);
            const double c1 =
                -(d1x * c1x + d1y * c1y + d1z * c1z) / (2.0 * delta);
            // C(φ) = a0/2 + a1 cos 2φ + b1 sin 2φ through C0, dC and C1.
            const double b1 = 0.5 * dc;
            const double a1 = (c0 - c1 + b1 * std::sin(2.0 * phi1))
                              / (1.0 - std::cos(2.0 * phi1));
            const double a0 = 2.0 * (c0 - a1);
            double phi_min = 0.5 * std::atan(b1 / a1);
            const double c_min = 0.5 * a0 + a1 * std::cos(2.0 * phi_min)
                                 + b1 * std::sin(2.0 * phi_min);
            if (c_min > c0)
                phi_min += 0.5 * units::pi;
            const double cs = std::cos(phi_min), sn = std::sin(phi_min);
            nx_ = cs * nx_ + sn * px;
            ny_ = cs * ny_ + sn * py;
            nz_ = cs * nz_ + sn * pz;
            normalize3_(nx_, ny_, nz_);
        }

        // Modified force at midpoint: invert parallel component (climb).
        double fpx = 0, fpy = 0, fpz = 0;
        for (int i = 0; i < n; ++i) {
            fpx += frame.span(kAtom_fx)[i];
            fpy += frame.span(kAtom_fy)[i];
            fpz += frame.span(kAtom_fz)[i];
        }
        // Use atom-0 force for single-particle saddle tests (n may be 1).
        fpx = frame.span(kAtom_fx)[0];
        fpy = frame.span(kAtom_fy)[0];
        fpz = frame.span(kAtom_fz)[0];
        const double f_par = fpx * nx_ + fpy * ny_ + fpz * nz_;
        // F_mod = F − 2 (F·N) N
        const double fmx = fpx - 2.0 * f_par * nx_;
        const double fmy = fpy - 2.0 * f_par * ny_;
        const double fmz = fpz - 2.0 * f_par * nz_;
        f_mod_ = std::sqrt(fmx * fmx + fmy * fmy + fmz * fmz);
        if (f_mod_ < f_tol) {
            converged_ = true;
            return it + 1;
        }

        // Translate midpoint along F_mod with adaptive step (cap by dR).
        double ux = fmx, uy = fmy, uz = fmz;
        normalize3_(ux, uy, uz);
        // Step size: min(dR, 0.5 * |F_mod| / max(k,1)) for inverted-harmonic scale.
        double step = dR;
        if (f_mod_ < 1.0)
            step = std::min(step, 0.5 * f_mod_);
        for (int i = 0; i < n; ++i) {
            frame.span(kAtom_x)[i] += step * ux;
            frame.span(kAtom_y)[i] += step * uy;
            frame.span(kAtom_z)[i] += step * uz;
        }
    }
    return max_iter;
}

} // namespace path
} // namespace cpu
} // namespace ATV
