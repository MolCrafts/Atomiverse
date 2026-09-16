/// @file src/cpu/path/dimer.cpp
#include "atomiverse/cpu/path/dimer.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"

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

        // --- Rotate N (finite-difference curvature) ---
        for (int rot = 0; rot < max_rotations; ++rot) {
            for (int i = 0; i < n; ++i) {
                f1.span(kAtom_x)[i] = frame.span(kAtom_x)[i] + delta * nx_;
                f1.span(kAtom_y)[i] = frame.span(kAtom_y)[i] + delta * ny_;
                f1.span(kAtom_z)[i] = frame.span(kAtom_z)[i] + delta * nz_;
                f2.span(kAtom_x)[i] = frame.span(kAtom_x)[i] - delta * nx_;
                f2.span(kAtom_y)[i] = frame.span(kAtom_y)[i] - delta * ny_;
                f2.span(kAtom_z)[i] = frame.span(kAtom_z)[i] - delta * nz_;
                f1.span(kAtom_fx)[i] = f1.span(kAtom_fy)[i] =
                    f1.span(kAtom_fz)[i] = 0;
                f2.span(kAtom_fx)[i] = f2.span(kAtom_fy)[i] =
                    f2.span(kAtom_fz)[i] = 0;
            }
            // Only rotate using the first atom's 3D mode for skeleton (single
            // particle / uniform N on all atoms). Multi-atom N is 3N-dim later.
            eval_force_(f1);
            eval_force_(f2);

            // F_R = (F1 − F2) / (2δ)  approximate −H N for atom 0
            const double frx =
                (f1.span(kAtom_fx)[0] - f2.span(kAtom_fx)[0]) / (2.0 * delta);
            const double fry =
                (f1.span(kAtom_fy)[0] - f2.span(kAtom_fy)[0]) / (2.0 * delta);
            const double frz =
                (f1.span(kAtom_fz)[0] - f2.span(kAtom_fz)[0]) / (2.0 * delta);

            // F_R ≈ −H N. Align N with the softest mode via a power-like step:
            // prefer direction that maximizes N·F_R (⇔ minimizes N·H·N).
            const double fr_n = frx * nx_ + fry * ny_ + frz * nz_;
            double tx = frx, ty = fry, tz = frz;
            if (fr_n < 0.0) {
                tx = -frx;
                ty = -fry;
                tz = -frz;
            }
            const double tnorm = std::sqrt(tx * tx + ty * ty + tz * tz);
            if (tnorm < rot_tol)
                break;
            // Mix toward the soft-mode estimate (stable under pure power iter).
            const double mix = 0.5;
            nx_ = (1.0 - mix) * nx_ + mix * (tx / tnorm);
            ny_ = (1.0 - mix) * ny_ + mix * (ty / tnorm);
            nz_ = (1.0 - mix) * nz_ + mix * (tz / tnorm);
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
