/// @file src/cpu/path/string_method.cpp
#include "atomiverse/cpu/path/string_method.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {
namespace path {

namespace {

void muller_brown_force(cpu::Frame& f) {
    constexpr double kA[4] = {-200.0, -100.0, -170.0, 15.0};
    constexpr double kAa[4] = {-1.0, -1.0, -6.5, 0.7};
    constexpr double kBb[4] = {0.0, 0.0, 11.0, 0.6};
    constexpr double kCc[4] = {-10.0, -10.0, -6.5, 0.7};
    constexpr double kX0[4] = {1.0, 0.0, -0.5, -1.0};
    constexpr double kY0[4] = {0.0, 0.5, 1.5, 1.0};
    const double x = f.span(kAtom_x)[0];
    const double y = f.span(kAtom_y)[0];
    double v = 0, dvdx = 0, dvdy = 0;
    for (int k = 0; k < 4; ++k) {
        const double dx = x - kX0[k];
        const double dy = y - kY0[k];
        const double e =
            kA[k]
            * std::exp(kAa[k] * dx * dx + kBb[k] * dx * dy + kCc[k] * dy * dy);
        v += e;
        dvdx += e * (2.0 * kAa[k] * dx + kBb[k] * dy);
        dvdy += e * (kBb[k] * dx + 2.0 * kCc[k] * dy);
    }
    f.meta.get(kEnergy) = v;
    f.span(kAtom_fx)[0] = -dvdx;
    f.span(kAtom_fy)[0] = -dvdy;
    f.span(kAtom_fz)[0] = 0;
}

cpu::Frame lerp_frame(const cpu::Frame& a, const cpu::Frame& b, double t) {
    cpu::Frame f = a;
    const int n = f.nlocal();
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] =
            (1 - t) * a.span(kAtom_x)[i] + t * b.span(kAtom_x)[i];
        f.span(kAtom_y)[i] =
            (1 - t) * a.span(kAtom_y)[i] + t * b.span(kAtom_y)[i];
        f.span(kAtom_z)[i] =
            (1 - t) * a.span(kAtom_z)[i] + t * b.span(kAtom_z)[i];
    }
    return f;
}

} // namespace

StringMethod::StringMethod(int n_images_, double step_, int max_iter_,
                           double f_tol_)
    : n_images(n_images_), step(step_), max_iter(max_iter_), f_tol(f_tol_) {
    if (n_images < 3)
        throw std::invalid_argument("StringMethod: n_images >= 3");
    if (!(step > 0.0))
        throw std::invalid_argument("StringMethod: step > 0");
}

void StringMethod::set_force_fn(ForceFn fn) { force_ = std::move(fn); }

void StringMethod::use_muller_brown() { force_ = muller_brown_force; }

void StringMethod::image(int i, cpu::Frame& out) const {
    out = images_.at(static_cast<size_t>(i));
}

void StringMethod::reparameterize_() {
    const int N = n_images;
    std::vector<double> s(static_cast<size_t>(N), 0.0);
    for (int i = 1; i < N; ++i) {
        double d2 = 0;
        const int n = images_[0].nlocal();
        for (int a = 0; a < n; ++a) {
            const double dx =
                images_[i].span(kAtom_x)[a] - images_[i - 1].span(kAtom_x)[a];
            const double dy =
                images_[i].span(kAtom_y)[a] - images_[i - 1].span(kAtom_y)[a];
            const double dz =
                images_[i].span(kAtom_z)[a] - images_[i - 1].span(kAtom_z)[a];
            d2 += dx * dx + dy * dy + dz * dz;
        }
        s[static_cast<size_t>(i)] =
            s[static_cast<size_t>(i - 1)] + std::sqrt(d2);
    }
    const double L = s.back();
    if (!(L > 0.0))
        return;
    std::vector<cpu::Frame> neu = images_;
    for (int i = 1; i < N - 1; ++i) {
        const double target = L * (static_cast<double>(i) / (N - 1));
        int j = 0;
        while (j < N - 1 && s[static_cast<size_t>(j + 1)] < target)
            ++j;
        const double s0 = s[static_cast<size_t>(j)];
        const double s1 = s[static_cast<size_t>(j + 1)];
        const double t = (s1 > s0) ? (target - s0) / (s1 - s0) : 0.0;
        neu[static_cast<size_t>(i)] = lerp_frame(images_[j], images_[j + 1], t);
    }
    images_.swap(neu);
}

void StringMethod::evolve_once_() {
    const int N = n_images;
    energies_.assign(static_cast<size_t>(N), 0.0);
    max_energy_ = -1e300;
    double max_fperp = 0;

    for (int i = 0; i < N; ++i) {
        force_(images_[static_cast<size_t>(i)]);
        energies_[static_cast<size_t>(i)] =
            images_[static_cast<size_t>(i)].meta.get(kEnergy);
        max_energy_ = std::max(max_energy_, energies_[static_cast<size_t>(i)]);
    }

    // Endpoints fixed.
    for (int i = 1; i < N - 1; ++i) {
        // Tangent from finite difference of neighbors.
        const int n = images_[0].nlocal();
        std::vector<double> tau(static_cast<size_t>(3 * n), 0.0);
        double tnorm = 0;
        for (int a = 0; a < n; ++a) {
            const double tx = images_[i + 1].span(kAtom_x)[a]
                              - images_[i - 1].span(kAtom_x)[a];
            const double ty = images_[i + 1].span(kAtom_y)[a]
                              - images_[i - 1].span(kAtom_y)[a];
            const double tz = images_[i + 1].span(kAtom_z)[a]
                              - images_[i - 1].span(kAtom_z)[a];
            tau[static_cast<size_t>(3 * a + 0)] = tx;
            tau[static_cast<size_t>(3 * a + 1)] = ty;
            tau[static_cast<size_t>(3 * a + 2)] = tz;
            tnorm += tx * tx + ty * ty + tz * tz;
        }
        tnorm = std::sqrt(tnorm);
        if (tnorm > 0) {
            for (double& v : tau)
                v /= tnorm;
        }
        // F_perp = F − (F·τ) τ ; step x += step * F_perp
        auto& im = images_[static_cast<size_t>(i)];
        double fdot = 0;
        for (int a = 0; a < n; ++a) {
            fdot +=
                im.span(kAtom_fx)[a] * tau[static_cast<size_t>(3 * a)]
                + im.span(kAtom_fy)[a] * tau[static_cast<size_t>(3 * a + 1)]
                + im.span(kAtom_fz)[a] * tau[static_cast<size_t>(3 * a + 2)];
        }
        double fperp2 = 0;
        for (int a = 0; a < n; ++a) {
            const double fpx =
                im.span(kAtom_fx)[a] - fdot * tau[static_cast<size_t>(3 * a)];
            const double fpy = im.span(kAtom_fy)[a]
                               - fdot * tau[static_cast<size_t>(3 * a + 1)];
            const double fpz = im.span(kAtom_fz)[a]
                               - fdot * tau[static_cast<size_t>(3 * a + 2)];
            fperp2 += fpx * fpx + fpy * fpy + fpz * fpz;
            im.span(kAtom_x)[a] += step * fpx;
            im.span(kAtom_y)[a] += step * fpy;
            im.span(kAtom_z)[a] += step * fpz;
        }
        max_fperp = std::max(max_fperp, std::sqrt(fperp2));
    }
    reparameterize_();
    converged_ = (max_fperp < f_tol);
}

int StringMethod::run(const cpu::Frame& start, const cpu::Frame& end) {
    if (!force_)
        throw std::runtime_error("StringMethod: force function not set");
    const int N = n_images;
    images_.clear();
    images_.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / (N - 1);
        images_.push_back(lerp_frame(start, end, t));
    }
    converged_ = false;
    int it = 0;
    for (; it < max_iter; ++it) {
        evolve_once_();
        if (converged_)
            break;
    }
    // Final energies after last reparam.
    max_energy_ = -1e300;
    energies_.resize(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        force_(images_[static_cast<size_t>(i)]);
        energies_[static_cast<size_t>(i)] =
            images_[static_cast<size_t>(i)].meta.get(kEnergy);
        max_energy_ = std::max(max_energy_, energies_[static_cast<size_t>(i)]);
    }
    return it + 1;
}

} // namespace path
} // namespace cpu
} // namespace ATV
