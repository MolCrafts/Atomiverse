#pragma once
#include "published_si.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace nc_example {
inline void near(double got, double want, double tolerance,
                 const char* quantity) {
    if (!std::isfinite(got) || !std::isfinite(want)
        || std::abs(got - want) > tolerance) {
        std::fprintf(stderr, "FAIL %s got=%.17g want=%.17g tol=%.2g\n",
                     quantity, got, want, tolerance);
        throw std::runtime_error(quantity);
    }
}

// Pure radial mathematics of the provider data. This does not reconstruct
// the PW Fourier operator or call a second electronic solver.
inline void check_operators(const PublishedSi& potential) {
    using Real = PublishedSi::Real;
    const auto data = potential.projectors();
    if (data.projectors.size() != 3 || data.coupling.size() != 9
        || data.projectors[0].l != 0 || data.projectors[1].l != 0
        || data.projectors[2].l != 1)
        throw std::runtime_error("published projector radial/l ordering");
    const auto overlap = [&](int a, int b) {
        const auto x = data.projectors[a].beta, y = data.projectors[b].beta;
        const auto n = std::min(x.size(), y.size());
        if (n != PublishedSi::projector_points || n >= data.radius.size())
            throw std::runtime_error(
                "projector must be the declared exact prefix");
        // Composite Simpson on this uniform odd mesh is the elementary radial
        // inner product integral beta_i*beta_j dr, not a PW implementation.
        Real integral = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const int weight = i == 0 || i + 1 == n ? 1 : (i % 2 ? 4 : 2);
            integral += weight * x[i] * y[i] * data.jacobian[i];
        }
        return integral / 3;
    };
    // Gaussian moment closed form: S11=S22=Spp=1, S12=sqrt(3/5).
    const double s12 = std::sqrt(3. / 5.);
    near(overlap(0, 0), 1, 3e-12, "normalized s1");
    near(overlap(1, 1), 1, 3e-12, "normalized s2");
    near(overlap(2, 2), 1, 3e-12, "normalized p1");
    near(overlap(0, 1), s12, 3e-12, "nonorthogonal s overlap");
    // Action on (p1+.4*p2)Y00 + .2*p1Y10: coefficients are D times the
    // radial inner products. Expected values use exact Gaussian moments.
    const std::array<double, 3> exact = {1 + .4 * s12, s12 + .4, .2};
    const std::array<double, 3> measured = {overlap(0, 0) + .4 * overlap(0, 1),
                                            overlap(1, 0) + .4 * overlap(1, 1),
                                            .2 * overlap(2, 2)};
    const std::array<double, 3> action = {
        8.95174150 * exact[0] - 2.70627082 * exact[1],
        -2.70627082 * exact[0] + 3.49378060 * exact[1], 2.43127673 * exact[2]};
    for (int i = 0; i < 3; ++i) {
        double got = 0;
        for (int j = 0; j < 3; ++j)
            got += data.coupling[i * 3 + j] * measured[j];
        near(got, action[i], 3e-11, "nonlocal radial action [Ha]");
    }
    const std::array<Real, 5> radius = {0, Real(.2), Real(.44), Real(1),
                                        Real(20)};
    std::array<Real, 5> value{}, derivative{};
    potential.local_r(radius, value, derivative);
    near(value[0],
         PublishedSi::c1
             - 4 * std::sqrt(2 / std::numbers::pi) / PublishedSi::rloc,
         1e-13, "local smooth origin [Ha]");
    near(derivative[0], 0, 0, "local origin derivative");
    near(value[4], -4. / 20, 1e-14, "Coulomb local tail [Ha]");
    near(derivative[4], 4. / 400, 1e-14, "Coulomb derivative [Ha/Bohr]");
    for (std::size_t i = 1; i < 4; ++i) {
        constexpr Real step = 1e-5;
        const std::array<Real, 2> displaced = {radius[i] - step,
                                               radius[i] + step};
        std::array<Real, 2> values{}, slopes{};
        potential.local_r(displaced, values, slopes);
        near(derivative[i], (values[1] - values[0]) / (2 * step), 2e-7,
             "local analytic derivative [Ha/Bohr]");
    }
    std::printf("PASS published GTH local limits/derivatives and normalized "
                "s/p nonlocal action\n");
}
} // namespace nc_example
