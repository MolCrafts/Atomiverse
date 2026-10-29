#include "atomiverse/cpu/planewave/internal/pw_radial_derivatives.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ATV::cpu::pw::internal {
namespace {
constexpr AccReal pi = AccReal(3.141592653589793238462643383279502884);
void require_nonnegative_finite(AccReal value) {
    if (!(value >= 0) || !std::isfinite(value))
        throw std::invalid_argument("NC reciprocal jet: invalid momentum");
}
} // namespace

RegularizedBesselJet regularized_bessel_jet(int l, AccReal x) {
    if (l < 0 || l > 2)
        throw std::invalid_argument("NC reciprocal jet: scalar l<=2 only");
    require_nonnegative_finite(x);
    // k_l(x)=j_l(x)/x^l has an even Taylor series. For |x|<=1 its
    // successive-term ratio is <=1/[2(k+1)(2l+2k+3)] and decreases.
    // Accumulate value and k'_l(x)/x directly, without subtracting two
    // almost equal values or dividing a vanishing f_l by q^l.
    if (x <= AccReal(1)) {
        const AccReal leading = l == 0   ? 1
                                : l == 1 ? AccReal(1.0 / 3)
                                         : AccReal(1.0 / 15);
        AccReal value = leading;
        AccReal derivative_term = -leading / AccReal(2 * l + 3);
        AccReal derivative = derivative_term;
        AccReal term = leading;
        const AccReal x2 = x * x;
        for (int k = 1; k <= 32; ++k) {
            term *= -x2 / (AccReal(2 * k) * AccReal(2 * l + 2 * k + 1));
            value += term;
            if (k > 1) {
                derivative_term *=
                    -x2 / (AccReal(2 * (k - 1)) * AccReal(2 * l + 2 * k + 1));
                derivative += derivative_term;
            }
            // Bound both tails using their next ratios; at x<=1 and k>=2
            // a geometric factor <=2 bounds every remaining term.
            if (k >= 2
                && 2 * std::abs(term) <= std::numeric_limits<AccReal>::epsilon()
                                             * std::abs(value)
                && 2 * std::abs(derivative_term)
                       <= std::numeric_limits<AccReal>::epsilon()
                              * std::abs(derivative))
                break;
        }
        return {value, derivative};
    }
    const AccReal j0 = std::sin(x) / x;
    const AccReal j1 = (j0 - std::cos(x)) / x;
    if (l == 0)
        return {j0, -j1 / x};
    const AccReal j2 = 3 * j1 / x - j0;
    if (l == 1)
        return {j1 / x, -j2 / (x * x)};
    const AccReal j3 = 5 * j2 / x - j1;
    return {j2 / (x * x), -j3 / (x * x * x)};
}

namespace {
template <bool NeedDerivative>
LocalFourierJet local_fourier_transform(const NcPotentialData& data, AccReal g,
                                        AccReal omega) {
    require_nonnegative_finite(g);
    if (!(omega > 0) || !std::isfinite(omega))
        throw std::invalid_argument("NC reciprocal jet: invalid volume");
    data.validate_radial_mesh();
    if (data.v_loc.size() != data.r.size())
        throw std::invalid_argument("NC reciprocal jet: local mesh mismatch");
    const bool zero = g * g < AccReal(1e-16); // same branch as current operator
    std::vector<AccReal> value(data.r.size()),
        derivative(NeedDerivative ? data.r.size() : 0);
    const AccReal z = static_cast<AccReal>(data.z_val);
    for (std::size_t ir = 0; ir < data.r.size(); ++ir) {
        const AccReal r = data.r[ir];
        if (zero) {
            value[ir] = r * (r * data.v_loc[ir] + z);
            continue;
        }
        const AccReal x = g * r;
        const AccReal a = r * (r * data.v_loc[ir] + z * std::erf(r));
        value[ir] = a * regularized_bessel_jet(0, x).value;
        if constexpr (NeedDerivative)
            derivative[ir] = -r * a * x * regularized_bessel_jet(1, x).value;
    }
    const AccReal pref = 4 * pi / omega;
    const AccReal integral = data.integrate_radial(value);
    if (zero)
        return {pref * integral, 0};
    const AccReal g2 = g * g;
    const AccReal screen = z * std::exp(-g2 / 4);
    AccReal slope = 0;
    if constexpr (NeedDerivative)
        slope = pref
                * (data.integrate_radial(derivative)
                   + screen * (AccReal(.5) / g + 2 / (g2 * g)));
    return {pref * (integral - screen / g2), slope};
}

template <bool NeedDerivative>
ProjectorRadialJet projector_radial_transform(const NcPotentialData& data,
                                              std::size_t projector,
                                              AccReal q) {
    require_nonnegative_finite(q);
    data.validate_radial_mesh();
    if (projector >= data.kb.size())
        throw std::invalid_argument("NC reciprocal jet: invalid projector");
    const auto& p = data.kb[projector];
    if (p.l < 0 || p.l > 2 || p.radial_points < 3
        || p.radial_points > p.beta.size() || p.beta.size() > data.r.size())
        throw std::invalid_argument("NC reciprocal jet: invalid radial prefix");
    std::vector<AccReal> value(p.radial_points),
        derivative(NeedDerivative ? p.radial_points : 0);
    for (std::size_t ir = 0; ir < p.radial_points; ++ir) {
        const AccReal r = data.r[ir];
        const auto bessel = regularized_bessel_jet(p.l, q * r);
        const AccReal weight = std::pow(r, p.l + 1) * p.beta[ir];
        value[ir] = weight * bessel.value;
        if constexpr (NeedDerivative)
            derivative[ir] = weight * r * r * bessel.slope_over_x;
    }
    AccReal slope_over_q = 0;
    if constexpr (NeedDerivative)
        slope_over_q = data.integrate_radial(derivative);
    return {data.integrate_radial(value), slope_over_q};
}
} // namespace

LocalFourierJet nc_vloc_fourier_jet(const NcPotentialData& data, AccReal g,
                                    AccReal omega) {
    return local_fourier_transform<true>(data, g, omega);
}
AccReal nc_vloc_fourier_value(const NcPotentialData& data, AccReal g,
                              AccReal omega) {
    return local_fourier_transform<false>(data, g, omega).value;
}
ProjectorRadialJet nc_projector_radial_jet(const NcPotentialData& data,
                                           std::size_t projector, AccReal q) {
    return projector_radial_transform<true>(data, projector, q);
}
AccReal nc_projector_radial_value(const NcPotentialData& data,
                                  std::size_t projector, AccReal q) {
    const auto radial = projector_radial_transform<false>(data, projector, q);
    return std::pow(q, data.kb[projector].l) * radial.value;
}

ProjectorCartesianJet nc_projector_cartesian_jet(const NcPotentialData& data,
                                                 std::size_t projector,
                                                 const Vec3& q) {
    for (int k = 0; k < 3; ++k)
        if (!std::isfinite(q[k]))
            throw std::invalid_argument("NC reciprocal jet: nonfinite q");
    const AccReal norm = std::sqrt(q.norm_squared());
    const auto radial = nc_projector_radial_jet(data, projector, norm);
    const int l = data.kb[projector].l;
    ProjectorCartesianJet out{2 * l + 1};
    std::array<AccReal, 5> h{};
    std::array<std::array<AccReal, 3>, 5> dh{};
    const AccReal x = q[0], y = q[1], z = q[2];
    if (l == 0) {
        h[0] = 1 / std::sqrt(4 * pi);
    } else if (l == 1) {
        const AccReal c = std::sqrt(3 / (4 * pi));
        h[0] = c * y;
        h[1] = c * z;
        h[2] = c * x;
        dh[0][1] = dh[1][2] = dh[2][0] = c;
    } else {
        const AccReal c = std::sqrt(15 / (4 * pi));
        const AccReal d = std::sqrt(5 / (16 * pi));
        h = {c * x * y, c * y * z, d * (2 * z * z - x * x - y * y), c * x * z,
             c * (x * x - y * y) / 2};
        dh[0] = {c * y, c * x, 0};
        dh[1] = {0, c * z, c * y};
        dh[2] = {-2 * d * x, -2 * d * y, 4 * d * z};
        dh[3] = {c * z, 0, c * x};
        dh[4] = {c * x, -c * y, 0};
    }
    for (int m = 0; m < out.components; ++m) {
        out.value[m] = radial.value * h[m];
        for (int k = 0; k < 3; ++k)
            out.gradient[m][k] =
                radial.value * dh[m][k] + radial.slope_over_q * q[k] * h[m];
    }
    return out;
}
} // namespace ATV::cpu::pw::internal
