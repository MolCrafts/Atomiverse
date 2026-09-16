/// @file cpu/planewave/internal/smearing.cpp
/// @brief Metallic occupation smearing implementations.
#include "atomiverse/cpu/planewave/internal/smearing.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {

constexpr AccReal kPi = AccReal(3.14159265358979323846);
constexpr AccReal kSqrtPi = AccReal(1.77245385090551602729);

AccReal clamp_exp_arg(AccReal x2) {
    // Avoid underflow; e^{-200} is already zero in double.
    return std::min(x2, AccReal(200));
}

/// Hermite polynomial H_n(x) via recurrence: H_0=1, H_1=2x, H_{n+1}=2x H_n - 2n H_{n-1}.
AccReal hermite(int n, AccReal x) {
    if (n == 0)
        return AccReal(1);
    if (n == 1)
        return AccReal(2) * x;
    AccReal h0 = AccReal(1);
    AccReal h1 = AccReal(2) * x;
    for (int k = 1; k < n; ++k) {
        AccReal h2 = AccReal(2) * x * h1 - AccReal(2) * AccReal(k) * h0;
        h0 = h1;
        h1 = h2;
    }
    return h1;
}

} // namespace

AccReal smear_f_spinless(AccReal x, SmearKind kind, int mp_order) {
    switch (kind) {
    case SmearKind::fermi_dirac: {
        // f = 1 / (1 + exp(x)); stable limits
        if (x > AccReal(36))
            return AccReal(0);
        if (x < AccReal(-36))
            return AccReal(1);
        return AccReal(1) / (AccReal(1) + std::exp(x));
    }
    case SmearKind::gaussian: {
        return AccReal(0.5) * std::erfc(static_cast<double>(x));
    }
    case SmearKind::methfessel_paxton: {
        if (mp_order < 1)
            throw std::invalid_argument("smear_f_spinless: mp_order must be >= "
                                        "1 for Methfessel-Paxton");
        // f_N(x) = 1/2 erfc(x) + Σ_{n=1}^N A_n H_{2n-1}(x) exp(-x²)
        // A_n = (-1)^n / (n! 4^n √π)
        AccReal f = AccReal(0.5) * std::erfc(static_cast<double>(x));
        const AccReal ex = std::exp(-clamp_exp_arg(x * x));
        AccReal inv_fact = AccReal(1); // 1/n!
        for (int n = 1; n <= mp_order; ++n) {
            inv_fact /= AccReal(n);
            // A_n = (-1)^n / (n! 4^n √π)
            const AccReal a_n = std::pow(AccReal(-1), n) * inv_fact
                                / (std::pow(AccReal(4), n) * kSqrtPi);
            f += a_n * hermite(2 * n - 1, x) * ex;
        }
        // Clamp tiny numerical overshoot
        if (f < AccReal(0))
            f = AccReal(0);
        if (f > AccReal(1))
            f = AccReal(1);
        return f;
    }
    }
    throw std::invalid_argument("smear_f_spinless: unknown kind");
}

AccReal smear_f_band(AccReal eig, AccReal mu, AccReal degauss, SmearKind kind,
                     int mp_order) {
    if (!(degauss > AccReal(0)))
        throw std::invalid_argument("smear_f_band: degauss must be > 0");
    const AccReal x = (eig - mu) / degauss;
    return AccReal(2) * smear_f_spinless(x, kind, mp_order);
}

AccReal smear_f_spinor(AccReal eig, AccReal mu, AccReal degauss, SmearKind kind,
                       int mp_order) {
    if (!(degauss > AccReal(0)))
        throw std::invalid_argument("smear_f_spinor: degauss must be > 0");
    const AccReal x = (eig - mu) / degauss;
    return smear_f_spinless(x, kind, mp_order);
}

AccReal smear_entropy_band(AccReal eig, AccReal mu, AccReal degauss,
                           SmearKind kind, int mp_order) {
    if (!(degauss > AccReal(0)))
        throw std::invalid_argument("smear_entropy_band: degauss must be > 0");
    const AccReal x = (eig - mu) / degauss;
    switch (kind) {
    case SmearKind::fermi_dirac: {
        AccReal f = smear_f_spinless(x, kind, mp_order);
        // s = -2 [f ln f + (1-f) ln(1-f)]
        constexpr AccReal eps = AccReal(1e-30);
        f = std::min(std::max(f, eps), AccReal(1) - eps);
        const AccReal omf = AccReal(1) - f;
        return AccReal(-2) * (f * std::log(f) + omf * std::log(omf));
    }
    case SmearKind::gaussian: {
        // QE w1gauss for n=0: (1/2 / √π) exp(-x²) ; band ×2 spins → exp/√π
        const AccReal ex = std::exp(-clamp_exp_arg(x * x));
        return ex / kSqrtPi;
    }
    case SmearKind::methfessel_paxton: {
        // QE-style: for MP-N, leading Gaussian piece plus Hermite corrections.
        // s_band = 2 * A_N * H_{2N}(x) * exp(-x²)  (QE w1gauss × spin)
        // Use the N-th term contribution (variational for free energy).
        if (mp_order < 1)
            throw std::invalid_argument(
                "smear_entropy_band: mp_order must be >= 1");
        const AccReal ex = std::exp(-clamp_exp_arg(x * x));
        AccReal inv_fact = AccReal(1);
        for (int n = 1; n <= mp_order; ++n)
            inv_fact /= AccReal(n);
        const AccReal a_n = std::pow(AccReal(-1), mp_order) * inv_fact
                            / (std::pow(AccReal(4), mp_order) * kSqrtPi);
        // spin degeneracy ×2
        return AccReal(2) * a_n * hermite(2 * mp_order, x) * ex;
    }
    }
    throw std::invalid_argument("smear_entropy_band: unknown kind");
}

AccReal smear_entropy_spinor(AccReal eig, AccReal mu, AccReal degauss,
                             SmearKind kind, int mp_order) {
    // Half the closed-shell band entropy (one spin channel).
    return AccReal(0.5)
           * smear_entropy_band(eig, mu, degauss, kind, mp_order);
}

namespace {

AccReal nelec_of_mu(const std::vector<AccReal>& evals,
                    const std::vector<AccReal>& weights, AccReal mu,
                    AccReal degauss, SmearKind kind, int mp_order,
                    AccReal spin_degen) {
    AccReal n = 0;
    const std::size_t m = evals.size();
    for (std::size_t i = 0; i < m; ++i)
        n += weights[i] * spin_degen
             * smear_f_spinless((evals[i] - mu) / degauss, kind, mp_order);
    return n;
}

} // namespace

AccReal find_fermi_level(const std::vector<AccReal>& evals,
                         const std::vector<AccReal>& weights, AccReal n_elec,
                         AccReal degauss, SmearKind kind, int mp_order,
                         AccReal spin_degen) {
    if (evals.empty())
        throw std::invalid_argument("find_fermi_level: empty spectrum");
    if (evals.size() != weights.size())
        throw std::invalid_argument(
            "find_fermi_level: evals/weights size mismatch");
    if (!(degauss > AccReal(0)))
        throw std::invalid_argument("find_fermi_level: degauss must be > 0");
    if (!(n_elec > AccReal(0)))
        throw std::invalid_argument("find_fermi_level: n_elec must be > 0");
    if (!(spin_degen > AccReal(0)))
        throw std::invalid_argument("find_fermi_level: spin_degen must be > 0");

    AccReal emin = evals[0];
    AccReal emax = evals[0];
    for (AccReal e : evals) {
        emin = std::min(emin, e);
        emax = std::max(emax, e);
    }
    // Pad the search window by several degauss widths.
    AccReal lo = emin - AccReal(20) * degauss;
    AccReal hi = emax + AccReal(20) * degauss;
    AccReal n_lo =
        nelec_of_mu(evals, weights, lo, degauss, kind, mp_order, spin_degen);
    AccReal n_hi =
        nelec_of_mu(evals, weights, hi, degauss, kind, mp_order, spin_degen);
    // Expand if needed (pathological empty/full windows).
    for (int expand = 0; expand < 20 && !(n_lo <= n_elec && n_elec <= n_hi);
         ++expand) {
        lo -= AccReal(20) * degauss;
        hi += AccReal(20) * degauss;
        n_lo = nelec_of_mu(evals, weights, lo, degauss, kind, mp_order,
                           spin_degen);
        n_hi = nelec_of_mu(evals, weights, hi, degauss, kind, mp_order,
                           spin_degen);
    }
    if (!(n_lo <= n_elec && n_elec <= n_hi)) {
        // Fall back to mid of spectrum — still better than throwing in SCF.
        return AccReal(0.5) * (emin + emax);
    }

    // Bisection
    for (int it = 0; it < 80; ++it) {
        const AccReal mid = AccReal(0.5) * (lo + hi);
        const AccReal n_mid = nelec_of_mu(evals, weights, mid, degauss, kind,
                                          mp_order, spin_degen);
        if (n_mid < n_elec)
            lo = mid;
        else
            hi = mid;
        if (std::abs(hi - lo) < AccReal(1e-14) * (AccReal(1) + std::abs(mid)))
            break;
    }
    return AccReal(0.5) * (lo + hi);
}

SmearFill fill_occupations(const std::vector<AccReal>& evals,
                           const std::vector<AccReal>& weights, AccReal mu,
                           AccReal degauss, SmearKind kind, int mp_order,
                           AccReal spin_degen) {
    SmearFill out;
    out.occ.resize(evals.size());
    for (std::size_t i = 0; i < evals.size(); ++i) {
        if (!(degauss > AccReal(0)))
            throw std::invalid_argument("fill_occupations: degauss must be > 0");
        const AccReal x = (evals[i] - mu) / degauss;
        const AccReal f = smear_f_spinless(x, kind, mp_order);
        out.occ[i] = spin_degen * f;
        out.n_occ += weights[i] * out.occ[i];
        // Entropy: compute spinless s and scale by spin_degen (not via
        // smear_entropy_band's hard-coded ×2, which mixed poorly with
        // spin_degen scaling when f is extreme).
        AccReal s_spinless = AccReal(0);
        switch (kind) {
        case SmearKind::fermi_dirac: {
            constexpr AccReal eps = AccReal(1e-30);
            AccReal fc = std::min(std::max(f, eps), AccReal(1) - eps);
            const AccReal omf = AccReal(1) - fc;
            s_spinless = -(fc * std::log(fc) + omf * std::log(omf));
            break;
        }
        case SmearKind::gaussian:
        case SmearKind::methfessel_paxton:
            // Half of the closed-shell band entropy weight.
            s_spinless = AccReal(0.5)
                         * smear_entropy_band(evals[i], mu, degauss, kind,
                                              mp_order);
            break;
        }
        if (std::isfinite(static_cast<double>(s_spinless)))
            out.entropy += weights[i] * spin_degen * s_spinless;
    }
    return out;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
