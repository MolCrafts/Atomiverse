/// @file cpu/planewave/internal/gth_upf.cpp
#include "atomiverse/cpu/planewave/internal/gth_upf.h"

#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {

/// GTH nonlocal projector [Bohr^{-3/2}], Goedecker et al. (1996) Eq. 3:
///   p_i^l(r) = sqrt(2) r^{l+2(i-1)} exp(-r^2/(2 r_l^2))
///              / ( r_l^{l+(4i-1)/2} sqrt(Gamma(l+(4i-1)/2)) )
/// Normalized so that \int p_i^l(r)^2 r^2 dr = 1.
/// @param i 1-based projector index within the channel.
AccReal gth_proj_r(AccReal r, int l, int i, AccReal rl) {
    const AccReal nu =
        static_cast<AccReal>(l) + AccReal(4 * i - 1) / AccReal(2);
    const AccReal norm =
        std::sqrt(AccReal(2)) / (std::pow(rl, nu) * std::sqrt(std::tgamma(nu)));
    return norm * std::pow(r, l + 2 * (i - 1))
           * std::exp(-r * r / (AccReal(2) * rl * rl));
}

} // namespace

void gth_local_r(const gpw::GthPotential& p, std::span<const AccReal> radius,
                 std::span<AccReal> value, std::span<AccReal> derivative) {
    const AccReal a = p.rloc, z = p.zion;
    const AccReal root = std::sqrt(AccReal(2) / AccReal(M_PI));
    for (std::size_t i = 0; i < radius.size(); ++i) {
        const AccReal r = radius[i], x = r / a, x2 = x * x;
        const AccReal e = std::exp(-x2 / 2);
        AccReal poly = 0, dpdx = 0;
        for (std::size_t j = p.cloc.size(); j-- > 0;) {
            dpdx = dpdx * x2 + 2 * x * poly;
            poly = poly * x2 + p.cloc[j];
        }
        AccReal vc, dvc;
        if (x < AccReal(1e-4)) {
            vc = -z * root / a
                 * (1 - x2 / 6 + x2 * x2 / 40 - x2 * x2 * x2 / 336);
            dvc = z * root / (a * a) * x
                  * (AccReal(1) / 3 - x2 / 10 + x2 * x2 / 56);
        } else {
            const AccReal erf = std::erf(x / std::sqrt(AccReal(2)));
            vc = -z * erf / r;
            dvc = z * (erf / (r * r) - root * e / (a * r));
        }
        value[i] = vc + e * poly;
        derivative[i] = dvc + e * (dpdx - x * poly) / a;
    }
}

NcPotentialData upf_from_gth(const gpw::GthPotential& pot, AccReal r_max,
                             int n_r) {
    if (!std::isfinite(r_max) || !(r_max > AccReal(1e-5)) || n_r < 3)
        throw std::invalid_argument(
            "upf_from_gth: r_max must be finite and > 1e-5 Bohr, and n_r >= 3");
    if (!std::isfinite(pot.rloc) || !(pot.rloc > 0))
        throw std::invalid_argument("upf_from_gth: GTH rloc must be > 0 for "
                                    + pot.name);

    NcPotentialData upf;
    upf.atomic_number = static_cast<int>(pot.element);
    upf.z_val = pot.zion;
    if (pot.cloc.size() > 4)
        throw std::invalid_argument(
            "upf_from_gth: GTH local polynomial supports "
            "at most four coefficients");
    for (const auto value : pot.cloc)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "upf_from_gth: GTH coefficients must be finite");
    for (const auto& ch : pot.channels) {
        if (ch.l < 0 || ch.l > 2 || ch.nproj < 0
            || ch.h.size() != static_cast<std::size_t>(ch.nproj) * ch.nproj)
            throw std::invalid_argument(
                "upf_from_gth: invalid GTH channel shape or angular momentum");
        if (ch.nproj && (!std::isfinite(ch.r) || ch.r <= 0))
            throw std::invalid_argument("upf_from_gth: GTH projector radius "
                                        "must be finite and positive");
        for (int i = 0; i < ch.nproj; ++i)
            for (int j = 0; j < ch.nproj; ++j)
                if (!std::isfinite(ch.hij(i, j))
                    || ch.hij(i, j) != ch.hij(j, i))
                    throw std::invalid_argument("upf_from_gth: GTH coupling "
                                                "must be finite and symmetric");
    }

    // Log mesh, the UPF convention. r_min is small enough that the r -> 0
    // limits above are never the integrand's dominant contribution.
    const AccReal r_min = AccReal(1e-5);
    const AccReal dx = std::log(r_max / r_min) / static_cast<AccReal>(n_r - 1);
    upf.r.resize(static_cast<std::size_t>(n_r));
    upf.rab.resize(static_cast<std::size_t>(n_r));
    for (int ir = 0; ir < n_r; ++ir) {
        upf.r[static_cast<std::size_t>(ir)] =
            r_min * std::exp(dx * static_cast<AccReal>(ir));
        upf.rab[static_cast<std::size_t>(ir)] =
            upf.r[static_cast<std::size_t>(ir)] * dx;
    }

    // The canonical radial representation uses Hartree directly.
    upf.v_loc.resize(upf.r.size());
    std::vector<AccReal> derivative(upf.r.size());
    gth_local_r(pot, upf.r, upf.v_loc, derivative);

    int l_max = 0;
    for (const auto& ch : pot.channels)
        l_max = std::max(l_max, ch.l);
    upf.l_max = l_max;

    // Projectors, flattened (channel, i) in table order. `beta` holds
    // r * p(r) — the PP_BETA convention setup_kb_projectors integrates
    // against as \int r * beta * j_l(Qr) dr.
    for (const auto& ch : pot.channels) {
        if (ch.nproj <= 0)
            continue;
        if (!(ch.r > 0))
            throw std::invalid_argument(
                "upf_from_gth: GTH projector radius must be > 0 for "
                + pot.name);
        for (int i = 1; i <= ch.nproj; ++i) {
            NcPotentialData::KBProjector kb;
            kb.l = ch.l;
            kb.beta.resize(upf.r.size());
            kb.radial_points = upf.r.size();
            for (std::size_t ir = 0; ir < upf.r.size(); ++ir)
                kb.beta[ir] = upf.r[ir]
                              * gth_proj_r(upf.r[ir], ch.l, i,
                                           static_cast<AccReal>(ch.r));
            upf.kb.push_back(std::move(kb));
        }
    }

    // D_ij: block-diagonal over channels, directly in Hartree.
    const int n_proj = upf.n_proj();
    if (n_proj > 0) {
        upf.d_ij.assign(static_cast<std::size_t>(n_proj) * n_proj, AccReal(0));
        int off = 0;
        for (const auto& ch : pot.channels) {
            if (ch.nproj <= 0)
                continue;
            for (int a = 0; a < ch.nproj; ++a)
                for (int b = 0; b < ch.nproj; ++b)
                    upf.d_ij[static_cast<std::size_t>(off + a) * n_proj + off
                             + b] = ch.hij(a, b);
            off += ch.nproj;
        }
    }

    // rho_atom intentionally left empty: GTH supplies no atomic pseudo-charge.
    upf.validate();
    return upf;
}

AccReal gth_vloc_gspace(const gpw::GthPotential& pot, AccReal g,
                        AccReal omega) {
    const AccReal rl = static_cast<AccReal>(pot.rloc);
    const AccReal z = static_cast<AccReal>(pot.zion);
    const AccReal gr = g * rl;
    const AccReal gr2 = gr * gr;
    const AccReal env = std::exp(AccReal(-0.5) * gr2);

    // Gaussian-polynomial part, Goedecker et al. (1996) Eq. 5.
    auto c = [&](std::size_t i) {
        return i < pot.cloc.size() ? static_cast<AccReal>(pot.cloc[i])
                                   : AccReal(0);
    };
    const AccReal poly = c(0) + c(1) * (AccReal(3) - gr2)
                         + c(2) * (AccReal(15) - AccReal(10) * gr2 + gr2 * gr2)
                         + c(3)
                               * (AccReal(105) - AccReal(105) * gr2
                                  + AccReal(21) * gr2 * gr2 - gr2 * gr2 * gr2);
    const AccReal sqrt_8pi3 =
        std::sqrt(AccReal(8) * AccReal(M_PI) * AccReal(M_PI) * AccReal(M_PI));
    const AccReal gauss = sqrt_8pi3 * rl * rl * rl / omega * env * poly;

    if (g * g < AccReal(1e-16)) {
        // G = 0: the divergent -4 pi Z/(Omega G^2) piece is cancelled against
        // the Ewald / Hartree conventions, leaving the finite remainder.
        return AccReal(2) * AccReal(M_PI) * z * rl * rl / omega + gauss;
    }
    return -AccReal(4) * AccReal(M_PI) * z / (omega * g * g) * env + gauss;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
