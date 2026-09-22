/// @file cpu/planewave/internal/gth_upf.cpp
#include "atomiverse/cpu/planewave/internal/gth_upf.h"

#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {

/// GTH local potential [Hartree], Goedecker et al. (1996) Eq. 1:
///   V_loc(r) = -(Z/r) erf(r/(sqrt(2) rloc))
///            + exp(-(r/rloc)^2/2) * sum_i C_i (r/rloc)^{2(i-1)}
AccReal gth_vloc_r(const gpw::GthPotential& pot, AccReal r) {
    const AccReal rl = static_cast<AccReal>(pot.rloc);
    const AccReal x = r / rl;
    const AccReal z = static_cast<AccReal>(pot.zion);

    // r -> 0: erf(r/(sqrt2 rl))/r -> sqrt(2/pi)/rl, finite.
    const AccReal erf_over_r =
        (r < AccReal(1e-10)) ? std::sqrt(AccReal(2) / AccReal(M_PI)) / rl
                             : std::erf(r / (std::sqrt(AccReal(2)) * rl)) / r;

    AccReal poly = 0, xp = 1;
    for (std::size_t i = 0; i < pot.cloc.size(); ++i) {
        poly += static_cast<AccReal>(pot.cloc[i]) * xp;
        xp *= x * x;
    }
    return -z * erf_over_r + std::exp(AccReal(-0.5) * x * x) * poly;
}

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

UPFData upf_from_gth(const gpw::GthPotential& pot, AccReal r_max, int n_r) {
    if (!(r_max > 0) || n_r < 2)
        throw std::invalid_argument(
            "upf_from_gth: r_max must be > 0 and n_r >= 2");
    if (!(pot.rloc > 0))
        throw std::invalid_argument("upf_from_gth: GTH rloc must be > 0 for "
                                    + pot.name);

    UPFData upf;
    upf.z_val = pot.zion;
    upf.pseudo_type = "NC"; // GTH is norm-conserving by construction.

    // Log mesh, the UPF convention. r_min is small enough that the r -> 0
    // limits above are never the integrand's dominant contribution.
    const AccReal r_min = AccReal(1e-5);
    const AccReal dx = std::log(r_max / r_min) / static_cast<AccReal>(n_r - 1);
    upf.r.resize(static_cast<std::size_t>(n_r));
    for (int ir = 0; ir < n_r; ++ir)
        upf.r[static_cast<std::size_t>(ir)] =
            r_min * std::exp(dx * static_cast<AccReal>(ir));

    // UPF stores the local potential in Rydberg (upf_vloc_to_gspace halves it).
    upf.v_loc.resize(upf.r.size());
    for (std::size_t ir = 0; ir < upf.r.size(); ++ir)
        upf.v_loc[ir] = AccReal(2) * gth_vloc_r(pot, upf.r[ir]);

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
            UPFData::KBProjector kb;
            kb.l = ch.l;
            kb.beta.resize(upf.r.size());
            for (std::size_t ir = 0; ir < upf.r.size(); ++ir)
                kb.beta[ir] = upf.r[ir]
                              * gth_proj_r(upf.r[ir], ch.l, i,
                                           static_cast<AccReal>(ch.r));
            upf.kb.push_back(std::move(kb));
        }
    }

    // D_ij: block-diagonal over channels, in Rydberg (setup_kb_projectors
    // applies the 0.5 Ry -> Ha conversion when it diagonalizes each l block).
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
                             + b] = AccReal(2) * ch.hij(a, b);
            off += ch.nproj;
        }
    }

    // rho_atom intentionally left empty: GTH supplies no atomic pseudo-charge.
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
