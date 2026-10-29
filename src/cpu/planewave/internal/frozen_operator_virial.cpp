#include "atomiverse/cpu/planewave/internal/frozen_operator_virial.h"
#include "atomiverse/cpu/planewave/internal/pw_radial_derivatives.h"
#include <cmath>
#include <stdexcept>
#include <vector>
namespace ATV::cpu::pw::internal {
namespace {
using C = std::complex<AccReal>;
constexpr AccReal pi = 3.14159265358979323846;
void orbital_shape(const GVectorList& gv, FrozenBandView o) {
    if (o.coefficients.size() != o.occupations.size() * gv.count())
        throw std::invalid_argument("band/full-complex G shape mismatch");
}
} // namespace

FrozenOperatorVirial kinetic_frozen_virial(const GVectorList& gv,
                                           FrozenBandView o) {
    orbital_shape(gv, o);
    FrozenOperatorVirial out;
    for (std::size_t band = 0; band < o.occupations.size(); ++band)
        for (int g = 0; g < gv.count(); ++g) {
            const auto q = gv.g[g] + gv.kvec;
            const auto weight =
                o.k_weight * o.occupations[band]
                * std::norm(o.coefficients[band * gv.count() + g]);
            out.energy_ha += .5 * weight * q.norm_squared();
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    out.virial_ha[3 * a + b] += weight * q[a] * q[b];
        }
    return out;
}
FrozenOperatorVirial hartree_frozen_virial(const GVectorList& gv,
                                           std::span<const C> rho,
                                           AccReal omega) {
    if (rho.size() != gv.g.size() || !(omega > 0))
        throw std::invalid_argument("density G/volume mismatch");
    FrozenOperatorVirial out;
    for (int g = 0; g < gv.count(); ++g) {
        const auto g2 = gv.g[g].norm_squared();
        if (g2 < AccReal(1e-16))
            continue;
        const auto e = 2 * pi * omega * std::norm(rho[g]) / g2;
        out.energy_ha += e;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                out.virial_ha[3 * a + b] +=
                    e * ((a == b ? 1. : 0.) - 2 * gv.g[g][a] * gv.g[g][b] / g2);
    }
    return out;
}
FrozenOperatorVirial local_potential_frozen_virial(const NcPotentialData& data,
                                                   const GVectorList& gv,
                                                   std::span<const Vec3> atoms,
                                                   std::span<const C> rho,
                                                   AccReal omega) {
    if (rho.size() != gv.g.size() || !(omega > 0))
        throw std::invalid_argument("local density G/volume mismatch");
    FrozenOperatorVirial out;
    for (int g = 0; g < gv.count(); ++g) {
        const auto norm = gv.g[g].norm();
        const auto jet = nc_vloc_fourier_jet(data, norm, omega);
        C structure{};
        for (const auto& r : atoms)
            structure += std::polar(AccReal(1), -gv.g[g].dot(r));
        const auto scalar = omega * std::real(std::conj(rho[g]) * structure);
        out.energy_ha += scalar * jet.value;
        if (norm > 0)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    out.virial_ha[3 * a + b] +=
                        scalar * jet.slope * gv.g[g][a] * gv.g[g][b] / norm;
    }
    for (int a = 0; a < 3; ++a)
        out.virial_ha[4 * a] += out.energy_ha;
    return out;
}
FrozenOperatorVirial nonlocal_frozen_virial(const NcPotentialData& data,
                                            const GVectorList& gv,
                                            std::span<const Vec3> atoms,
                                            FrozenBandView o, AccReal omega) {
    orbital_shape(gv, o);
    data.validate();
    if (!(omega > 0))
        throw std::invalid_argument("nonlocal volume");
    FrozenOperatorVirial out;
    const int np = data.n_proj();
    if (np == 0)
        return out;
    const AccReal pref = 4 * pi / std::sqrt(omega);
    // Geometry-only jets are cached across bands; D stays full in the original radial representation.
    std::vector<ProjectorCartesianJet> jets(static_cast<std::size_t>(np)
                                            * gv.count());
    for (int p = 0; p < np; ++p)
        for (int g = 0; g < gv.count(); ++g)
            jets[p * gv.count() + g] =
                nc_projector_cartesian_jet(data, p, gv.g[g] + gv.kvec);
    for (const auto& r : atoms)
        for (std::size_t band = 0; band < o.occupations.size(); ++band) {
            const auto weight =
                o.k_weight * o.occupations[band]; // No positive-only screening.
            std::vector<std::array<C, 5>> c(np);
            std::vector<std::array<std::array<C, 5>, 9>> eta(np);
            for (int p = 0; p < np; ++p) {
                const int ell = data.kb[p].l;
                const C il = ell == 0   ? C{1, 0}
                             : ell == 1 ? C{0, -1}
                                        : C{-1, 0};
                for (int g = 0; g < gv.count(); ++g) {
                    const auto q = gv.g[g] + gv.kvec;
                    const auto factor =
                        std::conj(pref * il * std::polar(AccReal(1), -q.dot(r)))
                        * o.coefficients[band * gv.count() + g];
                    const auto& jet = jets[p * gv.count() + g];
                    for (int m = 0; m < 2 * ell + 1; ++m) {
                        c[p][m] += factor * jet.value[m];
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                eta[p][3 * a + b][m] +=
                                    q[a] * factor * jet.gradient[m][b];
                    }
                }
            }
            for (int i = 0; i < np; ++i)
                for (int j = 0; j < np; ++j) {
                    if (data.kb[i].l != data.kb[j].l)
                        continue;
                    const auto d = data.d_ij[i * np + j];
                    for (int m = 0; m < 2 * data.kb[i].l + 1; ++m) {
                        out.energy_ha +=
                            weight * d
                            * std::real(std::conj(c[i][m]) * c[j][m]);
                        for (int ab = 0; ab < 9; ++ab)
                            out.virial_ha[ab] +=
                                2 * weight * d
                                * std::real(std::conj(c[i][m]) * eta[j][ab][m]);
                    }
                }
        }
    for (int a = 0; a < 3; ++a)
        out.virial_ha[4 * a] += out.energy_ha;
    return out;
}
} // namespace ATV::cpu::pw::internal
