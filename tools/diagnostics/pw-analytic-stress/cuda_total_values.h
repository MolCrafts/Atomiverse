#pragma once
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include <span>
namespace pw_frozen {
using namespace ATV;
using C = std::complex<AccReal>;
using internal_data = cpu::pw::internal::NcPotentialData;
using GList = cpu::pw::internal::GVectorList;
struct Orbitals {
    std::span<const C> coefficients;
    std::span<const AccReal> occupations;
    AccReal k_weight = 1;
};
} // namespace pw_frozen
#include "atomiverse/cpu/planewave/internal/poisson.h"
namespace pw_frozen {
using namespace ATV::cpu::pw::internal;
inline double value_local(const internal_data& d, const GList& g,
                          std::span<const Vec3> r, std::span<const C> rho,
                          double volume) {
    const auto vg = nc_vloc_to_gspace(d, g, volume);
    double e = 0;
    for (int i = 0; i < g.count(); ++i) {
        C structure{};
        for (const auto& a : r)
            structure += std::polar(1., -g.g[i].dot(a));
        e += volume * std::real(std::conj(rho[i]) * structure) * vg[i];
    }
    return e;
}
inline double value_nl(const internal_data& d, const GList& g,
                       std::span<const Vec3> r, Orbitals o, double volume) {
    std::vector<Vec3> atoms(r.begin(), r.end());
    std::vector<std::vector<PWWorkspace::KBProjG>> proj;
    setup_kb_projectors(proj, g, d, atoms, volume);
    double e = 0;
    for (std::size_t b = 0; b < o.occupations.size(); ++b)
        for (const auto& site : proj)
            for (const auto& p : site)
                for (int m = 0; m < 2 * p.l + 1; ++m) {
                    C overlap{};
                    for (int i = 0; i < g.count(); ++i)
                        overlap += std::conj(p.beta_G(m, i))
                                   * o.coefficients[b * g.count() + i];
                    e += o.k_weight * o.occupations[b] * p.d
                         * std::norm(overlap);
                }
    return e;
}
inline double value_h(const GList& g, std::span<const C> rho, double volume) {
    PWWorkspace ws;
    ws.gvec = &g;
    ws.n_g = g.count();
    ws.rho_G.resize({rho.size()});
    ws.V_H_G.resize({rho.size()});
    std::copy(rho.begin(), rho.end(), ws.rho_G.begin());
    PoissonSolver poisson;
    poisson.step(ws);
    double e = 0;
    for (int i = 0; i < g.count(); ++i)
        e += .5 * volume * std::real(std::conj(rho[i]) * ws.V_H_G[i]);
    return e;
}
inline double value_t(const GList& g, Orbitals o) {
    double e = 0;
    for (std::size_t b = 0; b < o.occupations.size(); ++b)
        for (int i = 0; i < g.count(); ++i)
            e += o.k_weight * o.occupations[b]
                 * std::norm(o.coefficients[b * g.count() + i]) * .5
                 * (g.g[i] + g.kvec).norm_squared();
    return e;
}
} // namespace pw_frozen
