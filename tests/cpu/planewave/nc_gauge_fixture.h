// Synthetic admitted NC operators: two radial projectors per s/p/d channel.
// Deliberately include signed/off-diagonal D, complex bands and unequal prefixes.
#pragma once
#include "atomiverse/cpu/planewave/internal/pw_radial_derivatives.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <span>

namespace nc_gauge_test {
using namespace ATV;
using namespace ATV::cpu::pw::internal;
using C = std::complex<AccReal>;
inline NcPotentialData potential(AccReal scale, bool unequal_prefix = false) {
    NcPotentialData d;
    d.atomic_number = d.z_val = 1;
    d.l_max = 2;
    constexpr int n = 2001;
    constexpr AccReal dr = .004;
    d.r.resize(n);
    d.rab.assign(n, dr);
    d.v_loc.resize(n);
    for (int i = 0; i < n; ++i) {
        const AccReal r = i * dr;
        d.r[i] = r;
        d.v_loc[i] = i ? -std::erf(r / (std::sqrt(2.) * .4)) / r
                       : -std::sqrt(2. / M_PI) / .4;
    }
    for (int ell = 0; ell <= 2; ++ell)
        for (AccReal alpha : {.7, 1.1}) {
            NcPotentialData::KBProjector p;
            p.l = ell;
            p.radial_points = n;
            p.beta.resize(n);
            for (int i = 0; i < n; ++i)
                p.beta[i] = scale * std::pow(d.r[i], ell + 1)
                            * std::exp(-alpha * d.r[i] * d.r[i]);
            d.kb.push_back(std::move(p));
        }
    if (unequal_prefix) {
        d.kb[1].radial_points = 578;
        d.kb[3].radial_points = 733;
        d.kb[5].radial_points = 888;
    }
    d.d_ij.assign(36, 0);
    for (int ell = 0; ell <= 2; ++ell) {
        const int p = 2 * ell;
        d.d_ij[6 * p + p] = .7 / (scale * scale);
        d.d_ij[6 * (p + 1) + p + 1] = -.4 / (scale * scale);
        d.d_ij[6 * p + p + 1] = d.d_ij[6 * (p + 1) + p] =
            -.21 / (scale * scale);
    }
    d.validate();
    return d;
}
inline GVectorList modes(bool nonzero_k) {
    GVectorList g;
    g.g = {{0, 0, 0},      {.3, -.2, .4},          {-.3, .2, -.4},
           {1.1, .4, -.7}, {-1.1, -.4, .7},        {-.7, .8, .2},
           {.7, -.8, -.2}, {1e-12, -2e-12, 3e-12}, {-1e-12, 2e-12, -3e-12}};
    g.nx = g.ny = g.nz = 16;
    if (nonzero_k)
        g.kvec = {.11, -.07, .03};
    for (int i = 0; i < g.count(); ++i) {
        g.h.push_back(i);
        g.k.push_back(0);
        g.l.push_back(0);
        g.ekin.push_back(.5 * (g.g[i] + g.kvec).norm_squared());
    }
    return g;
}
inline std::vector<C> wave(const GVectorList& g) {
    std::vector<C> c(3 * g.count());
    for (int b = 0; b < 3; ++b) {
        AccReal norm = 0;
        for (int i = 0; i < g.count(); ++i) {
            c[b * g.count() + i] = {std::sin(.37 * (i + 1) * (b + 1)),
                                    std::cos(.29 * (i + 2) * (b + 2))};
            norm += std::norm(c[b * g.count() + i]);
        }
        for (int i = 0; i < g.count(); ++i)
            c[b * g.count() + i] /= std::sqrt(norm);
    }
    return c;
}
inline void prepare(PWProblem& problem, const NcPotentialData& d,
                    const GVectorList& g, std::span<const C> c) {
    problem.gv0 = g;
    problem.gv_k = {g};
    problem.vol_bohr = 125;
    problem.n_bands = 3;
    problem.potential = std::make_shared<const NcPotentialData>(d);
    auto& ws = problem.ws;
    ws.gvec = &problem.gv0;
    ws.n_g = g.count();
    ws.omega = problem.vol_bohr;
    ws.atom_pos_bohr = {{.2, .4, .7}, {1.1, -.3, .5}};
    ws.fft_nx = g.nx;
    ws.fft_ny = g.ny;
    ws.fft_nz = g.nz;
    ws.nr = g.nx * g.ny * g.nz;
    ws.fft = std::make_unique<FFTPlan3D>(g.nx, g.ny, g.nz);
    ws.T_G.resize({std::size_t(g.count())});
    for (int i = 0; i < g.count(); ++i)
        ws.T_G[i] = g.ekin[i];
    ws.V_eff_r = xt::zeros<AccReal>({std::size_t(ws.nr)});
    ws.wfc = xt::zeros<C>({std::size_t(3), std::size_t(g.count())});
    std::copy(c.begin(), c.end(), ws.wfc.begin());
    problem.kb_proj_k.resize(1);
    setup_kb_projectors(problem.kb_proj_k[0], g, d, ws.atom_pos_bohr, ws.omega);
    ws.kb_proj = problem.kb_proj_k[0];
}
// Independent original full-D contraction: no eigenvalue rotation, packed
// device columns or projector amplitude admission used by the actual H path.
inline std::vector<C> full_d_h(const NcPotentialData& d, const PWProblem& p,
                               std::span<const C> c) {
    const auto& g = p.gv0;
    const int np = d.n_proj(), ng = g.count();
    std::vector<C> answer(c.size());
    for (int b = 0; b < 3; ++b)
        for (int i = 0; i < ng; ++i)
            answer[b * ng + i] = g.ekin[i] * c[b * ng + i];
    std::vector<ProjectorCartesianJet> jets(np * ng);
    for (int j = 0; j < np; ++j)
        for (int i = 0; i < ng; ++i)
            jets[j * ng + i] =
                nc_projector_cartesian_jet(d, j, g.g[i] + g.kvec);
    const AccReal pref = 4 * M_PI / std::sqrt(p.ws.omega);
    for (const auto& atom : p.ws.atom_pos_bohr) {
        std::vector<std::array<std::vector<C>, 5>> beta(np);
        for (int j = 0; j < np; ++j) {
            const int ell = d.kb[j].l;
            const C il = ell == 0 ? C{1, 0} : ell == 1 ? C{0, -1} : C{-1, 0};
            for (int m = 0; m < 2 * ell + 1; ++m) {
                beta[j][m].resize(ng);
                for (int i = 0; i < ng; ++i)
                    beta[j][m][i] =
                        pref * il
                        * std::polar(AccReal(1), -(g.g[i] + g.kvec).dot(atom))
                        * jets[j * ng + i].value[m];
            }
        }
        for (int b = 0; b < 3; ++b) {
            std::vector<std::array<C, 5>> overlap(np), weighted(np);
            for (int j = 0; j < np; ++j)
                for (int m = 0; m < 2 * d.kb[j].l + 1; ++m)
                    for (int i = 0; i < ng; ++i)
                        overlap[j][m] +=
                            std::conj(beta[j][m][i]) * c[b * ng + i];
            for (int j = 0; j < np; ++j)
                for (int k = 0; k < np; ++k)
                    if (d.kb[j].l == d.kb[k].l)
                        for (int m = 0; m < 2 * d.kb[j].l + 1; ++m)
                            weighted[j][m] +=
                                d.d_ij[j * np + k] * overlap[k][m];
            for (int j = 0; j < np; ++j)
                for (int m = 0; m < 2 * d.kb[j].l + 1; ++m)
                    for (int i = 0; i < ng; ++i)
                        answer[b * ng + i] += beta[j][m][i] * weighted[j][m];
        }
    }
    return answer;
}
} // namespace nc_gauge_test
