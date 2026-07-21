/// @file src/cpu/gpw/internal/gpw_forces.cpp
/// @brief Analytic GPW forces (see header).
///
/// Periodic-image consistent with the Γ-point lattice-sum energy: the Pulay
/// (overlap/kinetic/Vpp) derivatives use the periodic image sums, and the grid
/// term collocates the basis and its center-derivatives with the same
/// multi-image window as the SCF density. Reduces to the isolated path when the
/// cell is non-periodic (image counts collapse to 0).

#include "atomiverse/cpu/gpw/internal/gpw_forces.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/gth_integrals.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cpu/gpw/internal/mt_poisson.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"
#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"
#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

using linalg::RealMatrix;

Prim to_prim(const GridPrimitive& p) {
    Prim q;
    q.alpha = p.alpha;
    q.center[0] = p.center[0];
    q.center[1] = p.center[1];
    q.center[2] = p.center[2];
    q.l[0] = p.l[0];
    q.l[1] = p.l[1];
    q.l[2] = p.l[2];
    return q;
}

std::vector<GridBasisFunction> global_basis(const std::vector<GpwAtom>& atoms) {
    std::vector<GridBasisFunction> b;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            b.push_back(bf);
    return b;
}

// d φ / d(center_k) = 2α φ_{l_k+1} − l_k φ_{l_k−1}  (Gaussian center derivative).
GridBasisFunction deriv_bf(const GridBasisFunction& bf, int k) {
    GridBasisFunction d;
    d.reserve(bf.size() * 2);
    for (const auto& p : bf) {
        GridPrimitive up = p;
        up.l[k] += 1;
        up.coeff = p.coeff * 2.0 * p.alpha;
        d.push_back(up);
        if (p.l[k] > 0) {
            GridPrimitive dn = p;
            dn.coeff = p.coeff * (-static_cast<double>(p.l[k]));
            dn.l[k] -= 1;
            d.push_back(dn);
        }
    }
    return d;
}

// Γ-point periodic <a | Σ_C v_C | b> over the given GTH centers: the ket b is
// summed over its lattice images T (within `reach` of a), and each center C is
// taken at its nearest image to the bra·ket product center. Matches the energy's
// gth_matrix_element convention so the force derivative is consistent.
/// @param with_local Include the analytic per-pair V_loc^SR. Must be false when
///        the SCF put V_loc^SR on the grid (cfg.local_pp_on_grid): the grid
///        V_eff term then already carries its basis-derivative part, and
///        `add_local_pp_hf_forces` carries the operator-center part. Including it
///        here as well double-counts and differentiates the wrong functional.
double vpp_images(const GridBasisFunction& a, const GridBasisFunction& b,
                  const std::vector<const GpwAtom*>& centers, const double L[3],
                  const int m[3], double reach, bool with_local) {
    if (a.empty() || b.empty())
        return 0.0;
    const double* ac = a[0].center;
    const double* bc = b[0].center;
    const double reach2 = reach * reach;
    const bool per = m[0] || m[1] || m[2];
    double v = 0.0;
    for (const auto& p : a) {
        const Prim pp = to_prim(p);
        for (const auto& q0 : b) {
            const double cc = p.coeff * q0.coeff;
            if (cc == 0.0)
                continue;
            const double aa = p.alpha;
            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        const double sx = ac[0] - bc[0] - tx * L[0];
                        const double sy = ac[1] - bc[1] - ty * L[1];
                        const double sz = ac[2] - bc[2] - tz * L[2];
                        if (sx * sx + sy * sy + sz * sz > reach2)
                            continue;
                        GridPrimitive q = q0;
                        q.center[0] += tx * L[0];
                        q.center[1] += ty * L[1];
                        q.center[2] += tz * L[2];
                        const double bb = q.alpha, g = aa + bb;
                        double P[3];
                        for (int d = 0; d < 3; ++d)
                            P[d] = (aa * p.center[d] + bb * q.center[d]) / g;
                        const Prim qq = to_prim(q);
                        for (const GpwAtom* Cp : centers) {
                            const GpwAtom& C = *Cp;
                            double apos[3];
                            for (int d = 0; d < 3; ++d) {
                                apos[d] = C.pos[d];
                                if (per && L[d] > 0.0)
                                    apos[d] -= L[d]
                                               * std::nearbyint(
                                                   (C.pos[d] - P[d]) / L[d]);
                            }
                            if (with_local && !C.cloc.empty())
                                v += cc
                                     * gth_local_sr(pp, qq, apos, C.rloc,
                                                    C.cloc);
                            for (const auto& ch : C.nonlocal)
                                v += cc * gth_nonlocal(pp, qq, apos, ch);
                        }
                    }
        }
    }
    return v;
}

} // namespace

std::vector<std::array<double, 3>>
gpw_forces_analytic(const std::vector<GpwAtom>& atoms, const GpwConfig& cfg) {
    GPWSolver solver{cfg};
    solver.build(atoms);
    const GpwResult scf = solver.run(atoms);
    if (!scf.converged)
        throw std::runtime_error("gpw_forces_analytic: SCF did not converge");

    const std::vector<GridBasisFunction> basis = global_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    const int na = static_cast<int>(atoms.size());
    const bool periodic = !cfg.nonperiodic;
    const double eps = cfg.eps_default;

    // Map each atom to its contiguous block of global basis indices, and each
    // global basis index back to its owning atom.
    std::vector<int> atom_off(na + 1, 0);
    for (int a = 0; a < na; ++a)
        atom_off[a + 1] = atom_off[a] + static_cast<int>(atoms[a].basis.size());
    std::vector<int> owner(nb, 0);
    for (int a = 0; a < na; ++a)
        for (int u = atom_off[a]; u < atom_off[a + 1]; ++u)
            owner[u] = a;

    // Γ-point lattice-image extent + per-function radii (same as the SCF build).
    std::vector<double> radii(static_cast<std::size_t>(nb));
    double rmax = 0.0;
    for (int u = 0; u < nb; ++u) {
        radii[static_cast<std::size_t>(u)] =
            basis_function_radius(basis[u], eps);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(u)]);
    }
    int mimg[3];
    pair_image_counts(cfg.cell, periodic, 2.0 * rmax, mimg);

    const RealMatrix& P = scf.P;
    // Energy-weighted density matrix W_μν = Σ_i 2 ε_i C_μi C_νi (occupied).
    RealMatrix W = xt::zeros<double>({nb, nb});
    for (int i = 0; i < cfg.n_occ; ++i) {
        const double w = 2.0 * scf.eps[static_cast<std::size_t>(i)];
        for (int mu = 0; mu < nb; ++mu) {
            const double cmu = w * scf.C(mu, i);
            for (int nu = 0; nu < nb; ++nu)
                W(mu, nu) += cmu * scf.C(nu, i);
        }
    }

    // Center-derivative basis functions d φ_u / dR_u along each axis, plus their
    // radii (the l+1 component reaches slightly farther than φ_u).
    std::vector<std::array<GridBasisFunction, 3>> dphi(nb);
    std::vector<std::array<double, 3>> drad(nb);
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k) {
            dphi[u][k] = deriv_bf(basis[u], k);
            drad[u][k] = basis_function_radius(dphi[u][k], eps);
        }

    // Grid + multi-image collocation (same window as the SCF density).
    UniformGrid grid = make_uniform_grid(cfg.cell, cfg.cutoff_ry);
    const long ng = grid.npts();
    const double wgt = grid.weight();
    const CompactBasisGrid cbg =
        build_compact_basis_grid(grid, basis, periodic, eps);

    // Ionic core density n_c(r) (min-image; narrow cores, matches the SCF).
    std::vector<double> n_ion(static_cast<std::size_t>(ng), 0.0);
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        double s = 0.0;
        for (const auto& a : atoms) {
            const double s2 = a.rloc * a.rloc;
            const double norm = a.z_val * std::pow(2.0 * M_PI * s2, -1.5);
            double dx = r[0] - a.pos[0], dy = r[1] - a.pos[1],
                   dz = r[2] - a.pos[2];
            if (periodic) {
                dx -= grid.L[0] * std::nearbyint(dx / grid.L[0]);
                dy -= grid.L[1] * std::nearbyint(dy / grid.L[1]);
                dz -= grid.L[2] * std::nearbyint(dz / grid.L[2]);
            }
            s += norm * std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * s2));
        }
        n_ion[static_cast<std::size_t>(g)] = s;
    }

    std::vector<double> Pvec(static_cast<std::size_t>(nb) * nb);
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < nb; ++j)
            Pvec[static_cast<std::size_t>(i) * nb + j] = P(i, j);
    std::vector<double> rho = collocate_density(cbg, nb, Pvec); // multi-image χ
    std::vector<double> n_tot(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g)
        n_tot[g] = rho[g] - n_ion[g];

    const PoissonResult es = cfg.nonperiodic ? solve_poisson_mt(grid, n_tot)
                                             : solve_poisson(grid, n_tot);
    const XcOnGrid xc = cfg.xc == XcKind::PBE ? xc_grid_pbe(grid, rho)
                                              : xc_grid_pade(grid, rho);
    const std::vector<double>& v_es = es.v_r;
    // V_eff must be the SAME one the SCF converged against. When the local
    // short-range PP was collocated on the grid (the default), it is part of
    // V_eff — and then the analytic per-pair route must be OFF in the Pulay terms
    // (see internal/local_pp.h). Differentiating a different functional than the
    // SCF converged was a real bug: ZnO analytic-vs-FD 0.53 Ha/Bohr.
    const bool analytic_loc = !cfg.local_pp_on_grid;
    const std::vector<double> v_locsr =
        analytic_loc ? std::vector<double>()
                     : local_pp_grid(grid, atoms, periodic);

    std::vector<double> veff(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g) {
        veff[g] = es.v_r[g] + xc.vxc_r[g];
        if (!v_locsr.empty())
            veff[g] += v_locsr[g];
    }

    // g -> active index into cbg (for the grid term's Pχ lookup).
    std::vector<long> g2a(static_cast<std::size_t>(ng), -1);
    for (std::size_t a = 0; a < cbg.gidx.size(); ++a)
        g2a[static_cast<std::size_t>(cbg.gidx[a])] = static_cast<long>(a);

    // Multi-image collocation of every center-derivative function dφ_u/dR_u,k
    // (flat index f = u*3 + k), so the grid term's ∂χ_u is image-consistent.
    std::vector<GridBasisFunction> dflat(static_cast<std::size_t>(nb) * 3);
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k)
            dflat[static_cast<std::size_t>(u) * 3 + k] = dphi[u][k];
    const CompactBasisGrid cbgd =
        build_compact_basis_grid(grid, dflat, periodic, eps);

    std::vector<std::array<double, 3>> forces(na, {0.0, 0.0, 0.0});

    // ── AO-basis Pulay terms (periodic image sums) ──────────────────────────
    for (int A = 0; A < na; ++A) {
        for (int k = 0; k < 3; ++k) {
            double f = 0.0;
            const std::vector<const GpwAtom*> all = [&] {
                std::vector<const GpwAtom*> v(static_cast<std::size_t>(na));
                for (int a = 0; a < na; ++a)
                    v[static_cast<std::size_t>(a)] = &atoms[a];
                return v;
            }();
            const std::vector<const GpwAtom*> onlyA = {&atoms[A]};

            for (int mu = atom_off[A]; mu < atom_off[A + 1]; ++mu) {
                const GridBasisFunction& dmu = dphi[mu][k];
                const double rdmu = drad[mu][k];
                for (int nu = 0; nu < nb; ++nu) {
                    const double reach =
                        rdmu + radii[static_cast<std::size_t>(nu)];
                    const double s =
                        overlap(dmu, basis[nu], cfg.cell, mimg, reach);
                    const double t =
                        kinetic(dmu, basis[nu], cfg.cell, mimg, reach);
                    f += 2.0 * W(mu, nu) * s; // + W dS
                    f -= 2.0 * P(mu, nu) * t; // - P dT
                    f -=
                        2.0 * P(mu, nu)
                        * vpp_images(dmu, basis[nu], all, cfg.cell, mimg, reach,
                                     analytic_loc); // - P dV_pp(basis)
                }
            }
            // GTH operator-center derivative (∂_C, trans. inv.): +2 Σ_{μ,ν} P dμ·v_A·ν.
            for (int mu = 0; mu < nb; ++mu) {
                const GridBasisFunction& dmu = dphi[mu][k];
                const double rdmu = drad[mu][k];
                for (int nu = 0; nu < nb; ++nu) {
                    const double reach =
                        rdmu + radii[static_cast<std::size_t>(nu)];
                    f += 2.0 * P(mu, nu)
                         * vpp_images(dmu, basis[nu], onlyA, cfg.cell, mimg,
                                      reach, analytic_loc);
                }
            }
            forces[A][k] = f;
        }
    }

    // ── Grid term: -2 Σ_{μ∈A} Σ_g w χ_{dμ}(g) V_eff(g) (Σ_ν P_μν χ_ν(g)) ──────
    // Iterate the derivative collocation; for each (g, dφ_{u,k}) look up the
    // active cbg point at g and contract Pχ_u(g) against V_eff.
    for (std::size_t ad = 0; ad < cbgd.gidx.size(); ++ad) {
        const long g = cbgd.gidx[ad];
        const long a = g2a[static_cast<std::size_t>(g)];
        if (a < 0)
            continue; // χ_ν all zero here ⇒ Pχ = 0
        const double vw = veff[static_cast<std::size_t>(g)] * wgt;
        const std::size_t cs = cbg.off[static_cast<std::size_t>(a)];
        const std::size_t ce = cbg.off[static_cast<std::size_t>(a) + 1];
        const std::size_t ds = cbgd.off[ad];
        const std::size_t de = cbgd.off[ad + 1];
        for (std::size_t id = ds; id < de; ++id) {
            const int fdu = cbgd.fu[id]; // flat index u*3+k
            const int u = fdu / 3, k = fdu % 3;
            const double chi_dmu = cbgd.fv[id];
            // Pχ_u(g) = Σ_ν P_uν χ_ν(g)
            const double* Pu = &Pvec[static_cast<std::size_t>(u) * nb];
            double pchi = 0.0;
            for (std::size_t ib = cs; ib < ce; ++ib)
                pchi += Pu[cbg.fu[ib]] * cbg.fv[ib];
            forces[owner[u]][k] -= 2.0 * vw * chi_dmu * pchi;
        }
    }

    // ── Ionic Hellmann–Feynman: + ∫ V_es ∂n_c^A/∂R_A,k (min-image, matches SCF) ─
    for (int A = 0; A < na; ++A) {
        const GpwAtom& Aatom = atoms[A];
        const double s2 = Aatom.rloc * Aatom.rloc;
        const double norm = Aatom.z_val * std::pow(2.0 * M_PI * s2, -1.5);
        std::array<double, 3> acc = {0.0, 0.0, 0.0};
        for (long g = 0; g < ng; ++g) {
            double r[3];
            grid.point(g, r);
            double d[3] = {r[0] - Aatom.pos[0], r[1] - Aatom.pos[1],
                           r[2] - Aatom.pos[2]};
            if (periodic)
                for (int k = 0; k < 3; ++k)
                    d[k] -= grid.L[k] * std::nearbyint(d[k] / grid.L[k]);
            const double gauss =
                norm
                * std::exp(-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                           / (2.0 * s2));
            const double vg = v_es[static_cast<std::size_t>(g)] * gauss / s2;
            for (int k = 0; k < 3; ++k)
                acc[k] += vg * d[k];
        }
        for (int k = 0; k < 3; ++k)
            forces[A][k] += wgt * acc[k];
    }

    // ── Local-PP Hellmann–Feynman: the grid V_loc^SR moves with its atom ───────
    // The counterpart of dropping gth_local_sr from the Pulay terms. Only for the
    // grid route: in the analytic route this contribution is already inside the
    // per-pair operator-center derivative above.
    if (!analytic_loc)
        add_local_pp_hf_forces(grid, atoms, periodic, rho, forces);

    // Grimme DFT-D3 dispersion forces (additive, geometry-only).
    if (cfg.d3) {
        std::vector<int> Z(static_cast<std::size_t>(na));
        std::vector<std::array<double, 3>> pos(static_cast<std::size_t>(na));
        for (int i = 0; i < na; ++i) {
            Z[static_cast<std::size_t>(i)] =
                atoms[static_cast<std::size_t>(i)].z;
            pos[static_cast<std::size_t>(i)] = {
                atoms[static_cast<std::size_t>(i)].pos[0],
                atoms[static_cast<std::size_t>(i)].pos[1],
                atoms[static_cast<std::size_t>(i)].pos[2]};
        }
        const D3Result d3 = dftd3_pbe(Z, pos, cfg.cell, periodic, D3Params{});
        for (int i = 0; i < na; ++i)
            for (int k = 0; k < 3; ++k)
                forces[i][k] += d3.forces[static_cast<std::size_t>(i)][k];
    }

    return forces;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
