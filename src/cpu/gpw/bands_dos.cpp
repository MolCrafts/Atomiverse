/// @file cpu/gpw/bands_dos.cpp
/// @brief Band-structure post-processing on frozen GPW \(V_\mathrm{eff}\).

#include "atomiverse/cpu/gpw/bands_dos.h"

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"
#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

using internal::CplxMatrix;
using internal::KPoint;
using Cplx = std::complex<double>;

constexpr double kBuildScreen = 1e-12;

std::vector<GridBasisFunction> flatten_basis(const std::vector<GpwAtom>& atoms) {
    std::vector<GridBasisFunction> basis;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            basis.push_back(bf);
    return basis;
}

/// GTH Gaussian ionic cores (same form as gpw_solver.cpp ionic_density).
std::vector<double> ionic_density_grid(const UniformGrid& grid,
                                       const std::vector<GpwAtom>& atoms,
                                       bool periodic) {
    const long ng = grid.npts();
    std::vector<double> n(static_cast<std::size_t>(ng), 0.0);
#pragma omp parallel for schedule(static)
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
        n[static_cast<std::size_t>(g)] = s;
    }
    return n;
}

std::vector<KPoint> kpoints_from_frac(
    const std::vector<std::array<double, 3>>& kfrac,
    const std::vector<double>& weights) {
    std::vector<KPoint> kpts;
    kpts.reserve(kfrac.size());
    for (std::size_t i = 0; i < kfrac.size(); ++i) {
        KPoint kp;
        kp.frac[0] = kfrac[i][0];
        kp.frac[1] = kfrac[i][1];
        kp.frac[2] = kfrac[i][2];
        kp.weight = weights.empty() ? 1.0 / static_cast<double>(kfrac.size())
                                    : weights[i];
        kpts.push_back(kp);
    }
    return kpts;
}

std::vector<std::array<double, 3>>
lerp_segment(const std::array<double, 3>& a, const std::array<double, 3>& b,
             int n_seg, bool include_start) {
    std::vector<std::array<double, 3>> pts;
    if (n_seg < 2)
        n_seg = 2;
    const int start = include_start ? 0 : 1;
    for (int i = start; i < n_seg; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n_seg - 1);
        pts.push_back({a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
                       a[2] + t * (b[2] - a[2])});
    }
    return pts;
}

XcOnGrid eval_xc(XcKind xc, const UniformGrid& grid,
                 const std::vector<double>& rho) {
    switch (xc) {
    case XcKind::PBE:
        return xc_grid_pbe(grid, rho);
    case XcKind::PBEsol:
        return xc_grid_pbesol(grid, rho);
    case XcKind::RevPBE:
        return xc_grid_revpbe(grid, rho);
    case XcKind::RPBE:
        return xc_grid_rpbe(grid, rho);
    case XcKind::BLYP:
        return xc_grid_blyp(grid, rho);
    case XcKind::OptB88VdW:
        return xc_grid_optb88_vdw(grid, rho);
    case XcKind::PADE:
    default:
        return xc_grid_pade(grid, rho);
    }
}

void reject_unsupported(const GpwDft& m) {
    if (m.is_uks())
        throw std::invalid_argument(
            "gpw::Bands: collinear UKS band structure is not product yet "
            "(closed-shell RKS only; use n_occ, not n_alpha/n_beta)");
    if (m.smearing != Smearing::none)
        throw std::invalid_argument(
            "gpw::Bands: metallic smearing is not product yet "
            "(fixed integer occupations only)");
    if (m.nonperiodic)
        throw std::invalid_argument(
            "gpw::Bands: nonperiodic (MT) cells are not supported "
            "(need a periodic orthorhombic cell)");
    if (!m.local_pp_on_grid)
        throw std::invalid_argument(
            "gpw::Bands: requires local_pp_on_grid=true "
            "(grid-collocated GTH local SR PP)");
    if (m.ngrids > 1)
        throw std::invalid_argument(
            "gpw::Bands: multigrid NSCF is not product yet "
            "(set ngrids=1)");
    if (!m.plus_u.empty())
        throw std::invalid_argument(
            "gpw::Bands: DFT+U on the band path is not product yet "
            "(clear plus_u)");
    if (m.d3 || m.d3bj || m.d4 || m.ts || m.xdm || m.mbd || m.vdw_df
        || m.vdw_df2 || m.rvv10)
        throw std::invalid_argument(
            "gpw::Bands: dispersion on the band path is not product yet "
            "(clear d3/d3bj/d4/ts/xdm/mbd/vdw_df/vdw_df2/rvv10)");
}

} // namespace

double gap_from_spectrum(const std::vector<std::vector<double>>& evals,
                         int n_occ, double& e_vbm, double& e_cbm, int& vbm_k,
                         int& cbm_k, int& vbm_b, int& cbm_b) {
    if (evals.empty())
        throw std::invalid_argument("gpw::gap_from_spectrum: empty spectrum");
    if (n_occ < 1)
        throw std::invalid_argument("gpw::gap_from_spectrum: n_occ must be >= 1");

    e_vbm = -1e300;
    e_cbm = 1e300;
    vbm_k = cbm_k = 0;
    vbm_b = cbm_b = 0;
    bool have_vbm = false;
    bool have_cbm = false;

    for (std::size_t ik = 0; ik < evals.size(); ++ik) {
        const int nb = static_cast<int>(evals[ik].size());
        if (nb < n_occ)
            throw std::invalid_argument(
                "gpw::gap_from_spectrum: band count < n_occ at a k-point");
        const double v = evals[ik][static_cast<std::size_t>(n_occ - 1)];
        if (!have_vbm || v > e_vbm) {
            e_vbm = v;
            vbm_k = static_cast<int>(ik);
            vbm_b = n_occ - 1;
            have_vbm = true;
        }
        if (nb > n_occ) {
            const double c = evals[ik][static_cast<std::size_t>(n_occ)];
            if (!have_cbm || c < e_cbm) {
                e_cbm = c;
                cbm_k = static_cast<int>(ik);
                cbm_b = n_occ;
                have_cbm = true;
            }
        }
    }
    if (!have_vbm || !have_cbm)
        throw std::invalid_argument(
            "gpw::gap_from_spectrum: need occupied and empty states "
            "(raise n_bands above n_occ)");
    return e_cbm - e_vbm;
}

void Bands::set_diamond_path(int n_seg) {
    if (n_seg < 2)
        n_seg = 2;
    // Same FCC/diamond SPECIAL_POINT list as cpu::pw::Bands / CP2K twins.
    const std::array<double, 3> G{0.0, 0.0, 0.0};
    const std::array<double, 3> X{0.5, 0.0, 0.5};
    const std::array<double, 3> W{0.5, 0.25, 0.75};
    const std::array<double, 3> L{0.5, 0.5, 0.5};
    const std::array<double, 3> K{0.375, 0.375, 0.75};

    std::vector<std::array<double, 3>> path;
    auto append = [&](const std::array<double, 3>& a,
                      const std::array<double, 3>& b, bool include_start) {
        auto seg = lerp_segment(a, b, n_seg, include_start);
        path.insert(path.end(), seg.begin(), seg.end());
    };
    append(G, X, true);
    append(X, W, false);
    append(W, L, false);
    append(L, G, false);
    append(G, K, false);

    kfrac = std::move(path);
}

BandsResult Bands::compute(GpwDft& method,
                           const std::vector<GpwAtom>& atoms) const {
    if (n_bands < 2)
        throw std::invalid_argument("gpw::Bands::compute: n_bands must be >= 2");
    if (method.n_occ < 1)
        throw std::invalid_argument(
            "gpw::Bands::compute: method.n_occ must be >= 1 (closed-shell RKS)");
    if (n_bands < method.n_occ)
        throw std::invalid_argument(
            "gpw::Bands::compute: n_bands must be >= n_occ so the gap is defined");
    reject_unsupported(method);

    std::vector<std::array<double, 3>> path = kfrac;
    if (path.empty()) {
        Bands tmp;
        tmp.set_diamond_path(4);
        path = tmp.kfrac;
    }
    if (path.empty())
        throw std::invalid_argument("gpw::Bands::compute: empty k-path");

    // ── 1. Multi-k SCF density ──────────────────────────────────────────
    // Pure Γ (k_mesh={1,1,1} and empty k_points) has no P_k payload — force
    // the k path via an explicit Γ point so the density lives on P_k.
    const bool pure_gamma = method.k_points.empty() && method.k_mesh[0] == 1
                            && method.k_mesh[1] == 1 && method.k_mesh[2] == 1;
    std::vector<std::array<double, 3>> saved_kpts = method.k_points;
    std::vector<double> saved_kw = method.k_point_weights;
    int saved_mesh[3] = {method.k_mesh[0], method.k_mesh[1], method.k_mesh[2]};
    if (pure_gamma) {
        method.k_points = {{0.0, 0.0, 0.0}};
        method.k_point_weights = {1.0};
    }

    const GpwResult scf = method.run(atoms);

    // Restore method k settings (path is independent of the SCF mesh).
    method.k_points = saved_kpts;
    method.k_point_weights = saved_kw;
    method.k_mesh[0] = saved_mesh[0];
    method.k_mesh[1] = saved_mesh[1];
    method.k_mesh[2] = saved_mesh[2];

    if (scf.P_k.empty() || scf.k_frac.empty()
        || static_cast<int>(scf.P_k.size()) != scf.n_kpts)
        throw std::runtime_error(
            "gpw::Bands::compute: SCF did not populate multi-k P_k "
            "(need k_mesh / k_points on the method for the density)");

    // ── 2. Geometry + freeze V_eff from SCF {P_k} ───────────────────────
    const auto basis = flatten_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    if (nb < 1)
        throw std::runtime_error("gpw::Bands::compute: empty AO basis");

    std::vector<double> radii(static_cast<std::size_t>(nb), 0.0);
    double rmax = 0.0;
    for (int i = 0; i < nb; ++i) {
        radii[static_cast<std::size_t>(i)] =
            basis_function_radius(basis[static_cast<std::size_t>(i)],
                                  method.eps_default);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(i)]);
    }
    int mimg[3] = {0, 0, 0};
    pair_image_counts(method.cell, /*periodic=*/true, 2.0 * rmax, mimg);

    const UniformGrid grid = make_uniform_grid(method.cell, method.cutoff_ry);
    const std::vector<double> n_ion =
        ionic_density_grid(grid, atoms, /*periodic=*/true);
    const std::vector<double> v_locsr =
        local_pp_grid(grid, atoms, /*periodic=*/true);

    std::vector<KPoint> scf_kpts =
        kpoints_from_frac(scf.k_frac, scf.k_weights);
    const int nk_scf = static_cast<int>(scf_kpts.size());
    const internal::KpointBasisGrid kbg_scf =
        internal::build_kpoint_basis_grid(grid, basis, scf_kpts,
                                          method.eps_default);

    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.0);
    for (int ik = 0; ik < nk_scf; ++ik) {
        internal::collocate_density_k(kbg_scf, ik,
                                      scf.P_k[static_cast<std::size_t>(ik)],
                                      scf_kpts[static_cast<std::size_t>(ik)].weight,
                                      rho);
    }

    std::vector<double> n_tot(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g)
        n_tot[g] = rho[g] - n_ion[g];
    const PoissonResult es = solve_poisson(grid, n_tot);
    const XcOnGrid xc = eval_xc(method.xc, grid, rho);

    std::vector<double> veff(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g)
        veff[g] = es.v_r[g] + xc.vxc_r[g] + v_locsr[g];

    // ── 3. NSCF diagonalization on the path ─────────────────────────────
    std::vector<double> path_w(path.size(), 1.0 / static_cast<double>(path.size()));
    std::vector<KPoint> path_kpts = kpoints_from_frac(path, path_w);
    const int nk = static_cast<int>(path_kpts.size());

    std::vector<CplxMatrix> S_k, T_k;
    internal::build_overlap_kinetic_k(basis, radii, method.cell, mimg,
                                      kBuildScreen, path_kpts, S_k, T_k);
    std::vector<CplxMatrix> Vnl_k(static_cast<std::size_t>(nk),
                                  CplxMatrix(xt::zeros<Cplx>({nb, nb})));
    internal::add_nonlocal_kb_k(Vnl_k, basis, atoms, radii, method.cell, mimg,
                                kBuildScreen, path_kpts);

    const internal::KpointBasisGrid kbg_path =
        internal::build_kpoint_basis_grid(grid, basis, path_kpts,
                                          method.eps_default);

    BandsResult out;
    out.kfrac = path;
    out.nspin = 1;
    out.scf_converged = scf.converged;
    out.energy_ha = scf.e_total;
    out.eigenvalues_ha.resize(static_cast<std::size_t>(nk));

    for (int ik = 0; ik < nk; ++ik) {
        internal::KOrth orth(S_k[static_cast<std::size_t>(ik)],
                             method.lindep_threshold);
        const CplxMatrix Vg =
            internal::integrate_potential_k(kbg_path, ik, veff);
        const CplxMatrix H = T_k[static_cast<std::size_t>(ik)]
                             + Vnl_k[static_cast<std::size_t>(ik)] + Vg;
        auto [eps, C] = internal::diagonalize_k(H, orth);
        (void)C;
        const int m = static_cast<int>(eps.size());
        const int nrep = std::min(n_bands, m);
        if (nrep < method.n_occ)
            throw std::runtime_error(
                "gpw::Bands::compute: only " + std::to_string(m)
                + " independent directions at path k #" + std::to_string(ik)
                + " but n_occ=" + std::to_string(method.n_occ));
        out.eigenvalues_ha[static_cast<std::size_t>(ik)].assign(
            eps.begin(), eps.begin() + nrep);
    }

    if (n_bands > method.n_occ) {
        out.gap_ha =
            gap_from_spectrum(out.eigenvalues_ha, method.n_occ, out.e_vbm_ha,
                              out.e_cbm_ha, out.vbm_k, out.cbm_k, out.vbm_band,
                              out.cbm_band);
    }
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
