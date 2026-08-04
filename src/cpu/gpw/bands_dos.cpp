/// @file cpu/gpw/bands_dos.cpp
/// @brief Band-structure and multi-k DOS post-processing on GPW.

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

XcOnGridSpin eval_xc_spin(XcKind xc, const UniformGrid& grid,
                          const std::vector<double>& rho_a,
                          const std::vector<double>& rho_b) {
    switch (xc) {
    case XcKind::PBE:
        return xc_grid_pbe_spin(grid, rho_a, rho_b);
    case XcKind::PBEsol:
        return xc_grid_pbesol_spin(grid, rho_a, rho_b);
    case XcKind::RevPBE:
        return xc_grid_revpbe_spin(grid, rho_a, rho_b);
    case XcKind::RPBE:
        return xc_grid_rpbe_spin(grid, rho_a, rho_b);
    case XcKind::BLYP:
        return xc_grid_blyp_spin(grid, rho_a, rho_b);
    case XcKind::OptB88VdW:
        return xc_grid_optb88_vdw_spin(grid, rho_a, rho_b);
    case XcKind::PADE:
    default:
        return xc_grid_pade_lsd(grid, rho_a, rho_b);
    }
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
    const bool uks = method.is_uks();
    const bool use_smear = (method.smearing != Smearing::none);
    if (!uks) {
        if (method.n_occ < 1)
            throw std::invalid_argument(
                "gpw::Bands::compute: method.n_occ must be >= 1 "
                "(closed-shell RKS)");
        if (n_bands < method.n_occ)
            throw std::invalid_argument(
                "gpw::Bands::compute: n_bands must be >= n_occ so the gap is "
                "defined");
    } else {
        if (method.n_alpha < 1 || method.n_beta < 1)
            throw std::invalid_argument(
                "gpw::Bands::compute: UKS requires n_alpha, n_beta >= 1");
        const int n_need = std::max(method.n_alpha, method.n_beta);
        if (n_bands < n_need)
            throw std::invalid_argument(
                "gpw::Bands::compute: n_bands must cover max(n_alpha, n_beta)");
    }
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
    if (uks
        && (scf.P_k_beta.empty()
            || static_cast<int>(scf.P_k_beta.size()) != scf.n_kpts))
        throw std::runtime_error(
            "gpw::Bands::compute: UKS SCF did not populate multi-k P_k_beta");

    // ── 2. Geometry + freeze V_eff (RKS) or dual V_eff^σ (UKS) ─────────
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
    std::vector<double> rho_a, rho_b;
    if (uks) {
        rho_a.assign(static_cast<std::size_t>(grid.npts()), 0.0);
        rho_b.assign(static_cast<std::size_t>(grid.npts()), 0.0);
        for (int ik = 0; ik < nk_scf; ++ik) {
            const double wk = scf_kpts[static_cast<std::size_t>(ik)].weight;
            internal::collocate_density_k(
                kbg_scf, ik, scf.P_k[static_cast<std::size_t>(ik)], wk, rho_a);
            internal::collocate_density_k(
                kbg_scf, ik, scf.P_k_beta[static_cast<std::size_t>(ik)], wk,
                rho_b);
        }
        for (std::size_t g = 0; g < rho.size(); ++g)
            rho[g] = rho_a[g] + rho_b[g];
    } else {
        for (int ik = 0; ik < nk_scf; ++ik) {
            internal::collocate_density_k(
                kbg_scf, ik, scf.P_k[static_cast<std::size_t>(ik)],
                scf_kpts[static_cast<std::size_t>(ik)].weight, rho);
        }
    }

    std::vector<double> n_tot(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g)
        n_tot[g] = rho[g] - n_ion[g];
    const PoissonResult es = solve_poisson(grid, n_tot);

    std::vector<double> veff, veff_a, veff_b;
    if (uks) {
        const XcOnGridSpin xc = eval_xc_spin(method.xc, grid, rho_a, rho_b);
        veff_a.resize(rho.size());
        veff_b.resize(rho.size());
        for (std::size_t g = 0; g < rho.size(); ++g) {
            veff_a[g] = es.v_r[g] + xc.vxc_a[g] + v_locsr[g];
            veff_b[g] = es.v_r[g] + xc.vxc_b[g] + v_locsr[g];
        }
    } else {
        const XcOnGrid xc = eval_xc(method.xc, grid, rho);
        veff.resize(rho.size());
        for (std::size_t g = 0; g < rho.size(); ++g)
            veff[g] = es.v_r[g] + xc.vxc_r[g] + v_locsr[g];
    }

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
    out.nspin = uks ? 2 : 1;
    out.scf_converged = scf.converged;
    out.energy_ha = scf.e_total;
    out.e_fermi_ha = use_smear ? scf.e_fermi : 0.0;
    out.eigenvalues_ha.resize(static_cast<std::size_t>(nk));
    if (uks)
        out.eigenvalues_beta_ha.resize(static_cast<std::size_t>(nk));

    const int n_occ_min = uks ? std::max(method.n_alpha, method.n_beta)
                              : method.n_occ;

    for (int ik = 0; ik < nk; ++ik) {
        internal::KOrth orth(S_k[static_cast<std::size_t>(ik)],
                             method.lindep_threshold);
        auto diag_channel = [&](const std::vector<double>& v_eff,
                                std::vector<double>& row) {
            const CplxMatrix Vg =
                internal::integrate_potential_k(kbg_path, ik, v_eff);
            const CplxMatrix H = T_k[static_cast<std::size_t>(ik)]
                                 + Vnl_k[static_cast<std::size_t>(ik)] + Vg;
            auto [eps, C] = internal::diagonalize_k(H, orth);
            (void)C;
            const int m = static_cast<int>(eps.size());
            const int nrep = std::min(n_bands, m);
            if (nrep < n_occ_min)
                throw std::runtime_error(
                    "gpw::Bands::compute: only " + std::to_string(m)
                    + " independent directions at path k #"
                    + std::to_string(ik) + " but need >= "
                    + std::to_string(n_occ_min));
            row.assign(eps.begin(), eps.begin() + nrep);
        };

        if (uks) {
            diag_channel(veff_a, out.eigenvalues_ha[static_cast<std::size_t>(ik)]);
            diag_channel(veff_b,
                         out.eigenvalues_beta_ha[static_cast<std::size_t>(ik)]);
        } else {
            diag_channel(veff, out.eigenvalues_ha[static_cast<std::size_t>(ik)]);
        }
    }

    // Gap character for fixed-occupation paths only (metallic: E_F is the pin).
    if (!use_smear) {
        if (!uks && n_bands > method.n_occ) {
            out.gap_ha = gap_from_spectrum(
                out.eigenvalues_ha, method.n_occ, out.e_vbm_ha, out.e_cbm_ha,
                out.vbm_k, out.cbm_k, out.vbm_band, out.cbm_band);
        } else if (uks) {
            // Joint α/β VBM–CBM (integer-fill seed), same policy as Dos.
            double e_vbm = -1e300;
            double e_cbm = 1e300;
            bool have_vbm = false;
            bool have_cbm = false;
            int vk = 0, ck = 0, vb = 0, cb = 0;
            auto scan = [&](const std::vector<std::vector<double>>& ev,
                            int n_occ) {
                for (std::size_t ik = 0; ik < ev.size(); ++ik) {
                    const auto& row = ev[ik];
                    if (n_occ > 0 && static_cast<int>(row.size()) >= n_occ) {
                        const double v =
                            row[static_cast<std::size_t>(n_occ - 1)];
                        if (!have_vbm || v > e_vbm) {
                            e_vbm = v;
                            vk = static_cast<int>(ik);
                            vb = n_occ - 1;
                            have_vbm = true;
                        }
                    }
                    if (static_cast<int>(row.size()) > n_occ) {
                        const double c = row[static_cast<std::size_t>(n_occ)];
                        if (!have_cbm || c < e_cbm) {
                            e_cbm = c;
                            ck = static_cast<int>(ik);
                            cb = n_occ;
                            have_cbm = true;
                        }
                    }
                }
            };
            scan(out.eigenvalues_ha, method.n_alpha);
            scan(out.eigenvalues_beta_ha, method.n_beta);
            if (have_vbm && have_cbm) {
                out.e_vbm_ha = e_vbm;
                out.e_cbm_ha = e_cbm;
                out.gap_ha = e_cbm - e_vbm;
                out.vbm_k = vk;
                out.cbm_k = ck;
                out.vbm_band = vb;
                out.cbm_band = cb;
            }
        }
    }
    return out;
}

void gaussian_dos(const std::vector<double>& evals,
                  const std::vector<double>& weights, double e_min,
                  double e_max, int n_bins, double sigma,
                  std::vector<double>& energy, std::vector<double>& dos,
                  double spin_degeneracy) {
    if (evals.size() != weights.size())
        throw std::invalid_argument(
            "gaussian_dos: evals/weights size mismatch");
    if (n_bins < 2)
        throw std::invalid_argument("gaussian_dos: n_bins < 2");
    if (!(sigma > 0.0))
        throw std::invalid_argument("gaussian_dos: sigma must be > 0");
    if (!(e_max > e_min))
        throw std::invalid_argument("gaussian_dos: need e_max > e_min");
    if (!(spin_degeneracy > 0.0))
        throw std::invalid_argument(
            "gaussian_dos: spin_degeneracy must be > 0");

    energy.resize(static_cast<std::size_t>(n_bins));
    dos.assign(static_cast<std::size_t>(n_bins), 0.0);
    const double de = (e_max - e_min) / static_cast<double>(n_bins - 1);
    const double pref = spin_degeneracy
                        / (std::sqrt(2.0 * 3.14159265358979323846) * sigma);
    const double inv_2s2 = 1.0 / (2.0 * sigma * sigma);

    for (int i = 0; i < n_bins; ++i)
        energy[static_cast<std::size_t>(i)] =
            e_min + de * static_cast<double>(i);

    for (std::size_t n = 0; n < evals.size(); ++n) {
        const double e = evals[n];
        const double w = weights[n];
        for (int i = 0; i < n_bins; ++i) {
            const double x = energy[static_cast<std::size_t>(i)] - e;
            dos[static_cast<std::size_t>(i)] +=
                w * pref * std::exp(-x * x * inv_2s2);
        }
    }
}

double integrate_dos_to(const std::vector<double>& energy,
                        const std::vector<double>& dos, double e_cut) {
    if (energy.size() != dos.size())
        throw std::invalid_argument(
            "integrate_dos_to: energy/dos size mismatch");
    if (energy.size() < 2)
        throw std::invalid_argument("integrate_dos_to: need >= 2 bins");

    const double de = (energy.back() - energy.front())
                      / static_cast<double>(static_cast<int>(energy.size()) - 1);
    double integ = 0;
    for (std::size_t i = 0; i + 1 < energy.size(); ++i) {
        if (energy[i] > e_cut)
            break;
        const double e1 = energy[i];
        const double e2 = energy[i + 1];
        const double d1 = dos[i];
        const double d2 = dos[i + 1];
        if (e2 <= e_cut) {
            integ += 0.5 * (d1 + d2) * de;
        } else {
            const double t = (e_cut - e1) / (e2 - e1);
            const double d_c = d1 + t * (d2 - d1);
            integ += 0.5 * (d1 + d_c) * (e_cut - e1);
            break;
        }
    }
    return integ;
}

namespace {

std::vector<std::vector<double>>
cap_bands(const std::vector<std::vector<double>>& evals, int n_bands) {
    if (n_bands <= 0)
        return evals;
    std::vector<std::vector<double>> out(evals.size());
    for (std::size_t ik = 0; ik < evals.size(); ++ik) {
        const int m = static_cast<int>(evals[ik].size());
        const int take = std::min(n_bands, m);
        out[ik].assign(evals[ik].begin(),
                       evals[ik].begin() + static_cast<std::ptrdiff_t>(take));
    }
    return out;
}

void flatten_spectrum(const std::vector<std::vector<double>>& evals,
                      const std::vector<double>& k_weights, int n_bands_use,
                      std::vector<double>& flat_e, std::vector<double>& flat_w) {
    const int nk = static_cast<int>(evals.size());
    flat_e.clear();
    flat_w.clear();
    flat_e.reserve(static_cast<std::size_t>(nk * std::max(n_bands_use, 1)));
    flat_w.reserve(flat_e.capacity());
    for (int ik = 0; ik < nk; ++ik) {
        const double w = k_weights[static_cast<std::size_t>(ik)];
        const int m = static_cast<int>(evals[static_cast<std::size_t>(ik)].size());
        const int take = (n_bands_use > 0) ? std::min(n_bands_use, m) : m;
        for (int b = 0; b < take; ++b) {
            flat_e.push_back(
                evals[static_cast<std::size_t>(ik)][static_cast<std::size_t>(b)]);
            flat_w.push_back(w);
        }
    }
}

} // namespace

DosResult Dos::compute(GpwDft& method,
                       const std::vector<GpwAtom>& atoms) const {
    if (k_mesh < 1 && !use_method_k_list)
        throw std::invalid_argument("Dos::compute: k_mesh must be >= 1");
    if (!(sigma_ha > 0.0))
        throw std::invalid_argument("Dos::compute: sigma_ha must be > 0");
    if (n_bins < 2)
        throw std::invalid_argument("Dos::compute: n_bins must be >= 2");
    if (n_bands < 0)
        throw std::invalid_argument("Dos::compute: n_bands must be >= 0");

    if (!use_method_k_list) {
        method.k_points.clear();
        method.k_point_weights.clear();
        method.k_mesh[0] = k_mesh;
        method.k_mesh[1] = k_mesh;
        method.k_mesh[2] = k_mesh;
    }

    // Multi-k path requires a non-Γ mesh or an explicit list.
    const bool has_explicit = !method.k_points.empty();
    const bool multi =
        has_explicit || method.k_mesh[0] != 1 || method.k_mesh[1] != 1
        || method.k_mesh[2] != 1;
    if (!multi)
        throw std::invalid_argument(
            "Dos::compute: multi-k mesh required (set Dos::k_mesh >= 2, or "
            "pass use_method_k_list with a non-Γ GpwDft::k_mesh / k_points). "
            "Example: dos.k_mesh = 2; dos.compute(method, atoms);");

    const GpwResult scf = method.run(atoms);

    if (scf.eps_k.empty() || scf.k_weights.empty() || scf.n_kpts < 1)
        throw std::runtime_error(
            "Dos::compute: multi-k SCF left empty eps_k / k_weights "
            "(internal multi-k payload missing)");
    if (static_cast<int>(scf.eps_k.size()) != scf.n_kpts
        || static_cast<int>(scf.k_weights.size()) != scf.n_kpts)
        throw std::runtime_error(
            "Dos::compute: eps_k / k_weights length mismatch vs n_kpts");

    DosResult out;
    out.scf_converged = scf.converged;
    out.scf_energy_ha = scf.e_total;
    out.n_kpts = scf.n_kpts;
    out.e_fermi_ha = scf.e_fermi;

    const bool uks = method.is_uks();
    const bool use_smear = (method.smearing != Smearing::none);
    out.nspin = uks ? 2 : 1;

    // Cap band count; GPW diagonalizes the full AO spectrum.
    const int m0 = static_cast<int>(scf.eps_k[0].size());
    const int n_use = (n_bands > 0) ? std::min(n_bands, m0) : m0;
    if (n_use < 1)
        throw std::runtime_error("Dos::compute: no bands in eps_k");
    out.n_bands = n_use;

    auto evals = cap_bands(scf.eps_k, n_use);

    if (!uks) {
        // ── Restricted (RKS) ───────────────────────────────────────────
        if (method.n_occ < 1)
            throw std::invalid_argument(
                "Dos::compute: RKS requires method.n_occ >= 1");
        if (n_use < method.n_occ)
            throw std::invalid_argument(
                "Dos::compute: n_bands (" + std::to_string(n_use)
                + ") < n_occ (" + std::to_string(method.n_occ) + ")");

        if (n_use > method.n_occ) {
            int vk = 0, ck = 0, vb = 0, cb = 0;
            out.gap_ha = gap_from_spectrum(evals, method.n_occ, out.e_vbm_ha,
                                          out.e_cbm_ha, vk, ck, vb, cb);
        } else {
            // Only occupied spectrum: VBM = max of band n_occ-1; no CBM.
            out.e_vbm_ha = -1e300;
            for (const auto& row : evals) {
                const double v =
                    row[static_cast<std::size_t>(method.n_occ - 1)];
                if (v > out.e_vbm_ha)
                    out.e_vbm_ha = v;
            }
            out.e_cbm_ha = out.e_vbm_ha;
            out.gap_ha = 0;
        }

        std::vector<double> flat_e, flat_w;
        flatten_spectrum(evals, scf.k_weights, n_use, flat_e, flat_w);

        double emin = e_min_ha;
        double emax = e_max_ha;
        if (!(emax > emin)) {
            emin = *std::min_element(flat_e.begin(), flat_e.end())
                   - 4.0 * sigma_ha;
            emax = *std::max_element(flat_e.begin(), flat_e.end())
                   + 4.0 * sigma_ha;
        }

        gaussian_dos(flat_e, flat_w, emin, emax, n_bins, sigma_ha,
                     out.energy_ha, out.dos, /*spin_degeneracy=*/2.0);

        const double e_cut =
            use_smear ? out.e_fermi_ha : out.e_vbm_ha;
        out.n_elec_to_vbm = integrate_dos_to(out.energy_ha, out.dos, e_cut);
        return out;
    }

    // ── Collinear UKS multi-k ───────────────────────────────────────────
    if (scf.eps_k_beta.empty()
        || static_cast<int>(scf.eps_k_beta.size()) != scf.n_kpts)
        throw std::runtime_error(
            "Dos::compute: UKS multi-k missing eps_k_beta payload");
    if (method.n_alpha < 1 || method.n_beta < 1)
        throw std::invalid_argument(
            "Dos::compute: UKS requires n_alpha, n_beta >= 1");

    auto evals_b = cap_bands(scf.eps_k_beta, n_use);
    if (n_use < method.n_alpha || n_use < method.n_beta)
        throw std::invalid_argument(
            "Dos::compute: n_bands must cover max(n_alpha, n_beta)");

    // Joint VBM / CBM over α+β (integer-fill seed).
    {
        double e_vbm = -1e300;
        double e_cbm = 1e300;
        bool have_vbm = false;
        bool have_cbm = false;
        const auto scan = [&](const std::vector<std::vector<double>>& ev,
                              int n_occ) {
            for (const auto& row : ev) {
                if (n_occ > 0
                    && static_cast<int>(row.size()) >= n_occ) {
                    const double v =
                        row[static_cast<std::size_t>(n_occ - 1)];
                    if (!have_vbm || v > e_vbm) {
                        e_vbm = v;
                        have_vbm = true;
                    }
                }
                if (static_cast<int>(row.size()) > n_occ) {
                    const double c =
                        row[static_cast<std::size_t>(n_occ)];
                    if (!have_cbm || c < e_cbm) {
                        e_cbm = c;
                        have_cbm = true;
                    }
                }
            }
        };
        scan(evals, method.n_alpha);
        scan(evals_b, method.n_beta);
        if (have_vbm)
            out.e_vbm_ha = e_vbm;
        if (have_cbm && have_vbm) {
            out.e_cbm_ha = e_cbm;
            out.gap_ha = e_cbm - e_vbm;
        }
    }

    std::vector<double> flat_a, flat_b, flat_w, flat_all, flat_w_all;
    flatten_spectrum(evals, scf.k_weights, n_use, flat_a, flat_w);
    flatten_spectrum(evals_b, scf.k_weights, n_use, flat_b, flat_w);
    flat_all.reserve(flat_a.size() + flat_b.size());
    flat_w_all.reserve(flat_a.size() + flat_b.size());
    for (std::size_t i = 0; i < flat_a.size(); ++i) {
        flat_all.push_back(flat_a[i]);
        flat_w_all.push_back(flat_w[i]);
        flat_all.push_back(flat_b[i]);
        flat_w_all.push_back(flat_w[i]);
    }

    double emin = e_min_ha;
    double emax = e_max_ha;
    if (!(emax > emin)) {
        emin = *std::min_element(flat_all.begin(), flat_all.end())
               - 4.0 * sigma_ha;
        emax = *std::max_element(flat_all.begin(), flat_all.end())
               + 4.0 * sigma_ha;
    }

    gaussian_dos(flat_a, flat_w, emin, emax, n_bins, sigma_ha, out.energy_ha,
                 out.dos_alpha, /*spin_degeneracy=*/1.0);
    std::vector<double> e_beta;
    gaussian_dos(flat_b, flat_w, emin, emax, n_bins, sigma_ha, e_beta,
                 out.dos_beta, /*spin_degeneracy=*/1.0);
    gaussian_dos(flat_all, flat_w_all, emin, emax, n_bins, sigma_ha,
                 out.energy_ha, out.dos, /*spin_degeneracy=*/1.0);

    const double e_cut = use_smear ? out.e_fermi_ha : out.e_vbm_ha;
    out.n_elec_to_vbm = integrate_dos_to(out.energy_ha, out.dos, e_cut);
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
