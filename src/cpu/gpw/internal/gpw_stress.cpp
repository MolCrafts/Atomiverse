/// @file cpu/gpw/internal/gpw_stress.cpp
/// @brief GPW cell stress: Hellmann–Feynman fixed-P product path + multi-SCF FD.
///
/// Analytic product path (gpw_stress / gpw_stress_analytic_impl):
///   1. One full SCF at the reference geometry → density matrix P (Γ RKS),
///      spin pair {Pα,Pβ,Wα,Wβ} (Γ UKS), per-k {P_k, W_k} from C_occ_k /
///      eps_k (multi-k RKS, occ=2), or per-k spin {P_k^σ, W_k^σ} from
///      C_occ_k / C_occ_k_beta (multi-k UKS, occ=1). Dudarev DFT+U
///      (`plus_u`) is supported on Γ and multi-k: \(E_U[n(P,S)]\) (Γ) or
///      BZ-averaged \(E_U[n^\sigma(\{P_k^\sigma\},\{S_k\})]\) (k-path) enters
///      the fixed-P Lagrangian (Mulliken \(n\) at strained \(S\)/\(S_k\)).
///   2. Central FD of the *non-self-consistent* total energy E[P; ε] under
///      orthorhombic diagonal strain at fixed fractional coordinates. Geometry-
///      dependent T, V_pp / S_k,T_k,V_nl^k, grid, Poisson, XC (and optional
///      D3/D4/TS/XDM/MBD geometry-only dispersion and density-dependent
///      nonlocal vdW-DF/DF2/rVV10 \(E_{\mathrm{c}}^{\mathrm{nl}}\) at fixed
///      \(\rho_P\)) are rebuilt; the SCF is *not* re-converged. This is the
///      GPW form of the Hellmann–Feynman cell theorem used for plane-wave
///      Nielsen–Martin in this tree (Nielsen & Martin, PRB 32, 3780 (1985);
///      Quickstep energy surface). Multi-k sums the fixed-P Lagrangian over
///      sampled k.
///
/// Oracle path (gpw_stress_fd / gpw_stress_fd_impl): full multi-SCF strain-FD
/// of E_tot (2 SCFs per diagonal axis).
///
/// Off-diagonal shear remains 0 (orthorhombic GPW cell only). Sign matches
/// CP2K STRESS_TENSOR: σ_aa = −(1/Ω) ∂E/∂ε_aa.

#include "atomiverse/cpu/gpw/internal/gpw_stress.h"

#include <array>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/dft_plus_u.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"
#include "atomiverse/cpu/scf/dftd4.h"
#include "atomiverse/cpu/gpw/internal/gth_integrals.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cpu/gpw/internal/mt_poisson.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"
#include "atomiverse/cpu/gpw/internal/vdw_df.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"
#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"
#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {
namespace {

using linalg::RealMatrix;

GpwDft copy_gpw(const GpwDft& m) {
    GpwDft p;
    for (int d = 0; d < 3; ++d)
        p.cell[d] = m.cell[d];
    p.cutoff_ry = m.cutoff_ry;
    p.ngrids = m.ngrids;
    p.rel_cutoff = m.rel_cutoff;
    p.n_occ = m.n_occ;
    p.n_alpha = m.n_alpha;
    p.n_beta = m.n_beta;
    p.max_iter = m.max_iter;
    p.tol = m.tol;
    p.eps_default = m.eps_default;
    p.mix = m.mix;
    p.diis_depth = m.diis_depth;
    p.lindep_threshold = m.lindep_threshold;
    p.level_shift = m.level_shift;
    p.xc = m.xc;
    p.nonperiodic = m.nonperiodic;
    p.local_pp_on_grid = m.local_pp_on_grid;
    p.d3 = m.d3;
    p.d3bj = m.d3bj;
    p.d3_atm = m.d3_atm;
    p.d3_rcut_bohr = m.d3_rcut_bohr;
    p.d4 = m.d4;
    p.d4_atm = m.d4_atm;
    p.ts = m.ts;
    p.xdm = m.xdm;
    p.mbd = m.mbd;
    for (int d = 0; d < 3; ++d)
        p.mbd_k_mesh[d] = m.mbd_k_mesh[d];
    p.vdw_df = m.vdw_df;
    p.vdw_df2 = m.vdw_df2;
    p.rvv10 = m.rvv10;
    p.hirshfeld = m.hirshfeld;
    p.use_ot = m.use_ot;
    p.plus_u = m.plus_u;
    p.atom_spin_guess = m.atom_spin_guess;
    for (int d = 0; d < 3; ++d)
        p.k_mesh[d] = m.k_mesh[d];
    p.k_time_reversal = m.k_time_reversal;
    p.k_points = m.k_points;
    p.k_point_weights = m.k_point_weights;
    // Metallic free-energy path (FD/MP) must survive the stress-side copy;
    // otherwise a smear SCF silently degrades to integer occupations.
    p.smearing = m.smearing;
    p.degauss = m.degauss;
    p.mp_order = m.mp_order;
    p.extra_bands = m.extra_bands;
    return p;
}

/// Scale atom positions and all basis-function centers along axis `ax` by
/// factor `s` (fixed fractional coordinates under orthorhombic strain).
std::vector<GpwAtom> scale_axis(const std::vector<GpwAtom>& atoms, int ax,
                                double s) {
    std::vector<GpwAtom> out = atoms;
    for (auto& a : out) {
        a.pos[ax] *= s;
        for (auto& bf : a.basis)
            for (auto& p : bf)
                p.center[ax] *= s;
    }
    return out;
}

double scf_energy_ha(const std::vector<GpwAtom>& atoms, const GpwDft& method,
                     const linalg::RealMatrix* guess_C) {
    GpwDft solver = copy_gpw(method);
    solver.build(atoms);
    const GpwResult& r = solver.run(atoms, guess_C);
    if (!r.converged)
        throw std::runtime_error("gpw_stress_fd: SCF did not converge");
    return r.e_total;
}

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

// Γ-point periodic <a | Σ_C v_C | b> — same convention as gpw_forces / SCF.
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

double core_self_energy(const GpwAtom& a) {
    return a.z_val * a.z_val / (2.0 * a.rloc * std::sqrt(M_PI));
}

bool uses_kpath(const GpwDft& cfg) {
    // Match SCF dispatch: metallic smear always uses the Bloch path.
    return !cfg.k_points.empty() || cfg.k_mesh[0] != 1 || cfg.k_mesh[1] != 1
           || cfg.k_mesh[2] != 1 || cfg.smearing != Smearing::none;
}

constexpr double kBuildScreen = 1e-12;

using Cplx = std::complex<double>;
using CplxMatrix = internal::CplxMatrix;
using KPoint = internal::KPoint;
using KpointBasisGrid = internal::KpointBasisGrid;
using internal::build_overlap_kinetic_k;
using internal::add_nonlocal_kb_k;
using internal::build_kpoint_basis_grid;
using internal::collocate_density_k;
using internal::monkhorst_pack;

double ctrace_re(const CplxMatrix& P, const CplxMatrix& M, int nb) {
    double s = 0.0;
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < nb; ++j)
            s += (P(i, j) * M(j, i)).real();
    return s;
}

/// Occupied Bloch density \(P = \mathrm{occ}\sum_o c_o c_o^\dagger\).
/// RKS uses occ=2; UKS per-spin uses occ=1.
CplxMatrix density_from_occ(const std::vector<Cplx>& C_occ, int nb, int n_occ,
                            double occ = 2.0) {
    CplxMatrix P = xt::zeros<Cplx>({nb, nb});
    if (n_occ <= 0)
        return P;
    for (int i = 0; i < nb; ++i)
        for (int o = 0; o < n_occ; ++o) {
            const Cplx ci = occ * C_occ[static_cast<std::size_t>(i) * n_occ
                                       + static_cast<std::size_t>(o)];
            for (int j = 0; j < nb; ++j)
                P(i, j) +=
                    ci
                    * std::conj(C_occ[static_cast<std::size_t>(j) * n_occ
                                      + static_cast<std::size_t>(o)]);
        }
    return P;
}

/// Energy-weighted density \(W = \mathrm{occ}\sum_o \varepsilon_o c_o c_o^\dagger\).
CplxMatrix energy_weighted_from_occ(const std::vector<Cplx>& C_occ,
                                    const std::vector<double>& eps, int nb,
                                    int n_occ, double occ = 2.0) {
    CplxMatrix W = xt::zeros<Cplx>({nb, nb});
    if (n_occ <= 0)
        return W;
    for (int o = 0; o < n_occ; ++o) {
        const double w = occ * eps[static_cast<std::size_t>(o)];
        for (int i = 0; i < nb; ++i) {
            const Cplx ci =
                w * C_occ[static_cast<std::size_t>(i) * n_occ
                          + static_cast<std::size_t>(o)];
            for (int j = 0; j < nb; ++j)
                W(i, j) +=
                    ci
                    * std::conj(C_occ[static_cast<std::size_t>(j) * n_occ
                                      + static_cast<std::size_t>(o)]);
        }
    }
    return W;
}

/// Density from full Bloch MOs + fractional band occupations (metallic
/// free-energy path; matches forces: \(P=\sum_b f_b c_b c_b^\dagger\)).
CplxMatrix density_from_frac(const CplxMatrix& C,
                             const std::vector<double>& f) {
    const int nb = static_cast<int>(C.shape(0));
    const int m = static_cast<int>(C.shape(1));
    const int nfill = std::min(m, static_cast<int>(f.size()));
    CplxMatrix P = xt::zeros<Cplx>({nb, nb});
    for (int o = 0; o < nfill; ++o) {
        const double fo = f[static_cast<std::size_t>(o)];
        if (!(std::abs(fo) > 1e-14))
            continue;
        for (int i = 0; i < nb; ++i) {
            const Cplx ci = fo * C(i, o);
            for (int j = 0; j < nb; ++j)
                P(i, j) += ci * std::conj(C(j, o));
        }
    }
    return P;
}

/// Energy-weighted density from full Bloch MOs + fractional occupations
/// \(W=\sum_b f_b\varepsilon_b c_b c_b^\dagger\).
CplxMatrix energy_weighted_from_frac(const CplxMatrix& C,
                                     const std::vector<double>& eps,
                                     const std::vector<double>& f) {
    const int nb = static_cast<int>(C.shape(0));
    const int m = static_cast<int>(C.shape(1));
    const int nfill = std::min(
        m, std::min(static_cast<int>(eps.size()), static_cast<int>(f.size())));
    CplxMatrix W = xt::zeros<Cplx>({nb, nb});
    for (int o = 0; o < nfill; ++o) {
        const double fo = f[static_cast<std::size_t>(o)];
        if (!(std::abs(fo) > 1e-14))
            continue;
        const double w = fo * eps[static_cast<std::size_t>(o)];
        for (int i = 0; i < nb; ++i) {
            const Cplx ci = w * C(i, o);
            for (int j = 0; j < nb; ++j)
                W(i, j) += ci * std::conj(C(j, o));
        }
    }
    return W;
}

std::vector<KPoint> resolve_kpts(const GpwDft& cfg) {
    if (!cfg.k_points.empty()) {
        std::vector<KPoint> kpts;
        kpts.reserve(cfg.k_points.size());
        for (std::size_t i = 0; i < cfg.k_points.size(); ++i) {
            KPoint kp;
            kp.frac[0] = cfg.k_points[i][0];
            kp.frac[1] = cfg.k_points[i][1];
            kp.frac[2] = cfg.k_points[i][2];
            kp.weight = cfg.k_point_weights[i];
            kpts.push_back(kp);
        }
        return kpts;
    }
    return monkhorst_pack(cfg.k_mesh, cfg.k_time_reversal);
}

/// @brief Hellmann–Feynman fixed-{P_k,W_k} Lagrangian energy on a k mesh.
///
/// RKS:
///   E_L = Σ_k w_k [ Re Tr(P_k T_k) + Re Tr(P_k V_nl^k) − Re Tr(W_k S_k) ]
///       + E_locgrid[ρ] + E_es[ρ−n_c] + E_xc[ρ] + E_disp + E_U[n(P/2,{S_k})]
/// UKS (optional Pa_k / Pb_k for spin densities; P_k/W_k are spin totals):
///   E_L = Σ_k w_k [ Re Tr(P^tot T) + Re Tr(P^tot V_nl) − Re Tr(W^tot S) ]
///       + E_locgrid[ρα+ρβ] + E_es + E_xc[ρα,ρβ] + E_disp
///       + E_U[n^α({Pa_k},{S_k})] + E_U[n^β({Pb_k},{S_k})]
///
/// Dudarev \(E_U\) uses BZ-averaged Mulliken occupations at fixed spin
/// densities and strained \(S_k\) (same formula as multi-k SCF). \(W_k\) is
/// built from KS eigenvalues that already include \(V_U\).
///
/// Density via collocate_density_k. Requires local_pp_on_grid (k foothold).
double energy_fixed_P_k(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                        const std::vector<CplxMatrix>& P_k,
                        const std::vector<CplxMatrix>& W_k,
                        const std::vector<KPoint>& kpts,
                        const std::vector<CplxMatrix>* Pa_k = nullptr,
                        const std::vector<CplxMatrix>* Pb_k = nullptr,
                        const std::vector<double>* vol_ratio = nullptr) {
    const bool uks = Pa_k != nullptr;
    if (uks && Pb_k == nullptr)
        throw std::invalid_argument(
            "gpw_stress analytic k: UKS requires Pa_k and Pb_k");
    if (!uks && cfg.is_uks())
        throw std::invalid_argument(
            "gpw_stress analytic k: UKS requires spin densities Pa_k/Pb_k");
    if (cfg.nonperiodic)
        throw std::invalid_argument(
            "gpw_stress analytic k: nonperiodic not supported");
    if (!cfg.local_pp_on_grid)
        throw std::invalid_argument(
            "gpw_stress analytic k: requires local_pp_on_grid");
    // Spin XC path matches multi-k UKS SCF / forces and Γ-UKS stress
    // (PADE LSD + pure GGA spin: PBE / PBEsol / RevPBE / RPBE / BLYP / OptB88).
    if (uks && cfg.xc != XcKind::PADE && cfg.xc != XcKind::PBE
        && cfg.xc != XcKind::PBEsol && cfg.xc != XcKind::RevPBE
        && cfg.xc != XcKind::RPBE && cfg.xc != XcKind::BLYP
        && cfg.xc != XcKind::OptB88VdW)
        throw std::invalid_argument(
            "gpw_stress analytic k UKS: XcKind must be PADE, PBE, PBEsol, "
            "RevPBE, RPBE, BLYP, or OptB88VdW");
    const int n_disp = (cfg.d3 ? 1 : 0) + (cfg.d3bj ? 1 : 0) + (cfg.d4 ? 1 : 0)
                       + (cfg.ts ? 1 : 0) + (cfg.xdm ? 1 : 0)
                       + (cfg.mbd ? 1 : 0);
    if (n_disp > 1)
        throw std::invalid_argument(
            "gpw_stress analytic k: d3, d3bj, d4, ts, xdm, and mbd are "
            "mutually exclusive");

    const std::vector<GridBasisFunction> basis = global_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    const int nk = static_cast<int>(kpts.size());
    if (static_cast<int>(P_k.size()) != nk || static_cast<int>(W_k.size()) != nk)
        throw std::invalid_argument(
            "gpw_stress analytic k: P_k/W_k size mismatch vs kpts");
    if (uks
        && (static_cast<int>(Pa_k->size()) != nk
            || static_cast<int>(Pb_k->size()) != nk))
        throw std::invalid_argument(
            "gpw_stress analytic k: Pa_k/Pb_k size mismatch vs kpts");

    const double eps = cfg.eps_default;
    std::vector<double> radii(static_cast<std::size_t>(nb));
    double rmax = 0.0;
    for (int u = 0; u < nb; ++u) {
        radii[static_cast<std::size_t>(u)] =
            basis_function_radius(basis[u], eps);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(u)]);
    }
    int mimg[3];
    pair_image_counts(cfg.cell, /*periodic=*/true, 2.0 * rmax, mimg);

    std::vector<CplxMatrix> S_k, T_k;
    build_overlap_kinetic_k(basis, radii, cfg.cell, mimg, kBuildScreen, kpts,
                            S_k, T_k);
    std::vector<CplxMatrix> Vnl_k(static_cast<std::size_t>(nk),
                                  CplxMatrix(xt::zeros<Cplx>({nb, nb})));
    add_nonlocal_kb_k(Vnl_k, basis, atoms, radii, cfg.cell, mimg, kBuildScreen,
                      kpts);

    double e_kin = 0.0, e_nl = 0.0, e_pulay = 0.0;
    for (int ik = 0; ik < nk; ++ik) {
        const double wk = kpts[static_cast<std::size_t>(ik)].weight;
        e_kin +=
            wk * ctrace_re(P_k[static_cast<std::size_t>(ik)],
                           T_k[static_cast<std::size_t>(ik)], nb);
        e_nl += wk * ctrace_re(P_k[static_cast<std::size_t>(ik)],
                               Vnl_k[static_cast<std::size_t>(ik)], nb);
        e_pulay +=
            wk * ctrace_re(W_k[static_cast<std::size_t>(ik)],
                           S_k[static_cast<std::size_t>(ik)], nb);
    }

    UniformGrid grid = make_uniform_grid(cfg.cell, cfg.cutoff_ry);
    const long ng = grid.npts();
    const double wgt = grid.weight();
    const KpointBasisGrid kbg =
        build_kpoint_basis_grid(grid, basis, kpts, eps);

    std::vector<double> rho(static_cast<std::size_t>(ng), 0.0);
    std::vector<double> rho_a, rho_b;
    if (uks) {
        rho_a.assign(static_cast<std::size_t>(ng), 0.0);
        rho_b.assign(static_cast<std::size_t>(ng), 0.0);
        for (int ik = 0; ik < nk; ++ik) {
            const double wk = kpts[static_cast<std::size_t>(ik)].weight;
            collocate_density_k(kbg, ik, (*Pa_k)[static_cast<std::size_t>(ik)],
                                wk, rho_a);
            collocate_density_k(kbg, ik, (*Pb_k)[static_cast<std::size_t>(ik)],
                                wk, rho_b);
        }
        for (std::size_t g = 0; g < rho.size(); ++g)
            rho[g] = rho_a[g] + rho_b[g];
    } else {
        for (int ik = 0; ik < nk; ++ik)
            collocate_density_k(kbg, ik, P_k[static_cast<std::size_t>(ik)],
                                kpts[static_cast<std::size_t>(ik)].weight, rho);
    }

    // Ionic core density n_c (min-image; matches SCF / forces).
    std::vector<double> n_ion(static_cast<std::size_t>(ng), 0.0);
    double e_self = 0.0;
    for (const auto& a : atoms)
        e_self += core_self_energy(a);
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        double s = 0.0;
        for (const auto& a : atoms) {
            const double s2 = a.rloc * a.rloc;
            const double norm = a.z_val * std::pow(2.0 * M_PI * s2, -1.5);
            double dx = r[0] - a.pos[0], dy = r[1] - a.pos[1],
                   dz = r[2] - a.pos[2];
            dx -= grid.L[0] * std::nearbyint(dx / grid.L[0]);
            dy -= grid.L[1] * std::nearbyint(dy / grid.L[1]);
            dz -= grid.L[2] * std::nearbyint(dz / grid.L[2]);
            s += norm * std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * s2));
        }
        n_ion[static_cast<std::size_t>(g)] = s;
    }

    std::vector<double> n_tot(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g)
        n_tot[g] = rho[g] - n_ion[g];

    const PoissonResult es = solve_poisson(grid, n_tot);
    double e_xc = 0.0;
    if (uks) {
        const XcOnGridSpin xc =
            cfg.xc == XcKind::PBE
                ? xc_grid_pbe_spin(grid, rho_a, rho_b)
                : (cfg.xc == XcKind::PBEsol
                       ? xc_grid_pbesol_spin(grid, rho_a, rho_b)
                       : (cfg.xc == XcKind::RevPBE
                              ? xc_grid_revpbe_spin(grid, rho_a, rho_b)
                              : (cfg.xc == XcKind::RPBE
                                     ? xc_grid_rpbe_spin(grid, rho_a, rho_b)
                                     : (cfg.xc == XcKind::BLYP
                                            ? xc_grid_blyp_spin(grid, rho_a,
                                                                rho_b)
                                            : (cfg.xc == XcKind::OptB88VdW
                                                   ? xc_grid_optb88_vdw_spin(
                                                         grid, rho_a, rho_b)
                                                   : xc_grid_pade_lsd(
                                                         grid, rho_a,
                                                         rho_b))))));
        e_xc = xc.e_xc;
    } else {
        const XcOnGrid xc =
            cfg.xc == XcKind::PBE
                ? xc_grid_pbe(grid, rho)
                : (cfg.xc == XcKind::PBEsol
                       ? xc_grid_pbesol(grid, rho)
                       : (cfg.xc == XcKind::RevPBE
                              ? xc_grid_revpbe(grid, rho)
                              : (cfg.xc == XcKind::RPBE
                                     ? xc_grid_rpbe(grid, rho)
                                     : (cfg.xc == XcKind::BLYP
                                            ? xc_grid_blyp(grid, rho)
                                            : (cfg.xc == XcKind::OptB88VdW
                                                   ? xc_grid_optb88_vdw(
                                                         grid, rho)
                                                   : xc_grid_pade(grid,
                                                                  rho))))));
        e_xc = xc.e_xc;
    }

    const std::vector<double> v_locsr =
        local_pp_grid(grid, atoms, /*periodic=*/true);
    double e_locgrid = 0.0;
    for (std::size_t g = 0; g < rho.size(); ++g)
        e_locgrid += rho[g] * v_locsr[g];
    e_locgrid *= wgt;

    const double e_es = es.e_hartree - e_self;

    double e_disp = 0.0;
    if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
        // Density-dependent nonlocal correlation at fixed ρ_P on the strained
        // cell (Hellmann–Feynman fixed-P; matches multi-SCF FD of E_nl).
        e_disp = nonlocal_vdw_on_grid(grid, rho, cfg.vdw_df, cfg.vdw_df2,
                                      cfg.rvv10, /*potential=*/false)
                     .energy;
    } else if (cfg.d3 || cfg.d3bj || cfg.d4 || cfg.ts || cfg.xdm || cfg.mbd) {
        const int na = static_cast<int>(atoms.size());
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
        constexpr double kAng2Bohr = 1.0 / 0.52917720859;
        if (cfg.mbd) {
            // Geometry-only MBD@rsSCS, rebuilt at the strained cell exactly
            // as on the Gamma path (energy_fixed_P). The k path is already
            // periodic-only, so no nonperiodic guard is needed here;
            // mbd_k_mesh all-zero selects isolated free-atom MBD, non-zero
            // the Ewald + MBD k-mesh lattice sum.
            const bool mbd_per = cfg.mbd_k_mesh[0] > 0 || cfg.mbd_k_mesh[1] > 0
                                 || cfg.mbd_k_mesh[2] > 0;
            // Fixed Hirshfeld volumes from the reference SCF (density-chain
            // out of scope — same policy as ionic forces).
            const std::vector<double> empty;
            const std::vector<double>& vr =
                (vol_ratio != nullptr) ? *vol_ratio : empty;
            if (cfg.hirshfeld && vr.empty())
                throw std::runtime_error(
                    "gpw_stress MBD: hirshfeld requires SCF volume_ratios");
            e_disp = mbd_rsscs_pbe(Z, pos, cfg.cell, mbd_per, 0.83, 6.0, vr,
                                   mbd_per ? cfg.mbd_k_mesh : nullptr)
                         .energy;
        } else if (cfg.xdm) {
            constexpr double kRcutXdm = 50.0 * kAng2Bohr;
            e_disp =
                xdm_pbe(Z, pos, cfg.cell, /*periodic=*/true, kRcutXdm).energy;
        } else if (cfg.ts) {
            constexpr double kRcutTs = 50.0 * kAng2Bohr;
            const std::vector<double> empty;
            const std::vector<double>& vr =
                (vol_ratio != nullptr) ? *vol_ratio : empty;
            if (cfg.hirshfeld && vr.empty())
                throw std::runtime_error(
                    "gpw_stress TS: hirshfeld requires SCF volume_ratios");
            e_disp = ts_pbe(Z, pos, cfg.cell, /*periodic=*/true, kRcutTs, 0.94,
                            20.0, vr)
                         .energy;
        } else if (cfg.d4) {
            const double s9 = cfg.d4_atm ? 1.0 : 0.0;
            e_disp = cpu::scf::dftd4_energy(
                Z, pos, cfg.cell, /*periodic=*/true, /*s6=*/1.0,
                /*s8=*/0.95948085, /*a1=*/0.38574991, /*a2=*/4.80688534,
                /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0, /*charge=*/0.0,
                /*cn_cut=*/30.0, /*disp2_cut=*/60.0, s9);
        } else {
            const D3Result d3 =
                cfg.d3 ? dftd3_pbe(Z, pos, cfg.cell, true, D3Params{},
                                   cfg.d3_rcut_bohr,
                                   cfg.d3_atm)
                       : dftd3_pbe_bj(Z, pos, cfg.cell, true, D3Params{},
                                      cfg.d3_rcut_bohr, 1.0, 0.7875,
                                      0.4289, 4.4407,
                                      /*cp2k_cn=*/true, cfg.d3_atm);
            e_disp = d3.energy;
        }
    }

    // Dudarev DFT+U at fixed spin densities and strained S_k (BZ-averaged
    // Mulliken n({P_k},{S_k})). Serial: all k, no MPI allreduce on occ.
    double e_dftu = 0.0;
    if (!cfg.plus_u.empty()) {
        std::vector<int> z_keys;
        std::vector<PlusUHa> slots;
        z_keys.reserve(cfg.plus_u.size());
        slots.reserve(cfg.plus_u.size());
        for (const auto& pu : cfg.plus_u) {
            if (!(pu.u_minus_j_ev > 0.0) || pu.l < 0)
                continue;
            z_keys.push_back(static_cast<int>(pu.element));
            PlusUHa s;
            s.l = pu.l;
            s.U_ha = pu.u_minus_j_ev / ATV::units::kHartreeInEv;
            slots.push_back(s);
        }
        if (!slots.empty()) {
            std::vector<AoShellTag> tags;
            tags.reserve(static_cast<std::size_t>(nb));
            std::vector<int> z_of_atom(atoms.size());
            for (int ia = 0; ia < static_cast<int>(atoms.size()); ++ia) {
                const auto& a = atoms[static_cast<std::size_t>(ia)];
                z_of_atom[static_cast<std::size_t>(ia)] = a.z;
                const int nbf = static_cast<int>(a.basis.size());
                const int nl = static_cast<int>(a.basis_l.size());
                for (int k = 0; k < nbf; ++k) {
                    AoShellTag t;
                    t.atom = ia;
                    t.l = (k < nl) ? a.basis_l[static_cast<std::size_t>(k)]
                                   : -1;
                    tags.push_back(t);
                }
            }
            if (static_cast<int>(tags.size()) != nb)
                throw std::invalid_argument(
                    "gpw_stress analytic k: AO shell tags size != basis size");
            std::vector<double> weights(static_cast<std::size_t>(nk));
            for (int ik = 0; ik < nk; ++ik)
                weights[static_cast<std::size_t>(ik)] =
                    kpts[static_cast<std::size_t>(ik)].weight;
            RealMatrix V_discard;
            if (uks) {
                e_dftu += build_dftu_potential_k(
                    *Pa_k, S_k, weights, /*ik_begin=*/0, /*ik_stride=*/1,
                    /*allreduce_occ=*/false, tags, z_of_atom, slots, z_keys,
                    V_discard);
                e_dftu += build_dftu_potential_k(
                    *Pb_k, S_k, weights, /*ik_begin=*/0, /*ik_stride=*/1,
                    /*allreduce_occ=*/false, tags, z_of_atom, slots, z_keys,
                    V_discard);
            } else {
                // Closed-shell: two equal spin channels from P_k/2.
                std::vector<CplxMatrix> Pspin_k(static_cast<std::size_t>(nk));
                for (int ik = 0; ik < nk; ++ik)
                    Pspin_k[static_cast<std::size_t>(ik)] =
                        0.5 * P_k[static_cast<std::size_t>(ik)];
                e_dftu = 2.0
                         * build_dftu_potential_k(
                               Pspin_k, S_k, weights, /*ik_begin=*/0,
                               /*ik_stride=*/1, /*allreduce_occ=*/false, tags,
                               z_of_atom, slots, z_keys, V_discard);
            }
        }
    }

    return e_kin + e_nl + e_locgrid + e_es + e_xc + e_disp + e_dftu - e_pulay;
}

/// @brief Hellmann–Feynman Lagrangian energy at fixed (P, W) and geometry.
///
/// RKS:
///   E_L = Tr(P T) + Tr(P V_pp) + E_es[ρ_P] + E_xc[ρ_P] + E_disp
///         + E_U[n(P/2,S)] − Tr(W S)
/// UKS (P = Pα, optional P_beta / W_beta):
///   E_L = Tr((Pα+Pβ) T) + Tr((Pα+Pβ) V_pp) + E_es[ρα+ρβ−n_c]
///         + E_xc[ρα,ρβ] + E_disp + E_U[n(Pα,S)] + E_U[n(Pβ,S)]
///         − Tr(Wα S) − Tr(Wβ S)
///
/// Dudarev \(E_U\) is evaluated at fixed spin density and the strained
/// overlap (Mulliken occupation depends on \(S\)). \(W\) is built from KS
/// eigenvalues that already include \(V_U\).
///
/// At the SCF stationary point the total-energy derivative w.r.t. a geometry
/// parameter equals ∂E_L/∂λ at fixed P and energy-weighted density W
/// (Pulay −W·dS). Dropping −Tr(W S) leaves an incomplete fixed-P surface whose
/// strain FD does not match multi-SCF FD.
double energy_fixed_P(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                      const RealMatrix& P, const RealMatrix& W,
                      const RealMatrix* P_beta = nullptr,
                      const RealMatrix* W_beta = nullptr,
                      const std::vector<double>* vol_ratio = nullptr) {
    const int n_disp = (cfg.d3 ? 1 : 0) + (cfg.d3bj ? 1 : 0) + (cfg.d4 ? 1 : 0)
                       + (cfg.ts ? 1 : 0) + (cfg.mbd ? 1 : 0)
                       + (cfg.xdm ? 1 : 0);
    if (n_disp > 1)
        throw std::invalid_argument(
            "gpw_stress analytic: d3, d3bj, d4, ts, xdm, and mbd are mutually "
            "exclusive");

    const bool uks = P_beta != nullptr;
    if (uks && W_beta == nullptr)
        throw std::invalid_argument(
            "gpw_stress analytic: UKS requires W_beta with P_beta");
    if (!uks && cfg.is_uks())
        throw std::invalid_argument(
            "gpw_stress analytic: UKS configuration requires spin densities");

    const std::vector<GridBasisFunction> basis = global_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    if (static_cast<int>(P.shape(0)) != nb || static_cast<int>(P.shape(1)) != nb
        || static_cast<int>(W.shape(0)) != nb
        || static_cast<int>(W.shape(1)) != nb)
        throw std::invalid_argument(
            "gpw_stress analytic: P/W shape does not match basis");
    if (uks
        && (static_cast<int>(P_beta->shape(0)) != nb
            || static_cast<int>(P_beta->shape(1)) != nb
            || static_cast<int>(W_beta->shape(0)) != nb
            || static_cast<int>(W_beta->shape(1)) != nb))
        throw std::invalid_argument(
            "gpw_stress analytic: P_beta/W_beta shape does not match basis");

    const bool periodic = !cfg.nonperiodic;
    const double eps = cfg.eps_default;
    const bool analytic_loc = !cfg.local_pp_on_grid;

    std::vector<double> radii(static_cast<std::size_t>(nb));
    double rmax = 0.0;
    for (int u = 0; u < nb; ++u) {
        radii[static_cast<std::size_t>(u)] =
            basis_function_radius(basis[u], eps);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(u)]);
    }
    int mimg[3];
    pair_image_counts(cfg.cell, periodic, 2.0 * rmax, mimg);

    std::vector<const GpwAtom*> all(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        all[a] = &atoms[a];

    // One-electron: Tr(Ptot T) + Tr(Ptot V_pp) − Tr(Wtot S) (Pulay).
    // RKS: Ptot = P (closed-shell total), Wtot = W.
    // UKS: Ptot = Pα+Pβ, Wtot = Wα+Wβ (factor-1 energy-weighted densities).
    // Also assemble S for Dudarev Mulliken occupations when plus_u is on.
    const bool do_dftu = !cfg.plus_u.empty();
    RealMatrix Smat =
        do_dftu ? RealMatrix(xt::zeros<double>({nb, nb})) : RealMatrix();
    double e_kin = 0.0;
    double e_pp = 0.0;
    double e_pulay = 0.0;
    for (int mu = 0; mu < nb; ++mu) {
        if (basis[static_cast<std::size_t>(mu)].empty())
            continue;
        for (int nu = 0; nu < nb; ++nu) {
            if (basis[static_cast<std::size_t>(nu)].empty())
                continue;
            const double reach = radii[static_cast<std::size_t>(mu)]
                                 + radii[static_cast<std::size_t>(nu)];
            double s = 0.0, t = 0.0;
            contracted_ST_periodic(basis[mu], basis[nu], cfg.cell, mimg, reach,
                                   s, t);
            const double vpp = vpp_images(basis[mu], basis[nu], all, cfg.cell,
                                          mimg, reach, analytic_loc);
            const double p_tot =
                uks ? (P(mu, nu) + (*P_beta)(mu, nu)) : P(mu, nu);
            const double w_tot =
                uks ? (W(mu, nu) + (*W_beta)(mu, nu)) : W(mu, nu);
            e_kin += p_tot * t;
            e_pp += p_tot * vpp;
            e_pulay += w_tot * s;
            if (do_dftu)
                Smat(mu, nu) = s;
        }
    }

    UniformGrid grid = make_uniform_grid(cfg.cell, cfg.cutoff_ry);
    const long ng = grid.npts();
    const double wgt = grid.weight();
    const CompactBasisGrid cbg =
        build_compact_basis_grid(grid, basis, periodic, eps);

    auto mat_to_vec = [nb](const RealMatrix& M) {
        std::vector<double> v(static_cast<std::size_t>(nb) * nb);
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < nb; ++j)
                v[static_cast<std::size_t>(i) * nb + j] = M(i, j);
        return v;
    };

    std::vector<double> rho;
    std::vector<double> rho_a, rho_b;
    if (uks) {
        rho_a = collocate_density(cbg, nb, mat_to_vec(P));
        rho_b = collocate_density(cbg, nb, mat_to_vec(*P_beta));
        rho.resize(rho_a.size());
        for (std::size_t g = 0; g < rho.size(); ++g)
            rho[g] = rho_a[g] + rho_b[g];
    } else {
        rho = collocate_density(cbg, nb, mat_to_vec(P));
    }

    // Ionic core density n_c (min-image; matches SCF / forces).
    std::vector<double> n_ion(static_cast<std::size_t>(ng), 0.0);
    double e_self = 0.0;
    for (const auto& a : atoms)
        e_self += core_self_energy(a);
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

    std::vector<double> n_tot(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g)
        n_tot[g] = rho[g] - n_ion[g];

    const PoissonResult es = cfg.nonperiodic ? solve_poisson_mt(grid, n_tot)
                                             : solve_poisson(grid, n_tot);

    double e_xc = 0.0;
    if (uks) {
        // Spin XC path matches Γ-UKS SCF / forces (PADE LSD + pure GGA spin).
        if (cfg.xc != XcKind::PADE && cfg.xc != XcKind::PBE
            && cfg.xc != XcKind::PBEsol && cfg.xc != XcKind::RevPBE
            && cfg.xc != XcKind::RPBE && cfg.xc != XcKind::BLYP
            && cfg.xc != XcKind::OptB88VdW)
            throw std::invalid_argument(
                "gpw_stress analytic UKS: XcKind must be PADE, PBE, PBEsol, "
                "RevPBE, RPBE, BLYP, or OptB88VdW");
        const XcOnGridSpin xc =
            cfg.xc == XcKind::PBE
                ? xc_grid_pbe_spin(grid, rho_a, rho_b)
                : (cfg.xc == XcKind::PBEsol
                       ? xc_grid_pbesol_spin(grid, rho_a, rho_b)
                       : (cfg.xc == XcKind::RevPBE
                              ? xc_grid_revpbe_spin(grid, rho_a, rho_b)
                              : (cfg.xc == XcKind::RPBE
                                     ? xc_grid_rpbe_spin(grid, rho_a, rho_b)
                                     : (cfg.xc == XcKind::BLYP
                                            ? xc_grid_blyp_spin(grid, rho_a,
                                                                rho_b)
                                            : (cfg.xc == XcKind::OptB88VdW
                                                   ? xc_grid_optb88_vdw_spin(
                                                         grid, rho_a, rho_b)
                                                   : xc_grid_pade_lsd(
                                                         grid, rho_a,
                                                         rho_b))))));
        e_xc = xc.e_xc;
    } else {
        const XcOnGrid xc =
            cfg.xc == XcKind::PBE
                ? xc_grid_pbe(grid, rho)
                : (cfg.xc == XcKind::PBEsol
                       ? xc_grid_pbesol(grid, rho)
                       : (cfg.xc == XcKind::RevPBE
                              ? xc_grid_revpbe(grid, rho)
                              : (cfg.xc == XcKind::RPBE
                                     ? xc_grid_rpbe(grid, rho)
                                     : (cfg.xc == XcKind::BLYP
                                            ? xc_grid_blyp(grid, rho)
                                            : (cfg.xc == XcKind::OptB88VdW
                                                   ? xc_grid_optb88_vdw(
                                                         grid, rho)
                                                   : xc_grid_pade(grid,
                                                                  rho))))));
        e_xc = xc.e_xc;
    }

    double e_locgrid = 0.0;
    if (!analytic_loc) {
        const std::vector<double> v_locsr =
            local_pp_grid(grid, atoms, periodic);
        for (std::size_t g = 0; g < rho.size(); ++g)
            e_locgrid += rho[g] * v_locsr[g];
        e_locgrid *= wgt;
    }

    const double e_es = es.e_hartree - e_self;

    double e_disp = 0.0;
    if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
        e_disp = nonlocal_vdw_on_grid(grid, rho, cfg.vdw_df, cfg.vdw_df2,
                                      cfg.rvv10, /*potential=*/false)
                     .energy;
    } else if (cfg.d3 || cfg.d3bj || cfg.d4 || cfg.ts || cfg.xdm || cfg.mbd) {
        const int na = static_cast<int>(atoms.size());
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
        constexpr double kAng2Bohr = 1.0 / 0.52917720859;
        if (cfg.mbd) {
            const bool mbd_per = cfg.mbd_k_mesh[0] > 0 || cfg.mbd_k_mesh[1] > 0
                                 || cfg.mbd_k_mesh[2] > 0;
            if (mbd_per && cfg.nonperiodic)
                throw std::runtime_error(
                    "gpw_stress MBD: mbd_k_mesh requires a periodic cell");
            const std::vector<double> empty;
            const std::vector<double>& vr =
                (vol_ratio != nullptr) ? *vol_ratio : empty;
            if (cfg.hirshfeld && vr.empty())
                throw std::runtime_error(
                    "gpw_stress MBD: hirshfeld requires SCF volume_ratios");
            e_disp = mbd_rsscs_pbe(Z, pos, cfg.cell, mbd_per, 0.83, 6.0, vr,
                                  mbd_per ? cfg.mbd_k_mesh : nullptr)
                         .energy;
        } else if (cfg.xdm) {
            constexpr double kRcutXdm = 50.0 * kAng2Bohr;
            e_disp = xdm_pbe(Z, pos, cfg.cell, periodic, kRcutXdm).energy;
        } else if (cfg.ts) {
            constexpr double kRcutTs = 50.0 * kAng2Bohr;
            const std::vector<double> empty;
            const std::vector<double>& vr =
                (vol_ratio != nullptr) ? *vol_ratio : empty;
            if (cfg.hirshfeld && vr.empty())
                throw std::runtime_error(
                    "gpw_stress TS: hirshfeld requires SCF volume_ratios");
            e_disp =
                ts_pbe(Z, pos, cfg.cell, periodic, kRcutTs, 0.94, 20.0, vr)
                    .energy;
        } else if (cfg.d4) {
            const double s9 = cfg.d4_atm ? 1.0 : 0.0;
            e_disp = cpu::scf::dftd4_energy(
                Z, pos, cfg.cell, periodic, /*s6=*/1.0, /*s8=*/0.95948085,
                /*a1=*/0.38574991, /*a2=*/4.80688534, /*ga=*/3.0, /*gc=*/2.0,
                /*wf=*/6.0, /*charge=*/0.0, /*cn_cut=*/30.0, /*disp2_cut=*/60.0,
                s9);
        } else {
            const D3Result d3 =
                cfg.d3 ? dftd3_pbe(Z, pos, cfg.cell, periodic, D3Params{},
                                   cfg.d3_rcut_bohr,
                                   cfg.d3_atm)
                       : dftd3_pbe_bj(Z, pos, cfg.cell, periodic, D3Params{},
                                      cfg.d3_rcut_bohr, 1.0, 0.7875,
                                      0.4289, 4.4407,
                                      /*cp2k_cn=*/true, cfg.d3_atm);
            e_disp = d3.energy;
        }
    }

    // Dudarev DFT+U at fixed spin density and strained S (Mulliken n(P,S)).
    double e_dftu = 0.0;
    if (do_dftu) {
        std::vector<int> z_keys;
        std::vector<PlusUHa> slots;
        z_keys.reserve(cfg.plus_u.size());
        slots.reserve(cfg.plus_u.size());
        for (const auto& pu : cfg.plus_u) {
            if (!(pu.u_minus_j_ev > 0.0) || pu.l < 0)
                continue;
            z_keys.push_back(static_cast<int>(pu.element));
            PlusUHa s;
            s.l = pu.l;
            s.U_ha = pu.u_minus_j_ev / ATV::units::kHartreeInEv;
            slots.push_back(s);
        }
        if (!slots.empty()) {
            std::vector<AoShellTag> tags;
            tags.reserve(static_cast<std::size_t>(nb));
            std::vector<int> z_of_atom(atoms.size());
            for (int ia = 0; ia < static_cast<int>(atoms.size()); ++ia) {
                const auto& a = atoms[static_cast<std::size_t>(ia)];
                z_of_atom[static_cast<std::size_t>(ia)] = a.z;
                const int nbf = static_cast<int>(a.basis.size());
                const int nl = static_cast<int>(a.basis_l.size());
                for (int k = 0; k < nbf; ++k) {
                    AoShellTag t;
                    t.atom = ia;
                    t.l = (k < nl) ? a.basis_l[static_cast<std::size_t>(k)]
                                   : -1;
                    tags.push_back(t);
                }
            }
            if (uks) {
                e_dftu += dft_plus_u_energy(P, Smat, tags, z_of_atom, slots,
                                            z_keys);
                e_dftu += dft_plus_u_energy(*P_beta, Smat, tags, z_of_atom,
                                            slots, z_keys);
            } else {
                // Closed-shell: two equal spin channels from P/2.
                RealMatrix Pspin = xt::zeros<double>({nb, nb});
                for (int i = 0; i < nb; ++i)
                    for (int j = 0; j < nb; ++j)
                        Pspin(i, j) = 0.5 * P(i, j);
                e_dftu = 2.0
                         * dft_plus_u_energy(Pspin, Smat, tags, z_of_atom,
                                             slots, z_keys);
            }
        }
    }

    // −Tr(W S): Pulay contribution to the Lagrangian (constant −Tr(W S0) drops
    // out of the strain FD; only ∂S/∂ε matters).
    return e_kin + e_pp + e_locgrid + e_es + e_xc + e_disp + e_dftu - e_pulay;
}

/// Energy-weighted density W = Σ_i n_i ε_i c_i c_iᵀ (one spin channel).
RealMatrix energy_weighted_gamma(const RealMatrix& C,
                                 const std::vector<double>& eps, int n_occ,
                                 double occ) {
    const int nb = static_cast<int>(C.shape(0));
    RealMatrix W = xt::zeros<double>({nb, nb});
    for (int i = 0; i < n_occ; ++i) {
        const double w = occ * eps[static_cast<std::size_t>(i)];
        for (int mu = 0; mu < nb; ++mu) {
            const double cmu = w * C(mu, i);
            for (int nu = 0; nu < nb; ++nu)
                W(mu, nu) += cmu * C(nu, i);
        }
    }
    return W;
}

void validate_stress_args(const std::vector<GpwAtom>& atoms, double h,
                          const std::array<bool, 3>& axes, const char* tag) {
    if (atoms.empty())
        throw std::invalid_argument(std::string(tag) + ": empty atom list");
    if (!(h > 0.0) || !(h < 0.1))
        throw std::invalid_argument(std::string(tag)
                                    + ": strain step h out of range");
    if (!axes[0] && !axes[1] && !axes[2])
        throw std::invalid_argument(std::string(tag) + ": all axes masked");
}

} // namespace

std::array<double, 6> gpw_stress_analytic_impl(
    const std::vector<GpwAtom>& atoms, const GpwDft& method, double h,
    const std::array<bool, 3>& axes, const linalg::RealMatrix* guess_C) {
    validate_stress_args(atoms, h, axes, "gpw_stress");

    const double vol = method.cell[0] * method.cell[1] * method.cell[2];
    if (!(vol > 0.0))
        throw std::invalid_argument("gpw_stress: non-positive cell volume");

    // Reference SCF → fixed density (Gamma: real P/W; multi-k: P_k/W_k).
    GpwDft solver = copy_gpw(method);
    solver.build(atoms);
    const GpwResult scf = solver.run(atoms, guess_C);
    if (!scf.converged)
        throw std::runtime_error("gpw_stress: reference SCF did not converge");
    return gpw_stress_at_state_impl(atoms, method, scf, h, axes);
}

std::array<double, 6> gpw_stress_at_state_impl(const std::vector<GpwAtom>& atoms,
                                               const GpwDft& method,
                                               const GpwResult& scf, double h,
                                               const std::array<bool, 3>& axes) {
    validate_stress_args(atoms, h, axes, "gpw_stress");

    const double vol = method.cell[0] * method.cell[1] * method.cell[2];
    if (!(vol > 0.0))
        throw std::invalid_argument("gpw_stress: non-positive cell volume");
    if (!scf.converged)
        throw std::invalid_argument(
            "gpw_stress: supplied state is not a stationary payload "
            "(converged=false) — hand a converged SCF result or the "
            "linearized-XL shadow payload");

    const bool kpath = uses_kpath(method);
    const bool uks = method.is_uks();
    const int n_occ = method.n_occ;
    if (!uks && n_occ <= 0)
        throw std::invalid_argument(
            "gpw_stress analytic: n_occ must be positive for RKS");
    if (uks && method.n_alpha + method.n_beta < 1)
        throw std::invalid_argument(
            "gpw_stress analytic: UKS requires n_alpha+n_beta >= 1");

    RealMatrix P_gamma, W_gamma;
    RealMatrix P_beta, W_beta;
    const RealMatrix* P_beta_ptr = nullptr;
    const RealMatrix* W_beta_ptr = nullptr;
    std::vector<CplxMatrix> P_k, W_k;
    std::vector<CplxMatrix> Pa_k, Pb_k;
    const std::vector<CplxMatrix>* Pa_k_ptr = nullptr;
    const std::vector<CplxMatrix>* Pb_k_ptr = nullptr;
    std::vector<KPoint> kpts;
    if (kpath) {
        if (scf.C_occ_k.empty() || scf.eps_k.empty()
            || scf.C_occ_k.size() != scf.eps_k.size())
            throw std::runtime_error(
                "gpw_stress analytic k: missing C_occ_k / eps_k payload");
        kpts = resolve_kpts(method);
        if (static_cast<int>(kpts.size()) != static_cast<int>(scf.C_occ_k.size()))
            throw std::runtime_error(
                "gpw_stress analytic k: k-point count mismatch vs C_occ_k");
        const int nk = static_cast<int>(kpts.size());
        const int n_occ_a = uks ? method.n_alpha : n_occ;
        const int n_occ_b = uks ? method.n_beta : 0;
        // Metallic free-energy: full C_k + fractional f_k (same as forces).
        const bool frac =
            !scf.f_k.empty() && static_cast<int>(scf.f_k.size()) == nk;
        if (frac) {
            if (static_cast<int>(scf.C_k.size()) != nk)
                throw std::runtime_error(
                    "gpw_stress analytic k: metallic free-energy needs C_k + "
                    "f_k from store_k_payload");
            if (uks
                && (static_cast<int>(scf.C_k_beta.size()) != nk
                    || static_cast<int>(scf.f_k_beta.size()) != nk
                    || static_cast<int>(scf.eps_k_beta.size()) != nk))
                throw std::runtime_error(
                    "gpw_stress analytic k: UKS metallic needs C_k_beta / "
                    "f_k_beta / eps_k_beta");
        }
        // Rebuild nb from occupied payload (row-major nb * n_occ) or C_k.
        // When n_alpha==0 (edge), fall back to β payload or total basis.
        int nb = 0;
        if (frac) {
            nb = static_cast<int>(scf.C_k[0].shape(0));
            if (nb < 1)
                throw std::runtime_error(
                    "gpw_stress analytic k: empty C_k under smear");
        } else if (n_occ_a > 0) {
            nb = static_cast<int>(scf.C_occ_k[0].size()) / n_occ_a;
            if (nb * n_occ_a != static_cast<int>(scf.C_occ_k[0].size()) || nb < 1)
                throw std::runtime_error(
                    "gpw_stress analytic k: C_occ_k shape is not nb * n_occ");
        } else if (uks && n_occ_b > 0 && !scf.C_occ_k_beta.empty()) {
            nb = static_cast<int>(scf.C_occ_k_beta[0].size()) / n_occ_b;
            if (nb * n_occ_b
                    != static_cast<int>(scf.C_occ_k_beta[0].size())
                || nb < 1)
                throw std::runtime_error(
                    "gpw_stress analytic k: C_occ_k_beta shape is not nb * n_beta");
        } else {
            throw std::runtime_error(
                "gpw_stress analytic k: cannot recover AO dimension from C_occ");
        }
        if (uks
            && (static_cast<int>(scf.C_occ_k_beta.size()) != nk
                || static_cast<int>(scf.eps_k_beta.size()) != nk))
            throw std::runtime_error(
                "gpw_stress analytic k: missing C_occ_k_beta / eps_k_beta for UKS");
        P_k.resize(static_cast<std::size_t>(nk));
        W_k.resize(static_cast<std::size_t>(nk));
        if (uks) {
            Pa_k.resize(static_cast<std::size_t>(nk));
            Pb_k.resize(static_cast<std::size_t>(nk));
        }
        for (int ik = 0; ik < nk; ++ik) {
            if (uks) {
                if (frac) {
                    const auto& fa = scf.f_k[static_cast<std::size_t>(ik)];
                    const auto& fb = scf.f_k_beta[static_cast<std::size_t>(ik)];
                    Pa_k[static_cast<std::size_t>(ik)] = density_from_frac(
                        scf.C_k[static_cast<std::size_t>(ik)], fa);
                    CplxMatrix Wa = energy_weighted_from_frac(
                        scf.C_k[static_cast<std::size_t>(ik)],
                        scf.eps_k[static_cast<std::size_t>(ik)], fa);
                    Pb_k[static_cast<std::size_t>(ik)] =
                        fb.empty()
                            ? CplxMatrix(xt::zeros<Cplx>({nb, nb}))
                            : density_from_frac(
                                  scf.C_k_beta[static_cast<std::size_t>(ik)],
                                  fb);
                    CplxMatrix Wb =
                        fb.empty()
                            ? CplxMatrix(xt::zeros<Cplx>({nb, nb}))
                            : energy_weighted_from_frac(
                                  scf.C_k_beta[static_cast<std::size_t>(ik)],
                                  scf.eps_k_beta[static_cast<std::size_t>(ik)],
                                  fb);
                    P_k[static_cast<std::size_t>(ik)] =
                        Pa_k[static_cast<std::size_t>(ik)]
                        + Pb_k[static_cast<std::size_t>(ik)];
                    W_k[static_cast<std::size_t>(ik)] = Wa + Wb;
                } else {
                    if (static_cast<int>(
                            scf.eps_k[static_cast<std::size_t>(ik)].size())
                        < n_occ_a)
                        throw std::runtime_error(
                            "gpw_stress analytic k: eps_k shorter than n_alpha");
                    if (n_occ_b > 0
                        && static_cast<int>(
                               scf.eps_k_beta[static_cast<std::size_t>(ik)]
                                   .size())
                               < n_occ_b)
                        throw std::runtime_error(
                            "gpw_stress analytic k: eps_k_beta shorter than "
                            "n_beta");
                    // Factor-1 per spin (matches gpw_forces multi-k UKS).
                    Pa_k[static_cast<std::size_t>(ik)] = density_from_occ(
                        scf.C_occ_k[static_cast<std::size_t>(ik)], nb, n_occ_a,
                        /*occ=*/1.0);
                    CplxMatrix Wa = energy_weighted_from_occ(
                        scf.C_occ_k[static_cast<std::size_t>(ik)],
                        scf.eps_k[static_cast<std::size_t>(ik)], nb, n_occ_a,
                        /*occ=*/1.0);
                    Pb_k[static_cast<std::size_t>(ik)] =
                        n_occ_b > 0
                            ? density_from_occ(
                                  scf.C_occ_k_beta[static_cast<std::size_t>(ik)],
                                  nb, n_occ_b, /*occ=*/1.0)
                            : CplxMatrix(xt::zeros<Cplx>({nb, nb}));
                    CplxMatrix Wb =
                        n_occ_b > 0
                            ? energy_weighted_from_occ(
                                  scf.C_occ_k_beta[static_cast<std::size_t>(ik)],
                                  scf.eps_k_beta[static_cast<std::size_t>(ik)],
                                  nb, n_occ_b, /*occ=*/1.0)
                            : CplxMatrix(xt::zeros<Cplx>({nb, nb}));
                    P_k[static_cast<std::size_t>(ik)] =
                        Pa_k[static_cast<std::size_t>(ik)]
                        + Pb_k[static_cast<std::size_t>(ik)];
                    W_k[static_cast<std::size_t>(ik)] = Wa + Wb;
                }
            } else if (frac) {
                const auto& f = scf.f_k[static_cast<std::size_t>(ik)];
                P_k[static_cast<std::size_t>(ik)] = density_from_frac(
                    scf.C_k[static_cast<std::size_t>(ik)], f);
                W_k[static_cast<std::size_t>(ik)] = energy_weighted_from_frac(
                    scf.C_k[static_cast<std::size_t>(ik)],
                    scf.eps_k[static_cast<std::size_t>(ik)], f);
            } else {
                if (static_cast<int>(
                        scf.eps_k[static_cast<std::size_t>(ik)].size())
                    < n_occ)
                    throw std::runtime_error(
                        "gpw_stress analytic k: eps_k shorter than n_occ");
                P_k[static_cast<std::size_t>(ik)] = density_from_occ(
                    scf.C_occ_k[static_cast<std::size_t>(ik)], nb, n_occ);
                W_k[static_cast<std::size_t>(ik)] = energy_weighted_from_occ(
                    scf.C_occ_k[static_cast<std::size_t>(ik)],
                    scf.eps_k[static_cast<std::size_t>(ik)], nb, n_occ);
            }
        }
        if (uks) {
            Pa_k_ptr = &Pa_k;
            Pb_k_ptr = &Pb_k;
        }
    } else if (uks) {
        if (scf.P.size() == 0 || scf.P_beta.size() == 0)
            throw std::runtime_error(
                "gpw_stress analytic UKS: missing P_alpha / P_beta");
        P_gamma = scf.P;
        P_beta = scf.P_beta;
        const int nb = static_cast<int>(P_gamma.shape(0));
        if (static_cast<int>(scf.C.shape(0)) != nb
            || static_cast<int>(scf.C.shape(1)) < method.n_alpha
            || static_cast<int>(scf.eps.size()) < method.n_alpha)
            throw std::runtime_error(
                "gpw_stress analytic UKS: converged C/eps (α) missing for W");
        if (method.n_beta > 0
            && (static_cast<int>(scf.C_beta.shape(0)) != nb
                || static_cast<int>(scf.C_beta.shape(1)) < method.n_beta
                || static_cast<int>(scf.eps_beta.size()) < method.n_beta))
            throw std::runtime_error(
                "gpw_stress analytic UKS: converged C/eps (β) missing for W");
        // Factor-1 energy-weighted densities (matches gpw_forces UKS).
        W_gamma =
            energy_weighted_gamma(scf.C, scf.eps, method.n_alpha, /*occ=*/1.0);
        if (method.n_beta > 0)
            W_beta = energy_weighted_gamma(scf.C_beta, scf.eps_beta,
                                           method.n_beta, /*occ=*/1.0);
        else
            W_beta = xt::zeros<double>({nb, nb});
        P_beta_ptr = &P_beta;
        W_beta_ptr = &W_beta;
    } else {
        if (scf.P.size() == 0)
            throw std::runtime_error(
                "gpw_stress: reference SCF produced empty density matrix");
        P_gamma = scf.P;
        const int nb = static_cast<int>(P_gamma.shape(0));
        if (static_cast<int>(scf.eps.size()) < n_occ
            || static_cast<int>(scf.C.shape(0)) != nb
            || static_cast<int>(scf.C.shape(1)) < n_occ)
            throw std::runtime_error(
                "gpw_stress analytic: converged C/eps missing for W build");
        W_gamma =
            energy_weighted_gamma(scf.C, scf.eps, n_occ, /*occ=*/2.0);
    }

    std::array<double, 6> stress{};
    for (int a = 0; a < 3; ++a) {
        if (!axes[static_cast<std::size_t>(a)])
            continue;
        const double sp = 1.0 + h;
        const double sm = 1.0 - h;

        std::vector<GpwAtom> plus = scale_axis(atoms, a, sp);
        std::vector<GpwAtom> minus = scale_axis(atoms, a, sm);

        GpwDft mp = copy_gpw(method);
        GpwDft mm = copy_gpw(method);
        mp.cell[a] = method.cell[a] * sp;
        mm.cell[a] = method.cell[a] * sm;

        // Fixed-v Hirshfeld ratios from the reference SCF (empty ⇒ free-atom).
        const std::vector<double>* vr_ptr =
            (method.hirshfeld && (method.ts || method.mbd)
             && !scf.volume_ratios.empty())
                ? &scf.volume_ratios
                : nullptr;
        if (method.hirshfeld && (method.ts || method.mbd) && vr_ptr == nullptr)
            throw std::runtime_error(
                "gpw_stress: hirshfeld set but SCF volume_ratios empty");

        double ep = 0.0, em = 0.0;
        if (kpath) {
            ep = energy_fixed_P_k(plus, mp, P_k, W_k, kpts, Pa_k_ptr, Pb_k_ptr,
                                  vr_ptr);
            em = energy_fixed_P_k(minus, mm, P_k, W_k, kpts, Pa_k_ptr, Pb_k_ptr,
                                  vr_ptr);
        } else {
            ep = energy_fixed_P(plus, mp, P_gamma, W_gamma, P_beta_ptr,
                                W_beta_ptr, vr_ptr);
            em = energy_fixed_P(minus, mm, P_gamma, W_gamma, P_beta_ptr,
                                W_beta_ptr, vr_ptr);
        }
        // σ_aa = −(1/Ω) ∂E/∂ε_aa  (CP2K STRESS_TENSOR convention).
        stress[static_cast<std::size_t>(a)] = -(ep - em) / (2.0 * h) / vol;
    }
    stress[3] = stress[4] = stress[5] = 0.0;
    return stress;
}

std::array<double, 6> gpw_stress_fd_impl(const std::vector<GpwAtom>& atoms,
                                         const GpwDft& method, double h,
                                         const std::array<bool, 3>& axes,
                                         const linalg::RealMatrix* guess_C) {
    validate_stress_args(atoms, h, axes, "gpw_stress_fd");

    const double vol = method.cell[0] * method.cell[1] * method.cell[2];
    if (!(vol > 0.0))
        throw std::invalid_argument("gpw_stress_fd: non-positive cell volume");

    std::array<double, 6> stress{};
    for (int a = 0; a < 3; ++a) {
        if (!axes[static_cast<std::size_t>(a)])
            continue;
        const double sp = 1.0 + h;
        const double sm = 1.0 - h;

        std::vector<GpwAtom> plus = scale_axis(atoms, a, sp);
        std::vector<GpwAtom> minus = scale_axis(atoms, a, sm);

        GpwDft mp = copy_gpw(method);
        GpwDft mm = copy_gpw(method);
        mp.cell[a] = method.cell[a] * sp;
        mm.cell[a] = method.cell[a] * sm;

        const double ep = scf_energy_ha(plus, mp, guess_C);
        const double em = scf_energy_ha(minus, mm, guess_C);
        stress[static_cast<std::size_t>(a)] = -(ep - em) / (2.0 * h) / vol;
    }
    stress[3] = stress[4] = stress[5] = 0.0;
    return stress;
}

} // namespace internal

std::array<double, 6> gpw_stress(const std::vector<GpwAtom>& atoms,
                                 const GpwDft& method, double h,
                                 const std::array<bool, 3>& axes,
                                 const linalg::RealMatrix* guess_C) {
    return internal::gpw_stress_analytic_impl(atoms, method, h, axes, guess_C);
}

std::array<double, 6> gpw_stress(const std::vector<GpwAtom>& atoms,
                                 const GpwDft& method, const GpwResult& state,
                                 double h, const std::array<bool, 3>& axes) {
    return internal::gpw_stress_at_state_impl(atoms, method, state, h, axes);
}

std::array<double, 6> gpw_stress_fd(const std::vector<GpwAtom>& atoms,
                                    const GpwDft& method, double h,
                                    const std::array<bool, 3>& axes,
                                    const linalg::RealMatrix* guess_C) {
    return internal::gpw_stress_fd_impl(atoms, method, h, axes, guess_C);
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
