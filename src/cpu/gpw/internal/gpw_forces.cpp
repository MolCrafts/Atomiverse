/// @file src/cpu/gpw/internal/gpw_forces.cpp
/// @brief Analytic GPW forces (see header).
///
/// Periodic-image consistent with the Γ-point lattice-sum energy: the Pulay
/// (overlap/kinetic/Vpp) derivatives use the periodic image sums, and the grid
/// term collocates the basis and its center-derivatives with the same
/// multi-image window as the SCF density. Reduces to the isolated path when the
/// cell is non-periodic (image counts collapse to 0).
///
/// The k-point RKS/UKS path generalises the same assembly with Bloch-phased
/// complex S_k / T_k / V_nl and pre-summed complex χ^k on the grid
/// (see internal/kpoints.h). UKS uses dual V_eff^σ and C_occ_k / C_occ_k_beta.

#include "atomiverse/cpu/gpw/internal/gpw_forces.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/dft_plus_u.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/gth_integrals.h"
#include "atomiverse/cpu/gpw/internal/hirshfeld_volumes.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/cpu/gpw/internal/mt_poisson.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/cpu/gpw/internal/vdw_df.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"
#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"
#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/scf/dftd4.h"

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

namespace ATV {
namespace cpu {
namespace gpw {

namespace {
GpwDft copy_gpw(const GpwDft& m) {
    GpwDft p;
    for (int d = 0; d < 3; ++d)
        p.cell[d] = m.cell[d];
    p.cutoff_ry = m.cutoff_ry;
    p.ngrids = m.ngrids;
    p.rel_cutoff = m.rel_cutoff;
    p.n_occ = m.n_occ;
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
    p.hirshfeld = m.hirshfeld;
    p.use_ot = m.use_ot;
    for (int d = 0; d < 3; ++d)
        p.k_mesh[d] = m.k_mesh[d];
    p.k_time_reversal = m.k_time_reversal;
    p.k_points = m.k_points;
    p.k_point_weights = m.k_point_weights;
    p.plus_u = m.plus_u;
    p.atom_spin_guess = m.atom_spin_guess;
    p.n_alpha = m.n_alpha;
    p.n_beta = m.n_beta;
    p.smearing = m.smearing;
    p.degauss = m.degauss;
    p.mp_order = m.mp_order;
    p.extra_bands = m.extra_bands;
    return p;
}

bool uses_kpath(const GpwDft& cfg) {
    // Match SCF dispatch: metallic smear always rides the Bloch path
    // (including effective Γ via k_mesh = {1,1,1}).
    return !cfg.k_points.empty() || cfg.k_mesh[0] != 1 || cfg.k_mesh[1] != 1
           || cfg.k_mesh[2] != 1 || cfg.smearing != Smearing::none;
}

/// Volume ratios for TS/MBD force fold. Empty ⇒ free-atom (v=1). When
/// `cfg.hirshfeld`, ratios come from the SCF payload if present, else from the
/// density already assembled for Hellmann–Feynman (same ρ as energy fold).
/// Density-chain ∂v/∂R is out of scope — forces hold v fixed at this geometry.
std::vector<double>
ts_mbd_vol_ratio(const GpwDft& cfg, const UniformGrid& grid,
                 const std::vector<double>& rho, const std::vector<int>& Z,
                 const std::vector<std::array<double, 3>>& pos, bool periodic,
                 const std::vector<double>* scf_ratios) {
    if (!cfg.hirshfeld || !(cfg.ts || cfg.mbd))
        return {};
    if (scf_ratios != nullptr && !scf_ratios->empty()) {
        if (scf_ratios->size() != Z.size())
            throw std::runtime_error(
                "GPW forces: volume_ratios size mismatch vs atoms");
        return *scf_ratios;
    }
    if (rho.empty())
        throw std::runtime_error(
            "GPW forces: hirshfeld requires density or SCF volume_ratios");
    return hirshfeld_volume_ratios(grid, rho, Z, pos, cfg.cell, periodic);
}

// Solver-side (Z, ℓ, U_ha) tables + per-AO shell tags for cfg.plus_u.
// Mirrors gpw_solver.cpp / gpw_stress.cpp — a shared extraction is blocked
// by layering (internal/dft_plus_u.h cannot include gpw_solver.h).
// Returns false when no entry is active (U ≤ 0 everywhere).
bool pack_plus_u_tables(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                        int nb, std::vector<internal::AoShellTag>& tags,
                        std::vector<int>& z_of_atom,
                        std::vector<internal::PlusUHa>& slots,
                        std::vector<int>& z_keys) {
    slots.clear();
    z_keys.clear();
    for (const auto& pu : cfg.plus_u) {
        if (!(pu.u_minus_j_ev > 0.0) || pu.l < 0)
            continue;
        z_keys.push_back(static_cast<int>(pu.element));
        internal::PlusUHa s;
        s.l = pu.l;
        s.U_ha = pu.u_minus_j_ev / units::kHartreeInEv;
        slots.push_back(s);
    }
    if (slots.empty())
        return false;
    tags.clear();
    tags.reserve(static_cast<std::size_t>(nb));
    z_of_atom.assign(atoms.size(), 0);
    for (int ia = 0; ia < static_cast<int>(atoms.size()); ++ia) {
        const GpwAtom& a = atoms[static_cast<std::size_t>(ia)];
        z_of_atom[static_cast<std::size_t>(ia)] = a.z;
        const int nbf = static_cast<int>(a.basis.size());
        const int nl = static_cast<int>(a.basis_l.size());
        for (int k = 0; k < nbf; ++k) {
            internal::AoShellTag t;
            t.atom = ia;
            t.l = (k < nl) ? a.basis_l[static_cast<std::size_t>(k)] : -1;
            tags.push_back(t);
        }
    }
    if (static_cast<int>(tags.size()) != nb)
        throw std::runtime_error(
            "gpw_forces plus_u: AO shell tag count does not match basis size");
    return true;
}
} // namespace

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

namespace {

/// Shared force assembly at a fixed electronic state (no SCF).
/// RKS: `P` is the total closed-shell density; `P_beta == nullptr`.
/// UKS: `P` is \(P_\alpha\), `P_beta` is \(P_\beta\); one-electron Pulay uses
/// \(P_\mathrm{tot}=P_\alpha+P_\beta\); grid term uses spin-resolved V_eff.
std::vector<std::array<double, 3>>
assemble_gpw_forces(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                    const RealMatrix& P, const RealMatrix& W,
                    const RealMatrix* P_beta = nullptr,
                    const std::vector<double>* scf_vol_ratios = nullptr,
                    const std::vector<double>* scf_veff = nullptr,
                    const std::vector<double>* scf_veff_a = nullptr,
                    const std::vector<double>* scf_veff_b = nullptr,
                    const std::vector<double>* scf_v_es = nullptr,
                    const GpwResult* scf_grids = nullptr) {
    const std::vector<GridBasisFunction> basis = global_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    const int na = static_cast<int>(atoms.size());
    if (static_cast<int>(P.shape(0)) != nb || static_cast<int>(P.shape(1)) != nb
        || static_cast<int>(W.shape(0)) != nb
        || static_cast<int>(W.shape(1)) != nb)
        throw std::invalid_argument(
            "gpw_forces: P/W shape does not match the AO basis");
    const bool uks = P_beta != nullptr;
    if (uks
        && (static_cast<int>(P_beta->shape(0)) != nb
            || static_cast<int>(P_beta->shape(1)) != nb))
        throw std::invalid_argument(
            "gpw_forces: P_beta shape does not match the AO basis");
    const bool periodic = !cfg.nonperiodic;
    const double eps = cfg.eps_default;

    // Total density for spin-independent one-electron / ionic terms.
    RealMatrix Ptot = P;
    if (uks)
        Ptot = P + *P_beta;

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
    // Center-derivative basis functions d φ_u / dR_u along each axis, plus their
    // radii (the l+1 component reaches slightly farther than φ_u).
    std::vector<std::array<GridBasisFunction, 3>> dphi(nb);
    std::vector<std::array<double, 3>> drad(nb);
    double drmax = 0.0;
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k) {
            dphi[u][k] = deriv_bf(basis[u], k);
            drad[u][k] = basis_function_radius(dphi[u][k], eps);
            drmax = std::max(drmax, drad[u][k]);
        }

    // Image extent must cover the LARGEST pair reach any loop below asks for.
    // The Pulay and GTH operator-center loops ask for `drad[mu][k] + radii[nu]`,
    // and `deriv_bf` raises l by one, so `drmax > rmax`: sizing the images from
    // the undifferentiated `2*rmax` silently dropped images from the gradient
    // that the energy's own lattice sum kept. The truncation only bites when
    // images reach at all, which is why it vanished in a large enough cell.
    int mimg[3];
    pair_image_counts(cfg.cell, periodic, std::max(2.0 * rmax, drmax + rmax),
                      mimg);

    // ── Dudarev DFT+U geometric term ───────────────────────────────────────
    // n = ½(PS+SP)|shell depends on geometry through S, so at fixed P the +U
    // energy adds dE_U/dR = Tr[M dS] with M = ½(V̂P+PV̂) per spin. Folding
    // W ← W − M lets the existing W·dS Pulay term carry it (CP2K
    // dft_plus_u.F matrix_w correction shape). Uses the same image-summed
    // overlaps as the Pulay loop, so the derivative matches the SCF's S.
    RealMatrix W_fold;
    const RealMatrix* W_use = &W;
    {
        std::vector<internal::AoShellTag> tags;
        std::vector<int> z_of_atom, z_keys;
        std::vector<internal::PlusUHa> slots;
        if (pack_plus_u_tables(atoms, cfg, nb, tags, z_of_atom, slots,
                               z_keys)) {
            RealMatrix S = xt::zeros<double>({nb, nb});
            for (int mu = 0; mu < nb; ++mu)
                for (int nu = 0; nu < nb; ++nu) {
                    const double reach = radii[static_cast<std::size_t>(mu)]
                                         + radii[static_cast<std::size_t>(nu)];
                    S(mu, nu) =
                        overlap(basis[mu], basis[nu], cfg.cell, mimg, reach);
                }
            W_fold = W;
            if (uks) {
                internal::subtract_dftu_pulay(W_fold, P, S, /*scale=*/1.0, tags,
                                              z_of_atom, slots, z_keys);
                internal::subtract_dftu_pulay(W_fold, *P_beta, S,
                                              /*scale=*/1.0, tags, z_of_atom,
                                              slots, z_keys);
            } else {
                // Closed-shell: two identical spin channels from P_tot/2.
                const RealMatrix Pspin = 0.5 * Ptot;
                internal::subtract_dftu_pulay(W_fold, Pspin, S, /*scale=*/2.0,
                                              tags, z_of_atom, slots, z_keys);
            }
            W_use = &W_fold;
        }
    }

    // Grid + multi-image collocation (same window as the SCF density).
    UniformGrid grid = make_uniform_grid(cfg.cell, cfg.cutoff_ry);
    const long ng = grid.npts();
    const double wgt = grid.weight();
    // Reuse SCF cbg when single-grid SCF filled it (p2-force-kbg-reuse).
    CompactBasisGrid cbg_local;
    const CompactBasisGrid* cbg_ptr = nullptr;
    if (scf_grids && scf_grids->scf_cbg_valid
        && scf_grids->scf_cbg.npts == ng) {
        cbg_ptr = &scf_grids->scf_cbg;
    } else {
        cbg_local = build_compact_basis_grid(grid, basis, periodic, eps);
        cbg_ptr = &cbg_local;
    }
    const CompactBasisGrid& cbg = *cbg_ptr;

    auto mat_to_vec = [&](const RealMatrix& M) {
        std::vector<double> v(static_cast<std::size_t>(nb) * nb);
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < nb; ++j)
                v[static_cast<std::size_t>(i) * nb + j] = M(i, j);
        return v;
    };
    const std::vector<double> Ptot_vec = mat_to_vec(Ptot);
    std::vector<double> Pa_vec, Pb_vec;
    std::vector<double> rho, rho_a, rho_b;
    if (uks) {
        Pa_vec = mat_to_vec(P);
        Pb_vec = mat_to_vec(*P_beta);
        rho_a = collocate_density(cbg, nb, Pa_vec);
        rho_b = collocate_density(cbg, nb, Pb_vec);
        rho.resize(rho_a.size());
        for (std::size_t g = 0; g < rho.size(); ++g)
            rho[g] = rho_a[g] + rho_b[g];
    } else {
        rho = collocate_density(cbg, nb, Ptot_vec);
    }

    // V_eff must be the SAME one the SCF converged against. Prefer the
    // converged SCF payload (p2-force-grid-reuse) so Poisson+XC is not rebuilt
    // at fixed density; fall back to a full rebuild when missing.
    // Differentiating a different functional than the SCF converged was a
    // real bug: ZnO analytic-vs-FD 0.53 Ha/Bohr.
    const bool analytic_loc = !cfg.local_pp_on_grid;
    const bool reuse_uks = uks && scf_veff_a && scf_veff_b && scf_v_es
                           && scf_veff_a->size() == static_cast<std::size_t>(ng)
                           && scf_veff_b->size() == static_cast<std::size_t>(ng)
                           && scf_v_es->size() == static_cast<std::size_t>(ng);
    const bool reuse_rks = !uks && scf_veff && scf_v_es
                           && scf_veff->size() == static_cast<std::size_t>(ng)
                           && scf_v_es->size() == static_cast<std::size_t>(ng);

    std::vector<double> veff, veff_a, veff_b, v_es_store;
    const std::vector<double>* v_es_ptr = nullptr;
    if (reuse_uks) {
        veff_a = *scf_veff_a;
        veff_b = *scf_veff_b;
        v_es_ptr = scf_v_es;
    } else if (reuse_rks) {
        veff = *scf_veff;
        v_es_ptr = scf_v_es;
    } else {
        // Ionic core density n_c(r) (min-image; narrow cores, matches SCF).
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
                s += norm
                     * std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * s2));
            }
            n_ion[static_cast<std::size_t>(g)] = s;
        }
        std::vector<double> n_tot(rho.size());
        for (std::size_t g = 0; g < rho.size(); ++g)
            n_tot[g] = rho[g] - n_ion[g];

        const PoissonResult es = cfg.nonperiodic ? solve_poisson_mt(grid, n_tot)
                                                 : solve_poisson(grid, n_tot);
        const std::vector<double> v_locsr =
            analytic_loc ? std::vector<double>()
                         : local_pp_grid(grid, atoms, periodic);

        if (uks) {
            if (cfg.xc != XcKind::PADE && cfg.xc != XcKind::PBE
                && cfg.xc != XcKind::PBEsol && cfg.xc != XcKind::RevPBE
                && cfg.xc != XcKind::RPBE && cfg.xc != XcKind::BLYP
                && cfg.xc != XcKind::OptB88VdW)
                throw std::runtime_error(
                    "GPW UKS forces: XcKind must be PADE, PBE, PBEsol, RevPBE, "
                    "RPBE, BLYP, or OptB88VdW");
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
            veff_a.resize(rho.size());
            veff_b.resize(rho.size());
            for (std::size_t g = 0; g < rho.size(); ++g) {
                double vloc = v_locsr.empty() ? 0.0 : v_locsr[g];
                veff_a[g] = es.v_r[g] + xc.vxc_a[g] + vloc;
                veff_b[g] = es.v_r[g] + xc.vxc_b[g] + vloc;
            }
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
            veff.resize(rho.size());
            for (std::size_t g = 0; g < rho.size(); ++g) {
                veff[g] = es.v_r[g] + xc.vxc_r[g];
                if (!v_locsr.empty())
                    veff[g] += v_locsr[g];
            }
        }
        // SC nonlocal vdW potential on total density (both spins share V_nl).
        if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
            NonlocalVdwOnGrid nl =
                nonlocal_vdw_on_grid(grid, rho, cfg.vdw_df, cfg.vdw_df2,
                                     cfg.rvv10, /*potential=*/true);
            if (uks) {
                for (std::size_t g = 0; g < rho.size(); ++g) {
                    veff_a[g] += nl.v_r[g];
                    veff_b[g] += nl.v_r[g];
                }
            } else {
                for (std::size_t g = 0; g < rho.size(); ++g)
                    veff[g] += nl.v_r[g];
            }
        }
        v_es_store = es.v_r;
        v_es_ptr = &v_es_store;
    }
    const std::vector<double>& v_es = *v_es_ptr;

    // g -> active index into cbg (for the grid term's Pχ lookup).
    std::vector<long> g2a(static_cast<std::size_t>(ng), -1);
    for (std::size_t a = 0; a < cbg.gidx.size(); ++a)
        g2a[static_cast<std::size_t>(cbg.gidx[a])] = static_cast<long>(a);

    // Multi-image collocation of every center-derivative function dφ_u/dR_u,k
    // (flat index f = u*3 + k), so the grid term's ∂χ_u is image-consistent.
    // Lazy-cache on GpwResult when available (first force after SCF).
    std::vector<GridBasisFunction> dflat(static_cast<std::size_t>(nb) * 3);
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k)
            dflat[static_cast<std::size_t>(u) * 3 + k] = dphi[u][k];
    CompactBasisGrid cbgd_local;
    const CompactBasisGrid* cbgd_ptr = nullptr;
    if (scf_grids && scf_grids->force_cbgd_valid
        && scf_grids->force_cbgd.npts == ng) {
        cbgd_ptr = &scf_grids->force_cbgd;
    } else if (scf_grids) {
        scf_grids->force_cbgd =
            build_compact_basis_grid(grid, dflat, periodic, eps);
        scf_grids->force_cbgd_valid = true;
        cbgd_ptr = &scf_grids->force_cbgd;
    } else {
        cbgd_local = build_compact_basis_grid(grid, dflat, periodic, eps);
        cbgd_ptr = &cbgd_local;
    }
    const CompactBasisGrid& cbgd = *cbgd_ptr;

    std::vector<std::array<double, 3>> forces(na, {0.0, 0.0, 0.0});

    // ── AO-basis Pulay terms (periodic image sums) ──────────────────────────
    // Use P_tot for kinetic / V_pp (spin-independent operators).
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
                    double s = 0.0, t = 0.0;
                    contracted_ST_periodic(dmu, basis[nu], cfg.cell, mimg,
                                           reach, s, t);
                    f += 2.0 * (*W_use)(mu, nu) * s; // + W dS (+U fold inside)
                    f -= 2.0 * Ptot(mu, nu) * t;     // - P dT
                    f -=
                        2.0 * Ptot(mu, nu)
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
                    f += 2.0 * Ptot(mu, nu)
                         * vpp_images(dmu, basis[nu], onlyA, cfg.cell, mimg,
                                      reach, analytic_loc);
                }
            }
            forces[A][k] = f;
        }
    }

    // ── Grid term ──────────────────────────────────────────────────────────
    // RKS: -2 Σ_μ∈A Σ_g w χ_dμ V_eff (Pχ)_μ
    // UKS: -2 Σ_μ∈A Σ_g w χ_dμ [V_eff^α (Pα χ)_μ + V_eff^β (Pβ χ)_μ]
    for (std::size_t ad = 0; ad < cbgd.gidx.size(); ++ad) {
        const long g = cbgd.gidx[ad];
        const long a = g2a[static_cast<std::size_t>(g)];
        if (a < 0)
            continue; // χ_ν all zero here ⇒ Pχ = 0
        const std::size_t cs = cbg.off[static_cast<std::size_t>(a)];
        const std::size_t ce = cbg.off[static_cast<std::size_t>(a) + 1];
        const std::size_t ds = cbgd.off[ad];
        const std::size_t de = cbgd.off[ad + 1];
        const double w = wgt;
        for (std::size_t id = ds; id < de; ++id) {
            const int fdu = cbgd.fu[id]; // flat index u*3+k
            const int u = fdu / 3, k = fdu % 3;
            const double chi_dmu = cbgd.fv[id];
            if (uks) {
                const double* Pau = &Pa_vec[static_cast<std::size_t>(u) * nb];
                const double* Pbu = &Pb_vec[static_cast<std::size_t>(u) * nb];
                double pchi_a = 0.0, pchi_b = 0.0;
                for (std::size_t ib = cs; ib < ce; ++ib) {
                    pchi_a += Pau[cbg.fu[ib]] * cbg.fv[ib];
                    pchi_b += Pbu[cbg.fu[ib]] * cbg.fv[ib];
                }
                forces[owner[u]][k] -=
                    2.0 * w * chi_dmu
                    * (veff_a[static_cast<std::size_t>(g)] * pchi_a
                       + veff_b[static_cast<std::size_t>(g)] * pchi_b);
            } else {
                const double* Pu = &Ptot_vec[static_cast<std::size_t>(u) * nb];
                double pchi = 0.0;
                for (std::size_t ib = cs; ib < ce; ++ib)
                    pchi += Pu[cbg.fu[ib]] * cbg.fv[ib];
                forces[owner[u]][k] -= 2.0 * veff[static_cast<std::size_t>(g)]
                                       * w * chi_dmu * pchi;
            }
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

    // Dispersion forces (additive, geometry-only). D3/BJ, D4, TS, XDM, and
    // MBD@rsSCS are product paths (mutually exclusive).
    const int n_disp = (cfg.d3 ? 1 : 0) + (cfg.d3bj ? 1 : 0) + (cfg.d4 ? 1 : 0)
                       + (cfg.ts ? 1 : 0) + (cfg.xdm ? 1 : 0)
                       + (cfg.mbd ? 1 : 0);
    if (n_disp > 1)
        throw std::runtime_error("GPW forces: d3, d3bj, d4, ts, xdm, and mbd "
                                 "are mutually exclusive");
    if (cfg.d3 || cfg.d3bj || cfg.d4 || cfg.ts || cfg.xdm || cfg.mbd) {
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
        // Hirshfeld-scaled TS/MBD: fixed volume ratios from SCF density (or
        // recompute from rho if scf_vol_ratio_ not threaded — assemble uses rho).
        const std::vector<double> vol_ratio =
            ts_mbd_vol_ratio(cfg, grid, rho, Z, pos, periodic, scf_vol_ratios);
        if (cfg.mbd) {
            // Same isolated / periodic switch as fold_dispersion (mbd_k_mesh).
            const bool mbd_per = cfg.mbd_k_mesh[0] > 0 || cfg.mbd_k_mesh[1] > 0
                                 || cfg.mbd_k_mesh[2] > 0;
            if (mbd_per && cfg.nonperiodic)
                throw std::runtime_error(
                    "GPW forces MBD: mbd_k_mesh requires a periodic cell");
            const MbdResult mbd =
                mbd_rsscs_pbe(Z, pos, cfg.cell, mbd_per, 0.83, 6.0, vol_ratio,
                              mbd_per ? cfg.mbd_k_mesh : nullptr);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += mbd.forces[static_cast<std::size_t>(i)][k];
        } else if (cfg.xdm) {
            constexpr double kRcutXdm = 50.0 * kAng2Bohr;
            const XdmResult xdm = xdm_pbe(Z, pos, cfg.cell, periodic, kRcutXdm);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += xdm.forces[static_cast<std::size_t>(i)][k];
        } else if (cfg.ts) {
            constexpr double kRcutTs = 50.0 * kAng2Bohr;
            const TsResult ts = ts_pbe(Z, pos, cfg.cell, periodic, kRcutTs,
                                       0.94, 20.0, vol_ratio);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += ts.forces[static_cast<std::size_t>(i)][k];
        } else if (cfg.d4) {
            const double s9 = cfg.d4_atm ? 1.0 : 0.0;
            const cpu::scf::D4Result d4 =
                cpu::scf::dftd4(Z, pos, cfg.cell, periodic, /*s6=*/1.0,
                                /*s8=*/0.95948085, /*a1=*/0.38574991,
                                /*a2=*/4.80688534, /*ga=*/3.0, /*gc=*/2.0,
                                /*wf=*/6.0, /*charge=*/0.0, /*cn_cut=*/30.0,
                                /*disp2_cut=*/60.0, s9);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += d4.forces[static_cast<std::size_t>(i)][k];
        } else {
            const D3Result d3 =
                cfg.d3 ? dftd3_pbe(Z, pos, cfg.cell, periodic, D3Params{},
                                   cfg.d3_rcut_bohr, cfg.d3_atm)
                       : dftd3_pbe_bj(Z, pos, cfg.cell, periodic, D3Params{},
                                      cfg.d3_rcut_bohr, 1.0, 0.7875, 0.4289,
                                      4.4407,
                                      /*cp2k_cn=*/true, cfg.d3_atm);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += d3.forces[static_cast<std::size_t>(i)][k];
        }
    }
    return forces;
}

/// Energy-weighted density W = Σ_i n_i ε_i c_i c_iᵀ for one spin channel.
RealMatrix energy_weighted(const RealMatrix& C, const std::vector<double>& eps,
                           int n_occ, double occ) {
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

// ── k-point RKS / UKS force assembly ───────────────────────────────────────

using Cplx = std::complex<double>;
using internal::CplxMatrix;
using internal::KPoint;

constexpr double kTwoPiF = 6.283185307179586476925286766559;

// Phased ⟨a|b⟩_k and ⟨a|T|b⟩_k for one pair (ket shifted by lattice images).
void pair_ST_k(const GridBasisFunction& a, const GridBasisFunction& b,
               const double L[3], const int m[3], double reach,
               const std::vector<KPoint>& kpts, std::vector<Cplx>& s_out,
               std::vector<Cplx>& t_out) {
    const int nk = static_cast<int>(kpts.size());
    s_out.assign(static_cast<std::size_t>(nk), Cplx(0, 0));
    t_out.assign(static_cast<std::size_t>(nk), Cplx(0, 0));
    if (a.empty() || b.empty())
        return;
    const double reach2 = reach * reach;
    const double* ac = a[0].center;
    const double* bc = b[0].center;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                const double T[3] = {tx * L[0], ty * L[1], tz * L[2]};
                const double dx = ac[0] - bc[0] - T[0];
                const double dy = ac[1] - bc[1] - T[1];
                const double dz = ac[2] - bc[2] - T[2];
                if (dx * dx + dy * dy + dz * dz > reach2)
                    continue;
                double s_im = 0.0, t_im = 0.0;
                for (const auto& pa : a)
                    for (const auto& pb : b) {
                        GridPrimitive q = pb;
                        q.center[0] += T[0];
                        q.center[1] += T[1];
                        q.center[2] += T[2];
                        double s_p = 0.0, t_p = 0.0;
                        prim_ST(pa, q, s_p, t_p);
                        s_im += s_p;
                        t_im += t_p;
                    }
                for (int ik = 0; ik < nk; ++ik) {
                    const auto& kp = kpts[static_cast<std::size_t>(ik)];
                    const double arg =
                        kTwoPiF
                        * (kp.frac[0] * tx + kp.frac[1] * ty + kp.frac[2] * tz);
                    const Cplx phase(std::cos(arg), std::sin(arg));
                    s_out[static_cast<std::size_t>(ik)] += phase * s_im;
                    t_out[static_cast<std::size_t>(ik)] += phase * t_im;
                }
            }
}

// Phased ⟨a|V_nl|b⟩_k over a subset of GTH centers (no local SR — on grid).
void pair_Vnl_k(const GridBasisFunction& a, const GridBasisFunction& b,
                const std::vector<const GpwAtom*>& centers, const double L[3],
                const int m[3], double reach, const std::vector<KPoint>& kpts,
                std::vector<Cplx>& v_out) {
    const int nk = static_cast<int>(kpts.size());
    v_out.assign(static_cast<std::size_t>(nk), Cplx(0, 0));
    if (a.empty() || b.empty() || centers.empty())
        return;
    const double reach2 = reach * reach;
    const double* ac = a[0].center;
    const double* bc = b[0].center;
    const bool per = m[0] || m[1] || m[2];
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                const double T[3] = {tx * L[0], ty * L[1], tz * L[2]};
                const double dx = ac[0] - bc[0] - T[0];
                const double dy = ac[1] - bc[1] - T[1];
                const double dz = ac[2] - bc[2] - T[2];
                if (dx * dx + dy * dy + dz * dz > reach2)
                    continue;
                double v_im = 0.0;
                for (const auto& p : a) {
                    const Prim pp = to_prim(p);
                    for (const auto& q0 : b) {
                        const double cc = p.coeff * q0.coeff;
                        if (cc == 0.0)
                            continue;
                        GridPrimitive q = q0;
                        q.center[0] += T[0];
                        q.center[1] += T[1];
                        q.center[2] += T[2];
                        const double aa = p.alpha, bb = q.alpha, g = aa + bb;
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
                            for (const auto& ch : C.nonlocal)
                                v_im += cc * gth_nonlocal(pp, qq, apos, ch);
                        }
                    }
                }
                for (int ik = 0; ik < nk; ++ik) {
                    const auto& kp = kpts[static_cast<std::size_t>(ik)];
                    const double arg =
                        kTwoPiF
                        * (kp.frac[0] * tx + kp.frac[1] * ty + kp.frac[2] * tz);
                    const Cplx phase(std::cos(arg), std::sin(arg));
                    v_out[static_cast<std::size_t>(ik)] += phase * v_im;
                }
            }
}

/// Build spin-channel P/W from occupied Bloch MOs (uniform occupancy `occ`).
void add_occ_PW(const std::vector<Cplx>& Cflat, const std::vector<double>& eps,
                int nb, int n_occ, double occ, CplxMatrix& P, CplxMatrix& W) {
    if (static_cast<int>(Cflat.size()) != nb * n_occ)
        throw std::runtime_error(
            "gpw_forces kpoints: C_occ shape mismatch (nb*n_occ)");
    if (static_cast<int>(eps.size()) < n_occ)
        throw std::runtime_error("gpw_forces kpoints: eps shorter than n_occ");
    for (int o = 0; o < n_occ; ++o) {
        const double weps = occ * eps[static_cast<std::size_t>(o)];
        for (int mu = 0; mu < nb; ++mu) {
            const Cplx cmu = Cflat[static_cast<std::size_t>(mu) * n_occ
                                   + static_cast<std::size_t>(o)];
            for (int nu = 0; nu < nb; ++nu) {
                const Cplx cnu = Cflat[static_cast<std::size_t>(nu) * n_occ
                                       + static_cast<std::size_t>(o)];
                P(mu, nu) += occ * cmu * std::conj(cnu);
                W(mu, nu) += weps * cmu * std::conj(cnu);
            }
        }
    }
}

/// Free-energy P/W from full Bloch MOs + fractional band occupations.
/// \(P=\sum_b f_b\,c_b c_b^\dagger\), \(W=\sum_b f_b\varepsilon_b\,c_b c_b^\dagger\).
/// `f` is already in band form (RKS \([0,2]\)) or spinor form (UKS \([0,1]\)).
void add_frac_PW(const CplxMatrix& C, const std::vector<double>& eps,
                 const std::vector<double>& f, CplxMatrix& P, CplxMatrix& W) {
    const int nb = static_cast<int>(C.shape(0));
    const int m = static_cast<int>(C.shape(1));
    const int nfill = std::min(
        m, std::min(static_cast<int>(eps.size()), static_cast<int>(f.size())));
    for (int o = 0; o < nfill; ++o) {
        const double fo = f[static_cast<std::size_t>(o)];
        if (!(std::abs(fo) > 1e-14))
            continue;
        const double weps = fo * eps[static_cast<std::size_t>(o)];
        for (int mu = 0; mu < nb; ++mu) {
            const Cplx cmu = C(mu, o);
            for (int nu = 0; nu < nb; ++nu) {
                const Cplx cnu = C(nu, o);
                P(mu, nu) += fo * cmu * std::conj(cnu);
                W(mu, nu) += weps * cmu * std::conj(cnu);
            }
        }
    }
}

/// Analytic ionic forces after a converged k-point RKS or collinear UKS SCF.
/// Fixed occupations: `C_occ_k` / `eps_k` with occ=2 (RKS) or 1 (UKS).
/// Metallic smear (free-energy): full `C_k` + fractional `f_k` (and β).
std::vector<std::array<double, 3>>
assemble_gpw_forces_kpoints(const std::vector<GpwAtom>& atoms,
                            const GpwDft& cfg, const GpwResult& scf) {
    const std::vector<GridBasisFunction> basis = global_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    const int na = static_cast<int>(atoms.size());
    const bool uks = cfg.is_uks();
    const bool metallic = (cfg.smearing != Smearing::none);
    const int n_occ_a = uks ? cfg.n_alpha : cfg.n_occ;
    const int n_occ_b = uks ? cfg.n_beta : 0;
    const int nk = scf.n_kpts;
    if (nk < 1 || static_cast<int>(scf.eps_k.size()) != nk
        || static_cast<int>(scf.k_weights.size()) != nk)
        throw std::runtime_error(
            "gpw_forces kpoints: missing eps_k / k_weights payload from SCF");
    if (metallic) {
        if (static_cast<int>(scf.C_k.size()) != nk
            || static_cast<int>(scf.f_k.size()) != nk)
            throw std::runtime_error(
                "gpw_forces kpoints: metallic free-energy forces need C_k and "
                "f_k from store_k_payload (fractional occupations)");
        if (uks
            && (static_cast<int>(scf.C_k_beta.size()) != nk
                || static_cast<int>(scf.f_k_beta.size()) != nk
                || static_cast<int>(scf.eps_k_beta.size()) != nk))
            throw std::runtime_error(
                "gpw_forces kpoints: UKS metallic forces need C_k_beta / "
                "f_k_beta / eps_k_beta");
    } else {
        if (static_cast<int>(scf.C_occ_k.size()) != nk)
            throw std::runtime_error(
                "gpw_forces kpoints: missing C_occ_k payload from SCF");
        if (uks
            && (static_cast<int>(scf.C_occ_k_beta.size()) != nk
                || static_cast<int>(scf.eps_k_beta.size()) != nk))
            throw std::runtime_error("gpw_forces kpoints: missing C_occ_k_beta "
                                     "/ eps_k_beta for UKS");
    }
    if (cfg.nonperiodic)
        throw std::runtime_error(
            "gpw_forces kpoints: nonperiodic not supported on k-path");
    if (!cfg.local_pp_on_grid)
        throw std::runtime_error(
            "gpw_forces kpoints: requires local_pp_on_grid (energy foothold)");
    if (uks && cfg.xc != XcKind::PADE && cfg.xc != XcKind::PBE
        && cfg.xc != XcKind::PBEsol && cfg.xc != XcKind::RevPBE
        && cfg.xc != XcKind::RPBE && cfg.xc != XcKind::BLYP
        && cfg.xc != XcKind::OptB88VdW)
        throw std::runtime_error(
            "gpw_forces kpoints UKS: XcKind must be PADE, PBE, PBEsol, RevPBE, "
            "RPBE, BLYP, or OptB88VdW");

    // Rebuild k list from the SCF payload (frac + weight).
    std::vector<KPoint> kpts(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik) {
        kpts[static_cast<std::size_t>(ik)].frac[0] =
            scf.k_frac[static_cast<std::size_t>(ik)][0];
        kpts[static_cast<std::size_t>(ik)].frac[1] =
            scf.k_frac[static_cast<std::size_t>(ik)][1];
        kpts[static_cast<std::size_t>(ik)].frac[2] =
            scf.k_frac[static_cast<std::size_t>(ik)][2];
        kpts[static_cast<std::size_t>(ik)].weight =
            scf.k_weights[static_cast<std::size_t>(ik)];
    }

    // P_tot^k / W^k (and spin Pa^k / Pb^k for UKS grid).
    // Fixed: RKS occ=2; UKS P_tot=Pa+Pb, W=Wa+Wb with occ=1.
    // Metallic: fractional f from SCF (Mermin free-energy Hellmann–Feynman).
    std::vector<CplxMatrix> P_k(static_cast<std::size_t>(nk));
    std::vector<CplxMatrix> W_k(static_cast<std::size_t>(nk));
    std::vector<CplxMatrix> Pa_k, Pb_k;
    if (uks) {
        Pa_k.resize(static_cast<std::size_t>(nk));
        Pb_k.resize(static_cast<std::size_t>(nk));
    }
    for (int ik = 0; ik < nk; ++ik) {
        CplxMatrix P = xt::zeros<Cplx>({nb, nb});
        CplxMatrix W = xt::zeros<Cplx>({nb, nb});
        if (metallic) {
            if (uks) {
                CplxMatrix Pa = xt::zeros<Cplx>({nb, nb});
                CplxMatrix Wa = xt::zeros<Cplx>({nb, nb});
                CplxMatrix Pb = xt::zeros<Cplx>({nb, nb});
                CplxMatrix Wb = xt::zeros<Cplx>({nb, nb});
                add_frac_PW(scf.C_k[static_cast<std::size_t>(ik)],
                            scf.eps_k[static_cast<std::size_t>(ik)],
                            scf.f_k[static_cast<std::size_t>(ik)], Pa, Wa);
                add_frac_PW(scf.C_k_beta[static_cast<std::size_t>(ik)],
                            scf.eps_k_beta[static_cast<std::size_t>(ik)],
                            scf.f_k_beta[static_cast<std::size_t>(ik)], Pb, Wb);
                P = Pa + Pb;
                W = Wa + Wb;
                Pa_k[static_cast<std::size_t>(ik)] = std::move(Pa);
                Pb_k[static_cast<std::size_t>(ik)] = std::move(Pb);
            } else {
                add_frac_PW(scf.C_k[static_cast<std::size_t>(ik)],
                            scf.eps_k[static_cast<std::size_t>(ik)],
                            scf.f_k[static_cast<std::size_t>(ik)], P, W);
            }
        } else if (uks) {
            CplxMatrix Pa = xt::zeros<Cplx>({nb, nb});
            CplxMatrix Wa = xt::zeros<Cplx>({nb, nb});
            CplxMatrix Pb = xt::zeros<Cplx>({nb, nb});
            CplxMatrix Wb = xt::zeros<Cplx>({nb, nb});
            add_occ_PW(scf.C_occ_k[static_cast<std::size_t>(ik)],
                       scf.eps_k[static_cast<std::size_t>(ik)], nb, n_occ_a,
                       1.0, Pa, Wa);
            if (n_occ_b > 0)
                add_occ_PW(scf.C_occ_k_beta[static_cast<std::size_t>(ik)],
                           scf.eps_k_beta[static_cast<std::size_t>(ik)], nb,
                           n_occ_b, 1.0, Pb, Wb);
            P = Pa + Pb;
            W = Wa + Wb;
            Pa_k[static_cast<std::size_t>(ik)] = std::move(Pa);
            Pb_k[static_cast<std::size_t>(ik)] = std::move(Pb);
        } else {
            add_occ_PW(scf.C_occ_k[static_cast<std::size_t>(ik)],
                       scf.eps_k[static_cast<std::size_t>(ik)], nb, n_occ_a,
                       2.0, P, W);
        }
        P_k[static_cast<std::size_t>(ik)] = std::move(P);
        W_k[static_cast<std::size_t>(ik)] = std::move(W);
    }

    std::vector<int> atom_off(na + 1, 0);
    for (int a = 0; a < na; ++a)
        atom_off[a + 1] = atom_off[a] + static_cast<int>(atoms[a].basis.size());
    std::vector<int> owner(nb, 0);
    for (int a = 0; a < na; ++a)
        for (int u = atom_off[a]; u < atom_off[a + 1]; ++u)
            owner[u] = a;

    const double eps = cfg.eps_default;
    std::vector<double> radii(static_cast<std::size_t>(nb));
    double rmax = 0.0;
    for (int u = 0; u < nb; ++u) {
        radii[static_cast<std::size_t>(u)] =
            basis_function_radius(basis[u], eps);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(u)]);
    }
    // Same image-extent rule as the Γ path: the derivative loops below ask for
    // `drad[mu][k] + radii[nu]`, and `deriv_bf` raises l by one, so sizing the
    // images from the undifferentiated `2*rmax` would drop images the energy's
    // lattice sum keeps.
    double drmax = 0.0;
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k)
            drmax = std::max(drmax,
                             basis_function_radius(deriv_bf(basis[u], k), eps));
    int mimg[3];
    pair_image_counts(cfg.cell, /*periodic=*/true,
                      std::max(2.0 * rmax, drmax + rmax), mimg);

    // ── Dudarev DFT+U geometric term (multi-k) ─────────────────────────────
    // BZ-averaged n = Σ_k w_k ½Herm[(P_k S_k)+(S_k P_k)]|shell depends on
    // geometry through S_k: fold W_k ← W_k − ½(V̂P_k+P_kV̂) per spin so the
    // k-summed W·dS_k Pulay term carries dE_U/dR (same Γ device, Bloch-phased).
    {
        std::vector<internal::AoShellTag> tags;
        std::vector<int> z_of_atom, z_keys;
        std::vector<internal::PlusUHa> slots;
        if (pack_plus_u_tables(atoms, cfg, nb, tags, z_of_atom, slots,
                               z_keys)) {
            constexpr double kBuildScreen = 1e-12; // matches the SCF S_k build
            std::vector<CplxMatrix> S_k, T_k;
            internal::build_overlap_kinetic_k(basis, radii, cfg.cell, mimg,
                                              kBuildScreen, kpts, S_k, T_k);
            std::vector<double> weights(static_cast<std::size_t>(nk));
            for (int ik = 0; ik < nk; ++ik)
                weights[static_cast<std::size_t>(ik)] =
                    kpts[static_cast<std::size_t>(ik)].weight;
            if (uks) {
                internal::subtract_dftu_pulay_k(W_k, Pa_k, S_k, weights,
                                                /*scale=*/1.0, tags, z_of_atom,
                                                slots, z_keys);
                internal::subtract_dftu_pulay_k(W_k, Pb_k, S_k, weights,
                                                /*scale=*/1.0, tags, z_of_atom,
                                                slots, z_keys);
            } else {
                // Closed-shell: two identical spin channels from P_k/2.
                std::vector<CplxMatrix> Pspin_k(static_cast<std::size_t>(nk));
                for (int ik = 0; ik < nk; ++ik)
                    Pspin_k[static_cast<std::size_t>(ik)] =
                        0.5 * P_k[static_cast<std::size_t>(ik)];
                internal::subtract_dftu_pulay_k(W_k, Pspin_k, S_k, weights,
                                                /*scale=*/2.0, tags, z_of_atom,
                                                slots, z_keys);
            }
        }
    }

    std::vector<std::array<GridBasisFunction, 3>> dphi(nb);
    std::vector<std::array<double, 3>> drad(nb);
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k) {
            dphi[u][k] = deriv_bf(basis[u], k);
            drad[u][k] = basis_function_radius(dphi[u][k], eps);
        }

    UniformGrid grid = make_uniform_grid(cfg.cell, cfg.cutoff_ry);
    const long ng = grid.npts();
    const double wgt = grid.weight();
    // Reuse SCF χ^k when single-grid multi-k SCF filled it.
    internal::KpointBasisGrid kbg_local;
    const internal::KpointBasisGrid* kbg_ptr = nullptr;
    if (scf.scf_kbg && scf.scf_kbg->npts == ng && scf.scf_kbg->nb == nb
        && static_cast<int>(scf.scf_kbg->chi.size()) == nk) {
        kbg_ptr = scf.scf_kbg.get();
    } else {
        kbg_local = internal::build_kpoint_basis_grid(grid, basis, kpts, eps);
        kbg_ptr = &kbg_local;
    }
    const internal::KpointBasisGrid& kbg = *kbg_ptr;

    // Density collocate (needed for local-PP HF / Hirshfeld). Prefer SCF V_eff
    // / V_es to skip Poisson+XC rebuild at fixed density.
    std::vector<double> rho(static_cast<std::size_t>(ng), 0.0);
    std::vector<double> rho_a, rho_b;
    if (uks) {
        rho_a.assign(static_cast<std::size_t>(ng), 0.0);
        rho_b.assign(static_cast<std::size_t>(ng), 0.0);
        for (int ik = 0; ik < nk; ++ik) {
            const double wk = kpts[static_cast<std::size_t>(ik)].weight;
            internal::collocate_density_k(
                kbg, ik, Pa_k[static_cast<std::size_t>(ik)], wk, rho_a);
            internal::collocate_density_k(
                kbg, ik, Pb_k[static_cast<std::size_t>(ik)], wk, rho_b);
        }
        for (std::size_t g = 0; g < rho.size(); ++g)
            rho[g] = rho_a[g] + rho_b[g];
    } else {
        for (int ik = 0; ik < nk; ++ik)
            internal::collocate_density_k(
                kbg, ik, P_k[static_cast<std::size_t>(ik)],
                kpts[static_cast<std::size_t>(ik)].weight, rho);
    }

    const bool reuse_uks = uks
                           && scf.veff_a.size() == static_cast<std::size_t>(ng)
                           && scf.veff_b.size() == static_cast<std::size_t>(ng)
                           && scf.v_es.size() == static_cast<std::size_t>(ng);
    const bool reuse_rks = !uks
                           && scf.veff.size() == static_cast<std::size_t>(ng)
                           && scf.v_es.size() == static_cast<std::size_t>(ng);

    std::vector<double> veff, veff_a, veff_b, v_es_store;
    const std::vector<double>* v_es_ptr = nullptr;
    if (reuse_uks) {
        veff_a = scf.veff_a;
        veff_b = scf.veff_b;
        v_es_ptr = &scf.v_es;
    } else if (reuse_rks) {
        veff = scf.veff;
        v_es_ptr = &scf.v_es;
    } else {
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
                dx -= grid.L[0] * std::nearbyint(dx / grid.L[0]);
                dy -= grid.L[1] * std::nearbyint(dy / grid.L[1]);
                dz -= grid.L[2] * std::nearbyint(dz / grid.L[2]);
                s += norm
                     * std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * s2));
            }
            n_ion[static_cast<std::size_t>(g)] = s;
        }
        std::vector<double> n_tot(rho.size());
        for (std::size_t g = 0; g < rho.size(); ++g)
            n_tot[g] = rho[g] - n_ion[g];
        const PoissonResult es = solve_poisson(grid, n_tot);
        const std::vector<double> v_locsr = local_pp_grid(grid, atoms, true);
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
            veff_a.resize(rho.size());
            veff_b.resize(rho.size());
            for (std::size_t g = 0; g < rho.size(); ++g) {
                veff_a[g] = es.v_r[g] + xc.vxc_a[g] + v_locsr[g];
                veff_b[g] = es.v_r[g] + xc.vxc_b[g] + v_locsr[g];
            }
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
            veff.resize(rho.size());
            for (std::size_t g = 0; g < rho.size(); ++g)
                veff[g] = es.v_r[g] + xc.vxc_r[g] + v_locsr[g];
        }
        if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
            NonlocalVdwOnGrid nl =
                nonlocal_vdw_on_grid(grid, rho, cfg.vdw_df, cfg.vdw_df2,
                                     cfg.rvv10, /*potential=*/true);
            if (uks) {
                for (std::size_t g = 0; g < rho.size(); ++g) {
                    veff_a[g] += nl.v_r[g];
                    veff_b[g] += nl.v_r[g];
                }
            } else {
                for (std::size_t g = 0; g < rho.size(); ++g)
                    veff[g] += nl.v_r[g];
            }
        }
        v_es_store = es.v_r;
        v_es_ptr = &v_es_store;
    }
    const std::vector<double>& v_es = *v_es_ptr;

    // Derivative basis on grid (flat u*3+k); lazy-cache on scf.
    std::vector<GridBasisFunction> dflat(static_cast<std::size_t>(nb) * 3);
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k)
            dflat[static_cast<std::size_t>(u) * 3 + k] = dphi[u][k];
    const internal::KpointBasisGrid* kbgd_ptr = nullptr;
    if (scf.force_kbgd && scf.force_kbgd->npts == ng
        && scf.force_kbgd->nb == nb * 3
        && static_cast<int>(scf.force_kbgd->chi.size()) == nk) {
        kbgd_ptr = scf.force_kbgd.get();
    } else {
        scf.force_kbgd = std::make_shared<internal::KpointBasisGrid>(
            internal::build_kpoint_basis_grid(grid, dflat, kpts, eps));
        kbgd_ptr = scf.force_kbgd.get();
    }
    const internal::KpointBasisGrid& kbgd = *kbgd_ptr;

    std::vector<std::array<double, 3>> forces(na, {0.0, 0.0, 0.0});

    // ── AO Pulay: Σ_k w_k 2 Re[ W_μν* ⟨dμφ|ν⟩_k − P_μν* ⟨dμφ|T+Vnl|ν⟩_k ] ──
    // Parallel over basis functions (better load balance than atoms — O has
    // far more AOs than H). Each thread accumulates into a private force
    // scratch and merges once; the operator-center Vnl TI is parallel over
    // (A, μ) with the same scratch.
    std::vector<const GpwAtom*> all(static_cast<std::size_t>(na));
    for (int a = 0; a < na; ++a)
        all[static_cast<std::size_t>(a)] = &atoms[a];
    for (int k = 0; k < 3; ++k) {
#pragma omp parallel
        {
            std::vector<Cplx> s_k, t_k, v_k, vA_k;
            std::vector<double> f_loc(static_cast<std::size_t>(na), 0.0);
            // Bra-center Pulay: ∂/∂R_A on basis functions owned by A.
#pragma omp for schedule(dynamic) nowait
            for (int mu = 0; mu < nb; ++mu) {
                const int A = owner[static_cast<std::size_t>(mu)];
                const GridBasisFunction& dmu = dphi[mu][k];
                const double rdmu = drad[mu][k];
                double f = 0.0;
                for (int nu = 0; nu < nb; ++nu) {
                    const double reach =
                        rdmu + radii[static_cast<std::size_t>(nu)];
                    pair_ST_k(dmu, basis[nu], cfg.cell, mimg, reach, kpts, s_k,
                              t_k);
                    pair_Vnl_k(dmu, basis[nu], all, cfg.cell, mimg, reach, kpts,
                               v_k);
                    for (int ik = 0; ik < nk; ++ik) {
                        const double wk =
                            kpts[static_cast<std::size_t>(ik)].weight;
                        const Cplx Wmn =
                            W_k[static_cast<std::size_t>(ik)](mu, nu);
                        const Cplx Pmn =
                            P_k[static_cast<std::size_t>(ik)](mu, nu);
                        const Cplx s = s_k[static_cast<std::size_t>(ik)];
                        const Cplx t = t_k[static_cast<std::size_t>(ik)];
                        const Cplx v = v_k[static_cast<std::size_t>(ik)];
                        // 2 Re(W† s) with Hermitian W → 2 Re(conj(W)*s)
                        f += 2.0 * wk * std::real(std::conj(Wmn) * s);
                        f -= 2.0 * wk * std::real(std::conj(Pmn) * t);
                        f -= 2.0 * wk * std::real(std::conj(Pmn) * v);
                    }
                }
                f_loc[static_cast<std::size_t>(A)] += f;
            }
            // Operator-center TI for nonlocal on A: +2 Re P† ⟨dμφ|V_A|ν⟩.
            // Flatten over A so all atoms share the μ-loop work pool.
#pragma omp for schedule(dynamic) nowait
            for (int A = 0; A < na; ++A) {
                const std::vector<const GpwAtom*> onlyA = {&atoms[A]};
                double f = 0.0;
                for (int mu = 0; mu < nb; ++mu) {
                    const GridBasisFunction& dmu = dphi[mu][k];
                    const double rdmu = drad[mu][k];
                    for (int nu = 0; nu < nb; ++nu) {
                        const double reach =
                            rdmu + radii[static_cast<std::size_t>(nu)];
                        pair_Vnl_k(dmu, basis[nu], onlyA, cfg.cell, mimg, reach,
                                   kpts, vA_k);
                        for (int ik = 0; ik < nk; ++ik) {
                            const double wk =
                                kpts[static_cast<std::size_t>(ik)].weight;
                            const Cplx Pmn =
                                P_k[static_cast<std::size_t>(ik)](mu, nu);
                            f += 2.0 * wk
                                 * std::real(
                                     std::conj(Pmn)
                                     * vA_k[static_cast<std::size_t>(ik)]);
                        }
                    }
                }
                f_loc[static_cast<std::size_t>(A)] += f;
            }
#pragma omp critical
            {
                for (int a = 0; a < na; ++a)
                    forces[a][k] += f_loc[static_cast<std::size_t>(a)];
            }
        }
    }

    // ── Grid term ──────────────────────────────────────────────────────────
    // RKS: −2 Σ_k w_k Σ_g w Veff Re[ (∂χ_μ) (P χ†)_μ ]
    // UKS: −2 Σ_k w_k Σ_g w [Veff^α Re(∂χ y_α) + Veff^β Re(∂χ y_β)]
    // kbgd.chi[ik][g * (nb*3) + (u*3+k)] = ∂χ_u^k / ∂R_k at g.
    // Parallel over grid points with thread-private force scratch.
    const int nbd = nb * 3;
    for (int ik = 0; ik < nk; ++ik) {
        const double wk = kpts[static_cast<std::size_t>(ik)].weight;
        const auto& chi = kbg.chi[static_cast<std::size_t>(ik)];
        const auto& dchi = kbgd.chi[static_cast<std::size_t>(ik)];
#pragma omp parallel
        {
            std::vector<std::array<double, 3>> f_loc(na, {0.0, 0.0, 0.0});
            std::vector<Cplx> y(static_cast<std::size_t>(nb));
            std::vector<Cplx> ya, yb;
            if (uks) {
                ya.resize(static_cast<std::size_t>(nb));
                yb.resize(static_cast<std::size_t>(nb));
            }
#pragma omp for schedule(static) nowait
            for (long g = 0; g < ng; ++g) {
                const Cplx* v = chi.data() + static_cast<std::size_t>(g) * nb;
                const Cplx* dv =
                    dchi.data() + static_cast<std::size_t>(g) * nbd;
                if (uks) {
                    const CplxMatrix& Pak = Pa_k[static_cast<std::size_t>(ik)];
                    const CplxMatrix& Pbk = Pb_k[static_cast<std::size_t>(ik)];
                    for (int mu = 0; mu < nb; ++mu) {
                        Cplx sa(0, 0), sb(0, 0);
                        for (int nu = 0; nu < nb; ++nu) {
                            sa += Pak(mu, nu) * std::conj(v[nu]);
                            sb += Pbk(mu, nu) * std::conj(v[nu]);
                        }
                        ya[static_cast<std::size_t>(mu)] = sa;
                        yb[static_cast<std::size_t>(mu)] = sb;
                    }
                    const double va = veff_a[static_cast<std::size_t>(g)] * wgt;
                    const double vb = veff_b[static_cast<std::size_t>(g)] * wgt;
                    for (int u = 0; u < nb; ++u)
                        for (int kk = 0; kk < 3; ++kk) {
                            const Cplx dval =
                                dv[static_cast<std::size_t>(u) * 3
                                   + static_cast<std::size_t>(kk)];
                            f_loc[owner[u]][kk] -=
                                2.0 * wk
                                * (va
                                       * std::real(
                                           dval
                                           * ya[static_cast<std::size_t>(u)])
                                   + vb
                                         * std::real(
                                             dval
                                             * yb[static_cast<std::size_t>(
                                                 u)]));
                        }
                } else {
                    const double vw = veff[static_cast<std::size_t>(g)] * wgt;
                    if (vw == 0.0)
                        continue;
                    const CplxMatrix& Pk = P_k[static_cast<std::size_t>(ik)];
                    for (int mu = 0; mu < nb; ++mu) {
                        Cplx s(0, 0);
                        for (int nu = 0; nu < nb; ++nu)
                            s += Pk(mu, nu) * std::conj(v[nu]);
                        y[static_cast<std::size_t>(mu)] = s;
                    }
                    for (int u = 0; u < nb; ++u)
                        for (int kk = 0; kk < 3; ++kk) {
                            const Cplx dval =
                                dv[static_cast<std::size_t>(u) * 3
                                   + static_cast<std::size_t>(kk)];
                            f_loc[owner[u]][kk] -=
                                2.0 * wk * vw
                                * std::real(dval
                                            * y[static_cast<std::size_t>(u)]);
                        }
                }
            }
#pragma omp critical
            {
                for (int a = 0; a < na; ++a)
                    for (int kk = 0; kk < 3; ++kk)
                        forces[a][kk] += f_loc[a][kk];
            }
        }
    }

    // ── Ionic Hellmann–Feynman ────────────────────────────────────────────
    // Parallel over atoms (each A writes only forces[A][*]).
#pragma omp parallel for schedule(static) if (na > 1)
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
            for (int kk = 0; kk < 3; ++kk)
                d[kk] -= grid.L[kk] * std::nearbyint(d[kk] / grid.L[kk]);
            const double gauss =
                norm
                * std::exp(-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                           / (2.0 * s2));
            const double vg = v_es[static_cast<std::size_t>(g)] * gauss / s2;
            for (int kk = 0; kk < 3; ++kk)
                acc[kk] += vg * d[kk];
        }
        for (int kk = 0; kk < 3; ++kk)
            forces[A][kk] += wgt * acc[kk];
    }

    // Local-PP HF (grid route).
    add_local_pp_hf_forces(grid, atoms, /*periodic=*/true, rho, forces);

    // Geometry-only dispersion (same as Γ assemble_gpw_forces).
    // MBD was previously counted for mutual exclusion but not folded in —
    // multi-k product path must match Γ for d3/d3bj/d4/ts/xdm/mbd.
    {
        const bool periodic = true;
        const int na = static_cast<int>(atoms.size());
        const int n_disp = (cfg.d3 ? 1 : 0) + (cfg.d3bj ? 1 : 0)
                           + (cfg.d4 ? 1 : 0) + (cfg.ts ? 1 : 0)
                           + (cfg.xdm ? 1 : 0) + (cfg.mbd ? 1 : 0);
        if (n_disp > 1)
            throw std::runtime_error(
                "GPW k-forces: d3, d3bj, d4, ts, xdm, and mbd are mutually "
                "exclusive");
        if (cfg.d3 || cfg.d3bj || cfg.d4 || cfg.ts || cfg.xdm || cfg.mbd) {
            std::vector<int> Z(static_cast<std::size_t>(na));
            std::vector<std::array<double, 3>> pos(
                static_cast<std::size_t>(na));
            for (int i = 0; i < na; ++i) {
                Z[static_cast<std::size_t>(i)] =
                    atoms[static_cast<std::size_t>(i)].z;
                pos[static_cast<std::size_t>(i)] = {
                    atoms[static_cast<std::size_t>(i)].pos[0],
                    atoms[static_cast<std::size_t>(i)].pos[1],
                    atoms[static_cast<std::size_t>(i)].pos[2]};
            }
            constexpr double kAng2Bohr = 1.0 / 0.52917720859;
            // Prefer SCF volume_ratios (energy fold); fall back to k-summed rho.
            const std::vector<double> vol_ratio = ts_mbd_vol_ratio(
                cfg, grid, rho, Z, pos, periodic, &scf.volume_ratios);
            if (cfg.mbd) {
                // Same isolated / periodic switch as Γ + fold_dispersion.
                const bool mbd_per = cfg.mbd_k_mesh[0] > 0
                                     || cfg.mbd_k_mesh[1] > 0
                                     || cfg.mbd_k_mesh[2] > 0;
                if (mbd_per && cfg.nonperiodic)
                    throw std::runtime_error("GPW k-forces MBD: mbd_k_mesh "
                                             "requires a periodic cell");
                const MbdResult mbd = mbd_rsscs_pbe(
                    Z, pos, cfg.cell, mbd_per, 0.83, 6.0, vol_ratio,
                    mbd_per ? cfg.mbd_k_mesh : nullptr);
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        forces[i][k] +=
                            mbd.forces[static_cast<std::size_t>(i)][k];
            } else if (cfg.xdm) {
                constexpr double kRcutXdm = 50.0 * kAng2Bohr;
                const XdmResult xdm =
                    xdm_pbe(Z, pos, cfg.cell, periodic, kRcutXdm);
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        forces[i][k] +=
                            xdm.forces[static_cast<std::size_t>(i)][k];
            } else if (cfg.ts) {
                constexpr double kRcutTs = 50.0 * kAng2Bohr;
                const TsResult ts = ts_pbe(Z, pos, cfg.cell, periodic, kRcutTs,
                                           0.94, 20.0, vol_ratio);
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        forces[i][k] +=
                            ts.forces[static_cast<std::size_t>(i)][k];
            } else if (cfg.d4) {
                const double s9 = cfg.d4_atm ? 1.0 : 0.0;
                const cpu::scf::D4Result d4 = cpu::scf::dftd4(
                    Z, pos, cfg.cell, periodic, /*s6=*/1.0,
                    /*s8=*/0.95948085, /*a1=*/0.38574991, /*a2=*/4.80688534,
                    /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0, /*charge=*/0.0,
                    /*cn_cut=*/30.0, /*disp2_cut=*/60.0, s9);
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        forces[i][k] +=
                            d4.forces[static_cast<std::size_t>(i)][k];
            } else {
                const D3Result d3 =
                    cfg.d3 ? dftd3_pbe(Z, pos, cfg.cell, periodic, D3Params{},
                                       cfg.d3_rcut_bohr, cfg.d3_atm)
                           : dftd3_pbe_bj(Z, pos, cfg.cell, periodic,
                                          D3Params{}, cfg.d3_rcut_bohr, 1.0,
                                          0.7875, 0.4289, 4.4407,
                                          /*cp2k_cn=*/true, cfg.d3_atm);
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        forces[i][k] +=
                            d3.forces[static_cast<std::size_t>(i)][k];
            }
        }
    }

    return forces;
}

} // namespace

std::vector<std::array<double, 3>>
gpw_forces_from_scf(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                    const GpwResult& scf) {
    if (!scf.converged)
        throw std::runtime_error("gpw_forces_from_scf: SCF did not converge");

    // Multi-k payload is C_occ_k / eps_k (not Γ P/C/eps). Branch first so
    // product evaluators can reuse one SCF on the k-path (gpw-mbd-kpoint-forces)
    // and so a k-path result is never rejected for missing Γ orbitals.
    if (uses_kpath(cfg) || scf.n_kpts > 0)
        return assemble_gpw_forces_kpoints(atoms, cfg, scf);

    if (scf.P.size() == 0 || scf.C.size() == 0 || scf.eps.empty())
        throw std::runtime_error(
            "gpw_forces_from_scf: converged result missing P/C/eps");

    if (cfg.is_uks()) {
        // W = Wα + Wβ with occupancy 1 per spin orbital.
        RealMatrix W = energy_weighted(scf.C, scf.eps, cfg.n_alpha, 1.0);
        if (cfg.n_beta > 0) {
            if (scf.C_beta.size() == 0 || scf.eps_beta.empty()
                || scf.P_beta.size() == 0)
                throw std::runtime_error("gpw_forces_from_scf: UKS result "
                                         "missing β orbitals/density");
            W += energy_weighted(scf.C_beta, scf.eps_beta, cfg.n_beta, 1.0);
        }
        return assemble_gpw_forces(atoms, cfg, scf.P, W, &scf.P_beta,
                                   &scf.volume_ratios, nullptr, &scf.veff_a,
                                   &scf.veff_b, &scf.v_es, &scf);
    }

    // Closed-shell RKS: W_μν = Σ_i 2 ε_i C_μi C_νi (occupied).
    const int n_occ =
        cfg.n_occ > 0 ? cfg.n_occ : static_cast<int>(scf.eps.size());
    RealMatrix W = energy_weighted(scf.C, scf.eps, n_occ, 2.0);
    return assemble_gpw_forces(atoms, cfg, scf.P, W, nullptr,
                               &scf.volume_ratios, &scf.veff, nullptr, nullptr,
                               &scf.v_es, &scf);
}

std::vector<std::array<double, 3>>
gpw_forces_analytic(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                    const linalg::RealMatrix* guess_C) {
    GpwDft solver = copy_gpw(cfg);
    solver.build(atoms);
    // UKS has no guess_C warm-start on the energy path; keep the cold solve.
    const GpwResult scf =
        cfg.is_uks() ? solver.run(atoms, nullptr) : solver.run(atoms, guess_C);
    if (!scf.converged)
        throw std::runtime_error("gpw_forces_analytic: SCF did not converge");

    if (uses_kpath(cfg)) {
        if (guess_C != nullptr)
            throw std::invalid_argument(
                "gpw_forces_analytic: k-path does not accept guess_C "
                "(no-fallback)");
        return assemble_gpw_forces_kpoints(atoms, cfg, scf);
    }
    return gpw_forces_from_scf(atoms, cfg, scf);
}

std::vector<std::array<double, 3>>
gpw_forces_at_density(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                      const linalg::RealMatrix& P,
                      const linalg::RealMatrix& C_occ,
                      const linalg::RealMatrix& H) {
    if (cfg.is_uks())
        throw std::invalid_argument(
            "gpw_forces_at_density: UKS requires the dual-spin overload "
            "(P_a, C_a, H_a, P_b, C_b, H_b)");
    const int nb = static_cast<int>(P.shape(0));
    const int no = static_cast<int>(C_occ.shape(1));
    if (static_cast<int>(C_occ.shape(0)) != nb)
        throw std::invalid_argument(
            "gpw_forces_at_density: C_occ rows must equal AO dimension");
    if (static_cast<int>(H.shape(0)) != nb
        || static_cast<int>(H.shape(1)) != nb)
        throw std::invalid_argument(
            "gpw_forces_at_density: H shape must be nb × nb");
    if (no < 1)
        throw std::invalid_argument(
            "gpw_forces_at_density: C_occ has no occupied columns");

    // Off-BO energy-weighted density: W = 2 C (C† H C) C†
    // (at the SCF fixed point this recovers W = 2 C ε C†).
    const RealMatrix HC = xt::linalg::dot(H, C_occ);
    const RealMatrix Lambda = xt::linalg::dot(xt::transpose(C_occ), HC);
    const RealMatrix W =
        2.0
        * xt::linalg::dot(C_occ, xt::linalg::dot(Lambda, xt::transpose(C_occ)));
    return assemble_gpw_forces(atoms, cfg, P, W);
}

namespace {
/// Off-BO energy-weighted density for one spin channel (occ=1):
/// W = C (C† H C) C†. Empty C (n_occ=0) → zero W.
RealMatrix energy_weighted_fixed(const RealMatrix& C, const RealMatrix& H,
                                 int nb) {
    if (C.size() == 0 || static_cast<int>(C.shape(1)) < 1)
        return xt::zeros<double>(
            {static_cast<std::size_t>(nb), static_cast<std::size_t>(nb)});
    const RealMatrix HC = xt::linalg::dot(H, C);
    const RealMatrix Lambda = xt::linalg::dot(xt::transpose(C), HC);
    return xt::linalg::dot(C, xt::linalg::dot(Lambda, xt::transpose(C)));
}
} // namespace

std::vector<std::array<double, 3>> gpw_forces_at_density(
    const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
    const linalg::RealMatrix& P_a, const linalg::RealMatrix& C_a,
    const linalg::RealMatrix& H_a, const linalg::RealMatrix& P_b,
    const linalg::RealMatrix& C_b, const linalg::RealMatrix& H_b) {
    if (!cfg.is_uks())
        throw std::invalid_argument(
            "gpw_forces_at_density (UKS): cfg is not UKS — use the RKS "
            "overload");
    const int nb = static_cast<int>(P_a.shape(0));
    if (static_cast<int>(P_a.shape(1)) != nb
        || static_cast<int>(P_b.shape(0)) != nb
        || static_cast<int>(P_b.shape(1)) != nb)
        throw std::invalid_argument(
            "gpw_forces_at_density (UKS): P_a/P_b must be nb × nb");
    if (static_cast<int>(H_a.shape(0)) != nb
        || static_cast<int>(H_a.shape(1)) != nb
        || static_cast<int>(H_b.shape(0)) != nb
        || static_cast<int>(H_b.shape(1)) != nb)
        throw std::invalid_argument(
            "gpw_forces_at_density (UKS): H_a/H_b must be nb × nb");
    if (C_a.size() > 0 && static_cast<int>(C_a.shape(0)) != nb)
        throw std::invalid_argument(
            "gpw_forces_at_density (UKS): C_a rows must equal nb");
    if (C_b.size() > 0 && static_cast<int>(C_b.shape(0)) != nb)
        throw std::invalid_argument(
            "gpw_forces_at_density (UKS): C_b rows must equal nb");
    if (cfg.n_alpha > 0
        && (C_a.size() == 0 || static_cast<int>(C_a.shape(1)) != cfg.n_alpha))
        throw std::invalid_argument(
            "gpw_forces_at_density (UKS): C_a must be nb × n_alpha");
    if (cfg.n_beta > 0
        && (C_b.size() == 0 || static_cast<int>(C_b.shape(1)) != cfg.n_beta))
        throw std::invalid_argument(
            "gpw_forces_at_density (UKS): C_b must be nb × n_beta");

    RealMatrix W = energy_weighted_fixed(C_a, H_a, nb);
    W += energy_weighted_fixed(C_b, H_b, nb);
    return assemble_gpw_forces(atoms, cfg, P_a, W, &P_b);
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
