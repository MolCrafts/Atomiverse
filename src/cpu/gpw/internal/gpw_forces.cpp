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
#include "atomiverse/cpu/gpw/internal/gpw_geometry.h"

#include <array>
#include <chrono>
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
#include "atomiverse/cpu/gpw/internal/core_overlap.h"
#include "atomiverse/cpu/gpw/internal/dft_plus_u.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/gth_integrals.h"
#include "atomiverse/cpu/gpw/internal/hirshfeld_volumes.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/cpu/gpw/internal/mt_poisson.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
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
namespace internal {

namespace {

bool uses_kpath(const GpwDft& cfg) {
    // Match SCF dispatch: metallic smear always rides the Bloch path
    // (including effective Γ via k_mesh = {1,1,1}).
    return !cfg.k_points.empty() || cfg.k_mesh[0] != 1 || cfg.k_mesh[1] != 1
           || cfg.k_mesh[2] != 1
           || cfg.smearing != ATV::electronic::Smearing::none;
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
    return hirshfeld_volume_ratios(grid, rho, Z, pos, cfg.cell_bohr().data(),
                                   periodic);
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
assemble_gpw_forces(const GpwGeometry& geometry, const GpwDft& cfg,
                    const RealMatrix& P, const RealMatrix& W,
                    const RealMatrix* P_beta,
                    const std::vector<double>* scf_vol_ratios,
                    const GpwWorkspace* scf_potentials) {
    const auto& atoms = geometry.atoms;
    require_gpw_xc(cfg.xc);
    const bool profile = std::getenv("ATV_GPW_PROF") != nullptr;
    using ForceClock = std::chrono::steady_clock;
    const auto force_start =
        profile ? ForceClock::now() : ForceClock::time_point{};

    const auto& basis = geometry.ao.basis;
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

    const auto& radii = geometry.ao.radii;
    const double rmax = geometry.ao.rmax;
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
    pair_image_counts(geometry.cell.data(), periodic,
                      std::max(2.0 * rmax, drmax + rmax), mimg);

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
                    S(mu, nu) = overlap(basis[mu], basis[nu],
                                        geometry.cell.data(), mimg, reach);
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
    const auto& grid = geometry.grid;
    const long ng = grid.npts();
    const double wgt = grid.weight();
    const auto& cbg = geometry.cbg;
    const auto& mg = geometry.product_grid;
    const auto& products = geometry.products;
    const bool product_grid = cbg.npts == 0 && !mg.levels.empty();
    if (!product_grid && cbg.npts != ng) {
        throw std::logic_error("GPW force requires an assembled AO grid");
    }
    auto collocate = [&](const std::vector<double>& density) {
        return product_grid
                   ? multigrid_collocate_rho(mg, products, nb, density, eps)
                   : collocate_density(cbg, nb, density);
    };

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
        rho_a = collocate(Pa_vec);
        rho_b = collocate(Pb_vec);
        rho.resize(rho_a.size());
        for (std::size_t g = 0; g < rho.size(); ++g)
            rho[g] = rho_a[g] + rho_b[g];
    } else {
        rho = collocate(Ptot_vec);
    }

    // Accepted product-grid forces use the potentials of the same solve.
    // An independent shadow density has no accepted potentials to borrow.
    const bool analytic_loc = !cfg.local_pp_on_grid;
    const auto npts = static_cast<std::size_t>(ng);
    const bool reuse = scf_potentials && scf_potentials->v_es.size() == npts
                       && (uks ? scf_potentials->veff_a.size() == npts
                                     && scf_potentials->veff_b.size() == npts
                               : scf_potentials->veff.size() == npts);
    if (product_grid && scf_potentials && !reuse)
        throw std::logic_error(
            "GPW product-grid force needs its SCF potentials");
    GpwWorkspace rebuilt;
    if (!reuse) {
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
            if (cfg.xc != xc::XCType::PADE && cfg.xc != xc::XCType::PBE
                && cfg.xc != xc::XCType::PBEsol && cfg.xc != xc::XCType::RevPBE
                && cfg.xc != xc::XCType::RPBE && cfg.xc != xc::XCType::BLYP
                && cfg.xc != xc::XCType::OptB88VdW)
                throw std::runtime_error("GPW UKS forces: xc::XCType must be "
                                         "PADE, PBE, PBEsol, RevPBE, "
                                         "RPBE, BLYP, or OptB88VdW");
            const XcOnGridSpin xc =
                cfg.xc == xc::XCType::PBE
                    ? xc_grid_pbe_spin(grid, rho_a, rho_b)
                    : (cfg.xc == xc::XCType::PBEsol
                           ? xc_grid_pbesol_spin(grid, rho_a, rho_b)
                           : (cfg.xc == xc::XCType::RevPBE
                                  ? xc_grid_revpbe_spin(grid, rho_a, rho_b)
                                  : (cfg.xc == xc::XCType::RPBE
                                         ? xc_grid_rpbe_spin(grid, rho_a, rho_b)
                                         : (cfg.xc == xc::XCType::BLYP
                                                ? xc_grid_blyp_spin(grid, rho_a,
                                                                    rho_b)
                                                : (cfg.xc
                                                           == xc::XCType::
                                                               OptB88VdW
                                                       ? xc_grid_optb88_vdw_spin(
                                                             grid, rho_a, rho_b)
                                                       : xc_grid_pade_lsd(
                                                             grid, rho_a,
                                                             rho_b))))));
            rebuilt.veff_a.resize(rho.size());
            rebuilt.veff_b.resize(rho.size());
            for (std::size_t g = 0; g < rho.size(); ++g) {
                double vloc = v_locsr.empty() ? 0.0 : v_locsr[g];
                rebuilt.veff_a[g] = es.v_r[g] + xc.vxc_a[g] + vloc;
                rebuilt.veff_b[g] = es.v_r[g] + xc.vxc_b[g] + vloc;
            }
        } else {
            const XcOnGrid xc =
                cfg.xc == xc::XCType::PBE
                    ? xc_grid_pbe(grid, rho)
                    : (cfg.xc == xc::XCType::PBEsol
                           ? xc_grid_pbesol(grid, rho)
                           : (cfg.xc == xc::XCType::RevPBE
                                  ? xc_grid_revpbe(grid, rho)
                                  : (cfg.xc == xc::XCType::RPBE
                                         ? xc_grid_rpbe(grid, rho)
                                         : (cfg.xc == xc::XCType::BLYP
                                                ? xc_grid_blyp(grid, rho)
                                                : (cfg.xc
                                                           == xc::XCType::
                                                               OptB88VdW
                                                       ? xc_grid_optb88_vdw(
                                                             grid, rho)
                                                       : xc_grid_pade(grid,
                                                                      rho))))));
            rebuilt.veff.resize(rho.size());
            for (std::size_t g = 0; g < rho.size(); ++g) {
                rebuilt.veff[g] = es.v_r[g] + xc.vxc_r[g];
                if (!v_locsr.empty())
                    rebuilt.veff[g] += v_locsr[g];
            }
        }
        // SC nonlocal vdW potential on total density (both spins share V_nl).
        if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
            NonlocalVdwOnGrid nl =
                nonlocal_vdw_on_grid(grid, rho, cfg.vdw_df, cfg.vdw_df2,
                                     cfg.rvv10, /*potential=*/true);
            if (uks) {
                for (std::size_t g = 0; g < rho.size(); ++g) {
                    rebuilt.veff_a[g] += nl.v_r[g];
                    rebuilt.veff_b[g] += nl.v_r[g];
                }
            } else {
                for (std::size_t g = 0; g < rho.size(); ++g)
                    rebuilt.veff[g] += nl.v_r[g];
            }
        }
        rebuilt.v_es = es.v_r;
    }
    const auto& potentials = reuse ? *scf_potentials : rebuilt;
    const auto& v_es = potentials.v_es;
    const auto& veff = potentials.veff;
    const auto& veff_a = potentials.veff_a;
    const auto& veff_b = potentials.veff_b;

    std::vector<std::array<double, 3>> forces(na, {0.0, 0.0, 0.0});

    const auto force_prepared =
        profile ? ForceClock::now() : ForceClock::time_point{};
    // ── AO-basis Pulay terms (periodic image sums) ──────────────────────────
    // Use P_tot for kinetic / V_pp (spin-independent operators).
    std::vector<const GpwAtom*> all(static_cast<std::size_t>(na));
    for (int atom = 0; atom < na; ++atom)
        all[static_cast<std::size_t>(atom)] = &atoms[atom];
    // Each task owns one force component. Keep its integral accumulation
    // order fixed; integral scratch is local or thread-local.
#pragma omp parallel for collapse(2) schedule(static, 1)
    for (int A = 0; A < na; ++A) {
        for (int k = 0; k < 3; ++k) {
            double f = 0.0;
            const std::vector<const GpwAtom*> onlyA = {&atoms[A]};

            for (int mu = atom_off[A]; mu < atom_off[A + 1]; ++mu) {
                const GridBasisFunction& dmu = dphi[mu][k];
                const double rdmu = drad[mu][k];
                for (int nu = 0; nu < nb; ++nu) {
                    const double reach =
                        rdmu + radii[static_cast<std::size_t>(nu)];
                    double s = 0.0, t = 0.0;
                    contracted_ST_periodic(dmu, basis[nu], geometry.cell.data(),
                                           mimg, reach, s, t);
                    f += 2.0 * (*W_use)(mu, nu) * s; // + W dS (+U fold inside)
                    f -= 2.0 * Ptot(mu, nu) * t;     // - P dT
                    f -= 2.0 * Ptot(mu, nu)
                         * vpp_images(dmu, basis[nu], all, geometry.cell.data(),
                                      mimg, reach,
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
                         * vpp_images(dmu, basis[nu], onlyA,
                                      geometry.cell.data(), mimg, reach,
                                      analytic_loc);
                }
            }
            forces[A][k] = f;
        }
    }

    const auto force_ao =
        profile ? ForceClock::now() : ForceClock::time_point{};
    // Derivative of the same periodized product energy used by SCF.
    if (product_grid) {
        const auto ga = internal::multigrid_density_center_gradient(
            mg, products, nb, uks ? Pa_vec : Ptot_vec, uks ? veff_a : veff,
            owner, na, eps);
        std::vector<std::array<double, 3>> gb;
        if (uks)
            gb = internal::multigrid_density_center_gradient(
                mg, products, nb, Pb_vec, veff_b, owner, na, eps);
        for (int atom = 0; atom < na; ++atom)
            for (int axis = 0; axis < 3; ++axis)
                forces[atom][axis] -=
                    ga[atom][axis] + (uks ? gb[atom][axis] : 0);
    } else {
        // g -> active index into cbg (for the grid term's Pχ lookup).
        std::vector<long> g2a(static_cast<std::size_t>(ng), -1);
        for (std::size_t a = 0; a < cbg.gidx.size(); ++a)
            g2a[static_cast<std::size_t>(cbg.gidx[a])] = static_cast<long>(a);

        // Stream one AO's three center derivatives through the established
        // image-summed collocation kernel. Materializing all 3*nb functions
        // at once duplicates the large CSR construction payload and dominates
        // periodic force time/memory. Each task owns its compact grid and xyz
        // integral; the shared base CBG and SCF fields are read-only.
        std::vector<std::array<double, 3>> ao_grid(nb, {0.0, 0.0, 0.0});
        std::vector<std::array<double, 2>> profile_ao(nb, {0.0, 0.0});
#pragma omp parallel for schedule(dynamic, 1)
        for (int u = 0; u < nb; ++u) {
            const auto build_start =
                profile ? ForceClock::now() : ForceClock::time_point{};
            const std::vector<GridBasisFunction> derivatives{
                dphi[u][0], dphi[u][1], dphi[u][2]};
            const CompactBasisGrid cbgd =
                build_compact_basis_grid(grid, derivatives, periodic, eps);
            const auto contract_start =
                profile ? ForceClock::now() : ForceClock::time_point{};
            auto& integral = ao_grid[u];
            const double* Pa =
                &(uks ? Pa_vec : Ptot_vec)[static_cast<std::size_t>(u) * nb];
            const double* Pb =
                uks ? &Pb_vec[static_cast<std::size_t>(u) * nb] : nullptr;
            for (std::size_t ad = 0; ad < cbgd.gidx.size(); ++ad) {
                const long g = cbgd.gidx[ad];
                const long a = g2a[static_cast<std::size_t>(g)];
                if (a < 0)
                    continue; // All chi_nu are zero here, so P*chi is zero.
                const std::size_t cs = cbg.off[static_cast<std::size_t>(a)];
                const std::size_t ce = cbg.off[static_cast<std::size_t>(a) + 1];
                double pchi_a = 0.0, pchi_b = 0.0;
                for (std::size_t ib = cs; ib < ce; ++ib) {
                    pchi_a += Pa[cbg.fu[ib]] * cbg.fv[ib];
                    if (uks)
                        pchi_b += Pb[cbg.fu[ib]] * cbg.fv[ib];
                }
                const double v_pchi =
                    uks ? veff_a[static_cast<std::size_t>(g)] * pchi_a
                              + veff_b[static_cast<std::size_t>(g)] * pchi_b
                        : veff[static_cast<std::size_t>(g)] * pchi_a;
                for (std::size_t id = cbgd.off[ad]; id < cbgd.off[ad + 1]; ++id)
                    integral[cbgd.fu[id]] -= 2.0 * wgt * cbgd.fv[id] * v_pchi;
            }
            if (profile) {
                const auto end = ForceClock::now();
                profile_ao[u] = {std::chrono::duration<double, std::milli>(
                                     contract_start - build_start)
                                     .count(),
                                 std::chrono::duration<double, std::milli>(
                                     end - contract_start)
                                     .count()};
            }
        }
        // Fixed AO order makes the fold independent of OpenMP scheduling.
        for (int u = 0; u < nb; ++u)
            for (int axis = 0; axis < 3; ++axis)
                forces[owner[u]][axis] += ao_grid[u][axis];
        if (profile) {
            double build_ms = 0.0, contract_ms = 0.0;
            for (const auto& times : profile_ao) {
                build_ms += times[0];
                contract_ms += times[1];
            }
            std::fprintf(stderr,
                         "[gpw force grid] AO-task sums: build=%.1f "
                         "contract=%.1f ms\n",
                         build_ms, contract_ms);
        }
    }

    const auto force_grid =
        profile ? ForceClock::now() : ForceClock::time_point{};
    // ── Ionic Hellmann–Feynman: + ∫ V_es ∂n_c^A/∂R_A,k (min-image, matches SCF) ─
#pragma omp parallel for schedule(static)
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

    // Restore the point-ion interaction removed by Gaussian core smearing.
    // This density-independent term also applies to fixed-density forces.
    const auto core_overlap =
        make_core_overlap_model(atoms, geometry.cell.data(), periodic);
    add_core_overlap_forces(core_overlap, forces);

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
                mbd_rsscs_pbe(Z, pos, geometry.cell.data(), mbd_per, 0.83, 6.0,
                              vol_ratio, mbd_per ? cfg.mbd_k_mesh : nullptr);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += mbd.forces[static_cast<std::size_t>(i)][k];
        } else if (cfg.xdm) {
            constexpr double kRcutXdm = 50.0 * kAng2Bohr;
            const XdmResult xdm =
                xdm_pbe(Z, pos, geometry.cell.data(), periodic, kRcutXdm);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += xdm.forces[static_cast<std::size_t>(i)][k];
        } else if (cfg.ts) {
            constexpr double kRcutTs = 50.0 * kAng2Bohr;
            const TsResult ts = ts_pbe(Z, pos, geometry.cell.data(), periodic,
                                       kRcutTs, 0.94, 20.0, vol_ratio);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += ts.forces[static_cast<std::size_t>(i)][k];
        } else if (cfg.d4) {
            const double s9 = cfg.d4_atm ? 1.0 : 0.0;
            const cpu::scf::D4Result d4 = cpu::scf::dftd4(
                Z, pos, geometry.cell.data(), periodic, /*s6=*/1.0,
                /*s8=*/0.95948085, /*a1=*/0.38574991,
                /*a2=*/4.80688534, /*ga=*/3.0, /*gc=*/2.0,
                /*wf=*/6.0, /*charge=*/0.0, /*cn_cut=*/30.0,
                /*disp2_cut=*/60.0, s9);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += d4.forces[static_cast<std::size_t>(i)][k];
        } else {
            const D3Result d3 =
                cfg.d3 ? dftd3_pbe(Z, pos, geometry.cell.data(), periodic,
                                   D3Params{}, cfg.d3_rcut_bohr, cfg.d3_atm)
                       : dftd3_pbe_bj(Z, pos, geometry.cell.data(), periodic,
                                      D3Params{}, cfg.d3_rcut_bohr, 1.0, 0.7875,
                                      0.4289, 4.4407,
                                      /*cp2k_cn=*/true, cfg.d3_atm);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < 3; ++k)
                    forces[i][k] += d3.forces[static_cast<std::size_t>(i)][k];
        }
    }
    if (profile) {
        const auto end = ForceClock::now();
        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::fprintf(stderr,
                     "[gpw forces] setup=%.1f ao=%.1f grid=%.1f "
                     "ionic_dispersion=%.1f ms\n",
                     ms(force_start, force_prepared),
                     ms(force_prepared, force_ao), ms(force_ao, force_grid),
                     ms(force_grid, end));
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
assemble_gpw_forces_kpoints(const GpwGeometry& geometry, const GpwDft& cfg,
                            const GpwResult& scf,
                            const GpwWorkspace& workspace) {
    const auto& atoms = geometry.atoms;
    require_gpw_xc(cfg.xc);
    const auto& basis = geometry.ao.basis;
    const int nb = static_cast<int>(basis.size());
    const int na = static_cast<int>(atoms.size());
    const bool uks = cfg.is_uks();
    const bool metallic = (cfg.smearing != ATV::electronic::Smearing::none);
    const int n_occ_a = uks ? cfg.n_alpha : cfg.n_occ;
    const int n_occ_b = uks ? cfg.n_beta : 0;
    const auto& kpts = geometry.kpoints;
    const int nk = static_cast<int>(kpts.size());
    if (nk < 1 || scf.n_kpts != nk || static_cast<int>(scf.eps_k.size()) != nk)
        throw std::runtime_error("GPW k-point forces require the assembled "
                                 "sampling and SCF energies");
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
    if (uks && cfg.xc != xc::XCType::PADE && cfg.xc != xc::XCType::PBE
        && cfg.xc != xc::XCType::PBEsol && cfg.xc != xc::XCType::RevPBE
        && cfg.xc != xc::XCType::RPBE && cfg.xc != xc::XCType::BLYP
        && cfg.xc != xc::XCType::OptB88VdW)
        throw std::runtime_error("gpw_forces kpoints UKS: xc::XCType must be "
                                 "PADE, PBE, PBEsol, RevPBE, "
                                 "RPBE, BLYP, or OptB88VdW");

    const auto& mg = geometry.product_grid;
    const auto& products = geometry.products;
    const bool use_product_gradient = !mg.levels.empty();
    if (!use_product_gradient
        && static_cast<int>(geometry.kbg.chi.size()) != nk)
        throw std::logic_error(
            "GPW k-point forces require the assembled Bloch table");
    const bool borrow_density = !uks && (!metallic || use_product_gradient);
    const bool borrow_spin = uks && scf.iters > 0;
    if (borrow_spin
        && (workspace.accepted_P_k.size() != kpts.size()
            || workspace.accepted_P_k_beta.size() != kpts.size()))
        throw std::logic_error(
            "GPW k-point UKS forces require both accepted spin densities");
    const auto& accepted_P_k = scf.iters > 0 ? workspace.accepted_P_k : scf.P_k;
    if (borrow_density && static_cast<int>(accepted_P_k.size()) != nk)
        throw std::logic_error(
            "GPW k-point forces require the accepted density");

    // P_tot^k / W^k (and spin Pa^k / Pb^k for UKS grid).
    // Fixed: RKS occ=2; UKS P_tot=Pa+Pb, W=Wa+Wb with occ=1.
    // Metallic: fractional f from SCF (Mermin free-energy Hellmann–Feynman).
    std::vector<CplxMatrix> rebuilt_P_k;
    if (!borrow_density)
        rebuilt_P_k.resize(static_cast<std::size_t>(nk));
    const auto& P_k = borrow_density ? accepted_P_k : rebuilt_P_k;
    std::vector<CplxMatrix> W_k(static_cast<std::size_t>(nk));
    std::vector<CplxMatrix> rebuilt_Pa_k, rebuilt_Pb_k;
    if (uks && !borrow_spin) {
        rebuilt_Pa_k.resize(static_cast<std::size_t>(nk));
        rebuilt_Pb_k.resize(static_cast<std::size_t>(nk));
    }
    const auto& Pa_k = borrow_spin ? workspace.accepted_P_k : rebuilt_Pa_k;
    const auto& Pb_k = borrow_spin ? workspace.accepted_P_k_beta : rebuilt_Pb_k;
    for (int ik = 0; ik < nk; ++ik) {
        const auto k = static_cast<std::size_t>(ik);
        if (borrow_spin) {
            rebuilt_P_k[k] = Pa_k[k] + Pb_k[k];
            if (metallic) {
                W_k[k] =
                    internal::energy_weighted_from_frac(
                        scf.C_k[k], scf.eps_k[k], scf.f_k[k])
                    + internal::energy_weighted_from_frac(
                        scf.C_k_beta[k], scf.eps_k_beta[k], scf.f_k_beta[k]);
            } else {
                W_k[k] = internal::energy_weighted_from_occ(
                             scf.C_occ_k[k], scf.eps_k[k], nb, n_occ_a, 1.0)
                         + internal::energy_weighted_from_occ(
                             scf.C_occ_k_beta[k], scf.eps_k_beta[k], nb,
                             n_occ_b, 1.0);
            }
            continue;
        }
        if (borrow_density) {
            W_k[k] = metallic ? internal::energy_weighted_from_frac(
                                    scf.C_k[k], scf.eps_k[k], scf.f_k[k])
                              : internal::energy_weighted_from_occ(
                                    scf.C_occ_k[k], scf.eps_k[k], nb, n_occ_a);
            continue;
        }
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
                rebuilt_Pa_k[static_cast<std::size_t>(ik)] = std::move(Pa);
                rebuilt_Pb_k[static_cast<std::size_t>(ik)] = std::move(Pb);
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
            rebuilt_Pa_k[static_cast<std::size_t>(ik)] = std::move(Pa);
            rebuilt_Pb_k[static_cast<std::size_t>(ik)] = std::move(Pb);
        } else {
            add_occ_PW(scf.C_occ_k[static_cast<std::size_t>(ik)],
                       scf.eps_k[static_cast<std::size_t>(ik)], nb, n_occ_a,
                       2.0, P, W);
        }
        rebuilt_P_k[static_cast<std::size_t>(ik)] = std::move(P);
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
    const auto& radii = geometry.ao.radii;
    const double rmax = geometry.ao.rmax;
    std::vector<std::array<GridBasisFunction, 3>> dphi(nb);
    std::vector<std::array<double, 3>> drad(nb);
    double drmax = 0.0;
    for (int u = 0; u < nb; ++u)
        for (int k = 0; k < 3; ++k) {
            dphi[u][k] = deriv_bf(basis[u], k);
            drad[u][k] = basis_function_radius(dphi[u][k], eps);
            drmax = std::max(drmax, drad[u][k]);
        }
    int mimg[3];
    pair_image_counts(geometry.cell.data(), /*periodic=*/true,
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
            internal::build_overlap_kinetic_k(basis, radii,
                                              geometry.cell.data(), mimg,
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

    const auto& grid = geometry.grid;
    const long ng = grid.npts();
    const double wgt = grid.weight();
    const auto& kbg = geometry.kbg;
    auto collocate = [&](const std::vector<CplxMatrix>& density,
                         std::vector<std::vector<double>>& cells) {
        if (use_product_gradient) {
            cells = internal::build_bloch_density_stack(geometry, density);
            return multigrid_collocate_rho_cells(mg, products, nb, cells, eps);
        }
        std::vector<double> rho(static_cast<std::size_t>(ng), 0.0);
        for (int ik = 0; ik < nk; ++ik) {
            const auto k = static_cast<std::size_t>(ik);
            internal::collocate_density_k(kbg, ik, density[k], kpts[k].weight,
                                          rho);
        }
        return rho;
    };
    std::vector<double> rho;
    std::vector<double> rho_a, rho_b;
    std::vector<std::vector<double>> D, D_beta;
    if (uks) {
        rho_a = collocate(Pa_k, D);
        rho_b = collocate(Pb_k, D_beta);
        rho.resize(rho_a.size());
        for (std::size_t g = 0; g < rho.size(); ++g)
            rho[g] = rho_a[g] + rho_b[g];
    } else {
        rho = collocate(P_k, D);
    }

    const bool reuse_uks =
        uks && workspace.veff_a.size() == static_cast<std::size_t>(ng)
        && workspace.veff_b.size() == static_cast<std::size_t>(ng)
        && workspace.v_es.size() == static_cast<std::size_t>(ng);
    const bool reuse_rks =
        !uks && workspace.veff.size() == static_cast<std::size_t>(ng)
        && workspace.v_es.size() == static_cast<std::size_t>(ng);

    const bool reuse_potentials = reuse_rks || reuse_uks;
    if ((borrow_density || borrow_spin || use_product_gradient)
        && !reuse_potentials)
        throw std::logic_error(
            "GPW k-point forces require the accepted potentials");
    GpwWorkspace rebuilt;
    if (!reuse_potentials) {
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
                cfg.xc == xc::XCType::PBE
                    ? xc_grid_pbe_spin(grid, rho_a, rho_b)
                    : (cfg.xc == xc::XCType::PBEsol
                           ? xc_grid_pbesol_spin(grid, rho_a, rho_b)
                           : (cfg.xc == xc::XCType::RevPBE
                                  ? xc_grid_revpbe_spin(grid, rho_a, rho_b)
                                  : (cfg.xc == xc::XCType::RPBE
                                         ? xc_grid_rpbe_spin(grid, rho_a, rho_b)
                                         : (cfg.xc == xc::XCType::BLYP
                                                ? xc_grid_blyp_spin(grid, rho_a,
                                                                    rho_b)
                                                : (cfg.xc
                                                           == xc::XCType::
                                                               OptB88VdW
                                                       ? xc_grid_optb88_vdw_spin(
                                                             grid, rho_a, rho_b)
                                                       : xc_grid_pade_lsd(
                                                             grid, rho_a,
                                                             rho_b))))));
            rebuilt.veff_a.resize(rho.size());
            rebuilt.veff_b.resize(rho.size());
            for (std::size_t g = 0; g < rho.size(); ++g) {
                rebuilt.veff_a[g] = es.v_r[g] + xc.vxc_a[g] + v_locsr[g];
                rebuilt.veff_b[g] = es.v_r[g] + xc.vxc_b[g] + v_locsr[g];
            }
        } else {
            const XcOnGrid xc =
                cfg.xc == xc::XCType::PBE
                    ? xc_grid_pbe(grid, rho)
                    : (cfg.xc == xc::XCType::PBEsol
                           ? xc_grid_pbesol(grid, rho)
                           : (cfg.xc == xc::XCType::RevPBE
                                  ? xc_grid_revpbe(grid, rho)
                                  : (cfg.xc == xc::XCType::RPBE
                                         ? xc_grid_rpbe(grid, rho)
                                         : (cfg.xc == xc::XCType::BLYP
                                                ? xc_grid_blyp(grid, rho)
                                                : (cfg.xc
                                                           == xc::XCType::
                                                               OptB88VdW
                                                       ? xc_grid_optb88_vdw(
                                                             grid, rho)
                                                       : xc_grid_pade(grid,
                                                                      rho))))));
            rebuilt.veff.resize(rho.size());
            for (std::size_t g = 0; g < rho.size(); ++g)
                rebuilt.veff[g] = es.v_r[g] + xc.vxc_r[g] + v_locsr[g];
        }
        if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
            NonlocalVdwOnGrid nl =
                nonlocal_vdw_on_grid(grid, rho, cfg.vdw_df, cfg.vdw_df2,
                                     cfg.rvv10, /*potential=*/true);
            if (uks) {
                for (std::size_t g = 0; g < rho.size(); ++g) {
                    rebuilt.veff_a[g] += nl.v_r[g];
                    rebuilt.veff_b[g] += nl.v_r[g];
                }
            } else {
                for (std::size_t g = 0; g < rho.size(); ++g)
                    rebuilt.veff[g] += nl.v_r[g];
            }
        }
        rebuilt.v_es = es.v_r;
    }
    const auto& potentials = reuse_potentials ? workspace : rebuilt;
    const auto& veff = potentials.veff;
    const auto& veff_a = potentials.veff_a;
    const auto& veff_b = potentials.veff_b;
    const auto& v_es = potentials.v_es;

    std::vector<std::array<double, 3>> forces(na, {0.0, 0.0, 0.0});

    // Overlap and kinetic Pulay terms. Nonlocal force uses the same
    // separable Bloch projector overlaps as the SCF operator.
    internal::add_nonlocal_kb_k_forces(geometry, dphi, owner, P_k,
                                       /*screen=*/1e-12, forces);
    for (int k = 0; k < 3; ++k) {
#pragma omp parallel
        {
            std::vector<Cplx> s_k, t_k;
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
                    pair_ST_k(dmu, basis[nu], geometry.cell.data(), mimg, reach,
                              kpts, s_k, t_k);
                    for (int ik = 0; ik < nk; ++ik) {
                        const double wk =
                            kpts[static_cast<std::size_t>(ik)].weight;
                        const Cplx Wmn =
                            W_k[static_cast<std::size_t>(ik)](mu, nu);
                        const Cplx Pmn =
                            P_k[static_cast<std::size_t>(ik)](mu, nu);
                        const Cplx s = s_k[static_cast<std::size_t>(ik)];
                        const Cplx t = t_k[static_cast<std::size_t>(ik)];
                        // 2 Re(W† s) with Hermitian W → 2 Re(conj(W)*s)
                        f += 2.0 * wk * std::real(std::conj(Wmn) * s);
                        f -= 2.0 * wk * std::real(std::conj(Pmn) * t);
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

    if (use_product_gradient) {
        auto subtract_gradient =
            [&](const std::vector<std::vector<double>>& cells,
                const std::vector<double>& potential) {
                const auto gradient =
                    internal::multigrid_density_center_gradient_cells(
                        mg, products, nb, cells, potential, owner, na, eps);
                for (int a = 0; a < na; ++a)
                    for (int d = 0; d < 3; ++d)
                        forces[a][d] -= gradient[a][d];
            };
        subtract_gradient(D, uks ? veff_a : veff);
        if (uks)
            subtract_gradient(D_beta, veff_b);
    } else {
        // Derivative basis on grid (flat u*3+k), local to this force call.
        std::vector<GridBasisFunction> dflat(static_cast<std::size_t>(nb) * 3);
        for (int u = 0; u < nb; ++u)
            for (int k = 0; k < 3; ++k)
                dflat[static_cast<std::size_t>(u) * 3 + k] = dphi[u][k];
        const auto kbgd =
            internal::build_kpoint_basis_grid(grid, dflat, kpts, eps);

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
                    const Cplx* v =
                        chi.data() + static_cast<std::size_t>(g) * nb;
                    const Cplx* dv =
                        dchi.data() + static_cast<std::size_t>(g) * nbd;
                    if (uks) {
                        const CplxMatrix& Pak =
                            Pa_k[static_cast<std::size_t>(ik)];
                        const CplxMatrix& Pbk =
                            Pb_k[static_cast<std::size_t>(ik)];
                        for (int mu = 0; mu < nb; ++mu) {
                            Cplx sa(0, 0), sb(0, 0);
                            for (int nu = 0; nu < nb; ++nu) {
                                sa += Pak(mu, nu) * std::conj(v[nu]);
                                sb += Pbk(mu, nu) * std::conj(v[nu]);
                            }
                            ya[static_cast<std::size_t>(mu)] = sa;
                            yb[static_cast<std::size_t>(mu)] = sb;
                        }
                        const double va =
                            veff_a[static_cast<std::size_t>(g)] * wgt;
                        const double vb =
                            veff_b[static_cast<std::size_t>(g)] * wgt;
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
                                               * ya[static_cast<std::size_t>(
                                                   u)])
                                       + vb
                                             * std::real(
                                                 dval
                                                 * yb[static_cast<std::size_t>(
                                                     u)]));
                            }
                    } else {
                        const double vw =
                            veff[static_cast<std::size_t>(g)] * wgt;
                        if (vw == 0.0)
                            continue;
                        const CplxMatrix& Pk =
                            P_k[static_cast<std::size_t>(ik)];
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
                                    * std::real(
                                        dval * y[static_cast<std::size_t>(u)]);
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

    const auto core_overlap =
        make_core_overlap_model(atoms, geometry.cell.data(), true);
    add_core_overlap_forces(core_overlap, forces);

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
                    Z, pos, geometry.cell.data(), mbd_per, 0.83, 6.0, vol_ratio,
                    mbd_per ? cfg.mbd_k_mesh : nullptr);
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        forces[i][k] +=
                            mbd.forces[static_cast<std::size_t>(i)][k];
            } else if (cfg.xdm) {
                constexpr double kRcutXdm = 50.0 * kAng2Bohr;
                const XdmResult xdm =
                    xdm_pbe(Z, pos, geometry.cell.data(), periodic, kRcutXdm);
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        forces[i][k] +=
                            xdm.forces[static_cast<std::size_t>(i)][k];
            } else if (cfg.ts) {
                constexpr double kRcutTs = 50.0 * kAng2Bohr;
                const TsResult ts =
                    ts_pbe(Z, pos, geometry.cell.data(), periodic, kRcutTs,
                           0.94, 20.0, vol_ratio);
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        forces[i][k] +=
                            ts.forces[static_cast<std::size_t>(i)][k];
            } else if (cfg.d4) {
                const double s9 = cfg.d4_atm ? 1.0 : 0.0;
                const cpu::scf::D4Result d4 = cpu::scf::dftd4(
                    Z, pos, geometry.cell.data(), periodic, /*s6=*/1.0,
                    /*s8=*/0.95948085, /*a1=*/0.38574991, /*a2=*/4.80688534,
                    /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0, /*charge=*/0.0,
                    /*cn_cut=*/30.0, /*disp2_cut=*/60.0, s9);
                for (int i = 0; i < na; ++i)
                    for (int k = 0; k < 3; ++k)
                        forces[i][k] +=
                            d4.forces[static_cast<std::size_t>(i)][k];
            } else {
                const D3Result d3 =
                    cfg.d3
                        ? dftd3_pbe(Z, pos, geometry.cell.data(), periodic,
                                    D3Params{}, cfg.d3_rcut_bohr, cfg.d3_atm)
                        : dftd3_pbe_bj(Z, pos, geometry.cell.data(), periodic,
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
gpw_forces_from_scf(const GpwGeometry& geometry, const GpwDft& cfg,
                    const GpwResult& scf, const GpwWorkspace& workspace) {
    if (!scf.converged)
        throw std::runtime_error("gpw_forces_from_scf: SCF did not converge");

    // Multi-k payload is C_occ_k / eps_k (not Γ P/C/eps). Branch first so
    // product evaluators can reuse one SCF on the k-path (gpw-mbd-kpoint-forces)
    // and so a k-path result is never rejected for missing Γ orbitals.
    if (uses_kpath(cfg) || scf.n_kpts > 0)
        return assemble_gpw_forces_kpoints(geometry, cfg, scf, workspace);

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
        return assemble_gpw_forces(geometry, cfg, scf.P, W, &scf.P_beta,
                                   &scf.volume_ratios, &workspace);
    }

    // Closed-shell RKS: W_μν = Σ_i 2 ε_i C_μi C_νi (occupied).
    const int n_occ =
        cfg.n_occ > 0 ? cfg.n_occ : static_cast<int>(scf.eps.size());
    RealMatrix W = energy_weighted(scf.C, scf.eps, n_occ, 2.0);
    return assemble_gpw_forces(geometry, cfg, scf.P, W, nullptr,
                               &scf.volume_ratios, &workspace);
}

std::vector<std::array<double, 3>>
gpw_forces_analytic(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                    const linalg::RealMatrix* guess_C) {
    GpwDft solver = GpwDftAccess::clone(cfg);
    GpwDftAccess::build(solver, atoms);
    // UKS has no guess_C warm-start on the energy path; keep the cold solve.
    const auto& scf =
        GpwDftAccess::run(solver, atoms, cfg.is_uks() ? nullptr : guess_C);
    if (!scf.converged)
        throw std::runtime_error("gpw_forces_analytic: SCF did not converge");

    if (uses_kpath(cfg) && guess_C != nullptr)
        throw std::invalid_argument("gpw_forces_analytic: k-path does not "
                                    "accept guess_C (no-fallback)");
    return gpw_forces_from_scf(GpwDftAccess::geometry(solver), cfg, scf,
                               GpwDftAccess::workspace(solver));
}

std::vector<std::array<double, 3>> gpw_forces_at_density(
    const GpwGeometry& geometry, const GpwDft& cfg, const linalg::RealMatrix& P,
    const linalg::RealMatrix& C_occ, const linalg::RealMatrix& H,
    const GpwResult* matching_density, const GpwWorkspace* workspace) {
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
    if (matching_density && &P != &matching_density->P)
        throw std::invalid_argument(
            "fixed-density force payload does not own P");
    return assemble_gpw_forces(
        geometry, cfg, P, W, nullptr,
        matching_density ? &matching_density->volume_ratios : nullptr,
        matching_density ? workspace : nullptr);
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
    const GpwGeometry& geometry, const GpwDft& cfg,
    const linalg::RealMatrix& P_a, const linalg::RealMatrix& C_a,
    const linalg::RealMatrix& H_a, const linalg::RealMatrix& P_b,
    const linalg::RealMatrix& C_b, const linalg::RealMatrix& H_b,
    const GpwResult* matching_density, const GpwWorkspace* workspace) {
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
    if (matching_density
        && (&P_a != &matching_density->P || &P_b != &matching_density->P_beta))
        throw std::invalid_argument(
            "fixed-density UKS payload does not own both densities");
    return assemble_gpw_forces(
        geometry, cfg, P_a, W, &P_b,
        matching_density ? &matching_density->volume_ratios : nullptr,
        matching_density ? workspace : nullptr);
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
