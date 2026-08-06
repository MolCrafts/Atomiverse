/// @file src/cpu/gpw/internal/ks_grid_build.cpp
/// @brief Shared fine-grid KS build implementation.

#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"

#include "atomiverse/cpu/gpw/gpw_solver.h" // XcKind
#include "atomiverse/cpu/gpw/internal/mt_poisson.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"
#include "atomiverse/cpu/gpw/internal/vdw_df.h"
#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"

#include <stdexcept>

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

XcOnGrid evaluate_xc_rks(const UniformGrid& grid, const std::vector<double>& rho,
                         XcKind xc) {
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
        return xc_grid_pade(grid, rho);
    }
    return xc_grid_pade(grid, rho);
}

XcOnGridSpin evaluate_xc_uks(const UniformGrid& grid,
                             const std::vector<double>& rho_a,
                             const std::vector<double>& rho_b, XcKind xc) {
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
        return xc_grid_pade_lsd(grid, rho_a, rho_b);
    }
    return xc_grid_pade_lsd(grid, rho_a, rho_b);
}

namespace {

void check_density_sizes(const UniformGrid& grid, std::size_t n_rho,
                         std::size_t n_ion, const char* tag) {
    if (n_rho != static_cast<std::size_t>(grid.npts())
        || n_ion != static_cast<std::size_t>(grid.npts()))
        throw std::runtime_error(
            std::string("GPW ") + tag
            + ": density grid size mismatch (rho=" + std::to_string(n_rho)
            + " n_ion=" + std::to_string(n_ion)
            + " grid.npts=" + std::to_string(grid.npts()) + ")");
}

PoissonResult poisson(const UniformGrid& grid, const std::vector<double>& n_tot,
                      bool nonperiodic) {
    return nonperiodic ? solve_poisson_mt(grid, n_tot)
                       : solve_poisson(grid, n_tot);
}

} // namespace

KsGridRks ks_grid_build_rks(const UniformGrid& grid,
                            const std::vector<double>& rho,
                            const std::vector<double>& n_ion,
                            const std::vector<double>& v_locsr, XcKind xc,
                            bool nonperiodic, bool vdw_df, bool vdw_df2,
                            bool rvv10) {
    check_density_sizes(grid, rho.size(), n_ion.size(), "ks_grid_build_rks");
    std::vector<double> n_tot(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g)
        n_tot[g] = rho[g] - n_ion[g];
    const PoissonResult es = poisson(grid, n_tot, nonperiodic);
    const XcOnGrid xcg = evaluate_xc_rks(grid, rho, xc);

    KsGridRks out;
    out.e_xc = xcg.e_xc;
    out.e_hartree = es.e_hartree;
    out.v_es = es.v_r;
    out.veff.resize(rho.size());
    const bool have_loc = !v_locsr.empty();
    if (have_loc && v_locsr.size() != rho.size())
        throw std::runtime_error(
            "GPW ks_grid_build_rks: v_locsr size does not match density grid");
    for (std::size_t g = 0; g < rho.size(); ++g) {
        const double vloc = have_loc ? v_locsr[g] : 0.0;
        out.veff[g] = out.v_es[g] + xcg.vxc_r[g] + vloc;
        if (have_loc)
            out.e_locgrid += rho[g] * vloc;
    }
    out.e_locgrid *= grid.weight();

    if (vdw_df || vdw_df2 || rvv10) {
        NonlocalVdwOnGrid nl =
            nonlocal_vdw_on_grid(grid, rho, vdw_df, vdw_df2, rvv10,
                                 /*potential=*/true);
        out.e_nl_vdw = nl.energy;
        for (std::size_t g = 0; g < rho.size(); ++g)
            out.veff[g] += nl.v_r[g];
    }
    return out;
}

KsGridUks ks_grid_build_uks(const UniformGrid& grid,
                            const std::vector<double>& rho_a,
                            const std::vector<double>& rho_b,
                            const std::vector<double>& n_ion,
                            const std::vector<double>& v_locsr, XcKind xc,
                            bool nonperiodic, bool vdw_df, bool vdw_df2,
                            bool rvv10) {
    if (rho_a.size() != rho_b.size())
        throw std::runtime_error(
            "GPW ks_grid_build_uks: rho_a / rho_b size mismatch");
    check_density_sizes(grid, rho_a.size(), n_ion.size(), "ks_grid_build_uks");
    // Single pass: total ρ and n = ρ − n_ion (shared Poisson input).
    std::vector<double> rho(rho_a.size());
    std::vector<double> n_tot(rho_a.size());
    const std::size_t ng = rho_a.size();
#pragma omp parallel for schedule(static) if (ng > 4096)
    for (std::size_t g = 0; g < ng; ++g) {
        const double rt = rho_a[g] + rho_b[g];
        rho[g] = rt;
        n_tot[g] = rt - n_ion[g];
    }
    const PoissonResult es = poisson(grid, n_tot, nonperiodic);
    const XcOnGridSpin xcg = evaluate_xc_uks(grid, rho_a, rho_b, xc);

    KsGridUks out;
    out.e_xc = xcg.e_xc;
    out.e_hartree = es.e_hartree;
    out.v_es = es.v_r;
    out.veff_a.resize(ng);
    out.veff_b.resize(ng);
    const bool have_loc = !v_locsr.empty();
    if (have_loc && v_locsr.size() != ng)
        throw std::runtime_error(
            "GPW ks_grid_build_uks: v_locsr size does not match density grid");
    // Fuse V_eff^α/β assembly + optional ∫ρ V_loc (one grid walk).
    double e_loc = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : e_loc) if (ng > 4096)
    for (std::size_t g = 0; g < ng; ++g) {
        const double vloc = have_loc ? v_locsr[g] : 0.0;
        if (have_loc)
            e_loc += rho[g] * vloc;
        const double ves = out.v_es[g] + vloc;
        out.veff_a[g] = ves + xcg.vxc_a[g];
        out.veff_b[g] = ves + xcg.vxc_b[g];
    }
    out.e_locgrid = e_loc * grid.weight();

    if (vdw_df || vdw_df2 || rvv10) {
        NonlocalVdwOnGrid nl =
            nonlocal_vdw_on_grid(grid, rho, vdw_df, vdw_df2, rvv10,
                                 /*potential=*/true);
        out.e_nl_vdw = nl.energy;
#pragma omp parallel for schedule(static) if (ng > 4096)
        for (std::size_t g = 0; g < ng; ++g) {
            out.veff_a[g] += nl.v_r[g];
            out.veff_b[g] += nl.v_r[g];
        }
    }
    return out;
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
