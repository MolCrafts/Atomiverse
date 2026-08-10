/// @file regressions/architecture/arch-mg-main-structural/reg_arch-mg-main-structural.cpp
/// @brief Architecture gate: multigrid MAIN product path is a single composition.
///
/// Public + internal multigrid headers. Confirms the multi-level collocate /
/// integrate helpers compile and run, and that public `GpwDft` still exposes
/// `ngrids` / `rel_cutoff` for the REL_CUTOFF product path. Full energy pin
/// lives in twin `gpw-multigrid-kpoints` — this case is shape-only (no multi-iter
/// SCF).

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"

namespace {

ATV::cpu::gpw::GridBasisFunction s_gauss(double alpha, double cx, double cy,
                                         double cz) {
    ATV::cpu::gpw::GridPrimitive p;
    p.alpha = alpha;
    p.center[0] = cx;
    p.center[1] = cy;
    p.center[2] = cz;
    p.coeff = 1.0;
    return {p};
}

} // namespace

int main() {
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::build_products;
    using ATV::cpu::gpw::make_multigrid;
    using ATV::cpu::gpw::multigrid_collocate_rho;
    using ATV::cpu::gpw::multigrid_integrate_veff;

    // Public surface: multigrid members remain first-class on GpwDft.
    GpwDft method;
    method.ngrids = 4;
    method.rel_cutoff = 60.0;
    method.cutoff_ry = 150.0;
    if (method.ngrids != 4 || !(method.rel_cutoff > 0.0)) {
        std::fprintf(stderr, "FAIL: GpwDft ngrids/rel_cutoff surface missing\n");
        return 1;
    }

    // Tiny two-level ladder: both levels get products; helpers must return
    // finite fine-grid density and KS contribution without a full SCF.
    const double cell[3] = {6.0, 6.0, 6.0};
    auto mg = make_multigrid(cell, 60.0, 2, 12.0);
    if (mg.levels.size() != 2) {
        std::fprintf(stderr, "FAIL: expected 2 multigrid levels\n");
        return 1;
    }
    std::vector<ATV::cpu::gpw::GridBasisFunction> basis = {
        s_gauss(0.35, 1.5, 2.4, 1.8),
        s_gauss(0.80, 4.2, 3.3, 3.9),
    };
    const int nb = static_cast<int>(basis.size());
    const auto products =
        build_products(mg, basis, cell, 12.0, 1e-12, /*periodic=*/true);
    if (products.by_level.size() != 2 || products.by_level[0].empty()
        || products.by_level[1].empty()) {
        std::fprintf(stderr, "FAIL: product levels not both populated\n");
        return 1;
    }

    std::vector<double> P(static_cast<std::size_t>(nb) * nb, 0.0);
    P[0] = 1.0;
    P[static_cast<std::size_t>(nb) + 1] = 0.5;
    P[1] = P[static_cast<std::size_t>(nb)] = 0.1;

    const auto rho = multigrid_collocate_rho(mg, products, nb, P, 1e-12);
    if (rho.size() != static_cast<std::size_t>(mg.levels[0].npts())) {
        std::fprintf(stderr, "FAIL: multigrid_collocate_rho size mismatch\n");
        return 1;
    }
    double q = 0.0;
    for (double v : rho) {
        if (!std::isfinite(v)) {
            std::fprintf(stderr, "FAIL: non-finite density\n");
            return 1;
        }
        q += v;
    }
    q *= mg.levels[0].weight();
    if (!(q > 0.0)) {
        std::fprintf(stderr, "FAIL: fine density integral not positive (q=%g)\n",
                     q);
        return 1;
    }

    std::vector<double> veff(rho.size(), -0.1);
    const auto Vks = multigrid_integrate_veff(mg, products, nb, veff, 1e-12);
    if (Vks.size() != static_cast<std::size_t>(nb) * nb) {
        std::fprintf(stderr, "FAIL: multigrid_integrate_veff size mismatch\n");
        return 1;
    }
    for (double v : Vks) {
        if (!std::isfinite(v)) {
            std::fprintf(stderr, "FAIL: non-finite V_ks\n");
            return 1;
        }
    }

    std::printf(
        "PASS arch-mg-main-structural: MAIN multi-level product path helpers\n");
    return 0;
}
