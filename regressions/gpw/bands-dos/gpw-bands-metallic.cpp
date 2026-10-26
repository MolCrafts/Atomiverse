/// @file regressions/gpw/gpw-bands-metallic/reg_gpw-bands-metallic.cpp
/// @brief END-TO-END: multi-k GPW band structure under Fermi–Dirac smear (BCC Na).
///
/// SCF with SMEAR → freeze V_eff → NSCF path. Gates free energy / E_F character
/// vs offline CP2K 2026.1 SMEAR + BAND_STRUCTURE twin. See REFERENCE.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/bands_dos.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molpy.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_bands_metallic {

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

/// BCC Na conventional cell (2 atoms), lattice constant `a_ang` [Å].
ATV::cpu::Frame na_bcc(double a_ang) {
    using namespace ATV;
    const double frac[2][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 11;
        frame.span(kAtom_x)[i] = frac[i][0] * a_ang;
        frame.span(kAtom_y)[i] = frac[i][1] * a_ang;
        frame.span(kAtom_z)[i] = frac[i][2] * a_ang;
    }
    frame.box = cpu::Box::cubic(a_ang);
    return frame;
}

} // namespace

int run() {
    using namespace ATV::cpu::gpw;

    // Same electronic class as gpw-metallic-smearing / dos-metallic.
    constexpr double kFPinHa = -92.126330608378;
    constexpr double kFTolHa = 1e-3;
    // CP2K E_F class (dos-metallic twin); soft absolute window.
    constexpr double kEFermiCp2kHa = 0.05447820003643;
    constexpr double kEFermiTolHa = 0.05;
    constexpr double kNelec = 18.0;

    ATV::cpu::Frame frame = na_bcc(4.225);

    GpwDft method{{Kind{molrs::Element::Na,
                        GaussianBasis(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH"),
                        GthPotential(molrs::Element::Na, "GTH-PBE-q9")}},
                  150.0,
                  XcKind::PADE};
    method.n_occ = 9;
    method.max_iter = 80;
    method.tol = 1e-6;
    method.eps_default = 1e-7;
    method.diis_depth = 6;
    method.mix = 0.4;
    method.k_mesh[0] = method.k_mesh[1] = method.k_mesh[2] = 2;
    method.k_time_reversal = true;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    method.extra_bands = 4;
    method.local_pp_on_grid = true;
    method.build(frame);

    Bands bands;
    bands.n_bands = 16; // n_occ seed + ADDED_MOS-class empty
    // Fractional SPECIAL path (same list as diamond / CP2K twin points).
    bands.set_diamond_path(3);

    const auto t0 = std::chrono::steady_clock::now();
    const BandsResult br = bands.compute(method, frame);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    std::printf("gpw-bands-metallic: F=%.10f Ha  E_F=%.6f Ha  wall=%.3f s  "
                "nk=%zu  nspin=%d  conv=%d\n",
                br.energy_ha, br.e_fermi_ha, wall_s, br.kfrac.size(), br.nspin,
                br.scf_converged ? 1 : 0);

    check(br.scf_converged, "SCF converged");
    check(br.nspin == 1, "nspin=1 (RKS metallic)");
    check(std::isfinite(br.energy_ha) && std::isfinite(br.e_fermi_ha),
          "finite F and E_F");
    check(std::abs(br.energy_ha - kFPinHa) <= kFTolHa, "free energy pin");
    check(std::abs(br.e_fermi_ha - kEFermiCp2kHa) <= kEFermiTolHa,
          "E_F vs CP2K class");
    check(!br.eigenvalues_ha.empty(), "path eigenvalues present");
    check(br.kfrac.size() == br.eigenvalues_ha.size(), "kfrac/evals match");
    check(br.eigenvalues_beta_ha.empty(), "no β channel on RKS");

    // Eigenvalues sorted; spectrum spans both sides of E_F (metallic).
    int n_below = 0, n_above = 0;
    for (const auto& row : br.eigenvalues_ha) {
        check(static_cast<int>(row.size()) == bands.n_bands, "n_bands per k");
        for (std::size_t b = 1; b < row.size(); ++b)
            check(row[b] + 1e-10 >= row[b - 1], "evals sorted");
        for (double e : row) {
            if (e < br.e_fermi_ha)
                ++n_below;
            else
                ++n_above;
        }
    }
    check(n_below > 0 && n_above > 0, "path spectrum straddles E_F");
    std::printf("  n_below_EF=%d  n_above_EF=%d  (expect metallic)\n", n_below,
                n_above);
    (void)kNelec;

    std::printf("ATV_WALL_S %.6f\n", wall_s);

    if (g_fail) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("ALL CHECKS PASSED\n");
    return 0;
}

} // namespace atv_reg_gpw_gpw_bands_metallic

ATV_REGRESSION("gpw/bands-dos", "gpw-bands-metallic",
               atv_reg_gpw_gpw_bands_metallic::run);
