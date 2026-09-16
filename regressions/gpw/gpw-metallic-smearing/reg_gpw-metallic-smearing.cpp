/// @file regressions/gpw/gpw-metallic-smearing/reg_gpw-metallic-smearing.cpp
/// @brief END-TO-END: multi-k GPW RKS with Fermi–Dirac metallic smearing (BCC Na).
///
/// See REFERENCE.md. Public headers only. Requires ATV_BUILD_GPW.
/// Pins: electron count, finite Fermi level, free energy vs offline freeze.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859; // Å → Bohr

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

/// BCC conventional cubic: 2 Na at (0,0,0) and (½,½,½).
std::vector<ATV::cpu::gpw::GpwAtom> na_bcc(double a_bohr) {
    using namespace ATV;
    const cpu::gpw::GaussianBasis bNa(molrs::Element::Na,
                                      "DZVP-MOLOPT-SR-GTH");
    const cpu::gpw::GthPotential pNa(molrs::Element::Na, "GTH-PBE-q9");
    const double frac[2][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
    std::vector<cpu::gpw::GpwAtom> atoms;
    for (int i = 0; i < 2; ++i) {
        const double pos[3] = {frac[i][0] * a_bohr, frac[i][1] * a_bohr,
                               frac[i][2] * a_bohr};
        atoms.emplace_back(pos, bNa, pNa, 11);
    }
    return atoms;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GpwDft;
    using cpu::gpw::Smearing;
    using cpu::gpw::XcKind;

    // Offline free-energy pin — frozen first green (REFERENCE.md).
    const double f_pin_ha = -92.126330608378; // Ha free energy F
    const double f_tol_ha = 1e-4;             // Ha
    const double ne_tol = 1e-5;
    const double a_ang = 4.225;
    const double a = a_ang * kA2B;
    const double n_elec = 18.0; // 2 × GTH-PBE-q9

    auto atoms = na_bcc(a);

    GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = a;
    m.cutoff_ry = 150.0;
    m.n_occ = 9; // Ne/2
    m.max_iter = 80;
    m.tol = 1e-6;
    m.eps_default = 1e-7;
    m.xc = XcKind::PADE;
    m.diis_depth = 6;
    m.mix = 0.4;
    m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
    m.k_time_reversal = true;
    m.smearing = Smearing::fermi_dirac;
    m.degauss = 0.02; // Ry
    m.extra_bands = 4;

    m.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const auto& r = m.run(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s =
        std::chrono::duration<double>(t1 - t0).count();

    std::printf(
        "F=%.10f Ha  e_fermi=%.6f Ha  n_elec=%.8f  e_entropy=%.6e Ha  "
        "iters=%d  wall=%.3f s  converged=%d\n",
        r.e_total, r.e_fermi, r.n_elec_occ, r.e_entropy, r.iters, wall_s,
        r.converged ? 1 : 0);

    check(r.converged, "R1 SCF converged");
    check(std::isfinite(r.e_total) && std::isfinite(r.e_fermi),
          "finite free energy and e_fermi");
    check(std::fabs(r.n_elec_occ - n_elec) < ne_tol, "R2 electron count");
    check(std::fabs(r.e_fermi) < 1e3, "R3 Fermi level finite");
    check(std::fabs(r.e_total - f_pin_ha) < f_tol_ha, "R4 free energy pin");
    // Entropy term should be finite; for a metal it is typically non-zero.
    check(std::isfinite(r.e_entropy), "finite e_entropy");
    check(r.n_kpts > 1, "multi-k sampling active");

    if (g_fail) {
        std::printf("FAIL: gpw-metallic-smearing (%d checks)\n", g_fail);
        return 1;
    }
    std::printf("PASS: gpw-metallic-smearing (BCC Na FD smear)\n");
    return 0;
}
