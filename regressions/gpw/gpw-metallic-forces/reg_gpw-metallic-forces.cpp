/// @file regressions/gpw/gpw-metallic-forces/reg_gpw-metallic-forces.cpp
/// @brief END-TO-END: multi-k GPW RKS free-energy analytic forces (FD smear).
///
/// Periodic H₂ on a Monkhorst–Pack 1×1×2 mesh with Fermi–Dirac smearing:
/// analytic free-energy forces match central FD of \(F_{\mathrm{Mermin}}\).
/// Offline CP2K ENERGY_FORCE + SMEAR twin (BCC Na, same algorithm class)
/// documents fair wall. See REFERENCE.md. Public headers only.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

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

// Frozen first-green ATV (Release, OMP=8), 2026-08-01. Decisive: analytic
// vs free-energy FD max|dF| = 7.8e-6 Ha/Bohr (h = 1e-3 Bohr).
constexpr double kEPin = -1.118469277632;    // Ha free energy
constexpr double kETol = 1e-5;               // Ha
constexpr double kF0xPin = -1.4404059091e-2; // Ha/Bohr
constexpr double kF1xPin = 2.1449155103e-1;  // Ha/Bohr
constexpr double kForceTol = 1e-5;           // Ha/Bohr
constexpr double kFdTol = 5e-5;              // Ha/Bohr analytic vs FD

} // namespace

int main() {
    using namespace ATV::cpu::gpw;

    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    // Off-symmetry compressed bond so every Cartesian is nonzero.
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    auto make_cfg = []() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 100.0;
        m.n_occ = 1;
        m.tol = 1e-8;
        m.max_iter = 60;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.xc = XcKind::PADE;
        m.eps_default = 1e-8;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.smearing = Smearing::fermi_dirac;
        m.degauss = 0.02; // Ry
        m.extra_bands = 2;
        return m;
    };

    GpwDft energy = make_cfg();
    energy.build(atoms);

    const auto t0 = std::chrono::steady_clock::now();
    const auto& r = energy.run(atoms);
    const auto t_scf = std::chrono::steady_clock::now();
    const auto Fa = gpw_forces_analytic(atoms, make_cfg());
    const auto t1 = std::chrono::steady_clock::now();
    // Free-energy FD oracle (differentiates e_total = F_Mermin under smear).
    // Full 6N SCF × k-path — decisive identity, still ≪ 60 s on H2.
    const auto Ffd = gpw_forces_fd(atoms, make_cfg(), 1e-3);
    const auto t2 = std::chrono::steady_clock::now();

    const double wall_scf = std::chrono::duration<double>(t_scf - t0).count();
    const double wall_f = std::chrono::duration<double>(t1 - t_scf).count();
    const double wall_fd = std::chrono::duration<double>(t2 - t1).count();

    std::printf(
        "F=%.12f Ha  e_fermi=%.6f  Ne=%.8f  e_ent=%.3e  iters=%d  nk=%d  "
        "f_k=%zu  wall_scf=%.3f s  wall_F=%.3f s  wall_fd=%.3f s\n",
        r.e_total, r.e_fermi, r.n_elec_occ, r.e_entropy, r.iters, r.n_kpts,
        r.f_k.size(), wall_scf, wall_f, wall_fd);
    for (int a = 0; a < 2; ++a)
        std::printf("  Fa[%d]  = (% .10e, % .10e, % .10e)\n", a, Fa[a][0],
                    Fa[a][1], Fa[a][2]);
    for (int a = 0; a < 2; ++a)
        std::printf("  Ffd[%d] = (% .10e, % .10e, % .10e)\n", a, Ffd[a][0],
                    Ffd[a][1], Ffd[a][2]);

    check(r.converged, "R1 SCF converged");
    check(std::isfinite(r.e_total) && std::isfinite(r.e_fermi),
          "finite free energy and e_fermi");
    check(std::fabs(r.n_elec_occ - 2.0) < 1e-5, "R2 electron count Ne=2");
    // k-path product: mesh 1×1×2 (TR may fold to nk=1; path still k-aware).
    check(r.n_kpts >= 1 && !r.f_k.empty(), "k-path + fractional f_k payload");
    check(std::fabs(r.e_total - kEPin) < kETol, "R3 free energy pin");

    check(Fa.size() == 2 && Ffd.size() == 2, "two force vectors");
    double max_err = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            check(std::isfinite(Fa[a][k]) && std::isfinite(Ffd[a][k]),
                  "finite force component");
            max_err = std::max(max_err, std::fabs(Fa[a][k] - Ffd[a][k]));
        }
    }
    std::printf("  max|Fa-Ffd| = %.3e Ha/Bohr\n", max_err);
    check(max_err < kFdTol, "R4 analytic free-energy force vs FD");

    check(Fa[0][0] < 0.0 && Fa[1][0] > 0.0, "restoring F_x signs");
    check(std::fabs(Fa[0][0] - kF0xPin) < kForceTol, "R5 F0_x pin");
    check(std::fabs(Fa[1][0] - kF1xPin) < kForceTol, "R5 F1_x pin");

    if (g_fail) {
        std::printf("FAIL: gpw-metallic-forces (%d checks)\n", g_fail);
        return 1;
    }
    std::printf(
        "PASS: gpw-metallic-forces (H2 multi-k FD free-energy forces)\n");
    return 0;
}
