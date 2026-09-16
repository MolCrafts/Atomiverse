/// @file regressions/gpw/gpw-vdw-scf-potential/reg_gpw-vdw-scf-potential.cpp
/// @brief Γ GPW PBE + rVV10 self-consistent E_disp + analytic force FD gate.
///
/// Out-of-library consumer. SC nonlocal V_nl in SCF + forces (REFERENCE.md).
/// Offline CP2K 2026.1 TYPE RVV10 pin; no live oracle in ctest.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859;

// Same water dimer as gpw-rvv10-ice / gpw-vdw-df-ice [Å].
const double kAng[6][3] = {
    {0.000000, 0.000000, 0.000000},  // O
    {0.758602, 0.000000, 0.504284},  // H
    {0.758602, 0.000000, -0.504284}, // H
    {3.000000, 0.000000, 0.000000},  // O
    {3.758602, 0.000000, 0.504284},  // H
    {3.758602, 0.000000, -0.504284}, // H
};
const int kZ[6] = {8, 1, 1, 8, 1, 1};

std::vector<ATV::cpu::gpw::GpwAtom> make_atoms(const double ang[6][3]) {
    using namespace ATV::cpu::gpw;
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {ang[i][0] * kA2B, ang[i][1] * kA2B,
                               ang[i][2] * kA2B};
        if (kZ[i] == 8)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }
    return atoms;
}

ATV::cpu::gpw::GpwDft make_gpw() {
    ATV::cpu::gpw::GpwDft gpw;
    const double L = 12.0 * kA2B;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.n_occ = 8;
    gpw.max_iter = 40;
    gpw.tol = 1e-7;
    gpw.mix = 0.4;
    gpw.xc = ATV::cpu::gpw::XcKind::PBE;
    gpw.nonperiodic = false;
    gpw.rvv10 = true;
    return gpw;
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;

    // Offline CP2K 2026.1 SC Dispersion (TYPE RVV10) — see REFERENCE.md.
    // Post-SCF ATV was 6.11535e-2; SC should land nearer CP2K 6.11647e-2.
    constexpr double kEDispCp2k = 6.116467960721e-02; // Ha
    constexpr double kEDispTol = 1.0e-4;              // Ha

    auto atoms = make_atoms(kAng);
    GpwDft gpw = make_gpw();
    gpw.build(atoms);
    const GpwResult res = gpw.run(atoms);

    if (!res.converged) {
        std::fprintf(stderr, "FAIL: SCF did not converge (iters=%d)\n",
                     res.iters);
        return 1;
    }
    if (!std::isfinite(res.e_disp)
        || std::abs(res.e_disp - kEDispCp2k) > kEDispTol) {
        std::fprintf(stderr,
                     "FAIL: SC E_disp rVV10 got=%.14e want=%.14e ±%.1e Ha\n",
                     res.e_disp, kEDispCp2k, kEDispTol);
        return 1;
    }

    // Analytic forces at SC density.
    const auto f_ana = gpw_forces_from_scf(atoms, gpw, res);
    // Central FD of E_total w.r.t. O0 x (H-bond axis) — non-trivial component.
    // h in Bohr.
    constexpr double h = 5.0e-4;
    double ang_p[6][3], ang_m[6][3];
    for (int i = 0; i < 6; ++i)
        for (int k = 0; k < 3; ++k) {
            ang_p[i][k] = kAng[i][k];
            ang_m[i][k] = kAng[i][k];
        }
    ang_p[0][0] += h / kA2B; // Å
    ang_m[0][0] -= h / kA2B;

    auto atoms_p = make_atoms(ang_p);
    auto atoms_m = make_atoms(ang_m);
    GpwDft gpw_p = make_gpw();
    GpwDft gpw_m = make_gpw();
    gpw_p.build(atoms_p);
    gpw_m.build(atoms_m);
    const GpwResult rp = gpw_p.run(atoms_p);
    const GpwResult rm = gpw_m.run(atoms_m);
    if (!rp.converged || !rm.converged) {
        std::fprintf(stderr, "FAIL: FD SCF did not converge\n");
        return 1;
    }
    // Force = −dE/dR; E in Ha, R in Bohr.
    const double f_fd_x = -(rp.e_total - rm.e_total) / (2.0 * h);
    const double f_ana_x = f_ana[0][0];
    const double df = std::abs(f_fd_x - f_ana_x);
    constexpr double kForceTol = 5.0e-4; // Ha/Bohr

    std::printf("SC E_disp=%.14e Ha (CP2K %.14e ± %.1e); "
                "F_O0_x ana=%.6e FD=%.6e |Δ|=%.3e Ha/Bohr\n",
                res.e_disp, kEDispCp2k, kEDispTol, f_ana_x, f_fd_x, df);

    if (!(std::abs(f_ana_x) > 1.0e-5)) {
        std::fprintf(stderr,
                     "FAIL: expected non-trivial F_O0_x, got ana=%.6e\n",
                     f_ana_x);
        return 1;
    }
    if (!std::isfinite(f_ana_x) || !std::isfinite(f_fd_x) || df > kForceTol) {
        std::fprintf(stderr,
                     "FAIL: force FD gate |F_ana - F_FD|=%.3e > %.1e Ha/Bohr "
                     "(ana=%.6e FD=%.6e)\n",
                     df, kForceTol, f_ana_x, f_fd_x);
        return 1;
    }

    std::printf("PASS: gpw-vdw-scf-potential rVV10 SC E_disp + force FD\n");
    return 0;
}
