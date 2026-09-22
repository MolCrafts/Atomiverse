/// @file regressions/gpw/gpw-mbd-forces/reg_gpw-mbd-forces.cpp
/// @brief Water-dimer free-atom MBD@rsSCS E_disp + force dual-gate.
///
/// Out-of-library consumer: public GPW headers only. Geometry and pins in
/// REFERENCE.md (libMBD offline; no live oracle in ctest).

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::mbd_rsscs_pbe;
    using cpu::gpw::MbdResult;
    using cpu::gpw::XcKind;

    constexpr double kA2B = 1.0 / 0.52917720859;
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    const int Zarr[6] = {8, 1, 1, 8, 1, 1};
    std::vector<int> Z(Zarr, Zarr + 6);
    std::vector<std::array<double, 3>> pos(6);
    for (int i = 0; i < 6; ++i)
        for (int d = 0; d < 3; ++d)
            pos[static_cast<std::size_t>(i)][d] = ang[i][d] * kA2B;

    // Offline pin (REFERENCE.md): free-atom MBD@rsSCS, isolated.
    constexpr double kEDispRef = -1.4928804171407961e-03; // Ha
    constexpr double kETol = 1.0e-8;                      // Ha
    constexpr double kFTol = 1.0e-6;                      // Ha/Bohr
    constexpr double kSumFTol = 1.0e-8;                   // Ha/Bohr

    const double cell0[3] = {0.0, 0.0, 0.0};
    const MbdResult mbd = mbd_rsscs_pbe(Z, pos, cell0, /*periodic=*/false);
    if (!std::isfinite(mbd.energy)
        || std::abs(mbd.energy - kEDispRef) > kETol) {
        std::fprintf(stderr,
                     "FAIL: E_disp MBD got=%.14e want=%.14e tol=%.1e Ha\n",
                     mbd.energy, kEDispRef, kETol);
        return 1;
    }
    if (mbd.forces.size() != 6) {
        std::fprintf(stderr, "FAIL: forces size %zu (want 6)\n",
                     mbd.forces.size());
        return 1;
    }

    // Dual-gate: independent central FD of energy (h=1e-5) vs product forces.
    const double h = 1.0e-5;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    auto pos_fd = pos;
    for (std::size_t i = 0; i < 6; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos_fd[i][d];
            pos_fd[i][d] = save + h;
            const double ep = mbd_rsscs_pbe(Z, pos_fd, cell0, false).energy;
            pos_fd[i][d] = save - h;
            const double em = mbd_rsscs_pbe(Z, pos_fd, cell0, false).energy;
            pos_fd[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - mbd.forces[i][d]));
            sum[d] += mbd.forces[i][d];
        }
    }
    if (max_err > kFTol) {
        std::fprintf(stderr,
                     "FAIL: max |F_product-F_fd|=%.3e > tol=%.1e Ha/Bohr\n",
                     max_err, kFTol);
        return 1;
    }
    const double sum_norm =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    if (sum_norm > kSumFTol) {
        std::fprintf(stderr, "FAIL: |sum F|=%.3e > tol=%.1e Ha/Bohr\n",
                     sum_norm, kSumFTol);
        return 1;
    }

    // Product path: short GPW SCF with mbd=true reports same E_disp.
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double p[3] = {pos[static_cast<std::size_t>(i)][0],
                             pos[static_cast<std::size_t>(i)][1],
                             pos[static_cast<std::size_t>(i)][2]};
        if (Zarr[i] == 8)
            atoms.emplace_back(p, bO, pO, 8);
        else
            atoms.emplace_back(p, bH, pH, 1);
    }

    GpwDft gpw;
    const double L = 12.0 * kA2B;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.n_occ = 8;
    gpw.max_iter = 8;
    gpw.tol = 1e-6;
    gpw.mix = 0.4;
    gpw.xc = XcKind::PBE;
    gpw.nonperiodic = false;
    gpw.mbd = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);
    if (!std::isfinite(res.e_disp)
        || std::abs(res.e_disp - kEDispRef) > kETol) {
        std::fprintf(stderr, "FAIL: GpwDft e_disp got=%.14e want=%.14e\n",
                     res.e_disp, kEDispRef);
        return 1;
    }
    if (!std::isfinite(res.e_total)) {
        std::fprintf(stderr, "FAIL: e_total non-finite\n");
        return 1;
    }

    std::printf("PASS: MBD@rsSCS free-atom water dimer E_disp=%.14e Ha "
                "(libMBD pin); max|F-F_fd|=%.2e Ha/Bohr; |sum F|=%.2e; "
                "F_O0=(%.6e, %.6e, %.6e) Ha/Bohr; e_total=%.8f Ha (iters=%d)\n",
                mbd.energy, max_err, sum_norm, mbd.forces[0][0],
                mbd.forces[0][1], mbd.forces[0][2], res.e_total, res.iters);
    return 0;
}
