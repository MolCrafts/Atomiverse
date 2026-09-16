/// @file regressions/gpw/gpw-hirshfeld-kpoint-forces/reg_gpw-hirshfeld-kpoint-forces.cpp
/// @brief Multi-k GPW forces fold Hirshfeld-scaled TS/MBD volumes.
///
/// Out-of-library consumer: public GPW headers only. Offline pins in
/// REFERENCE.md (libMBD / product dual; no live oracle in ctest).
/// CP2K 2026.1 has no stock TS/MBD — same policy as gpw-hirshfeld-forces.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;
    using clock = std::chrono::steady_clock;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::gpw_forces_from_scf;
    using cpu::gpw::mbd_rsscs_pbe;
    using cpu::gpw::MbdResult;
    using cpu::gpw::ts_pbe;
    using cpu::gpw::TsResult;
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

    // Same dual volume table as reg_gpw-hirshfeld-forces / mbd-hirshfeld.
    const std::vector<double> kVol = {0.7403, 0.6529, 0.6529,
                                      0.7153, 0.5784, 0.5784};

    constexpr double kEMbdHirsh = -1.1479717148086300e-03; // Ha
    constexpr double kETol = 1.0e-8;                       // Ha
    constexpr double kFTol = 1.0e-6;                       // Ha/Bohr
    constexpr double kSumFTol = 1.0e-8;                    // Ha/Bohr
    constexpr double kWiringTol = 1.0e-7;                  // Ha/Bohr
    constexpr double kWallBudget = 60.0;                   // s

    const auto t0 = clock::now();
    int nfail = 0;
    auto fail = [&](const char* msg) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++nfail;
    };

    // ── A/B: isolated volume-scaled MBD energy + force dual-gate ──────────
    const double cell0[3] = {0.0, 0.0, 0.0};
    const MbdResult mbd =
        mbd_rsscs_pbe(Z, pos, cell0, /*periodic=*/false, 0.83, 6.0, kVol);
    if (!std::isfinite(mbd.energy)
        || std::abs(mbd.energy - kEMbdHirsh) > kETol) {
        std::fprintf(stderr,
                     "FAIL: E_disp MBD+Hirsh volumes got=%.14e want=%.14e\n",
                     mbd.energy, kEMbdHirsh);
        ++nfail;
    }
    if (mbd.forces.size() != 6) {
        fail("MBD forces size != 6");
    } else {
        const double h = 1.0e-5;
        double max_err = 0.0;
        double sum[3] = {0.0, 0.0, 0.0};
        auto pos_fd = pos;
        for (std::size_t i = 0; i < 6; ++i) {
            for (int d = 0; d < 3; ++d) {
                const double save = pos_fd[i][d];
                pos_fd[i][d] = save + h;
                const double ep =
                    mbd_rsscs_pbe(Z, pos_fd, cell0, false, 0.83, 6.0, kVol)
                        .energy;
                pos_fd[i][d] = save - h;
                const double em =
                    mbd_rsscs_pbe(Z, pos_fd, cell0, false, 0.83, 6.0, kVol)
                        .energy;
                pos_fd[i][d] = save;
                const double fd = -(ep - em) / (2.0 * h);
                max_err =
                    std::max(max_err, std::abs(fd - mbd.forces[i][d]));
                sum[d] += mbd.forces[i][d];
            }
        }
        const double sum_norm =
            std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
        if (max_err > kFTol) {
            std::fprintf(stderr,
                         "FAIL: MBD max|F-F_fd|=%.3e > tol=%.1e Ha/Bohr\n",
                         max_err, kFTol);
            ++nfail;
        }
        if (sum_norm > kSumFTol) {
            std::fprintf(stderr, "FAIL: MBD |sum F|=%.3e > tol\n", sum_norm);
            ++nfail;
        }
        std::printf("  MBD+Hirsh isolated: E=%.14e  max|F-F_fd|=%.2e  "
                    "|sum F|=%.2e\n",
                    mbd.energy, max_err, sum_norm);
    }

    // ── C: isolated volume-scaled TS force dual-gate ──────────────────────
    const TsResult ts =
        ts_pbe(Z, pos, cell0, /*periodic=*/false, 94.2, 0.94, 20.0, kVol);
    if (!std::isfinite(ts.energy) || !(ts.energy < 0.0)) {
        fail("TS Hirsh energy non-finite or non-attractive");
    } else {
        const double h = 1.0e-5;
        double max_err = 0.0;
        auto pos_fd = pos;
        for (std::size_t i = 0; i < 6; ++i) {
            for (int d = 0; d < 3; ++d) {
                const double save = pos_fd[i][d];
                pos_fd[i][d] = save + h;
                const double ep =
                    ts_pbe(Z, pos_fd, cell0, false, 94.2, 0.94, 20.0, kVol)
                        .energy;
                pos_fd[i][d] = save - h;
                const double em =
                    ts_pbe(Z, pos_fd, cell0, false, 94.2, 0.94, 20.0, kVol)
                        .energy;
                pos_fd[i][d] = save;
                const double fd = -(ep - em) / (2.0 * h);
                max_err = std::max(max_err, std::abs(fd - ts.forces[i][d]));
            }
        }
        if (max_err > kFTol) {
            std::fprintf(stderr,
                         "FAIL: TS max|F-F_fd|=%.3e > tol=%.1e Ha/Bohr\n",
                         max_err, kFTol);
            ++nfail;
        }
        const TsResult ts_free =
            ts_pbe(Z, pos, cell0, /*periodic=*/false, 94.2, 0.94, 20.0);
        double max_df = 0.0;
        for (std::size_t i = 0; i < 6; ++i)
            for (int d = 0; d < 3; ++d)
                max_df = std::max(
                    max_df, std::abs(ts.forces[i][d] - ts_free.forces[i][d]));
        if (max_df < 1e-8) {
            std::fprintf(stderr,
                         "FAIL: Hirshfeld volumes must move TS forces "
                         "(max|dF|=%.3e)\n",
                         max_df);
            ++nfail;
        }
        std::printf("  TS+Hirsh isolated: E=%.14e  max|F-F_fd|=%.2e  "
                    "max|F_h-F_free|=%.2e\n",
                    ts.energy, max_err, max_df);
    }

    // ── D/E: multi-k product wiring on H₂ (1×1×2) ─────────────────────────
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};
    std::vector<int> Zh = {1, 1};
    std::vector<std::array<double, 3>> posh = {
        {rA[0], rA[1], rA[2]}, {rB[0], rB[1], rB[2]}};
    const double cell_h2[3] = {12.0, 12.0, 12.0};
    constexpr double kRcutTs = 50.0 * kA2B;

    auto make_cfg = [](bool with_ts, bool with_mbd, bool with_hirsh) {
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
        m.ts = with_ts;
        m.mbd = with_mbd;
        m.hirshfeld = with_hirsh;
        return m;
    };

    // D: multi-k TS + hirshfeld
    {
        GpwDft cfg = make_cfg(/*ts=*/true, /*mbd=*/false, /*hirsh=*/true);
        cfg.build(atoms);
        const auto& scf = cfg.run(atoms);
        if (!scf.converged) {
            fail("multi-k TS+hirsh SCF did not converge");
        } else if (scf.n_kpts < 1 || scf.C_occ_k.empty()) {
            fail("multi-k TS+hirsh missing k-path payload");
        } else if (scf.volume_ratios.size() != 2) {
            fail("multi-k TS+hirsh volume_ratios size != 2");
        } else if (!std::isfinite(scf.e_disp) || !(scf.e_disp < 0.0)) {
            fail("multi-k TS+hirsh e_disp non-finite or non-attractive");
        } else {
            const auto F_h = gpw_forces_from_scf(atoms, cfg, scf);
            GpwDft plain = make_cfg(false, false, false);
            const auto F_plain = gpw_forces_from_scf(atoms, plain, scf);
            const TsResult geom_h =
                ts_pbe(Zh, posh, cell_h2, /*periodic=*/true, kRcutTs, 0.94,
                       20.0, scf.volume_ratios);
            const TsResult geom_free =
                ts_pbe(Zh, posh, cell_h2, /*periodic=*/true, kRcutTs);
            if (std::abs(scf.e_disp - geom_h.energy) > kETol) {
                std::fprintf(stderr,
                             "FAIL: multi-k TS e_disp=%.14e vs geom_h=%.14e\n",
                             scf.e_disp, geom_h.energy);
                ++nfail;
            }
            double max_wire = 0.0;
            double max_wire_free = 0.0;
            for (int a = 0; a < 2; ++a) {
                for (int d = 0; d < 3; ++d) {
                    const double dF =
                        F_h[static_cast<std::size_t>(a)][d]
                        - F_plain[static_cast<std::size_t>(a)][d];
                    max_wire = std::max(
                        max_wire,
                        std::abs(dF
                                 - geom_h.forces[static_cast<std::size_t>(a)]
                                               [d]));
                    max_wire_free = std::max(
                        max_wire_free,
                        std::abs(dF
                                 - geom_free.forces[static_cast<std::size_t>(a)]
                                                   [d]));
                }
            }
            if (max_wire > kWiringTol) {
                std::fprintf(stderr,
                             "FAIL: multi-k TS+Hirsh wiring max|d|=%.3e "
                             "> tol=%.1e\n",
                             max_wire, kWiringTol);
                ++nfail;
            }
            if (max_wire_free < 1e-9) {
                fail("multi-k TS forces still match free-atom (hirsh not "
                     "applied)");
            }
            std::printf(
                "  multi-k TS+Hirsh H2: e_disp=%.14e  n_kpts=%d  "
                "max|wire_h|=%.2e  max|wire_free|=%.2e  v0=%.4f\n",
                scf.e_disp, scf.n_kpts, max_wire, max_wire_free,
                scf.volume_ratios[0]);
        }
    }

    // E: multi-k MBD + hirshfeld (isolated MBD when mbd_k_mesh default 0)
    {
        GpwDft cfg = make_cfg(/*ts=*/false, /*mbd=*/true, /*hirsh=*/true);
        cfg.build(atoms);
        const auto& scf = cfg.run(atoms);
        if (!scf.converged) {
            fail("multi-k MBD+hirsh SCF did not converge");
        } else if (scf.n_kpts < 1 || scf.C_occ_k.empty()) {
            fail("multi-k MBD+hirsh missing k-path payload");
        } else if (scf.volume_ratios.size() != 2) {
            fail("multi-k MBD+hirsh volume_ratios size != 2");
        } else if (!std::isfinite(scf.e_disp) || !(scf.e_disp < 0.0)) {
            fail("multi-k MBD+hirsh e_disp non-finite or non-attractive");
        } else {
            const auto F_h = gpw_forces_from_scf(atoms, cfg, scf);
            GpwDft plain = make_cfg(false, false, false);
            const auto F_plain = gpw_forces_from_scf(atoms, plain, scf);
            const MbdResult geom_h =
                mbd_rsscs_pbe(Zh, posh, cell0, /*periodic=*/false, 0.83, 6.0,
                              scf.volume_ratios);
            const MbdResult geom_free =
                mbd_rsscs_pbe(Zh, posh, cell0, /*periodic=*/false);
            if (std::abs(scf.e_disp - geom_h.energy) > kETol) {
                std::fprintf(stderr,
                             "FAIL: multi-k MBD e_disp=%.14e vs geom_h=%.14e\n",
                             scf.e_disp, geom_h.energy);
                ++nfail;
            }
            double max_wire = 0.0;
            double max_wire_free = 0.0;
            for (int a = 0; a < 2; ++a) {
                for (int d = 0; d < 3; ++d) {
                    const double dF =
                        F_h[static_cast<std::size_t>(a)][d]
                        - F_plain[static_cast<std::size_t>(a)][d];
                    max_wire = std::max(
                        max_wire,
                        std::abs(dF
                                 - geom_h.forces[static_cast<std::size_t>(a)]
                                               [d]));
                    max_wire_free = std::max(
                        max_wire_free,
                        std::abs(dF
                                 - geom_free.forces[static_cast<std::size_t>(a)]
                                                   [d]));
                }
            }
            if (max_wire > kWiringTol) {
                std::fprintf(stderr,
                             "FAIL: multi-k MBD+Hirsh wiring max|d|=%.3e "
                             "> tol=%.1e\n",
                             max_wire, kWiringTol);
                ++nfail;
            }
            if (max_wire_free < 1e-9) {
                fail("multi-k MBD forces still match free-atom (hirsh not "
                     "applied)");
            }
            std::printf(
                "  multi-k MBD+Hirsh H2: e_disp=%.14e  n_kpts=%d  "
                "max|wire_h|=%.2e  max|wire_free|=%.2e  v0=%.4f\n",
                scf.e_disp, scf.n_kpts, max_wire, max_wire_free,
                scf.volume_ratios[0]);
        }
    }

    const double wall =
        std::chrono::duration<double>(clock::now() - t0).count();
    if (wall > kWallBudget) {
        std::fprintf(stderr, "FAIL: wall %.2f s > budget %.1f s\n", wall,
                     kWallBudget);
        ++nfail;
    }

    if (nfail != 0) {
        std::fprintf(stderr, "FAIL: %d gate(s) failed (wall=%.2f s)\n", nfail,
                     wall);
        return 1;
    }
    std::printf("PASS: gpw-hirshfeld-kpoint-forces (MBD/TS fixed-v dual + "
                "multi-k H2 TS/MBD+Hirsh wiring; wall=%.2f s)\n",
                wall);
    return 0;
}
