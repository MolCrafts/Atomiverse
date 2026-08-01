/// @file regressions/gpw/gpw-hirshfeld-kpoint-stress/reg_gpw-hirshfeld-kpoint-stress.cpp
/// @brief Multi-k GPW analytic cell stress folds Hirshfeld-scaled TS/MBD.
///
/// Out-of-library consumer: public GPW headers only. Offline pins in
/// REFERENCE.md (libMBD / product dual; no live oracle in ctest).
/// CP2K 2026.1 has no stock TS/MBD — same policy as gpw-hirshfeld-kpoint-forces.
///
/// Product definition (density-chain ∂v/∂ε out of scope):
///   σ_aa = −Ω⁻¹ ∂E_L[{P_k},{W_k};ε]/∂ε_aa |_{v fixed at multi-k SCF}.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859;
constexpr double kStrainH = 2.0e-3;
constexpr double kEMbdHirsh = -1.1479717148086300e-03; // Ha
constexpr double kETol = 1.0e-8;                       // Ha
constexpr double kSigmaDualTol = 1.0e-12;              // Ha/Bohr³ (geom dual)
constexpr double kWireTol = 1.0e-8;                    // Ha/Bohr³
constexpr double kWallBudget = 60.0;                   // s

/// Geometry-only Voigt diagonal stress from central FD of E_disp under
/// orthorhombic strain at fixed fractional coordinates and fixed volumes.
template <typename EvalFn>
std::array<double, 3> sigma_disp_fd(const EvalFn& eval_energy,
                                    const std::vector<std::array<double, 3>>& pos0,
                                    const double cell0[3], double h) {
    const double vol = cell0[0] * cell0[1] * cell0[2];
    std::array<double, 3> sig{{0.0, 0.0, 0.0}};
    for (int a = 0; a < 3; ++a) {
        const double sp = 1.0 + h;
        const double sm = 1.0 - h;
        std::vector<std::array<double, 3>> pos_p = pos0;
        std::vector<std::array<double, 3>> pos_m = pos0;
        double cell_p[3] = {cell0[0], cell0[1], cell0[2]};
        double cell_m[3] = {cell0[0], cell0[1], cell0[2]};
        cell_p[a] *= sp;
        cell_m[a] *= sm;
        for (auto& p : pos_p)
            p[static_cast<std::size_t>(a)] *= sp;
        for (auto& p : pos_m)
            p[static_cast<std::size_t>(a)] *= sm;
        const double ep = eval_energy(pos_p, cell_p);
        const double em = eval_energy(pos_m, cell_m);
        sig[static_cast<std::size_t>(a)] = -(ep - em) / (2.0 * h) / vol;
    }
    return sig;
}

} // namespace

int main() {
    using namespace ATV;
    using clock = std::chrono::steady_clock;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::gpw_stress;
    using cpu::gpw::mbd_rsscs_pbe;
    using cpu::gpw::MbdResult;
    using cpu::gpw::ts_pbe;
    using cpu::gpw::TsResult;
    using cpu::gpw::XcKind;

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

    // Same dual volume table as reg_gpw-hirshfeld-forces / kpoint-forces.
    const std::vector<double> kVol = {0.7403, 0.6529, 0.6529,
                                      0.7153, 0.5784, 0.5784};

    const auto t0 = clock::now();
    int nfail = 0;
    auto fail = [&](const char* msg) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++nfail;
    };

    // ── A: isolated volume-scaled MBD energy pin ──────────────────────────
    const double cell0_iso[3] = {0.0, 0.0, 0.0};
    const MbdResult mbd =
        mbd_rsscs_pbe(Z, pos, cell0_iso, /*periodic=*/false, 0.83, 6.0, kVol);
    if (!std::isfinite(mbd.energy)
        || std::abs(mbd.energy - kEMbdHirsh) > kETol) {
        std::fprintf(stderr,
                     "FAIL: E_disp MBD+Hirsh volumes got=%.14e want=%.14e\n",
                     mbd.energy, kEMbdHirsh);
        ++nfail;
    } else {
        std::printf("  A MBD+Hirsh isolated: E=%.14e Ha\n", mbd.energy);
    }

    // ── B: geometry stress dual under strain (fixed dual volumes) ─────────
    // Periodic water dimer cube — pure-RS TS + isolated MBD with fixed v.
    const double L = 12.0 * kA2B;
    const double cell_per[3] = {L, L, L};
    constexpr double kRcutTs = 50.0 * kA2B;
    {
        auto ts_e = [&](const std::vector<std::array<double, 3>>& p,
                        const double c[3]) {
            return ts_pbe(Z, p, c, /*periodic=*/true, kRcutTs, 0.94, 20.0, kVol)
                .energy;
        };
        auto mbd_e = [&](const std::vector<std::array<double, 3>>& p,
                         const double c[3]) {
            (void)c;
            return mbd_rsscs_pbe(Z, p, cell0_iso, /*periodic=*/false, 0.83, 6.0,
                                 kVol)
                .energy;
        };
        // Isolated MBD is geometry-only (no cell images); strain still moves
        // nuclear coordinates → σ from FD of E_disp is well-defined on the
        // reference volume Ω of the unstrained cube.
        const auto sig_ts = sigma_disp_fd(ts_e, pos, cell_per, kStrainH);
        const auto sig_mbd = sigma_disp_fd(mbd_e, pos, cell_per, kStrainH);
        // Self-consistency of the helper (rebuild with same call).
        const auto sig_ts2 = sigma_disp_fd(ts_e, pos, cell_per, kStrainH);
        const auto sig_mbd2 = sigma_disp_fd(mbd_e, pos, cell_per, kStrainH);
        double max_ts = 0.0, max_mbd = 0.0, max_ts_abs = 0.0, max_mbd_abs = 0.0;
        for (int a = 0; a < 3; ++a) {
            max_ts = std::max(
                max_ts, std::abs(sig_ts[static_cast<std::size_t>(a)]
                                 - sig_ts2[static_cast<std::size_t>(a)]));
            max_mbd = std::max(
                max_mbd, std::abs(sig_mbd[static_cast<std::size_t>(a)]
                                  - sig_mbd2[static_cast<std::size_t>(a)]));
            max_ts_abs = std::max(
                max_ts_abs, std::abs(sig_ts[static_cast<std::size_t>(a)]));
            max_mbd_abs = std::max(
                max_mbd_abs, std::abs(sig_mbd[static_cast<std::size_t>(a)]));
        }
        if (max_ts > kSigmaDualTol || max_mbd > kSigmaDualTol) {
            fail("B geometry stress dual non-reproducible");
        }
        if (!(max_ts_abs > 1e-12) || !(max_mbd_abs > 1e-12)) {
            fail("B fixed-v TS/MBD stress must be nonzero under strain");
        }
        // Free-atom residual: volumes must move σ.
        auto ts_free_e = [&](const std::vector<std::array<double, 3>>& p,
                             const double c[3]) {
            return ts_pbe(Z, p, c, /*periodic=*/true, kRcutTs).energy;
        };
        const auto sig_free = sigma_disp_fd(ts_free_e, pos, cell_per, kStrainH);
        double max_df = 0.0;
        for (int a = 0; a < 3; ++a)
            max_df = std::max(
                max_df, std::abs(sig_ts[static_cast<std::size_t>(a)]
                                 - sig_free[static_cast<std::size_t>(a)]));
        if (max_df < 1e-12) {
            std::fprintf(stderr,
                         "FAIL: Hirshfeld volumes must move TS stress "
                         "(max|dσ|=%.3e)\n",
                         max_df);
            ++nfail;
        }
        std::printf(
            "  B geom dual: σ_TS={%.3e,%.3e,%.3e}  σ_MBD={%.3e,%.3e,%.3e}  "
            "max|σ_h-σ_free|=%.2e Ha/Bohr3\n",
            sig_ts[0], sig_ts[1], sig_ts[2], sig_mbd[0], sig_mbd[1],
            sig_mbd[2], max_df);
    }

    // ── C/D: multi-k product wiring on H₂ (1×1×2) ─────────────────────────
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

    // C: multi-k TS + hirshfeld stress
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
            GpwDft plain = make_cfg(false, false, false);
            const auto sig_h = gpw_stress(atoms, cfg, scf, kStrainH);
            const auto sig_plain = gpw_stress(atoms, plain, scf, kStrainH);

            auto ts_h = [&](const std::vector<std::array<double, 3>>& p,
                            const double c[3]) {
                return ts_pbe(Zh, p, c, /*periodic=*/true, kRcutTs, 0.94, 20.0,
                              scf.volume_ratios)
                    .energy;
            };
            auto ts_free = [&](const std::vector<std::array<double, 3>>& p,
                               const double c[3]) {
                return ts_pbe(Zh, p, c, /*periodic=*/true, kRcutTs).energy;
            };
            const auto sig_geom =
                sigma_disp_fd(ts_h, posh, cell_h2, kStrainH);
            const auto sig_geom_free =
                sigma_disp_fd(ts_free, posh, cell_h2, kStrainH);

            // Energy must match volume-scaled TS at SCF ratios.
            const TsResult geom_h =
                ts_pbe(Zh, posh, cell_h2, /*periodic=*/true, kRcutTs, 0.94,
                       20.0, scf.volume_ratios);
            if (std::abs(scf.e_disp - geom_h.energy) > kETol) {
                std::fprintf(stderr,
                             "FAIL: multi-k TS e_disp=%.14e vs geom_h=%.14e\n",
                             scf.e_disp, geom_h.energy);
                ++nfail;
            }

            double max_wire = 0.0;
            double max_wire_free = 0.0;
            for (int a = 0; a < 3; ++a) {
                const double shift = sig_h[static_cast<std::size_t>(a)]
                                     - sig_plain[static_cast<std::size_t>(a)];
                max_wire = std::max(
                    max_wire,
                    std::abs(shift - sig_geom[static_cast<std::size_t>(a)]));
                max_wire_free = std::max(
                    max_wire_free,
                    std::abs(shift
                             - sig_geom_free[static_cast<std::size_t>(a)]));
                // Off-diagonals of product stress must stay zero.
                if (std::abs(sig_h[static_cast<std::size_t>(a + 3)]) > 1e-12)
                    fail("non-zero off-diagonal stress (TS+hirsh)");
            }
            if (max_wire > kWireTol) {
                std::fprintf(stderr,
                             "FAIL: multi-k TS+Hirsh stress wiring max|d|=%.3e "
                             "> tol=%.1e\n",
                             max_wire, kWireTol);
                ++nfail;
            }
            if (max_wire_free < 1e-12) {
                fail("multi-k TS stress still matches free-atom (hirsh not "
                     "applied)");
            }
            std::printf(
                "  C multi-k TS+Hirsh H2: e_disp=%.14e  n_kpts=%d  "
                "max|wire_h|=%.2e  max|wire_free|=%.2e  v0=%.4f\n",
                scf.e_disp, scf.n_kpts, max_wire, max_wire_free,
                scf.volume_ratios[0]);
        }
    }

    // D: multi-k MBD + hirshfeld stress (isolated MBD when mbd_k_mesh = 0)
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
            GpwDft plain = make_cfg(false, false, false);
            const auto sig_h = gpw_stress(atoms, cfg, scf, kStrainH);
            const auto sig_plain = gpw_stress(atoms, plain, scf, kStrainH);

            auto mbd_h = [&](const std::vector<std::array<double, 3>>& p,
                             const double c[3]) {
                (void)c;
                return mbd_rsscs_pbe(Zh, p, cell0_iso, /*periodic=*/false, 0.83,
                                     6.0, scf.volume_ratios)
                    .energy;
            };
            auto mbd_free = [&](const std::vector<std::array<double, 3>>& p,
                                const double c[3]) {
                (void)c;
                return mbd_rsscs_pbe(Zh, p, cell0_iso, /*periodic=*/false)
                    .energy;
            };
            const auto sig_geom =
                sigma_disp_fd(mbd_h, posh, cell_h2, kStrainH);
            const auto sig_geom_free =
                sigma_disp_fd(mbd_free, posh, cell_h2, kStrainH);

            const MbdResult geom_h =
                mbd_rsscs_pbe(Zh, posh, cell0_iso, /*periodic=*/false, 0.83,
                              6.0, scf.volume_ratios);
            if (std::abs(scf.e_disp - geom_h.energy) > kETol) {
                std::fprintf(stderr,
                             "FAIL: multi-k MBD e_disp=%.14e vs geom_h=%.14e\n",
                             scf.e_disp, geom_h.energy);
                ++nfail;
            }

            double max_wire = 0.0;
            double max_wire_free = 0.0;
            for (int a = 0; a < 3; ++a) {
                const double shift = sig_h[static_cast<std::size_t>(a)]
                                     - sig_plain[static_cast<std::size_t>(a)];
                max_wire = std::max(
                    max_wire,
                    std::abs(shift - sig_geom[static_cast<std::size_t>(a)]));
                max_wire_free = std::max(
                    max_wire_free,
                    std::abs(shift
                             - sig_geom_free[static_cast<std::size_t>(a)]));
                if (std::abs(sig_h[static_cast<std::size_t>(a + 3)]) > 1e-12)
                    fail("non-zero off-diagonal stress (MBD+hirsh)");
            }
            if (max_wire > kWireTol) {
                std::fprintf(stderr,
                             "FAIL: multi-k MBD+Hirsh stress wiring max|d|=%.3e "
                             "> tol=%.1e\n",
                             max_wire, kWireTol);
                ++nfail;
            }
            if (max_wire_free < 1e-12) {
                fail("multi-k MBD stress still matches free-atom (hirsh not "
                     "applied)");
            }
            std::printf(
                "  D multi-k MBD+Hirsh H2: e_disp=%.14e  n_kpts=%d  "
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
    std::printf("PASS: gpw-hirshfeld-kpoint-stress (MBD/TS fixed-v dual + "
                "multi-k H2 TS/MBD+Hirsh stress wiring; wall=%.2f s)\n",
                wall);
    return 0;
}
