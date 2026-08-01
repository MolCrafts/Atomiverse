/// @file regressions/gpw/gpw-d3-atm-kpoint-forces/reg_gpw-d3-atm-kpoint-forces.cpp
/// @brief END-TO-END: multi-k GPW analytic ionic forces with DFT-D3 three-body
///        ATM (C9), vs offline CP2K ENERGY_FORCE twins on a Monkhorst–Pack mesh.
///
/// Capability: multi-k force assembly folds `dftd3_pbe(..., calculate_c9 =
/// cfg.d3_atm)` identically to Γ (`gpw_forces.cpp` k path). Twin of
/// `gpw-d3-atm-forces` (Γ periodic) on the same 6 Å water-dimer cell for the
/// ATM pin, with electronic mesh 1×1×2 on a cheap H₂ product wiring twin
/// (same pattern as `gpw-mbd-kpoint-forces`).
///
/// Hard pins from two offline CP2K 2026.1 runs (`ENERGY_FORCE`,
/// `&KPOINTS MONKHORST-PACK 1 1 2`, `CALCULATE_C9_TERM` T/F, otherwise
/// identical) on the water dimer:
///   1. E_disp — printed "Dispersion energy", ATM on and off (geometry-only;
///      bit-identical to the Γ pin).
///   2. F_ATM — F(C9) − F(two-body). Electronic force cancels; pure ATM
///      gradient (matches Γ ΔF to print residual).
/// Product dual: multi-k H₂ SCF + force assembly (d3_atm on − off) on the
/// same density equals the standalone ATM ΔF. Multi-k payload: C_occ_k.
///
/// Public API only. No live CP2K. See REFERENCE.md.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

constexpr double kAng2Bohr = 1.8897259885789;

// ── Offline CP2K 2026.1 multi-k ENERGY_FORCE pins (see REFERENCE.md) ─────
// Periodic 6 Å water dimer, R_CUTOFF 10.0, EPS_SCF 1e-8, MP 1×1×2.
constexpr double kEDispAtmCp2k = -0.00195809486938; // Ha (CALCULATE_C9 .TRUE.)
constexpr double kEDispTwoCp2k = -0.00195887984277; // Ha (.FALSE.)
constexpr double kEDispTol = 1.0e-9;

// FORCES| Atomic forces [hartree/bohr], atom order O H H O H H.
// Multi-k, CALCULATE_C9_TERM .TRUE.:
constexpr double kFAtmCp2k[6][3] = {
    {-7.76100703E-02, 4.42032292E-10, 4.26650979E-10},
    {3.84410315E-02, 1.33918034E-11, 1.30001789E-01},
    {3.84410315E-02, 1.23244443E-11, -1.30001789E-01},
    {-7.76098924E-02, 5.14119447E-10, 5.36972620E-10},
    {3.84411038E-02, 1.31896386E-11, 1.30001779E-01},
    {3.84411038E-02, 1.43579460E-11, -1.30001779E-01},
};
// Multi-k, CALCULATE_C9_TERM .FALSE.:
constexpr double kFTwoCp2k[6][3] = {
    {-7.76105812E-02, 4.42038340E-10, 4.26653893E-10},
    {3.84413680E-02, 1.33916184E-11, 1.30001230E-01},
    {3.84413680E-02, 1.23242605E-11, -1.30001230E-01},
    {-7.76105827E-02, 5.14117684E-10, 5.36989844E-10},
    {3.84413679E-02, 1.31894361E-11, 1.30001230E-01},
    {3.84413679E-02, 1.43582194E-11, -1.30001230E-01},
};

// ATM-only force pin: difference cancels electronic force to ~1e-15;
// floor is FORCES| print resolution vs ATM signal ~6.9e-7 Ha/Bohr.
constexpr double kFAtmTol = 1.0e-8;  // Ha/Bohr
constexpr double kFAtmMin = 1.0e-7;  // Ha/Bohr engage floor
// Product multi-k wiring (H2): force assembly (d3_atm on-off) vs standalone.
constexpr double kWiringTol = 1.0e-7; // Ha/Bohr
constexpr double kWallBudget = 60.0;  // s
// Fair wall (REFERENCE.md): offline product multi-k water-dimer ENERGY_FORCE
// twin (same geometry/method class as the CP2K pin), not the cheap H2 wiring
// gate below. Recorded T_atv / T_cp2k fill the PASS line for provenance.
constexpr double kTAtvFairSec = 14.97; // s — product SCF+force, OMP 8, 150 Ry
constexpr double kTCp2kSec = 1.51;     // s — CP2K cold ENERGY_FORCE, 1×8

void atm_diff_errors(const std::vector<std::array<double, 3>>& f_on,
                     const std::vector<std::array<double, 3>>& f_off,
                     const double cp2k_on[6][3], const double cp2k_off[6][3],
                     double& max_err, double& max_signal) {
    max_err = 0.0;
    max_signal = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double want = cp2k_on[i][k] - cp2k_off[i][k];
            const double got = f_on[static_cast<std::size_t>(i)]
                                   [static_cast<std::size_t>(k)]
                               - f_off[static_cast<std::size_t>(i)]
                                      [static_cast<std::size_t>(k)];
            max_err = std::max(max_err, std::abs(got - want));
            max_signal = std::max(max_signal, std::abs(want));
        }
    }
}

} // namespace

int main() {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // ── (1) Standalone ATM vs CP2K multi-k ΔF pin (water dimer 6 Å) ───────
    constexpr double L = 6.0; // Å
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double raw[6][3] = {
        {0.000000, 0.000000, 0.000000},  {0.758602, 0.000000, 0.504284},
        {0.758602, 0.000000, -0.504284}, {3.000000, 0.000000, 0.000000},
        {3.758602, 0.000000, 0.504284},  {3.758602, 0.000000, -0.504284},
    };

    std::vector<int> Zv(6);
    std::vector<std::array<double, 3>> pos(6);
    for (int i = 0; i < 6; ++i) {
        Zv[static_cast<std::size_t>(i)] = z[i];
        pos[static_cast<std::size_t>(i)] = {raw[i][0] * kAng2Bohr,
                                            raw[i][1] * kAng2Bohr,
                                            raw[i][2] * kAng2Bohr};
    }
    const double Lb = L * kAng2Bohr;
    const double cell[3] = {Lb, Lb, Lb};
    // CP2K R_CUTOFF 10.0 Å is a half cutoff: pairs/triples sum to 2×.
    const double rcut = 2.0 * 10.0 * kAng2Bohr;

    const cpu::gpw::D3Result d3_on = cpu::gpw::dftd3_pbe(
        Zv, pos, cell, /*periodic=*/true, cpu::gpw::D3Params{}, rcut,
        /*calculate_c9=*/true);
    const cpu::gpw::D3Result d3_off = cpu::gpw::dftd3_pbe(
        Zv, pos, cell, /*periodic=*/true, cpu::gpw::D3Params{}, rcut,
        /*calculate_c9=*/false);

    if (!std::isfinite(d3_on.energy)
        || std::abs(d3_on.energy - kEDispAtmCp2k) > kEDispTol
        || std::abs(d3_off.energy - kEDispTwoCp2k) > kEDispTol) {
        std::printf("FAIL: E_disp  ATM got=%.14e want=%.14e  two-body "
                    "got=%.14e want=%.14e (tol %.1e Ha)\n",
                    d3_on.energy, kEDispAtmCp2k, d3_off.energy, kEDispTwoCp2k,
                    kEDispTol);
        return 1;
    }
    double err_atm = 0.0;
    double sig_atm = 0.0;
    atm_diff_errors(d3_on.forces, d3_off.forces, kFAtmCp2k, kFTwoCp2k, err_atm,
                    sig_atm);
    if (err_atm > kFAtmTol || sig_atm < kFAtmMin) {
        std::printf("FAIL: multi-k ATM force pin  max|dF-cp2k|=%.3e (tol %.1e)  "
                    "signal=%.3e (min %.1e) Ha/Bohr\n",
                    err_atm, kFAtmTol, sig_atm, kFAtmMin);
        return 1;
    }
    std::printf("standalone ATM: E_disp=%.14e (|d|=%.2e)  max|dF-cp2k|=%.3e  "
                "signal=%.3e Ha/Bohr\n",
                d3_on.energy, std::abs(d3_on.energy - kEDispAtmCp2k), err_atm,
                sig_atm);

    // ── (2) Product multi-k SCF + force assembly wiring (H2 1×1×2) ────────
    // Cheap product twin (same mesh class as the CP2K ENERGY_FORCE twin)
    // so the e2e wall stays under 60 s. ATM pin above already exercised
    // lattice-image triples on the water dimer.
    const auto basis = cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    std::vector<cpu::gpw::GpwAtom> atoms = {
        cpu::gpw::GpwAtom(rA, basis, pot, 1),
        cpu::gpw::GpwAtom(rB, basis, pot, 1)};

    auto make_cfg = [](bool atm) {
        cpu::gpw::GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 100.0;
        m.n_occ = 1;
        m.tol = 1e-8;
        m.max_iter = 60;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.xc = cpu::gpw::XcKind::PADE;
        m.eps_default = 1e-8;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.d3 = true;
        m.d3_atm = atm;
        m.d3_rcut_bohr = 20.0;
        return m;
    };

    const auto t_prod0 = clock::now();
    cpu::gpw::GpwDft cfg_atm = make_cfg(true);
    cfg_atm.build(atoms);
    const auto& scf = cfg_atm.run(atoms);
    if (!scf.converged) {
        std::printf("FAIL: multi-k SCF did not converge (iters=%d)\n",
                    scf.iters);
        return 1;
    }
    if (scf.n_kpts < 1 || scf.C_occ_k.empty()) {
        std::printf("FAIL: multi-k payload missing (n_kpts=%d C_occ_k.empty=%d)\n",
                    scf.n_kpts, scf.C_occ_k.empty() ? 1 : 0);
        return 1;
    }
    if (!std::isfinite(scf.e_disp)) {
        std::printf("FAIL: product multi-k e_disp non-finite\n");
        return 1;
    }

    const auto F_on = cpu::gpw::gpw_forces_from_scf(atoms, cfg_atm, scf);
    cpu::gpw::GpwDft cfg_two = make_cfg(false);
    cfg_two.build(atoms);
    const auto F_off = cpu::gpw::gpw_forces_from_scf(atoms, cfg_two, scf);
    const double t_prod_s =
        std::chrono::duration<double>(clock::now() - t_prod0).count();

    std::vector<int> Zh = {1, 1};
    std::vector<std::array<double, 3>> posh = {
        {rA[0], rA[1], rA[2]}, {rB[0], rB[1], rB[2]}};
    const double cell_h[3] = {12.0, 12.0, 12.0};
    const cpu::gpw::D3Result h_on = cpu::gpw::dftd3_pbe(
        Zh, posh, cell_h, /*periodic=*/true, cpu::gpw::D3Params{}, 20.0,
        /*calculate_c9=*/true);
    const cpu::gpw::D3Result h_off = cpu::gpw::dftd3_pbe(
        Zh, posh, cell_h, /*periodic=*/true, cpu::gpw::D3Params{}, 20.0,
        /*calculate_c9=*/false);

    double max_wire = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double on = F_on[static_cast<std::size_t>(a)][k];
            const double off = F_off[static_cast<std::size_t>(a)][k];
            if (!std::isfinite(on) || !std::isfinite(off)) {
                std::printf("FAIL: non-finite force atom %d axis %d\n", a, k);
                return 1;
            }
            const double f_atm =
                h_on.forces[static_cast<std::size_t>(a)][k]
                - h_off.forces[static_cast<std::size_t>(a)][k];
            max_wire = std::max(max_wire, std::abs((on - off) - f_atm));
        }
    }
    if (max_wire > kWiringTol) {
        std::printf("FAIL: multi-k wiring max|(F_on-F_off)-F_ATM|=%.3e > "
                    "%.1e Ha/Bohr\n",
                    max_wire, kWiringTol);
        return 1;
    }

    const double wall_s =
        std::chrono::duration<double>(clock::now() - t0).count();
    if (wall_s > kWallBudget) {
        std::printf("FAIL: wall=%.2f s exceeds %.0f s e2e budget\n", wall_s,
                    kWallBudget);
        return 1;
    }

    const double ratio = kTAtvFairSec / kTCp2kSec;
    std::printf("PASS: gpw-d3-atm-kpoint-forces  max|dF-cp2k|=%.3e (sig %.3e)  "
                "wiring=%.3e  nk=%d  iters=%d  T_atv_h2=%.2f s  "
                "T_atv_fair=%.2f s  T_cp2k=%.2f s  ratio=%.3f  wall=%.2f s\n",
                err_atm, sig_atm, max_wire, scf.n_kpts, scf.iters, t_prod_s,
                kTAtvFairSec, kTCp2kSec, ratio, wall_s);
    return 0;
}
