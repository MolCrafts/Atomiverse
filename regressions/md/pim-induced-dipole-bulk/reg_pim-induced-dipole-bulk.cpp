/// @file regressions/md/pim-induced-dipole-bulk/reg_pim-induced-dipole-bulk.cpp
/// @brief END-TO-END: bulk PIM induced dipoles vs the REAL MetalWalls binary.
///
/// Two gates on the same two-site frame (see REFERENCE.md for the raw
/// MetalWalls output, the box-size ladder and every unit conversion):
///
///   A. thole_a = 0 — the configuration MetalWalls can express.  Pinned to
///      MetalWalls c8ad175d output (atomic units, extrapolated to the isolated
///      limit over a 50 -> 3200 Å box ladder).  This is the oracle gate.
///   B. thole_a = 0.39 — Thole/Applequist exponential damping.  MetalWalls
///      implements NO Thole damping (its only short-range multipole screening
///      is Tang–Toennies on the charge->dipole field), so this stays a
///      literature/closed-form pin and is labelled as such.
///
/// Public headers + link atomiverse only. Flat main(); exits non-zero on fail.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"

using namespace ATV;

namespace {

// CODATA-2018, the same values MetalWalls' own output headers quote.
constexpr double kHartree_eV = 27.211386245988;   // 1 Ha [eV]
constexpr double kBohr_Ang = 0.529177210903;      // 1 a0 [Å]

// Geometry: q=+1 α=1.0 Å³ at origin; q=-1 α=1.5 Å³ at (3,0,0); L=50 Å cubic.
void build_frame(cpu::Frame& frame) {
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = 3.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.0;
    frame.span(kAtom_charge)[0] = 1.0;
    frame.span(kAtom_charge)[1] = -1.0;
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 2;
    frame.span(kAtom_mass)[0] = frame.span(kAtom_mass)[1] = 1.0;
    frame.span(kAtom_tag)[0] = 1;
    frame.span(kAtom_tag)[1] = 2;
    frame.span(kAtom_fx)[0] = frame.span(kAtom_fy)[0] =
        frame.span(kAtom_fz)[0] = 0;
    frame.span(kAtom_fx)[1] = frame.span(kAtom_fy)[1] =
        frame.span(kAtom_fz)[1] = 0;
    frame.box = cpu::Box::cubic(50.0);
    frame.meta.get(kEnergy) = 0;
}

} // namespace

int main() {
    bool ok = true;

    // ── A. MetalWalls oracle gate (undamped multipoles) ──────────────────
    //
    // MetalWalls c8ad175d, mpirun -n 1, inputs at
    //   tools/metalwalls_oracle/inputs/metalwalls/pim-induced-dipole-bulk/
    // run over L = 50,100,200,400,800,1600,3200 Å and Richardson-extrapolated
    // in 1/L³ to the isolated limit (MetalWalls is Ewald-periodic; the
    // Atomiverse component is min-image/cutoff, so the isolated limit is the
    // only common ground — see REFERENCE.md).
    const double mw_mu0_au = 2.3523561717334e-01;  // [e·a0]
    const double mw_mu1_au = 3.4109164490134e-01;  // [e·a0]
    const double mw_epol_ha = -8.9660039191731e-03; // [Ha]
    const double mw_f0_au = 6.7922032485569e-03;   // [Ha/a0]

    // Dipoles are pure Gaussian-unit geometry (the Coulomb constant never
    // enters μ = αE), so they gate at 1e-9 relative; measured 2.6e-14.
    const double tol_mu_rel = 1e-9;
    // Energy and force carry ATV's units::coulomb = 14.399645 eV·Å/e², an
    // 8-digit truncation of the exact 14.399645478456…, i.e. a hard
    // −3.3227e-08 relative floor. Measured deviation is exactly that value;
    // after rescaling by the constant the agreement is 6e-14.
    const double tol_ef_rel = 1e-7;

    cpu::Frame frame_mw(2);
    build_frame(frame_mw);

    cpu::PimInducedDipole pim_mw(20.0, /*alpha_by_type=*/{0.0, 1.0, 1.5},
                                 /*thole_a=*/0.0, 1e-14, 500);
    cpu::Evaluate eval_mw;
    eval_mw.on(evaluate::Phase::evaluate).add_ref(pim_mw);
    eval_mw.build();
    eval_mw.run(frame_mw);

    const double mu0_au = pim_mw.last_mu()[0] / kBohr_Ang;
    const double mu1_au = pim_mw.last_mu()[3] / kBohr_Ang;
    const double epol_ha = frame_mw.meta.get(kEnergy) / kHartree_eV;
    const double f0_au =
        frame_mw.span(kAtom_fx)[0] * kBohr_Ang / kHartree_eV;
    const double f1_au =
        frame_mw.span(kAtom_fx)[1] * kBohr_Ang / kHartree_eV;

    std::printf("[MW gate] thole_a=0, %d SCC iters\n", pim_mw.last_pol_iters());
    std::printf("  mu0  = %.16e e.a0   MW %.16e  rel %+.3e\n", mu0_au,
                mw_mu0_au, (mu0_au - mw_mu0_au) / mw_mu0_au);
    std::printf("  mu1  = %.16e e.a0   MW %.16e  rel %+.3e\n", mu1_au,
                mw_mu1_au, (mu1_au - mw_mu1_au) / mw_mu1_au);
    std::printf("  Epol = %.16e Ha     MW %.16e  rel %+.3e\n", epol_ha,
                mw_epol_ha, (epol_ha - mw_epol_ha) / std::fabs(mw_epol_ha));
    std::printf("  F0x  = %.16e Ha/a0  MW %.16e  rel %+.3e\n", f0_au, mw_f0_au,
                (f0_au - mw_f0_au) / mw_f0_au);

    if (!(std::fabs(mu0_au - mw_mu0_au) <= tol_mu_rel * std::fabs(mw_mu0_au))) {
        std::printf("FAIL mu0 vs MetalWalls\n");
        ok = false;
    }
    if (!(std::fabs(mu1_au - mw_mu1_au) <= tol_mu_rel * std::fabs(mw_mu1_au))) {
        std::printf("FAIL mu1 vs MetalWalls\n");
        ok = false;
    }
    if (!(std::fabs(epol_ha - mw_epol_ha)
          <= tol_ef_rel * std::fabs(mw_epol_ha))) {
        std::printf("FAIL Epol vs MetalWalls\n");
        ok = false;
    }
    if (!(std::fabs(f0_au - mw_f0_au) <= tol_ef_rel * std::fabs(mw_f0_au))) {
        std::printf("FAIL F0x vs MetalWalls\n");
        ok = false;
    }
    if (!(std::fabs(f1_au + mw_f0_au) <= tol_ef_rel * std::fabs(mw_f0_au))) {
        std::printf("FAIL F1x vs MetalWalls (action-reaction)\n");
        ok = false;
    }
    if (!pim_mw.last_pol_converged()) {
        std::printf("FAIL MW-gate SCC not converged\n");
        ok = false;
    }

    // ── B. Thole/Applequist pin — NOT a MetalWalls number ────────────────
    //
    // MetalWalls has no Thole damping (grep 'thole' over the MetalWalls tree
    // at c8ad175d: zero hits). Closed-form two-site Applequist SCC with
    // van Duijnen/Swart exponential λ3/λ5/λ7; see REFERENCE.md.
    const double E_pin = -0.12040946414707571; // [eV]
    const double f0x_pin = 0.040341072868572643;
    const double f1x_pin = -0.040341072868572643;
    const double tol_e = 1e-9;
    const double tol_f = 1e-8;

    cpu::Frame frame(2);
    build_frame(frame);

    cpu::PimInducedDipole pim(20.0, /*alpha_by_type=*/{0.0, 1.0, 1.5},
                              /*thole_a=*/0.39, 1e-12, 200);
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(pim);
    eval.build();
    eval.run(frame);

    const double E = frame.meta.get(kEnergy);
    const double f0x = frame.span(kAtom_fx)[0];
    const double f0y = frame.span(kAtom_fy)[0];
    const double f0z = frame.span(kAtom_fz)[0];
    const double f1x = frame.span(kAtom_fx)[1];
    const double f1y = frame.span(kAtom_fy)[1];
    const double f1z = frame.span(kAtom_fz)[1];

    std::printf("[Thole pin] E=%.16e want=%.16e\n", E, E_pin);
    std::printf("  F0=(%.16e, %.16e, %.16e)\n", f0x, f0y, f0z);
    std::printf("  F1=(%.16e, %.16e, %.16e)\n", f1x, f1y, f1z);
    std::printf("  e_pol=%.16e iters=%d\n", pim.last_pol_energy(),
                pim.last_pol_iters());

    if (!(std::fabs(E - E_pin) <= tol_e)) {
        std::printf("FAIL energy got=%.16e want=%.16e tol=%.1e\n", E, E_pin,
                    tol_e);
        ok = false;
    }
    if (!(std::fabs(pim.last_pol_energy() - E_pin) <= tol_e)) {
        std::printf("FAIL last_pol_energy got=%.16e want=%.16e\n",
                    pim.last_pol_energy(), E_pin);
        ok = false;
    }
    if (!(std::fabs(f0x - f0x_pin) <= tol_f
          && std::fabs(f1x - f1x_pin) <= tol_f)) {
        std::printf(
            "FAIL force got F0x=%.16e F1x=%.16e want %.16e / %.16e tol=%.1e\n",
            f0x, f1x, f0x_pin, f1x_pin, tol_f);
        ok = false;
    }
    if (!(std::fabs(f0y) < 1e-12 && std::fabs(f0z) < 1e-12
          && std::fabs(f1y) < 1e-12 && std::fabs(f1z) < 1e-12)) {
        std::printf("FAIL transverse forces non-zero\n");
        ok = false;
    }
    if (!(std::fabs(f0x + f1x) < 1e-12)) {
        std::printf("FAIL action-reaction\n");
        ok = false;
    }
    if (!pim.last_pol_converged()) {
        std::printf("FAIL SCC not converged\n");
        ok = false;
    }
    if (!(E < 0.0)) {
        std::printf("FAIL E_pol should be negative\n");
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS: pim-induced-dipole-bulk vs MetalWalls c8ad175d "
                "(undamped) + Thole closed form\n");
    return 0;
}
