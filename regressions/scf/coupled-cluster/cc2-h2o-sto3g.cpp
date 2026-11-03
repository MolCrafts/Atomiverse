/// @file regressions/scf/coupled-cluster/cc2-h2o-sto3g.cpp
/// @brief H2O CC2/STO-3G correlation + total energy vs offline PySCF pin
///        (REFERENCE.md).
///
/// H2/STO-3G has T1 = 0, so CC2 equals MP2. This geometry has a nonzero
/// singles amplitude (PySCF t1_norm = 9.663e-3) and a CC2 correlation
/// energy 1.231e-4 Ha below MP2. Out-of-library consumer: public headers
/// only. Product gate: e_corr and e_tot to 1e-6 Ha.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_cc2_h2o_sto3g {

namespace {

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.15f want=%.15f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-28s %18.12f (want %18.12f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

ATV::cpu::Frame water() {
    using namespace ATV;
    // Geometry (Å) — pin_scf `water` preset.
    cpu::Frame frame(3);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.span(kAtom_atomic_number)[0] = 8;
    frame.span(kAtom_atomic_number)[1] = 1;
    frame.span(kAtom_atomic_number)[2] = 1;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int run() {
    using namespace ATV;

    // Pins — REFERENCE.md (PySCF 2.14.0 rccsd.RCCSD cc2=True / RHF STO-3G).
    constexpr double kEHf = -74.963023138462859;  // Ha
    constexpr double kECorr = -0.035668770406499; // Ha
    constexpr double kETot = -74.998691908869361; // Ha
    // Same geometry, PySCF mp.MP2. Distinct from CC2 by 1.231e-4 Ha.
    constexpr double kECorrMp2 = -0.035545651646918; // Ha
    constexpr double kTol = 1e-6;                    // Ha

    cpu::Frame frame = water();
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    rhf.tol = 1e-12;
    rhf.max_iter = 100;
    cpu::scf::CC2<decltype(rhf)> cc2{rhf};

    // kEnergy is part of the Frame schema (0 until a writer publishes).
    // run(const Frame&) must leave that value untouched.
    cpu::Frame frozen = water();
    const double energy_before = frozen.meta.get(kEnergy);
    const cpu::scf::CC2Result quiet =
        cc2.run(static_cast<const cpu::Frame&>(frozen));
    if (frozen.meta.get(kEnergy) != energy_before) {
        std::fprintf(stderr,
                     "FAIL: CC2::run(const Frame&) wrote kEnergy "
                     "(got %.15f, was %.15f)\n",
                     frozen.meta.get(kEnergy), energy_before);
        return 1;
    }
    if (!quiet.converged) {
        std::fprintf(stderr, "FAIL: CC2 amplitudes did not converge (const)\n");
        return 1;
    }
    if (!check("E_corr const [Ha]", quiet.e_corr, kECorr, kTol))
        return 1;

    const cpu::scf::CC2Result live = cc2.run(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G (H2O) did not converge\n");
        return 1;
    }
    if (!live.converged || !cc2.converged()) {
        std::fprintf(stderr, "FAIL: CC2 amplitudes did not converge\n");
        return 1;
    }

    std::printf("H2O CC2/STO-3G (RHF reference, T1 nonzero):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kEHf, kTol))
        return 1;
    if (!check("E_corr CC2 [Ha]", live.e_corr, kECorr, kTol))
        return 1;
    if (!check("E_tot CC2 [Ha]", live.e_tot, kETot, kTol))
        return 1;
    if (!check("E_tot = E_HF+E_corr", live.e_tot, rhf.energy() + live.e_corr,
               1e-12))
        return 1;
    if (std::abs(live.e_corr - kECorrMp2) <= kTol) {
        std::fprintf(stderr,
                     "FAIL: CC2 correlation collapsed onto MP2 "
                     "(got %.15f, MP2 pin %.15f)\n",
                     live.e_corr, kECorrMp2);
        return 1;
    }
    if (!frame.meta.contains(kEnergy.key)) {
        std::fprintf(stderr,
                     "FAIL: CC2::run(Frame&) did not publish kEnergy\n");
        return 1;
    }
    if (!check("kEnergy [eV]", frame.meta.get(kEnergy),
               live.e_tot * units::kHartreeInEv, 1e-8))
        return 1;

    std::printf("  iterations                 %d\n", live.iterations);
    std::printf("\nPASS: cc2-h2o-sto3g\n");
    return 0;
}

} // namespace atv_reg_scf_cc2_h2o_sto3g

ATV_REGRESSION("scf/coupled-cluster", "cc2-h2o-sto3g",
               atv_reg_scf_cc2_h2o_sto3g::run);
