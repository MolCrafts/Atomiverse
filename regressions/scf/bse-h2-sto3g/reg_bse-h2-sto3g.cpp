/// @file regressions/scf/bse-h2-sto3g/reg_bse-h2-sto3g.cpp
/// @brief H₂ RHF/STO-3G singlet BSE@G0W0 S1 vs offline PySCF-integral pin.
///
/// Public-API consumer only. Geometry, basis, and oracle numbers are pinned
/// in REFERENCE.md (PySCF 2.14.0 offline; no live import in ctest).
///
/// Observable: lowest singlet BSE excitation [eV] from `RHF::bse()` —
/// spectral G0W0 poles + static dRPA-screened W (exact 4-centre ERIs).
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.12f want=%.12f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-36s %14.8f (want %14.8f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

} // namespace

int main() {
    using namespace ATV;

    // Geometry [Å] — REFERENCE.md (H–H = 0.74 Å on +z).
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.74;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);

    // Offline pins (REFERENCE.md): exact-ERI BSE@spectral-G0W0 twin.
    constexpr double kERhfHa = -1.116759307396;
    constexpr double kS1Ev = 25.956271542931; // primary gate
    constexpr double kTdaEv = 26.809052342076;
    constexpr double kGapEv = 35.007583892680; // G0W0 QP gap (sanity)
    constexpr double kTolEv = 1e-4;            // job gate
    constexpr double kTolEHa = 1e-6;

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;
    rhf.run(f);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF did not converge\n");
        return 1;
    }

    std::printf("H2 RHF/STO-3G singlet BSE@G0W0:\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kERhfHa, kTolEHa))
        return 1;

    const cpu::scf::BSEResult bse = rhf.bse(/*n_states=*/1, /*tda=*/false);
    if (bse.energy_ev.empty()) {
        std::fprintf(stderr, "FAIL: BSE returned no roots\n");
        return 1;
    }
    if (!bse.gw.converged) {
        std::fprintf(stderr, "FAIL: underlying G0W0 did not converge\n");
        return 1;
    }

    std::printf("  S1 = %.10f eV  (G0W0 gap = %.8f eV)\n", bse.energy_ev[0],
                bse.gw.gap_ev);

    if (!check("BSE S1 [eV]", bse.energy_ev[0], kS1Ev, kTolEv))
        return 1;
    if (!check("G0W0 gap [eV]", bse.gw.gap_ev, kGapEv, kTolEv))
        return 1;

    // TDA-BSE companion (not the primary gate).
    const cpu::scf::BSEResult tda = rhf.bse(/*n_states=*/1, /*tda=*/true);
    if (!check("BSE-TDA S1 [eV]", tda.energy_ev[0], kTdaEv, kTolEv))
        return 1;

    // Physical sanity: full BSE below TDA; BSE below the QP gap on this system.
    if (!(bse.energy_ev[0] < tda.energy_ev[0])) {
        std::fprintf(stderr, "FAIL: full BSE S1 %.6f not < TDA %.6f eV\n",
                     bse.energy_ev[0], tda.energy_ev[0]);
        return 1;
    }
    if (!(bse.energy_ev[0] < bse.gw.gap_ev)) {
        std::fprintf(stderr, "FAIL: BSE S1 %.6f eV not < QP gap %.6f eV\n",
                     bse.energy_ev[0], bse.gw.gap_ev);
        return 1;
    }

    // Ha consistency via CODATA constant (regression uses eV surface).
    const double s1_ha = bse.energy_ev[0] / units::kHartreeInEv;
    if (!std::isfinite(s1_ha) || s1_ha <= 0.0) {
        std::fprintf(stderr, "FAIL: non-physical S1 Ha = %g\n", s1_ha);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
