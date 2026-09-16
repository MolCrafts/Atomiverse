/// @file regressions/scf/g0w0-h2-sto3g/reg_g0w0-h2-sto3g.cpp
/// @brief H₂ RHF/STO-3G spectral G0W0 quasiparticle gap vs PySCF pin.
///
/// Public-API consumer only. Geometry, basis, and oracle numbers are pinned
/// in REFERENCE.md (PySCF 2.14.0 offline; no live import in ctest).
///
/// Observable: HOMO–LUMO QP gap [eV] (and IP / EA) from `RHF::g0w0()` —
/// frequency-exact G0W0 with dRPA screening (PySCF `gw.gw_exact.GWExact` twin).
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

    // Geometry [Å] — REFERENCE.md (bond on +z; any axis is equivalent).
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.74;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);

    // PySCF 2.14.0 spectral G0W0 pins (REFERENCE.md).
    constexpr double kERhfHa = -1.116759307396;
    constexpr double kGapEv = 35.007583892680; // primary gate
    constexpr double kIpEv = 16.244045825851;
    constexpr double kEaEv = 18.763538066829;
    constexpr double kQpHomoHa = -0.596957673490;
    constexpr double kQpLumoHa = 0.689547305573;
    constexpr double kTolGapEv = 1e-4; // job gate
    constexpr double kTolEHa = 1e-6;
    constexpr double kTolQpHa = 1e-5;

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;
    rhf.run(f);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF did not converge\n");
        return 1;
    }

    std::printf("H2 RHF/STO-3G spectral G0W0 (dRPA screening):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kERhfHa, kTolEHa))
        return 1;

    const cpu::scf::G0W0Result gw = rhf.g0w0();
    if (!gw.converged) {
        std::fprintf(stderr, "FAIL: G0W0 QP Newton did not converge\n");
        return 1;
    }
    if (gw.n_occ != 1 || gw.qp_energy_ha.size() != 2) {
        std::fprintf(stderr, "FAIL: unexpected MO counts n_occ=%d n_mo=%zu\n",
                     gw.n_occ, gw.qp_energy_ha.size());
        return 1;
    }

    std::printf("  QP HOMO = %.10f Ha  LUMO = %.10f Ha\n", gw.qp_energy_ha[0],
                gw.qp_energy_ha[1]);
    std::printf("  gap = %.8f eV  IP = %.8f eV  EA = %.8f eV\n", gw.gap_ev,
                gw.ip_ev, gw.ea_ev);

    if (!check("QP gap [eV]", gw.gap_ev, kGapEv, kTolGapEv))
        return 1;
    if (!check("IP [eV]", gw.ip_ev, kIpEv, kTolGapEv))
        return 1;
    if (!check("EA [eV]", gw.ea_ev, kEaEv, kTolGapEv))
        return 1;
    if (!check("QP HOMO [Ha]", gw.qp_energy_ha[0], kQpHomoHa, kTolQpHa))
        return 1;
    if (!check("QP LUMO [Ha]", gw.qp_energy_ha[1], kQpLumoHa, kTolQpHa))
        return 1;

    // Gap consistency with Ha values via CODATA constant.
    const double gap_from_ha =
        (gw.qp_energy_ha[1] - gw.qp_energy_ha[0]) * units::kHartreeInEv;
    if (!check("gap from Ha [eV]", gap_from_ha, gw.gap_ev, 1e-9))
        return 1;

    // Physical sanity: G0W0 opens the HF gap on this tiny system.
    const double gap_hf_ev =
        (gw.mf_energy_ha[1] - gw.mf_energy_ha[0]) * units::kHartreeInEv;
    if (!(gw.gap_ev > gap_hf_ev)) {
        std::fprintf(stderr, "FAIL: G0W0 gap %.6f eV not > HF gap %.6f eV\n",
                     gw.gap_ev, gap_hf_ev);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
