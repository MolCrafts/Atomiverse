/// @file regressions/scf/blyp-water-sto3g/reg_blyp-water-sto3g.cpp
/// @brief H₂O / STO-3G restricted BLYP total energy vs offline PySCF pin.
///
/// Out-of-library consumer: public `cpu/scf/methods.h` + `cpu/frame/frame.h`
/// + `core/units.h` only. Geometry, basis, and oracle energy are pinned in
/// REFERENCE.md (PySCF 2.14.0 offline; no live import in ctest).
///
/// Pure GGA path: `XCType::BLYP` = B88 exchange + LYP correlation at unit
/// weights (`hybrid_exchange_fraction == 0`).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // Geometry [Å] — pin_scf.py water preset / REFERENCE.md.
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8; // O
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1; // H
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1; // H
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);

    // PySCF 2.14.0 pin (REFERENCE.md): rks --geom water --basis sto-3g --xc blyp
    constexpr double kEnergyRef = -75.277447482665; // Ha
    constexpr double kTol = 1.0e-6;                 // Ha

    // Pure-GGA metadata: no exact exchange; GGA density-gradient path.
    if (cpu::scf::hybrid_exchange_fraction(cpu::scf::XCType::BLYP) != 0.0) {
        std::fprintf(stderr, "FAIL: BLYP hybrid_exchange_fraction != 0\n");
        return 1;
    }
    if (!cpu::scf::is_gga(cpu::scf::XCType::BLYP)) {
        std::fprintf(stderr, "FAIL: BLYP is_gga == false\n");
        return 1;
    }

    cpu::scf::RKS rks(cpu::scf::XCType::BLYP);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.tol = 1e-10;
    rks.max_iter = 100;
    rks.run(frame);

    if (!rks.converged()) {
        std::fprintf(stderr, "FAIL: RKS/BLYP/STO-3G(H2O) did not converge\n");
        return 1;
    }

    const double got_ha = rks.energy();
    const double from_frame = frame.meta.get(kEnergy) / units::kHartreeInEv;

    if (!std::isfinite(got_ha) || std::abs(got_ha - kEnergyRef) > kTol) {
        std::fprintf(
            stderr,
            "FAIL: E_BLYP/STO-3G(H2O) got=%.12f want=%.12f tol=%.1e Ha\n",
            got_ha, kEnergyRef, kTol);
        return 1;
    }
    // Frame energy must track the method energy (eV ↔ Ha conversion).
    if (!std::isfinite(from_frame) || std::abs(from_frame - got_ha) > 1e-10) {
        std::fprintf(stderr,
                     "FAIL: frame energy %.12f Ha != rks.energy() %.12f Ha\n",
                     from_frame, got_ha);
        return 1;
    }

    std::printf("PASS: E_BLYP/STO-3G(H2O) = %.12f Ha (want %.12f +/- %.1e) "
                "[PySCF 2.14.0 offline pin; pure GGA B88+LYP]\n",
                got_ha, kEnergyRef, kTol);
    return 0;
}
