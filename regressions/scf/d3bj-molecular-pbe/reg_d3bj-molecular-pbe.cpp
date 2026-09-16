/// @file regressions/scf/d3bj-molecular-pbe/reg_d3bj-molecular-pbe.cpp
/// @brief H₂O molecular DFT-D3 (BJ, PBE params) E_disp vs offline PySCF pin.
///
/// Out-of-library consumer: public `cpu/scf/dftd3.h` + `cpu/frame/frame.h`
/// only. Geometry and oracle energy are pinned in REFERENCE.md
/// (PySCF 2.14.0 + pyscf-dispersion offline; no live import in ctest).

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dftd3.h"

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

    // PySCF 2.14.0 + pyscf-dispersion pin (REFERENCE.md): d3bj / water / PBE.
    constexpr double kEDispRef = -0.0003594686557095473; // Ha
    constexpr double kTol = 1.0e-6;                      // Ha

    const double got = cpu::scf::D3BJ{}.energy(frame);

    if (!std::isfinite(got) || std::abs(got - kEDispRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_disp D3BJ/PBE(H2O) got=%.12e want=%.12e "
                     "tol=%.1e Ha\n",
                     got, kEDispRef, kTol);
        return 1;
    }

    std::printf("PASS: E_disp D3BJ/PBE(H2O) = %.12e Ha "
                "(want %.12e +/- %.1e) [PySCF 2.14.0 offline pin]\n",
                got, kEDispRef, kTol);
    return 0;
}
