/// @file regressions/scf/dft-d4-h2o-pbe/reg_dft-d4-h2o-pbe.cpp
/// @brief H₂O molecular DFT-D4 (PBE params) E_disp vs offline PySCF pin.
///
/// Out-of-library consumer: public `cpu/scf/dftd4.h` + `cpu/frame/frame.h`
/// only. Geometry and oracle energy are pinned in REFERENCE.md
/// (PySCF 2.14.0 + pyscf-dispersion offline; no live import in ctest).
///
/// Optional argv[1] = N: evaluate energy N times (for fair_bench multi-iter
/// wall twin; default 1 for the accuracy gate).

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dftd4.h"

int main(int argc, char** argv) {
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

    // PySCF 2.14.0 + pyscf-dispersion pin (REFERENCE.md): d4 / water / PBE.
    constexpr double kEDispRef = -0.00019530371933102527; // Ha
    constexpr double kTol = 1.0e-6;                       // Ha

    const int n_iter = (argc > 1) ? std::atoi(argv[1]) : 1;
    const int n = (n_iter > 0) ? n_iter : 1;

    const cpu::scf::D4 d4{};
    double got = 0.0;
    for (int i = 0; i < n; ++i)
        got = d4.energy(frame);

    if (!std::isfinite(got) || std::abs(got - kEDispRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_disp D4/PBE(H2O) got=%.12e want=%.12e "
                     "tol=%.1e Ha\n",
                     got, kEDispRef, kTol);
        return 1;
    }

    std::printf("PASS: E_disp D4/PBE(H2O) = %.12e Ha "
                "(want %.12e +/- %.1e) [PySCF 2.14.0 offline pin] iters=%d\n",
                got, kEDispRef, kTol, n);
    return 0;
}
