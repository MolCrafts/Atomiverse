#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include <cmath>
#include <cstdio>
using namespace ATV;
int main() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0.757;
    f.span(kAtom_y)[1] = 0.586;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = -0.757;
    f.span(kAtom_y)[2] = 0.586;
    f.span(kAtom_z)[2] = 0;
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.box = cpu::Box::cubic(20);
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-9;
    auto mp2 = rhf.mp2(f);
    std::printf("MP2 Ecorr=%.8f Ha Etot=%.8f Ha\n", mp2.e_corr, mp2.e_tot);
    // Textbook: MP2 correlation negative for H2O/STO-3G (~ -0.03 to -0.1 Ha)
    if (!(mp2.e_corr < -1e-4) || !std::isfinite(mp2.e_tot)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
