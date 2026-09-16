#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include <cmath>
#include <cstdio>
using namespace ATV;
int main() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0.1173;
    f.span(kAtom_x)[1] = 0;
    f.span(kAtom_y)[1] = 0.7572;
    f.span(kAtom_z)[1] = -0.4692;
    f.span(kAtom_x)[2] = 0;
    f.span(kAtom_y)[2] = -0.7572;
    f.span(kAtom_z)[2] = -0.4692;
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.box = cpu::Box::cubic(20);
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-9;
    rhf.ri = cpu::scf::RI::JK;
    rhf.aux_basis = cpu::scf::AuxBasis::Def2JKFIT;
    auto m = rhf.mp2(f);
    std::printf("RI-MP2 e_corr=%.8f Ha ri=%d\n", m.e_corr, (int)m.ri);
    // Hard-coded window: H2O/STO-3G MP2 correlation ~ -0.03..-0.1 Ha
    if (!m.ri || !(m.e_corr < -1e-3) || !(m.e_corr > -0.2)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
