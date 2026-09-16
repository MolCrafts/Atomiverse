/// @file examples/02_rhf.cu
/// @brief CUDA RHF tutorial — host path uses concrete cpu::scf::RHF; device
///        evaluator comparison is optional when ATV_HAS_CUDA.
///
/// Build: -DATV_BUILD_CUDA=ON -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON

#include <iomanip>
#include <iostream>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

int main() {
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_x)[1] = 0.74;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_z)[1] = 0.0;
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_mass)[1] = 1.008;
    frame.box = cpu::Box::cubic(100.0);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-10};
    rhf.run(frame);
    std::cout << std::setprecision(10) << "CPU RHF E = " << rhf.energy()
              << " Ha (converged=" << rhf.converged() << ")\n";
    // CUDA SCF comparison lives in tests/cuda; this example is the Frame API.
    return rhf.converged() ? 0 : 1;
}
