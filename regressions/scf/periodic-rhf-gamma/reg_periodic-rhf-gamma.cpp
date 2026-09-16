/// @file regressions/scf/periodic-rhf-gamma/reg_periodic-rhf-gamma.cpp
/// @brief H₂ / STO-3G Γ-point periodic RHF in a 20 Å cube.
///
/// Public API only: `RHF{STO3G}` + `periodic = true` + `run(frame)`.
/// The 20 Å cell is larger than the STO-3G pair reach, so the truncated
/// lattice has only T = 0 and the energy equals molecular RHF. Oracle
/// is PySCF 2.14.0 `pbc.scf.RHF` on the same cell (REFERENCE.md).

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0.0;
    frame.span(kAtom_x)[1] = frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    rhf.periodic = true;
    rhf.tol = 1e-10;
    rhf.max_iter = 50;
    rhf.run(frame);

    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: periodic RHF did not converge\n");
        return 1;
    }

    // PySCF 2.14.0 pbc.scf.RHF / molecular-limit pin — see REFERENCE.md.
    constexpr double kEpinHa = -1.116759307396;
    constexpr double kTolHa = 1.0e-5;

    const double e_ha = rhf.energy();
    const double diff = std::fabs(e_ha - kEpinHa);
    std::printf("E_RHF(periodic Γ / STO-3G H2, 20 A) got=%.16f want=%.16f "
                "|diff|=%.3e tol=%.0e  periodic=%d\n",
                e_ha, kEpinHa, diff, kTolHa, rhf.periodic ? 1 : 0);
    if (!std::isfinite(e_ha) || diff > kTolHa) {
        std::fprintf(stderr, "FAIL: periodic RHF energy off pin\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
