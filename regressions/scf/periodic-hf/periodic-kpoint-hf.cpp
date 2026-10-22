/// @file regressions/scf/periodic-kpoint-hf/reg_periodic-kpoint-hf.cpp
/// @brief H₂ / STO-3G KRHF and triplet KUHF, 4 Bohr cube.
///
/// Public API only. The cube is smaller than the STO-3G pair reach.
/// Oracle: PySCF 2.14.0 `pbc.scf.KRHF` / `KUHF`, FFTDF, exxdiv='ewald'
/// (tools/pyscf_oracle/pin_scf.py). See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/krhf.h"
#include "atomiverse/cpu/scf/kuhf.h"
#include "harness/registry.h"

namespace atv_reg_scf_periodic_kpoint_hf {

namespace {

ATV::cpu::Frame h2_cell() {
    using namespace ATV;
    // PySCF cell.unit = 'Bohr': lattice 4 Bohr, bond 1.4 Bohr.
    const double L_A = 4.0 * units::kBohrInAngstrom;
    const double z_A = 1.4 * units::kBohrInAngstrom;
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0.0;
    frame.span(kAtom_x)[1] = frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = z_A;
    frame.box = cpu::Box::cubic(L_A);
    return frame;
}

int gate(const char* name, double got, double want, bool conv, int iters,
         double e_ev) {
    constexpr double kTolHa = 1.0e-6;
    const double diff = std::fabs(got - want);
    std::printf("%s got=%.16f want=%.16f |diff|=%.3e tol=%.0e iters=%d\n", name,
                got, want, diff, kTolHa, iters);
    if (!conv) {
        std::fprintf(stderr, "FAIL: %s did not converge\n", name);
        return 1;
    }
    if (!std::isfinite(got) || diff > kTolHa) {
        std::fprintf(stderr, "FAIL: %s energy off pin\n", name);
        return 1;
    }
    if (std::fabs(e_ev - got * ATV::units::kHartreeInEv) > 1e-8) {
        std::fprintf(stderr, "FAIL: %s kEnergy [eV] does not match energy()\n",
                     name);
        return 1;
    }
    return 0;
}

} // namespace

int run() {
    using namespace ATV;
    int rc = 0;

    {
        cpu::Frame frame = h2_cell();
        cpu::scf::KRHF krhf{cpu::scf::BasisSet::STO3G, {1, 1, 2}, 0, 1e-10, 40};
        krhf.run(frame);
        rc |=
            gate("KRHF 1x1x2", krhf.energy(), -1.3541438798330374,
                 krhf.converged(), krhf.iterations(), frame.meta.get(kEnergy));
    }
    {
        cpu::Frame frame = h2_cell();
        cpu::scf::KRHF krhf{cpu::scf::BasisSet::STO3G, {2, 1, 1}, 0, 1e-10, 40};
        krhf.run(frame);
        rc |=
            gate("KRHF 2x1x1", krhf.energy(), -1.4471748215213558,
                 krhf.converged(), krhf.iterations(), frame.meta.get(kEnergy));
    }
    {
        // PySCF KUHF.nelec with cell.spin = 2 on this 2-point mesh: (3, 1).
        cpu::Frame frame = h2_cell();
        cpu::scf::KUHF kuhf{cpu::scf::BasisSet::STO3G,
                            {2, 1, 1},
                            /*n_alpha=*/3,
                            /*n_beta=*/1,
                            0,
                            1e-10,
                            40};
        kuhf.run(frame);
        rc |=
            gate("KUHF triplet 2x1x1", kuhf.energy(), -0.7658470471886634,
                 kuhf.converged(), kuhf.iterations(), frame.meta.get(kEnergy));
    }
    if (rc == 0)
        std::printf("PASS\n");
    return rc == 0 ? 0 : 1;
}

} // namespace atv_reg_scf_periodic_kpoint_hf

ATV_REGRESSION("scf/periodic-hf", "periodic-kpoint-hf",
               atv_reg_scf_periodic_kpoint_hf::run);
