#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#include "cuda_uks_domain_fixture.h"
#include <cstdio>
#include <fstream>
#include <iomanip>
using namespace ATV;
using pw_domain_fixture::geometry;

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("legacy value output path");
        std::ofstream values(argv[1]);
        values << std::setprecision(17);
        cuda::pw::PW method(std::make_shared<cpu::pw::GthPseudopotential>(
                                molrs::Element::H, "GTH-PBE-q1"),
                            2., 1, cpu::xc::XCType::LSDA);
        method.spin_polarized = true;
        method.multiplicity = 1;
        method.max_iter = 200;
        method.tol = 1e-10;
        method.mix_beta = .2;
        method.pulay_depth = 6;
        method.extra_bands = 0;
        cuda::Evaluate parent(true, true);
        parent.set_evaluator_ref(method);
        parent.build();
        for (double box : {12., 4.}) {
            auto f = geometry(box);
            parent.run(f);
            const auto& r = method.last_result();
            values << box << ' ' << r.energy_ha << ' ' << r.converged << ' '
                   << r.iterations;
            std::printf(
                "LEGACY box_A=%.17g E_Ha=%.17g converged=%d iterations=%d\n",
                box, r.energy_ha, r.converged, r.iterations);
            for (int i = 0; i < 2; ++i)
                for (auto col : {kAtom_fx, kAtom_fy, kAtom_fz}) {
                    double v = f.span(col)[i];
                    if (!std::isfinite(v))
                        throw std::runtime_error("legacy force finite");
                    values << ' ' << v;
                    std::printf("LEGACY_F box=%.17g atom=%d eVA=%.17g\n", box,
                                i, v);
                }
            for (double w : f.meta.get(kStress)) {
                if (!std::isfinite(w))
                    throw std::runtime_error("legacy W finite");
                values << ' ' << w;
                std::printf("LEGACY_W box=%.17g eV=%.17g\n", box, w);
            }
            values << '\n';
        }
        std::puts(
            "PASS LEGACY_REFERENCE accepted1480_current_joint_UKS_legacy=1 "
            "SCF=2 additional_SCF=0");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL LEGACY_REFERENCE %s\n", e.what());
        return 1;
    }
}
