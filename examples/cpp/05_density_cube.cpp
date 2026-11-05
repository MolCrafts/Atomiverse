/// @file examples/cpp/05_density_cube.cpp
/// @brief Write the RHF density and an orbital of H2 as Gaussian cube files.
///
/// One concept: a method's result handle feeds recorder components. The RHF
/// runs in the `evaluate` phase of a one-pass `cpu::Evaluate` driver; two
/// cube writers read its converged density/orbitals in `finalize`.
///
/// Build:  -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON, target `05_density_cube`
/// Output: h2_density.cube, h2_homo.cube in the working directory.

#include <iostream>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/record/density_cube_writer.h"
#include "atomiverse/cpu/record/orbital_cube_writer.h"
#include "atomiverse/cpu/scf/methods.h"
#include "molecules.h"

using namespace ATV;

int main() {
    cpu::Frame frame = examples::h2(0.74);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    rhf.with_gradient(false);
    auto handle = rhf.result_handle();

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(std::move(rhf));
    eval.on(evaluate::Phase::finalize)
        .add(cpu::recorder::DensityCubeWriter{handle, "h2_density.cube"});
    eval.on(evaluate::Phase::finalize)
        .add(cpu::recorder::OrbitalCubeWriter{handle, "h2_homo.cube",
                                              /*mo_index=*/0});
    eval.build();
    eval.run(frame);
    std::cout << "wrote h2_density.cube and h2_homo.cube\n";
    return 0;
}
