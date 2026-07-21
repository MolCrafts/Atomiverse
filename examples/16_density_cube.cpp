/// @file examples/16_density_cube.cpp
/// @brief Single-point RHF then density/orbital cube writers via result_handle.

#include <iostream>

#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/recorder/density_cube_writer.h"
#include "atomiverse/cpu/recorder/orbital_cube_writer.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/facade/stage.h"

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
    frame.simbox = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
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
