/// @file examples/09_gpw.cpp
/// @brief GPW DFT single-point energy for an isolated water molecule.
///
/// The third electronic-structure family in these tutorials: GPW (Gaussian and
/// Plane Waves). The Kohn-Sham problem is solved in a contracted Gaussian basis
/// (like cpu::scf) while the Hartree + XC potentials are evaluated on a uniform
/// real-space grid via FFT (like cpu::pw). This is the method behind CP2K's
/// Quickstep, so it reproduces CP2K's numbers for the same input.
///
/// This mirrors CP2K's getting-started h2o.inp: isolated H2O, PBE,
/// DZVP-MOLOPT-GTH basis + GTH-PBE pseudopotentials, Martyna-Tuckerman isolated
/// electrostatics. The user-facing component is cpu::gpw::GPWEvaluator (role
/// Evaluator, capability Energy), run through cpu::single_point exactly like
/// cpu::pw::PWEvaluator and cpu::scf::SCF.
///
/// Build:
///   cmake -B build -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
///                  -DATV_BUILD_EXAMPLES=ON                 # needs FFTW3
///   cmake --build build --target 09_gpw
///
/// Run (no external data needed — the GTH/MOLOPT tables are in-tree):
///   ./build/examples/09_gpw
///
/// CP2K reference total energy for this input: -17.21233714 Hartree.

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "atomiverse/cpu/driver/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

/// @brief Water molecule from the CP2K h2o.inp tutorial (coordinates in Å).
cpu::Frame make_water() {
    cpu::Frame f(3);
    const int z[3] = {8, 1, 1};
    const double x[3] = {5.0000, 5.7586, 4.2414};
    const double y[3] = {5.0000, 5.0000, 5.0000};
    const double zc[3] = {5.0000, 5.5043, 5.5043};
    for (int i = 0; i < 3; ++i) {
        f.span(ATV::kAtom_type)[i] = z[i];
        f.span(ATV::kAtom_x)[i] = x[i];
        f.span(ATV::kAtom_y)[i] = y[i];
        f.span(ATV::kAtom_z)[i] = zc[i];
    }
    f.simbox = cpu::Box::cubic(10.0); // 10 Å box
    return f;
}

} // namespace

int main() {

    cpu::Frame frame = make_water();

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GpwEvaluatorConfig cfg;
    cfg.cutoff_ry = 400.0;
    cfg.xc = cpu::gpw::XcKind::PBE;
    cfg.nonperiodic = true; // isolated molecule (Martyna-Tuckerman)
    cfg.kinds = {O, H};

    cpu::gpw::GPWEvaluator eval(cfg);
    cpu::single_point(frame, eval);

    const double e_ha = eval.last_energy_hartree();
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "GPW H2O single-point (PBE, DZVP-MOLOPT-GTH, MT)\n"
              << "  converged : "
              << (eval.last_result().converged ? "yes" : "no") << " in "
              << eval.last_result().iters << " iters\n"
              << "  E_total   : " << e_ha << " Hartree   ("
              << frame.meta.get(ATV::kEnergy) << " eV)\n"
              << "  CP2K ref  : -17.21233714 Hartree\n"
              << "  |dE|      : " << std::abs(e_ha + 17.21233714458516)
              << " Hartree\n";
    return 0;
}
