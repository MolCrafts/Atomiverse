/// @file regressions/forcefield/cuda-gaff-fragment/reg_cuda-gaff-fragment.cu
/// @brief Full CUDA GAFF fragment against independently printed OpenMM pins.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"
#include "atomiverse/cuda/potentials/dihedral_multi_harmonic.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <type_traits>

using namespace ATV;
int main() try {
    constexpr double ev = 1.0 / 23.060547830619026;
    constexpr double xyz[8][3] = {
        {2, 2, 2},     {3.5, 2.1, 2},   {4.2, 3.3, 2.1}, {5.5, 3.5, 3},
        {3.2, 8, 2.7}, {4.7, 8.1, 2.7}, {5.4, 9.3, 2.8}, {6.7, 9.5, 3.7}};
    constexpr double charge[4] = {-0.1, 0.15, -0.2, 0.15};
    cpu::Frame f(8);
    f.box = cpu::Box::cubic(24);
    for (int i = 0; i < 8; ++i) {
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
        f.span(kAtom_charge)[i] = charge[i % 4];
        f.span(kAtom_type)[i] = i % 4 == 2 ? 1 : 0;
        f.span(kAtom_mass)[i] = i % 4 == 2 ? 16 : 12.01;
        f.span(kAtom_tag)[i] = i;
    }
    f.rebuild_tag_index(0, 8);
    for (int o : {0, 4}) {
        for (int i = 0; i < 3; ++i)
            f.add_bond(o + i, o + i + 1, 0);
        f.add_angle(o, o + 1, o + 2, 0);
        f.add_angle(o + 1, o + 2, o + 3, 0);
        f.add_dihedral(o, o + 1, o + 2, o + 3, 0);
    }
    ATV::cuda::Evaluate eval;
    eval.set_neighbor(ATV::cuda::NeighborList{9, 1});
    eval.set_evaluator(ATV::cuda::PairLJCutCoulLong{
        {0.1094 * ev, 0.17 * ev}, {3.39967, 3.000012}, 0.35, 9}
                           .set_special_bonds_amber());
    eval.set_evaluator(ATV::cuda::BondHarmonic{{300.9 * ev}, {1.5375}});
    eval.set_evaluator(
        ATV::cuda::AngleHarmonic{{46.4 * ev}, {109.560047 * units::pi / 180}});
    // .1*(1-cos(2 phi)) + .383*(1+cos(3 phi)), exactly the oracle input.
    eval.set_evaluator(ATV::cuda::DihedralMultiHarmonic{
        {{.583 * ev, -1.149 * ev, -.2 * ev, 1.532 * ev, 0}}});
    eval.on(evaluate::Phase::evaluate)
        .add(ATV::cuda::KSpacePME{0.35, 1, 5}, 100);
    eval.build();
    eval.run(f);
    std::ifstream pin(ATV_GAFF_ORACLE_PATH);
    pin.exceptions(std::ios::failbit | std::ios::badbit);
    double expected_energy;
    pin >> expected_energy;
    const double energy_error = std::abs(f.meta.get(kEnergy) - expected_energy);
    double force_error = 0;
    for (int i = 0; i < 8; ++i) {
        double expected[3];
        pin >> expected[0] >> expected[1] >> expected[2];
        const double actual[3] = {f.span(kAtom_fx)[i], f.span(kAtom_fy)[i],
                                  f.span(kAtom_fz)[i]};
        for (int d = 0; d < 3; ++d) {
            if (!std::isfinite(actual[d]))
                return 1;
            force_error =
                std::max(force_error, std::abs(actual[d] - expected[d]));
        }
    }
    constexpr double tolerance =
        std::is_same_v<ATV::cuda::real_t, float> ? 5e-4 : 5e-5;
    std::printf("CUDA GAFF: E=%.15g eV energy_error=%.6g eV "
                "max_force_error=%.6g eV/A\n",
                f.meta.get(kEnergy), energy_error, force_error);
    return std::isfinite(energy_error) && energy_error < tolerance
                   && force_error < tolerance
               ? 0
               : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cuda-gaff-fragment: %s\n", e.what());
    return 1;
}
