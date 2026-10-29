// Gauge invariance of the admitted NC Hamiltonian, without a SCF solve.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/dfpt.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/internal/pw_forces.h"
#include "nc_gauge_fixture.h"
#include <gtest/gtest.h>

namespace {
using namespace nc_gauge_test;
TEST(PWNCGaugeCPU, FullHamiltonianPreservesProjectorNormalization) {
    for (bool nonzero_k : {false, true}) {
        const auto g = modes(nonzero_k);
        const auto c = wave(g);
        const auto original = potential(1, nonzero_k);
        PWProblem ref;
        prepare(ref, original, g, c);
        const auto expected = full_d_h(original, ref, c);
        AccReal nonlocal_norm = 0;
        for (std::size_t i = 0; i < c.size(); ++i)
            nonlocal_norm +=
                std::norm(expected[i] - g.ekin[i % g.count()] * c[i]);
        ASSERT_GT(nonlocal_norm,
                  1e-4); // The old screens must be distinguishable.
        for (AccReal scale : {AccReal(1), AccReal(1e9), AccReal(1e-35)}) {
            SCOPED_TRACE(::testing::Message()
                         << "k=" << nonzero_k << " scale=" << scale);
            const auto d = potential(scale, nonzero_k);
            PWProblem p;
            prepare(p, d, g, c);
            std::vector<C> actual(c.size());
            for (int b = 0; b < 3; ++b)
                pw_apply_h(p.ws, c.data() + b * g.count(),
                           actual.data() + b * g.count());
            for (std::size_t i = 0; i < c.size(); ++i) {
                const AccReal gate = 1e-8 + 1e-7 * std::abs(expected[i]);
                EXPECT_LE(std::abs(actual[i] - expected[i]), gate)
                    << "component " << i;
            }
        }
    }
}

// No SCF or XC point law: force is linear in the supplied occupation when its
// orbital density scales with that occupation. One ion avoids a large ionic
// background obscuring the tiny electronic force. Signed controls test only
// these prepared operators, not a signed-density SCF model.
TEST(PWNCGaugeCPU, FinalBandForcesRetainTinySignedOccupations) {
    const auto g = modes(false);
    const auto c = wave(g);
    PWProblem p;
    prepare(p, potential(1, true), g, c);
    auto& ws = p.ws;
    ws.atom_pos_bohr.resize(1);
    p.kb_proj_k[0].resize(1);
    ws.kb_proj = p.kb_proj_k[0];
    ws.nc = (g.nx / 2 + 1) * g.ny * g.nz;
    ws.eigenvalues = xt::zeros<AccReal>({std::size_t(3)});

    cpu::Frame frame(1);
    const double bohr =
        units::convert(1., units::Length::Bohr, units::Length::Angstrom);
    Mat3 cell;
    for (int axis = 0; axis < 3; ++axis)
        cell(axis, axis) = 5 * bohr;
    frame.box.set_cell(cell);
    frame.span(kAtom_x)[0] = ws.atom_pos_bohr[0][0] * bohr;
    frame.span(kAtom_y)[0] = ws.atom_pos_bohr[0][1] * bohr;
    frame.span(kAtom_z)[0] = ws.atom_pos_bohr[0][2] * bohr;
    frame.span(kAtom_type)[0] = 1;

    DensityBuilder density;
    density.prepare(ws);
    const auto evaluate = [&](double occupation) {
        FinalBandState state;
        FinalBandBlock block;
        block.ik = 0;
        block.k_weight = .37;
        block.coefficients = ws.wfc;
        block.occupations = {occupation, 0, 0};
        state.local_blocks.push_back(block);
        density.begin(ws);
        density.accumulate_kpoint(ws, block.k_weight, 3,
                                  block.occupations.data());
        xt::xarray<C> rho;
        density.extract(ws, rho);
        state.orbital_density_G.assign(rho.begin(), rho.end());
        return pw_forces_from_bands(frame, p, state);
    };
    const auto zero = evaluate(0), full = evaluate(1);
    double scale = 0;
    for (int axis = 0; axis < 3; ++axis)
        scale = std::max(scale, std::abs(full[0][axis] - zero[0][axis]));
    ASSERT_GT(scale, 1e-3);
    for (double occupation : {5e-15, -5e-15, 1e-16}) {
        SCOPED_TRACE(::testing::Message() << "occupation=" << occupation);
        const auto tail = evaluate(occupation);
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_NEAR((tail[0][axis] - zero[0][axis]) / occupation,
                        full[0][axis] - zero[0][axis], 1e-8 + 1e-7 * scale);
    }
}

// Solve a nontrivial projected response with known solution and an independent
// full-D RHS. This checks the public internal DFPT linear solver without a
// ground-state or response SCF. Private finite-q ionic perturbations are
// separately qualified by the opt-in gauge diagnostic.
TEST(PWNCGaugeCPU, SternheimerResponsePreservesProjectorNormalization) {
    const auto g = modes(true);
    const auto c = wave(g);
    const int ng = g.count();
    const C* occupied = c.data() + ng;
    const auto project = [&](std::span<C> x) {
        C overlap{};
        for (int i = 0; i < ng; ++i)
            overlap += std::conj(occupied[i]) * x[i];
        for (int i = 0; i < ng; ++i)
            x[i] -= occupied[i] * overlap;
    };
    std::vector<C> expected(c.begin(), c.begin() + ng);
    project(expected);
    std::vector<C> all(3 * ng);
    std::copy(expected.begin(), expected.end(), all.begin());
    const auto original = potential(1, true);
    PWProblem ref;
    prepare(ref, original, g, c);
    auto rhs = full_d_h(original, ref, all);
    rhs.resize(ng);
    constexpr AccReal eigenvalue = -2;
    for (int i = 0; i < ng; ++i)
        rhs[i] -= eigenvalue * expected[i];
    project(rhs);
    for (AccReal scale : {AccReal(1), AccReal(1e9), AccReal(1e-35)}) {
        SCOPED_TRACE(::testing::Message() << "scale=" << scale);
        const auto d = potential(scale, true);
        PWProblem p;
        prepare(p, d, g, c);
        std::vector<C> response(ng), residual(ng);
        auto* scratch = static_cast<fftw_complex*>(
            fftw_malloc(sizeof(fftw_complex) * p.ws.nr));
        ASSERT_NE(scratch, nullptr);
        std::unique_ptr<fftw_complex, decltype(&fftw_free)> work(scratch,
                                                                 fftw_free);
        const int iterations =
            sternheimer_cg(p.ws, occupied, 1, eigenvalue, rhs.data(),
                           response.data(), 1e-11, 80, work.get());
        ASSERT_LE(iterations, 80);
        pw_apply_h(p.ws, response.data(), residual.data(), work.get());
        for (int i = 0; i < ng; ++i)
            residual[i] -= eigenvalue * response[i];
        project(residual);
        AccReal equation_error = 0, rhs_norm = 0;
        for (int i = 0; i < ng; ++i) {
            EXPECT_LE(std::abs(response[i] - expected[i]),
                      1e-8 + 1e-7 * std::abs(expected[i]));
            equation_error += std::norm(residual[i] - rhs[i]);
            rhs_norm += std::norm(rhs[i]);
        }
        EXPECT_LT(std::sqrt(equation_error / rhs_norm), 2e-11);
    }
}
} // namespace
