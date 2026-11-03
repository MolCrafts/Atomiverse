// Frozen-input final-band component: no SCF, public Driver or stress calls.
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <string>

using namespace ATV;
using namespace ATV::cpu::pw::internal;
namespace {
using Complex = std::complex<AccReal>;
struct Case {
    int mesh;
    int style; // 0 restricted, 1 populated spin, 2 fixed empty beta
    electronic::Smearing smearing;
    std::string name;
};
cpu::Frame silicon() {
    cpu::Frame frame(2);
    Mat3 h;
    constexpr double columns[3][3] = {
        {.10, 2.70, 2.65}, {2.75, .05, 2.70}, {2.65, 2.75, .15}};
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            h(r, c) = columns[c][r];
    frame.box.set_cell(h);
    constexpr double positions[2][3] = {{.2, .3, .4}, {1.57, 1.66, 1.75}};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_atomic_number)[i] = 14;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] = 28.0855;
        frame.span(kAtom_x)[i] = positions[i][0];
        frame.span(kAtom_y)[i] = positions[i][1];
        frame.span(kAtom_z)[i] = positions[i][2];
    }
    return frame;
}
void expect_same_result(const cpu::pw::PWResult& a,
                        const cpu::pw::PWResult& b) {
    EXPECT_DOUBLE_EQ(a.energy, b.energy);
    EXPECT_DOUBLE_EQ(a.energy_ha, b.energy_ha);
    EXPECT_EQ(a.converged, b.converged);
    EXPECT_EQ(a.iterations, b.iterations);
    EXPECT_DOUBLE_EQ(a.magnetization, b.magnetization);
    EXPECT_DOUBLE_EQ(a.e_fermi, b.e_fermi);
    EXPECT_DOUBLE_EQ(a.n_elec_occ, b.n_elec_occ);
    EXPECT_DOUBLE_EQ(a.e_entropy, b.e_entropy);
    EXPECT_EQ(a.n_kpts, b.n_kpts);
    EXPECT_EQ(a.n_kpts_local, b.n_kpts_local);
}
template <class Array>
void expect_same_array(const Array& a, const Array& b) {
    ASSERT_EQ(a.shape(), b.shape());
    EXPECT_TRUE(std::equal(a.begin(), a.end(), b.begin()));
}
class PwFinalState : public ::testing::TestWithParam<Case> {};
TEST_P(PwFinalState, PreservesReferenceAndFillsOneFrozenHamiltonianSweep) {
    const auto c = GetParam();
    const auto frame = silicon();
    auto potential = std::make_shared<cpu::pw::GthPseudopotential>(
        molrs::Element::Si, "GTH-PBE-q4");
    cpu::pw::PW method{potential, 12, c.mesh, cpu::xc::XCType::LSDA};
    method.tol = 1e-10;
    method.spin_polarized = c.style != 0;
    method.multiplicity = c.style == 2 ? 9 : 1;
    method.smearing = c.smearing;
    method.degauss = .08; // Ry; finite-temperature component fixture only.
    method.extra_bands = 2;
    PWProblem problem;
    setup_pw_problem(frame, *potential, 12, c.mesh, problem);
    ASSERT_EQ(problem.n_elec, 8);
    ASSERT_EQ(problem.potential->n_proj(), 3);
    if (c.style) {
        problem.ws.nspin = 2;
        problem.ws.rho_spin_G.resize(2);
        problem.ws.rho_spin_G[0] = (c.style == 2 ? 1. : .65) * problem.ws.rho_G;
        problem.ws.rho_spin_G[1] = (c.style == 2 ? 0. : .35) * problem.ws.rho_G;
    }
    // Stale last-k orbital/potential scratch must neither be treated as the
    // accepted all-k eigenset nor be overwritten during derivative extraction.
    problem.ws.wfc = xt::ones<Complex>({std::size_t(1), std::size_t(1)});
    problem.ws.eigenvalues = xt::ones<AccReal>({std::size_t(1)});
    problem.ws.V_H_G.fill(7);
    problem.ws.V_xc_G.fill(-5);
    // These stale caches are audit sentinels, not initialized physical inputs.
    problem.ws.rho_G_old.fill(Complex(.75, -.25));
    problem.ws.V_eff_G.fill(Complex(1.25, -.5));
    problem.ws.T_G.fill(3.5);
    const auto rho = problem.ws.rho_G, old = problem.ws.rho_G_old,
               vh = problem.ws.V_H_G, vxc = problem.ws.V_xc_G,
               veff = problem.ws.V_eff_G, coeff = problem.ws.wfc;
    const auto eigenvalues = problem.ws.eigenvalues;
    const auto spin_rho = problem.ws.rho_spin_G;
    const auto* binding = problem.ws.gvec;
    const auto* fft = problem.ws.fft.get();
    const int ng = problem.ws.n_g;
    cpu::pw::PWResult reference;
    reference.energy = 17;
    reference.energy_ha = 8.5;
    reference.iterations = 7;
    reference.converged = false;
    reference.e_fermi = .2;
    reference.e_entropy = -.4;
    reference.n_elec_occ = 6.25;
    reference.magnetization = .3;
    reference.n_kpts = 11;
    reference.n_kpts_local = 3;
    const auto state =
        extract_final_bands(problem, method, reference, c.style == 2 ? 8 : 6);
    expect_same_result(state.reference_result, reference);
    EXPECT_EQ(state.reference_density_G,
              std::vector<Complex>(rho.begin(), rho.end()));
    ASSERT_EQ(state.reference_spin_density_G.size(), spin_rho.size());
    for (std::size_t spin = 0; spin < spin_rho.size(); ++spin)
        EXPECT_EQ(
            state.reference_spin_density_G[spin],
            std::vector<Complex>(spin_rho[spin].begin(), spin_rho[spin].end()));
    EXPECT_EQ(problem.ws.gvec, binding);
    EXPECT_EQ(problem.ws.fft.get(), fft);
    EXPECT_EQ(problem.ws.n_g, ng);
    expect_same_array(problem.ws.rho_G, rho);
    expect_same_array(problem.ws.rho_G_old, old);
    expect_same_array(problem.ws.V_H_G, vh);
    expect_same_array(problem.ws.V_xc_G, vxc);
    expect_same_array(problem.ws.V_eff_G, veff);
    expect_same_array(problem.ws.wfc, coeff);
    expect_same_array(problem.ws.eigenvalues, eigenvalues);
    ASSERT_EQ(problem.ws.rho_spin_G.size(), spin_rho.size());
    for (std::size_t s = 0; s < spin_rho.size(); ++s)
        expect_same_array(problem.ws.rho_spin_G[s], spin_rho[s]);

    const int nk = c.mesh * c.mesh * c.mesh;
    ASSERT_EQ(state.local_blocks.size(), std::size_t(nk * (c.style ? 2 : 1)));
    EXPECT_EQ(state.local_band_solves, nk * (c.style == 1 ? 2 : 1));
    double number = 0, magnetization = 0, min_f = 0;
    for (const auto& block : state.local_blocks) {
        ASSERT_GE(block.ik, 0);
        ASSERT_LT(block.ik, nk);
        EXPECT_DOUBLE_EQ(block.k_weight, problem.kg.weights[block.ik]);
        ASSERT_EQ(block.occupations.size(), block.epsilon_ha.size());
        ASSERT_EQ(block.coefficients.dimension(), 2);
        EXPECT_EQ(block.coefficients.shape(0), block.occupations.size());
        EXPECT_EQ(block.coefficients.shape(1), problem.gv_k[block.ik].g.size());
        EXPECT_TRUE(
            std::is_sorted(block.epsilon_ha.begin(), block.epsilon_ha.end()));
        double orthogonality_error = 0;
        for (std::size_t a = 0; a < block.occupations.size(); ++a)
            for (std::size_t b = 0; b < block.occupations.size(); ++b) {
                Complex overlap{};
                for (std::size_t g = 0; g < block.coefficients.shape(1); ++g)
                    overlap += std::conj(block.coefficients(a, g))
                               * block.coefficients(b, g);
                orthogonality_error =
                    std::max(orthogonality_error,
                             std::abs(overlap - (a == b ? 1. : 0.)));
            }
        EXPECT_LT(orthogonality_error, 1e-10);
        for (auto f : block.occupations) {
            number += block.k_weight * f;
            magnetization += (block.spin ? -1 : 1) * block.k_weight * f;
            min_f = std::min(min_f, double(f));
        }
        if (c.smearing == electronic::Smearing::none)
            for (std::size_t b = 0; b < block.occupations.size(); ++b)
                EXPECT_DOUBLE_EQ(block.occupations[b],
                                 b < std::size_t(c.style == 2 ? 8 : 4)
                                     ? (c.style ? 1 : 2)
                                     : 0);
    }
    EXPECT_NEAR(number, 8, 1e-7);
    EXPECT_NEAR(state.auxiliary_electron_count, number, 1e-10);
    EXPECT_LT(state.max_local_eigen_residual_ha, 1e-8);
    ASSERT_EQ(state.orbital_density_G.size(), problem.gv0.g.size());
    EXPECT_NEAR(state.orbital_density_G[0].real() * problem.ws.omega, 8, 1e-7);
    if (c.style) {
        ASSERT_EQ(state.orbital_spin_density_G.size(), 2);
        const double density_magnetization =
            (state.orbital_spin_density_G[0][0]
             - state.orbital_spin_density_G[1][0])
                .real()
            * problem.ws.omega;
        EXPECT_NEAR(density_magnetization, magnetization, 1e-7);
        EXPECT_NEAR(density_magnetization, state.auxiliary_magnetization, 1e-7);
        if (c.smearing != electronic::Smearing::none)
            EXPECT_GT(std::abs(density_magnetization), 1e-4)
                << "Shared mu must allow M to change from the fixed-filling "
                   "seed";
        if (c.style == 2)
            for (auto value : state.orbital_spin_density_G[1])
                EXPECT_EQ(value, Complex{});
    } else {
        EXPECT_TRUE(state.orbital_spin_density_G.empty());
        EXPECT_DOUBLE_EQ(state.auxiliary_magnetization, 0);
    }
    if (c.smearing == electronic::Smearing::methfessel_paxton) {
        EXPECT_LT(min_f, -1e-8); // Detect forbidden positive-only filling.
        EXPECT_LT(state.auxiliary_entropy, 0); // Signed generalized entropy.
    } else if (c.smearing == electronic::Smearing::none) {
        EXPECT_DOUBLE_EQ(state.auxiliary_mu_ha, 0);
        EXPECT_DOUBLE_EQ(state.auxiliary_entropy, 0);
    } else {
        EXPECT_GT(state.auxiliary_entropy, 0);
    }
}
std::vector<Case> cases() {
    std::vector<Case> result;
    const char* names[] = {"Fixed", "FD", "MP", "Gaussian"};
    for (int mesh : {1, 2})
        for (int style = 0; style < 3; ++style)
            for (auto smearing :
                 {electronic::Smearing::none, electronic::Smearing::fermi_dirac,
                  electronic::Smearing::gaussian,
                  electronic::Smearing::methfessel_paxton}) {
                if (style == 2 && smearing != electronic::Smearing::none)
                    continue;
                result.push_back({mesh, style, smearing,
                                  std::string(style == 0   ? "RKS"
                                              : style == 1 ? "UKS"
                                                           : "EmptyBeta")
                                      + "_" + names[int(smearing)] + "_K"
                                      + std::to_string(mesh)});
            }
    return result;
}
INSTANTIATE_TEST_SUITE_P(FrozenInput, PwFinalState,
                         ::testing::ValuesIn(cases()),
                         [](const auto& p) { return p.param.name; });
} // namespace
