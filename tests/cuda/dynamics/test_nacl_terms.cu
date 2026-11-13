/// @file tests/cuda/dynamics/test_nacl_terms.cu
/// @brief CPU <-> CUDA parity of the molten-NaCl force terms on a jittered
///        configuration, one term set per case: energy, forces and virial
///        of `PairBorn` (with and without the tail), of the Ewald pair
///        (`CoulLong` + `KSpaceEwald`) and of the full driven model, and the
///        curvature each supplies to the configurational temperature —
///        per-atom H·F, Laplacian and Laplacian gradient, and the sums N, D.
///
/// Both drivers take one 1e-9 ps velocity-Verlet step from the same frame
/// (how `ConfigurationalTemperature` is sampled in production), so the two
/// evaluations are at the same configuration to round-off.
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "atomiverse/cpu/dynamics/configurational_temperature.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cuda/dynamics/configurational_temperature.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"

#include "nacl_parity.h"

using namespace ATV;
using nacl_parity::Atom;
using nacl_parity::Terms;

namespace {

/// Relative to the largest reference magnitude of each quantity. FP64: the
/// two backends differ by summation order and the fixed-point force grid.
constexpr double kTol = nacl_parity::kDouble ? 1e-9 : 1e-4;
constexpr double kStep = 1e-9; // [ps]

/// Per-atom curvature rows by tag: H·F, ∇²U, ∇ Σ∇²U.
struct Curvature {
    std::array<double, 3> hf, grad;
    double lap;
};

struct Evaluation {
    cpu::Frame frame;
    std::map<std::int64_t, Curvature> rows;
    double force_sq = 0, laplacian = 0, temperature = 0;
};

Evaluation evaluate_cpu(const Terms& terms) {
    Evaluation out{nacl_fixture::nacl(1500.0, 3)};
    // Borrowed: declared first so that it outlives the driver.
    cpu::ConfigurationalTemperature observer;
    cpu::md::MD sim;
    nacl_parity::mount(sim, terms);
    sim.set_observer_ref(observer);
    sim.set_integrator(cpu::md::VelocityVerlet{kStep});
    sim.build();
    sim.run(out.frame, 1);
    const int n = out.frame.nlocal();
    const auto& a = observer.assembly();
    for (int i = 0; i < n; ++i) {
        Curvature c;
        for (int d = 0; d < 3; ++d) {
            c.hf[d] = a.hessian_force()[d * n + i];
            c.grad[d] = a.laplacian_gradient()[d * n + i];
        }
        c.lap = a.laplacian()[i];
        out.rows[out.frame.span(kAtom_tag)[i]] = c;
    }
    out.force_sq = observer.last().force_sq;
    out.laplacian = observer.last().laplacian;
    out.temperature = observer.last().temperature;
    return out;
}

std::vector<double> download(const double* device, int n) {
    std::vector<double> host(static_cast<std::size_t>(n));
    EXPECT_EQ(cudaMemcpy(host.data(), device, n * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
}

Evaluation evaluate_cuda(const Terms& terms) {
    Evaluation out{nacl_fixture::nacl(1500.0, 3)};
    // Borrowed: declared first so that it outlives the driver.
    cuda::ConfigurationalTemperature observer;
    cuda::md::MD sim;
    nacl_parity::mount(sim, terms);
    sim.set_observer_ref(observer);
    sim.set_integrator(cuda::md::VelocityVerlet{kStep});
    sim.build();
    sim.run(out.frame, 1);
    const int n = out.frame.nlocal();
    const cuda::CurvatureRows& r = observer.assembly().rows();
    const std::vector<double> h[3] = {download(r.hx, n), download(r.hy, n),
                                      download(r.hz, n)};
    const std::vector<double> g[3] = {download(r.gx, n), download(r.gy, n),
                                      download(r.gz, n)};
    const std::vector<double> lap = download(r.lap, n);
    for (int i = 0; i < n; ++i) {
        Curvature c;
        for (int d = 0; d < 3; ++d) {
            c.hf[d] = h[d][i];
            c.grad[d] = g[d][i];
        }
        c.lap = lap[i];
        out.rows[out.frame.span(kAtom_tag)[i]] = c;
    }
    out.force_sq = observer.last().force_sq;
    out.laplacian = observer.last().laplacian;
    out.temperature = observer.last().temperature;
    return out;
}

/// Largest deviation of one curvature field over the atoms, relative to
/// its largest reference magnitude.
template <class Get>
double worst_row(const Evaluation& got, const Evaluation& want, Get get,
                 const char* what) {
    double scale = 0.0, diff = 0.0;
    for (const auto& [tag, w] : want.rows) {
        const Curvature& g = got.rows.at(tag);
        for (int d = 0; d < 3; ++d) {
            scale = std::max(scale, std::abs(get(w, d)));
            diff = std::max(diff, std::abs(get(g, d) - get(w, d)));
        }
    }
    EXPECT_GT(scale, 0.0) << what << " is vacuous";
    return diff / scale;
}

struct Case {
    std::string name;
    Terms terms;
};

class CudaNaClTerms : public testing::TestWithParam<Case> {};

} // namespace

TEST_P(CudaNaClTerms, EnergyForceAndVirialMatchCpu) {
    const Terms& terms = GetParam().terms;
    const Evaluation want = evaluate_cpu(terms);
    const Evaluation got = evaluate_cuda(terms);
    const auto g = nacl_parity::atoms(got.frame);
    const auto w = nacl_parity::atoms(want.frame);
    ASSERT_EQ(g.size(), w.size());
    EXPECT_LT(nacl_parity::worst(g, w, &Atom::f, "force"), kTol);

    const double e_want = want.frame.meta.get(kEnergy);
    ASSERT_GT(std::abs(e_want), 1e-3) << "vacuous energy";
    EXPECT_NEAR(got.frame.meta.get(kEnergy), e_want, kTol * std::abs(e_want));
    const double pair_want = want.frame.meta.get(kEnergyPair);
    EXPECT_NEAR(got.frame.meta.get(kEnergyPair), pair_want,
                kTol * std::abs(e_want));

    const auto& sg = got.frame.meta.get(kStress);
    const auto& sw = want.frame.meta.get(kStress);
    double scale = 0.0;
    for (int k = 0; k < 6; ++k)
        scale = std::max(scale, std::abs(sw[k]));
    ASSERT_GT(scale, 1e-3) << "vacuous virial";
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(sg[k], sw[k], kTol * scale) << "virial " << k;
}

TEST_P(CudaNaClTerms, CurvatureMatchesCpu) {
    const Terms& terms = GetParam().terms;
    const Evaluation want = evaluate_cpu(terms);
    const Evaluation got = evaluate_cuda(terms);
    ASSERT_EQ(got.rows.size(), want.rows.size());
    EXPECT_LT(worst_row(
                  got, want, [](const Curvature& c, int d) { return c.hf[d]; },
                  "H·F"),
              kTol);
    EXPECT_LT(worst_row(
                  got, want, [](const Curvature& c, int) { return c.lap; },
                  "Laplacian"),
              kTol);
    EXPECT_LT(worst_row(
                  got, want,
                  [](const Curvature& c, int d) { return c.grad[d]; },
                  "Laplacian gradient"),
              kTol);
    ASSERT_GT(want.force_sq, 0.0);
    ASSERT_GT(std::abs(want.laplacian), 0.0);
    EXPECT_NEAR(got.force_sq, want.force_sq, kTol * want.force_sq);
    EXPECT_NEAR(got.laplacian, want.laplacian, kTol * std::abs(want.laplacian));
    EXPECT_NEAR(got.temperature, want.temperature,
                kTol * std::abs(want.temperature));
}

INSTANTIATE_TEST_SUITE_P(
    Sets, CudaNaClTerms,
    testing::Values(Case{"Born", {true, false, false, 0.0}},
                    Case{"BornTail", {true, false, true, 0.0}},
                    Case{"Ewald", {false, true, false, 0.0}},
                    Case{"DrivenMelt", {true, true, true, 0.5}}),
    [](const testing::TestParamInfo<Case>& info) { return info.param.name; });

// The field is applied work, not interaction energy: with it on, the force
// the temperature uses is the frame force minus q E, so N and T_C1 are
// those of the undriven melt while the frame force is not.
TEST(CudaNaClTermsDrive, TheFieldIsExcludedFromTheConfigurationalTemperature) {
    const Evaluation free = evaluate_cuda({true, true, false, 0.0});
    const Evaluation driven = evaluate_cuda({true, true, false, 0.5});
    const auto f0 = nacl_parity::atoms(free.frame);
    const auto f1 = nacl_parity::atoms(driven.frame);
    EXPECT_GT(nacl_parity::worst(f1, f0, &Atom::f, "force"), 1e-3)
        << "the field did not act on the frame force";
    EXPECT_NEAR(driven.force_sq, free.force_sq, kTol * free.force_sq);
    EXPECT_NEAR(driven.laplacian, free.laplacian,
                kTol * std::abs(free.laplacian));
}
