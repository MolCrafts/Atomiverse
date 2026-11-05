/// @file tests/cuda/gpw/test_gpw_forces.cu
/// @brief Unit: CUDA GPW device force kernels vs host twins + private state guards.
///
/// UNIT SCOPE: ionic HF / local-PP HF kernels on synthetic grids; public
/// private supplied-state force fail-closed guards. Full H₂ dual-gate lives in
/// regressions/gpw/rks — never under ctest as e2e SCF.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#include "atomiverse/cuda/gpw/internal/gpw_access.cuh"
#include "atomiverse/cuda/gpw/internal/gpw_forces.cuh"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::make_uniform_grid;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::internal::GpwAtom;
using ATV::cpu::xc::XCType;
using ATV::cuda::gpw::internal::CollocateGridDesc;
using ATV::cuda::gpw::internal::DeviceForceAtom;
using ATV::cuda::gpw::internal::kMaxCloc;

namespace {

void check_cuda(cudaError_t e) {
    ASSERT_EQ(e, cudaSuccess) << cudaGetErrorString(e);
}

template <typename T>
T* d_alloc(std::size_t n) {
    T* p = nullptr;
    check_cuda(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

CollocateGridDesc to_coll(const UniformGrid& g) {
    CollocateGridDesc d;
    for (int i = 0; i < 3; ++i) {
        d.n[i] = g.n[i];
        d.L[i] = g.L[i];
        d.origin[i] = g.origin[i];
    }
    return d;
}

// Host twin of ionic HF (matches assemble_gpw_forces ionic term).
void host_ionic_hf(const UniformGrid& grid, const std::vector<GpwAtom>& atoms,
                   const std::vector<double>& v_es, bool periodic,
                   std::vector<std::array<double, 3>>& forces) {
    const long ng = grid.npts();
    const double wgt = grid.weight();
    for (std::size_t A = 0; A < atoms.size(); ++A) {
        const GpwAtom& Aatom = atoms[A];
        const double s2 = Aatom.rloc * Aatom.rloc;
        if (!(s2 > 0.0) || Aatom.z_val == 0.0)
            continue;
        const double norm = Aatom.z_val * std::pow(2.0 * M_PI * s2, -1.5);
        std::array<double, 3> acc = {0.0, 0.0, 0.0};
        for (long g = 0; g < ng; ++g) {
            double r[3];
            grid.point(g, r);
            double d[3] = {r[0] - Aatom.pos[0], r[1] - Aatom.pos[1],
                           r[2] - Aatom.pos[2]};
            if (periodic)
                for (int k = 0; k < 3; ++k)
                    d[k] -= grid.L[k] * std::nearbyint(d[k] / grid.L[k]);
            const double gauss =
                norm
                * std::exp(-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                           / (2.0 * s2));
            const double vg = v_es[static_cast<std::size_t>(g)] * gauss / s2;
            for (int k = 0; k < 3; ++k)
                acc[k] += vg * d[k];
        }
        for (int k = 0; k < 3; ++k)
            forces[A][k] += wgt * acc[k];
    }
}

} // namespace

TEST(CudaGpwForces, IonicHfKernelMatchesHost) {
    const double cell[3] = {6.0, 6.0, 6.0};
    UniformGrid grid = make_uniform_grid(cell, 40.0);
    const long ng = grid.npts();
    ASSERT_GT(ng, 0);

    // One H-like Gaussian core off-center so force is nonzero.
    GpwAtom a;
    a.z_val = 1.0;
    a.rloc = 0.2;
    a.pos[0] = 2.7;
    a.pos[1] = 3.0;
    a.pos[2] = 3.0;
    std::vector<GpwAtom> atoms = {a};

    // Synthetic V_es = x (linear → nonzero ionic force).
    std::vector<double> v_es(static_cast<std::size_t>(ng));
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        v_es[static_cast<std::size_t>(g)] = r[0];
    }

    std::vector<std::array<double, 3>> F_host(1, {0.0, 0.0, 0.0});
    host_ionic_hf(grid, atoms, v_es, true, F_host);

    DeviceForceAtom ha{};
    ha.pos[0] = a.pos[0];
    ha.pos[1] = a.pos[1];
    ha.pos[2] = a.pos[2];
    ha.z_val = a.z_val;
    ha.rloc = a.rloc;
    ha.n_cloc = 0;

    double* d_ves = d_alloc<double>(static_cast<std::size_t>(ng));
    double* d_f = d_alloc<double>(3);
    DeviceForceAtom* d_a = d_alloc<DeviceForceAtom>(1);
    check_cuda(cudaMemcpy(d_ves, v_es.data(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_a, &ha, sizeof(ha), cudaMemcpyHostToDevice));
    check_cuda(cudaMemset(d_f, 0, 3 * sizeof(double)));

    ATV::cuda::gpw::internal::accumulate_ionic_hf_forces_device(
        to_coll(grid), d_a, 1, d_ves, true, d_f);
    check_cuda(cudaDeviceSynchronize());

    double F_dev[3] = {0, 0, 0};
    check_cuda(
        cudaMemcpy(F_dev, d_f, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_ves);
    cudaFree(d_f);
    cudaFree(d_a);

    // Absolute tol: small grid, analytic Gaussian — should be tight.
    EXPECT_NEAR(F_dev[0], F_host[0][0], 1e-10);
    EXPECT_NEAR(F_dev[1], F_host[0][1], 1e-10);
    EXPECT_NEAR(F_dev[2], F_host[0][2], 1e-10);
    EXPECT_GT(std::abs(F_host[0][0]), 1e-8); // non-vacuous
}

TEST(CudaGpwForces, LocalPpHfKernelMatchesHost) {
    const double cell[3] = {6.0, 6.0, 6.0};
    UniformGrid grid = make_uniform_grid(cell, 40.0);
    const long ng = grid.npts();

    GpwAtom a;
    a.z_val = 1.0;
    a.rloc = 0.2;
    a.pos[0] = 2.8;
    a.pos[1] = 3.0;
    a.pos[2] = 3.0;
    a.cloc = {-4.0, 0.6}; // typical GTH-ish
    std::vector<GpwAtom> atoms = {a};

    // Synthetic density offset from nucleus.
    std::vector<double> rho(static_cast<std::size_t>(ng), 0.0);
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        const double dx = r[0] - 3.2, dy = r[1] - 3.0, dz = r[2] - 3.0;
        rho[static_cast<std::size_t>(g)] =
            std::exp(-(dx * dx + dy * dy + dz * dz) / 0.5);
    }

    std::vector<std::array<double, 3>> F_host(1, {0.0, 0.0, 0.0});
    ATV::cpu::gpw::add_local_pp_hf_forces(grid, atoms, true, rho, F_host);

    DeviceForceAtom ha{};
    ha.pos[0] = a.pos[0];
    ha.pos[1] = a.pos[1];
    ha.pos[2] = a.pos[2];
    ha.z_val = a.z_val;
    ha.rloc = a.rloc;
    ha.n_cloc = static_cast<int>(a.cloc.size());
    for (int k = 0; k < ha.n_cloc && k < kMaxCloc; ++k)
        ha.cloc[k] = a.cloc[static_cast<std::size_t>(k)];

    double* d_rho = d_alloc<double>(static_cast<std::size_t>(ng));
    double* d_f = d_alloc<double>(3);
    DeviceForceAtom* d_a = d_alloc<DeviceForceAtom>(1);
    check_cuda(cudaMemcpy(d_rho, rho.data(),
                          static_cast<size_t>(ng) * sizeof(double),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_a, &ha, sizeof(ha), cudaMemcpyHostToDevice));
    check_cuda(cudaMemset(d_f, 0, 3 * sizeof(double)));

    ATV::cuda::gpw::internal::accumulate_local_pp_hf_forces_device(
        to_coll(grid), d_a, 1, d_rho, true, d_f);
    check_cuda(cudaDeviceSynchronize());

    double F_dev[3] = {0, 0, 0};
    check_cuda(
        cudaMemcpy(F_dev, d_f, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_rho);
    cudaFree(d_f);
    cudaFree(d_a);

    EXPECT_NEAR(F_dev[0], F_host[0][0], 1e-10);
    EXPECT_NEAR(F_dev[1], F_host[0][1], 1e-10);
    EXPECT_NEAR(F_dev[2], F_host[0][2], 1e-10);
    EXPECT_GT(std::abs(F_host[0][0]), 1e-8);
}

namespace {
ATV::cpu::Frame h2_force_frame() {
    ATV::cpu::Frame frame(2);
    frame.box = ATV::cpu::Box::cubic(12 * ATV::units::kBohrInAngstrom);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_atomic_number)[i] = 1;
        frame.span(ATV::kAtom_x)[i] = (5.5 + i) * ATV::units::kBohrInAngstrom;
        frame.span(ATV::kAtom_y)[i] = 6 * ATV::units::kBohrInAngstrom;
        frame.span(ATV::kAtom_z)[i] = 6 * ATV::units::kBohrInAngstrom;
    }
    return frame;
}
ATV::cuda::gpw::GpwDft h_force_method() {
    return {
        {{molrs::Element::H, GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
          GthPotential(molrs::Element::H, "GTH-PADE-q1")}},
        80.0,
        XCType::PADE};
}
} // namespace

TEST(CudaGpwForces, FreeFnRejectsMissingGrids) {
    auto frame = h2_force_frame();
    auto method = h_force_method();
    method.n_occ = 1;
    method.build(frame);
    ATV::cuda::gpw::GpwResult scf;
    scf.converged = true;
    scf.P = xt::zeros<double>({2, 2});
    scf.C = xt::zeros<double>({2, 2});
    scf.eps = {0.0, 0.0};
    // Unit-only state injection: no public Frame entry accepts missing grids.
    const std::vector<GpwAtom> atoms;
    EXPECT_THROW(ATV::cuda::gpw::internal::GpwDftAccess::forces_from_state(
                     method, atoms, scf),
                 std::runtime_error);
}

TEST(CudaGpwForces, UksDoesNotHitFirstSliceReject) {
    auto frame = h2_force_frame();
    auto method = h_force_method();
    method.n_alpha = 1;
    method.n_beta = 0;
    method.build(frame);
    ATV::cuda::gpw::GpwResult scf;
    scf.converged = true;
    const std::vector<GpwAtom> atoms;
    try {
        (void)ATV::cuda::gpw::internal::GpwDftAccess::forces_from_state(
            method, atoms, scf);
        FAIL() << "expected host-assembly throw";
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("UKS forces not on device"), std::string::npos)
            << msg;
    }
}

namespace {
struct ResidentTestBuffers {
    std::vector<double*> pointers;
    ~ResidentTestBuffers() {
        for (auto* pointer : pointers)
            cudaFree(pointer);
    }
    double* upload(const std::vector<double>& source, cudaStream_t stream) {
        double* pointer = nullptr;
        const auto error = cudaMalloc(&pointer, source.size() * sizeof(double));
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        pointers.push_back(pointer);
        check_cuda(cudaMemcpyAsync(pointer, source.data(),
                                   source.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        check_cuda(cudaStreamSynchronize(stream));
        return pointer;
    }
};
struct ResidentTestStream {
    cudaStream_t stream = nullptr;
    ResidentTestStream() {
        const auto error = cudaStreamCreate(&stream);
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }
    ~ResidentTestStream() {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
};
CollocateGridDesc resident_test_grid() {
    CollocateGridDesc grid;
    for (int k = 0; k < 3; ++k) {
        grid.n[k] = 2;
        grid.L[k] = 2;
        grid.origin[k] = 0;
    }
    grid.periodic = false;
    return grid;
}
} // namespace

TEST(CudaGpwForces, ResidentSparseAoAndGridContractionsUseAcceptedDeviceState) {
    using ATV::cuda::gpw::internal::AoForceEntry;
    using ATV::cuda::gpw::internal::GpwForceWorkspace;
    ResidentTestStream execution;
    ResidentTestBuffers scientific;
    // A two-AO sparse grid with a derivative point outside the chi support.
    // Its huge derivative value must be ignored by g2a, as in the old kernel.
    ATV::cpu::gpw::CompactBasisGrid cbg, cbgd;
    cbg.npts = cbgd.npts = 8;
    cbg.gidx = {1, 6};
    cbg.off = {0, 2, 4};
    cbg.fu = {0, 1, 0, 1};
    cbg.fv = {.8, .3, -.2, .7};
    cbgd.gidx = {1, 4, 6};
    cbgd.off = {0, 2, 3, 5};
    cbgd.fu = {0, 4, 2, 1, 5};
    cbgd.fv = {.1, -.2, 100, .05, .12};
    std::vector<int> owner = {0, 0};
    std::vector<DeviceForceAtom> atoms(1); // No ionic/local operator terms.
    std::vector<AoForceEntry> entries = {
        {0, 0, .5, -.3}, {1, 1, -.4, .2}, {2, 3, 0, -.6}, {0, 1, 1.2, 0}};
    std::vector<double> P = {1, .2, .2, .7}, W = {.9, -.15, -.15, .3};
    std::vector<double> veff(8, 0), v_es(8, 0);
    veff[1] = .7;
    veff[6] = -.4;
    auto* d_P = scientific.upload(P, execution.stream);
    auto* d_W = scientific.upload(W, execution.stream);
    auto* d_veff = scientific.upload(veff, execution.stream);
    auto* d_v_es = scientific.upload(v_es, execution.stream);
    GpwForceWorkspace workspace;
    workspace.build(resident_test_grid(), cbg, cbgd, owner, atoms, entries, {},
                    execution.stream);
    // Host mirrors no longer contain usable science or even geometry. The
    // force body must consume its immutable cold tables and device P/W/grids.
    const double poison = std::numeric_limits<double>::quiet_NaN();
    for (auto* host : {&P, &W, &veff, &v_es, &cbg.fv, &cbgd.fv})
        std::fill(host->begin(), host->end(), poison);
    for (auto& entry : entries)
        entry.overlap = entry.one_body = poison;
    std::fill(owner.begin(), owner.end(), -1);
    workspace.evaluate(d_P, d_W, d_veff, d_v_es, true);
    double actual_force[3];
    std::vector<double> rho(8);
    check_cuda(cudaMemcpyAsync(actual_force, workspace.d_forces(),
                               sizeof(actual_force), cudaMemcpyDeviceToHost,
                               execution.stream));
    check_cuda(cudaMemcpyAsync(rho.data(), workspace.d_rho(),
                               rho.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, execution.stream));
    check_cuda(cudaStreamSynchronize(execution.stream));
    // Independent scalar matrix products for the predeclared tiny fixture:
    // AO=(−.03,.10,−.42); grid=(−.1204,.1012,.0432).
    EXPECT_NEAR(actual_force[0], -.1504, 1e-13);
    EXPECT_NEAR(actual_force[1], .2012, 1e-13);
    EXPECT_NEAR(actual_force[2], -.3768, 1e-13);
    for (int point = 0; point < 8; ++point)
        EXPECT_NEAR(rho[point],
                    point == 1   ? .799
                    : point == 6 ? .327
                                 : 0,
                    1e-14);
    // evaluate() may reuse the same immutable geometry for another resident
    // call; force scratch is reset, so the second result must not double.
    workspace.evaluate(d_P, d_W, d_veff, d_v_es, true);
    check_cuda(cudaMemcpyAsync(actual_force, workspace.d_forces(),
                               sizeof(actual_force), cudaMemcpyDeviceToHost,
                               execution.stream));
    check_cuda(cudaStreamSynchronize(execution.stream));
    EXPECT_NEAR(actual_force[0], -.1504, 1e-13);
    EXPECT_NEAR(actual_force[1], .2012, 1e-13);
    EXPECT_NEAR(actual_force[2], -.3768, 1e-13);
}

TEST(CudaGpwForces, ResidentEmptyAoEntriesAndZeroChargeNeedNoWeightMatrix) {
    using ATV::cuda::gpw::internal::GpwForceWorkspace;
    ResidentTestStream execution;
    ResidentTestBuffers scientific;
    ATV::cpu::gpw::CompactBasisGrid cbg, cbgd;
    cbg.npts = cbgd.npts = 8;
    cbg.gidx = {2};
    cbg.off = {0, 2};
    cbg.fu = {0, 1};
    cbg.fv = {.8, .3};
    cbgd.gidx = {2};
    cbgd.off = {0, 1};
    cbgd.fu = {0};
    cbgd.fv = {.1};
    const std::vector<double> zero_P(4, 0), potential(8, .7), zero_es(8, 0);
    auto* d_P = scientific.upload(zero_P, execution.stream);
    auto* d_veff = scientific.upload(potential, execution.stream);
    auto* d_v_es = scientific.upload(zero_es, execution.stream);
    GpwForceWorkspace workspace;
    workspace.build(resident_test_grid(), cbg, cbgd, {0, 0},
                    std::vector<DeviceForceAtom>(1), {}, {}, execution.stream);
    EXPECT_THROW(workspace.evaluate(nullptr, nullptr, d_veff, d_v_es, true),
                 std::invalid_argument);
    workspace.evaluate(d_P, nullptr, d_veff, d_v_es, true);
    double force[3];
    std::vector<double> rho(8);
    check_cuda(cudaMemcpyAsync(force, workspace.d_forces(), sizeof(force),
                               cudaMemcpyDeviceToHost, execution.stream));
    check_cuda(cudaMemcpyAsync(rho.data(), workspace.d_rho(),
                               rho.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, execution.stream));
    check_cuda(cudaStreamSynchronize(execution.stream));
    for (double value : force)
        EXPECT_DOUBLE_EQ(value, 0);
    for (double value : rho)
        EXPECT_DOUBLE_EQ(value, 0);
}

TEST(CudaGpwForces, ResidentCoreOverlapPairsMatchEnergyDerivativeAndSkipSelf) {
    using ATV::cuda::gpw::internal::GpwForceWorkspace;
    using ATV::gaussian_core::Pair;
    ResidentTestStream execution;
    ResidentTestBuffers scientific;
    ATV::cpu::gpw::CompactBasisGrid cbg, cbgd;
    cbg.npts = cbgd.npts = 8;
    cbg.off = cbgd.off = {0};
    // Cross a CUDA block boundary; each canonical pair updates both atoms.
    constexpr int count = 257;
    std::vector<Pair> pairs(count, {0, 1, {.6, -.8, 1.2}, 2.5, .7});
    // A periodic self-image contributes energy/virial but no atomic force.
    pairs.push_back({0, 0, {2, 0, 0}, 9, .3});
    const std::vector<double> zero_P(1, 0), potential(8, 0);
    auto* d_P = scientific.upload(zero_P, execution.stream);
    auto* d_potential = scientific.upload(potential, execution.stream);
    GpwForceWorkspace workspace;
    workspace.build(resident_test_grid(), cbg, cbgd, {0},
                    std::vector<DeviceForceAtom>(2), {}, pairs,
                    execution.stream);
    const auto geometry_bytes = workspace.geometry_bytes();
    // Device force evaluation cannot use the old host pair geometry.
    for (auto& pair : pairs) {
        pair.atom_a = pair.atom_b = -1;
        pair.charge_product = std::numeric_limits<double>::quiet_NaN();
    }
    workspace.evaluate(d_P, nullptr, d_potential, d_potential, false);
    double force[6];
    check_cuda(cudaMemcpyAsync(force, workspace.d_forces(), sizeof(force),
                               cudaMemcpyDeviceToHost, execution.stream));
    check_cuda(cudaStreamSynchronize(execution.stream));
    auto energy = [](const std::array<double, 3>& displacement) {
        const double r = std::sqrt(displacement[0] * displacement[0]
                                   + displacement[1] * displacement[1]
                                   + displacement[2] * displacement[2]);
        return 2.5 * std::erfc(.7 * r) / r;
    };
    for (int k = 0; k < 3; ++k) {
        std::array<double, 3> plus = {.6, -.8, 1.2}, minus = plus;
        constexpr double h = 1e-5;
        plus[k] += h;
        minus[k] -= h;
        const double expected =
            -count * (energy(plus) - energy(minus)) / (2 * h);
        EXPECT_NEAR(force[k], expected, 1e-8);
        EXPECT_NEAR(force[k + 3], -expected, 1e-8);
        EXPECT_NEAR(force[k] + force[k + 3], 0, 1e-13);
    }
    EXPECT_EQ(workspace.geometry_bytes(), geometry_bytes);
    // Reusing scratch resets the force instead of accumulating twice.
    workspace.evaluate(d_P, nullptr, d_potential, d_potential, false);
    double repeat[6];
    check_cuda(cudaMemcpyAsync(repeat, workspace.d_forces(), sizeof(repeat),
                               cudaMemcpyDeviceToHost, execution.stream));
    check_cuda(cudaStreamSynchronize(execution.stream));
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(repeat[k], force[k], 1e-12);
}
