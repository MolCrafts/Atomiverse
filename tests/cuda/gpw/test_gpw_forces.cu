/// @file tests/cuda/gpw/test_gpw_forces.cu
/// @brief Unit: CUDA GPW device force kernels vs host twins + free-fn shape.
///
/// UNIT SCOPE: ionic HF / local-PP HF kernels on synthetic grids; public
/// free-fn force path fail-closed guards. Full H₂ dual-gate lives in
/// regressions/gpw/cuda-gpw-device-forces — never under ctest as e2e SCF.
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#include "atomiverse/cuda/gpw/internal/gpw_forces.cuh"

#include <cmath>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::XcKind;
using ATV::cpu::gpw::make_uniform_grid;
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
    check_cuda(cudaMemcpy(F_dev, d_f, 3 * sizeof(double),
                          cudaMemcpyDeviceToHost));
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
    a.cloc = { -4.0, 0.6 }; // typical GTH-ish
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
    check_cuda(cudaMemcpy(F_dev, d_f, 3 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_rho);
    cudaFree(d_f);
    cudaFree(d_a);

    EXPECT_NEAR(F_dev[0], F_host[0][0], 1e-10);
    EXPECT_NEAR(F_dev[1], F_host[0][1], 1e-10);
    EXPECT_NEAR(F_dev[2], F_host[0][2], 1e-10);
    EXPECT_GT(std::abs(F_host[0][0]), 1e-8);
}

TEST(CudaGpwForces, FreeFnRejectsMissingGrids) {
    ATV::cuda::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
    m.cutoff_ry = 80.0;
    m.n_occ = 1;
    m.xc = XcKind::PADE;
    m.local_pp_on_grid = true;
    m.ngrids = 1;
    ATV::cuda::gpw::GpwResult scf;
    scf.converged = true;
    scf.P = xt::zeros<double>({2, 2});
    scf.C = xt::zeros<double>({2, 2});
    scf.eps = {0.0, 0.0};
    // empty veff → throw before geometry work
    auto atoms = std::vector<GpwAtom>{};
    EXPECT_THROW(ATV::cuda::gpw::gpw_forces_from_scf(atoms, m, scf),
                 std::runtime_error);
}

TEST(CudaGpwForces, UksDoesNotHitFirstSliceReject) {
    ATV::cuda::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
    m.cutoff_ry = 80.0;
    m.n_occ = 0;
    m.n_alpha = 1;
    m.n_beta = 0;
    m.xc = XcKind::PADE;
    ATV::cuda::gpw::GpwResult scf;
    scf.converged = true;
    auto atoms = std::vector<GpwAtom>{};
    try {
        (void)ATV::cuda::gpw::gpw_forces_from_scf(atoms, m, scf);
        FAIL() << "expected host-assembly throw";
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("UKS forces not on device"), std::string::npos)
            << msg;
    }
}
