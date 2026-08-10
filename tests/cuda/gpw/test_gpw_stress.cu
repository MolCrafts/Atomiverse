/// @file tests/cuda/gpw/test_gpw_stress.cu
/// @brief Unit: public `cuda::gpw::gpw_stress` fail-closed guards (no e2e SCF).
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;

namespace {

std::vector<ATV::cpu::gpw::GpwAtom> h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    return {ATV::cpu::gpw::GpwAtom(rA, basis, pot, 1),
            ATV::cpu::gpw::GpwAtom(rB, basis, pot, 1)};
}

} // namespace

TEST(CudaGpwStress, FailClosedMultiK) {
    auto atoms = h2_atoms();
    ATV::cuda::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
    m.cutoff_ry = 80.0;
    m.n_occ = 1;
    m.xc = XcKind::PADE;
    m.k_mesh[0] = 2;
    m.k_mesh[1] = 1;
    m.k_mesh[2] = 1;
    // build itself rejects multi-k; stress also guards.
    EXPECT_THROW(m.build(atoms), std::invalid_argument);
    ATV::cpu::gpw::GpwResult fake;
    fake.converged = true;
    EXPECT_THROW(ATV::cuda::gpw::gpw_stress(atoms, m, fake),
                 std::invalid_argument);
}

TEST(CudaGpwStress, FailClosedUnconvergedState) {
    auto atoms = h2_atoms();
    ATV::cuda::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
    m.cutoff_ry = 80.0;
    m.n_occ = 1;
    m.xc = XcKind::PADE;
    m.build(atoms);
    ATV::cpu::gpw::GpwResult fake;
    fake.converged = false;
    EXPECT_THROW(ATV::cuda::gpw::gpw_stress(atoms, m, fake),
                 std::invalid_argument);
}
