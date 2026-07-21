/// @file test_md.cu
/// @brief Integration test: NVE MD with cuda::PairLJCut + output components.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/driver/cuda_driver.cuh"
#include "atomiverse/cuda/md/init.cuh"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/observer/rdf.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/recorder/recorder.cuh"
#include "atomiverse/cuda/recorder/thermo.cuh"

using namespace ATV;

namespace {

/// Create a simple cubic lattice of cuda::PairLJCut particles.
cpu::Frame make_lj_system(int n_side, double density) {
    int n = n_side * n_side * n_side;
    double box_len = std::cbrt(double(n) / density);
    double spacing = box_len / double(n_side);

    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.atoms.x[idx] = (double(ix) + double(0.5)) * spacing;
                frame.atoms.y[idx] = (double(iy) + double(0.5)) * spacing;
                frame.atoms.z[idx] = (double(iz) + double(0.5)) * spacing;
                frame.atoms.mass[idx] = double(1.0);
                frame.atoms.type[idx] = 0;
                idx++;
            }
    frame.box = Mat3::cubic(static_cast<double>(box_len));

    // Small random velocities
    std::srand(42);
    for (int i = 0; i < n; ++i) {
        frame.atoms.vx[i] =
            double(0.001)
            * (double(std::rand()) / double(RAND_MAX) - double(0.5));
        frame.atoms.vy[i] =
            double(0.001)
            * (double(std::rand()) / double(RAND_MAX) - double(0.5));
        frame.atoms.vz[i] =
            double(0.001)
            * (double(std::rand()) / double(RAND_MAX) - double(0.5));
    }

    // Remove center-of-mass velocity
    double mvx = 0, mvy = 0, mvz = 0;
    for (int i = 0; i < n; ++i) {
        mvx += frame.atoms.vx[i];
        mvy += frame.atoms.vy[i];
        mvz += frame.atoms.vz[i];
    }
    mvx /= double(n);
    mvy /= double(n);
    mvz /= double(n);
    for (int i = 0; i < n; ++i) {
        frame.atoms.vx[i] -= mvx;
        frame.atoms.vy[i] -= mvy;
        frame.atoms.vz[i] -= mvz;
    }

    return frame;
}

} // anonymous namespace

// ── Structural / taxonomy tests (link 02-md reparent) ────────────────

/// cuda::md::MD is a loop node over the MD phase domain.
///
/// It used to assert `is_base_of<cuda::DynamicsDriver, MD>` — "MD is in the Dynamics
/// category". The category is gone (termination is a LoopHandler, not a class), so that
/// assertion now asserts nothing. What still has content is the stage domain: MD is a
/// `CUDADriver` over `md::Phase`, which is what makes an `md::Phase`-bound component
/// dispatchable in it and what a cross-workflow misbind is checked against.
TEST(MDTaxonomy, IsALoopNodeOverMDPhase) {
    static_assert(
        std::is_base_of_v<cuda::CUDADriver<ATV::md::Phase>, cuda::md::MD>,
        "cuda::md::MD must be a CUDADriver over md::Phase");
    EXPECT_TRUE(
        (std::is_base_of_v<cuda::CUDADriver<ATV::md::Phase>, cuda::md::MD>));
}

/// The Zero* infrastructure components are phase-agnostic: they expose no
/// hardcoded default_bindings(), so each concrete driver binds them to
/// its own phase enum via explicit on(phase).add().
TEST(MDTaxonomy, ZeroComponentsArePhaseAgnostic) {
    EXPECT_TRUE(cuda::md::ZeroForces{}.default_bindings().empty());
    EXPECT_TRUE(cuda::md::ZeroEnergy{}.default_bindings().empty());
    EXPECT_TRUE(cuda::md::ZeroKinetic{}.default_bindings().empty());
}

/// An MD setup with a Force-capable evaluator but no Integrator-role
/// component must be rejected by `MD::validate_components()` at build().
TEST(MDTaxonomy, ThrowsOnMissingIntegrator) {
    double cutoff = double(3.0);
    cuda::md::MD sim;
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
    EXPECT_THROW(sim.build(), std::runtime_error);
}

/// An MD setup with an Integrator but no Force-capable evaluator must be
/// rejected by `MD::validate_components()` at build().
TEST(MDTaxonomy, ThrowsOnMissingForceWriter) {
    cuda::md::MD sim;
    sim.add(cuda::md::VelocityVerlet{double(0.001)});
    EXPECT_THROW(sim.build(), std::runtime_error);
}

TEST(MD, NVEEnergyConservation) {
    auto frame = make_lj_system(4, double(0.8)); // 64 atoms
    double cutoff = double(3.0);

    // First run: get initial total energy
    cpu::Frame frame1 = frame;
    {
        cuda::md::MD sim;
        sim.add(cuda::md::VelocityVerlet{double(0.001)});
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
        sim.build();
        sim.run(frame1, 1);
    }
    double e_initial = frame1.scalars.energy + frame1.scalars.kinetic;

    // Run 100 steps
    cpu::Frame frame2 = frame;
    {
        cuda::md::MD sim;
        sim.add(cuda::md::VelocityVerlet{double(0.001)});
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
        sim.build();
        sim.run(frame2, 100);
    }
    double e_final = frame2.scalars.energy + frame2.scalars.kinetic;

    // KE should be positive after integration
    EXPECT_GT(frame2.scalars.kinetic, double(0));

    // Total energy should be conserved within tolerance
    // (small dt = 0.001 fs, 100 steps, float precision)
    double tol = std::abs(e_initial) * double(0.01) + double(0.01);
    EXPECT_NEAR(static_cast<double>(e_final), static_cast<double>(e_initial),
                static_cast<double>(tol));
}

TEST(MD, ThermoOutput) {
    auto frame = make_lj_system(4, double(0.8));
    double cutoff = double(3.0);

    cuda::md::MD sim;
    sim.add(cuda::md::VelocityVerlet{double(0.001)});
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{cutoff}, -10)
        .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
    sim.on(md::Phase::end_of_step).add(cuda::Thermo{5});
    sim.build();

    // Should print thermo output without crashing
    testing::internal::CaptureStdout();
    sim.run(frame, 10);
    std::string output = testing::internal::GetCapturedStdout();

    // Should contain header and data lines
    EXPECT_NE(output.find("Step"), std::string::npos);
    EXPECT_NE(output.find("Temp"), std::string::npos);
    EXPECT_NE(output.find("PotEng"), std::string::npos);
}

TEST(MD, RecorderXYZ) {
    auto frame = make_lj_system(3, double(0.8)); // 27 atoms
    double cutoff = double(3.0);
    std::string filename = "test_traj.xyz";

    {
        cuda::md::MD sim;
        sim.add(cuda::md::VelocityVerlet{double(0.001)});
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
        sim.on(md::Phase::end_of_step).add(cuda::Recorder{filename, 3});
        sim.build();
        sim.run(frame, 9);
    }

    // Check file exists and has content
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());

    // Count frames: each frame starts with atom count line
    int frame_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        // Atom count line
        if (line == "27")
            frame_count++;
    }
    // freq=3, steps 0..8 -> steps 0,3,6 = 3 frames
    EXPECT_EQ(frame_count, 3);

    // Cleanup
    std::remove(filename.c_str());
}

TEST(MD, RDFOutput) {
    auto frame = make_lj_system(4, double(0.8));
    double cutoff = double(3.0);
    std::string filename = "test_rdf.dat";

    {
        cuda::md::MD sim;
        sim.add(cuda::md::VelocityVerlet{double(0.001)});
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
        sim.on(md::Phase::end_of_step).add(cuda::RDF{cutoff, 50, filename});
        sim.build();
        sim.run(frame, 5);
    }

    // Check file exists
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());

    // Read data: skip header, check we have bins
    std::string line;
    int data_lines = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        data_lines++;
    }
    EXPECT_EQ(data_lines, 50);

    // Cleanup
    std::remove(filename.c_str());
}

TEST(MD, InitVelocity) {
    // Create system with zero velocities
    int n = 64;
    cpu::Frame frame(n);
    double box_len = std::cbrt(double(n) / double(0.8));
    double spacing = box_len / double(4);
    int idx = 0;
    for (int ix = 0; ix < 4; ++ix)
        for (int iy = 0; iy < 4; ++iy)
            for (int iz = 0; iz < 4; ++iz) {
                frame.atoms.x[idx] = (double(ix) + double(0.5)) * spacing;
                frame.atoms.y[idx] = (double(iy) + double(0.5)) * spacing;
                frame.atoms.z[idx] = (double(iz) + double(0.5)) * spacing;
                frame.atoms.mass[idx] = double(1.0);
                idx++;
            }
    frame.box = Mat3::cubic(static_cast<double>(box_len));

    // All velocities should be zero initially
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(frame.atoms.vx[i], double(0));
    }

    double target_temp = double(300.0);
    double cutoff = double(3.0);

    // Run 1 step with InitVelocity — velocities should be set
    {
        cuda::md::MD sim;
        sim.add(cuda::md::InitVelocity{target_temp, 42});
        sim.add(cuda::md::VelocityVerlet{double(0.001)});
        sim.on(md::Phase::force)
            .add(cuda::NeighborList{cutoff}, -10)
            .add(cuda::PairLJCut{double(1.0), double(1.0), cutoff});
        sim.build();
        sim.run(frame, 1);
    }

    // Velocities should now be non-zero
    bool has_nonzero = false;
    for (int i = 0; i < n; ++i) {
        if (frame.atoms.vx[i] != double(0) || frame.atoms.vy[i] != double(0)
            || frame.atoms.vz[i] != double(0))
            has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);

    // Center-of-mass velocity should be near zero
    double mvx = 0, mvy = 0, mvz = 0, M = 0;
    for (int i = 0; i < n; ++i) {
        mvx += frame.atoms.mass[i] * frame.atoms.vx[i];
        mvy += frame.atoms.mass[i] * frame.atoms.vy[i];
        mvz += frame.atoms.mass[i] * frame.atoms.vz[i];
        M += frame.atoms.mass[i];
    }
    // After 1 step of VV, COM velocity may drift slightly from forces,
    // but should still be small
    EXPECT_NEAR(static_cast<double>(mvx / M), 0.0, 0.01);
    EXPECT_NEAR(static_cast<double>(mvy / M), 0.0, 0.01);
    EXPECT_NEAR(static_cast<double>(mvz / M), 0.0, 0.01);

    // Kinetic energy should be positive (velocities were generated)
    EXPECT_GT(frame.scalars.kinetic, double(0));

    // Temperature should be in the right ballpark (within factor 3 for 64 atoms)
    int ndof = 3 * n - 3;
    double temp =
        double(2) * frame.scalars.kinetic / (double(ndof) * kBoltzmann);
    EXPECT_GT(static_cast<double>(temp),
              static_cast<double>(target_temp) * 0.3);
    EXPECT_LT(static_cast<double>(temp),
              static_cast<double>(target_temp) * 3.0);
}
