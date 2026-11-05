#include "atomiverse/cpu/scf/methods.h"
/// @file tests/cpu/recorder/test_density_cube.cpp
/// @brief Tests for cpu::recorder::DensityCubeWriter and OrbitalCubeWriter.
///
/// Core acceptance: the grid-integrated density ≈ n_electrons within a
/// grid-resolution-dependent tolerance.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/record/density_cube_writer.h"
#include "atomiverse/cpu/record/orbital_cube_writer.h"
#include "atomiverse/cpu/scf/internal/scf_types.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

cpu::Frame make_h2(double bond_length) {
    cpu::Frame f(2);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = bond_length;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_mass)[0] = 1.008;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.span(ATV::kAtom_atomic_number)[0] = 1;
    f.span(ATV::kAtom_atomic_number)[1] = 1;
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

struct ParsedCube {
    int n_atoms;
    double origin[3];
    int dims[3];
    double spacing[3]; // diagonal
    std::vector<double> values;
    double voxel() const { return spacing[0] * spacing[1] * spacing[2]; }
};

// Parse a Gaussian cube file that our writer produced (diagonal axes).
ParsedCube parse_cube(const std::string& path) {
    std::ifstream is(path);
    if (!is)
        throw std::runtime_error("parse_cube: cannot open " + path);
    std::string line;
    std::getline(is, line); // title1
    std::getline(is, line); // title2
    ParsedCube c;
    std::getline(is, line);
    {
        std::istringstream ss(line);
        ss >> c.n_atoms >> c.origin[0] >> c.origin[1] >> c.origin[2];
    }
    for (int axis = 0; axis < 3; ++axis) {
        std::getline(is, line);
        std::istringstream ss(line);
        double ax, ay, az;
        ss >> c.dims[axis] >> ax >> ay >> az;
        c.spacing[axis] = (axis == 0) ? ax : (axis == 1) ? ay : az;
    }
    for (int a = 0; a < c.n_atoms; ++a)
        std::getline(is, line);
    long total = static_cast<long>(c.dims[0]) * c.dims[1] * c.dims[2];
    c.values.reserve(total);
    double v;
    while (is >> v)
        c.values.push_back(v);
    if (static_cast<long>(c.values.size()) != total)
        throw std::runtime_error("parse_cube: incomplete value grid");
    return c;
}

} // namespace

TEST(DensityCubeTest, H2DensityIntegratesToTwoElectrons) {
    auto frame = make_h2(0.74);
    cpu::scf::RHF scf{cpu::scf::BasisSet::STO3G};
    auto handle = scf.result_handle();

    auto tmp = std::filesystem::temp_directory_path() / "atv_rho_h2.cube";
    std::error_code ec;
    std::filesystem::remove(tmp, ec);

    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.add_criterion(cpu::ForceConverged{/*ftol=*/0.5});
    opt.on(minimizer::Phase::force).add(std::move(scf));
    opt.on(minimizer::Phase::end_of_step)
        .add(cpu::recorder::DensityCubeWriter{
            handle, tmp.string(), /*padding=*/6.0, /*spacing=*/0.2});
    opt.build();
    opt.run(frame, 1);

    ASSERT_TRUE(std::filesystem::exists(tmp));
    auto cube = parse_cube(tmp.string());
    EXPECT_EQ(cube.n_atoms, 2);
    double sum = 0;
    for (double v : cube.values)
        sum += v;
    double integrated = sum * cube.voxel();
    // 0.2-Bohr grid over a 6-Bohr-padded H2 box -- about 1% accurate.
    EXPECT_NEAR(integrated, 2.0, 0.1)
        << "grid-integrated density should approximate 2 electrons";

    std::filesystem::remove(tmp, ec);
}

TEST(OrbitalCubeTest, WritesOrbitalFileForHomoH2) {
    auto frame = make_h2(0.74);
    cpu::scf::RHF scf{cpu::scf::BasisSet::STO3G};
    auto handle = scf.result_handle();

    auto tmp = std::filesystem::temp_directory_path() / "atv_mo_h2.cube";
    std::error_code ec;
    std::filesystem::remove(tmp, ec);

    // H2 RHF: 2 electrons → n_alpha = 1 → HOMO is orbital 0.
    constexpr int homo_idx = 0;
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.add_criterion(cpu::ForceConverged{/*ftol=*/0.5});
    opt.on(minimizer::Phase::force).add(std::move(scf));
    opt.on(minimizer::Phase::end_of_step)
        .add(cpu::recorder::OrbitalCubeWriter{
            handle, tmp.string(), homo_idx, cpu::recorder::OrbitalSpin::Alpha,
            /*padding=*/5.0, /*spacing=*/0.3});
    opt.build();
    opt.run(frame, 1);

    ASSERT_TRUE(std::filesystem::exists(tmp));
    auto cube = parse_cube(tmp.string());
    EXPECT_EQ(cube.n_atoms, 2);
    // Orbital should be normalized: integral of |psi|^2 ≈ 1.
    double norm = 0;
    double v_min = cube.values.empty() ? 0.0 : cube.values[0];
    double v_max = v_min;
    for (double v : cube.values) {
        norm += v * v;
        if (v < v_min)
            v_min = v;
        if (v > v_max)
            v_max = v;
    }
    norm *= cube.voxel();
    EXPECT_NEAR(norm, 1.0, 0.1);
    // Phase gauge: largest lobe is positive so density-style +iso viewers
    // (MolVis default channel) show a surface without a sign flip.
    EXPECT_GE(v_max, -v_min);
    std::filesystem::remove(tmp, ec);
}

TEST(DensityCubeTest, NullHandleThrows) {
    cpu::scf::MeanFieldOutputs empty{};
    EXPECT_THROW(cpu::recorder::DensityCubeWriter(empty, "/tmp/x.cube"),
                 std::invalid_argument);
    EXPECT_THROW(
        cpu::recorder::OrbitalCubeWriter(empty, "/tmp/x.cube", 0,
                                         cpu::recorder::OrbitalSpin::Alpha),
        std::invalid_argument);
}

// ── LAMMPS `minimize` output step convention ─────────────────────────────
//
// src/min.cpp Min::setup → Output::setup writes the initial state (step 0);
// iteration i ends at completed iteration i (`++update->ntimestep`) and
// writes when on stride; a stop before MAXITER forces the final state
// (Min::run); `last_dump` writes no step twice. The cube is overwritten each
// write, so the file holds the last state written — checked through its
// atom block against an independent minimization of exactly that many
// iterations (the oracle is never the writer).

namespace {

/// @brief Stop after `n` iterations (any LAMMPS stop condition but MAXITER).
class StopAfter : public cpu::Criterion {
  public:
    explicit StopAfter(BigInt n) : n_(n) {}
    std::string name() const override { return "stop_after"; }
    void launch(const cpu::LaunchContext&) override {}
    Verdict verdict(const cpu::LoopState& s) override {
        return s.iteration + 1 >= n_ ? Verdict::Stop : Verdict::Continue;
    }

  private:
    BigInt n_;
};

/// @brief H2 stretched to 0.9 Å, so every LBFGS iteration moves it.
cpu::Frame make_stretched_h2() { return make_h2(0.9); }

void mount_h2(cpu::minimizer::Minimize& opt, cpu::scf::RHF scf) {
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.add_criterion(cpu::ForceConverged{1e-30, 1e-30, 1e-30, 1e-30, 1e-30});
    opt.on(minimizer::Phase::force).add(std::move(scf));
}

/// @brief Oracle geometry [Bohr] after exactly `iters` iterations.
std::vector<std::array<double, 3>> h2_after(int iters) {
    cpu::Frame frame = make_stretched_h2();
    if (iters > 0) {
        cpu::minimizer::Minimize opt;
        mount_h2(opt, cpu::scf::RHF{cpu::scf::BasisSet::STO3G});
        opt.build();
        opt.run(frame, iters);
    }
    std::vector<std::array<double, 3>> r;
    for (int i = 0; i < frame.nlocal(); ++i)
        r.push_back({frame.span(kAtom_x)[i] * units::kAngstromInBohr,
                     frame.span(kAtom_y)[i] * units::kAngstromInBohr,
                     frame.span(kAtom_z)[i] * units::kAngstromInBohr});
    return r;
}

/// @brief The atom block of a cube file [Bohr].
std::vector<std::array<double, 3>> cube_atoms(const std::string& path) {
    std::ifstream is(path);
    std::string line;
    std::getline(is, line);
    std::getline(is, line);
    std::getline(is, line);
    int n = 0;
    std::istringstream(line) >> n;
    for (int axis = 0; axis < 3; ++axis)
        std::getline(is, line);
    std::vector<std::array<double, 3>> r;
    for (int a = 0; a < n; ++a) {
        std::getline(is, line);
        std::istringstream ss(line);
        int z = 0;
        double q = 0;
        std::array<double, 3> p{};
        ss >> z >> q >> p[0] >> p[1] >> p[2];
        r.push_back(p);
    }
    return r;
}

void expect_geometry(const std::vector<std::array<double, 3>>& got,
                     const std::vector<std::array<double, 3>>& want) {
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i)
        for (int d = 0; d < 3; ++d)
            EXPECT_NEAR(got[i][d], want[i][d], 2e-6) << "atom " << i;
}

} // namespace

// The oracle separates the iterations below at the cube's 1e-6 precision.
TEST(CubeWriterSteps, OracleSeparatesIterations) {
    for (int i = 0; i < 3; ++i) {
        const auto a = h2_after(i), b = h2_after(i + 1);
        EXPECT_GT(std::abs(a[1][0] - b[1][0]), 1e-4) << "iteration " << i;
    }
}

// Early stop after 3 iterations at stride 2: writes 0, 2 and the final 3.
TEST(CubeWriterSteps, DensityEarlyStopEndsOnTheFinalState) {
    auto tmp = std::filesystem::temp_directory_path() / "atv_rho_steps.cube";
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    cpu::scf::RHF scf{cpu::scf::BasisSet::STO3G};
    cpu::recorder::DensityCubeWriter writer{scf.result_handle(), tmp.string(),
                                            /*padding=*/4.0, /*spacing=*/0.5,
                                            /*stride=*/2};
    {
        cpu::Frame frame = make_stretched_h2();
        cpu::minimizer::Minimize opt;
        mount_h2(opt, std::move(scf));
        opt.add_criterion(StopAfter{3});
        opt.on(minimizer::Phase::end_of_step).add_ref(writer);
        opt.build();
        EXPECT_EQ(opt.run(frame, 50), 3);
    }
    EXPECT_EQ(writer.n_writes(), 3) << "steps 0, 2 and the final 3";
    EXPECT_EQ(writer.last_step_written(), 3);
    expect_geometry(cube_atoms(tmp.string()), h2_after(3));
    std::filesystem::remove(tmp, ec);
}

// Budget exhausted (MAXITER) at stride 2 over 3 iterations: 0 and 2 only;
// no forced final write.
TEST(CubeWriterSteps, OrbitalBudgetExhaustionWritesOnStrideOnly) {
    auto tmp = std::filesystem::temp_directory_path() / "atv_mo_steps.cube";
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    cpu::scf::RHF scf{cpu::scf::BasisSet::STO3G};
    cpu::recorder::OrbitalCubeWriter writer{
        scf.result_handle(), tmp.string(), 0, cpu::recorder::OrbitalSpin::Alpha,
        /*padding=*/4.0,
        /*spacing=*/0.5,
        /*stride=*/2};
    {
        cpu::Frame frame = make_stretched_h2();
        cpu::minimizer::Minimize opt;
        mount_h2(opt, std::move(scf));
        opt.on(minimizer::Phase::end_of_step).add_ref(writer);
        opt.build();
        EXPECT_EQ(opt.run(frame, 3), 3);
    }
    EXPECT_EQ(writer.n_writes(), 2) << "steps 0 and 2";
    EXPECT_EQ(writer.last_step_written(), 2);
    expect_geometry(cube_atoms(tmp.string()), h2_after(2));
    std::filesystem::remove(tmp, ec);
}
