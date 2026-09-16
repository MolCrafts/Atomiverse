#include "atomiverse/cpu/scf/methods.h"
/// @file tests/cpu/recorder/test_density_cube.cpp
/// @brief Tests for cpu::recorder::DensityCubeWriter and OrbitalCubeWriter.
///
/// Core acceptance: the grid-integrated density ≈ n_electrons within a
/// grid-resolution-dependent tolerance.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
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
    f.span(ATV::kAtom_type)[0] = 1;
    f.span(ATV::kAtom_type)[1] = 1;
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
    cpu::scf::RHF scf{};
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
    cpu::scf::RHF scf{};
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
