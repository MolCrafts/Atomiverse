/// @file cpu/record/test_density_grid_cube.cpp
/// @brief DensityGrid Cube conversion preserves sample positions and value order.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/density.h"
#include "atomiverse/cpu/record/cube.h"
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
struct CubeFile {
    std::filesystem::path path =
        std::filesystem::temp_directory_path()
        / ("atv-density-grid-"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count())
           + ".cube");
    ~CubeFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

ATV::cpu::DensityGrid density_grid() {
    ATV::cpu::DensityGrid grid;
    grid.nx = 2;
    grid.ny = 3;
    grid.nz = 4;
    grid.ox = 100.00000001;
    grid.oy = -2.25;
    grid.oz = 0.125;
    grid.lx = 4.0;
    grid.ly = 9.0;
    grid.lz = 2.0;
    for (int i = 0; i < 24; ++i)
        grid.rho.push_back(0.125 * (i + 1));
    return grid;
}
} // namespace

TEST(DensityGridCube, WriteCubePreservesHalfOpenGridAndPhysicalGeometry) {
    ATV::cpu::Frame geometry(1);
    geometry.span(ATV::kAtom_type)[0] = 1;
    geometry.span(ATV::kAtom_x)[0] = 100.00000001;
    geometry.span(ATV::kAtom_y)[0] = -2.25;
    geometry.span(ATV::kAtom_z)[0] = 0.125;
    const auto grid = density_grid();
    CubeFile output;
    ATV::cpu::recorder::write_cube(output.path.string(), "density", "samples",
                                   geometry, grid);
    std::ifstream input(output.path);
    ASSERT_TRUE(input.good());
    std::string line;
    std::getline(input, line);
    EXPECT_EQ(line, "density");
    std::getline(input, line);
    EXPECT_EQ(line, "samples");
    int count;
    double x, y, z;
    ASSERT_TRUE(bool(input >> count >> x >> y >> z));
    EXPECT_EQ(count, 1);
    constexpr double factor = ATV::units::kAngstromInBohr;
    EXPECT_NEAR(x, grid.ox * factor, 1e-13);
    EXPECT_NEAR(y, grid.oy * factor, 1e-14);
    EXPECT_NEAR(z, grid.oz * factor, 1e-14);
    for (int axis = 0; axis < 3; ++axis) {
        ASSERT_TRUE(bool(input >> count >> x >> y >> z));
        const int dimensions[] = {2, 3, 4};
        const double spacing[] = {2.0, 3.0, 0.5};
        EXPECT_EQ(count, dimensions[axis]);
        EXPECT_NEAR(x, axis == 0 ? spacing[axis] * factor : 0.0, 1e-14);
        EXPECT_NEAR(y, axis == 1 ? spacing[axis] * factor : 0.0, 1e-14);
        EXPECT_NEAR(z, axis == 2 ? spacing[axis] * factor : 0.0, 1e-14);
    }
    double charge;
    ASSERT_TRUE(bool(input >> count >> charge >> x >> y >> z));
    EXPECT_EQ(count, 1);
    EXPECT_NEAR(x, double(geometry.span(ATV::kAtom_x)[0]) * factor, 1e-13);
    for (double expected : grid.rho) {
        double value;
        ASSERT_TRUE(bool(input >> value));
        EXPECT_DOUBLE_EQ(value, expected);
    }
    double extra;
    EXPECT_FALSE(bool(input >> extra));
}

TEST(DensityGridCube, WriteCubeRejectsInvalidGridBeforeCreatingOutput) {
    ATV::cpu::Frame geometry(1);
    CubeFile output;
    auto grid = density_grid();
    grid.nx = 0;
    EXPECT_THROW(ATV::cpu::recorder::write_cube(output.path.string(), "density",
                                                "samples", geometry, grid),
                 std::invalid_argument);
    grid = density_grid();
    grid.lx = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(ATV::cpu::recorder::write_cube(output.path.string(), "density",
                                                "samples", geometry, grid),
                 std::invalid_argument);
    grid = density_grid();
    grid.rho.pop_back();
    EXPECT_THROW(ATV::cpu::recorder::write_cube(output.path.string(), "density",
                                                "samples", geometry, grid),
                 std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(output.path));
}
