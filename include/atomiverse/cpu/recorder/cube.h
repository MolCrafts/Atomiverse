/// @file cpu/recorder/cube.h
/// @brief Gaussian ".cube" volumetric file I/O helpers.
///
/// Shared by `DensityCubeWriter` and `OrbitalCubeWriter`. The cube format is
/// fixed: all coordinates are written in Bohr (the SCF convention), values
/// are printed 6 per line in exponential form.
///
/// Specification: http://paulbourke.net/dataformats/cube/
#pragma once

#include <array>
#include <string>
#include <vector>

#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
class Frame;
namespace recorder {

/// @brief Uniform Cartesian grid in Bohr (cube-file native units).
struct CubeGrid {
    std::array<double, 3> origin;  ///< Grid origin [Bohr].
    std::array<int, 3> dims;       ///< Number of samples along x,y,z.
    std::array<double, 3> spacing; ///< Sample spacing [Bohr] along each axis.

    /// @brief Grid voxel volume [Bohr^3].
    double voxel() const { return spacing[0] * spacing[1] * spacing[2]; }

    /// @brief Total number of samples.
    long n_points() const {
        return static_cast<long>(dims[0]) * dims[1] * dims[2];
    }
};

/// @brief Build a uniform grid that wraps the molecule with a padding margin.
///
/// @param shells  Basis shells (centers already in Bohr).
/// @param padding Extra margin around the shell-center bounding box [Bohr].
/// @param spacing Sample spacing [Bohr] (isotropic for all three axes).
/// @return CubeGrid centered on the bounding box.
CubeGrid bounding_box_grid(const std::vector<scf::native::Shell>& shells,
                           double padding = 4.0, double spacing = 0.25);

/// @brief Write a Gaussian cube file.
///
/// Atom block positions and nuclear charges come from `frame` (Å → Bohr).
///
/// @param filename Output path.
/// @param title1   First title line.
/// @param title2   Second title line.
/// @param frame    Geometry (Å) + Z in type column.
/// @param grid     Grid description (all Bohr).
/// @param values   Grid values in row-major order (ix, iy, iz) with
///                 iz the fastest axis -- cube-file convention.
/// @throws std::runtime_error if the file cannot be opened.
void write_cube(const std::string& filename, const std::string& title1,
                const std::string& title2, const Frame& frame,
                const CubeGrid& grid, const std::vector<double>& values);

} // namespace recorder
} // namespace cpu
} // namespace ATV
