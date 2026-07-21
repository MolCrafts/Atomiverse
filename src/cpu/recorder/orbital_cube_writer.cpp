/// @file cpu/recorder/orbital_cube_writer.cpp
/// @brief Molecular-orbital cube writer implementation.

#include "atomiverse/cpu/recorder/orbital_cube_writer.h"

#include <stdexcept>

#include "atomiverse/cpu/scf/basis_eval.h"

namespace ATV {
namespace cpu {
namespace recorder {

OrbitalCubeWriter::OrbitalCubeWriter(scf::MeanFieldOutputs handle,
                                     std::string filename, int mo_index,
                                     OrbitalSpin spin, double padding,
                                     double spacing, int stride)
    : handle_(std::move(handle)), filename_(std::move(filename)),
      mo_index_(mo_index), spin_(spin), auto_grid_(true), padding_(padding),
      spacing_(spacing), stride_(stride < 1 ? 1 : stride) {
    if (!handle_.result || !handle_.shells || !handle_.ready)
        throw std::invalid_argument("OrbitalCubeWriter: null MeanFieldOutputs");
}

OrbitalCubeWriter::OrbitalCubeWriter(scf::MeanFieldOutputs handle,
                                     std::string filename, int mo_index,
                                     OrbitalSpin spin, CubeGrid grid,
                                     int stride)
    : handle_(std::move(handle)), filename_(std::move(filename)),
      mo_index_(mo_index), spin_(spin), auto_grid_(false), grid_(grid),
      padding_(0), spacing_(0), stride_(stride < 1 ? 1 : stride) {
    if (!handle_.result || !handle_.shells || !handle_.ready)
        throw std::invalid_argument("OrbitalCubeWriter: null MeanFieldOutputs");
}

void OrbitalCubeWriter::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_rank != 0)
        return;
    if (!*handle_.ready)
        return;
    if ((ctx.step % stride_) != 0)
        return;

    const auto& shells = *handle_.shells;
    const scf::SlaterDeterminant& det = *handle_.result;
    const int n_basis = static_cast<int>(shells.size());

    const auto& C = det.coefficients(spin_ == OrbitalSpin::Alpha
                                         ? scf::SlaterDeterminant::Spin::alpha
                                         : scf::SlaterDeterminant::Spin::beta);
    if (C.size() == 0)
        throw std::runtime_error(
            "OrbitalCubeWriter: empty MO matrix for requested spin");
    const int n_mo = static_cast<int>(C.shape()[1]);
    if (mo_index_ < 0 || mo_index_ >= n_mo)
        throw std::runtime_error("OrbitalCubeWriter: mo_index out of range");

    CubeGrid grid =
        auto_grid_ ? bounding_box_grid(shells, padding_, spacing_) : grid_;

    std::vector<double> values;
    values.resize(static_cast<size_t>(grid.n_points()), 0.0);

    const int nx = grid.dims[0];
    const int ny = grid.dims[1];
    const int nz = grid.dims[2];

    for (int ix = 0; ix < nx; ++ix) {
        double rx = grid.origin[0] + ix * grid.spacing[0];
        for (int iy = 0; iy < ny; ++iy) {
            double ry = grid.origin[1] + iy * grid.spacing[1];
            for (int iz = 0; iz < nz; ++iz) {
                double rz = grid.origin[2] + iz * grid.spacing[2];
                double psi = 0.0;
                for (int mu = 0; mu < n_basis; ++mu)
                    psi += C(mu, mo_index_)
                           * scf::eval_basis(shells[mu], rx, ry, rz);
                long idx = (static_cast<long>(ix) * ny + iy) * nz + iz;
                values[idx] = psi;
            }
        }
    }

    write_cube(filename_, "Molecular orbital from Atomiverse SCF",
               std::string("MO index=") + std::to_string(mo_index_) + " spin="
                   + (spin_ == OrbitalSpin::Alpha ? "alpha" : "beta"),
               ctx.frame, grid, values);
    ++n_writes_;
}

} // namespace recorder
} // namespace cpu
} // namespace ATV
