/// @file cpu/recorder/density_cube_writer.cpp
/// @brief Electron-density cube writer implementation.

#include "atomiverse/cpu/recorder/density_cube_writer.h"

#include <stdexcept>

#include "atomiverse/cpu/scf/basis_eval.h"

namespace ATV {
namespace cpu {
namespace recorder {

DensityCubeWriter::DensityCubeWriter(scf::MeanFieldOutputs handle,
                                     std::string filename, double padding,
                                     double spacing, int stride)
    : handle_(std::move(handle)), filename_(std::move(filename)),
      auto_grid_(true), padding_(padding), spacing_(spacing),
      stride_(stride < 1 ? 1 : stride) {
    if (!handle_.result || !handle_.shells || !handle_.ready)
        throw std::invalid_argument("DensityCubeWriter: null MeanFieldOutputs");
}

DensityCubeWriter::DensityCubeWriter(scf::MeanFieldOutputs handle,
                                     std::string filename, CubeGrid grid,
                                     int stride)
    : handle_(std::move(handle)), filename_(std::move(filename)),
      auto_grid_(false), grid_(grid), padding_(0), spacing_(0),
      stride_(stride < 1 ? 1 : stride) {
    if (!handle_.result || !handle_.shells || !handle_.ready)
        throw std::invalid_argument("DensityCubeWriter: null MeanFieldOutputs");
}

void DensityCubeWriter::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_rank != 0)
        return;
    if (!*handle_.ready)
        return;
    if ((ctx.step % stride_) != 0)
        return;

    const auto& shells = *handle_.shells;
    const scf::SlaterDeterminant& det = *handle_.result;
    const int n_basis = static_cast<int>(shells.size());

    // Total density P^alpha + P^beta (factor-2 closed-shell density for the
    // restricted form). By-value; hoisted here once, never inside the loops.
    const xt::xtensor<double, 2> P_total = det.density_total();

    CubeGrid grid =
        auto_grid_ ? bounding_box_grid(shells, padding_, spacing_) : grid_;

    std::vector<double> values;
    values.resize(static_cast<size_t>(grid.n_points()), 0.0);
    std::vector<double> phi(n_basis, 0.0);

    const int nx = grid.dims[0];
    const int ny = grid.dims[1];
    const int nz = grid.dims[2];

    double integrated = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
        double rx = grid.origin[0] + ix * grid.spacing[0];
        for (int iy = 0; iy < ny; ++iy) {
            double ry = grid.origin[1] + iy * grid.spacing[1];
            for (int iz = 0; iz < nz; ++iz) {
                double rz = grid.origin[2] + iz * grid.spacing[2];
                for (int mu = 0; mu < n_basis; ++mu)
                    phi[mu] = scf::eval_basis(shells[mu], rx, ry, rz);
                double rho = 0.0;
                for (int mu = 0; mu < n_basis; ++mu) {
                    double row = 0.0;
                    for (int nu = 0; nu < n_basis; ++nu)
                        row += P_total(mu, nu) * phi[nu];
                    rho += phi[mu] * row;
                }
                long idx = (static_cast<long>(ix) * ny + iy) * nz + iz;
                values[idx] = rho;
                integrated += rho;
            }
        }
    }
    last_integral_ = integrated * grid.voxel();

    write_cube(filename_, "Electron density from Atomiverse SCF",
               "rho(r) in e / Bohr^3", ctx.frame, grid, values);
    ++n_writes_;
}

} // namespace recorder
} // namespace cpu
} // namespace ATV
