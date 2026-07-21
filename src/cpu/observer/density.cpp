/// @file cpu/observer/density.cpp
/// @brief Electron density grid evaluation:
///        rho(r) = sum_{mu,nu} P_{mu,nu} phi_mu(r) phi_nu(r).
///
/// Basis data comes from the canonical auto-generated STO-3G table
/// (`scf/basis_data/sto3g.h`, PySCF bit-for-bit) resolved through the
/// shared builder `scf::build_shell_list`, and grid-point evaluation
/// uses the shared `scf::eval_basis` (`scf/basis_eval.h`) -- the same
/// path the cube writers use. No basis constants are duplicated here.

#include "atomiverse/cpu/observer/density.h"

#include <algorithm>
#include <cstddef>
#include <limits>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/basis_data/sto3g.h"
#include "atomiverse/cpu/scf/basis_eval.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"

namespace ATV {
namespace cpu {

namespace {

// True when every atomic number has STO-3G data in the canonical table.
// Preserves the documented empty-grid contract of eval_density_grid
// (build_shell_list throws on unsupported Z instead of returning empty).
bool sto3g_covers(const std::vector<int>& type_id) {
    const auto& table = scf::basis_data::sto3g();
    for (const int Z : type_id) {
        if (Z < 1 || Z >= static_cast<int>(table.size())
            || table[static_cast<std::size_t>(Z)].shells.empty())
            return false;
    }
    return true;
}

// Resolve the STO-3G shell list for the given geometry via the shared
// builder. Positions arrive in Angstrom; shells live in Bohr.
scf::ShellList sto3g_shells(const std::vector<int>& type_id,
                            const std::vector<double>& x,
                            const std::vector<double>& y,
                            const std::vector<double>& z) {
    scf::AtomList atoms;
    atoms.atoms.resize(type_id.size());
    int n_electrons = 0;
    for (std::size_t i = 0; i < type_id.size(); ++i) {
        atoms.atoms[i].Z = type_id[i];
        atoms.atoms[i].pos[0] = x[i] * units::kAngstromInBohr;
        atoms.atoms[i].pos[1] = y[i] * units::kAngstromInBohr;
        atoms.atoms[i].pos[2] = z[i] * units::kAngstromInBohr;
        n_electrons += type_id[i];
    }
    atoms.n_electrons = n_electrons;
    return scf::ShellList{atoms, scf::BasisSet::STO3G};
}

} // namespace

// ── eval_density_grid ─────────────────────────────────────────────────────────

DensityGrid eval_density_grid(const std::vector<int>& type_id,
                              const std::vector<double>& x,
                              const std::vector<double>& y,
                              const std::vector<double>& z, const double* P,
                              int n_basis, int nx, int ny, int nz,
                              double padding) {
    DensityGrid grid;
    if (nx <= 0 || ny <= 0 || nz <= 0 || n_basis <= 0 || type_id.empty())
        return grid;
    if (!sto3g_covers(type_id))
        return grid;

    const scf::ShellList shell_list = sto3g_shells(type_id, x, y, z);
    if (shell_list.n_basis != n_basis)
        return grid;

    // Grid bounds from atom positions + padding
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();
    for (size_t i = 0; i < x.size(); ++i) {
        min_x = std::min(min_x, x[i]);
        min_y = std::min(min_y, y[i]);
        min_z = std::min(min_z, z[i]);
        max_x = std::max(max_x, x[i]);
        max_y = std::max(max_y, y[i]);
        max_z = std::max(max_z, z[i]);
    }
    min_x -= padding;
    min_y -= padding;
    min_z -= padding;
    max_x += padding;
    max_y += padding;
    max_z += padding;

    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.ox = min_x;
    grid.oy = min_y;
    grid.oz = min_z;
    grid.lx = std::max(max_x - min_x, 1e-6);
    grid.ly = std::max(max_y - min_y, 1e-6);
    grid.lz = std::max(max_z - min_z, 1e-6);

    const size_t n = static_cast<size_t>(n_basis);
    const size_t total = static_cast<size_t>(nx) * static_cast<size_t>(ny)
                         * static_cast<size_t>(nz);
    grid.rho.assign(total, 0.0);
    std::vector<double> phi(n, 0.0);

    for (int ix = 0; ix < nx; ++ix) {
        const double rx = min_x + (static_cast<double>(ix) / nx) * grid.lx;
        const double rx_bohr = rx * units::kAngstromInBohr;
        for (int iy = 0; iy < ny; ++iy) {
            const double ry = min_y + (static_cast<double>(iy) / ny) * grid.ly;
            const double ry_bohr = ry * units::kAngstromInBohr;
            for (int iz = 0; iz < nz; ++iz) {
                const double rz =
                    min_z + (static_cast<double>(iz) / nz) * grid.lz;
                const double rz_bohr = rz * units::kAngstromInBohr;

                // Evaluate all basis functions at this point (shared
                // evaluator -- same as the cube writers).
                for (size_t mu = 0; mu < n; ++mu)
                    phi[mu] = scf::eval_basis(shell_list.shells[mu], rx_bohr,
                                              ry_bohr, rz_bohr);

                // rho(r) = sum_{mu,nu} P_{mu,nu} phi_mu phi_nu
                double val = 0.0;
                for (size_t mu = 0; mu < n; ++mu)
                    for (size_t nu = 0; nu < n; ++nu)
                        val += P[mu * n + nu] * phi[mu] * phi[nu];

                grid.rho[(static_cast<size_t>(ix) * ny + iy) * nz + iz] = val;
            }
        }
    }
    return grid;
}

} // namespace cpu
} // namespace ATV
