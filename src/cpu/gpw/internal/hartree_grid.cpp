/// @file src/cpu/gpw/internal/hartree_grid.cpp
/// @brief Implementation of the Hartree energy / Coulomb KS block (see header).

#include "atomiverse/cpu/gpw/internal/hartree_grid.h"

#include "atomiverse/cpu/gpw/internal/poisson_fft.h"

namespace ATV {
namespace cpu {
namespace gpw {

HartreeOnGrid hartree_grid(const UniformGrid& grid,
                           const std::vector<GridBasisFunction>& basis,
                           const std::vector<double>& P) {
    const std::vector<double> rho = collocate_density(grid, basis, P);
    const PoissonResult pois = solve_poisson(grid, rho);
    HartreeOnGrid out;
    out.e_hartree = pois.e_hartree;
    out.j_matrix = integrate_potential(grid, basis, pois.v_r);
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
