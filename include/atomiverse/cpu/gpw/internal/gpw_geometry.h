/// @file cpu/gpw/internal/gpw_geometry.h
/// @brief Geometry and AO tables belonging to a single GPW assembly.
#pragma once

#include "atomiverse/cpu/gpw/internal/gpw_atom.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"

namespace ATV::cpu::gpw::internal {

struct GpwAoBasis {
    std::vector<GridBasisFunction> basis;
    std::vector<double> radii;
    double rmax = 0;
    int mimg[3] = {0, 0, 0};

    GpwAoBasis(const std::vector<GpwAtom>& atoms, const double cell[3],
               bool periodic, double eps);
    int nb() const { return static_cast<int>(basis.size()); }
};

enum class GpwGridTables { none, compact, products, compact_and_products };

struct GpwGeometry {
    std::vector<GpwAtom> atoms;
    std::array<double, 3> cell;
    GpwAoBasis ao;
    UniformGrid grid;
    CompactBasisGrid cbg; ///< Only the periodic Γ single-grid path uses CBG.
    MultiGrid product_grid;
    ProductSet products;
    double product_rel_cutoff = 0; ///< Exponent-to-level calibration [Ry].
    std::vector<KPoint> kpoints;
    KpointBasisGrid kbg; ///< Dense Bloch AO table on the multi-k single grid.
    /// exp(-2πi k·t), indexed by kpoints and products.cells.
    std::vector<std::vector<std::complex<double>>> bloch_cell_phases;

    GpwGeometry(std::vector<GpwAtom> atoms, std::array<double, 3> cell,
                double cutoff_ry, double eps, bool periodic,
                GpwGridTables tables, int product_levels = 1,
                double rel_cutoff = 0);

    void assemble_bloch_grid(std::vector<KPoint> sampled, double eps);
};

} // namespace ATV::cpu::gpw::internal
