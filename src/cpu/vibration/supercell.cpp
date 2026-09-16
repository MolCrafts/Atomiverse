/// @file cpu/vibration/supercell.cpp
/// @brief Expand a unit-cell Frame into an \(n_a\times n_b\times n_c\) supercell.

#include "atomiverse/cpu/vibration/supercell.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"

namespace ATV {
namespace cpu {
namespace vib {

cpu::Frame expand_supercell(const cpu::Frame& unit, int na, int nb, int nc) {
    if (na < 1 || nb < 1 || nc < 1)
        throw std::invalid_argument(
            "expand_supercell: na, nb, nc must each be ≥ 1");
    const int n_unit = unit.nlocal();
    if (n_unit <= 0)
        throw std::invalid_argument("expand_supercell: unit cell has no atoms");

    const int n_sc = n_unit * na * nb * nc;
    cpu::Frame sc(n_sc);

    const Mat3& cell = unit.box.cell();
    // Rows of cell are lattice vectors a, b, c [Å].
    const double ax = cell(0, 0), ay = cell(0, 1), az = cell(0, 2);
    const double bx = cell(1, 0), by = cell(1, 1), bz = cell(1, 2);
    const double cx = cell(2, 0), cy = cell(2, 1), cz = cell(2, 2);

    Mat3 sc_cell;
    sc_cell(0, 0) = ax * na;
    sc_cell(0, 1) = ay * na;
    sc_cell(0, 2) = az * na;
    sc_cell(1, 0) = bx * nb;
    sc_cell(1, 1) = by * nb;
    sc_cell(1, 2) = bz * nb;
    sc_cell(2, 0) = cx * nc;
    sc_cell(2, 1) = cy * nc;
    sc_cell(2, 2) = cz * nc;
    sc.box.set_cell(sc_cell);

    const auto ux = unit.span(kAtom_x);
    const auto uy = unit.span(kAtom_y);
    const auto uz = unit.span(kAtom_z);
    const auto ut = unit.span(kAtom_type);
    const auto um = unit.span(kAtom_mass);

    auto sx = sc.span(kAtom_x);
    auto sy = sc.span(kAtom_y);
    auto sz = sc.span(kAtom_z);
    auto st = sc.span(kAtom_type);
    auto sm = sc.span(kAtom_mass);
    auto stag = sc.span(kAtom_tag);

    int out = 0;
    for (int ia = 0; ia < na; ++ia) {
        for (int ib = 0; ib < nb; ++ib) {
            for (int ic = 0; ic < nc; ++ic) {
                const double ox = ia * ax + ib * bx + ic * cx;
                const double oy = ia * ay + ib * by + ic * cy;
                const double oz = ia * az + ib * bz + ic * cz;
                for (int i = 0; i < n_unit; ++i) {
                    sx[out] = static_cast<StorageReal>(
                        static_cast<double>(ux[i]) + ox);
                    sy[out] = static_cast<StorageReal>(
                        static_cast<double>(uy[i]) + oy);
                    sz[out] = static_cast<StorageReal>(
                        static_cast<double>(uz[i]) + oz);
                    st[out] = ut[i];
                    sm[out] = um[i];
                    stag[out] = static_cast<TagInt>(out + 1);
                    ++out;
                }
            }
        }
    }
    return sc;
}

} // namespace vib
} // namespace cpu
} // namespace ATV
