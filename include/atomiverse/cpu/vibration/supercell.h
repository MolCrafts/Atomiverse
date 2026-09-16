/// @file cpu/vibration/supercell.h
/// @brief Expand a unit-cell Frame into an \(n_a\times n_b\times n_c\) supercell.
///
/// Rows of `frame.box.cell()` are lattice vectors \(\mathbf{a},\mathbf{b},
/// \mathbf{c}\) [Å]. The supercell uses \(\mathbf{a}'=n_a\mathbf{a}\), etc.,
/// and places one image of every unit-cell atom at
/// \(\mathbf{R} + i\mathbf{a} + j\mathbf{b} + k\mathbf{c}\).
#pragma once

#include <stdexcept>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Build a supercell Frame from a unit cell.
///
/// Copies type, mass, and Cartesian positions; assigns unique tags
/// `1..N_sc`. Does not copy topology (bonds/angles) — force engines that
/// need topology must rebuild it. Ghost rows are empty (`nghost=0`).
///
/// @param unit Unit-cell system (positions [Å], box set).
/// @param na Supercell repeats along \(\mathbf{a}\) (≥1).
/// @param nb Supercell repeats along \(\mathbf{b}\) (≥1).
/// @param nc Supercell repeats along \(\mathbf{c}\) (≥1).
/// @return Supercell with \(N_\mathrm{sc} = N_\mathrm{unit}\cdot n_a n_b n_c\) atoms.
/// @throws std::invalid_argument if any factor < 1 or unit has no atoms.
cpu::Frame expand_supercell(const cpu::Frame& unit, int na, int nb, int nc);

} // namespace vib
} // namespace cpu
} // namespace ATV
