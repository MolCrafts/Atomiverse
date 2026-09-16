/// @file cpu/multiscale/qm_mask.h
/// @brief Per-atom QM region mask column for multiscale couplings.
///
/// **How to declare a QM region (v1 runtime column):**
/// @code
///   frame.blocks.at("atoms").insert(
///       "qm_z", ColumnSpec{/*dtype=*/ColumnDtype::I32,
///                          /*wire_id=*/kDynamicFieldId});
///   auto qm = frame.span(cpu::kAtomQmZ);
///   qm[0] = 8;  // QM oxygen (Z = 8)
///   qm[1] = 1;  // QM hydrogen
///   // remaining rows stay 0 → MM
/// @endcode
///
/// Value **0 = MM**; **> 0 = QM with atomic number Z = value**. The force-field
/// `type` column keeps its ordinary meaning. The column is runtime-only
/// (`kDynamicFieldId`) and does not cross the molrs boundary in v1.
///
/// **Covalent QM/MM links** are not a separate user list: at
/// `Qmmm*::build`, cuts are derived from this mask × existing
/// `num_bond`/`bond_atom` topology. No bonds ⇒ zero links (valid for LJ
/// particle QM/MM). See harness `frame-region-fields`.
#pragma once

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {

/// @brief Per-atom QM mask / Z column on the `atoms` block.
///
/// Dynamic runtime field (`wire_id = kDynamicFieldId`). Insert before
/// `QmmmElectrostatic::build()`; missing or all-zero throws with a how-to.
inline constexpr ColKey<int> kAtomQmZ{"atoms", "qm_z", kDynamicFieldId,
                                      ColDType::Int};

} // namespace cpu
} // namespace ATV
