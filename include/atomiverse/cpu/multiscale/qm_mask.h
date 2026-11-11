/// @file cpu/multiscale/qm_mask.h
/// @brief Per-atom QM region mask column for multiscale couplings.
///
/// **How to declare a QM region (v1 runtime column):**
/// @code
///   frame.block("atoms").insert(
///       "qm", ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
///                        kDynamicFieldId});
///   auto qm = frame.span(cpu::kAtomQm);
///   qm[0] = 1;  // QM
///   qm[1] = 1;  // QM
///   // remaining rows stay 0 → MM
/// @endcode
///
/// Value **0 = MM**, **1 = QM**; anything else throws. The column is region
/// membership only: a QM atom's species is its `kAtom_atomic_number` (Z,
/// must be > 0), and the force-field `type` column keeps its ordinary
/// meaning for the MM terms. The mask is runtime-only (`kDynamicFieldId`)
/// and does not cross the molpy boundary in v1.
///
/// **Covalent QM/MM links** are not a separate user list: at
/// `Qmmm*::build`, cuts are derived from this mask × existing
/// `num_bond`/`bond_atom` topology. No bonds ⇒ zero links (valid for LJ
/// particle QM/MM). See harness `frame-region-fields`.
#pragma once

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {

/// @brief Per-atom QM region mask on the `atoms` block (0 = MM, 1 = QM).
///
/// Dynamic runtime field (`wire_id = kDynamicFieldId`). Insert before
/// `QmmmElectrostatic::build()`; missing, all-zero or a value other than
/// 0 / 1 throws with a how-to. Species come from `kAtom_atomic_number`.
inline constexpr ColKey<int> kAtomQm{"atoms", "qm", kDynamicFieldId,
                                     ColDType::Int};

} // namespace cpu
} // namespace ATV
