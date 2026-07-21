/// @file cpu/scf/internal/basis_workspace.h
/// @brief **PRIVATE** AtomList / ShellList workspace — not a user type.
///
/// Frame is the public molecule. These types are Bohr-unit scratch for
/// integral kernels only.
#pragma once

#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Private: atoms in Bohr + electron count (from Frame + charge).
///
/// Ghost atoms (BSSE counterpoise): listed by local Frame index. Their
/// nuclear charge is zeroed (`atoms[i].Z = 0`) so nuclear attraction,
/// nuclear repulsion, and electron counting exclude them, while
/// `basis_z[i]` keeps the true elemental Z for ShellList table lookup
/// so their basis functions still enter the AO space.
struct AtomList {
    std::vector<native::Atom> atoms;
    /// @brief True elemental Z for each atom (parallel to `atoms`). Used
    ///        only for basis-table lookup; equals `atoms[i].Z` for real
    ///        atoms and the pre-ghost Z for ghost atoms.
    std::vector<int> basis_z;
    int n_electrons = 0;

    AtomList() = default;
    /// @brief Build from Frame geometry.
    /// @param frame       Geometry (Å) + Z in `kAtom_type`.
    /// @param charge      Molecular charge (0 = neutral).
    /// @param ghost_atoms Local atom indices treated as ghosts (basis only;
    ///                    Z → 0 for nuclear/electron terms). Empty = none.
    /// @throws std::out_of_range when a ghost index is outside
    ///         `[0, nlocal)`.
    AtomList(const Frame& frame, int charge = 0,
             const std::vector<int>& ghost_atoms = {});
};

/// @brief Private: shells + owned coefficient storage for one basis.
struct ShellList {
    std::vector<native::Shell> shells;
    std::vector<std::vector<double>> shell_storage;
    int n_basis = 0;
    BasisSet basis = BasisSet::STO3G;

    ShellList() = default;
    /// @brief Orbital shell list. Basis-table lookup uses `basis_z[a]`
    ///        when non-empty, else `atoms[a].Z` (manual AtomList builders).
    /// @param atoms Atom list (Bohr positions + Z / basis_z).
    /// @param basis Orbital basis identifier.
    ShellList(const AtomList& atoms, BasisSet basis);

    /// @brief Auxiliary (density-fitting) shell list — same table→shell
    ///        conversion as the orbital path, but allows l up to kMaxL
    ///        (g) since aux shells only reach the 2-/3-center Rys paths.
    ///        `basis` keeps its default value (not meaningful here).
    /// @param atoms Atom list (Bohr positions + Z).
    /// @param aux   Auxiliary basis identifier (throws when an atom has
    ///              no table row — no fallback).
    ShellList(const AtomList& atoms, AuxBasis aux);

    ShellList(const ShellList& o) { copy_from(o); }
    ShellList(ShellList&& o) noexcept { move_from(std::move(o)); }
    ShellList& operator=(const ShellList& o) {
        if (this != &o)
            copy_from(o);
        return *this;
    }
    ShellList& operator=(ShellList&& o) noexcept {
        if (this != &o)
            move_from(std::move(o));
        return *this;
    }

  private:
    void copy_from(const ShellList& o);
    void move_from(ShellList&& o) noexcept;
    void rebind_pointers();
};

} // namespace scf
} // namespace cpu
} // namespace ATV
