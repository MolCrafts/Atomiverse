/// @file cpu/scf/basis_set.cpp
/// @brief Implementation of AtomList and ShellList builders.

#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/basis_data/aug_ccpvdz.h"
#include "atomiverse/cpu/scf/basis_data/def2_jkfit.h"
#include "atomiverse/cpu/scf/basis_data/def2_svp.h"
#include "atomiverse/cpu/scf/basis_data/def2_tzvp.h"
#include "atomiverse/cpu/scf/basis_data/pople_631g.h"
#include "atomiverse/cpu/scf/basis_data/pople_631gs.h"
#include "atomiverse/cpu/scf/basis_data/pople_631pgs.h"
#include "atomiverse/cpu/scf/basis_data/sto3g.h"
#include "atomiverse/cpu/scf/cart.h"

namespace ATV {
namespace cpu {
namespace scf {

const char* basis_set_name(BasisSet basis) {
    switch (basis) {
    case BasisSet::STO3G:
        return "STO-3G";
    case BasisSet::Pople631G:
        return "6-31G";
    case BasisSet::Pople631Gs:
        return "6-31G*";
    case BasisSet::Pople631pGs:
        return "6-31+G*";
    case BasisSet::Def2SVP:
        return "def2-SVP";
    case BasisSet::Def2TZVP:
        return "def2-TZVP";
    case BasisSet::AugCcPVDZ:
        return "aug-cc-pVDZ";
    }
    return "unknown";
}

const char* aux_basis_name(AuxBasis aux) {
    switch (aux) {
    case AuxBasis::Def2JKFIT:
        return "def2-universal-jkfit";
    }
    return "unknown";
}

basis_data::AngularConvention angular_convention(BasisSet basis) {
    switch (basis) {
    case BasisSet::STO3G:
    case BasisSet::Pople631G:
    case BasisSet::Pople631Gs:
    case BasisSet::Pople631pGs:
        return basis_data::AngularConvention::Cartesian;
    case BasisSet::Def2SVP:
    case BasisSet::Def2TZVP:
    case BasisSet::AugCcPVDZ:
        return basis_data::AngularConvention::Spherical;
    }
    throw std::runtime_error("angular_convention: unknown BasisSet");
}

// ── ShellList copy / move rebind ─────────────────────────────────────

void ShellList::rebind_pointers() {
    const int n = n_basis;
    for (int i = 0; i < n; ++i) {
        const int np = shells[i].n_prim;
        shells[i].alpha = shell_storage[i].data();
        shells[i].coeff = shell_storage[i].data() + np;
    }
}

void ShellList::copy_from(const ShellList& o) {
    shells = o.shells;
    shell_storage = o.shell_storage;
    n_basis = o.n_basis;
    basis = o.basis;
    rebind_pointers();
}

void ShellList::move_from(ShellList&& o) noexcept {
    shells = std::move(o.shells);
    shell_storage = std::move(o.shell_storage);
    n_basis = o.n_basis;
    basis = o.basis;
    rebind_pointers();
}

// ── AtomList constructor ─────────────────────────────────────────────

AtomList::AtomList(const Frame& frame, int charge,
                   const std::vector<int>& ghost_atoms) {
    const int natm = static_cast<int>(frame.nlocal());

    // Validate ghost indices up front (no fallback / silent clamp).
    std::vector<char> is_ghost(static_cast<std::size_t>(natm), 0);
    for (int g : ghost_atoms) {
        if (g < 0 || g >= natm) {
            throw std::out_of_range("cpu::scf::AtomList: ghost atom index "
                                    + std::to_string(g) + " is outside [0, "
                                    + std::to_string(natm) + ")");
        }
        is_ghost[static_cast<std::size_t>(g)] = 1;
    }

    atoms.resize(static_cast<std::size_t>(natm));
    basis_z.resize(static_cast<std::size_t>(natm));
    int nelec = -charge;
    const int* __restrict sp_type = frame.span(kAtom_type);
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);
    for (int i = 0; i < natm; ++i) {
        const int Z = sp_type[i];
        // True elemental Z always stored for basis-table lookup.
        basis_z[static_cast<std::size_t>(i)] = Z;
        // Ghost: basis stays, nuclear charge and electron count drop to 0.
        atoms[static_cast<std::size_t>(i)].Z =
            is_ghost[static_cast<std::size_t>(i)] ? 0 : Z;
        // Frame positions are in Angstrom; SCF integrals live in Bohr.
        using units::convert;
        using units::Length;
        atoms[static_cast<std::size_t>(i)].pos[0] =
            convert(sp_x[i], Length::Angstrom, Length::Bohr);
        atoms[static_cast<std::size_t>(i)].pos[1] =
            convert(sp_y[i], Length::Angstrom, Length::Bohr);
        atoms[static_cast<std::size_t>(i)].pos[2] =
            convert(sp_z[i], Length::Angstrom, Length::Bohr);
        if (!is_ghost[static_cast<std::size_t>(i)])
            nelec += Z;
    }
    n_electrons = nelec;
}

// ── Generic builder driven by a checked-in basis_data table ──────────
namespace {
//
// Walks a `std::vector<basis_data::AtomBasis>` indexed by Z and emits
// one native::Shell per Cartesian AO, applying the physical GTO
// normalization (phys_norm_s .. phys_norm_g) to the stored contraction
// coefficients. `set_name` labels error messages (orbital BasisSet or
// auxiliary AuxBasis name); `max_l` is the caller's angular cap —
// kMaxLOrbital (f) for orbital bases whose shells feed the 4-center
// engines, kMaxL (g) for density-fitting auxiliary bases that only ever
// reach the 2-/3-center Rys paths.

/// @brief Elemental Z used for basis-table lookup.
///        Prefer `basis_z[a]` (true Z for ghosts); fall back to
///        `atoms[a].Z` for manually assembled AtomList (e.g. density fitting).
int basis_lookup_z(const AtomList& atoms, int a) {
    if (!atoms.basis_z.empty()) {
        if (a < 0 || a >= static_cast<int>(atoms.basis_z.size())) {
            throw std::out_of_range(
                "cpu::scf::ShellList: basis_z index out of range");
        }
        return atoms.basis_z[static_cast<std::size_t>(a)];
    }
    return atoms.atoms[static_cast<std::size_t>(a)].Z;
}

ShellList
build_shell_list_from_table(const AtomList& atoms,
                            const std::vector<basis_data::AtomBasis>& table,
                            const char* set_name, int max_l) {
    const int natm = static_cast<int>(atoms.atoms.size());

    // First pass: count AOs and validate element coverage.
    int n_basis = 0;
    for (int a = 0; a < natm; ++a) {
        const int Z = basis_lookup_z(atoms, a);
        if (Z < 1 || Z >= static_cast<int>(table.size())
            || table[static_cast<std::size_t>(Z)].shells.empty()) {
            throw std::runtime_error(std::string("cpu::scf::ShellList: '")
                                     + set_name + "' has no data for Z="
                                     + std::to_string(Z));
        }
        for (const auto& sh : table[static_cast<std::size_t>(Z)].shells) {
            if (sh.l < 0 || sh.l > max_l) {
                throw std::runtime_error(
                    std::string("cpu::scf::ShellList: '") + set_name + "' l="
                    + std::to_string(sh.l) + " exceeds the supported maximum l="
                    + std::to_string(max_l) + ".");
            }
            n_basis += n_cart_of(sh.l);
        }
    }

    ShellList out;
    out.n_basis = n_basis;
    out.shells.resize(static_cast<std::size_t>(n_basis));
    out.shell_storage.resize(static_cast<std::size_t>(n_basis));

    int shell_idx = 0;
    for (int a = 0; a < natm; ++a) {
        const int Z = basis_lookup_z(atoms, a);
        const double cx = atoms.atoms[static_cast<std::size_t>(a)].pos[0];
        const double cy = atoms.atoms[static_cast<std::size_t>(a)].pos[1];
        const double cz = atoms.atoms[static_cast<std::size_t>(a)].pos[2];

        for (const auto& sh : table[static_cast<std::size_t>(Z)].shells) {
            const int np = static_cast<int>(sh.alpha.size());

            auto emit_shell = [&](int l, int cart, double norm_at(double)) {
                auto& store =
                    out.shell_storage[static_cast<std::size_t>(shell_idx)];
                store.resize(static_cast<std::size_t>(2 * np));
                for (int p = 0; p < np; ++p) {
                    store[static_cast<std::size_t>(p)] =
                        sh.alpha[static_cast<std::size_t>(p)];
                    store[static_cast<std::size_t>(np + p)] =
                        sh.coeff[static_cast<std::size_t>(p)]
                        * norm_at(sh.alpha[static_cast<std::size_t>(p)]);
                }
                auto& s = out.shells[static_cast<std::size_t>(shell_idx)];
                s.n_prim = np;
                s.alpha = store.data();
                s.coeff = store.data() + np;
                s.center[0] = cx;
                s.center[1] = cy;
                s.center[2] = cz;
                s.l = l;
                s.cart = cart;
                ++shell_idx;
            };

            // Emit one native::Shell per Cartesian component. All
            // components of a given l share the same orientation-independent
            // normalization (libcint / PySCF cart convention).
            const int n_cart = n_cart_of(sh.l);
            double (*norm_at)(double) = (sh.l == 0)   ? native::phys_norm
                                        : (sh.l == 1) ? native::phys_norm_p
                                        : (sh.l == 2) ? native::phys_norm_d
                                        : (sh.l == 3) ? native::phys_norm_f
                                                      : native::phys_norm_g;
            for (int cart = 0; cart < n_cart; ++cart)
                emit_shell(sh.l, cart, norm_at);
        }
    }
    return out;
}

} // namespace

ShellList::ShellList(const AtomList& atoms, BasisSet basis_id) {
    const char* name = basis_set_name(basis_id);
    ShellList tmp;
    switch (basis_id) {
    case BasisSet::STO3G:
        tmp = build_shell_list_from_table(atoms, basis_data::sto3g(), name,
                                          kMaxLOrbital);
        break;
    case BasisSet::Pople631G:
        tmp = build_shell_list_from_table(atoms, basis_data::pople_631g(), name,
                                          kMaxLOrbital);
        break;
    case BasisSet::Pople631Gs:
        tmp = build_shell_list_from_table(atoms, basis_data::pople_631gs(),
                                          name, kMaxLOrbital);
        break;
    case BasisSet::Pople631pGs:
        tmp = build_shell_list_from_table(atoms, basis_data::pople_631pgs(),
                                          name, kMaxLOrbital);
        break;
    case BasisSet::AugCcPVDZ:
        tmp = build_shell_list_from_table(atoms, basis_data::aug_ccpvdz(), name,
                                          kMaxLOrbital);
        break;
    case BasisSet::Def2SVP:
        tmp = build_shell_list_from_table(atoms, basis_data::def2_svp(), name,
                                          kMaxLOrbital);
        break;
    case BasisSet::Def2TZVP:
        tmp = build_shell_list_from_table(atoms, basis_data::def2_tzvp(), name,
                                          kMaxLOrbital);
        break;
    default:
        throw std::runtime_error("cpu::scf::ShellList: unknown BasisSet");
    }
    tmp.basis = basis_id;
    *this = std::move(tmp);
}

ShellList::ShellList(const AtomList& atoms, AuxBasis aux) {
    ShellList tmp;
    switch (aux) {
    case AuxBasis::Def2JKFIT:
        // Auxiliary shells may carry l = 4 (g) — they are consumed only
        // by the 2-/3-center Rys density-fitting paths, never by the
        // 4-center orbital engines, so the cap is kMaxL rather than
        // kMaxLOrbital.
        tmp = build_shell_list_from_table(atoms, basis_data::def2_jkfit(),
                                          aux_basis_name(aux), kMaxL);
        break;
    default:
        throw std::runtime_error("cpu::scf::ShellList: unknown AuxBasis");
    }
    *this = std::move(tmp);
}

} // namespace scf
} // namespace cpu
} // namespace ATV
