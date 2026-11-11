/// @file cpu/scf/internal/cartesian_shell_pack.h
/// @brief Host pack of Cartesian contracted shells for the CUDA l>1 RHF path.
///
/// Setup only: exponents, raw contraction coefficients, and AO maps. No
/// integrals. The device step evaluates them (`launch_1e_integrals_lgt1`,
/// `launch_eri_lgt1`, l ≤ 3). PySCF `scf.hf.get_jk` / `scf.hf.get_veff` consume
/// Cartesian `int2e` (`pyscf.scf._vhf.direct` with `intor='int2e_cart'`).
/// Spherical bases then apply the same CartToSph sandwich as the CPU pack.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/gto.h"
#include "atomiverse/cpu/scf/basis_data/accessors.h"
#include "atomiverse/cpu/scf/basis_data/aug_ccpvdz.h"
#include "atomiverse/cpu/scf/basis_data/ccpvdz.h"
#include "atomiverse/cpu/scf/basis_data/def2_svp.h"
#include "atomiverse/cpu/scf/basis_data/def2_tzvp.h"
#include "atomiverse/cpu/scf/basis_data/pople_631g.h"
#include "atomiverse/cpu/scf/basis_data/pople_631gs.h"
#include "atomiverse/cpu/scf/basis_data/pople_631pgs.h"
#include "atomiverse/cpu/scf/basis_data/sto3g.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/cart.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Highest angular momentum `launch_eri_lgt1` evaluates (Cartesian f).
inline constexpr int kCudaCartesianLgt1Max = 3;

/// @brief Named when a shell has l ≥ 4 (no device g-ERI).
inline constexpr const char* kMissingKernelEriG = "launch_eri_g";

/// @brief Named when Cartesian l ≤ 2 forces are requested (energy kernel only).
inline constexpr const char* kMissingKernelForcesLgt1 =
    "launch_rhf_forces_lgt1";

/// @brief Which CUDA RHF integral path a basis + nuclear charges take.
enum class CudaRhfRouteKind {
    Sto3gSp,       ///< Uniform STO-3G, every shell l ≤ 1 (`launch_*_sp`).
    CartesianLgt1, ///< Cartesian shells, max l ≤ 3 (spherical uses c2s).
    MissingKernel, ///< Device kernel does not exist; do not host-fill ERIs.
};

/// @brief Route plus the kernel name to throw when @ref MissingKernel.
struct CudaRhfRoute {
    CudaRhfRouteKind kind = CudaRhfRouteKind::MissingKernel;
    /// @brief Non-null only for @ref CudaRhfRouteKind::MissingKernel.
    const char* missing_kernel = nullptr;
};

/// @brief Ragged contracted-shell list (one angular shell, not one AO).
///
/// Coefficients are the table values, **not** physically normalized.
/// `launch_*_lgt1` multiplies by `contraction_norm(l, alpha)`.
struct CartesianShellPack {
    int n_shells = 0;
    int n_ao = 0;
    int max_l = 0;
    std::vector<double> alpha;
    std::vector<double> coeff;
    std::vector<int> nprim;
    std::vector<int> prim_off;
    std::vector<int> L;
    std::vector<int> atom_of;
    std::vector<int> ao_start;
    std::vector<int> ao_shell;
    std::vector<int> ao_cart;
};

/// @brief libcint / PySCF Cartesian primitive norm for shell angular momentum.
/// @param l Angular momentum (0, 1, 2, or 3).
/// @param alpha Gaussian exponent [Bohr^-2].
/// @return Physical normalization factor.
/// @throws std::invalid_argument when `l` is outside 0..3.
inline double contraction_norm(int l, double alpha) {
    switch (l) {
    case 0:
        return ATV::phys_norm_s(alpha);
    case 1:
        return ATV::phys_norm_p(alpha);
    case 2:
        return ATV::phys_norm_d(alpha);
    case 3:
        return ATV::phys_norm_f(alpha);
    default:
        throw std::invalid_argument(
            "contraction_norm: l=" + std::to_string(l)
            + " is outside the Cartesian l<=3 device envelope");
    }
}

/// @brief Basis table for `basis` (empty shells mark an unsupported Z).
inline const std::vector<basis_data::AtomBasis>& basis_table(BasisSet basis) {
    switch (basis) {
    case BasisSet::STO3G:
        return basis_data::sto3g();
    case BasisSet::Pople631G:
        return basis_data::pople_631g();
    case BasisSet::Pople631Gs:
        return basis_data::pople_631gs();
    case BasisSet::Pople631pGs:
        return basis_data::pople_631pgs();
    case BasisSet::Def2SVP:
        return basis_data::def2_svp();
    case BasisSet::Def2TZVP:
        return basis_data::def2_tzvp();
    case BasisSet::AugCcPVDZ:
        return basis_data::aug_ccpvdz();
    case BasisSet::CcPVDZ:
        return basis_data::ccpvdz();
    }
    throw std::invalid_argument("basis_table: unknown BasisSet");
}

/// @brief Walk contracted shells of `basis` on `Z[0..natm)`.
///
/// One call, used by the pack and by the route classifier, so the
/// missing-element check is not a second copy of the table loop.
/// @param fn Called as `fn(atom, shell)` in table order.
/// @throws std::invalid_argument when `natm<=0` or `Z` is null.
/// @throws std::runtime_error when an element has no row, or a contraction
///         is empty.
template <typename Fn>
void for_each_contracted_shell(BasisSet basis, const int* Z, int natm,
                               Fn&& fn) {
    if (natm <= 0 || Z == nullptr) {
        throw std::invalid_argument(
            "pack_cartesian_shells: need a non-empty nuclear-charge list");
    }
    const auto& table = basis_table(basis);
    const char* name = basis_set_name(basis);
    for (int a = 0; a < natm; ++a) {
        const int z = Z[a];
        if (z < 1 || static_cast<std::size_t>(z) >= table.size()
            || table[static_cast<std::size_t>(z)].shells.empty()) {
            throw std::runtime_error(std::string("pack_cartesian_shells: '")
                                     + name + "' has no data for Z="
                                     + std::to_string(z));
        }
        for (const auto& sh : table[static_cast<std::size_t>(z)].shells) {
            if (sh.alpha.empty() || sh.alpha.size() != sh.coeff.size()) {
                throw std::runtime_error(std::string("pack_cartesian_shells: '")
                                         + name + "' Z=" + std::to_string(z)
                                         + " has an empty contraction");
            }
            fn(a, sh);
        }
    }
}

/// @brief Pack Cartesian shells for `Z[0..natm)`.
///
/// Shell order matches `ShellList`: table order per atom, Cartesian
/// components in `fill_cart_comps` order (not emitted here — one record
/// per angular shell). Geometry is not stored; the device expands centers
/// from `atom_of`.
/// @param basis Orbital basis.
/// @param Z Nuclear charges / elemental numbers [natm].
/// @param natm Atom count.
/// @throws std::invalid_argument when `natm<=0` or `Z` is null.
/// @throws std::runtime_error when an element has no row in the table.
inline CartesianShellPack pack_cartesian_shells(BasisSet basis, const int* Z,
                                                int natm) {
    CartesianShellPack out;
    int ao = 0;
    for_each_contracted_shell(basis, Z, natm, [&](int a, const auto& sh) {
        const int np = static_cast<int>(sh.alpha.size());
        const int s = out.n_shells;
        out.nprim.push_back(np);
        out.prim_off.push_back(static_cast<int>(out.alpha.size()));
        out.L.push_back(sh.l);
        out.atom_of.push_back(a);
        out.ao_start.push_back(ao);
        out.alpha.insert(out.alpha.end(), sh.alpha.begin(), sh.alpha.end());
        out.coeff.insert(out.coeff.end(), sh.coeff.begin(), sh.coeff.end());
        if (sh.l > out.max_l)
            out.max_l = sh.l;
        const int ncart = n_cart_of(sh.l);
        for (int c = 0; c < ncart; ++c) {
            out.ao_shell.push_back(s);
            out.ao_cart.push_back(c);
            ++ao;
        }
        ++out.n_shells;
    });
    out.n_ao = ao;
    return out;
}

/// @brief True when every atom is uniform-primitive s/p STO-3G.
inline bool sto3g_sp_uniform(BasisSet basis, const int* Z, int natm) {
    if (basis != BasisSet::STO3G || natm <= 0 || Z == nullptr)
        return false;
    int n_prim0 = -1;
    for (int i = 0; i < natm; ++i) {
        if (!basis_data::is_sp_sto3g(Z[i]))
            return false;
        const int np = basis_data::sp_sto3g_n_prim(Z[i]);
        if (n_prim0 < 0)
            n_prim0 = np;
        else if (np != n_prim0)
            return false;
    }
    return true;
}

/// @brief Pick the CUDA RHF integral route. Does not compute integrals.
/// @param basis Orbital basis.
/// @param Z Nuclear charges [natm].
/// @param natm Atom count.
inline CudaRhfRoute classify_cuda_rhf(BasisSet basis, const int* Z, int natm) {
    if (sto3g_sp_uniform(basis, Z, natm))
        return {CudaRhfRouteKind::Sto3gSp, nullptr};
    int max_l = 0;
    for_each_contracted_shell(basis, Z, natm, [&](int, const auto& sh) {
        if (sh.l > max_l)
            max_l = sh.l;
    });
    if (max_l > kCudaCartesianLgt1Max)
        return {CudaRhfRouteKind::MissingKernel, kMissingKernelEriG};
    // Spherical l ≤ 3 stays on this route. The fock build sandwiches
    // S, T, V, and the Cartesian ERI with CartToSph (integrals.cpp).
    return {CudaRhfRouteKind::CartesianLgt1, nullptr};
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
