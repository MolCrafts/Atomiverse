/// @file cpu/scf/integrals.cpp
/// @brief IntegralPack constructor — production compositor over the
///        private integral members (AtomList / ShellList +
///        IntegralPack::overlap/kinetic/nuclear_attraction/eri_dense).
///
/// This is the canonical entry point for assembling an `IntegralPack` from a
/// `Frame` and a `BasisSet`. The single-responsibility integral kernels are
/// private static members of `IntegralPack`, composed here:
///
/// @code
///   IntegralPack pack{frame, cpu::scf::BasisSet::STO3G};
///   // pack.S, pack.T, pack.V, pack.Hcore, pack.V_nuc are filled;
///   // pack.ERI additionally with IntegralPack{frame, basis, 0, true}.
/// @endcode

#include "atomiverse/cpu/scf/internal/integrals.h"

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/ecp_potential.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"
#include "atomiverse/cpu/scf/internal/ecp_integrals.h"
#include "atomiverse/cpu/scf/internal/one_electron.h"
#include "atomiverse/cpu/scf/internal/cart2sph.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
#include "atomiverse/cpu/scf/internal/shell_pair.h"

#include <xtensor-blas/xlinalg.hpp>

namespace ATV {
namespace cpu {
namespace scf {

IntegralPack::IntegralPack(const Frame& frame, BasisSet basis, int charge,
                           bool compute_eri, std::string ecp_name,
                           const std::vector<int>& ghost_atoms,
                           bool compute_pvp) {
    const auto t_total = LogClock::now();

    AtomList atom_list{frame, charge, ghost_atoms};
    ShellList shell_list{atom_list, basis};

    // ── ECP: valence-only description for atoms in the named table ───────
    // Resolved BEFORE the integral kernels so nuclear attraction and
    // nuclear repulsion see Z_val = Z − n_core and the electron count drops
    // by n_core per hit atom. The shell list above used the full Z (basis
    // rows are keyed by the element), so this runs after its construction.
    ecp = std::move(ecp_name);
    std::vector<internal::EcpCenter> ecp_centers;
    if (!ecp.empty()) {
        const auto& table = EcpPotential::family_table(ecp); // throws unknown
        for (auto& at : atom_list.atoms) {
            if (at.Z < 1 || at.Z >= static_cast<int>(table.size()))
                continue;
            const auto& row = table[static_cast<std::size_t>(at.Z)];
            if (row.channels.empty())
                continue; // all-electron element in this family
            ecp_centers.push_back(internal::EcpCenter{
                {at.pos[0], at.pos[1], at.pos[2]}, &row});
            atom_list.n_electrons -= row.n_core;
            at.Z -= row.n_core;
        }
        has_ecp = !ecp_centers.empty();
    }

    ATV_INFO(
        "[integrals] start natm={} basis={} n_basis={} electrons={} charge={}"
        " ecp={} ecp_atoms={}",
        static_cast<int>(atom_list.atoms.size()), basis_set_name(basis),
        shell_list.n_basis, atom_list.n_electrons, charge,
        ecp.empty() ? "-" : ecp.c_str(),
        static_cast<int>(ecp_centers.size()));

    n_basis = shell_list.n_basis;
    n_electrons = atom_list.n_electrons;

    const auto t_1e = LogClock::now();
    S = overlap(shell_list);
    T = kinetic(shell_list);
    V = nuclear_attraction(shell_list, atom_list);
    if (compute_pvp)
        W = pvp(shell_list, atom_list);
    if (has_ecp) {
        // Semi-local ECP matrix U in the Cartesian engine basis, added into
        // the nuclear-attraction block so the spherical sandwich below (and
        // Hcore = T + V) transforms it along with everything else.
        // (x2c is mutually exclusive with ECP at the method level; W is
        // never assembled together with U.)
        internal::EcpIntegrals ecp_ints{shell_list.shells,
                                        std::move(ecp_centers)};
        V += ecp_ints.matrix();
    }
    Hcore = T + V;
    ATV_INFO("[integrals] one-electron done ms={:.1f}", elapsed_ms(t_1e));

    if (compute_eri) {
        const auto t_2e = LogClock::now();
        ERI = eri_dense(shell_list);
        ATV_INFO("[integrals] two-electron done ms={:.1f}", elapsed_ms(t_2e));
    }

    // Spherical-convention bases (def2 family): the Cartesian engines above
    // stay untouched; the pack-level matrices are sandwiched to the 2l+1
    // spherical AO basis with the block-diagonal engine-convention transform
    // (CartToSph::block per shell block). Fock builds bridge back through
    // `c2s` (HamiltonianBuilder), so the transform lives here exactly once.
    if (angular_convention(basis) == basis_data::AngularConvention::Spherical) {
        const internal::CartToSph c2s_tables;
        const auto blocks = blocks_from_shells(shell_list.shells);
        int n_sph = 0;
        for (const auto& b : blocks)
            n_sph += 2 * b.l + 1;
        n_basis_cart = n_basis;
        c2s = xt::zeros<double>(
            {std::size_t(n_basis_cart), std::size_t(n_sph)});
        // Per-block construction, self-normalized against the engine's own
        // Cartesian overlap: the engine primitive-normalizes but does NOT
        // unit-normalize contracted AOs, so the tabulated block(l) (which
        // assumes unit-normalized components) is first rescaled by
        // 1/sqrt(S_ii) per Cartesian row, and each spherical column is then
        // normalized in the S_cart metric. Row ratios (the double-factorial
        // pitfall) come from block(l); the column normalization makes the
        // spherical AOs exactly unit — convention-independent by
        // construction, degenerating to block(l) when S_ii == 1.
        int col = 0;
        for (const auto& b : blocks) {
            const auto& m = c2s_tables.block(b.l);
            const int nc = b.n_cart;
            const int ns = 2 * b.l + 1;
            for (int j = 0; j < ns; ++j) {
                std::vector<double> v(nc);
                for (int i = 0; i < nc; ++i)
                    v[i] = m(i, j)
                           / std::sqrt(S(b.ao_start + i, b.ao_start + i));
                double nrm2 = 0.0;
                for (int i = 0; i < nc; ++i)
                    for (int k = 0; k < nc; ++k)
                        nrm2 +=
                            v[i] * S(b.ao_start + i, b.ao_start + k) * v[k];
                const double inv = 1.0 / std::sqrt(nrm2);
                for (int i = 0; i < nc; ++i)
                    c2s(b.ao_start + i, col + j) = v[i] * inv;
            }
            col += ns;
        }
        const auto ct = xt::transpose(c2s);
        S = xt::linalg::dot(ct, xt::linalg::dot(S, c2s));
        T = xt::linalg::dot(ct, xt::linalg::dot(T, c2s));
        V = xt::linalg::dot(ct, xt::linalg::dot(V, c2s));
        if (W.size() > 0)
            W = xt::linalg::dot(ct, xt::linalg::dot(W, c2s));
        Hcore = T + V;
        if (ERI.size() > 0)
            ERI = eri_to_spherical(ERI, c2s);
        n_basis = n_sph;
        ATV_INFO("[integrals] spherical convention: n_basis {} -> {}",
                 n_basis_cart, n_basis);
    }

    V_nuc = nuclear_repulsion_energy(atom_list);

    // IntegralPack preserves owned shell storage because downstream
    // consumers (UKS grid, gradient, ASPC) hold references to it.
    atoms = atom_list.atoms;
    shell_data = shell_list.shell_storage;
    shells.resize(shell_list.n_basis);
    for (int i = 0; i < shell_list.n_basis; ++i) {
        shells[i] = shell_list.shells[i];
        const int np = shells[i].n_prim;
        shells[i].alpha = shell_data[i].data();
        shells[i].coeff = shell_data[i].data() + np;
    }

    ATV_INFO("[integrals] done total_ms={:.1f} v_nuc={:.6f}",
             elapsed_ms(t_total), V_nuc);
}

xt::xtensor<double, 4>
IntegralPack::eri_to_spherical(const xt::xtensor<double, 4>& eri_cart,
                               const xt::xtensor<double, 2>& c2s) {
    // Four-index transform, one index at a time (small systems only — the
    // dense path is validation/MP2/FCI scale by charter).
    const std::size_t nc = c2s.shape()[0];
    const std::size_t n_sph = c2s.shape()[1];
    auto e1 = xt::reshape_view(eri_cart, {nc * nc * nc, nc});
    xt::xtensor<double, 2> h1 = xt::linalg::dot(e1, c2s); // l index
    // rotate axes: (ijk,l)->(l,ijk) etc., four times
    xt::xtensor<double, 4> cur =
        xt::reshape_view(h1, {nc, nc, nc, n_sph});
    for (int rep = 0; rep < 3; ++rep) {
        xt::xtensor<double, 4> perm = xt::transpose(cur, {3, 0, 1, 2});
        const auto sh = perm.shape();
        auto flat = xt::reshape_view(perm, {sh[0] * sh[1] * sh[2], sh[3]});
        xt::xtensor<double, 2> hx = xt::linalg::dot(flat, c2s);
        cur = xt::reshape_view(hx, {sh[0], sh[1], sh[2], n_sph});
    }
    return xt::transpose(cur, {3, 0, 1, 2});
}

xt::xtensor<double, 4> IntegralPack::eri_working() const {
    if (shells.empty())
        throw std::runtime_error(
            "IntegralPack::eri_working: empty shell workspace — the pack "
            "was not built from a Frame");
    // Local view of the stored shells: the Shell descriptors point into this
    // pack's owned shell_data, which stays alive for the whole call.
    ShellList shell_list;
    shell_list.shells = shells;
    shell_list.n_basis = static_cast<int>(shells.size());
    xt::xtensor<double, 4> eri = eri_dense(shell_list);
    if (spherical())
        eri = eri_to_spherical(eri, c2s);
    return eri;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
