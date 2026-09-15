/// @file cpu/scf/internal/minao.cpp
/// @brief MINAO projector implementation.
///
/// Algorithm (matches pyscf.scf.hf.init_guess_by_minao exactly):
///   1. For each atom, emit the MINAO basis shells at that atom's center.
///      Each contraction is renormalized so <phi|phi>=1 in the target
///      conventions (physical-normalization primitives).
///   2. Compute cross-overlap S_cross[i, j] = <target_i | minao_j>.
///   3. Solve the overdetermined-looking projection:
///        mo = solve(S_target, S_cross)
///      (S_target is square [n_t, n_t], S_cross is [n_t, n_m]).
///      Equivalent to PySCF's `lib.cho_solve(s22, s21)`.
///   4. Assemble density:
///        P = mo * diag(occ) * mo^T
///      where occ is the per-atom fractional-occupation vector appended
///      in AO order.
#include "atomiverse/cpu/scf/internal/minao.h"

#include <cmath>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/minao_data.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {

namespace {

// Per-AO storage: exponents concatenated with normalized-coeffs,
// plus the native::Shell view pointing into this storage.
struct MinaoAO {
    std::vector<double> data; // [n_prim alpha] + [n_prim coeff (normalized)]
    native::Shell shell;
    double occupation;
};

// Build all MINAO AOs for the molecule. `storage` holds the per-AO exponent
// and coefficient data; `shell` views into the owned vectors, so `storage`
// must outlive any use of the shell pointers.
static void build_minao_ao_list(const internal::IntegralView& integrals,
                                std::vector<MinaoAO>& out) {
    out.clear();
    for (const auto& atom : integrals.atoms) {
        const auto* basis = minao::element_basis(atom.Z);
        if (basis == nullptr)
            throw std::runtime_error("MINAO: unsupported element Z="
                                     + std::to_string(atom.Z));

        int ao_in_elem = 0;
        for (int sh = 0; sh < basis->n_shells; ++sh) {
            const auto& mshell = basis->shells[sh];
            const int np = mshell.n_prim;
            for (int c = 0; c < mshell.n_contract; ++c) {
                // One contraction -> (2l+1) Cartesian AOs sharing the same radial part.
                // Compute the normalized-primitive coefficients once per contraction.
                std::vector<double> base(2 * np);
                for (int p = 0; p < np; ++p) {
                    base[p] = mshell.alpha[p];
                    // primitive physical normalization
                    double N = (mshell.l == 0)
                                   ? native::phys_norm(mshell.alpha[p])
                                   : native::phys_norm_p(mshell.alpha[p]);
                    // coeff layout in minao_data: row-major [n_prim][n_contract]
                    double raw_c = mshell.coeff[p * mshell.n_contract + c];
                    base[np + p] = raw_c * N;
                }

                // Renormalize the contracted function: <phi|phi>=1.
                // We build a temporary Shell, compute overlap, divide coeffs by sqrt.
                native::Shell probe;
                probe.n_prim = np;
                probe.alpha = base.data();
                probe.coeff = base.data() + np;
                probe.center[0] = atom.pos[0];
                probe.center[1] = atom.pos[1];
                probe.center[2] = atom.pos[2];
                probe.l = mshell.l;
                probe.cart = 0; // l=0 -> 0; for l=1 the three Cartesians share
                // the same radial self-overlap, so cart=0 suffices.
                double self = native::overlap(probe, probe);
                double inv_norm = 1.0 / std::sqrt(self);
                for (int p = 0; p < np; ++p)
                    base[np + p] *= inv_norm;

                const int degen = 2 * mshell.l + 1; // l=0 -> 1, l=1 -> 3
                for (int cart = 0; cart < degen; ++cart) {
                    MinaoAO ao;
                    ao.data = base; // one copy per AO so pointers are stable
                    ao.shell.n_prim = np;
                    ao.shell.alpha = ao.data.data();
                    ao.shell.coeff = ao.data.data() + np;
                    ao.shell.center[0] = atom.pos[0];
                    ao.shell.center[1] = atom.pos[1];
                    ao.shell.center[2] = atom.pos[2];
                    ao.shell.l = mshell.l;
                    ao.shell.cart = cart;
                    ao.occupation = basis->occupations[ao_in_elem];
                    out.push_back(std::move(ao));
                    ++ao_in_elem;
                }
            }
        }
    }
}

} // namespace

bool minao_supported(const internal::IntegralView& integrals) {
    // Spherical-convention packs: the MINAO cross-overlap machinery is
    // Cartesian; the MINAO→spherical projection is deferred. Falling back to
    // the Hcore guess keeps the converged energy identical — the guess only
    // affects the iteration count.
    if (integrals.spherical())
        return false;
    // ECP packs: the MINAO reference densities are all-electron; projecting
    // them onto a valence-only pack would over-count the (removed) core.
    // Falling back to the Hcore guess keeps the converged energy identical.
    if (integrals.has_ecp)
        return false;
    for (const auto& atom : integrals.atoms) {
        if (minao::element_basis(atom.Z) == nullptr)
            return false;
    }
    return true;
}

RealMatrix build_minao_density(const internal::IntegralView& integrals) {
    // ── 1. Build MINAO AO list (and owning storage) ─────────────────────
    std::vector<MinaoAO> minao_aos;
    build_minao_ao_list(integrals, minao_aos);

    const int n_target = integrals.n_basis;
    const int n_minao = static_cast<int>(minao_aos.size());

    // ── 2. Target-basis shells already normalized by build_integrals
    //        — just borrow the pointers.
    //        Reconstruct native::Shell views from the internal::IntegralWorkspace to avoid
    //        depending on its internal shell_data layout past its lifetime.
    //        Use the already-stored `shells` vector directly.
    const auto& target_shells = integrals.shells;
    if (static_cast<int>(target_shells.size()) != n_target)
        throw std::runtime_error(
            "MINAO: target shell count != n_basis (expected one Shell per AO)");

    // ── 3. Cross-overlap S_cross[i, j] = <target_i | minao_j> ──────────
    RealMatrix S_cross = xt::zeros<double>({n_target, n_minao});
    for (int i = 0; i < n_target; ++i) {
        for (int j = 0; j < n_minao; ++j) {
            S_cross(i, j) =
                native::overlap(target_shells[i], minao_aos[j].shell);
        }
    }

    // ── 4. Solve S_target * mo = S_cross   (PySCF's cho_solve) ─────────
    //       xt::linalg::solve dispatches to LAPACK dgesv.
    RealMatrix S_target = integrals.S; // copy (dgesv overwrites A)
    RealMatrix mo = xt::linalg::solve(S_target, S_cross);

    // ── 5. Assemble P = mo * diag(occ) * mo^T ──────────────────────────
    //        Rewrite as (mo * occ) · mo^T for a single DGEMM.
    RealMatrix mo_scaled = xt::zeros<double>({n_target, n_minao});
    for (int i = 0; i < n_target; ++i)
        for (int j = 0; j < n_minao; ++j)
            mo_scaled(i, j) = mo(i, j) * minao_aos[j].occupation;
    RealMatrix P = xt::linalg::dot(mo_scaled, xt::transpose(mo));

    // Symmetrize (guard against round-off asymmetry).
    P = 0.5 * (P + xt::transpose(P));
    return P;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
