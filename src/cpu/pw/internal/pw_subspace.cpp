/// @file cpu/pw/internal/pw_subspace.cpp
/// @brief Davidson subspace utilities (moved verbatim from pw_davidson.cpp).

#include "atomiverse/cpu/pw/internal/pw_subspace.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {

/// Deterministic LCG in [-0.5, 0.5) for the atomic+random initial-guess
/// perturbation. Seeded per subspace-vector index so the result is
/// independent of which MPI rank handles which k-point (std::rand's
/// global stream would make the wavefunction guess rank-layout
/// dependent).
struct TinyRng {
    std::uint32_t state;
    explicit TinyRng(std::uint32_t seed) : state(seed * 2654435761u + 1u) {}
    AccReal next() {
        state = state * 1664525u + 1013904223u;
        return AccReal(state) / AccReal(4294967296.0) - AccReal(0.5);
    }
};

// ── Real spherical harmonics Y_lm(Ĝ) ──

AccReal y00(AccReal, AccReal, AccReal) {
    return AccReal(1) / std::sqrt(AccReal(4.0 * M_PI));
}

void y1m(AccReal x, AccReal y, AccReal z, AccReal r, AccReal* out) {
    AccReal n = std::sqrt(AccReal(3) / (AccReal(4.0 * M_PI)));
    AccReal ir = (r > AccReal(1e-16)) ? AccReal(1) / r : AccReal(0);
    out[0] = n * y * ir; // m=-1 (p_y)
    out[1] = n * z * ir; // m=0  (p_z)
    out[2] = n * x * ir; // m=+1 (p_x)
}

void y2m(AccReal x, AccReal y, AccReal z, AccReal r, AccReal* out) {
    AccReal ir2 = (r > AccReal(1e-16)) ? AccReal(1) / (r * r) : AccReal(0);
    out[0] = std::sqrt(AccReal(15) / (AccReal(4.0 * M_PI))) * x * y
             * ir2; // m=-2 (d_xy)
    out[1] = std::sqrt(AccReal(15) / (AccReal(4.0 * M_PI))) * y * z
             * ir2; // m=-1 (d_yz)
    out[2] = std::sqrt(AccReal(5) / (AccReal(16.0 * M_PI)))
             * (AccReal(3) * z * z * ir2 - AccReal(1)); // m=0 (d_z²)
    out[3] = std::sqrt(AccReal(15) / (AccReal(4.0 * M_PI))) * x * z
             * ir2; // m=+1 (d_xz)
    out[4] = std::sqrt(AccReal(15) / (AccReal(16.0 * M_PI))) * (x * x - y * y)
             * ir2; // m=+2 (d_x²-y²)
}

} // namespace

void compute_ylm(AccReal x, AccReal y, AccReal z, AccReal r, int l,
                 AccReal* out) {
    switch (l) {
    case 0:
        out[0] = y00(x, y, z);
        break;
    case 1:
        y1m(x, y, z, r, out);
        break;
    case 2:
        y2m(x, y, z, r, out);
        break;
    default:
        for (int m = 0; m < n_m_for_l(l); ++m)
            out[m] = 0;
        break;
    }
}

void jacobi_eigh(int n, AccReal* A, AccReal* eig, AccReal* vecs,
                 int max_sweep) {
    // Initialize eigenvectors to identity
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            vecs[i * n + j] = (i == j) ? AccReal(1) : AccReal(0);
        eig[i] = A[i * n + i];
    }

    for (int sweep = 0; sweep < max_sweep; ++sweep) {
        AccReal max_off = 0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                max_off = std::max(max_off, std::abs(A[i * n + j]));
        if (max_off < AccReal(1e-14))
            break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                AccReal apq = A[p * n + q];
                if (std::abs(apq) < AccReal(1e-14))
                    continue;
                AccReal app = A[p * n + p], aqq = A[q * n + q];
                AccReal theta =
                    AccReal(0.5) * std::atan2(AccReal(2) * apq, aqq - app);
                AccReal c = std::cos(theta), s = std::sin(theta);

                // Rotate A
                A[p * n + p] =
                    c * c * app + s * s * aqq - AccReal(2) * c * s * apq;
                A[q * n + q] =
                    s * s * app + c * c * aqq + AccReal(2) * c * s * apq;
                A[p * n + q] = A[q * n + p] = AccReal(0);
                for (int r = 0; r < n; ++r) {
                    if (r == p || r == q)
                        continue;
                    AccReal arp = A[r * n + p], arq = A[r * n + q];
                    A[r * n + p] = A[p * n + r] = c * arp - s * arq;
                    A[r * n + q] = A[q * n + r] = s * arp + c * arq;
                }
                // Rotate eigenvectors
                for (int r = 0; r < n; ++r) {
                    AccReal vrp = vecs[r * n + p], vrq = vecs[r * n + q];
                    vecs[r * n + p] = c * vrp - s * vrq;
                    vecs[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i)
        eig[i] = A[i * n + i];

    // Sort eigenpairs ascending: callers take the first n_bands columns
    // as the occupied bands, which is only correct in ascending order.
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return eig[a] < eig[b]; });
    std::vector<AccReal> eig_s(n), vecs_s(static_cast<size_t>(n) * n);
    for (int k = 0; k < n; ++k) {
        eig_s[k] = eig[order[k]];
        for (int i = 0; i < n; ++i)
            vecs_s[i * n + k] = vecs[i * n + order[k]];
    }
    for (int i = 0; i < n; ++i)
        eig[i] = eig_s[i];
    for (size_t i = 0; i < vecs_s.size(); ++i)
        vecs[i] = vecs_s[i];
}

int init_subspace(const PWWorkspace& ws, int n_bands,
                  std::vector<std::complex<AccReal>>& sub_vectors) {
    int ng = ws.n_g;
    int nb = n_bands;

    // ── Initialize subspace: lowest T(G) plane waves ──
    struct DiagElem {
        AccReal val;
        int idx;
    };
    std::vector<DiagElem> diag(ng);
    for (int i = 0; i < ng; ++i)
        diag[i] = {ws.T_G[i], i};
    std::sort(
        diag.begin(), diag.end(),
        [](const DiagElem& a, const DiagElem& b) { return a.val < b.val; });

    // Count atomic wavefunction states: n_atoms × Σ (2l+1)
    int n_atom_states = 0;
    for (size_t ia = 0; ia < ws.atom_wfc.size(); ++ia)
        for (size_t iw = 0; iw < ws.atom_wfc[ia].size(); ++iw)
            n_atom_states += 2 * ws.atom_wfc[ia][iw].l + 1;

    int n_sub;
    if (n_atom_states >= nb) {
        n_sub = n_atom_states; // QE: start from atomic wavefunctions
    } else {
        n_sub = std::min(ng, nb * 4);
    }
    sub_vectors.assign(static_cast<size_t>(n_sub) * ng,
                       std::complex<AccReal>(0, 0));

    if (n_atom_states >= nb) {
        // Initialize subspace with atomic wavefunctions (QE atomic_wfc_mod.f90)
        // ψ_{nlm}(G) = i^l · Y_lm(Ĝ) · chi_G(G) · exp(-iG·R)
        int j = 0;
        auto& gv = *ws.gvec;
        for (size_t ia = 0; ia < ws.atom_wfc.size() && j < n_sub; ++ia) {
            AccReal ax = ws.atom_pos_bohr[ia][0], ay = ws.atom_pos_bohr[ia][1],
                    az = ws.atom_pos_bohr[ia][2];
            for (size_t iw = 0; iw < ws.atom_wfc[ia].size() && j < n_sub;
                 ++iw) {
                int l = ws.atom_wfc[ia][iw].l;
                int nm = 2 * l + 1;
                for (int m = 0; m < nm && j < n_sub; ++m) {
                    for (int ig = 0; ig < ng; ++ig) {
                        // Bloch states live on q = k+G.
                        AccReal Gx = gv.g[ig][0] + gv.kvec[0],
                                Gy = gv.g[ig][1] + gv.kvec[1],
                                Gz = gv.g[ig][2] + gv.kvec[2];
                        AccReal Gr = std::sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
                        AccReal chi = ws.atom_wfc[ia][iw].chi_G[ig];
                        if (std::abs(chi) < AccReal(1e-30)) {
                            sub_vectors[j * ng + ig] =
                                std::complex<AccReal>(0, 0);
                            continue;
                        }
                        // i^l factor (QE: lphase = (0,1)^l)
                        std::complex<AccReal> il(1, 0);
                        if (l == 1)
                            il = std::complex<AccReal>(0, 1);
                        else if (l == 2)
                            il = std::complex<AccReal>(-1, 0);
                        // Y_lm(Ĝ)
                        std::vector<AccReal> ylm(nm);
                        compute_ylm(Gx, Gy, Gz, Gr, l, ylm.data());
                        // Structure factor S(G) = exp(-iG·R)
                        AccReal phase = Gx * ax + Gy * ay + Gz * az;
                        std::complex<AccReal> S(std::cos(phase),
                                                -std::sin(phase));
                        sub_vectors[j * ng + ig] = il * ylm[m] * chi * S;
                    }
                    // QE startingwfc='atomic+random': add random perturbation
                    // (deterministic per state index — see TinyRng).
                    TinyRng rng(static_cast<std::uint32_t>(j + 1));
                    for (int ig = 0; ig < ng; ++ig) {
                        AccReal rnd = rng.next();
                        sub_vectors[j * ng + ig] += std::complex<AccReal>(
                            rnd * AccReal(1e-4), AccReal(0));
                    }
                    ++j;
                }
            }
        }
        n_sub = j; // actual count
    } else {
        // Fallback: lowest T(G) plane waves
        for (int j = 0; j < n_sub; ++j) {
            int g_idx = diag[j].idx;
            for (int ig = 0; ig < ng; ++ig)
                sub_vectors[j * ng + ig] = (ig == g_idx)
                                               ? std::complex<AccReal>(1, 0)
                                               : std::complex<AccReal>(0, 0);
        }
    }

    // ── Orthonormalize (modified Gram-Schmidt, drop dependent columns) ──
    // The Davidson Rayleigh-Ritz treats the subspace as orthonormal; the
    // atomic-wavefunction guess is strongly non-orthogonal for multi-atom
    // cells (bonding/antibonding overlap), which biases the eigenpairs and
    // puts a noise floor under the SCF density. Plane-wave columns pass
    // through unchanged.
    int kept = 0;
    for (int j = 0; j < n_sub; ++j) {
        auto* vj = &sub_vectors[static_cast<size_t>(j) * ng];
        for (int pass = 0; pass < 2; ++pass) {
            for (int k = 0; k < kept; ++k) {
                const auto* vk = &sub_vectors[static_cast<size_t>(k) * ng];
                std::complex<AccReal> ov(0, 0);
                for (int ig = 0; ig < ng; ++ig)
                    ov += std::conj(vk[ig]) * vj[ig];
                for (int ig = 0; ig < ng; ++ig)
                    vj[ig] -= ov * vk[ig];
            }
        }
        AccReal nrm2 = 0;
        for (int ig = 0; ig < ng; ++ig)
            nrm2 += std::norm(vj[ig]);
        if (nrm2 < AccReal(1e-12))
            continue; // linearly dependent — drop
        AccReal inv = AccReal(1) / std::sqrt(nrm2);
        auto* vk = &sub_vectors[static_cast<size_t>(kept) * ng];
        for (int ig = 0; ig < ng; ++ig)
            vk[ig] = vj[ig] * inv;
        ++kept;
    }
    n_sub = kept;

    // Top up with lowest-T(G) plane waves if drops left fewer than nb.
    for (int d = 0; n_sub < nb && d < ng; ++d) {
        auto* vj = &sub_vectors[static_cast<size_t>(n_sub) * ng];
        for (int ig = 0; ig < ng; ++ig)
            vj[ig] = std::complex<AccReal>(0, 0);
        vj[diag[d].idx] = std::complex<AccReal>(1, 0);
        for (int k = 0; k < n_sub; ++k) {
            const auto* vk = &sub_vectors[static_cast<size_t>(k) * ng];
            std::complex<AccReal> ov(0, 0);
            for (int ig = 0; ig < ng; ++ig)
                ov += std::conj(vk[ig]) * vj[ig];
            for (int ig = 0; ig < ng; ++ig)
                vj[ig] -= ov * vk[ig];
        }
        AccReal nrm2 = 0;
        for (int ig = 0; ig < ng; ++ig)
            nrm2 += std::norm(vj[ig]);
        if (nrm2 < AccReal(1e-6))
            continue;
        AccReal inv = AccReal(1) / std::sqrt(nrm2);
        for (int ig = 0; ig < ng; ++ig)
            vj[ig] *= inv;
        ++n_sub;
    }
    return n_sub;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
