/// @file cpu/planewave/internal/davidson.cpp
/// @brief Iterative block Davidson with FFT-based H·ψ and KB non-local potential.
///
/// Subspace utilities (Y_lm, Jacobi eigensolver, initial subspace) live in
/// internal/pw_subspace.cpp, shared with the CUDA band solver.
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/planewave/internal/pw_subspace.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <fftw3.h>
#include <stdexcept>
#include <vector>
#include <xtensor-blas/xblas.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

using internal::n_m_for_l;

namespace {

// ── Pad G-sphere vector to full c2c FFT grid ──

void pad_to_fft(const PWWorkspace& ws, const std::complex<AccReal>* psi_g,
                fftw_complex* psi_fft) {
    int nr = ws.nr;
    int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    auto& gv = *ws.gvec;
    for (int i = 0; i < nr; ++i)
        psi_fft[i][0] = psi_fft[i][1] = 0;
    for (int ig = 0; ig < ws.n_g; ++ig) {
        int ix = ((gv.h[ig] % nx) + nx) % nx;
        int iy = ((gv.k[ig] % ny) + ny) % ny;
        int iz = ((gv.l[ig] % nz) + nz) % nz;
        int idx = ix + nx * (iy + ny * iz);
        if (idx >= 0 && idx < nr) {
            psi_fft[idx][0] = static_cast<double>(psi_g[ig].real());
            psi_fft[idx][1] = static_cast<double>(psi_g[ig].imag());
        }
    }
}

// ── Extract from FFT grid back to G-sphere ──

void extract_from_fft(const PWWorkspace& ws, const fftw_complex* psi_fft,
                      std::complex<AccReal>* psi_g) {
    int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    auto& gv = *ws.gvec;
    for (int ig = 0; ig < ws.n_g; ++ig) {
        int ix = ((gv.h[ig] % nx) + nx) % nx;
        int iy = ((gv.k[ig] % ny) + ny) % ny;
        int iz = ((gv.l[ig] % nz) + nz) % nz;
        int idx = ix + nx * (iy + ny * iz);
        psi_g[ig] =
            std::complex<AccReal>(static_cast<AccReal>(psi_fft[idx][0]),
                                  static_cast<AccReal>(psi_fft[idx][1]));
    }
}

// ── Apply Hamiltonian: (Hψ)(G) = T(G)ψ(G) + FFT[V_eff(r)·IFFT[ψ(G)]] + V_NL·ψ ──

void apply_hamiltonian(const PWWorkspace& ws,
                       const std::complex<AccReal>* psi_g,
                       std::complex<AccReal>* Hpsi_g, fftw_complex* work_buf) {
    int ng = ws.n_g, nr = ws.nr;

    // Step 1: IFFT ψ(G) → ψ(r)  (c2c backward)
    pad_to_fft(ws, psi_g, work_buf);
    ws.fft->c2c_bwd(work_buf); // now work_buf = ψ(r)

    // Step 2: ψ(r) *= V_eff(r)
    for (int i = 0; i < nr; ++i) {
        AccReal V = ws.V_eff_r[i];
        work_buf[i][0] *= V;
        work_buf[i][1] *= V;
    }

    // Step 3: FFT V_eff(r)·ψ(r) → (Vψ)(G)  (c2c forward)
    ws.fft->c2c_fwd(work_buf); // now work_buf = FFT[V_eff·ψ]

    // QE: tgwave_r2g divides by N. FFTW c2c_fwd does not → divide by N.
    // This gives the correct off-diagonal V_eff for density development.
    extract_from_fft(ws, work_buf, Hpsi_g);
    AccReal inv_N = AccReal(1) / AccReal(nr);
    for (int ig = 0; ig < ng; ++ig)
        Hpsi_g[ig] = Hpsi_g[ig] * inv_N + ws.T_G[ig] * psi_g[ig];

    // Step 5: Non-local KB potential: V_NL = Σ_a Σ_lm D_l β_lm^a ⟨β_lm^a|ψ⟩
    for (const auto& atom_proj : ws.kb_proj) {
        for (const auto& projector : atom_proj) {
            const int nm = n_m_for_l(projector.l);
            if (projector.l < 0 || projector.l > 2
                || projector.beta_G.dimension() != 2
                || projector.beta_G.shape()[0] != static_cast<std::size_t>(nm)
                || projector.beta_G.shape()[1] != static_cast<std::size_t>(ng))
                throw std::invalid_argument(
                    "PW Hamiltonian: KB columns must be prepared for the "
                    "current k-sphere");
            const AccReal D = projector.d;
            if (std::abs(D) < AccReal(1e-16))
                continue;
            std::array<std::complex<AccReal>, 5> c_lm{};
            const auto* beta = projector.beta_G.data();
            // The m-major table is an ng-by-nm column-major B matrix:
            // c = B†ψ, then Hψ += D Bc, without packing or another table.
            using Index = xt::blas_index_t;
            const Index rows = static_cast<Index>(ng);
            const Index columns = static_cast<Index>(nm);
            const std::complex<AccReal> one(1, 0), zero(0, 0);
            cxxblas::gemv<Index>(cxxblas::ColMajor, cxxblas::ConjTrans, rows,
                                 columns, one, beta, rows, psi_g, Index(1),
                                 zero, c_lm.data(), Index(1));
            cxxblas::gemv<Index>(cxxblas::ColMajor, cxxblas::NoTrans, rows,
                                 columns, std::complex<AccReal>(D, 0), beta,
                                 rows, c_lm.data(), Index(1), one, Hpsi_g,
                                 Index(1));
        }
    }
}

} // namespace

void pw_apply_h(const PWWorkspace& ws, const std::complex<AccReal>* psi_g,
                std::complex<AccReal>* hpsi_g, fftw_complex* work_buf) {
    apply_hamiltonian(ws, psi_g, hpsi_g, work_buf);
}

void pw_apply_h(const PWWorkspace& ws, const std::complex<AccReal>* psi_g,
                std::complex<AccReal>* hpsi_g) {
    FFTWBuffer<fftw_complex> buf(static_cast<std::size_t>(ws.nr));
    apply_hamiltonian(ws, psi_g, hpsi_g, buf.data());
}

// ── Public interface ──

void DavidsonSolver::prepare(const PWWorkspace& ws) {
    if (ws.nr > 0)
        work_buf_.resize(static_cast<std::size_t>(ws.nr));
}

void DavidsonSolver::step(PWWorkspace& ws) {
    int ng = ws.n_g;
    int nb = n_bands_;
    int nr = ws.nr;

    ws.eigenvalues.resize({static_cast<std::size_t>(nb)});

    // FFT work buffer for H·ψ — pre-allocated in prepare(); guard against
    // direct low-level use that skipped prepare().
    if (work_buf_.size() < static_cast<std::size_t>(nr))
        work_buf_.resize(static_cast<std::size_t>(nr));
    auto* work_buf = work_buf_.data();

    // ── Initialize subspace (atomic wavefunctions or lowest-T(G) PWs) ──
    std::vector<std::complex<AccReal>> sub_vectors;
    int n_sub = internal::init_subspace(ws, nb, sub_vectors);
    ws.wfc.resize({static_cast<std::size_t>(nb), static_cast<std::size_t>(ng)});
    std::vector<std::complex<AccReal>> H_sub_vectors(n_sub * ng);
    std::vector<bool> H_computed(n_sub, false);

    // ── Davidson iteration ──
    const int max_david = max_iter_;
    const AccReal tolerance2 = tol_ * tol_;

    for (int idavid = 0; idavid < max_david; ++idavid) {
        // Compute H·v_j for new subspace vectors
        for (int j = 0; j < n_sub; ++j) {
            if (H_computed[j])
                continue;
            apply_hamiltonian(ws, &sub_vectors[j * ng], &H_sub_vectors[j * ng],
                              work_buf);
            H_computed[j] = true;
        }

        // Rows hold basis vectors. Hs = V^dagger H V in the physical
        // column-vector convention; only the first factor is conjugated.
        const std::array<std::size_t, 2> shape{static_cast<std::size_t>(n_sub),
                                               static_cast<std::size_t>(ng)};
        const auto V = xt::adapt(sub_vectors, shape);
        const auto HV = xt::adapt(H_sub_vectors, shape);
        linalg::CplxMatrix Hs = xt::linalg::dot(xt::conj(V), xt::transpose(HV));
        for (int mu = 0; mu < n_sub; ++mu) {
            Hs(mu, mu) = {Hs(mu, mu).real(), 0};
            for (int nu = mu + 1; nu < n_sub; ++nu)
                Hs(nu, mu) = std::conj(Hs(mu, nu));
        }
        std::vector<double> sub_eig;
        linalg::CplxMatrix sub_vecs;
        linalg::eigh_hermitian(Hs, sub_eig, sub_vecs);

        // H is fixed during the subspace solve: H(Vc)=(HV)c. Rotate the
        // retained eigenvectors by ordinary transpose, preserving Bloch phases.
        const linalg::CplxMatrix coefficients =
            xt::transpose(xt::view(sub_vecs, xt::all(), xt::range(0, nb)));
        const linalg::CplxMatrix ritz = xt::linalg::dot(coefficients, V);
        const linalg::CplxMatrix hritz = xt::linalg::dot(coefficients, HV);

        // Compute residuals and check convergence
        std::vector<std::complex<AccReal>> residuals(nb * ng);
        std::vector<AccReal> residual_norm2(nb, 0);
        for (int b = 0; b < nb; ++b) {
            // R = H·ψ - ε·ψ
            for (int ig = 0; ig < ng; ++ig) {
                auto& R = residuals[b * ng + ig];
                R = hritz(b, ig) - sub_eig[b] * ritz(b, ig);
                residual_norm2[b] += std::norm(R);
            }

            // Store eigenvalue
            ws.eigenvalues[b] = sub_eig[b];

            // Copy Ritz vector to wfc
            for (int ig = 0; ig < ng; ++ig)
                ws.wfc(b, ig) = ritz(b, ig);
        }
        if (*std::max_element(residual_norm2.begin(), residual_norm2.end())
            <= tolerance2)
            break;

        // Always add residuals to build crystal field (QE approach)

        // Add preconditioned residuals for unconverged bands
        if (idavid < max_david - 1 && n_sub < ng) {
            int n_added = 0;
            for (int b = 0; b < nb && n_sub < ng; ++b) {
                // Converged band: appending its residual would only
                // inject rounding noise into the subspace.
                if (residual_norm2[b] <= tolerance2)
                    continue;

                // Precondition: δ(G) = -R(G) / (T(G) - ε). The clamp keeps
                // the T(G) ≈ ε crossing from amplifying residual noise by
                // orders of magnitude (the preconditioner only needs to be
                // roughly right for Davidson to converge).
                std::vector<std::complex<AccReal>> delta(ng);
                AccReal eps = sub_eig[b];
                for (int ig = 0; ig < ng; ++ig) {
                    AccReal denom = ws.T_G[ig] - eps;
                    // Damp near-degenerate contributions
                    AccReal damp = std::abs(denom);
                    if (damp < AccReal(1e-2))
                        denom = (denom >= 0 ? AccReal(1e-2) : AccReal(-1e-2));
                    delta[ig] = -residuals[b * ng + ig] / denom;
                }

                // Normalize first so the drop threshold below is
                // scale-free, then orthogonalize with TWO Gram-Schmidt
                // passes ("twice is enough"): a single pass loses
                // orthogonality once the subspace is large, and the
                // standard Rayleigh-Ritz then drifts unboundedly below
                // the true eigenvalues.
                AccReal pre_norm = 0;
                for (int ig = 0; ig < ng; ++ig)
                    pre_norm += std::norm(delta[ig]);
                if (pre_norm < AccReal(1e-24))
                    continue;
                AccReal inv_pre = AccReal(1) / std::sqrt(pre_norm);
                for (int ig = 0; ig < ng; ++ig)
                    delta[ig] *= inv_pre;
                for (int pass = 0; pass < 2; ++pass) {
                    for (int mu = 0; mu < n_sub; ++mu) {
                        std::complex<AccReal> ovlp(0, 0);
                        for (int ig = 0; ig < ng; ++ig)
                            ovlp += std::conj(sub_vectors[mu * ng + ig])
                                    * delta[ig];
                        for (int ig = 0; ig < ng; ++ig)
                            delta[ig] -= ovlp * sub_vectors[mu * ng + ig];
                    }
                }

                // Drop only true rounding-level remainders (the 2-pass
                // CGS makes small-but-genuine new directions accurate).
                AccReal delta_norm = 0;
                for (int ig = 0; ig < ng; ++ig)
                    delta_norm += std::norm(delta[ig]);
                if (delta_norm < AccReal(1e-24))
                    continue;
                AccReal inv_norm = AccReal(1) / std::sqrt(delta_norm);
                for (int ig = 0; ig < ng; ++ig)
                    delta[ig] *= inv_norm;

                // Add to subspace — grow the storage BEFORE writing the
                // new column. (The historical PW_Davidson wrote slot
                // n_sub first and resized afterwards: a heap-buffer
                // overflow of one full column per expansion, latent
                // until the allocator layout exposed it.)
                int j = n_sub++;
                sub_vectors.resize(static_cast<std::size_t>(n_sub) * ng);
                H_sub_vectors.resize(static_cast<std::size_t>(n_sub) * ng);
                for (int ig = 0; ig < ng; ++ig)
                    sub_vectors[j * ng + ig] = delta[ig];
                H_computed.push_back(false);
                ++n_added;
            }
            if (n_added == 0)
                break; // all bands converged / spanned — done
        }
    }
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
