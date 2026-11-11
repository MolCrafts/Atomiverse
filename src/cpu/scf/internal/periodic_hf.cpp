/// @file cpu/scf/internal/periodic_hf.cpp
/// @brief Γ-point truncated lattice sums for periodic RHF.

#include "atomiverse/cpu/scf/internal/periodic_hf.h"

#include <cmath>
#include <vector>

#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/rys_eri.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {
namespace {

native::ShellBlock shift_block(const native::ShellBlock& b, const double T[3]) {
    native::ShellBlock s = b;
    s.center[0] += T[0];
    s.center[1] += T[1];
    s.center[2] += T[2];
    return s;
}

void foreach_image(const LatticeSpec& spec, bool include_zero, const auto& fn) {
    for (int tx = -spec.m[0]; tx <= spec.m[0]; ++tx) {
        for (int ty = -spec.m[1]; ty <= spec.m[1]; ++ty) {
            for (int tz = -spec.m[2]; tz <= spec.m[2]; ++tz) {
                if (!include_zero && tx == 0 && ty == 0 && tz == 0)
                    continue;
                const double T[3] = {tx * spec.L[0], ty * spec.L[1],
                                     tz * spec.L[2]};
                fn(T);
            }
        }
    }
}

} // namespace

LatticeSpec lattice_spec(const double L[3], double rcut) {
    LatticeSpec s;
    for (int d = 0; d < 3; ++d) {
        s.L[d] = L[d];
        // |t| L <= rcut ⇒ |t| <= floor(rcut/L). ceil would pull in a
        // nearest-neighbour shell even when L > rcut (the 20 Å H2 case).
        s.m[d] = (L[d] > 0.0 && rcut > 0.0)
                     ? static_cast<int>(std::floor(rcut / L[d] + 1e-12))
                     : 0;
    }
    return s;
}

double shell_pair_rcut(const std::vector<native::Shell>& shells,
                       double log_eps) {
    double a_min = 1e300;
    for (const auto& sh : shells) {
        for (int p = 0; p < sh.n_prim; ++p) {
            if (sh.alpha[p] > 0.0 && sh.alpha[p] < a_min)
                a_min = sh.alpha[p];
        }
    }
    if (!(a_min < 1e299))
        return 0.0;
    return 2.0 * std::sqrt(log_eps / a_min);
}

void exchange_lattice(const std::vector<native::ShellBlock>& blocks,
                      const RealMatrix& P, const LatticeSpec& spec,
                      RealMatrix& K, double omega) {
    const auto n = P.shape()[0];
    K = RealMatrix({n, n});
    K.fill(0.0);
    const int nb = static_cast<int>(blocks.size());

    foreach_image(spec, /*include_zero=*/true, [&](const double T[3]) {
        for (int ia = 0; ia < nb; ++ia) {
            const auto& A = blocks[static_cast<std::size_t>(ia)];
            for (int ib = 0; ib < nb; ++ib) {
                const auto& B = blocks[static_cast<std::size_t>(ib)];
                for (int ic = 0; ic < nb; ++ic) {
                    const auto C =
                        shift_block(blocks[static_cast<std::size_t>(ic)], T);
                    for (int id = 0; id < nb; ++id) {
                        const auto D = shift_block(
                            blocks[static_cast<std::size_t>(id)], T);
                        const int na = A.n_cart, nb_c = B.n_cart;
                        const int nc = C.n_cart, nd = D.n_cart;
                        std::vector<double> buf(
                            static_cast<std::size_t>(na * nb_c * nc * nd), 0.0);
                        native::rys_eri_block(A, B, C, D, buf.data());
                        if (omega > 0.0) {
                            std::vector<double> erf(buf.size(), 0.0);
                            native::rys_eri_erf_block(A, B, C, D, omega,
                                                      erf.data());
                            for (std::size_t i = 0; i < buf.size(); ++i)
                                buf[i] -= erf[i];
                        }
                        for (int a = 0; a < na; ++a) {
                            const int mu = A.ao_start + a;
                            for (int b = 0; b < nb_c; ++b) {
                                const int lam = B.ao_start + b;
                                for (int c = 0; c < nc; ++c) {
                                    const int nu = C.ao_start + c;
                                    for (int d = 0; d < nd; ++d) {
                                        const int sig = D.ao_start + d;
                                        const double val =
                                            buf[((a * nb_c + b) * nc + c) * nd
                                                + d];
                                        K(mu, nu) += val * P(lam, sig);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });
}

void apply_gamma_images(IntegralWorkspace& ws, const LatticeSpec& spec) {
    if (!spec.has_extra())
        return;

    const auto blocks = blocks_from_shells(ws.shells);

    foreach_image(spec, /*include_zero=*/false, [&](const double T[3]) {
        for (const auto& A : blocks) {
            for (const auto& B0 : blocks) {
                const auto B = shift_block(B0, T);
                const int na = A.n_cart, nb = B.n_cart;
                std::vector<double> sbuf(static_cast<std::size_t>(na * nb),
                                         0.0);
                std::vector<double> tbuf(sbuf.size(), 0.0);
                std::vector<double> vbuf(sbuf.size(), 0.0);
                native::overlap_block(A, B, sbuf.data());
                native::kinetic_block(A, B, tbuf.data());
                native::nuclear_block(A, B, ws.atoms.data(),
                                      static_cast<int>(ws.atoms.size()),
                                      vbuf.data());
                for (int ia = 0; ia < na; ++ia) {
                    const int i = A.ao_start + ia;
                    for (int ib = 0; ib < nb; ++ib) {
                        const int j = B.ao_start + ib;
                        const int idx = ia * nb + ib;
                        ws.S(i, j) += sbuf[idx];
                        ws.T(i, j) += tbuf[idx];
                        ws.V(i, j) += vbuf[idx];
                    }
                }
            }
        }

        const int natm = static_cast<int>(ws.atoms.size());
        for (int a = 0; a < natm; ++a) {
            for (int b = 0; b < natm; ++b) {
                const double dx =
                    ws.atoms[a].pos[0] - ws.atoms[b].pos[0] - T[0];
                const double dy =
                    ws.atoms[a].pos[1] - ws.atoms[b].pos[1] - T[1];
                const double dz =
                    ws.atoms[a].pos[2] - ws.atoms[b].pos[2] - T[2];
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-15)
                    continue;
                ws.V_nuc += 0.5
                            * static_cast<double>(ws.atoms[a].Z * ws.atoms[b].Z)
                            / r;
            }
        }
    });

    ws.Hcore = ws.T + ws.V;
}

void fill_eri_lattice(const std::vector<native::ShellBlock>& blocks,
                      const LatticeSpec& spec, xt::xtensor<double, 4>& ERI) {
    foreach_image(spec, /*include_zero=*/true, [&](const double T[3]) {
        for (const auto& A : blocks) {
            for (const auto& B : blocks) {
                for (const auto& C0 : blocks) {
                    const auto C = shift_block(C0, T);
                    for (const auto& D0 : blocks) {
                        const auto D = shift_block(D0, T);
                        const int na = A.n_cart, nb = B.n_cart;
                        const int nc = C.n_cart, nd = D.n_cart;
                        std::vector<double> buf(
                            static_cast<std::size_t>(na * nb * nc * nd), 0.0);
                        native::rys_eri_block(A, B, C, D, buf.data());
                        for (int ia = 0; ia < na; ++ia) {
                            const int i = A.ao_start + ia;
                            for (int ib = 0; ib < nb; ++ib) {
                                const int j = B.ao_start + ib;
                                for (int ic = 0; ic < nc; ++ic) {
                                    const int k = C.ao_start + ic;
                                    for (int id = 0; id < nd; ++id) {
                                        const int l = D.ao_start + id;
                                        ERI(i, j, k, l) +=
                                            buf[((ia * nb + ib) * nc + ic) * nd
                                                + id];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
