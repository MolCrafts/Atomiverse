#include "atomiverse/cpu/scf/internal/orbital_canonicalization.h"
#include <algorithm>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>
namespace ATV::cpu::scf::internal {
RealMatrix semicanonicalize(const RealMatrix& F, RealMatrix& C, RealVector& eig,
                            int n_occ) {
    const int n = static_cast<int>(C.shape()[0]);
    const int m = static_cast<int>(C.shape()[1]);
    if (F.shape()[0] != C.shape()[0] || F.shape()[1] != C.shape()[0]
        || n_occ < 0 || n_occ > m)
        throw std::invalid_argument(
            "semicanonicalize: incompatible Fock/coefficient shapes or "
            "occupation count");
    const int nv = m - n_occ;
    const RealMatrix Fmo =
        xt::linalg::dot(xt::transpose(C), RealMatrix(xt::linalg::dot(F, C)));

    eig = xt::zeros<double>({static_cast<std::size_t>(m)});
    RealMatrix C_new = C;

    // Occupied block: C_occ ← C_occ · U_o with F_oo U_o = U_o diag(e_o).
    RealMatrix Uo = xt::eye<double>(static_cast<std::size_t>(n_occ));
    if (n_occ > 0) {
        RealMatrix Foo = xt::zeros<double>({n_occ, n_occ});
        for (int i = 0; i < n_occ; ++i)
            for (int j = 0; j < n_occ; ++j)
                Foo(i, j) = Fmo(i, j);
        auto [eo, U] = xt::linalg::eigh(Foo);
        Uo = U;
        for (int i = 0; i < n_occ; ++i)
            eig(i) = eo(i);
        for (int mu = 0; mu < n; ++mu)
            for (int i = 0; i < n_occ; ++i) {
                double s = 0.0;
                for (int j = 0; j < n_occ; ++j)
                    s += C(mu, j) * Uo(j, i);
                C_new(mu, i) = s;
            }
    }

    // Virtual block: C_virt ← C_virt · U_v.
    RealMatrix Uv = xt::eye<double>(static_cast<std::size_t>(std::max(nv, 0)));
    if (nv > 0) {
        RealMatrix Fvv = xt::zeros<double>({nv, nv});
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b)
                Fvv(a, b) = Fmo(n_occ + a, n_occ + b);
        auto [ev, U] = xt::linalg::eigh(Fvv);
        Uv = U;
        for (int a = 0; a < nv; ++a)
            eig(n_occ + a) = ev(a);
        for (int mu = 0; mu < n; ++mu)
            for (int a = 0; a < nv; ++a) {
                double s = 0.0;
                for (int b = 0; b < nv; ++b)
                    s += C(mu, n_occ + b) * Uv(b, a);
                C_new(mu, n_occ + a) = s;
            }
    }
    C = C_new;

    // Gradient block in the semicanonical basis: Uvᵀ F_vo Uo.
    if (nv == 0 || n_occ == 0)
        return xt::zeros<double>({nv, n_occ});
    const RealMatrix Fvo =
        xt::view(Fmo, xt::range(n_occ, m), xt::range(0, n_occ));
    return xt::linalg::dot(xt::transpose(Uv),
                           RealMatrix(xt::linalg::dot(Fvo, Uo)));
}

} // namespace ATV::cpu::scf::internal
