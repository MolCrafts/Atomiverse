/// @file cpu/scf/internal/cart2sph.cpp
/// @brief CartToSph tables — PySCF-convention constants + engine-convention
///        double-factorial rescale. @see cart2sph.h for the conventions.

#include "atomiverse/cpu/scf/internal/cart2sph.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/scf/cart.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

static_assert(kMaxL == 4, "resize CartToSph::matrix_/block_ arrays");

/// (2n-1)!! with (-1)!! = 1.
double dfact(int n) {
    double r = 1.0;
    for (int k = 2 * n - 1; k > 1; k -= 2)
        r *= k;
    return r;
}

} // namespace

CartToSph::CartToSph() {
    // l = 0, 1: identities (PySCF normalized='sp' convention).
    matrix_[0] = xt::xtensor<double, 2>({1, 1});
    matrix_[0](0, 0) = 1.0;
    matrix_[1] = xt::zeros<double>({std::size_t(3), std::size_t(3)});
    for (int i = 0; i < 3; ++i)
        matrix_[1](i, i) = 1.0;

    // l = 2 — rows xx,xy,xz,yy,yz,zz; cols m = -2..+2. Constants are the
    // real-Y_2m Cartesian coefficients (pinned vs PySCF 2.14.0 at 1e-12 in
    // test_cart2sph.cpp): a = (1/4)sqrt(5/pi) etc., written out digit-exact.
    static const double kD[6][5] = {
        {0.0, 0.0, -0.31539156525252005, 0.0, 0.5462742152960396},
        {1.0925484305920792, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0925484305920792, 0.0},
        {0.0, 0.0, -0.31539156525252005, 0.0, -0.5462742152960396},
        {0.0, 1.0925484305920792, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.6307831305050401, 0.0, 0.0},
    };
    matrix_[2] = xt::zeros<double>({std::size_t(6), std::size_t(5)});
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 5; ++j)
            matrix_[2](i, j) = kD[i][j];

    // l = 3 — rows xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz; cols m = -3..+3.
    static const double kF[10][7] = {
        {0.0, 0.0, 0.0, 0.0, -0.4570457994644657, 0.0, 0.5900435899266435},
        {1.7701307697799304, 0.0, -0.4570457994644657, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, -1.1195289977703462, 0.0, 1.4453057213202771, 0.0},
        {0.0, 0.0, 0.0, 0.0, -0.4570457994644657, 0.0, -1.7701307697799304},
        {0.0, 2.8906114426405543, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.8281831978578629, 0.0, 0.0},
        {-0.5900435899266435, 0.0, -0.4570457994644657, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, -1.1195289977703462, 0.0, -1.4453057213202771, 0.0},
        {0.0, 0.0, 1.8281831978578629, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.7463526651802308, 0.0, 0.0, 0.0},
    };
    matrix_[3] = xt::zeros<double>({std::size_t(10), std::size_t(7)});
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 7; ++j)
            matrix_[3](i, j) = kF[i][j];

    // l = 4 — rows xxxx,xxxy,xxxz,xxyy,xxyz,xxzz,xyyy,xyyz,xyzz,xzzz,
    //              yyyy,yyyz,yyzz,yzzz,zzzz; cols m = -4..+4. Digit-exact
    // copy of `pyscf.gto.mole.cart2sph(4, normalized='sp')` (PySCF
    // 2.14.0), exercised end-to-end against PySCF spherical 2-center
    // integrals by tests/cpu/scf/test_density_fitting.cpp. g shells
    // appear only in density-fitting auxiliary bases (def2 JKFIT).
    static const double kG[15][9] = {
        {0.0, 0.0, 0.0, 0.0, 0.31735664074561293, 0.0, -0.47308734787878, 0.0,
         0.6258357354491761},
        {2.5033429417967046, 0.0, -0.94617469575756, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, -2.0071396306718676, 0.0,
         1.7701307697799304, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.6347132814912259, 0.0, 0.0, 0.0,
         -3.755014412695057},
        {0.0, 5.310392309339791, 0.0, -2.0071396306718676, 0.0, 0.0, 0.0, 0.0,
         0.0},
        {0.0, 0.0, 0.0, 0.0, -2.5388531259649034, 0.0, 2.8385240872726802,
         0.0, 0.0},
        {-2.5033429417967046, 0.0, -0.94617469575756, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, -2.0071396306718676, 0.0,
         -5.310392309339791, 0.0},
        {0.0, 0.0, 5.6770481745453605, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 2.676186174229157, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.31735664074561293, 0.0, 0.47308734787878, 0.0,
         0.6258357354491761},
        {0.0, -1.7701307697799304, 0.0, -2.0071396306718676, 0.0, 0.0, 0.0,
         0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, -2.5388531259649034, 0.0, -2.8385240872726802,
         0.0, 0.0},
        {0.0, 0.0, 0.0, 2.676186174229157, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.8462843753216345, 0.0, 0.0, 0.0, 0.0},
    };
    matrix_[4] = xt::zeros<double>({std::size_t(15), std::size_t(9)});
    for (int i = 0; i < 15; ++i)
        for (int j = 0; j < 9; ++j)
            matrix_[4](i, j) = kG[i][j];

    // Engine convention: block(l) = D(l) · matrix(l), with
    // D_i = sqrt(4π (2lx−1)!!(2ly−1)!!(2lz−1)!! / (2l+1)!!) for l ≥ 2
    // (s/p are unit-normalized in both conventions ⇒ D = 1).
    const double kFourPi = 4.0 * M_PI;
    for (int l = 0; l <= kMaxL; ++l) {
        block_[l] = matrix_[l];
        if (l < 2)
            continue;
        int row = 0;
        // Cartesian component order matches cart.h / native integrals:
        // lx descending, then ly descending (xx,xy,xz,yy,yz,zz for d).
        for (int lx = l; lx >= 0; --lx) {
            for (int ly = l - lx; ly >= 0; --ly) {
                const int lz = l - lx - ly;
                const double denom = dfact(l + 1); // (2l+1)!!
                const double scale = std::sqrt(
                    kFourPi * dfact(lx) * dfact(ly) * dfact(lz) / denom);
                for (std::size_t j = 0; j < block_[l].shape(1); ++j)
                    block_[l](row, j) *= scale;
                ++row;
            }
        }
    }
}

const xt::xtensor<double, 2>& CartToSph::matrix(int l) const {
    if (l < 0 || l > kMaxL)
        throw std::out_of_range("CartToSph::matrix: l=" + std::to_string(l)
                                + " outside supported range 0.."
                                + std::to_string(kMaxL));
    return matrix_[l];
}

const xt::xtensor<double, 2>& CartToSph::block(int l) const {
    if (l < 0 || l > kMaxL)
        throw std::out_of_range("CartToSph::block: l=" + std::to_string(l)
                                + " outside supported range 0.."
                                + std::to_string(kMaxL));
    return block_[l];
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
