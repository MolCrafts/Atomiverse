/// @file cpu/planewave/internal/g_vector.cpp
/// @brief G-vector enumeration implementation.

#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/core/units.h"
#include <algorithm>
#include <cmath>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

GVectorList enumerate_g_vectors(const cpu::Box& cell_bohr, AccReal ecut,
                                const Vec3& k) {
    auto recip = cell_bohr.recip_cell();
    Vec3 b0{recip(0, 0), recip(0, 1), recip(0, 2)};
    Vec3 b1{recip(1, 0), recip(1, 1), recip(1, 2)};
    Vec3 b2{recip(2, 0), recip(2, 1), recip(2, 2)};

    AccReal b0n = b0.norm(), b1n = b1.norm(), b2n = b2.norm();
    AccReal gmax = std::sqrt(AccReal(2) * ecut);
    int hmax = static_cast<int>(gmax / b0n) + 1;
    int kmax = static_cast<int>(gmax / b1n) + 1;
    int lmax = static_cast<int>(gmax / b2n) + 1;

    // Index window per axis. The integer index of G along axis i is
    // exactly h_i = (a_i . G) / 2pi, so over the cutoff ball
    // |k + G| <= gmax the index runs in -f_i +/- gmax |a_i| / 2pi, where
    // f_i = (a_i . k) / 2pi is the fractional coordinate of k. The
    // historical bound +/-(gmax/|b_i| + 1) ignored both the k offset and
    // the obliquity of the reciprocal basis (gmax/|b_i| underestimates
    // the projection extent for non-orthogonal cells), silently clipping
    // sphere-edge G-vectors — harmless-looking for small |k| but fatal
    // for Berry-phase strings sampled at fractional labels near 1.
    const Mat3& A = cell_bohr.cell(); // rows a_i [Bohr]
    int lo[3], hi[3];
    for (int i = 0; i < 3; ++i) {
        const AccReal an = std::sqrt(A(i, 0) * A(i, 0) + A(i, 1) * A(i, 1)
                                     + A(i, 2) * A(i, 2));
        const AccReal two_pi = AccReal(2) * units::pi;
        const AccReal fi =
            (A(i, 0) * k[0] + A(i, 1) * k[1] + A(i, 2) * k[2]) / two_pi;
        const AccReal hw = gmax * an / two_pi;
        lo[i] = static_cast<int>(std::floor(-fi - hw)) - 1;
        hi[i] = static_cast<int>(std::ceil(-fi + hw)) + 1;
    }

    struct GV {
        Vec3 v;
        AccReal e;
        int hi, ki, li;
    };
    std::vector<GV> tmp;

    for (int h = lo[0]; h <= hi[0]; ++h) {
        for (int kk = lo[1]; kk <= hi[1]; ++kk) {
            for (int l = lo[2]; l <= hi[2]; ++l) {
                Vec3 g = b0 * AccReal(h) + b1 * AccReal(kk) + b2 * AccReal(l);
                Vec3 kpg = {g[0] + k[0], g[1] + k[1], g[2] + k[2]};
                AccReal e = AccReal(0.5) * kpg.norm_squared();
                if (e <= ecut + AccReal(1e-12)) {
                    tmp.push_back({g, e, h, kk, l});
                }
            }
        }
    }

    std::sort(tmp.begin(), tmp.end(), [](const GV& a, const GV& b) {
        return a.v.norm_squared() < b.v.norm_squared();
    });

    GVectorList result;
    result.kvec = k;
    result.g.reserve(tmp.size());
    result.ekin.reserve(tmp.size());
    result.h.reserve(tmp.size());
    result.k.reserve(tmp.size());
    result.l.reserve(tmp.size());
    for (auto& t : tmp) {
        result.g.push_back(t.v);
        result.ekin.push_back(t.e);
        result.h.push_back(t.hi);
        result.k.push_back(t.ki);
        result.l.push_back(t.li);
    }

    // FFT grid dimensions: must be even (FFTW requirement)
    int nx = 2 * (hmax + 1);
    int ny = 2 * (kmax + 1);
    int nz = 2 * (lmax + 1);
    if (nx % 2)
        ++nx;
    if (ny % 2)
        ++ny;
    if (nz % 2)
        ++nz;
    result.nx = nx;
    result.ny = ny;
    result.nz = nz;

    return result;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
