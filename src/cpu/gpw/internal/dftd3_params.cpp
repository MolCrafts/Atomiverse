/// @file cpu/gpw/internal/dftd3_params.cpp
/// @brief D3Params constructor over the in-tree static tables (see header).
///
/// The covalent-radius (rcov) and r2r4 tables are the compiled-in D3 reference
/// values (basis_data/dftd3_tables.h), processed exactly as CP2K does:
///   rcov  -> (4/3) * rcov_Pyykko[Ang] * (1/0.52917721) [Bohr]   (k2 folded in)
///   r2r4  -> sqrt(0.5 * r2r4_raw * sqrt(Z))                     (C8 geom. avg.)
///   r0ab  -> r0ab_tri[Ang] * (1/0.52917721) [Bohr]  (packed lower triangle)

#include "atomiverse/cpu/gpw/internal/dftd3_params.h"

#include <cmath>

#include "atomiverse/cpu/gpw/basis_data/dftd3_tables.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

/// CP2K physcon: bohr = 1/angstrom = Angstrom -> Bohr conversion factor.
constexpr double kAngToBohr = 1.0 / 0.52917720859;
/// D3 coordination-number prefactor k2 (CP2K qs_dispersion_pairpot.F).
constexpr double kK2 = 4.0 / 3.0;

/// @brief Decode a CP2K-encoded element index (Z + 1000*(ref-1)) -> (Z, ref).
/// Mirrors qs_dispersion_d3::limit with ref_stride = 1000.
void decode_index(int encoded, int& z, int& ref_index) {
    z = encoded;
    ref_index = 1;
    while (z > 1000) {
        z -= 1000;
        ref_index += 1;
    }
}

/// Fill the compiled-in element tables (rcov, r2r4, r0ab).
void fill_element_tables(D3Params& p) {
    for (int z = 1; z <= 103 && z <= kMaxElem; ++z) {
        p.rcov[z] = kK2 * basis_data::kRcovPyykko[z - 1] * kAngToBohr;
        const double dum = 0.5 * basis_data::kR2r4Raw[z - 1]
                           * std::sqrt(static_cast<double>(z));
        p.r2r4[z] = std::sqrt(dum);
    }
    // Unpack the lower-triangular r0ab table: k runs i=1..N, j=1..i.
    int k = 0;
    for (int i = 1; i <= 103; ++i) {
        for (int j = 1; j <= i; ++j) {
            const double v = basis_data::kR0abTri[k] * kAngToBohr;
            ++k;
            if (i <= kMaxElem && j <= kMaxElem) {
                p.r0ab[i][j] = v;
                p.r0ab[j][i] = v;
            }
        }
    }
}

/// Insert one CN-resolved C6 reference record, symmetrized over (a,b).
void add_c6_record(D3Params& p, double c6, double fiat, double fjat, double cna,
                   double cnb) {
    int za = 0, ra = 0, zb = 0, rb = 0;
    decode_index(static_cast<int>(std::lround(fiat)), za, ra);
    decode_index(static_cast<int>(std::lround(fjat)), zb, rb);
    if (za < 1 || zb < 1 || za > kMaxElem || zb > kMaxElem)
        return;
    p.c6ab[za][zb].push_back({c6, cna, cnb});
    if (za != zb || cna != cnb)
        p.c6ab[zb][za].push_back({c6, cnb, cna});
}

} // namespace

D3Params::D3Params() {
    fill_element_tables(*this);
    for (int i = 0; i < basis_data::kNC6RefSupported; ++i) {
        const double* r = basis_data::kC6RefSupported[i];
        add_c6_record(*this, r[0], r[1], r[2], r[3], r[4]);
    }
}

bool D3Params::supports_element(int z) {
    for (int s : basis_data::kD3SupportedZ)
        if (s == z)
            return true;
    return false;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
