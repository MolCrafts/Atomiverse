/// @file src/cpu/semiempirical/dftb0.cpp
/// Non-SCC DFTB0 (H/C/N/O): transcribed mio-1-1 SK tables for every pair.
#include "atomiverse/cpu/semiempirical/dftb0.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h"
#include "atomiverse/cpu/semiempirical/internal/mio_sk_block.h"
#include "atomiverse/cpu/semiempirical/internal/mio_sk_table.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace {

using basis_data::DftbElementRow;
using basis_data::DftbPairSkRow;
using basis_data::kDftbElements;
using basis_data::kDftbNElements;
using basis_data::kDftbNPairSk;
using basis_data::kDftbPairSk;

const DftbElementRow& element_row(int z) {
    for (std::size_t i = 0; i < kDftbNElements; ++i)
        if (kDftbElements[i].z == z)
            return kDftbElements[i];
    throw std::runtime_error("DFTB0: unsupported element Z=" + std::to_string(z)
                             + " (static mio-1-1 tables cover H/C/N/O only)");
}

const DftbPairSkRow& pair_sk(int za, int zb) {
    for (std::size_t i = 0; i < kDftbNPairSk; ++i)
        if (kDftbPairSk[i].z_a == za && kDftbPairSk[i].z_b == zb)
            return kDftbPairSk[i];
    throw std::runtime_error("DFTB0: no SK table for pair Z="
                             + std::to_string(za) + "-" + std::to_string(zb));
}

/// Non-SCC total energy at the current Frame geometry [eV].
double energy_non_scc(const Frame& f, const StorageReal* x, const StorageReal* y,
                      const StorageReal* z, const int* type, int n_at,
                      const std::vector<const DftbElementRow*>& el,
                      const std::vector<int>& ao0, int n_ao, int n_occ) {
    linalg::RealMatrix H0 = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix S = xt::zeros<double>({n_ao, n_ao});

    auto dist_vec = [&](int i, int j, double& dx, double& dy, double& dz) {
        dx = static_cast<double>(x[j] - x[i]);
        dy = static_cast<double>(y[j] - y[i]);
        dz = static_cast<double>(z[j] - z[i]);
        f.box.min_image(dx, dy, dz);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        H0(ia, ia) = ea.e_s;
        S(ia, ia) = 1.0;
        if (ea.n_ao == 4) {
            for (int p = 1; p < 4; ++p) {
                H0(ia + p, ia + p) = ea.e_p;
                S(ia + p, ia + p) = 1.0;
            }
        }
        for (int b = a + 1; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            double dx, dy, dz;
            const double r = dist_vec(a, b, dx, dy, dz);
            const double rcut = internal::MioSkTable{ea.z, eb.z}.cutoff();
            if (r > rcut)
                continue;
            const int ib = ao0[static_cast<std::size_t>(b)];
            const auto& sk_ab = pair_sk(type[a], type[b]);
            const auto& sk_ba = pair_sk(type[b], type[a]);
            internal::MioSketchPair{ea, eb, sk_ab}.add_block(dx, dy, dz, r, H0,
                                                             S, ia, ib);
            internal::MioSketchPair{eb, ea, sk_ba}.add_block(
                -dx, -dy, -dz, r, H0, S, ib, ia);
        }
    }

    auto sol = linalg::diagonalize_fock_generalized(H0, S);
    const auto& eps = sol.first;

    double e_band = 0;
    for (int o = 0; o < n_occ; ++o)
        e_band += 2.0 * eps(o);

    double e_repulsive = 0;
    for (int a = 0; a < n_at; ++a)
        for (int b = a + 1; b < n_at; ++b) {
            double dx, dy, dz;
            const double r = dist_vec(a, b, dx, dy, dz);
            e_repulsive +=
                internal::MioSkTable{type[a], type[b]}.repulsive(r);
        }

    return e_band + e_repulsive;
}

} // namespace

void DFTB0::build(const BuildContext& /*ctx*/) {}

void DFTB0::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n_at = f.nlocal();
    if (n_at < 1)
        throw std::invalid_argument("DFTB0: empty frame");

    const auto type = f.span(kAtom_type);
    StorageReal* x = f.span(kAtom_x);
    StorageReal* y = f.span(kAtom_y);
    StorageReal* z = f.span(kAtom_z);
    AccReal* fx = f.span(kAtom_fx);
    AccReal* fy = f.span(kAtom_fy);
    AccReal* fz = f.span(kAtom_fz);
    double& energy_meta = f.meta.get(kEnergy);

    std::vector<const DftbElementRow*> el(static_cast<std::size_t>(n_at));
    std::vector<int> ao0(static_cast<std::size_t>(n_at) + 1, 0);
    double n_elec = 0;
    for (int a = 0; a < n_at; ++a) {
        el[static_cast<std::size_t>(a)] = &element_row(type[a]);
        n_elec += el[static_cast<std::size_t>(a)]->n_val;
        ao0[static_cast<std::size_t>(a) + 1] =
            ao0[static_cast<std::size_t>(a)]
            + el[static_cast<std::size_t>(a)]->n_ao;
    }
    const int n_ao = ao0[static_cast<std::size_t>(n_at)];
    if (n_ao < 1)
        throw std::runtime_error("DFTB0: no basis functions");
    if (std::fabs(n_elec - std::round(n_elec)) > 1e-9
        || static_cast<int>(std::lround(n_elec)) % 2 != 0)
        throw std::runtime_error(
            "DFTB0: odd/non-integer valence electron count not supported");
    const int n_occ = static_cast<int>(std::lround(n_elec)) / 2;

    energy_ = energy_non_scc(f, x, y, z, type, n_at, el, ao0, n_ao, n_occ);
    energy_meta += energy_;

    if (with_forces) {
        constexpr double h_fd = 1.0e-4; // Å
        for (int a = 0; a < n_at; ++a) {
            for (int k = 0; k < 3; ++k) {
                StorageReal* col = (k == 0) ? x : (k == 1) ? y : z;
                const StorageReal saved = col[a];
                col[a] = saved + static_cast<StorageReal>(h_fd);
                const double ep =
                    energy_non_scc(f, x, y, z, type, n_at, el, ao0, n_ao,
                                   n_occ);
                col[a] = saved - static_cast<StorageReal>(h_fd);
                const double em =
                    energy_non_scc(f, x, y, z, type, n_at, el, ao0, n_ao,
                                   n_occ);
                col[a] = saved;
                const double fi = -(ep - em) / (2.0 * h_fd);
                if (k == 0)
                    fx[a] += static_cast<AccReal>(fi);
                else if (k == 1)
                    fy[a] += static_cast<AccReal>(fi);
                else
                    fz[a] += static_cast<AccReal>(fi);
            }
        }
    }
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
