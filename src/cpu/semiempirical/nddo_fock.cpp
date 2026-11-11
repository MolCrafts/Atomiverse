/// @file src/cpu/semiempirical/nddo_fock.cpp
/// @brief NDDO reference fock1/wstore/helect ports for NDDO methods.

#include "atomiverse/cpu/semiempirical/nddo_fock.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/semiempirical/nddo_two_electron.h"
#include "atomiverse/semiempirical/am1_parameters.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

std::size_t square_index(int row0, int col0, int dim) {
    return static_cast<std::size_t>(row0) * static_cast<std::size_t>(dim)
           + static_cast<std::size_t>(col0);
}

int local_pair_index0(int a1, int b1) {
    const int hi = std::max(a1, b1);
    const int lo = std::min(a1, b1);
    return (hi * (hi - 1)) / 2 + lo - 1;
}

void set_matrix(std::vector<double>& values, int dim, int row1, int col1,
                double value) {
    values[square_index(row1 - 1, col1 - 1, dim)] = value;
}

double matrix_value(const std::vector<double>& values, int dim, int row0,
                    int col0) {
    return values[square_index(row0, col0, dim)];
}

void validate_packed_size(const char* name, int n_orbitals,
                          const std::vector<double>& packed) {
    const std::size_t expected = static_cast<std::size_t>(n_orbitals)
                                 * static_cast<std::size_t>(n_orbitals + 1)
                                 / 2u;
    if (packed.size() != expected) {
        throw std::runtime_error(std::string(name)
                                 + " size does not match NDDO orbital count");
    }
}

/// Below this many atoms the pair loop is too short to amortize threads.
constexpr int kParallelPairAtoms = 16;

/// Largest two-center integral cache [bytes]: 5.8 MB holds a 205-atom and
/// 64 MB a 674-atom C/H/O/F/Br molecule; above the budget the Fock build
/// recomputes blocks per pair.
constexpr std::size_t kTwoCenterCacheBudget = std::size_t{2} << 30;

/// Coulomb and exchange contributions of one (i, j) atom pair's two-center
/// block `w` (row = atom-i orbital pair, n_pairs_j columns) to the Fock
/// matrix. The pair's density blocks are gathered once, the 10/10/16-entry
/// Fock blocks are formed locally and scattered once, so the inner loops do
/// no packed-index arithmetic. Same terms as NDDO reference `fock2`.
void accumulate_two_center_pair(const NDDOAtom& atom_i, const NDDOAtom& atom_j,
                                const double* w, int n_pairs_j,
                                const std::vector<double>& total_density,
                                const std::vector<double>& spin_density,
                                std::vector<double>& fock) {
    const int ni = atom_i.n_orbitals, nj = atom_j.n_orbitals;
    const int fi = atom_i.first_ao, fj = atom_j.first_ao;
    double p_ii[10], p_jj[10], s_ij[4][4];
    double f_ii[10] = {}, f_jj[10] = {}, f_ij[4][4] = {};
    double m_i[10], m_j[10];
    int a_of[10], b_of[10], c_of[10], d_of[10];
    int pi = 0;
    for (int a = 0; a < ni; ++a)
        for (int b = 0; b <= a; ++b, ++pi) {
            p_ii[pi] = total_density[packed_index(fi + a, fi + b)];
            m_i[pi] = a == b ? 1.0 : 2.0;
            a_of[pi] = a;
            b_of[pi] = b;
        }
    int pj = 0;
    for (int c = 0; c < nj; ++c)
        for (int d = 0; d <= c; ++d, ++pj) {
            p_jj[pj] = total_density[packed_index(fj + c, fj + d)];
            m_j[pj] = c == d ? 1.0 : 2.0;
            c_of[pj] = c;
            d_of[pj] = d;
        }
    for (int a = 0; a < ni; ++a)
        for (int c = 0; c < nj; ++c)
            s_ij[a][c] = spin_density[packed_index(fi + a, fj + c)];
    const int npi = pi, npj = pj;
    for (pi = 0; pi < npi; ++pi) {
        const double* row = w + static_cast<std::size_t>(pi) * n_pairs_j;
        const int a = a_of[pi], b = b_of[pi];
        double coulomb_i = 0.0;
        for (pj = 0; pj < npj; ++pj) {
            const double wij = row[pj];
            coulomb_i += m_j[pj] * p_jj[pj] * wij;
            f_jj[pj] += m_i[pi] * p_ii[pi] * wij;
            const int c = c_of[pj], d = d_of[pj];
            const double exchange = 0.25 * m_i[pi] * m_j[pj] * wij;
            f_ij[a][c] -= exchange * s_ij[b][d];
            f_ij[a][d] -= exchange * s_ij[b][c];
            f_ij[b][c] -= exchange * s_ij[a][d];
            f_ij[b][d] -= exchange * s_ij[a][c];
        }
        f_ii[pi] += coulomb_i;
    }
    for (pi = 0; pi < npi; ++pi)
        fock[packed_index(fi + a_of[pi], fi + b_of[pi])] += f_ii[pi];
    for (pj = 0; pj < npj; ++pj)
        fock[packed_index(fj + c_of[pj], fj + d_of[pj])] += f_jj[pj];
    for (int a = 0; a < ni; ++a) {
        double* fock_row = fock.data() + packed_index(fi + a, fj);
        for (int c = 0; c < nj; ++c)
            fock_row[c] += f_ij[a][c];
    }
}

int pair_thread_count(int n_atoms) {
#ifdef _OPENMP
    if (n_atoms >= kParallelPairAtoms)
        return std::max(1, omp_get_max_threads());
#endif
    (void)n_atoms;
    return 1;
}

/// Pair-block storage per atom pair (i > j): rows of atom-i orbital pairs.
std::size_t pair_block_size(const NDDOAtom& a, const NDDOAtom& b) {
    const auto pa =
        static_cast<std::size_t>(a.n_orbitals * (a.n_orbitals + 1) / 2);
    const auto pb =
        static_cast<std::size_t>(b.n_orbitals * (b.n_orbitals + 1) / 2);
    return pa * pb;
}

} // namespace

double NDDOOneCenterIntegrals::at(int row, int col) const {
    if (row < 0 || col < 0 || row >= dimension || col >= dimension) {
        throw std::runtime_error("NDDO one-center integral index out of range");
    }
    return values[square_index(row, col, dimension)];
}

NDDOOneCenterIntegrals
NDDOOneCenterIntegralBuilder::build(const NDDOAtom& atom) const {
    const auto* p = atom.params;
    if (p == nullptr) {
        throw std::runtime_error(
            "AM1 parameters unavailable for one-center integrals");
    }
    if (atom.n_orbitals > 4) {
        throw std::runtime_error(
            "AM1 d-orbital one-center integrals are not ported yet");
    }

    NDDOOneCenterIntegrals out;
    out.atomic_number = atom.atomic_number;
    out.dimension = atom.n_orbitals * (atom.n_orbitals + 1) / 2;
    out.values.assign(static_cast<std::size_t>(out.dimension)
                          * static_cast<std::size_t>(out.dimension),
                      0.0);

    set_matrix(out.values, out.dimension, 1, 1, p->gss);
    if (atom.n_orbitals > 2) {
        const int ip = 1;
        const int ipx = ip + 2;
        const int ipy = ip + 5;
        const int ipz = ip + 9;
        set_matrix(out.values, out.dimension, ipx, ip, p->gsp);
        set_matrix(out.values, out.dimension, ipy, ip, p->gsp);
        set_matrix(out.values, out.dimension, ipz, ip, p->gsp);
        set_matrix(out.values, out.dimension, ip, ipx, p->gsp);
        set_matrix(out.values, out.dimension, ip, ipy, p->gsp);
        set_matrix(out.values, out.dimension, ip, ipz, p->gsp);
        set_matrix(out.values, out.dimension, ipx, ipx, p->gpp);
        set_matrix(out.values, out.dimension, ipy, ipy, p->gpp);
        set_matrix(out.values, out.dimension, ipz, ipz, p->gpp);
        set_matrix(out.values, out.dimension, ipy, ipx, p->gp2);
        set_matrix(out.values, out.dimension, ipz, ipx, p->gp2);
        set_matrix(out.values, out.dimension, ipz, ipy, p->gp2);
        set_matrix(out.values, out.dimension, ipx, ipy, p->gp2);
        set_matrix(out.values, out.dimension, ipx, ipz, p->gp2);
        set_matrix(out.values, out.dimension, ipy, ipz, p->gp2);
        set_matrix(out.values, out.dimension, ip + 1, ip + 1, p->hsp);
        set_matrix(out.values, out.dimension, ip + 3, ip + 3, p->hsp);
        set_matrix(out.values, out.dimension, ip + 6, ip + 6, p->hsp);
        const double hpp = 0.5 * (p->gpp - p->gp2);
        set_matrix(out.values, out.dimension, ip + 4, ip + 4, hpp);
        set_matrix(out.values, out.dimension, ip + 7, ip + 7, hpp);
        set_matrix(out.values, out.dimension, ip + 8, ip + 8, hpp);
    }
    return out;
}

std::vector<double>
NDDOFockBuilder::add_one_center(const NDDOSystem& system,
                                const std::vector<double>& packed_hcore,
                                const std::vector<double>& total_density,
                                const std::vector<double>& spin_density) const {
    validate_packed_size("packed_hcore", system.n_orbitals, packed_hcore);
    validate_packed_size("total_density", system.n_orbitals, total_density);
    validate_packed_size("spin_density", system.n_orbitals, spin_density);

    std::vector<double> fock = packed_hcore;
    NDDOOneCenterIntegralBuilder integral_builder;

    for (const auto& atom : system.atoms) {
        const auto w = integral_builder.build(atom);
        const int ia = atom.first_ao;
        const int ib = atom.last_ao;
        for (int i = ia; i <= ib; ++i) {
            const int iw = i - ia + 1;
            for (int j = ia; j <= i; ++j) {
                const int jw = j - ia + 1;
                const std::size_t ij = packed_index(i, j);
                const int ijw = local_pair_index0(iw, jw);
                double sum = 0.0;

                for (int k = ia; k <= ib; ++k) {
                    const int kw = k - ia + 1;
                    for (int l = ia; l <= ib; ++l) {
                        const int lw = l - ia + 1;
                        const std::size_t ijp = packed_index(k, l);
                        const int klw = local_pair_index0(kw, lw);
                        const int ikw = local_pair_index0(kw, jw);
                        const int jlw = local_pair_index0(lw, iw);
                        sum +=
                            total_density[ijp]
                                * matrix_value(w.values, w.dimension, ijw, klw)
                            - spin_density[ijp]
                                  * matrix_value(w.values, w.dimension, ikw,
                                                 jlw);
                    }
                }
                fock[ij] += sum;
            }
        }
    }
    return fock;
}

std::vector<double>
NDDOFockBuilder::add_two_center(const NDDOSystem& system,
                                const cpu::Frame& frame,
                                const std::vector<double>& one_center_fock,
                                const std::vector<double>& total_density,
                                const std::vector<double>& spin_density) {
    return add_two_center(system, frame.span(kAtom_x), frame.span(kAtom_y),
                          frame.span(kAtom_z), one_center_fock, total_density,
                          spin_density);
}

std::vector<double> NDDOFockBuilder::add_two_center(
    const NDDOSystem& system, const StorageReal* sp_x, const StorageReal* sp_y,
    const StorageReal* sp_z, const std::vector<double>& one_center_fock,
    const std::vector<double>& total_density,
    const std::vector<double>& spin_density) {
    validate_packed_size("one_center_fock", system.n_orbitals, one_center_fock);
    validate_packed_size("total_density", system.n_orbitals, total_density);
    validate_packed_size("spin_density", system.n_orbitals, spin_density);
    if (sp_x == nullptr || sp_y == nullptr || sp_z == nullptr) {
        throw std::runtime_error(
            "NDDOFockBuilder::add_two_center requires non-null coordinates");
    }

    std::vector<double> fock = one_center_fock;
    // One bounded s/p block per thread is overwritten per pair.
    accumulate_pairs_(
        system, total_density, spin_density, fock,
        [&](int i, int j, NDDOTwoCenterIntegralBlock& w) {
            build_two_center_integral_block(system, sp_x, sp_y, sp_z, i, j, w);
            return std::pair<const double*, int>{w.values.data(), w.n_pairs_j};
        });
    return fock;
}

bool NDDOFockBuilder::cache_two_center(const NDDOSystem& system,
                                       const StorageReal* sp_x,
                                       const StorageReal* sp_y,
                                       const StorageReal* sp_z) {
    if (sp_x == nullptr || sp_y == nullptr || sp_z == nullptr)
        throw std::runtime_error(
            "NDDOFockBuilder::cache_two_center requires non-null coordinates");
    release_two_center();
    const int n_atoms = static_cast<int>(system.atoms.size());
    const std::size_t n_pairs =
        static_cast<std::size_t>(n_atoms) * (n_atoms > 0 ? n_atoms - 1 : 0) / 2;
    pair_offset_.resize(n_pairs + 1);
    std::size_t total = 0, p = 0;
    for (int i = 1; i < n_atoms; ++i)
        for (int j = 0; j < i; ++j, ++p) {
            pair_offset_[p] = total;
            total += pair_block_size(system.atoms[i], system.atoms[j]);
        }
    pair_offset_[p] = total;
    if (total > kTwoCenterCacheBudget / sizeof(double)) {
        pair_offset_.clear();
        return false;
    }
    pair_values_.resize(total);
    const int n_threads = pair_thread_count(n_atoms);
    std::exception_ptr failure;
#pragma omp parallel num_threads(n_threads) if (n_threads > 1)
    {
        NDDOTwoCenterIntegralBlock w;
#pragma omp for schedule(static, 1)
        for (int i = 1; i < n_atoms; ++i) {
            try {
                std::size_t q = static_cast<std::size_t>(i) * (i - 1) / 2;
                for (int j = 0; j < i; ++j, ++q) {
                    build_two_center_integral_block(system, sp_x, sp_y, sp_z, i,
                                                    j, w);
                    std::copy_n(w.values.data(),
                                pair_offset_[q + 1] - pair_offset_[q],
                                pair_values_.data() + pair_offset_[q]);
                }
            } catch (...) {
#pragma omp critical(nddo_cache_failure)
                if (!failure)
                    failure = std::current_exception();
            }
        }
    }
    if (failure) {
        release_two_center();
        std::rethrow_exception(failure);
    }
    cached_atoms_ = n_atoms;
    return true;
}

void NDDOFockBuilder::release_two_center() noexcept {
    pair_values_.clear();
    pair_offset_.clear();
    cached_atoms_ = -1;
}

std::vector<double>
NDDOFockBuilder::build_cached(const NDDOSystem& system,
                              const std::vector<double>& packed_hcore,
                              const std::vector<double>& total_density,
                              const std::vector<double>& spin_density) {
    if (cached_atoms_ != static_cast<int>(system.atoms.size()))
        throw std::logic_error(
            "NDDOFockBuilder::build_cached without a matching cache");
    std::vector<double> fock =
        add_one_center(system, packed_hcore, total_density, spin_density);
    validate_packed_size("total_density", system.n_orbitals, total_density);
    validate_packed_size("spin_density", system.n_orbitals, spin_density);
    accumulate_pairs_(system, total_density, spin_density, fock,
                      [&](int i, int j, NDDOTwoCenterIntegralBlock&) {
                          const std::size_t q =
                              static_cast<std::size_t>(i) * (i - 1) / 2 + j;
                          const int pj = system.atoms[j].n_orbitals
                                         * (system.atoms[j].n_orbitals + 1) / 2;
                          return std::pair<const double*, int>{
                              pair_values_.data() + pair_offset_[q], pj};
                      });
    return fock;
}

template <typename BlockFn>
void NDDOFockBuilder::accumulate_pairs_(
    const NDDOSystem& system, const std::vector<double>& total_density,
    const std::vector<double>& spin_density, std::vector<double>& fock,
    BlockFn&& block) {
    const int n_atoms = static_cast<int>(system.atoms.size());
    // Each pair writes its (ii), (jj) and (ij) Fock blocks, so threads
    // accumulate privately and are summed once; tiny systems stay serial.
    const int n_threads = pair_thread_count(n_atoms);
    const std::size_t packed_size = fock.size();
    thread_fock_.resize(static_cast<std::size_t>(n_threads));
    std::exception_ptr failure;
#pragma omp parallel num_threads(n_threads) if (n_threads > 1)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& local = thread_fock_[static_cast<std::size_t>(tid)];
        local.assign(packed_size, 0.0);
        NDDOTwoCenterIntegralBlock w;
        // Rows grow with i; round-robin rows balance the triangle and keep
        // the summation order fixed for a given thread count.
#pragma omp for schedule(static, 1)
        for (int i = 1; i < n_atoms; ++i) {
            try {
                for (int j = 0; j < i; ++j) {
                    const auto [values, n_pairs_j] = block(i, j, w);
                    accumulate_two_center_pair(system.atoms[i], system.atoms[j],
                                               values, n_pairs_j, total_density,
                                               spin_density, local);
                }
            } catch (...) {
#pragma omp critical(nddo_fock_failure)
                if (!failure)
                    failure = std::current_exception();
            }
        }
    }
    if (failure)
        std::rethrow_exception(failure);
    for (int t = 0; t < n_threads; ++t) {
        const auto& local = thread_fock_[static_cast<std::size_t>(t)];
        for (std::size_t k = 0; k < packed_size; ++k)
            fock[k] += local[k];
    }
}

std::vector<double>
NDDOFockBuilder::build(const NDDOSystem& system, const cpu::Frame& frame,
                       const std::vector<double>& packed_hcore,
                       const std::vector<double>& total_density,
                       const std::vector<double>& spin_density) {
    return build(system, frame.span(kAtom_x), frame.span(kAtom_y),
                 frame.span(kAtom_z), packed_hcore, total_density,
                 spin_density);
}

std::vector<double>
NDDOFockBuilder::build(const NDDOSystem& system, const StorageReal* sp_x,
                       const StorageReal* sp_y, const StorageReal* sp_z,
                       const std::vector<double>& packed_hcore,
                       const std::vector<double>& total_density,
                       const std::vector<double>& spin_density) {
    return add_two_center(
        system, sp_x, sp_y, sp_z,
        add_one_center(system, packed_hcore, total_density, spin_density),
        total_density, spin_density);
}

double electronic_energy_ev(int n_orbitals, const std::vector<double>& density,
                            const std::vector<double>& hcore,
                            const std::vector<double>& fock) {
    validate_packed_size("density", n_orbitals, density);
    validate_packed_size("hcore", n_orbitals, hcore);
    validate_packed_size("fock", n_orbitals, fock);

    // Compensated (Neumaier) sums: the terms cancel to a total ~1e-3 of
    // their absolute sum, so naive summation leaves ~1e-10 eV of round-off
    // in a 200-atom energy -- the size of the default energy_tol, which made
    // the SCF chase noise. Same terms, same helect formula.
    struct Compensated {
        double sum = 0.0, carry = 0.0;
        void add(double x) {
            const double t = sum + x;
            carry +=
                std::abs(sum) >= std::abs(x) ? (sum - t) + x : (x - t) + sum;
            sum = t;
        }
        double value() const { return sum + carry; }
    } diagonal, off_diagonal;
    std::size_t k = 0;
    for (int i = 0; i < n_orbitals; ++i) {
        for (int j = 0; j < i; ++j, ++k)
            off_diagonal.add(density[k] * (hcore[k] + fock[k]));
        diagonal.add(density[k] * (hcore[k] + fock[k]));
        ++k;
    }
    return off_diagonal.value() + 0.5 * diagonal.value();
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
