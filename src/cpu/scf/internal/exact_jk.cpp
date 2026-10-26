/// @file cpu/scf/internal/exact_jk.cpp
#include "atomiverse/cpu/scf/internal/exact_jk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/scf/cart.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace ATV::cpu::scf::internal {
namespace {
double resident_mb() {
#if defined(__linux__)
    std::ifstream status("/proc/self/statm");
    std::size_t total = 0, resident = 0;
    if (status >> total >> resident)
        return static_cast<double>(resident) * sysconf(_SC_PAGESIZE) / 1e6;
#elif defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count)
        == KERN_SUCCESS)
        return static_cast<double>(info.resident_size) / 1e6;
#endif
    return 0;
}
std::size_t pair_index(std::size_t a, std::size_t b) {
    if (a < b)
        std::swap(a, b);
    return a * (a + 1) / 2 + b;
}

// Visit each symmetry-unique AO quartet once. Block-pair order is not AO-pair
// order, so only a diagonal block quartet needs the ab >= cd filter.
template <typename Consume>
void visit_quartets(const ShellBlockPairList& pairs, int left,
                    Consume&& consume, double density_bound = 1.0,
                    const std::vector<RealMatrix>* transforms = nullptr,
                    const std::vector<int>* starts = nullptr) {
    const auto& bra = pairs.pairs[left];
    const auto& A = pairs.blocks[bra.si];
    const auto& B = pairs.blocks[bra.sj];
    double block[kMaxNCartOrbital * kMaxNCartOrbital * kMaxNCartOrbital
                 * kMaxNCartOrbital];
    for (int right = 0; right <= left; ++right) {
        const auto& ket = pairs.pairs[right];
        if (bra.schwarz * ket.schwarz * density_bound < pairs.threshold)
            continue;
        const auto& C = pairs.blocks[ket.si];
        const auto& D = pairs.blocks[ket.sj];
        const int size = A.n_cart * B.n_cart * C.n_cart * D.n_cart;
        std::fill_n(block, size, 0.0);
        native::eri_block(A, B, C, D, block);
        std::array<int, 4> dims{A.n_cart, B.n_cart, C.n_cart, D.n_cart};
        std::array<int, 4> offsets{A.ao_start, B.ao_start, C.ao_start,
                                   D.ao_start};
        if (transforms) {
            // Transform one shell quartet, never a dense n^4 intermediate.
            // Each axis is a small contraction with the existing basis map.
            double scratch[kMaxNCartOrbital * kMaxNCartOrbital
                           * kMaxNCartOrbital * kMaxNCartOrbital];
            const int shells[] = {bra.si, bra.sj, ket.si, ket.sj};
            for (int axis = 0; axis < 4; ++axis) {
                const auto& T = (*transforms)[shells[axis]];
                const int old_dim = dims[axis], new_dim = T.shape()[1];
                int before = 1, after = 1;
                for (int i = 0; i < axis; ++i)
                    before *= dims[i];
                for (int i = axis + 1; i < 4; ++i)
                    after *= dims[i];
                for (int i = 0; i < before; ++i)
                    for (int s = 0; s < new_dim; ++s)
                        for (int j = 0; j < after; ++j) {
                            double v = 0;
                            for (int c = 0; c < old_dim; ++c)
                                v += T(c, s)
                                     * block[(i * old_dim + c) * after + j];
                            scratch[(i * new_dim + s) * after + j] = v;
                        }
                std::copy_n(scratch, before * new_dim * after, block);
                dims[axis] = new_dim;
                offsets[axis] = (*starts)[shells[axis]];
            }
        }
        for (int ia = 0; ia < dims[0]; ++ia) {
            const int a = offsets[0] + ia;
            for (int ib = 0; ib < dims[1]; ++ib) {
                const int b = offsets[1] + ib;
                if (a < b)
                    continue;
                const auto ab = pair_index(a, b);
                for (int ic = 0; ic < dims[2]; ++ic) {
                    const int c = offsets[2] + ic;
                    for (int id = 0; id < dims[3]; ++id) {
                        const int d = offsets[3] + id;
                        if (c < d || (left == right && ab < pair_index(c, d)))
                            continue;
                        const double v =
                            block[((ia * dims[1] + ib) * dims[2] + ic) * dims[3]
                                  + id];
                        consume(a, b, c, d, v);
                    }
                }
            }
        }
    }
}

// Distinct permutations only; no density symmetry assumption. The same
// contraction handles RHF, open-shell and all four real GHF spin blocks.
inline void contract(int n, const double* p, double* j, double* k, int a, int b,
                     int c, int d, double v) {
    // J is symmetric even for nonsymmetric densities. Combine the two
    // density orientations before scattering; K still uses all permutations.
    const double jab = v * (p[c + d * n] + (c != d ? p[d + c * n] : 0));
    j[a + b * n] += jab;
    if (a != b)
        j[b + a * n] += jab;
    if (a != c || b != d) {
        const double jcd = v * (p[a + b * n] + (a != b ? p[b + a * n] : 0));
        j[c + d * n] += jcd;
        if (c != d)
            j[d + c * n] += jcd;
    }
    const auto add = [&](int i, int q, int r, int s) {
        k[i + r * n] += v * p[q + s * n];
    };
    add(a, b, c, d);
    if (a != b)
        add(b, a, c, d);
    if (c != d)
        add(a, b, d, c);
    if (a != b && c != d)
        add(b, a, d, c);
    if (a != c || b != d) {
        add(c, d, a, b);
        if (c != d)
            add(d, c, a, b);
        if (a != b)
            add(c, d, b, a);
        if (a != b && c != d)
            add(d, c, b, a);
    }
}
} // namespace

void ExactJK::prepare(const IntegralView& ints, double memory_mb, bool mpi) {
    if (!std::isfinite(memory_mb) || memory_mb < 0)
        throw std::invalid_argument(
            "max_memory_mb must be finite and nonnegative");
    const auto start = LogClock::now();
    has_previous_.fill(false);
    previous_p_ = {};
    previous_j_ = {};
    previous_k_ = {};
    rank_ = 0;
    ranks_ = 1;
    if (mpi) {
#ifdef ATV_HAS_MPI
        int initialized = 0, finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized)
            MPI_Finalized(&finalized);
        if (!initialized || finalized)
            throw std::runtime_error(
                "HF mpi=true requires an active MPI runtime on every rank");
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks_);
#else
        throw std::runtime_error("HF mpi=true requires ATV_USE_MPI=ON");
#endif
    }
    // Release the previous geometry before deciding/allocating the next cache.
    std::vector<double>().swap(eri_);
    pairs_ = {};
    c2s_ = {};
    // Explicit dense inputs are used by periodic/post-HF callers and synthetic
    // unit fixtures. Compress them; never select the old dense J/K algorithm.
    const bool dense = ints.ERI.size() != 0;
    n_ = ints.n_basis;
    if (n_ <= 0)
        throw std::invalid_argument("ExactJK requires a nonempty AO basis");
    const auto n = static_cast<std::size_t>(n_);
    const auto np = n * (n + 1) / 2;
    if (np > std::numeric_limits<std::size_t>::max() / (np + 1))
        throw std::overflow_error("ExactJK s8 size overflow");
    const auto count = np * (np + 1) / 2;
    if (!dense) {
        if (ints.shells.empty())
            throw std::invalid_argument("ExactJK requires shells or ERIs");
        // Match PySCF's default direct-SCF integral threshold.
        pairs_ = ShellBlockPairList(ints, 1e-13);
    }
    // PySCF RHF._is_mem_enough: nbf**4 / 1e6 + current_memory < .95 * max_memory.
    // Use actual s8 bytes as well, since n**4 underestimates tiny s8 arrays.
    const double estimate =
        std::max(std::pow(static_cast<double>(n), 4),
                 static_cast<double>(count) * sizeof(double));
    int cache = dense || estimate / 1e6 + resident_mb() < .95 * memory_mb;
#ifdef ATV_HAS_MPI
    if (ranks_ > 1)
        MPI_Allreduce(MPI_IN_PLACE, &cache, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
#endif
    if (cache) {
        eri_.assign(count, 0.0);
        if (dense) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int a = 0; a < n_; ++a)
                for (int b = 0; b <= a; ++b) {
                    const auto ab = pair_index(a, b);
                    for (int c = 0; c <= a; ++c)
                        for (int d = 0; d <= c; ++d) {
                            const auto cd = pair_index(c, d);
                            if (cd <= ab)
                                eri_[pair_index(ab, cd)] = ints.ERI(a, b, c, d);
                        }
                }
        } else {
            std::vector<RealMatrix> transforms;
            std::vector<int> starts;
            if (ints.spherical()) {
                int offset = 0;
                for (const auto& block : pairs_.blocks) {
                    const int ns = 2 * block.l + 1;
                    RealMatrix T = xt::zeros<double>({block.n_cart, ns});
                    for (int i = 0; i < block.n_cart; ++i)
                        for (int j = 0; j < ns; ++j)
                            T(i, j) = ints.c2s(block.ao_start + i, offset + j);
                    transforms.push_back(std::move(T));
                    starts.push_back(offset);
                    offset += ns;
                }
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int left = 0; left < static_cast<int>(pairs_.pairs.size());
                 ++left)
                visit_quartets(
                    pairs_, left,
                    [&](int a, int b, int c, int d, double v) {
                        eri_[pair_index(pair_index(a, b), pair_index(c, d))] =
                            v;
                    },
                    1.0, transforms.empty() ? nullptr : &transforms,
                    transforms.empty() ? nullptr : &starts);
        }
    } else {
        n_ = ints.n_ao_cart();
        if (ints.spherical())
            c2s_ = ints.c2s;
    }
    ATV_INFO("[exact_jk] cache_bytes={} build_ms={:.3f} mpi_ranks={}",
             cache_bytes(), elapsed_ms(start), ranks_);
}

void ExactJK::compute(const RealMatrix& P, RealMatrix& J, RealMatrix& K) const {
    const auto start = LogClock::now();
    const std::size_t nao = c2s_.size() ? c2s_.shape()[1] : n_;
    if (P.shape()[0] != nao || P.shape()[1] != nao)
        throw std::invalid_argument("ExactJK density dimension mismatch");
    const RealMatrix cart =
        c2s_.size()
            ? RealMatrix(xt::linalg::dot(
                  c2s_, RealMatrix(xt::linalg::dot(P, xt::transpose(c2s_)))))
            : P;
    double density_bound = 0;
    for (double value : cart)
        density_bound = std::max(density_bound, std::abs(value));
    // Conservative global density bound for all eight J/K permutations.
    density_bound *= 8;
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    const std::size_t nn = static_cast<std::size_t>(n_) * n_;
    std::vector<std::vector<double>> js(nt, std::vector<double>(nn));
    std::vector<std::vector<double>> ks(nt, std::vector<double>(nn));
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const auto consume = [&](int a, int b, int c, int d, double v) {
            if (v != 0.0)
                contract(n_, cart.data(), js[tid].data(), ks[tid].data(), a, b,
                         c, d, v);
        };
        if (cached()) {
            // As in PySCF nr_incore, schedule canonical AO-pair rows in
            // small dynamic chunks, keeping heavy rows balanced across threads.
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
            for (std::int64_t ab = rank_;
                 ab < static_cast<std::int64_t>(n_) * (n_ + 1) / 2;
                 ab += ranks_) {
                const int a =
                    static_cast<int>((std::sqrt(8.0 * ab + 1) - 1) / 2);
                const int b = ab - static_cast<std::int64_t>(a) * (a + 1) / 2;
                const double* row = eri_.data() + ab * (ab + 1) / 2;
                std::size_t cd = 0;
                for (int c = 0; c <= a; ++c)
                    for (int d = 0; d <= (c == a ? b : c); ++d, ++cd)
                        consume(a, b, c, d, row[cd]);
            }
        } else {
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
            for (int left = rank_; left < static_cast<int>(pairs_.pairs.size());
                 left += ranks_)
                visit_quartets(pairs_, left, consume, density_bound);
        }
    }
    J = xt::zeros<double>({n_, n_});
    K = xt::zeros<double>({n_, n_});
    for (int t = 0; t < nt; ++t)
        for (std::size_t i = 0; i < nn; ++i) {
            J.data()[i] += js[t][i];
            K.data()[i] += ks[t][i];
        }
#ifdef ATV_HAS_MPI
    if (ranks_ > 1) {
        // Only the calling thread enters MPI; OpenMP regions have completed.
        // Chunk counts to stay within the portable MPI int-count interface.
        for (std::size_t offset = 0; offset < nn;) {
            const int count = static_cast<int>(std::min(
                nn - offset,
                static_cast<std::size_t>(std::numeric_limits<int>::max())));
            MPI_Allreduce(MPI_IN_PLACE, J.data() + offset, count, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, K.data() + offset, count, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);
            offset += count;
        }
    }
#endif
    if (c2s_.size()) {
        J = xt::linalg::dot(xt::transpose(c2s_),
                            RealMatrix(xt::linalg::dot(J, c2s_)));
        K = xt::linalg::dot(xt::transpose(c2s_),
                            RealMatrix(xt::linalg::dot(K, c2s_)));
    }
    ATV_DEBUG("[exact_jk] jk_ms={:.3f}", elapsed_ms(start));
}

void ExactJK::step(const RealMatrix& P, RealMatrix& J, RealMatrix& K,
                   int channel) const {
    if (channel < 0 || channel >= 4)
        throw std::invalid_argument(
            "ExactJK channel must be between zero and three");
    if (cached()) {
        compute(P, J, K);
        return;
    }
    const RealMatrix delta =
        has_previous_[channel] ? RealMatrix(P - previous_p_[channel]) : P;
    compute(delta, J, K);
    if (has_previous_[channel]) {
        J += previous_j_[channel];
        K += previous_k_[channel];
    }
    previous_p_[channel] = P;
    previous_j_[channel] = J;
    previous_k_[channel] = K;
    has_previous_[channel] = true;
}
} // namespace ATV::cpu::scf::internal
