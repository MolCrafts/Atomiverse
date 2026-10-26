/// @file cpu/scf/internal/exact_jk.h
/// @brief Memory-budgeted exact Coulomb/exchange with eightfold ERI storage.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "atomiverse/cpu/scf/internal/shell_pair.h"

namespace ATV::cpu::scf::internal {

/// One exact J/K engine: cache s8 integrals when they fit, otherwise evaluate
/// screened shell quartets during contraction. Both routes use the same
/// permutation contraction, including nonsymmetric GHF spin densities.
class ExactJK {
  public:
    /// Rebuild for a geometry. Budget is process memory in MB (95% admission);
    /// zero selects the automatic engine's bounded-memory execution.
    void prepare(const IntegralView& ints, double memory_mb = 4000.0,
                 bool mpi = false);

    /// J_ij = sum_kl (ij|kl) P_lk; K_ik = sum_jl (ij|kl) P_jl.
    /// Input/output are in the working AO basis, including spherical AOs.
    void compute(const RealMatrix& P, RealMatrix& J, RealMatrix& K) const;

    /// SCF update: cached ERIs contract full densities; the bounded-memory
    /// route contracts delta densities and accumulates J/K, as in PySCF.
    /// Channels keep separate histories for the four real GHF spin blocks.
    void step(const RealMatrix& P, RealMatrix& J, RealMatrix& K,
              int channel = 0) const;

    std::size_t cache_bytes() const { return eri_.size() * sizeof(double); }
    bool cached() const { return !eri_.empty(); }

  private:
    int n_ = 0;
    ShellBlockPairList pairs_;
    RealMatrix c2s_;
    std::vector<double> eri_;
    int rank_ = 0, ranks_ = 1;
    mutable std::array<RealMatrix, 4> previous_p_, previous_j_, previous_k_;
    mutable std::array<bool, 4> has_previous_{};
};

} // namespace ATV::cpu::scf::internal
