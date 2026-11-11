/// @file cpu/scf/internal/ccsd_spatial.h
/// @brief Closed-shell spatial-orbital CCSD, one amplitude iteration.
///
/// The product loops are `cpu::scf::CCSD` and `cpu::scf::CC2`
/// (`Driver<cc::Phase>`). This type is the engine those loops call:
/// `update` is PySCF 2.14.0 `pyscf/cc/rccsd.py` `update_amps` (Hirata et
/// al., J. Chem. Phys. 120, 2581 (2004)). `cc2` selects that function's
/// `if cc.cc2` doubles residual (Christiansen, Koch & Jørgensen, Chem.
/// Phys. Lett. 243, 409 (1995)); the singles residual stays full CCSD.
/// `mix` is the accept step of `pyscf/cc/ccsd.py` `kernel` /
/// `CCSD.run_diis`, and `energy` is `rccsd.energy`. `RCCSD.kernel` calls
/// `CCSDBase.ccsd`, which runs this order. `pyscf/cc/ccsd.py`
/// `update_amps` raises on `cc2` and is not the residual. Nothing here
/// sets a converged flag.
///
/// Reference: Scuseria, Janssen, Schaefer, J. Chem. Phys. 89, 7382 (1988);
/// Hirata et al., J. Chem. Phys. 120, 2581 (2004).
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/internal/cc_diis.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief One closed-shell spatial CCSD amplitude state.
///
/// Construction builds denominators, the MP2 T₂ guess, and the DIIS
/// buffers. Each later call is one amplitude stage. This type does not
/// decide convergence.
struct CcsdSpatialCycle {
    /// @brief Denominators, MP2 T₂ guess, and DIIS buffers.
    /// @param ints Integral pack with the dense ERI tensor materialized.
    /// @param det Converged closed-shell RHF determinant.
    /// @param e_scf_ha SCF total energy [Hartree].
    /// @param diis_on When true, `mix` extrapolates; when false it copies
    ///        the Jacobi amplitudes (same fixed point).
    /// @param cc2_on When true, the doubles residual is `rccsd.update_amps`
    ///        with `cc.cc2`. When false, the residual is full CCSD.
    /// @throws std::invalid_argument on open-shell / missing ERI / non-aufbau
    ///         denominators (no silent fallback). No virtuals is a trivial
    ///         zero-correlation state.
    CcsdSpatialCycle(const internal::IntegralView& ints,
                     const SlaterDeterminant& det, double e_scf_ha,
                     bool diis_on, bool cc2_on = false);

    /// @brief One `rccsd.update_amps` step. Stores the Jacobi increment.
    ///        Doubles follow `if cc.cc2` when `cc2` is set.
    void update();

    /// @brief DIIS extrapolant, or the raw Jacobi amplitudes when DIIS is off.
    void mix();

    /// @brief `rccsd.energy` of the current amplitudes [Hartree]. Caches
    ///        `e_corr`.
    /// @return Correlation energy [Hartree].
    double energy();

    /// @brief max|Δt₁|, max|Δt₂| from the last `update` [dimensionless].
    /// @return Amplitude residual. Zero before the first update.
    double residual() const { return std::max(dt1, dt2); }

    xt::xtensor<double, 4> eri;
    xt::xtensor<double, 2> t1;
    xt::xtensor<double, 4> t2;
    xt::xtensor<double, 2> t1new;
    xt::xtensor<double, 4> t2new;
    xt::xtensor<double, 2> eia;
    xt::xtensor<double, 4> eijab;
    internal::CcAmplitudeDiis diis{8};
    std::vector<double> amp;
    std::vector<double> err;
    int no = 0;
    int nv = 0;
    std::size_t No = 0;
    std::size_t Nv = 0;
    /// @brief SCF total stored at construction [Hartree].
    double e_scf = 0.0;
    /// @brief Correlation energy from the last `energy` [Hartree].
    double e_corr = 0.0;
    double dt1 = 0.0;
    double dt2 = 0.0;
    bool use_diis = true;
    /// @brief Doubles residual is the CC2 truncation of `update_amps`.
    bool cc2 = false;
    /// @brief True when the reference has no virtual orbitals.
    bool trivial = false;

  private:
    double OO(int i, int j, int k, int l) const { return eri(i, j, k, l); }
    double OVOV(int i, int a, int j, int b) const {
        return eri(i, no + a, j, no + b);
    }
    double OOVV(int i, int j, int a, int b) const {
        return eri(i, j, no + a, no + b);
    }
    double OVVO(int i, int a, int b, int j) const {
        return eri(i, no + a, no + b, j);
    }
    double OVVV(int i, int a, int b, int c) const {
        return eri(i, no + a, no + b, no + c);
    }
    double OVOO(int i, int a, int j, int k) const {
        return eri(i, no + a, j, k);
    }
    double VVVV(int a, int b, int c, int d) const {
        return eri(no + a, no + b, no + c, no + d);
    }
};

} // namespace scf
} // namespace cpu
} // namespace ATV
