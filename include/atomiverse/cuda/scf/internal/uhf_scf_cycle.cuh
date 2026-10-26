/// @file cuda/scf/internal/uhf_scf_cycle.cuh
/// @brief internal::UhfScfCycle — private UHF workspace assembly.
///
/// Owned only by `cuda::scf::UHF`. Not a Driver and not a loop: the SCF
/// iteration is `cuda::Driver<scf::Phase>`. One stage per call.
///
/// Transplanted from PySCF (read, not re-derived):
///   - `pyscf.scf.hf.kernel` — stage order get_fock / eig / make_rdm1 /
///     energy_tot / convergence
///   - `pyscf.scf.uhf.make_rdm1` — P^σ = C_occ^σ (C_occ^σ)^T (occupation 1)
///   - `pyscf.scf.uhf.get_veff` — V^σ = J[Pα+Pβ] − K[P^σ]
///   - `pyscf.scf.uhf.energy_elec` — e1 = Tr(h Pα)+Tr(h Pβ),
///     e2 = ½ Tr(Vα Pα)+½ Tr(Vβ Pβ)
///   - `pyscf.scf.uhf.get_occ` — lowest n_α / n_β eigenvectors
///   - `pyscf.scf.uhf.get_fock` — per-spin level shift after DIIS
///   - `pyscf.scf.uhf.UHF.init_guess_by_minao` — dma = dmb = ½ P_MINAO
///     (open shell: `_break_dm_spin_symm` is a no-op). Hcore is the fallback
///     when an element has no MINAO table.
///   - `pyscf.scf.diis.CDIIS.update` — device CDIIS, one Pulay system per spin
///   - `pyscf.grad.uhf.grad_elec` / `get_veff` / `make_rdm1e` — forces
///
/// Not this type: `pyscf.scf.dhf` (Dirac), `pyscf.pbc.scf.khf` (k-point),
/// `pyscf.x2c.sfx2c1e_grad` (x2c-1e forces), `pyscf.df.df_jk.density_fit`
/// (RI JK selector), `pyscf.hessian.rhf.Hessian` (CPHF; `Hessian<UHF>`).
#pragma once

#include <memory>

#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

class EigenSolver;
class DIISMixer;
class UHFFockDevice;
class UHFDensity;
class ConvergenceCheck;
class UhfMinaoGuess;

/// @brief Beta-spin matrices. Alpha lives on `SCFWorkspace` between stages.
struct UhfBeta {
    prec_t* H = nullptr;
    prec_t* P = nullptr;
    prec_t* P_old = nullptr;
    prec_t* C = nullptr;
    prec_t* C_prime = nullptr;
    prec_t* eig = nullptr;
    prec_t* work = nullptr;
    prec_t* F_prime = nullptr;
    int n_occ = 0;
};

/// @brief Private CUDA UHF workspace (device integrals only).
class UhfScfCycle {
  public:
    /// @brief STO-3G s/p shell list already packed by the driver.
    UhfScfCycle(int n_shells, int n_ao, int n_prim, const double* alpha_mol,
                const double* coeff_mol, const int* L_mol,
                const int* atom_of_mol, int n_alpha, int n_beta, double tol,
                int max_iter, double level_shift, const int* atomic_numbers,
                int natm);
    ~UhfScfCycle();

    void build(void* stream, cuda::Frame& frame);
    void begin(void* stream);
    void step_fock(void* stream, cuda::Frame& frame, int step);
    void step_mix(void* stream, cuda::Frame& frame, int step);
    void step_diagonalize(void* stream, cuda::Frame& frame, int step);
    void step_density(void* stream, cuda::Frame& frame, int step);
    void step_check(void* stream, cuda::Frame& frame, int step);
    bool poll_converged(void* stream, int step);
    void finish(void* stream);
    void destroy(void* stream, cuda::Frame& frame);

    const SCFWorkspace& workspace() const { return workspace_; }
    const UhfBeta& beta() const { return beta_; }
    /// @brief True — this cycle has no host-ERI path.
    bool device_integrals() const { return true; }

  private:
    double tol_;
    double level_shift_ = 0.0;
    bool built_ = false;
    bool polishing_ = false;
    bool canonicalized_ = false;

    SCFWorkspace workspace_;
    UhfBeta beta_;

    std::unique_ptr<EigenSolver> eigen_;
    std::unique_ptr<DIISMixer> diis_alpha_;
    std::unique_ptr<DIISMixer> diis_beta_;
    std::unique_ptr<UHFFockDevice> fock_;
    std::unique_ptr<UHFDensity> density_;
    std::unique_ptr<ConvergenceCheck> convergence_;
    std::unique_ptr<UhfMinaoGuess> guess_;
};

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
