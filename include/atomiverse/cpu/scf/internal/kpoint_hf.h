/// @file cpu/scf/internal/kpoint_hf.h
/// @brief Periodic k-point restricted Hartree–Fock (FFT density fitting).
///
/// Transplanted from PySCF 2.14.0 (read, not re-derived):
///   - pyscf/pbc/scf/khf.py: `get_occ`, `make_rdm1`, `energy_elec`,
///     `KSCF.get_veff` (vhf = J − ½K), `KRHF`
///   - pyscf/pbc/scf/kuhf.py: `get_occ`, `make_rdm1`, `energy_elec`,
///     `KUHF.get_veff` (J = Ja+Jb, vhf = J−K per spin), `KUHF.nelec`
///   - pyscf/pbc/scf/hf.py: `energy_nuc` (= `cell.ewald`)
///   - pyscf/pbc/df/fft.py: `get_nuc`
///   - pyscf/pbc/df/fft_jk.py: `get_j_kpts`, `get_k_kpts`
///   - pyscf/pbc/tools/pbc.py: `get_coulG` (exxdiv='ewald'), `madelung`,
///     `fft` / `ifft` (NumPy normalization)
///   - pyscf/pbc/gto/cell.py: `estimate_ke_cutoff`, `cutoff_to_mesh`,
///     `ewald`, `make_kpts` (with_gamma, wrap_around off)
///
/// v1 is all-electron s functions on an orthorhombic cell. l > 0 throws
/// naming the missing collocation kernel — no host molecular-ERI fill.
/// KUHF reuses the same FFT J/K; only the spin assembly differs.
#pragma once

#include <array>
#include <complex>
#include <memory>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief One primitive as PySCF stores it after `cell.build`:
///        libcint contraction coefficient (radial norm included) and l.
struct PgtoSpec {
    int l = 0;
    double alpha = 0.0;
    double c_libcint = 0.0;
};

/// @brief FFT mesh for an orthorhombic cell. PySCF `estimate_ke_cutoff` +
///        `cutoff_to_mesh` at `precision` (default cell.precision = 1e-8).
/// @param L_bohr     Orthorhombic edges [Bohr].
/// @param prims      Primitives (steepest sets the cutoff).
/// @param nprim      Count.
/// @param precision  Integral target (dimensionless threshold).
/// @return Odd mesh {nx,ny,nz}.
std::array<int, 3> fft_mesh_orthorhombic(const double L_bohr[3],
                                         const PgtoSpec* prims, int nprim,
                                         double precision);

/// @brief Real-space lattice cutoff [Bohr] from PySCF `cell._estimate_rcut`
///        on the most diffuse primitive (overlap + kinetic penalty).
double lattice_rcut(const PgtoSpec* prims, int nprim, double precision);

/// @brief Ewald nuclear repulsion [Hartree], PySCF `pbc.gto.cell.ewald`
///        for a 3D orthorhombic cell (Martin App. F2, as corrected there).
/// @param Z       Nuclear charges [e], length `natm`.
/// @param R       Positions [Bohr], `R[a][3]`.
/// @param natm    Atom count.
/// @param L_bohr  Orthorhombic edges [Bohr].
/// @param precision Relative cutoff (PySCF `cell.precision`).
double ewald_nuclear(const double* Z, const double (*R)[3], int natm,
                     const double L_bohr[3], double precision);

/// @brief Γ-centered Monkhorst–Pack points [1/Bohr], PySCF `make_kpts`
///        with `with_gamma_point` and `wrap_around=False` (the library
///        default). Gamma is index 0. `k = s · b`, `b = 2π inv(aᵀ)`.
void monkhorst_kpts(const int nmesh[3], const double L_bohr[3],
                    std::vector<std::array<double, 3>>& k_out);

/// @brief In-place 3D FFT, NumPy `fftn` / `ifftn` normalization and
///        C-order index `iz + nz*(iy + ny*ix)`.
void fft3(std::vector<std::complex<double>>& data, int nx, int ny, int nz,
          bool inverse);

/// @brief Converged KRHF total energy [Hartree] for an orthorhombic Frame.
///
/// Writes nothing. Throws if the cell is not orthorhombic, the basis has
/// l > 0, the electron count across the mesh is odd, or the SCF does not
/// reach `tol` in `max_iter`.
/// @brief Density-independent KRHF quantities (geometry, basis, Hcore, AO).
///        The SCF step consumes these; it does not rebuild them.
struct KrhfStatic {
    int mesh[3]{};
    int nk = 0;
    int nao = 0;
    int ng = 0;
    int nocc = 0;
    double vol = 0.0;
    double enuc = 0.0;
    std::vector<std::array<double, 3>> kpts;
    std::vector<double> coords;
    std::vector<double> coul_j;
    std::vector<double> coul_k;
    /// AO values, index ((k * ng) + g) * nao + mu.
    std::vector<std::complex<double>> ao;
    /// Core Hamiltonian, index ((k * nao) + mu) * nao + nu.
    std::vector<std::complex<double>> H;
    /// Canonical X, index ((k * nao) + i) * nao + j, columns [0, nmo).
    std::vector<std::complex<double>> X;
    std::vector<int> nmo;
};

/// @brief Build the density-independent KRHF workspace (no SCF).
KrhfStatic krhf_assemble(const Frame& frame, BasisSet basis,
                         const int k_mesh[3], int charge);

/// @brief One restricted k-point SCF state. The driver walks it; this type
///        does not decide convergence.
///
/// Steps follow PySCF 2.14.0 `pyscf/pbc/scf/khf.py` (`KRHF`,
/// `KSCF.get_fock`, `energy_elec`, `eig`, `make_rdm1`) and
/// `pyscf/scf/hf.py` `kernel`, with DIIS left off (`get_fock` `diis=None`).
struct KrhfCycle;

/// @brief Destroy a cycle allocated by `krhf_cycle_open`.
void krhf_cycle_delete(KrhfCycle* cycle);

/// @brief Owning pointer. The deleter is a function so callers need not
///        see the cycle body.
using KrhfCycleOwner = std::unique_ptr<KrhfCycle, void (*)(KrhfCycle*)>;

/// @brief Assemble the cell and the core-Hamiltonian density guess.
/// @param frame Orthorhombic geometry [Å].
/// @param basis Gaussian basis (s functions).
/// @param k_mesh Monkhorst–Pack divisions.
/// @param charge Cell charge [e].
KrhfCycleOwner krhf_cycle_open(const Frame& frame, BasisSet basis,
                               const int k_mesh[3], int charge);

/// @brief `KSCF.get_fock` + `energy_elec` for the current density.
///        vhf = J − ½K. Stores the total energy [Hartree] including E_nn.
void krhf_cycle_fock(KrhfCycle& cycle);

/// @brief `KRHF.eig`: canonical orthogonalization then Hermitian diagonalize.
void krhf_cycle_diagonalize(KrhfCycle& cycle);

/// @brief `get_occ` + `make_rdm1` from the orbitals just diagonalized.
void krhf_cycle_density(KrhfCycle& cycle);

/// @brief Last total energy [Hartree], nuclear repulsion included.
double krhf_cycle_energy(const KrhfCycle& cycle);

/// @brief |ΔE| [Hartree] against the previous fock energy, or +∞ on the
///        first call. Records the current energy as the next baseline.
///        Does not decide convergence.
double krhf_cycle_delta(KrhfCycle& cycle);

struct KrhfEnergy {
    double hartree = 0.0;
    int iterations = 0;
    bool converged = false;
};

/// @brief Converged KUHF total energy [Hartree].
///
/// `n_alpha` and `n_beta` are mesh totals (`KUHF.nelec` in
/// `pyscf/pbc/scf/kuhf.py`): `cell.spin = n_alpha - n_beta` is not
/// multiplied by the number of k-points. `n_alpha + n_beta` must equal
/// `(ΣZ)*nk − charge`. Same s-function orthorhombic envelope as KRHF.
KrhfEnergy kuhf_energy(const Frame& frame, BasisSet basis, const int k_mesh[3],
                       int n_alpha, int n_beta, int charge, double tol,
                       int max_iter);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
