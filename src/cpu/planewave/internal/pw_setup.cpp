/// @file cpu/planewave/internal/pw_setup.cpp
/// @brief Host-side plane-wave SCF problem assembly (moved verbatim from
///        the pre-loop section of the solver) shared by both backends.

#include "atomiverse/cpu/planewave/internal/pw_setup.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/gth_upf.h"
#include "atomiverse/cpu/planewave/internal/upf_reader.h"
#include "atomiverse/cpu/planewave/qe_units.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {

/// True when this process should emit console logs — MPI rank 0, or any
/// run where MPI is not built/initialized. Mirrors the explicit rank
/// gate `PW::run()` applies around its ATV_INFO/ATV_DEBUG calls.
bool log_rank0() {
#ifdef ATV_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank == 0;
    }
#endif
    return true;
}

void setup_upf_vloc(PWWorkspace& ws, const internal::GVectorList& gv,
                    const internal::UPFData& upf, AccReal omega) {
    auto vg = upf_vloc_to_gspace(upf, gv, omega);
    // Atom form factor × structure factor Σ_a e^{-iG·R_a}.
    // vg[0] already holds only the finite part (QE "αZ" term); the
    // divergent −4πZ/ΩG² jellium piece is cancelled against the
    // Ewald/Hartree G=0 conventions.
    const auto& pos = ws.atom_pos_bohr;
    for (int ig = 0; ig < ws.n_g; ++ig) {
        AccReal s_re = 0, s_im = 0;
        for (size_t ia = 0; ia < pos.size(); ++ia) {
            AccReal phase = gv.g[ig][0] * pos[ia][0] + gv.g[ig][1] * pos[ia][1]
                            + gv.g[ig][2] * pos[ia][2];
            s_re += std::cos(phase);
            s_im -= std::sin(phase);
        }
        ws.V_loc_G[ig] = vg[ig] * std::complex<AccReal>(s_re, s_im);
    }
}

std::vector<Vec3> pos_to_bohr(const cpu::Frame& frame) {
    AccReal a2b = AccReal(1.8897259886);
    std::vector<Vec3> pos;
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);
    for (size_t i = 0; i < frame.nlocal(); ++i)
        pos.push_back({static_cast<AccReal>(sp_x[i]) * a2b,
                       static_cast<AccReal>(sp_y[i]) * a2b,
                       static_cast<AccReal>(sp_z[i]) * a2b});
    return pos;
}

// Spherical Bessel functions j_l(x)
AccReal sph_bessel_j0(AccReal x) {
    if (std::abs(x) < AccReal(1e-8))
        return AccReal(1) - x * x / AccReal(6);
    return std::sin(x) / x;
}
AccReal sph_bessel_j1(AccReal x) {
    if (std::abs(x) < AccReal(1e-8))
        return x / AccReal(3);
    return (std::sin(x) / (x * x)) - (std::cos(x) / x);
}
AccReal sph_bessel_j2(AccReal x) {
    if (std::abs(x) < AccReal(1e-8))
        return x * x / AccReal(15);
    AccReal x2 = x * x, x3 = x * x2;
    return (AccReal(3) / x3 - AccReal(1) / x) * std::sin(x)
           - (AccReal(3) / x2) * std::cos(x);
}

using SphBesselFn = AccReal (*)(AccReal);
SphBesselFn sph_bessel_for_l(int l) {
    switch (l) {
    case 0:
        return sph_bessel_j0;
    case 1:
        return sph_bessel_j1;
    case 2:
        return sph_bessel_j2;
    default:
        // No-fallback rule: silently substituting j0 for l > 2 would
        // produce a wrong (not failing) nonlocal potential.
        throw std::invalid_argument("sph_bessel_for_l: angular momentum l="
                                    + std::to_string(l)
                                    + " not supported (l <= 2)");
    }
}

} // namespace

/// Build Kleinman-Bylander non-local projectors in G-space for one
/// k-sphere. The radial table is evaluated at q = |k+G| — the projector
/// of a Bloch state lives on k+G, so the table is k-dependent and must
/// be rebuilt per k-point (size gv.count(), which also differs per k).
void setup_kb_projectors(std::vector<std::vector<PWWorkspace::KBProjG>>& out,
                         const internal::GVectorList& gv,
                         const internal::UPFData& upf,
                         const std::vector<Vec3>& pos_bohr) {
    int n_atoms = static_cast<int>(pos_bohr.size());
    int ng = gv.count();
    out.assign(n_atoms, {});

    // Precompute f_l(q) for each projector (radial integral, same for all atoms)
    int n_proj = upf.n_proj();
    upf.validate_radial_mesh();
    std::size_t radial_points = 0;
    for (const auto& projector : upf.kb) {
        if (projector.radial_points < 3
            || projector.radial_points > projector.beta.size()
            || projector.beta.size() > upf.r.size())
            throw std::invalid_argument("KB projectors: declare a valid radial "
                                        "integration domain within PP_R");
        radial_points = std::max(radial_points, projector.radial_points);
    }
    std::vector<std::vector<AccReal>> fl_all(n_proj);
    std::vector<int> proj_l(n_proj);
    for (size_t ip = 0; ip < upf.kb.size(); ++ip) {
        auto& kb = upf.kb[ip];
        proj_l[ip] = kb.l;
        auto j_l = sph_bessel_for_l(kb.l);
        fl_all[ip].resize(ng, AccReal(0));
        const auto& r = upf.r;
        const auto& beta_r = kb.beta;
        std::vector<AccReal> integrand(radial_points);
        for (int ig = 0; ig < ng; ++ig) {
            Vec3 q{gv.g[ig][0] + gv.kvec[0], gv.g[ig][1] + gv.kvec[1],
                   gv.g[ig][2] + gv.kvec[2]};
            AccReal Q = std::sqrt(q.norm_squared());
            for (std::size_t ir = 0; ir < radial_points; ++ir)
                integrand[ir] =
                    ir < beta_r.size()
                        ? r[ir] * beta_r[ir] * j_l(Q * r[ir])
                        : AccReal(0); // outside a supplied compact table
            fl_all[ip][ig] = upf.integrate_radial(integrand);
        }
    }

    // D is a real symmetric matrix in Ry. Distinct angular momenta cannot
    // couple in this spherical KB representation; never silently discard an
    // incomplete or unsupported matrix.
    if (upf.d_ij.size() != static_cast<std::size_t>(n_proj) * n_proj)
        throw std::invalid_argument("KB projectors: D_ij shape mismatch");
    for (int i = 0; i < n_proj; ++i)
        for (int j = 0; j < n_proj; ++j) {
            const auto value =
                upf.d_ij[static_cast<std::size_t>(i) * n_proj + j];
            if (!std::isfinite(value)
                || value != upf.d_ij[static_cast<std::size_t>(j) * n_proj + i])
                throw std::invalid_argument(
                    "KB projectors: D_ij must be finite and symmetric");
            if (proj_l[i] != proj_l[j] && value != AccReal(0))
                throw std::invalid_argument(
                    "KB projectors: D_ij couples different angular momenta");
        }

    std::map<int, std::vector<int>> angular_blocks;
    for (int ip = 0; ip < n_proj; ++ip)
        angular_blocks[proj_l[ip]].push_back(ip);
    std::vector<AccReal> eff_D(n_proj);
    std::vector<int> rotated_l(n_proj);
    std::vector<std::vector<AccReal>> fl_trans(n_proj);
    int ip_out = 0;
    for (const auto& [l, indices] : angular_blocks) {
        const int nl = static_cast<int>(indices.size());
        linalg::RealMatrix block = xt::zeros<double>({nl, nl});
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nl; ++j)
                block(i, j) =
                    upf.d_ij[static_cast<std::size_t>(indices[i]) * n_proj
                             + indices[j]];
        // Canonical LAPACK eigensolve: columns U(:,k), D = U diag(lambda) U^T.
        // Handles every block size, including already-diagonal permutations.
        const auto [eigenvalues, eigenvectors] = xt::linalg::eigh(block);
        for (int k = 0; k < nl; ++k, ++ip_out) {
            fl_trans[ip_out].assign(ng, AccReal(0));
            for (int i = 0; i < nl; ++i)
                for (int ig = 0; ig < ng; ++ig)
                    fl_trans[ip_out][ig] +=
                        eigenvectors(i, k) * fl_all[indices[i]][ig];
            eff_D[ip_out] = eigenvalues(k) * AccReal(0.5); // Ry -> Ha, once.
            rotated_l[ip_out] = l;
        }
    }

    auto ng_sz = static_cast<std::size_t>(ng);
    for (int ia = 0; ia < n_atoms; ++ia) {
        out[ia].resize(n_proj);
        for (int ip = 0; ip < n_proj; ++ip) {
            out[ia][ip].l = rotated_l[ip];
            out[ia][ip].d = eff_D[ip];
            out[ia][ip].f_G.resize({ng_sz});
            for (int ig = 0; ig < ng; ++ig)
                out[ia][ip].f_G[ig] = fl_trans[ip][ig];
        }
    }
}

namespace {

/// Build atomic wavefunctions in G-space for the initial guess on one
/// k-sphere; the radial transform is evaluated at q = |k+G|
/// (QE atwfc_mod.f90: chi_G = (4π/√Ω) ∫ χ(r)·j_l(qr)·r dr).
void setup_atomic_wfc(std::vector<std::vector<PWWorkspace::AtomWfc>>& out,
                      const internal::GVectorList& gv,
                      const internal::UPFData& upf, int n_atoms,
                      AccReal omega) {
    int ng = gv.count();
    AccReal pref = AccReal(4.0 * M_PI) / std::sqrt(omega);

    std::vector<std::vector<AccReal>> chi_g_all(upf.nwfc);
    for (int iw = 0; iw < upf.nwfc; ++iw) {
        int l = upf.lchi[iw];
        auto j_l = sph_bessel_for_l(l);
        const auto& r = upf.r;
        const auto& chi_r = upf.chi[iw];
        if (chi_r.size() != r.size())
            throw std::invalid_argument("Atomic wavefunction must match PP_R");
        std::vector<AccReal> integrand(r.size());

        chi_g_all[iw].resize(ng, AccReal(0));
        for (int ig = 0; ig < ng; ++ig) {
            Vec3 q{gv.g[ig][0] + gv.kvec[0], gv.g[ig][1] + gv.kvec[1],
                   gv.g[ig][2] + gv.kvec[2]};
            AccReal Q = std::sqrt(q.norm_squared());
            for (std::size_t ir = 0; ir < r.size(); ++ir)
                integrand[ir] = chi_r[ir] * j_l(Q * r[ir]) * r[ir];
            chi_g_all[iw][ig] = upf.integrate_radial(integrand) * pref;
        }
    }

    // Store for each atom (radial part shared; structure factor applied later)
    out.assign(n_atoms, {});
    auto ng_sz = static_cast<std::size_t>(ng);
    for (int ia = 0; ia < n_atoms; ++ia) {
        out[ia].resize(upf.nwfc);
        for (int iw = 0; iw < upf.nwfc; ++iw) {
            out[ia][iw].l = upf.lchi[iw];
            out[ia][iw].chi_G.resize({ng_sz});
            for (int ig = 0; ig < ng; ++ig)
                out[ia][iw].chi_G[ig] = chi_g_all[iw][ig];
        }
    }
}

} // namespace

/// Radial → G-space transform of the local pseudopotential, QE-style
/// (vloc_mod.f90): V_loc(r) → −Z/r at large r, so the bare integrand
/// r²·V·j0(Gr) does not decay and the quadrature diverges. Add and
/// subtract an erf-screened Coulomb: the difference V + Z·erf(r)/r
/// decays like erfc, and the screened Coulomb has the analytic
/// transform −(4πZ/ΩG²)·exp(−G²/4).
///
/// The G=0 element gets the finite part only (QE's "αZ" term):
/// V(0) = (4π/Ω)∫ r²(V(r) + Z/r) dr — the divergent jellium piece is
/// cancelled against the Ewald/Hartree G=0 conventions exactly as for
/// the local potential. All outputs in Hartree.
std::vector<AccReal> upf_vloc_to_gspace(const internal::UPFData& upf,
                                        const internal::GVectorList& gv,
                                        AccReal omega) {
    std::vector<AccReal> vg(gv.count(), AccReal(0));
    const auto& r = upf.r;
    const auto& vl = upf.v_loc;
    if (vl.size() != r.size())
        throw std::invalid_argument("Local potential PP_LOCAL must match PP_R");
    std::vector<AccReal> integrand(r.size());
    AccReal inv_omega = AccReal(1) / omega;
    AccReal ry_to_ha = AccReal(0.5); // UPF stores V_loc in Ry
    AccReal z = static_cast<AccReal>(upf.z_val);

    for (int ig = 0; ig < gv.count(); ++ig) {
        AccReal G = std::sqrt(gv.g[ig].norm_squared());
        AccReal g2 = gv.g[ig].norm_squared();
        bool g_zero = g2 < AccReal(1e-16);
        for (std::size_t ir = 0; ir < r.size(); ++ir) {
            const AccReal radius = r[ir];
            const AccReal potential = vl[ir] * ry_to_ha;
            if (g_zero) {
                // r²(V + Z/r): exactly zero beyond the core radius.
                integrand[ir] = radius * (radius * potential + z);
            } else {
                AccReal Gr = G * radius;
                AccReal j0 = (Gr < AccReal(1e-8))
                                 ? AccReal(1) - Gr * Gr / AccReal(6)
                                 : std::sin(Gr) / Gr;
                AccReal v_scr = radius * potential + z * std::erf(radius);
                integrand[ir] = radius * v_scr * j0;
            }
        }
        const AccReal sum = upf.integrate_radial(integrand);
        if (g_zero)
            vg[ig] = AccReal(4.0 * M_PI) * sum * inv_omega;
        else
            vg[ig] = AccReal(4.0 * M_PI) * inv_omega
                     * (sum - z * std::exp(-g2 / AccReal(4)) / g2);
    }
    return vg;
}

int valence_from_z(int z) {
    switch (z) {
    case 1:
        return 1;
    case 2:
        return 2;
    case 3:
        return 1;
    case 4:
        return 2;
    case 5:
        return 3;
    case 6:
        return 4;
    case 7:
        return 5;
    case 8:
        return 6;
    case 9:
        return 7;
    case 10:
        return 8;
    case 11:
        return 1;
    case 12:
        return 2;
    case 13:
        return 3;
    case 14:
        return 4;
    case 15:
        return 5;
    case 16:
        return 6;
    case 17:
        return 7;
    case 18:
        return 8;
    default:
        // No-fallback rule: a 0-electron atom would be silently wrong.
        throw std::invalid_argument(
            "valence_from_z: no fallback valence count for Z="
            + std::to_string(z) + " (H..Ar supported); supply a UPF");
    }
}

void require_closed_shell(int n_elec, const char* where) {
    if (n_elec % 2 == 0)
        return;
    throw std::invalid_argument(
        std::string(where) + ": " + std::to_string(n_elec)
        + " valence electrons is not closed-shell — use spin_polarized "
          "for open shell, or set smearing for metallic occupations");
}

UPFData load_pseudopotential(const cpu::Frame& frame,
                             const std::string& upf_path,
                             const std::string& gth_name) {
    // A plane-wave run has no default pseudopotential. Bare valence-charge
    // Coulomb — the old "toy" path — is not representable in a plane-wave
    // basis at any finite cutoff, so it produced numbers that neither were
    // right nor converged with ecut. Exactly one real source is required.
    if (upf_path.empty() == gth_name.empty())
        throw std::invalid_argument(
            "load_pseudopotential: set exactly one of PW::upf_path (a "
            "norm-conserving UPF file) or PW::gth_name (an in-tree analytic "
            "GTH parameterization such as \"GTH-PBE-q4\"). Both were "
            + std::string(upf_path.empty() ? "empty" : "set") + ".");

    if (!upf_path.empty())
        return read_upf_v1(upf_path);

    // Single-species, as for the UPF path: one potential serves every atom.
    // Element identity comes from the Frame's atomic numbers.
    const int n_atoms = static_cast<int>(frame.nlocal());
    if (n_atoms <= 0)
        throw std::invalid_argument(
            "load_pseudopotential: frame has no atoms, so PW::gth_name "
            "cannot be resolved to an element.");
    const int* __restrict sp_type = frame.span(kAtom_type);
    const int z0 = sp_type[0];
    for (int i = 1; i < n_atoms; ++i)
        if (sp_type[i] != z0)
            throw std::invalid_argument(
                "load_pseudopotential: PW::gth_name selects one potential for "
                "the whole cell, but the frame mixes atomic numbers. "
                "Multi-species plane-wave runs are not implemented.");
    return upf_from_gth(
        gpw::GthPotential(static_cast<molrs::Element>(z0), gth_name));
}

void setup_pw_problem(const cpu::Frame& frame, const std::string& upf_path,
                      const std::string& gth_name, double ecut_ry, int k_mesh,
                      PWProblem& out) {
    int n_atoms = static_cast<int>(frame.nlocal());
    const cpu::Box& box_ang = frame.box;
    cpu::Box box_bohr;
    box_bohr.set_cell(box_ang.cell() * AccReal(1.8897259886));

    AccReal ecut_ht = static_cast<AccReal>(ecut_ry) * AccReal(0.5);
    AccReal vol_bohr = box_bohr.volume();
    out.vol_bohr = vol_bohr;

    auto pos_bohr = pos_to_bohr(frame);

    const bool rank0 = log_rank0();

    const internal::UPFData upf_early =
        load_pseudopotential(frame, upf_path, gth_name);
    // Retained for the branch sites below: a pseudopotential is now always
    // present, from a file or from the analytic tables.
    const bool has_upf = true;

    int n_elec = 0;
    std::vector<AccReal> valence_charges;
    const int* __restrict sp_type = frame.span(kAtom_type);
    for (int i = 0; i < n_atoms; ++i) {
        int z = sp_type[i];
        int v = has_upf ? static_cast<int>(
                              std::lround(static_cast<double>(upf_early.z_val)))
                        : valence_from_z(z);
        n_elec += v;
        valence_charges.push_back(static_cast<AccReal>(v));
    }
    out.n_elec = n_elec;
    out.n_bands = std::max(n_elec / 2, 1);

    if (rank0) {
        ATV_INFO("[pw] ========== PW SCF ==========");
        ATV_INFO("[pw] atoms={} elec={} bands={} ecut_ry={} {}", n_atoms,
                 n_elec, out.n_bands, ecut_ry,
                 gth_name.empty() ? "UPF" : gth_name.c_str());
    }

    // === Ewald ===
    std::vector<Vec3> pos_ang;
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);
    for (int i = 0; i < n_atoms; ++i)
        pos_ang.push_back({static_cast<AccReal>(sp_x[i]),
                           static_cast<AccReal>(sp_y[i]),
                           static_cast<AccReal>(sp_z[i])});
    // Ewald uses the same charges as V_loc: pseudopotential valence.
    auto& ewald_charges = valence_charges;
    internal::EwaldSummation ewald(pos_ang, ewald_charges, box_ang, ecut_ht);
    AccReal ew_total = ewald.total_energy();

    // Charged-system / jellium G=0 term: -π Q² / (2 α² Ω). NB: QE's
    // ewald.f90 writes -charge²/(4·alpha)·(2π/Ω), but QE's `alpha` is the
    // Gaussian exponent η² — our EwaldSummation::alpha() is η, so the
    // denominator needs α². (With α¹ the term acquires a spurious
    // cell-size dependence that tilted the Si E(V) curve by ~0.04 Ry
    // across the EOS scan — comparable to the entire physical span.)
    AccReal ew_charge = 0;
    for (auto& z : ewald_charges)
        ew_charge += z;
    AccReal ew_alpha = ewald.alpha();
    AccReal ew_g0 = -ew_charge * ew_charge / (AccReal(4) * ew_alpha * ew_alpha)
                    * AccReal(2.0 * M_PI) / vol_bohr;
    ew_total += ew_g0;
    out.ew_total = ew_total;

    if (rank0) {
        ATV_DEBUG("[pw] --- Ewald (Ha) ---");
        ATV_DEBUG("[pw]   real={} recip={} self={}", ewald.real_energy(),
                  ewald.recip_energy(), ewald.self_energy());
        ATV_DEBUG("[pw]   total={} Ha ({} Ry)", ew_total, ew_total * 2.0);
    }

    // === G-vectors ===
    // gv0 is the DENSITY sphere: ρ = |ψ|² has spectral content out to
    // 2·G_max(wfc), i.e. ecutrho = 4·ecut (QE default). All density-side
    // quantities (ρ, V_H, V_xc, V_loc, V_eff, mixing, energy sums) live
    // on this sphere; the per-k wavefunction spheres stay at ecut.
    out.gv0 = internal::enumerate_g_vectors(box_bohr, AccReal(4) * ecut_ht,
                                            Vec3::zeros());
    auto& gv0 = out.gv0;
    if (rank0) {
        ATV_DEBUG("[pw] --- G-vectors ---");
        ATV_DEBUG("[pw]   N_G={} G_min={} G_max={}", gv0.count(),
                  std::sqrt(gv0.g[1].norm_squared()),
                  std::sqrt(gv0.g[gv0.count() - 1].norm_squared()));
        ATV_DEBUG("[pw]   FFT dims(hkl): {}x{}x{}", gv0.nx, gv0.ny, gv0.nz);
    }

    // k-points
    out.kg = internal::generate_mp_mesh(k_mesh, k_mesh, k_mesh);

    // === Workspace ===
    PWWorkspace& ws = out.ws;
    // Orbital guesses are local to this SCF solve. A new frame/cell or a
    // retry starts from a fresh physical setup, never a failed prior solve.
    ws.wfc = xt::xarray<std::complex<AccReal>>{};
    ws.n_g = gv0.count();
    ws.gvec = &gv0;
    auto ng = static_cast<std::size_t>(ws.n_g);
    ws.rho_G.resize({ng});
    ws.rho_G_old.resize({ng});
    ws.V_H_G.resize({ng});
    ws.V_xc_G.resize({ng});
    ws.V_eff_G.resize({ng});
    ws.V_loc_G.resize({ng});
    ws.T_G.resize({ng});
    ws.mix_beta = AccReal(0.7); // QE default: mixing_beta = 0.7
    ws.omega = vol_bohr;        // cell volume for real-space integration
    ws.atom_pos_bohr =
        pos_bohr; // atom positions for augmentation structure factor

    // Initial density: reuse the UPF already parsed for valence counts.
    const internal::UPFData& upf = upf_early;

    for (int i = 0; i < ws.n_g; ++i)
        ws.rho_G[i] = std::complex<AccReal>(0, 0);

    if (has_upf && !upf.rho_atom.empty()
        && upf.rho_atom.size() == upf.r.size()) {
        // G-space transform of the atomic pseudocharge. UPF PP_RHOATOM
        // stores 4πr²ρ(r) (integrates to Z_val with plain dr), so
        // ρ_atom(G) = (1/Ω) ∫ PP_RHOATOM(r) j₀(Gr) dr.
        std::vector<AccReal> integrand(upf.r.size());
        for (int ig = 0; ig < ws.n_g; ++ig) {
            AccReal G = std::sqrt(gv0.g[ig].norm_squared());
            for (std::size_t ir = 0; ir < upf.r.size(); ++ir) {
                AccReal Gr = G * upf.r[ir];
                AccReal j0 = (Gr < AccReal(1e-8))
                                 ? AccReal(1) - Gr * Gr / AccReal(6)
                                 : std::sin(Gr) / Gr;
                integrand[ir] = upf.rho_atom[ir] * j0;
            }
            const AccReal sum = upf.integrate_radial(integrand);
            AccReal rho_g_atom = sum / vol_bohr;
            // Structure factor: Σ_a exp(-i G·R_a)
            AccReal s_re = 0, s_im = 0;
            for (int ia = 0; ia < n_atoms; ++ia) {
                AccReal phase = gv0.g[ig][0] * pos_bohr[ia][0]
                                + gv0.g[ig][1] * pos_bohr[ia][1]
                                + gv0.g[ig][2] * pos_bohr[ia][2];
                s_re += std::cos(phase);
                s_im -= std::sin(phase);
            }
            ws.rho_G[ig] = rho_g_atom * std::complex<AccReal>(s_re, s_im);
        }
        // Renormalize: total charge must be n_elec
        AccReal total_charge = ws.rho_G[0].real() * vol_bohr;
        if (std::abs(total_charge) > AccReal(1e-16)) {
            AccReal rescale = AccReal(n_elec) / total_charge;
            for (int ig = 0; ig < ws.n_g; ++ig)
                ws.rho_G[ig] *= rescale;
        }
    } else {
        // Fallback: uniform density
        ws.rho_G[0] = std::complex<AccReal>(AccReal(n_elec) / vol_bohr, 0);
    }

    // gv0 is the density sphere (4*ecut) and its dimensions already cover
    // every density Fourier index. They also avoid aliasing G-G' between
    // retained wavefunctions; a second doubling only oversamples the XC grid.
    ws.fft_nx = gv0.nx;
    ws.fft_ny = gv0.ny;
    ws.fft_nz = gv0.nz;
    ws.nr = ws.fft_nx * ws.fft_ny * ws.fft_nz;
    ws.nc = (ws.fft_nx / 2 + 1) * ws.fft_ny * ws.fft_nz;
    ws.rho_r.resize({static_cast<std::size_t>(ws.nr)});
    ws.V_xc_r.resize({static_cast<std::size_t>(ws.nr)});
    ws.fft =
        std::make_unique<internal::FFTPlan3D>(ws.fft_nx, ws.fft_ny, ws.fft_nz);

    if (rank0)
        ATV_DEBUG("[pw]   Density FFT grid: {}x{}x{} (nr={})", ws.fft_nx,
                  ws.fft_ny, ws.fft_nz, ws.nr);

    // === V_loc ===
    if (has_upf) {
        if (rank0)
            ATV_DEBUG("[pw] --- UPF: Z_val={} l_max={} n_proj={} ---",
                      upf.z_val, upf.l_max, upf.kb.size());
        setup_upf_vloc(ws, gv0, upf, vol_bohr);

        // Setup augmentation form factor: rho_atom(G) / Q_atom
        ws.q_ij = upf.q_ij;
        ws.n_proj_total = upf.n_proj();
        ws.aug_form_factor_G.resize({static_cast<std::size_t>(ws.n_g)});
        if (!upf.rho_atom.empty() && upf.rho_atom.size() == upf.r.size()) {
            // PP_RHOATOM stores 4πr²ρ(r): Q = ∫ PP_RHOATOM dr,
            // form factor (G) = (1/ΩQ) ∫ PP_RHOATOM j₀(Gr) dr.
            const AccReal Q_atom = upf.integrate_radial(upf.rho_atom);
            AccReal inv_Q = (std::abs(Q_atom) > AccReal(1e-16))
                                ? AccReal(1) / Q_atom
                                : AccReal(0);
            std::vector<AccReal> integrand(upf.r.size());
            for (int ig = 0; ig < ws.n_g; ++ig) {
                AccReal G = std::sqrt(gv0.g[ig].norm_squared());
                for (std::size_t ir = 0; ir < upf.r.size(); ++ir) {
                    AccReal Gr = G * upf.r[ir];
                    AccReal j0 = (Gr < AccReal(1e-8))
                                     ? AccReal(1) - Gr * Gr / AccReal(6)
                                     : std::sin(Gr) / Gr;
                    integrand[ir] = upf.rho_atom[ir] * j0;
                }
                const AccReal sum = upf.integrate_radial(integrand);
                ws.aug_form_factor_G[ig] = sum * inv_Q / vol_bohr;
            }
        } else {
            for (int ig = 0; ig < ws.n_g; ++ig)
                ws.aug_form_factor_G[ig] = AccReal(0);
        }
    }
    if (rank0)
        ATV_DEBUG("[pw]   V_loc(G) [1]={} [10]={}", ws.V_loc_G[1].real(),
                  ws.V_loc_G[10].real());

    // Pre-compute k-point G-vectors and the per-k nonlocal / wfc tables
    // (radial parts live on q = k+G, so size and values are k-dependent).
    out.gv_k.clear();
    out.gv_k.reserve(out.kg.kpoints.size());
    {
        // MP mesh k-points are fractional (crystal) coordinates;
        // enumerate_g_vectors needs Cartesian [1/Bohr].
        auto recip = box_bohr.recip_cell();
        for (auto& kf : out.kg.kpoints) {
            Vec3 kc{
                recip(0, 0) * kf[0] + recip(0, 1) * kf[1] + recip(0, 2) * kf[2],
                recip(1, 0) * kf[0] + recip(1, 1) * kf[1] + recip(1, 2) * kf[2],
                recip(2, 0) * kf[0] + recip(2, 1) * kf[1]
                    + recip(2, 2) * kf[2]};
            out.gv_k.push_back(
                internal::enumerate_g_vectors(box_bohr, ecut_ht, kc));
        }
    }

    out.kb_proj_k.clear();
    out.atom_wfc_k.clear();
    if (has_upf) {
        auto n_k = out.gv_k.size();
        out.kb_proj_k.resize(n_k);
        out.atom_wfc_k.resize(n_k);
        for (size_t ik = 0; ik < n_k; ++ik) {
            setup_kb_projectors(out.kb_proj_k[ik], out.gv_k[ik], upf, pos_bohr);
            if (upf.nwfc > 0)
                setup_atomic_wfc(out.atom_wfc_k[ik], out.gv_k[ik], upf, n_atoms,
                                 vol_bohr);
        }
        if (rank0)
            ATV_DEBUG("[pw]   KB projectors: {} proj, {} atoms, {} k-spheres "
                      "(atomic wfc: {})",
                      upf.kb.size(), n_atoms, n_k, upf.nwfc);
    }
}

void rebuild_k_tables(PWProblem& out, const cpu::Frame& frame,
                      const std::string& upf_path, const std::string& gth_name,
                      double ecut_ry, const std::vector<Vec3>& kfrac) {
    if (kfrac.empty())
        throw std::invalid_argument("rebuild_k_tables: empty k-list");

    const cpu::Box& box_ang = frame.box;
    cpu::Box box_bohr;
    box_bohr.set_cell(box_ang.cell() * AccReal(1.8897259886));
    AccReal ecut_ht = static_cast<AccReal>(ecut_ry) * AccReal(0.5);
    AccReal vol_bohr = box_bohr.volume();
    auto pos_bohr = pos_to_bohr(frame);
    int n_atoms = static_cast<int>(pos_bohr.size());

    out.kg.kpoints = kfrac;
    const AccReal w = AccReal(1) / AccReal(static_cast<int>(kfrac.size()));
    out.kg.weights.assign(kfrac.size(), w);

    auto recip = box_bohr.recip_cell();
    out.gv_k.clear();
    out.gv_k.reserve(kfrac.size());
    for (const auto& kf : kfrac) {
        Vec3 kc{recip(0, 0) * kf[0] + recip(0, 1) * kf[1] + recip(0, 2) * kf[2],
                recip(1, 0) * kf[0] + recip(1, 1) * kf[1] + recip(1, 2) * kf[2],
                recip(2, 0) * kf[0] + recip(2, 1) * kf[1]
                    + recip(2, 2) * kf[2]};
        out.gv_k.push_back(
            internal::enumerate_g_vectors(box_bohr, ecut_ht, kc));
    }

    out.kb_proj_k.clear();
    out.atom_wfc_k.clear();
    {
        // Same pseudopotential source as setup_pw_problem — a k-path rebuild
        // must not silently switch potentials.
        const int* __restrict sp_t = frame.span(kAtom_type);
        auto upf = gth_name.empty()
                       ? internal::read_upf_v1(upf_path)
                       : internal::upf_from_gth(gpw::GthPotential(
                             static_cast<molrs::Element>(sp_t[0]), gth_name));
        const auto n_k = out.gv_k.size();
        out.kb_proj_k.resize(n_k);
        out.atom_wfc_k.resize(n_k);
        for (size_t ik = 0; ik < n_k; ++ik) {
            setup_kb_projectors(out.kb_proj_k[ik], out.gv_k[ik], upf, pos_bohr);
            if (upf.nwfc > 0)
                setup_atomic_wfc(out.atom_wfc_k[ik], out.gv_k[ik], upf, n_atoms,
                                 vol_bohr);
        }
    }
}

void solve_pulay_coefficients(const AccReal* B, int m, AccReal* c_out) {
    // Solve B·c = 1 (with constraint Σc=1 via Lagrangian)
    // Augmented system: [B  1] [c] = [0]
    //                    [1  0] [λ]   [1]
    const int depth = m;
    int n = depth + 1;
    std::vector<AccReal> A(n * n, AccReal(0));
    for (int i = 0; i < depth; ++i)
        for (int j = 0; j < depth; ++j)
            A[i * n + j] = B[i * depth + j];
    for (int i = 0; i < depth; ++i)
        A[i * n + depth] = A[depth * n + i] = AccReal(1);

    std::vector<AccReal> rhs(n, AccReal(0));
    rhs[depth] = AccReal(1);

    // Solve via Gaussian elimination
    std::vector<AccReal> c(n, AccReal(0));
    auto aug = A;
    auto b = rhs;
    for (int k = 0; k < n; ++k) {
        // Pivot
        int imax = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(aug[i * n + k]) > std::abs(aug[imax * n + k]))
                imax = i;
        if (imax != k) {
            for (int j = 0; j < n; ++j)
                std::swap(aug[k * n + j], aug[imax * n + j]);
            std::swap(b[k], b[imax]);
        }
        if (std::abs(aug[k * n + k]) < AccReal(1e-14))
            continue;
        for (int i = k + 1; i < n; ++i) {
            AccReal factor = aug[i * n + k] / aug[k * n + k];
            for (int j = k; j < n; ++j)
                aug[i * n + j] -= factor * aug[k * n + j];
            b[i] -= factor * b[k];
        }
    }
    for (int k = n - 1; k >= 0; --k) {
        AccReal sum = b[k];
        for (int j = k + 1; j < n; ++j)
            sum -= aug[k * n + j] * c[j];
        if (std::abs(aug[k * n + k]) > AccReal(1e-14))
            c[k] = sum / aug[k * n + k];
    }
    for (int i = 0; i < depth; ++i)
        c_out[i] = c[i];
}

void pulay_mix(xt::xarray<std::complex<AccReal>>& output,
               const xt::xarray<std::complex<AccReal>>& input,
               std::vector<xt::xarray<std::complex<AccReal>>>& residual_history,
               std::vector<xt::xarray<std::complex<AccReal>>>& density_history,
               AccReal fraction, int history_depth) {
    const auto n = input.size();
    if (n == 0 || output.size() != n || history_depth < 0
        || residual_history.size() != density_history.size()
        || !std::isfinite(fraction))
        throw std::invalid_argument("PW Pulay mix: invalid shape or controls");
    const auto finite_vector = [n](const auto& values) {
        if (values.size() != n)
            return false;
        for (const auto& z : values)
            if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
                return false;
        return true;
    };
    if (!finite_vector(input) || !finite_vector(output))
        throw std::invalid_argument("PW Pulay mix: nonfinite density");
    for (std::size_t i = 0; i < residual_history.size(); ++i)
        if (!finite_vector(residual_history[i])
            || !finite_vector(density_history[i]))
            throw std::invalid_argument("PW Pulay mix: invalid history");

    xt::xarray<std::complex<AccReal>> residual = output - input;
    if (!finite_vector(residual))
        throw std::invalid_argument("PW Pulay mix: nonfinite residual");
    if (history_depth == 0) {
        output = input + fraction * residual;
        residual_history.clear();
        density_history.clear();
        return;
    }
    residual_history.push_back(residual);
    density_history.push_back(input);
    while (static_cast<int>(residual_history.size()) > history_depth) {
        residual_history.erase(residual_history.begin());
        density_history.erase(density_history.begin());
    }

    for (;;) {
        const int depth = static_cast<int>(residual_history.size());
        if (depth == 1) {
            output = input + fraction * residual;
            return;
        }
        std::vector<AccReal> B(depth * depth, AccReal(0));
        AccReal scale = 0;
        for (int i = 0; i < depth; ++i) {
            for (int j = i; j < depth; ++j) {
                AccReal dot = 0;
                for (std::size_t g = 0; g < n; ++g)
                    dot += std::real(std::conj(residual_history[i][g])
                                     * residual_history[j][g]);
                B[i * depth + j] = B[j * depth + i] = dot;
            }
            scale = std::max(scale, B[i * depth + i]);
        }
        // A common Gram scale leaves the constrained minimum unchanged and
        // prevents the shared solver's absolute pivot guard from discarding
        // physically small density residuals near self-consistency.
        if (scale > AccReal(0) && std::isfinite(scale))
            for (auto& value : B)
                value /= scale;
        std::vector<AccReal> coefficients(depth, AccReal(0));
        solve_pulay_coefficients(B.data(), depth, coefficients.data());
        AccReal sum = 0;
        bool valid = std::isfinite(scale);
        for (const auto value : coefficients) {
            valid = valid && std::isfinite(value);
            sum += value;
        }
        valid = valid && std::isfinite(sum)
                && std::abs(sum - AccReal(1)) < AccReal(1e-10);
        if (valid) {
            xt::xarray<std::complex<AccReal>> mixed =
                xt::zeros<std::complex<AccReal>>({n});
            for (int i = 0; i < depth; ++i)
                for (std::size_t g = 0; g < n; ++g)
                    mixed[g] += (coefficients[i] / sum)
                                * (density_history[i][g]
                                   + fraction * residual_history[i][g]);
            if (finite_vector(mixed)) {
                output = std::move(mixed);
                return;
            }
        }
        // An unusable singular history is reduced to its newest independent
        // information; the one-entry endpoint is ordinary linear mixing.
        residual_history.erase(residual_history.begin());
        density_history.erase(density_history.begin());
    }
}

PWEnergyTerms combine_energy_terms(AccReal e_band, AccReal e_h,
                                   AccReal e_xc_int, AccReal e_loc,
                                   AccReal e_xc, AccReal ew_total,
                                   AccReal e_aug) {
    PWEnergyTerms t;
    t.e_h = e_h;
    t.e_xc_int = e_xc_int;
    t.e_loc = e_loc;
    // one-electron = Σ ε_i - ∫ρ V_H - ∫ρ V_xc
    t.e_one = e_band - AccReal(2.0) * t.e_h - t.e_xc_int;
    // total = one-electron + Hartree + XC + Ewald + augmentation
    t.total = t.e_one + t.e_h + e_xc + ew_total + e_aug;
    return t;
}

PWEnergyTerms iteration_energy(const PWWorkspace& ws, AccReal e_band,
                               AccReal ew_total, AccReal vol_bohr) {
    // Energy decomposition
    // Parseval: ∫ρ(r)v(r)d³r = Ω·Σ_G Re[ρ*(G)·v(G)] — both real AND
    // imaginary parts contribute (ρ_G is complex for any cell without
    // inversion symmetry about the origin; taking Re·Re only silently
    // dropped the Im·Im half and tilted E(V)).
    AccReal e_h = 0, e_xc_int = 0, e_loc = 0;
    for (int i = 0; i < ws.n_g; ++i) {
        std::complex<AccReal> rh = ws.rho_G[i];
        e_h += AccReal(0.5) * std::real(std::conj(rh) * ws.V_H_G[i]);
        if (ws.nspin == 1)
            e_xc_int += std::real(std::conj(rh) * ws.V_xc_G[i]);
        e_loc += std::real(std::conj(rh) * ws.V_loc_G[i]);
    }
    e_loc *= vol_bohr;
    e_h *= vol_bohr;
    // XCBuilder already integrates rho_up*v_up + rho_down*v_down,
    // including the volume; the scratch V_xc_G holds only the last spin.
    e_xc_int = ws.nspin == 2 ? ws.e_xc_int_spin : e_xc_int * vol_bohr;

    return combine_energy_terms(e_band, e_h, e_xc_int, e_loc, ws.E_xc, ew_total,
                                ws.E_aug);
}

void log_iteration_energy(const PWEnergyTerms& terms, const PWWorkspace& ws,
                          AccReal ew_total, int iter) {
    if (!log_rank0())
        return;
    ATV_DEBUG("[pw] iter={} one_e={:.6f} hartree={:.6f} xc={:.6f} "
              "ewald={:.6f} total_ha={:.8f} ({:.8f} Ry)",
              iter + 1, terms.e_one, terms.e_h, ws.E_xc, ew_total, terms.total,
              terms.total / qe_units::rydberg_to_hartree);
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
