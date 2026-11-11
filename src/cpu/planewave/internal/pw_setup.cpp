/// @file cpu/planewave/internal/pw_setup.cpp
/// @brief Host-side plane-wave SCF problem assembly (moved verbatim from
///        the pre-loop section of the solver) shared by both backends.

#include "atomiverse/cpu/planewave/internal/pw_setup.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/pulay_coefficients.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/pw_radial_derivatives.h"
#include "atomiverse/cpu/planewave/internal/pw_subspace.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/electronic/norm_conserving_potential.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
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

std::complex<AccReal> structure_factor(const Vec3& g,
                                       const std::vector<Vec3>& positions) {
    AccReal re = 0, im = 0;
    for (const auto& pos : positions) {
        const AccReal phase = g[0] * pos[0] + g[1] * pos[1] + g[2] * pos[2];
        re += std::cos(phase);
        im -= std::sin(phase);
    }
    return {re, im};
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
AccReal sph_bessel_j0(AccReal x) { return regularized_bessel_jet(0, x).value; }
AccReal sph_bessel_j1(AccReal x) {
    return x * regularized_bessel_jet(1, x).value;
}
AccReal sph_bessel_j2(AccReal x) {
    return x * x * regularized_bessel_jet(2, x).value;
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
void setup_kb_projectors(KBProjectors& out, const internal::GVectorList& gv,
                         const internal::NcPotentialData& upf,
                         const std::vector<Vec3>& pos_bohr, AccReal omega) {
    if (!(omega > AccReal(0)) || !std::isfinite(omega))
        throw std::invalid_argument(
            "KB projectors: cell volume must be finite and positive");
    int n_atoms = static_cast<int>(pos_bohr.size());
    int ng = gv.count();
    out.assign(n_atoms, {});

    // Precompute f_l(q) for each projector (radial integral, same for all atoms)
    int n_proj = upf.n_proj();
    upf.validate_radial_mesh();
    for (const auto& projector : upf.kb) {
        if (projector.radial_points < 3
            || projector.radial_points > projector.beta.size()
            || projector.beta.size() > upf.r.size())
            throw std::invalid_argument("KB projectors: declare a valid radial "
                                        "integration domain within PP_R");
    }
    std::vector<std::vector<AccReal>> fl_all(n_proj);
    std::vector<int> proj_l(n_proj);
    for (size_t ip = 0; ip < upf.kb.size(); ++ip) {
        auto& kb = upf.kb[ip];
        proj_l[ip] = kb.l;
        fl_all[ip].resize(ng, AccReal(0));
        for (int ig = 0; ig < ng; ++ig) {
            Vec3 q{gv.g[ig][0] + gv.kvec[0], gv.g[ig][1] + gv.kvec[1],
                   gv.g[ig][2] + gv.kvec[2]};
            AccReal Q = std::sqrt(q.norm_squared());
            fl_all[ip][ig] = nc_projector_radial_value(upf, ip, Q);
        }
    }

    // D is a real symmetric matrix in Ha. Distinct angular momenta cannot
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
            eff_D[ip_out] = eigenvalues(k);
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
            const int l = rotated_l[ip];
            const int nm = n_m_for_l(l);
            auto& beta = out[ia][ip].beta_G;
            beta = xt::zeros<std::complex<AccReal>>(
                {static_cast<std::size_t>(nm), ng_sz});
            const AccReal pref = AccReal(4.0 * M_PI) / std::sqrt(omega);
            const std::complex<AccReal> im_l =
                l == 1 ? std::complex<AccReal>(0, -1)
                       : std::complex<AccReal>(l == 2 ? -1 : 1, 0);
            std::array<AccReal, 5> ylm{};
            for (int ig = 0; ig < ng; ++ig) {
                const AccReal fl = fl_trans[ip][ig];
                out[ia][ip].f_G[ig] = fl;
                if (fl == AccReal(0))
                    continue;
                const Vec3 q = gv.g[ig] + gv.kvec;
                const AccReal phase = q[0] * pos_bohr[ia][0]
                                      + q[1] * pos_bohr[ia][1]
                                      + q[2] * pos_bohr[ia][2];
                compute_ylm(q[0], q[1], q[2], std::sqrt(q.norm_squared()), l,
                            ylm.data());
                const std::complex<AccReal> atomic_phase(std::cos(phase),
                                                         -std::sin(phase));
                for (int m = 0; m < nm; ++m)
                    beta(m, ig) = pref * im_l * ylm[m] * fl * atomic_phase;
            }
        }
    }
}

namespace {

/// Build atomic wavefunctions in G-space for the initial guess on one
/// k-sphere; the radial transform is evaluated at q = |k+G|
/// (QE atwfc_mod.f90: chi_G = (4π/√Ω) ∫ χ(r)·j_l(qr)·r dr).
void setup_atomic_wfc(AtomicOrbitals& out, const internal::GVectorList& gv,
                      const internal::NcPotentialData& upf, int n_atoms,
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
std::vector<AccReal> nc_vloc_to_gspace(const internal::NcPotentialData& upf,
                                       const internal::GVectorList& gv,
                                       AccReal omega) {
    std::vector<AccReal> vg(gv.count(), AccReal(0));
    for (int ig = 0; ig < gv.count(); ++ig) {
        AccReal G = std::sqrt(gv.g[ig].norm_squared());
        vg[ig] = nc_vloc_fourier_value(upf, G, omega);
    }
    return vg;
}

void require_closed_shell(int n_elec, const char* where) {
    if (n_elec % 2 == 0)
        return;
    throw std::invalid_argument(
        std::string(where) + ": " + std::to_string(n_elec)
        + " valence electrons is not closed-shell — use spin_polarized "
          "for open shell, or set smearing for metallic occupations");
}

std::vector<std::vector<AccReal>> nc_vloc_to_gspace(
    std::span<const std::shared_ptr<const NcPotentialData>> species,
    const GVectorList& gv, AccReal omega) {
    std::vector<std::vector<AccReal>> form_factors;
    form_factors.reserve(species.size());
    for (const auto& nc : species)
        form_factors.push_back(nc_vloc_to_gspace(*nc, gv, omega));
    return form_factors;
}

static void setup_species_k_tables(PWKMesh& out, const PWIons& ions) {
    const auto n_atoms = ions.atom_species.size();
    const auto n_k = out.gv_k.size();
    out.kb_proj_k.assign(n_k, {});
    out.atom_wfc_k.assign(n_k, {});
    for (std::size_t ik = 0; ik < n_k; ++ik) {
        out.kb_proj_k[ik].resize(n_atoms);
        out.atom_wfc_k[ik].resize(n_atoms);
    }
    for (std::size_t kind = 0; kind < ions.species.size(); ++kind) {
        const auto& nc = *ions.species[kind];
        std::vector<std::size_t> atoms;
        std::vector<Vec3> positions;
        for (std::size_t ia = 0; ia < n_atoms; ++ia) {
            if (ions.atom_species[ia] != kind)
                continue;
            atoms.push_back(ia);
            positions.push_back(ions.positions[ia]);
        }
        if (atoms.empty())
            continue;
        for (std::size_t ik = 0; ik < n_k; ++ik) {
            KBProjectors projectors;
            setup_kb_projectors(projectors, out.gv_k[ik], nc, positions,
                                ions.vol_bohr);
            for (std::size_t j = 0; j < atoms.size(); ++j)
                out.kb_proj_k[ik][atoms[j]] = std::move(projectors[j]);
            if (nc.nwfc > 0) {
                AtomicOrbitals orbitals;
                setup_atomic_wfc(orbitals, out.gv_k[ik], nc,
                                 static_cast<int>(atoms.size()), ions.vol_bohr);
                for (std::size_t j = 0; j < atoms.size(); ++j)
                    out.atom_wfc_k[ik][atoms[j]] = std::move(orbitals[j]);
            }
        }
    }
}

// Keep integer mode labels and the quadrature geometry fixed while changing
// their metric. Sorting or re-enumerating at the strained cutoff would give a
// discontinuous band space rather than the derivative of the reference basis.
static GVectorList strained_g_vectors(const GVectorList& reference,
                                      const cpu::Box& box_bohr,
                                      const Vec3& kcart) {
    GVectorList out = reference;
    out.kvec = kcart;
    const auto recip = box_bohr.recip_cell();
    for (int ig = 0; ig < out.count(); ++ig) {
        for (int axis = 0; axis < 3; ++axis)
            out.g[ig][axis] = recip(axis, 0) * AccReal(out.h[ig])
                              + recip(axis, 1) * AccReal(out.k[ig])
                              + recip(axis, 2) * AccReal(out.l[ig]);
        const Vec3 q = out.g[ig] + kcart;
        out.ekin[ig] = AccReal(.5) * q.norm_squared();
    }
    return out;
}

static PWKMesh assemble_k_mesh(const PWIons& ions, const cpu::Box& box_bohr,
                               AccReal ecut_ht, KPointGrid grid,
                               std::span<const GVectorList> reference) {
    PWKMesh mesh;
    mesh.kg = std::move(grid);
    mesh.gv_k.reserve(mesh.kg.kpoints.size());
    const auto recip = box_bohr.recip_cell();
    for (std::size_t ik = 0; ik < mesh.kg.kpoints.size(); ++ik) {
        const auto& kf = mesh.kg.kpoints[ik];
        const Vec3 kc{
            recip(0, 0) * kf[0] + recip(0, 1) * kf[1] + recip(0, 2) * kf[2],
            recip(1, 0) * kf[0] + recip(1, 1) * kf[1] + recip(1, 2) * kf[2],
            recip(2, 0) * kf[0] + recip(2, 1) * kf[1] + recip(2, 2) * kf[2]};
        mesh.gv_k.push_back(
            !reference.empty() ? strained_g_vectors(reference[ik], box_bohr, kc)
                               : enumerate_g_vectors(box_bohr, ecut_ht, kc));
    }
    setup_species_k_tables(mesh, ions);
    return mesh;
}

static void setup_pw_problem_impl(
    const cpu::Frame& frame,
    std::vector<std::shared_ptr<const NcPotentialData>> species,
    NcComposition composition, double ecut_ry, int k_mesh, PWSetup& out,
    PWWorkspace& ws, const PWStrainReference* reference) {
    if (!std::isfinite(ecut_ry) || ecut_ry <= 0 || k_mesh < 1)
        throw std::invalid_argument(
            "PW setup requires a positive cutoff and k-mesh");
    int n_atoms = static_cast<int>(frame.nlocal());
    const cpu::Box& box_ang = frame.box;
    cpu::Box box_bohr;
    box_bohr.set_cell(box_ang.cell() * AccReal(1.8897259886));

    AccReal ecut_ht = static_cast<AccReal>(ecut_ry) * AccReal(0.5);
    AccReal vol_bohr = box_bohr.volume();
    if (!std::isfinite(vol_bohr) || vol_bohr <= 0)
        throw std::invalid_argument(
            "PW setup requires a finite positive cell volume");
    out.ions.vol_bohr = vol_bohr;
    out.ions.species = std::move(species);
    out.ions.atom_species = std::move(composition.atom_species);
    out.ions.ionic_charges = std::move(composition.ionic_charges);

    out.ions.positions = pos_to_bohr(frame);
    const auto& pos_bohr = out.ions.positions;

    const bool rank0 = log_rank0();

    const int n_elec = composition.n_elec;
    out.ions.n_elec = n_elec;
    out.n_bands = std::max(n_elec / 2, 1);

    if (rank0) {
        ATV_INFO("[pw] ========== PW SCF ==========");
        ATV_INFO("[pw] atoms={} elec={} bands={} ecut_ry={} NC species={}",
                 n_atoms, n_elec, out.n_bands, ecut_ry,
                 out.ions.species.size());
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
    internal::EwaldSummation ewald(pos_ang, out.ions.ionic_charges, box_ang);
    const AccReal ew_total = ewald.total_energy();
    out.ions.ew_total = ew_total;

    if (rank0) {
        ATV_DEBUG("[pw] --- Ewald (Ha) ---");
        ATV_DEBUG("[pw]   real={} recip={} self={} background={}",
                  ewald.real_energy(), ewald.recip_energy(),
                  ewald.self_energy(), ewald.background_energy());
        ATV_DEBUG("[pw]   total={} Ha ({} Ry)", ew_total, ew_total * 2.0);
    }

    // === G-vectors ===
    // gv0 is the DENSITY sphere: ρ = |ψ|² has spectral content out to
    // 2·G_max(wfc), i.e. ecutrho = 4·ecut (QE default). All density-side
    // quantities (ρ, V_H, V_xc, V_loc, V_eff, mixing, energy sums) live
    // on this sphere; the per-k wavefunction spheres stay at ecut.
    out.gv0 = reference ? strained_g_vectors(reference->density_modes, box_bohr,
                                             Vec3::zeros())
                        : internal::enumerate_g_vectors(
                              box_bohr, AccReal(4) * ecut_ht, Vec3::zeros());
    auto& gv0 = out.gv0;
    if (rank0) {
        ATV_DEBUG("[pw] --- G-vectors ---");
        ATV_DEBUG("[pw]   N_G={} G_min={} G_max={}", gv0.count(),
                  std::sqrt(gv0.g[1].norm_squared()),
                  std::sqrt(gv0.g[gv0.count() - 1].norm_squared()));
        ATV_DEBUG("[pw]   FFT dims(hkl): {}x{}x{}", gv0.nx, gv0.ny, gv0.nz);
    }

    // === Workspace ===
    // Orbital guesses are local to this SCF solve. A new frame/cell or a
    // retry starts from a fresh physical setup, never a failed prior solve.
    ws.wfc = xt::xarray<std::complex<AccReal>>{};
    ws.bind({gv0});
    auto ng = static_cast<std::size_t>(ws.n_g);
    ws.rho_G.resize({ng});
    ws.rho_G_old.resize({ng});
    ws.V_H_G.resize({ng});
    ws.V_xc_G.resize({ng});
    ws.V_eff_G.resize({ng});
    out.V_loc_G.assign(ng, {});
    ws.V_loc_G = out.V_loc_G;
    ws.mix_beta = AccReal(0.7); // QE default: mixing_beta = 0.7
    ws.omega = vol_bohr;        // cell volume for real-space integration
    ws.atom_pos_bohr =
        pos_bohr; // atom positions for local/nonlocal structure factors

    // Assemble local potential and starting density with the same phases.
    for (int ig = 0; ig < ws.n_g; ++ig) {
        ws.rho_G[ig] = {};
    }
    for (std::size_t kind = 0; kind < out.ions.species.size(); ++kind) {
        std::vector<Vec3> positions;
        for (std::size_t ia = 0; ia < out.ions.atom_species.size(); ++ia)
            if (out.ions.atom_species[ia] == kind)
                positions.push_back(pos_bohr[ia]);
        if (positions.empty())
            continue;
        const auto& nc = *out.ions.species[kind];
        const auto vg = nc_vloc_to_gspace(nc, gv0, vol_bohr);
        // Providers without an atomic pseudocharge seed a uniform share.
        if (nc.rho_atom.empty())
            ws.rho_G[0] += AccReal(positions.size()) * nc.z_val / vol_bohr;
        std::vector<AccReal> integrand(nc.rho_atom.size());
        for (int ig = 0; ig < ws.n_g; ++ig) {
            const auto phase = structure_factor(gv0.g[ig], positions);
            out.V_loc_G[ig] += vg[ig] * phase;
            if (integrand.empty())
                continue;
            // PP_RHOATOM stores 4πr²ρ(r), integrated against j₀(Gr).
            const AccReal g = gv0.g[ig].norm();
            for (std::size_t ir = 0; ir < integrand.size(); ++ir) {
                const AccReal gr = g * nc.r[ir];
                const AccReal j0 = gr < AccReal(1e-8)
                                       ? AccReal(1) - gr * gr / AccReal(6)
                                       : std::sin(gr) / gr;
                integrand[ir] = nc.rho_atom[ir] * j0;
            }
            ws.rho_G[ig] += (nc.integrate_radial(integrand) / vol_bohr) * phase;
        }
    }
    const AccReal total_charge = ws.rho_G[0].real() * vol_bohr;
    if (std::abs(total_charge) > AccReal(1e-16)) {
        const AccReal rescale = AccReal(n_elec) / total_charge;
        for (int ig = 0; ig < ws.n_g; ++ig)
            ws.rho_G[ig] *= rescale;
    } else {
        ws.rho_G[0] = AccReal(n_elec) / vol_bohr;
    }

    if (rank0)
        ATV_DEBUG("[pw]   V_loc(G=0)={}", ws.V_loc_G[0].real());

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

    out.mesh = assemble_k_mesh(
        out.ions, box_bohr, ecut_ht,
        reference ? reference->grid : generate_mp_mesh(k_mesh, k_mesh, k_mesh),
        reference ? reference->wavefunction_modes
                  : std::span<const GVectorList>{});
}

void setup_pw_problem(const cpu::Frame& frame, NcProviders potentials,
                      double ecut_ry, int k_mesh, PWSetup& out,
                      PWWorkspace& ws) {
    auto composition = assign_nc_species(frame, potentials);
    if (!std::isfinite(ecut_ry) || ecut_ry <= 0 || k_mesh < 1)
        throw std::invalid_argument(
            "PW setup requires a positive cutoff and k-mesh");
    std::vector<std::shared_ptr<const NcPotentialData>> species;
    species.reserve(potentials.size());
    for (const auto& potential : potentials)
        species.push_back(std::make_shared<const NcPotentialData>(*potential));
    setup_pw_problem_impl(frame, std::move(species), std::move(composition),
                          ecut_ry, k_mesh, out, ws, nullptr);
}

void setup_pw_problem_fixed_basis(const cpu::Frame& frame,
                                  PWStrainReference reference, double ecut_ry,
                                  int k_mesh, PWSetup& out, PWWorkspace& ws) {
    if (&reference.density_modes == &out.gv0
        || reference.density_modes.count() == 0
        || reference.wavefunction_modes.size() != reference.grid.kpoints.size()
        || reference.grid.kpoints.empty())
        throw std::invalid_argument(
            "PW fixed-basis setup requires a distinct complete reference");
    const auto mesh = static_cast<std::size_t>(k_mesh);
    if (k_mesh < 1 || mesh * mesh * mesh != reference.grid.kpoints.size())
        throw std::invalid_argument(
            "PW fixed-basis k-mesh differs from reference");
    auto composition = assign_nc_species(frame, reference.ions.species);
    setup_pw_problem_impl(frame, reference.ions.species, std::move(composition),
                          ecut_ry, k_mesh, out, ws, &reference);
    const auto same_modes = [](const GVectorList& a, const GVectorList& b) {
        return a.h == b.h && a.k == b.k && a.l == b.l && a.nx == b.nx
               && a.ny == b.ny && a.nz == b.nz && a.count() == b.count();
    };
    if (!same_modes(out.gv0, reference.density_modes)
        || out.mesh.gv_k.size() != reference.wavefunction_modes.size())
        throw std::logic_error("PW fixed-basis density modes changed");
    for (std::size_t ik = 0; ik < out.mesh.gv_k.size(); ++ik)
        if (!same_modes(out.mesh.gv_k[ik], reference.wavefunction_modes[ik]))
            throw std::logic_error("PW fixed-basis wavefunction modes changed");
}

PWKMesh make_pw_k_mesh(const PWIons& ions, const cpu::Box& box, double ecut_ry,
                       const std::vector<Vec3>& kfrac) {
    if (kfrac.empty())
        throw std::invalid_argument("PW k-mesh: empty k-list");
    cpu::Box box_bohr;
    box_bohr.set_cell(box.cell() * AccReal(1.8897259886));
    KPointGrid grid;
    grid.kpoints = kfrac;
    grid.weights.assign(kfrac.size(), AccReal(1) / AccReal(kfrac.size()));
    return assemble_k_mesh(ions, box_bohr, AccReal(ecut_ry) * AccReal(.5),
                           std::move(grid), {});
}

void solve_pulay_coefficients(const AccReal* B, int m, AccReal* c_out) {
    // Same unit-sum system as cpu::linalg::pulay_coefficients. The previous
    // bordered matrix used +1 on the constraint row; that only flips the
    // Lagrange multiplier. A rejected solve leaves nonfinite coefficients so
    // pulay_mix drops the oldest history entry.
    if (m < 1)
        return;
    std::vector<double> gram(static_cast<std::size_t>(m)
                             * static_cast<std::size_t>(m));
    std::vector<double> solved(static_cast<std::size_t>(m), 0.0);
    for (std::size_t i = 0; i < gram.size(); ++i)
        gram[i] = static_cast<double>(B[i]);
    if (!cpu::linalg::try_pulay_coefficients(gram.data(), m, solved.data())) {
        for (int i = 0; i < m; ++i)
            c_out[i] = std::numeric_limits<AccReal>::quiet_NaN();
        return;
    }
    for (int i = 0; i < m; ++i)
        c_out[i] = static_cast<AccReal>(solved[static_cast<std::size_t>(i)]);
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
                                   AccReal e_xc, AccReal ew_total) {
    PWEnergyTerms t;
    t.e_h = e_h;
    t.e_xc_int = e_xc_int;
    t.e_loc = e_loc;
    // one-electron = Σ ε_i - ∫ρ V_H - ∫ρ V_xc
    t.e_one = e_band - AccReal(2.0) * t.e_h - t.e_xc_int;
    // total = one-electron + Hartree + XC + Ewald
    t.total = t.e_one + t.e_h + e_xc + ew_total;
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

    return combine_energy_terms(e_band, e_h, e_xc_int, e_loc, ws.E_xc,
                                ew_total);
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
