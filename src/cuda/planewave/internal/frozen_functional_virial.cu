#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/pw_radial_derivatives.h"
#include "atomiverse/cpu/planewave/internal/xc_quadrature.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/planewave/internal/frozen_functional_virial.cuh"
#include "atomiverse/cuda/planewave/internal/grid_map.cuh"
#include "atomiverse/cuda/planewave/internal/pw_forces.cuh"
#include "atomiverse/cuda/planewave/internal/xc_frozen_virial.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace ATV::cuda::pw::internal {
namespace dev = ::ATV::cuda::pw::internal;
namespace host = ::ATV::cpu::pw::internal;
using namespace ATV;
namespace {
constexpr int block = 256;
constexpr int aa[6] = {0, 1, 2, 0, 0, 1}, bb[6] = {0, 1, 2, 1, 2, 2};
void require(bool yes, const char* why) {
    if (!yes)
        throw std::runtime_error(why);
}
void blas_check(cublasStatus_t status) {
    require(status == CUBLAS_STATUS_SUCCESS, "total prototype cuBLAS");
}
__global__ void add_ionic(double* F, const double* ionic, int n) {
    int i = blockIdx.x * block + threadIdx.x;
    if (i < 3 * n)
        F[i] += ionic[i];
}
__global__ void sum_terms(dev::FrozenOperatorVirial* t) {
    int lane = threadIdx.x;
    if (lane >= 10)
        return;
    double value = 0;
    for (int i = 0; i < 6; ++i)
        value += lane ? t[i].virial_ha[lane - 1] : t[i].energy_ha;
    if (lane)
        t[6].virial_ha[lane - 1] = value;
    else
        t[6].energy_ha = value;
}
__global__ void publish_candidate(cuda::Frame target, const double* F,
                                  const dev::FrozenOperatorVirial* terms, int n,
                                  bool force, bool stress, double force_scale,
                                  double physical_energy_ev,
                                  double hartree_ev) {
    const int i = blockIdx.x * block + threadIdx.x;
    if (force && i < n) {
        cuda::add_force(target.atoms.fx, i, F[3 * i] * force_scale,
                        target.force_fault);
        cuda::add_force(target.atoms.fy, i, F[3 * i + 1] * force_scale,
                        target.force_fault);
        cuda::add_force(target.atoms.fz, i, F[3 * i + 2] * force_scale,
                        target.force_fault);
    }
    if (i == 0) {
        target.scalars->energy += physical_energy_ev;
        if (stress) {
            const int a[6] = {0, 1, 2, 0, 0, 1}, b[6] = {0, 1, 2, 1, 2, 2};
            for (int v = 0; v < 6; ++v)
                target.scalars->stress[v] +=
                    .5
                    * (terms[6].virial_ha[3 * a[v] + b[v]]
                       + terms[6].virial_ha[3 * b[v] + a[v]])
                    * hartree_ev;
        }
    }
}
} // namespace
namespace {
__global__ void certificate_kernel(
    const FrozenOperatorVirial* terms, const double* force,
    const cufftDoubleComplex* reference, const cufftDoubleComplex* current,
    const cufftDoubleComplex* orbital, int ng0, int atoms, bool want_force,
    bool want_stress, FrozenDerivativeCertificate* out, double force_scale,
    double hartree_ev, double volume) {
    __shared__ double anti[256], delta[256];
    __shared__ int finite[256], same[256];
    int t = threadIdx.x;
    double maxanti = 0, diff = 0;
    int good = 1, unchanged = 1;
    if (want_force)
        for (int i = t; i < 3 * atoms; i += 256)
            good &= isfinite(force[i]) && isfinite(force[i] * force_scale);
    if (want_stress) {
        for (int i = t; i < 70; i += 256) {
            const auto& term = terms[i / 10];
            double v = i % 10 ? term.virial_ha[i % 10 - 1] : term.energy_ha;
            good &= isfinite(v);
        }
        if (t < 9) {
            int a = t / 3, b = t % 3;
            maxanti =
                fabs(terms[6].virial_ha[t] - terms[6].virial_ha[3 * b + a]);
            const double w =
                .5 * (terms[6].virial_ha[t] + terms[6].virial_ha[3 * b + a]);
            good &=
                isfinite(w * hartree_ev) && isfinite(w * hartree_ev / volume);
        }
    }
    for (int i = t; i < ng0; i += 256) {
        const auto r = reference[i], c = current[i], o = orbital[i];
        good &= isfinite(r.x) && isfinite(r.y) && isfinite(c.x) && isfinite(c.y)
                && isfinite(o.x) && isfinite(o.y);
        unchanged &= __double_as_longlong(r.x) == __double_as_longlong(c.x)
                     && __double_as_longlong(r.y) == __double_as_longlong(c.y);
        diff += (r.x - o.x) * (r.x - o.x) + (r.y - o.y) * (r.y - o.y);
    }
    anti[t] = maxanti;
    delta[t] = diff;
    finite[t] = good;
    same[t] = unchanged;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (t < step) {
            anti[t] = fmax(anti[t], anti[t + step]);
            delta[t] += delta[t + step];
            finite[t] &= finite[t + step];
            same[t] &= same[t + step];
        }
        __syncthreads();
    }
    if (t == 0)
        *out = {anti[0], delta[0], finite[0], same[0]};
}

__global__ void spin_domain_kernel(FrozenSpinDomainInput input,
                                   FrozenSpinDomainStatus* output) {
    __shared__ double minimum[2][256], polarization[256];
    __shared__ unsigned violations[256], finite[256];
    double lo[2] = {INFINITY, INFINITY}, zmax = 0;
    unsigned flags = 0, valid = 1;
    for (std::size_t i = threadIdx.x; i < static_cast<std::size_t>(input.count);
         i += blockDim.x) {
        const double a = input.rho[0][i], b = input.rho[1][i];
        valid &= isfinite(a) && isfinite(b);
        lo[0] = fmin(lo[0], a);
        lo[1] = fmin(lo[1], b);
        // This is an interior-domain qualification, not physical screening.
        // Floors/cutoff track the existing point law. Near-full polarization
        // lies outside the qualified interior for LSDA as well as GGA;
        // the latter explicitly regulates zeta at 1-1e-14.
        if (a <= 1e-15 || b <= 1e-15)
            flags |= 1u;
        const double pa = fmax(a, 0.), pb = fmax(b, 0.), total = pa + pb;
        valid &= isfinite(total);
        if (total <= 1e-12)
            flags |= 2u;
        if (total > 0 && isfinite(total)) {
            const double zeta = (pa - pb) / total;
            zmax = fmax(zmax, fabs(zeta));
            if (zeta <= (-1. + 1e-14) || zeta >= (1. - 1e-14))
                flags |= 4u;
        }
        if (input.gga) {
            double g[2][3];
            for (int s = 0; s < 2; ++s)
                for (int d = 0; d < 3; ++d) {
                    g[s][d] = input.gradient[s][d][i];
                    valid &= isfinite(g[s][d]);
                }
            const double aa =
                g[0][0] * g[0][0] + g[0][1] * g[0][1] + g[0][2] * g[0][2];
            const double bb =
                g[1][0] * g[1][0] + g[1][1] * g[1][1] + g[1][2] * g[1][2];
            const double ab =
                g[0][0] * g[1][0] + g[0][1] * g[1][1] + g[0][2] * g[1][2];
            const double average = .5 * (aa + bb), sigma = aa + 2 * ab + bb;
            valid &= isfinite(aa) && isfinite(bb) && isfinite(ab)
                     && isfinite(average) && isfinite(sigma);
            if (ab < -average || ab > average || sigma < 0)
                flags |= 8u;
        }
    }
    minimum[0][threadIdx.x] = lo[0];
    minimum[1][threadIdx.x] = lo[1];
    polarization[threadIdx.x] = zmax;
    violations[threadIdx.x] = flags;
    finite[threadIdx.x] = valid;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step) {
            for (int s = 0; s < 2; ++s)
                minimum[s][threadIdx.x] = fmin(minimum[s][threadIdx.x],
                                               minimum[s][threadIdx.x + step]);
            polarization[threadIdx.x] = fmax(polarization[threadIdx.x],
                                             polarization[threadIdx.x + step]);
            violations[threadIdx.x] |= violations[threadIdx.x + step];
            finite[threadIdx.x] &= finite[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *output = {{minimum[0][0], minimum[1][0]},
                   polarization[0],
                   violations[0],
                   finite[0]};
}
struct FixedSpinChargePacket {
    double f_count[2]{}, orbital_charge[2]{}, rho_charge[2]{};
    int finite = 1, prefix_exact = 1;
};
__global__ void sum_spin_density(const cufftDoubleComplex* a,
                                 const cufftDoubleComplex* b,
                                 cufftDoubleComplex* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        total[i] = {a[i].x + b[i].x, a[i].y + b[i].y};
}
__global__ void certify_spin_reference(const cufftDoubleComplex* before_a,
                                       const cufftDoubleComplex* before_b,
                                       const cufftDoubleComplex* after_a,
                                       const cufftDoubleComplex* after_b, int n,
                                       FrozenDerivativeCertificate* packet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        auto a = before_a[i], b = before_b[i], c = after_a[i], d = after_b[i];
        if (__double_as_longlong(a.x) != __double_as_longlong(c.x)
            || __double_as_longlong(a.y) != __double_as_longlong(c.y)
            || __double_as_longlong(b.x) != __double_as_longlong(d.x)
            || __double_as_longlong(b.y) != __double_as_longlong(d.y))
            atomicAnd(&packet->reference_unchanged, 0);
    }
}
__global__ void fixed_spin_charge_packet(const cufftDoubleComplex* C,
                                         const double* f, int ng, int nb,
                                         FixedSpinOccupations occupied,
                                         const cufftDoubleComplex* ra,
                                         const cufftDoubleComplex* rb, int zero,
                                         double omega,
                                         FixedSpinChargePacket* out) {
    if (threadIdx.x || blockIdx.x)
        return;
    FixedSpinChargePacket v{};
    const int count[2] = {occupied.alpha, occupied.beta};
    for (int spin = 0; spin < 2; ++spin) {
        for (int b = 0; b < nb; ++b) {
            int band = spin * nb + b;
            double occ = f[band], expected = b < count[spin] ? 1. : 0.;
            v.finite &= isfinite(occ);
            v.prefix_exact &=
                __double_as_longlong(occ) == __double_as_longlong(expected);
            v.f_count[spin] += occ;
            double norm = 0;
            for (int g = 0; g < ng; ++g) {
                auto c = C[std::size_t(band) * ng + g];
                norm += c.x * c.x + c.y * c.y;
            }
            v.orbital_charge[spin] += occ * norm;
        }
        auto rho = spin ? rb[zero] : ra[zero];
        v.rho_charge[spin] = omega * rho.x;
        v.finite &=
            isfinite(rho.x) && isfinite(rho.y) && isfinite(v.f_count[spin])
            && isfinite(v.orbital_charge[spin]) && isfinite(v.rho_charge[spin]);
    }
    *out = v;
}
} // namespace
void classify_frozen_spin_domain(FrozenSpinDomainInput input,
                                 FrozenSpinDomainStatus* output,
                                 cudaStream_t stream) {
    if (input.count <= 0 || !input.rho[0] || !input.rho[1] || !output)
        throw std::invalid_argument("spin XC domain input/status shape");
    if (input.gga)
        for (int s = 0; s < 2; ++s)
            for (int d = 0; d < 3; ++d)
                if (!input.gradient[s][d])
                    throw std::invalid_argument(
                        "spin XC domain gradient shape");
    spin_domain_kernel<<<1, 256, 0, stream>>>(input, output);
    ATV_CUDA_CHECK(cudaGetLastError());
}
void certify_frozen_derivatives(
    const FrozenOperatorVirial* terms, const double* force,
    const cufftDoubleComplex* reference, const cufftDoubleComplex* current,
    const cufftDoubleComplex* orbital, int ng0, int atoms, bool want_force,
    bool want_stress, FrozenDerivativeCertificate* out, double force_scale,
    double hartree_ev, double volume, cudaStream_t stream) {
    if (ng0 < 0 || atoms < 0 || !out
        || (ng0 && (!reference || !current || !orbital))
        || (want_force && atoms && !force) || (want_stress && !terms))
        throw std::invalid_argument("certificate shape/input");
    certificate_kernel<<<1, 256, 0, stream>>>(
        terms, force, reference, current, orbital, ng0, atoms, want_force,
        want_stress, out, force_scale, hartree_ev, volume);
    ATV_CUDA_CHECK(cudaGetLastError());
}
struct FrozenFunctionalWorkspace::Impl {
    const cpu::Frame& frame;
    host::PWProblem& prob;
    cudaStream_t stream;
    int bands, ng, ng0, natoms;
    int packed_bands = 0, zero_index = -1;
    std::optional<FixedSpinOccupations> fixed_spin;
    cpu::xc::XCType functional;
    electronic::Smearing filling;
    double width_ha;
    bool need_force, need_stress;
    std::size_t bytes = 0;
    bool used = false;
    // Cold host sources and all readback destinations outlive release()'s drain.
    std::array<std::vector<double>, 3> G, q;
    std::vector<double> R, slope, local_value, local_slope, positions,
        occupation, ionic_force;
    std::vector<cufftDoubleComplex> structure;
    std::vector<dev::FrozenProjectorColumn> columns;
    std::unique_ptr<host::EwaldSummation> ionic;
    FrozenFunctionalOutcome value;
    FrozenOperatorVirial ionic_term{}, host_total{};
    FrozenDerivativeCertificate host_certificate{};
    FrozenDerivativeCertificate* device_certificate = nullptr;
    std::vector<double> force_values, final_norms;
    std::array<std::vector<double>, 2> spin_final_norms;
    FrozenSpinDomainStatus host_domain{};
    FrozenSpinDomainStatus* device_domain = nullptr;
    bool unsupported_domain = false;
    FixedSpinChargePacket host_spin_charge{};
    FixedSpinChargePacket* device_spin_charge = nullptr;
    std::vector<void*> memory;
    std::unique_ptr<dev::FFTPlan3D> base, quad;
    cublasHandle_t blas = nullptr;
    dev::FrozenReciprocalView Gview{}, qview{};
    dev::FrozenLocalDeviceView local{};
    dev::FrozenNonlocalDeviceGeometry nonlocal{};
    dev::FrozenNonlocalDeviceScratch nl_scratch{};
    int *dh = nullptr, *dk = nullptr, *dl = nullptr;
    cufftDoubleComplex *dC = nullptr, *rho_orb = nullptr, *rho_ref = nullptr,
                       *half = nullptr, *qC = nullptr, *residual_work = nullptr;
    double *rho_acc = nullptr, *df = nullptr, *forces = nullptr,
           *d_ionic_force = nullptr, *device_norms = nullptr;
    dev::FrozenOperatorVirial* terms = nullptr;
    std::array<cufftDoubleComplex*, 2> rho_ref_spin{}, rho_orb_spin{};
    FrozenSpinWorkspaceView spin_xc{};
    FrozenLsdaWorkspaceView xc{};
    FrozenPbeWorkspaceView gga{};
    template <class T>
    T* allocate(std::size_t n) {
        T* p = nullptr;
        if (!n)
            return p;
        ATV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        try {
            memory.push_back(p);
        } catch (...) {
            cudaFree(p);
            throw;
        }
        bytes += n * sizeof(T);
        return p;
    }
    template <class T>
    T* upload(const std::vector<T>& v) {
        T* p = allocate<T>(v.size());
        if (p)
            ATV_CUDA_CHECK(cudaMemcpyAsync(p, v.data(), v.size() * sizeof(T),
                                           cudaMemcpyHostToDevice, stream));
        return p;
    }
    void release() noexcept {
        if (stream)
            cudaStreamSynchronize(stream);
        quad.reset();
        base.reset();
        if (blas)
            cublasDestroy(blas);
        blas = nullptr;
        for (auto* p : memory)
            cudaFree(p);
        memory.clear();
    }
    Impl(const cpu::Frame& f, host::PWProblem& p, int nb, cpu::xc::XCType type,
         electronic::Smearing kind, double width,
         std::optional<FixedSpinOccupations> spin, cudaStream_t s,
         bool want_force, bool want_stress)
        : frame(f), prob(p), stream(s), bands(nb), fixed_spin(spin),
          ng(p.gv_k.empty() ? 0 : p.gv_k[0].count()), ng0(p.gv0.count()),
          natoms(f.nlocal()), functional(type), filling(kind), width_ha(width),
          need_force(want_force), need_stress(want_stress) {
        try {
            const bool filling_valid =
                fixed_spin
                    ? (filling == electronic::Smearing::none
                       && fixed_spin->alpha > 0 && fixed_spin->beta > 0
                       && fixed_spin->alpha <= bands
                       && fixed_spin->beta <= bands
                       && static_cast<long long>(fixed_spin->alpha)
                                  + fixed_spin->beta
                              == p.n_elec)
                    : (filling == electronic::Smearing::none
                           ? p.n_elec == 2LL * bands
                           : (filling == electronic::Smearing::fermi_dirac
                              && p.n_elec > 0 && p.n_elec <= 2LL * bands
                              && std::isfinite(width_ha) && width_ha > 0));
            require((functional == cpu::xc::XCType::LSDA
                     || cpu::xc::is_pbe_family(functional))
                        && stream && bands > 0 && bands <= ng
                        && p.kg.kpoints.size() == 1 && p.gv_k.size() == 1
                        && ng > 0 && ng0 > 0 && filling_valid,
                    "frozen functional admission");
            require(!fixed_spin || bands <= std::numeric_limits<int>::max() / 2,
                    "packed spin band-count overflow");
            packed_bands = fixed_spin ? 2 * bands : bands;
            require(std::size_t(packed_bands)
                        <= std::numeric_limits<std::size_t>::max()
                               / std::size_t(ng) / sizeof(cufftDoubleComplex),
                    "packed coefficient extent overflow");
            if (fixed_spin && need_force)
                require(1LL * ng * packed_bands
                            <= std::numeric_limits<int>::max(),
                        "packed force index overflow");
            value.spin_channels = fixed_spin ? 2 : 1;
            if (fixed_spin) {
                for (int channel = 0; channel < 2; ++channel)
                    spin_final_norms[channel].resize(2 * bands);
                for (int i = 0; i < ng0; ++i)
                    if (p.gv0.h[i] == 0 && p.gv0.k[i] == 0 && p.gv0.l[i] == 0) {
                        require(zero_index < 0, "duplicate density zero mode");
                        zero_index = i;
                    }
                require(zero_index >= 0, "missing density zero mode");
            }
            value.forces_requested = need_force;
            value.stress_requested = need_stress;
            final_norms.resize(2 * bands);
            if (need_force)
                force_values.resize(3 * natoms);
            occupation.assign(
                packed_bands,
                filling == electronic::Smearing::none && !fixed_spin ? 2. : 0.);
            if (fixed_spin) {
                for (int b = 0; b < fixed_spin->alpha; ++b)
                    occupation[b] = 1;
                for (int b = 0; b < fixed_spin->beta; ++b)
                    occupation[bands + b] = 1;
            }
            std::vector<Vec3> pos_ang;
            for (int i = 0; i < natoms; ++i) {
                pos_ang.push_back({f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                                   f.span(kAtom_z)[i]});
                for (int a = 0; a < 3; ++a)
                    positions.push_back(p.ws.atom_pos_bohr[i][a]);
            }
            ionic = std::make_unique<host::EwaldSummation>(
                pos_ang, std::vector<AccReal>(natoms, p.potential->z_val),
                f.box);
            ionic_term.energy_ha = ionic->total_energy();
            for (int v = 0; v < 6; ++v) {
                ionic_term.virial_ha[3 * aa[v] + bb[v]] = ionic->virial_ha()[v];
                ionic_term.virial_ha[3 * bb[v] + aa[v]] = ionic->virial_ha()[v];
            }
            for (const auto& F : ionic->forces_bohr())
                for (int a = 0; a < 3; ++a)
                    ionic_force.push_back(F[a]);
            for (int i = 0; i < ng0; ++i) {
                for (int a = 0; a < 3; ++a)
                    G[a].push_back(p.gv0.g[i][a]);
                const auto jet = host::nc_vloc_fourier_jet(
                    *p.potential, p.gv0.g[i].norm(), p.vol_bohr);
                local_value.push_back(jet.value);
                if (need_stress)
                    local_slope.push_back(jet.slope);
                std::complex<double> S{};
                for (const auto& r : p.ws.atom_pos_bohr)
                    S += std::polar(1., -p.gv0.g[i].dot(r));
                if (need_stress)
                    structure.push_back({S.real(), S.imag()});
            }
            for (int i = 0; i < ng; ++i)
                for (int a = 0; a < 3; ++a)
                    q[a].push_back((p.gv_k[0].g[i] + p.gv_k[0].kvec)[a]);
            for (int j = 0; j < p.potential->n_proj(); ++j)
                for (int i = 0; i < ng; ++i) {
                    const auto jet = host::nc_projector_radial_jet(
                        *p.potential, j,
                        (p.gv_k[0].g[i] + p.gv_k[0].kvec).norm());
                    R.push_back(jet.value);
                    if (need_stress)
                        slope.push_back(jet.slope_over_q);
                }
            for (int atom = 0; atom < natoms; ++atom)
                for (int l = 0; l <= 2; ++l)
                    for (int m = 0; m < 2 * l + 1; ++m) {
                        int first = columns.size(), size = 0;
                        for (const auto& pr : p.potential->kb)
                            if (pr.l == l)
                                ++size;
                        for (int j = 0; j < p.potential->n_proj(); ++j)
                            if (p.potential->kb[j].l == l)
                                columns.push_back({atom, l, m, j, first, size});
                    }
            base = std::make_unique<dev::FFTPlan3D>(p.ws.fft_nx, p.ws.fft_ny,
                                                    p.ws.fft_nz, stream);
            if (need_stress)
                quad = std::make_unique<dev::FFTPlan3D>(
                    host::xc_quadrature_dimension(p.ws.fft_nx,
                                                  cpu::xc::is_gga(functional)),
                    host::xc_quadrature_dimension(p.ws.fft_ny,
                                                  cpu::xc::is_gga(functional)),
                    host::xc_quadrature_dimension(p.ws.fft_nz,
                                                  cpu::xc::is_gga(functional)),
                    stream);
            blas_check(cublasCreate(&blas));
            blas_check(cublasSetStream(blas, stream));
            blas_check(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
            Gview = {upload(G[0]), upload(G[1]), upload(G[2]), ng0};
            qview = {upload(q[0]), upload(q[1]), upload(q[2]), ng};
            local = {upload(local_value), upload(local_slope),
                     upload(structure)};
            nonlocal = {qview,
                        upload(R),
                        upload(slope),
                        upload(p.potential->d_ij),
                        upload(positions),
                        upload(columns),
                        p.potential->n_proj(),
                        int(columns.size()),
                        4 * M_PI / std::sqrt(p.vol_bohr)};
            const auto bc = std::size_t(columns.size()) * packed_bands;
            if (fixed_spin && need_force)
                require(columns.size()
                            <= std::size_t(std::numeric_limits<int>::max())
                                   / packed_bands,
                        "packed projector force index overflow");
            nl_scratch = {
                allocate<cufftDoubleComplex>(std::size_t(ng) * columns.size()),
                allocate<cufftDoubleComplex>(bc),
                allocate<cufftDoubleComplex>(bc),
                allocate<cufftDoubleComplex>(bc)};
            dh = upload(p.gv0.h);
            dk = upload(p.gv0.k);
            dl = upload(p.gv0.l);
            df = filling == electronic::Smearing::none
                     ? upload(occupation)
                     : allocate<double>(bands);
            if (need_force)
                d_ionic_force = upload(ionic_force);
            dC = allocate<cufftDoubleComplex>(std::size_t(ng) * packed_bands);
            if (need_force)
                qC = allocate<cufftDoubleComplex>(std::size_t(ng)
                                                  * packed_bands);
            residual_work =
                allocate<cufftDoubleComplex>(std::size_t(ng) * bands);
            device_norms = allocate<double>(2 * bands);
            rho_orb = allocate<cufftDoubleComplex>(ng0);
            rho_ref = allocate<cufftDoubleComplex>(ng0);
            if (fixed_spin) {
                for (int channel = 0; channel < 2; ++channel) {
                    rho_ref_spin[channel] = allocate<cufftDoubleComplex>(ng0);
                    rho_orb_spin[channel] = allocate<cufftDoubleComplex>(ng0);
                }
                device_spin_charge = allocate<FixedSpinChargePacket>(1);
            }
            half = allocate<cufftDoubleComplex>(base->nc());
            rho_acc = allocate<double>(base->nr());
            if (need_force)
                forces = allocate<double>(3 * natoms);
            if (need_stress) {
                terms = allocate<dev::FrozenOperatorVirial>(7);
                ATV_CUDA_CHECK(cudaMemcpyAsync(
                    terms + 5, &ionic_term, sizeof(dev::FrozenOperatorVirial),
                    cudaMemcpyHostToDevice, stream));
                if (fixed_spin) {
                    for (int channel = 0; channel < 2; ++channel) {
                        spin_xc.scaled_density[channel] =
                            allocate<cufftDoubleComplex>(ng0);
                        spin_xc.rho[channel] = allocate<double>(quad->nr());
                        spin_xc.v[channel] = allocate<double>(quad->nr());
                    }
                    spin_xc.half = allocate<cufftDoubleComplex>(quad->nc());
                    spin_xc.epsilon = allocate<double>(quad->nr());
                    spin_xc.energy = allocate<double>(1);
                    spin_xc.energy_partials = allocate<double>(256);
                    spin_xc.virial_partials = allocate<double>(9 * 256);
                    spin_xc.output = terms + 4;
                    if (cpu::xc::is_gga(functional)) {
                        spin_xc.derivative = allocate<cufftDoubleComplex>(ng0);
                        for (int channel = 0; channel < 2; ++channel)
                            for (int axis = 0; axis < 3; ++axis) {
                                spin_xc.raw_gradient[channel][axis] =
                                    allocate<double>(quad->nr());
                                spin_xc.flux[channel][axis] =
                                    allocate<double>(quad->nr());
                            }
                    }
                } else {
                    xc = {allocate<cufftDoubleComplex>(ng0),
                          allocate<cufftDoubleComplex>(quad->nc()),
                          allocate<double>(quad->nr()),
                          allocate<double>(quad->nr()),
                          allocate<double>(quad->nr()),
                          allocate<double>(1),
                          allocate<double>(256),
                          allocate<double>(cpu::xc::is_gga(functional) ? 9 * 256
                                                                       : 256),
                          terms + 4};
                    if (cpu::xc::is_gga(functional)) {
                        gga = {xc.scaled_density,
                               allocate<cufftDoubleComplex>(ng0),
                               xc.half,
                               xc.rho,
                               xc.v,
                               xc.epsilon,
                               {},
                               {},
                               xc.energy,
                               xc.energy_partials,
                               xc.virial_partials,
                               xc.output};
                        for (int d = 0; d < 3; ++d) {
                            gga.raw_gradient[d] = allocate<double>(quad->nr());
                            gga.flux[d] = allocate<double>(quad->nr());
                        }
                    }
                }
            }
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
            device_certificate = allocate<FrozenDerivativeCertificate>(1);
            if (fixed_spin && need_stress)
                device_domain = allocate<FrozenSpinDomainStatus>(1);
            value.explicit_device_bytes = bytes;
        } catch (...) {
            release();
            throw;
        }
    }
    ~Impl() { release(); }
    FrozenFinishResult finish(DeviceDensity& den, DeviceBandSolver& solver,
                              const cpu::pw::PWResult& reference,
                              std::vector<std::array<double, 3>>* force_out,
                              std::array<double, 6>* stress_out,
                              cuda::Frame* target) {
        require(!used, "derivative workspace already consumed");
        const double physical_energy_ev =
            reference.energy * cpu::pw::qe_units::rydberg_to_ev;
        require(std::isfinite(physical_energy_ev)
                    && std::isfinite(reference.energy_ha),
                "nonfinite physical reference energy");
        const double conversion = units::convert(
            1., units::Force::HartreePerBohr, units::Force::eVPerAngstrom);
        const double volume = frame.box.volume();
        require(std::isfinite(conversion) && std::isfinite(volume)
                    && volume > 0,
                "invalid publication units/volume");
        used = true;
        require(solver.stream() == stream && den.stream() == stream,
                "derivative stream mismatch");
        ATV_CUDA_CHECK(cudaMemcpyAsync(rho_ref, den.device_rho_G(),
                                       ng0 * sizeof(cufftDoubleComplex),
                                       cudaMemcpyDeviceToDevice, stream));
        auto density_from_current = [&](cufftDoubleComplex* target,
                                        std::span<const double> f,
                                        bool restricted_fixed) {
            ATV_CUDA_CHECK(cudaMemsetAsync(
                rho_acc, 0, base->nr() * sizeof(double), stream));
            solver.bind_grids(rho_acc, den.effective_potential());
            try {
                if (restricted_fixed)
                    solver.accumulate_density(1., prob.vol_bohr, bands, 2.);
                else
                    solver.accumulate_final_density(1., prob.vol_bohr, f);
            } catch (...) {
                solver.bind_grids(den.density_accumulator(),
                                  den.effective_potential());
                throw;
            }
            solver.bind_grids(den.density_accumulator(),
                              den.effective_potential());
            base->r2c(rho_acc, half);
            gather_halfcomplex_to_g(half, dh, dk, dl, ng0, target, base->nx(),
                                    base->ny(), base->nz(), 1. / base->nr(),
                                    stream);
        };
        std::optional<SpinDensityDeviceView> spin_view;
        require(den.spin_polarized() == bool(fixed_spin),
                "density spin shape does not match frozen state");
        if (fixed_spin) {
            spin_view = den.spin_density_device_view();
            require(spin_view->count == ng0 && spin_view->stream == stream,
                    "spin density view shape/stream mismatch");
            for (int channel = 0; channel < 2; ++channel)
                ATV_CUDA_CHECK(cudaMemcpyAsync(
                    rho_ref_spin[channel], spin_view->rho[channel],
                    ng0 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice,
                    stream));
            den.hartree();
            den.exchange_correlation_spin();
            for (int channel = 0; channel < 2; ++channel) {
                den.assemble_veff_spin(channel);
                solver.bind_kpoint(0);
                value.spin_auxiliary[channel] = solver.solve_final_bands(
                    prob.ws,
                    {ng, bands, dC + std::size_t(channel) * ng * bands,
                     residual_work, device_norms, spin_final_norms[channel]},
                    20);
                require(value.spin_auxiliary[channel].complete,
                        "independent spin auxiliary eigenstate incomplete");
                density_from_current(
                    rho_orb_spin[channel],
                    std::span<const double>(occupation)
                        .subspan(std::size_t(channel) * bands, bands),
                    false);
            }
            value.auxiliary = {
                value.spin_auxiliary[0].complete
                    && value.spin_auxiliary[1].complete,
                std::max(value.spin_auxiliary[0].max_residual_ha,
                         value.spin_auxiliary[1].max_residual_ha),
                std::max(value.spin_auxiliary[0].max_normalization_error,
                         value.spin_auxiliary[1].max_normalization_error)};
            value.auxiliary_electron_count = prob.n_elec;
            sum_spin_density<<<(ng0 + block - 1) / block, block, 0, stream>>>(
                rho_orb_spin[0], rho_orb_spin[1], rho_orb, ng0);
            ATV_CUDA_CHECK(cudaGetLastError());
        } else {
            den.hartree();
            den.exchange_correlation();
            den.assemble_veff();
            solver.bind_kpoint(0);
            value.auxiliary = solver.solve_final_bands(
                prob.ws,
                {ng, bands, dC, residual_work, device_norms, final_norms}, 20);
            require(value.auxiliary.complete,
                    "auxiliary final eigenstate residual incomplete");
            value.auxiliary_electron_count = prob.n_elec;
            if (filling == electronic::Smearing::fermi_dirac) {
                const std::vector<AccReal> spectrum(prob.ws.eigenvalues.begin(),
                                                    prob.ws.eigenvalues.end());
                require(spectrum.size() == static_cast<std::size_t>(bands),
                        "final spectrum shape");
                std::vector<AccReal> weights(bands, 1.);
                value.auxiliary_mu_ha = electronic::find_fermi_level(
                    spectrum, weights, AccReal(prob.n_elec), width_ha, filling,
                    1, 2);
                const auto fill = electronic::fill_occupations(
                    spectrum, weights, value.auxiliary_mu_ha, width_ha, filling,
                    1, 2);
                require(fill.occ.size() == static_cast<std::size_t>(bands)
                            && std::isfinite(fill.n_occ)
                            && std::abs(fill.n_occ - prob.n_elec) < 1e-8
                            && std::isfinite(fill.entropy),
                        "final filling electron count/entropy");
                for (auto f : fill.occ)
                    require(std::isfinite(f) && f >= 0 && f <= 2,
                            "invalid final Fermi occupation");
                occupation.assign(fill.occ.begin(), fill.occ.end());
                value.auxiliary_entropy = fill.entropy;
                value.auxiliary_electron_count = fill.n_occ;
                ATV_CUDA_CHECK(cudaMemcpyAsync(df, occupation.data(),
                                               bands * sizeof(double),
                                               cudaMemcpyHostToDevice, stream));
            }

            density_from_current(rho_orb, occupation,
                                 filling == electronic::Smearing::none);
        }
        const FrozenBandDeviceView state{dC, df, packed_bands, 1};
        project_frozen_nonlocal(nonlocal, state, nl_scratch, blas, stream);
        if (need_force) {
            ATV_CUDA_CHECK(cudaMemsetAsync(
                forces, 0, 3 * natoms * sizeof(double), stream));
            accumulate_local_forces_device(
                rho_orb, Gview.x, Gview.y, Gview.z, local.value,
                nonlocal.positions, ng0, natoms, prob.vol_bohr, forces, stream);
            accumulate_projected_nonlocal_forces(nonlocal, state, nl_scratch,
                                                 qC, blas, forces, stream);
            add_ionic<<<(3 * natoms + block - 1) / block, block, 0, stream>>>(
                forces, d_ionic_force, natoms);
            ATV_CUDA_CHECK(cudaGetLastError());
        }
        if (need_stress) {
            ATV_CUDA_CHECK(cudaMemsetAsync(
                terms, 0, 5 * sizeof(FrozenOperatorVirial), stream));
            kinetic_frozen_virial(qview, state, terms, stream);
            hartree_frozen_virial(Gview, rho_orb, prob.vol_bohr, terms + 1,
                                  stream);
            local_potential_frozen_virial(Gview, rho_orb, prob.vol_bohr, local,
                                          terms + 2, stream);
            add_projected_nonlocal_virial(nonlocal, state, nl_scratch, blas,
                                          terms + 3, stream);
            if (fixed_spin)
                evaluate_spin_frozen_virial(functional, *quad, dh, dk, dl,
                                            Gview.x, Gview.y, Gview.z, ng0,
                                            rho_orb_spin[0], rho_orb_spin[1], 1,
                                            prob.vol_bohr, spin_xc, stream);
            else if (cpu::xc::is_gga(functional))
                evaluate_pbe_frozen_virial(
                    functional, *quad, dh, dk, dl, Gview.x, Gview.y, Gview.z,
                    ng0, rho_orb, 1, prob.vol_bohr, gga, stream);
            else
                evaluate_lsda_frozen_virial(*quad, dh, dk, dl, ng0, rho_orb, 1,
                                            prob.vol_bohr, xc, stream);
            sum_terms<<<1, 32, 0, stream>>>(terms);
            ATV_CUDA_CHECK(cudaGetLastError());
        }
        certify_frozen_derivatives(terms, forces, rho_ref, den.device_rho_G(),
                                   rho_orb, ng0, natoms, need_force,
                                   need_stress, device_certificate, conversion,
                                   units::kHartreeInEv, volume, stream);
        if (fixed_spin) {
            certify_spin_reference<<<(ng0 + block - 1) / block, block, 0,
                                     stream>>>(
                rho_ref_spin[0], rho_ref_spin[1], spin_view->rho[0],
                spin_view->rho[1], ng0, device_certificate);
            ATV_CUDA_CHECK(cudaGetLastError());
            fixed_spin_charge_packet<<<1, 1, 0, stream>>>(
                dC, df, ng, bands, *fixed_spin, rho_orb_spin[0],
                rho_orb_spin[1], zero_index, prob.vol_bohr, device_spin_charge);
            ATV_CUDA_CHECK(cudaGetLastError());
            ATV_CUDA_CHECK(cudaMemcpyAsync(
                &host_spin_charge, device_spin_charge, sizeof(host_spin_charge),
                cudaMemcpyDeviceToHost, stream));
            if (need_stress) {
                FrozenSpinDomainInput domain{
                    {spin_xc.rho[0], spin_xc.rho[1]},
                    {},
                    quad->nr(),
                    cpu::xc::is_pbe_family(functional)};
                for (int channel = 0; channel < 2; ++channel)
                    for (int d = 0; d < 3; ++d)
                        domain.gradient[channel][d] =
                            spin_xc.raw_gradient[channel][d];
                classify_frozen_spin_domain(domain, device_domain, stream);
                ATV_CUDA_CHECK(cudaMemcpyAsync(&host_domain, device_domain,
                                               sizeof(host_domain),
                                               cudaMemcpyDeviceToHost, stream));
            }
        }
        ATV_CUDA_CHECK(cudaMemcpyAsync(&host_certificate, device_certificate,
                                       sizeof(host_certificate),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        require(host_certificate.finite && host_certificate.reference_unchanged
                    && std::isfinite(host_certificate.density_mismatch_squared),
                "accepted density/reference certificate failed");
        if (fixed_spin) {
            require(host_spin_charge.finite && host_spin_charge.prefix_exact,
                    "fixed spin occupation/charge certificate failed");
            const int expected[2] = {fixed_spin->alpha, fixed_spin->beta};
            for (int channel = 0; channel < 2; ++channel) {
                require(std::abs(host_spin_charge.f_count[channel]
                                 - expected[channel])
                                < 1e-8
                            && std::abs(host_spin_charge.orbital_charge[channel]
                                        - expected[channel])
                                   < 1e-8
                            && std::abs(host_spin_charge.rho_charge[channel]
                                        - expected[channel])
                                   < 1e-8,
                        "fixed spin electron counts do not match physical "
                        "occupations");
                value.fixed_device_f_count[channel] =
                    host_spin_charge.f_count[channel];
                value.spin_orbital_electrons[channel] =
                    host_spin_charge.orbital_charge[channel];
                value.spin_rho_G0_electrons[channel] =
                    host_spin_charge.rho_charge[channel];
            }
            value.fixed_spin_prefix_exact = true;
        }
        if (fixed_spin && need_stress) {
            value.spin_domain = host_domain;
            value.spin_domain_evaluated = true;
            require(host_domain.finite, "nonfinite orbital spin XC domain");
            if (host_domain.violations) {
                unsupported_domain = true;
                return FrozenFinishResult::unsupported_domain;
            }
        }
        require(host_certificate.finite
                    && host_certificate.max_antisymmetric_ha <= 1e-9,
                "derivative finite/antisymmetry/reference certificate failed");
        value.certificate = host_certificate;
        std::vector<std::array<double, 3>> forces_ev(force_out ? natoms : 0);
        if (force_out)
            ATV_CUDA_CHECK(cudaMemcpyAsync(force_values.data(), forces,
                                           3 * natoms * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
        if (stress_out && !target)
            ATV_CUDA_CHECK(cudaMemcpyAsync(&host_total, terms + 6,
                                           sizeof(host_total),
                                           cudaMemcpyDeviceToHost, stream));
        if (force_out || (stress_out && !target))
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (force_out)
            for (int i = 0; i < natoms; ++i)
                for (int a = 0; a < 3; ++a)
                    forces_ev[i][a] = force_values[3 * i + a] * conversion;
        std::array<double, 6> sigma{};
        if (stress_out && !target)
            for (int v = 0; v < 6; ++v)
                sigma[v] = .5
                           * (host_total.virial_ha[3 * aa[v] + bb[v]]
                              + host_total.virial_ha[3 * bb[v] + aa[v]])
                           * units::kHartreeInEv / frame.box.volume();
        value.computed = true;
        if (target) {
            publish_candidate<<<(natoms + block - 1) / block, block, 0,
                                stream>>>(
                *target, forces, terms, natoms, need_force, need_stress,
                conversion, physical_energy_ev, units::kHartreeInEv);
            ATV_CUDA_CHECK(cudaGetLastError());
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        if (force_out)
            *force_out = std::move(forces_ev);
        if (stress_out && !target)
            *stress_out = sigma;
        value.published = true;
        return FrozenFinishResult::complete;
    }
    void commit_fallback(DeviceDensity& den, const cpu::pw::PWResult& reference,
                         const std::vector<std::array<double, 3>>& force_ev,
                         const std::array<double, 6>& sigma,
                         cuda::Frame* target) {
        require(used && unsupported_domain && !value.computed
                    && !value.published,
                "fallback requires finite unsupported uncommitted candidate");
        require(den.stream() == stream, "fallback borrowed stream mismatch");
        const double energy_ev =
            reference.energy * cpu::pw::qe_units::rydberg_to_ev;
        const double conversion = units::kHartreePerBohrInEvPerAngstrom;
        const double volume = frame.box.volume();
        require(std::isfinite(energy_ev) && std::isfinite(reference.energy_ha)
                    && std::isfinite(volume) && volume > 0
                    && std::isfinite(conversion),
                "nonfinite fallback physical reference/units");
        require(!need_force
                    || force_ev.size() == static_cast<std::size_t>(natoms),
                "fallback force candidate shape");
        if (need_force)
            for (int i = 0; i < natoms; ++i)
                for (int a = 0; a < 3; ++a) {
                    require(std::isfinite(force_ev[i][a]),
                            "nonfinite fallback force");
                    force_values[3 * i + a] = force_ev[i][a] / conversion;
                    require(std::isfinite(force_values[3 * i + a]),
                            "nonfinite fallback converted force");
                }
        host_total = {};
        if (need_stress)
            for (int lane = 0; lane < 6; ++lane) {
                const double extensive = sigma[lane] * volume;
                const double w = extensive / units::kHartreeInEv;
                require(std::isfinite(sigma[lane]) && std::isfinite(extensive)
                            && std::isfinite(w),
                        "nonfinite fallback stress/virial conversion");
                host_total.virial_ha[3 * aa[lane] + bb[lane]] = w;
                host_total.virial_ha[3 * bb[lane] + aa[lane]] = w;
            }
        // Upload sources and certificate destination are owner members, alive
        // through the normal fence and release()'s exceptional drain.
        if (need_force)
            ATV_CUDA_CHECK(cudaMemcpyAsync(forces, force_values.data(),
                                           3 * natoms * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));
        if (need_stress)
            ATV_CUDA_CHECK(cudaMemcpyAsync(terms + 6, &host_total,
                                           sizeof(host_total),
                                           cudaMemcpyHostToDevice, stream));
        certify_frozen_derivatives(terms, forces, rho_ref, den.device_rho_G(),
                                   rho_orb, ng0, natoms, need_force,
                                   need_stress, device_certificate, conversion,
                                   units::kHartreeInEv, volume, stream);
        const auto current = den.spin_density_device_view();
        require(current.count == ng0 && current.stream == stream,
                "fallback accepted spin view shape/stream mismatch");
        certify_spin_reference<<<(ng0 + block - 1) / block, block, 0, stream>>>(
            rho_ref_spin[0], rho_ref_spin[1], current.rho[0], current.rho[1],
            ng0, device_certificate);
        ATV_CUDA_CHECK(cudaGetLastError());
        ATV_CUDA_CHECK(cudaMemcpyAsync(&host_certificate, device_certificate,
                                       sizeof(host_certificate),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        // The symmetric nine lanes above are reconstructed from legacy six
        // sigma, not qualification of a native nine-component derivative.
        require(host_certificate.finite && host_certificate.reference_unchanged
                    && std::isfinite(host_certificate.density_mismatch_squared),
                "fallback finite/reference certificate failed");
        if (target) {
            publish_candidate<<<(natoms + block - 1) / block, block, 0,
                                stream>>>(*target, forces, terms, natoms,
                                          need_force, need_stress, conversion,
                                          energy_ev, units::kHartreeInEv);
            ATV_CUDA_CHECK(cudaGetLastError());
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        value.certificate = host_certificate;
        value.computed = true;
        value.published = true;
        value.legacy_fallback = true;
    }
};
FrozenFunctionalWorkspace::FrozenFunctionalWorkspace(
    const cpu::Frame& f, host::PWProblem& p, int b, cpu::xc::XCType type,
    electronic::Smearing kind, double width,
    std::optional<FixedSpinOccupations> spin, cudaStream_t stream, bool force,
    bool stress)
    : impl_(std::make_unique<Impl>(f, p, b, type, kind, width, spin, stream,
                                   force, stress)) {}
FrozenFunctionalWorkspace::~FrozenFunctionalWorkspace() = default;
FrozenFinishResult FrozenFunctionalWorkspace::finish(
    DeviceDensity& den, DeviceBandSolver& solver, const cpu::pw::PWResult& r,
    std::vector<std::array<double, 3>>* f, std::array<double, 6>* w,
    cuda::Frame* target) {
    return impl_->finish(den, solver, r, f, w, target);
}
void FrozenFunctionalWorkspace::commit_fallback(
    DeviceDensity& den, const cpu::pw::PWResult& result,
    const std::vector<std::array<double, 3>>& force,
    const std::array<double, 6>& sigma, cuda::Frame* target) {
    impl_->commit_fallback(den, result, force, sigma, target);
}
FrozenCandidateView FrozenFunctionalWorkspace::candidate() const {
    require(impl_->value.computed && !impl_->value.legacy_fallback,
            "native candidate is not certified");
    FrozenCandidateView view{
        &impl_->prob,   {impl_->dC, impl_->df, impl_->packed_bands, 1},
        impl_->rho_ref, impl_->rho_orb,
        impl_->terms,   impl_->forces,
        impl_->value};
    view.reference_spin_density = {impl_->rho_ref_spin[0],
                                   impl_->rho_ref_spin[1]};
    view.orbital_spin_density = {impl_->rho_orb_spin[0],
                                 impl_->rho_orb_spin[1]};
    view.fixed_spin = impl_->fixed_spin;
    view.live_bands = impl_->bands;
    return view;
}
} // namespace ATV::cuda::pw::internal
