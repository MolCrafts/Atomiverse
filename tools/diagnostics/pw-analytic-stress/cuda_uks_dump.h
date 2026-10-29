#pragma once
#include "cuda_uks_detail.h"

#include <fstream>
#include <type_traits>
namespace private_dump {
template <class T>
void put(std::ostream& s, const T& x) {
    static_assert(std::is_trivially_copyable_v<T>);
    s.write(reinterpret_cast<const char*>(&x), sizeof(x));
    require(bool(s), "write same-state diagnostic dump");
}
template <class T>
void get(std::istream& s, T& x) {
    static_assert(std::is_trivially_copyable_v<T>);
    s.read(reinterpret_cast<char*>(&x), sizeof(x));
    require(bool(s), "read same-state diagnostic dump");
}
template <class T>
void putv(std::ostream& s, const std::vector<T>& v) {
    put(s, std::uint64_t(v.size()));
    for (const auto& x : v)
        put(s, x);
}
template <class T>
void getv(std::istream& s, std::vector<T>& v) {
    std::uint64_t n;
    get(s, n);
    require(n < 10000000, "bounded same-state diagnostic vector");
    v.resize(n);
    for (auto& x : v)
        get(s, x);
}
inline void put3(std::ostream& s, const ATV::Vec3& v) {
    for (int a = 0; a < 3; ++a)
        put(s, v[a]);
}
inline void get3(std::istream& s, ATV::Vec3& v) {
    for (int a = 0; a < 3; ++a)
        get(s, v[a]);
}
inline void putg(std::ostream& s, const hp::GVectorList& g) {
    put(s, g.nx);
    put(s, g.ny);
    put(s, g.nz);
    put3(s, g.kvec);
    putv(s, g.h);
    putv(s, g.k);
    putv(s, g.l);
    putv(s, g.ekin);
    for (const auto& v : g.g)
        put3(s, v);
}
inline void getg(std::istream& s, hp::GVectorList& g) {
    get(s, g.nx);
    get(s, g.ny);
    get(s, g.nz);
    get3(s, g.kvec);
    getv(s, g.h);
    getv(s, g.k);
    getv(s, g.l);
    getv(s, g.ekin);
    require(g.h.size() == g.k.size() && g.h.size() == g.l.size()
                && g.h.size() == g.ekin.size(),
            "dump G shapes");
    g.g.resize(g.h.size());
    for (auto& v : g.g)
        get3(s, v);
}
inline void write(const std::string& path) {
    const auto& d = diagnostic;
    require(d.want_dump, "requested diagnostic array dump");
    std::ofstream s(path, std::ios::binary);
    put(s, std::uint64_t(0x554b535354415431));
    put(s, d.volume);
    putg(s, d.wave);
    putg(s, d.density);
    put(s, std::uint64_t(d.positions.size()));
    for (const auto& v : d.positions)
        put3(s, v);
    putv(s, d.coefficients);
    putv(s, d.occupations);
    putv(s, d.rho_ref);
    putv(s, d.rho_orb);
    for (const auto& r : d.spin_ref)
        putv(s, r);
    for (const auto& r : d.spin_orb)
        putv(s, r);
    put(s, d.terms);
    putv(s, d.forces);
    put(s, d.outcome.forces_requested);
    put(s, d.outcome.stress_requested);
    const auto& p = *d.potential;
    put(s, p.atomic_number);
    put(s, p.z_val);
    put(s, p.l_max);
    putv(s, p.r);
    putv(s, p.rab);
    putv(s, p.v_loc);
    putv(s, p.rho_atom);
    putv(s, p.d_ij);
    put(s, std::uint64_t(p.kb.size()));
    for (const auto& k : p.kb) {
        put(s, k.l);
        putv(s, k.beta);
        put(s, std::uint64_t(k.radial_points));
    }
    put(s, std::uint64_t(p.chi.size()));
    for (const auto& c : p.chi)
        putv(s, c);
    putv(s, p.lchi);
    putv(s, p.oc);
    put(s, p.nwfc);
    std::printf("SAME_STATE_DUMP path=%s bytes=%lld diagnostic_only=1\n",
                path.c_str(), static_cast<long long>(s.tellp()));
}
inline void read(const std::string& path) {
    auto& d = diagnostic;
    std::ifstream s(path, std::ios::binary);
    std::uint64_t magic;
    get(s, magic);
    require(magic == 0x554b535354415431, "same-state dump format");
    get(s, d.volume);
    getg(s, d.wave);
    getg(s, d.density);
    std::uint64_t n;
    get(s, n);
    require(n == 3, "Si3 dump actual atoms");
    d.positions.resize(n);
    for (auto& v : d.positions)
        get3(s, v);
    getv(s, d.coefficients);
    getv(s, d.occupations);
    getv(s, d.rho_ref);
    getv(s, d.rho_orb);
    for (auto& r : d.spin_ref)
        getv(s, r);
    for (auto& r : d.spin_orb)
        getv(s, r);
    get(s, d.terms);
    getv(s, d.forces);
    get(s, d.outcome.forces_requested);
    get(s, d.outcome.stress_requested);
    auto p = std::make_shared<hp::NcPotentialData>();
    get(s, p->atomic_number);
    get(s, p->z_val);
    get(s, p->l_max);
    getv(s, p->r);
    getv(s, p->rab);
    getv(s, p->v_loc);
    getv(s, p->rho_atom);
    getv(s, p->d_ij);
    get(s, n);
    require(n < 100, "dump potential projector count");
    p->kb.resize(n);
    for (auto& k : p->kb) {
        get(s, k.l);
        getv(s, k.beta);
        get(s, n);
        k.radial_points = n;
    }
    get(s, n);
    require(n < 100, "dump potential radial state count");
    p->chi.resize(n);
    for (auto& c : p->chi)
        getv(s, c);
    getv(s, p->lchi);
    getv(s, p->oc);
    get(s, p->nwfc);
    p->validate();
    d.potential = p;
    require(d.coefficients.size() == 14 * std::size_t(d.wave.count())
                && d.occupations.size() == 14,
            "actual packed dump shape");
    std::printf(
        "READ_SAME_STATE_DUMP path=%s GPU_calls=0 SCF=0 band_solves=0\n",
        path.c_str());
}
} // namespace private_dump
