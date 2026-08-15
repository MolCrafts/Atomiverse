/// @file cpu/task/task_codec.cpp
/// @brief TaskSpec v5 typed-sum codec; no historic decoder.

#include "atomiverse/cpu/task/task_codec.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ATV::cpu::task {
namespace {

constexpr char kTaskMagic[4] = {'A', 'T', 'V', 'T'};
constexpr char kResultMagic[4] = {'A', 'T', 'V', 'R'};

struct Writer {
    std::vector<std::byte>& out;

    template <typename T>
    void raw(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::size_t at = out.size();
        out.resize(at + sizeof(T));
        std::memcpy(out.data() + at, &value, sizeof(T));
    }
    void u8(std::uint8_t v) { raw(v); }
    void u16(std::uint16_t v) { raw(v); }
    void u32(std::uint32_t v) { raw(v); }
    void i32(std::int32_t v) { raw(v); }
    void u64(std::uint64_t v) { raw(v); }
    void i64(std::int64_t v) { raw(v); }
    void f64(double v) { raw(v); }
    void vec(const std::vector<double>& values) {
        u64(values.size());
        for (double value : values)
            f64(value);
    }
    void string(const std::string& value) {
        u64(value.size());
        const std::size_t at = out.size();
        out.resize(at + value.size());
        std::memcpy(out.data() + at, value.data(), value.size());
    }
};

struct Reader {
    const std::byte* data;
    std::size_t size;
    std::size_t at = 0;

    template <typename T>
    T raw() {
        static_assert(std::is_trivially_copyable_v<T>);
        if (sizeof(T) > size - at)
            throw std::runtime_error("task_codec: truncated buffer at byte "
                                     + std::to_string(at));
        T value;
        std::memcpy(&value, data + at, sizeof(T));
        at += sizeof(T);
        return value;
    }
    std::uint8_t u8() { return raw<std::uint8_t>(); }
    std::uint16_t u16() { return raw<std::uint16_t>(); }
    std::uint32_t u32() { return raw<std::uint32_t>(); }
    std::int32_t i32() { return raw<std::int32_t>(); }
    std::uint64_t u64() { return raw<std::uint64_t>(); }
    std::int64_t i64() { return raw<std::int64_t>(); }
    double f64() { return raw<double>(); }
    std::vector<double> vec() {
        const std::uint64_t n = u64();
        if (n > (size - at) / sizeof(double))
            throw std::runtime_error("task_codec: truncated vector payload");
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(n));
        for (std::uint64_t i = 0; i < n; ++i)
            values.push_back(f64());
        return values;
    }
    std::string string() {
        const std::uint64_t n = u64();
        if (n > size - at)
            throw std::runtime_error("task_codec: truncated string payload");
        std::string value(reinterpret_cast<const char*>(data + at),
                          static_cast<std::size_t>(n));
        at += static_cast<std::size_t>(n);
        return value;
    }
    void require_end() const {
        if (at != size)
            throw std::runtime_error("task_codec: trailing "
                                     + std::to_string(size - at)
                                     + " bytes after payload");
    }
};

void write(Writer& w, const PairLJCutParams& p) {
    w.f64(p.epsilon);
    w.f64(p.sigma);
    w.f64(p.cutoff);
}
void write(Writer& w, const NeighborListParams& p) {
    w.f64(p.cutoff);
    w.f64(p.skin);
}
void write(Writer& w, const BondHarmonicParams& p) {
    w.vec(p.K);
    w.vec(p.r0);
}
void write(Writer& w, const AngleHarmonicParams& p) {
    w.vec(p.K);
    w.vec(p.theta0);
}
void write(Writer& w, const DihedralMultiHarmonicParams& p) {
    w.u64(p.Cn.size());
    for (const auto& values : p.Cn)
        w.vec(values);
}
void write(Writer& w, const VelocityVerletParams& p) { w.f64(p.dt); }
void write(Writer& w, const InitVelocityParams& p) {
    w.f64(p.temperature);
    w.u32(p.seed);
}
void write(Writer& w, const NoseHooverParams& p) {
    w.f64(p.T_start);
    w.f64(p.tau_T);
    w.f64(p.dt);
}
void write(Writer& w, const ThermoParams& p) { w.i32(p.freq); }
// The only MD payload with a text field (the numeric `vec` fields above are
// already variable-length): `string` writes a u64 byte-count prefix followed
// by the raw bytes, so the reader knows where the filename ends and the
// cadence begins.
void write(Writer& w, const XyzRecorderParams& p) {
    w.string(p.filename);
    w.i32(p.freq);
}
void write(Writer& w, const FixLangevinParams& p) {
    w.f64(p.T_start);
    w.f64(p.T_stop);
    w.f64(p.damp);
    w.f64(p.dt);
    w.u32(p.seed);
}
void write(Writer& w, const FixCsvrParams& p) {
    w.f64(p.T_start);
    w.f64(p.tau);
    w.f64(p.dt);
    w.u32(p.seed);
}
void write(Writer& w, const NoseHooverNptParams& p) {
    w.f64(p.T_target);
    w.f64(p.tau_T);
    w.f64(p.P_target);
    w.f64(p.tau_P);
    w.f64(p.dt);
    w.f64(p.drag);
    w.i32(p.mtchain);
    w.i32(p.mpchain);
    for (bool axis : p.couple_axis)
        w.u8(axis ? 1 : 0);
    w.f64(p.T_stop);
}
void write(Writer& w, const ParrinelloRahmanNptParams& p) {
    w.f64(p.T_target);
    w.f64(p.P_target);
    w.f64(p.tau_T);
    w.f64(p.tau_P);
    w.f64(p.dt);
}
void write(Writer& w, const BerendsenBarostatParams& p) {
    w.f64(p.P_target);
    w.f64(p.tau_P);
    w.f64(p.dt);
    w.f64(p.bulk_modulus);
}
void write(Writer& w, const PairCoulCutParams& p) { w.f64(p.cutoff); }
void write(Writer& w, const CoulLongParams& p) {
    w.f64(p.alpha);
    w.f64(p.cutoff);
}
void write(Writer& w, const KSpacePmeParams& p) {
    w.f64(p.alpha);
    w.f64(p.grid_spacing);
    w.i32(p.order);
}
void write(Writer& w, const AM1EvaluatorParams& p) {
    w.i32(p.charge);
    w.i32(p.multiplicity);
    w.i32(p.max_iter);
    w.f64(p.density_tol);
    w.f64(p.energy_tol);
}
// The widest MD payload (fifteen fields) and the second one leading with a
// text field. Field order is the struct's declaration order; `bool` crosses
// as a single byte and both enum mirrors as i32, matching `read_pw` below.
void write(Writer& w, const PwEvaluatorParams& p) {
    w.string(p.upf_path);
    w.string(p.gth_name);
    w.f64(p.ecut_ry);
    w.i32(p.k_mesh);
    w.f64(p.tol);
    w.i32(p.max_iter);
    w.f64(p.mix_beta);
    w.i32(p.pulay_depth);
    w.i32(p.xc);
    w.u8(p.spin_polarized ? 1 : 0);
    w.i32(p.multiplicity);
    w.i32(p.smearing);
    w.f64(p.degauss);
    w.i32(p.mp_order);
    w.i32(p.extra_bands);
}
void write(Writer& w, const LBFGSParams& p) {
    w.i32(p.memory);
    w.f64(p.alpha0);
    w.i32(p.max_ls);
}
// The six remaining minimize optimizers. Field order is each mirror's
// declaration order, which is also the member declaration order of the C++
// class the materializer replays — so a reordered struct is a wire break, and
// is caught by the field-exact roundtrip.
void write(Writer& w, const CGParams& p) { w.f64(p.alpha0); }
void write(Writer& w, const FIREParams& p) {
    w.f64(p.dt0);
    w.f64(p.dt_max);
    w.i32(p.n_min);
    w.f64(p.f_inc);
    w.f64(p.f_dec);
    w.f64(p.alpha0);
    w.f64(p.f_alpha);
    w.f64(p.max_step);
}
void write(Writer& w, const IsotropicCellRelaxParams& p) {
    w.f64(p.P_target);
    w.f64(p.alpha0);
    w.f64(p.max_scale);
    w.f64(p.bulk_mod);
}
void write(Writer& w, const AnisotropicCellRelaxParams& p) {
    w.f64(p.P_target);
    w.f64(p.alpha0);
    w.f64(p.max_scale);
    w.f64(p.bulk_mod);
    // One byte per axis, in x/y/z order — the `NoseHooverNptParams` couple
    // mask precedent. A single byte for all three would survive a symmetric
    // payload and silently mangle an asymmetric one.
    w.u8(p.relax_x ? 1 : 0);
    w.u8(p.relax_y ? 1 : 0);
    w.u8(p.relax_z ? 1 : 0);
}
void write(Writer& w, const TriclinicCellRelaxParams& p) {
    w.f64(p.alpha0);
    w.f64(p.max_strain);
    w.f64(p.bulk_mod);
}
void write(Writer& w, const JointIsotropicRelaxParams& p) {
    w.f64(p.P_target);
    w.f64(p.atom_step);
    w.f64(p.alpha0);
    w.f64(p.max_scale);
    w.f64(p.bulk_mod);
}
void write(Writer& w, const ForceConvergedParams& p) {
    w.f64(p.ftol);
    w.f64(p.rms_ftol);
    w.f64(p.max_dtol);
    w.f64(p.rms_dtol);
    w.f64(p.etol);
}
void write(Writer& w, const PressureConvergedParams& p) {
    w.f64(p.P_target);
    w.f64(p.ptol);
}
void write(Writer& w, const StressConvergedParams& p) {
    w.f64(p.sigma_target);
    w.f64(p.stol);
    w.u8(p.check_x ? 1 : 0);
    w.u8(p.check_y ? 1 : 0);
    w.u8(p.check_z ? 1 : 0);
}

template <typename Variant>
void write_variant(Writer& w, const Variant& params) {
    std::visit([&](const auto& p) { write(w, p); }, params);
}

/// @brief Write the `MinimizeTask` stop policy: id then payload.
///
/// The criterion is the LAST thing the Minimize packer lays down, so the id
/// has to precede its fields — a decoder reading a fixed five doubles there
/// ends up short or long, and `require_end()` is what turns that into an error
/// instead of a silently wrong tolerance.
///
/// @param w         Byte sink.
/// @param criterion The typed stop policy of the task being packed.
void write_criterion(Writer& w, const MinimizeCriterionParams& criterion) {
    w.u16(static_cast<std::uint16_t>(criterion_id(criterion)));
    std::visit([&](const auto& c) { write(w, c); }, criterion);
}

ComponentId decode_component_id(std::uint16_t raw) {
    switch (raw) {
    case 0:
        return ComponentId::PairLJCut;
    case 1:
        return ComponentId::NeighborList;
    case 2:
        return ComponentId::BondHarmonic;
    case 3:
        return ComponentId::AngleHarmonic;
    case 4:
        return ComponentId::DihedralMultiHarmonic;
    case 5:
        return ComponentId::VelocityVerlet;
    case 6:
        return ComponentId::InitVelocity;
    case 7:
        return ComponentId::NoseHoover;
    case 8:
        return ComponentId::Thermo;
    case 9:
        return ComponentId::AM1Evaluator;
    case 10:
        return ComponentId::LBFGS;
    case 11:
        return ComponentId::FixLangevin;
    case 12:
        return ComponentId::FixCSVR;
    case 13:
        return ComponentId::NoseHooverNPT;
    case 14:
        return ComponentId::ParrinelloRahmanNPT;
    case 15:
        return ComponentId::BerendsenBarostat;
    case 16:
        return ComponentId::PairCoulCut;
    case 17:
        return ComponentId::CoulLong;
    case 18:
        return ComponentId::KSpacePME;
    case 19:
        return ComponentId::XyzRecorder;
    case 20:
        return ComponentId::PWEvaluator;
    case 21:
        return ComponentId::CG;
    case 22:
        return ComponentId::FIRE;
    case 23:
        return ComponentId::IsotropicCellRelax;
    case 24:
        return ComponentId::AnisotropicCellRelax;
    case 25:
        return ComponentId::TriclinicCellRelax;
    case 26:
        return ComponentId::JointIsotropicRelax;
    default:
        throw std::runtime_error("task_codec: unknown component_id "
                                 + std::to_string(raw));
    }
}

/// @brief Is this component id a Minimize-only one?
///
/// Seven optimizers share one home, so `wrong_task` asks this rather than
/// spelling seven `||` arms at the point where it builds the diagnostic.
///
/// @param id A decoded component id.
/// @return True for the optimizers `MinimizeTask` alone may carry.
bool is_minimize_only(ComponentId id) noexcept {
    switch (id) {
    case ComponentId::LBFGS:
    case ComponentId::CG:
    case ComponentId::FIRE:
    case ComponentId::IsotropicCellRelax:
    case ComponentId::AnisotropicCellRelax:
    case ComponentId::TriclinicCellRelax:
    case ComponentId::JointIsotropicRelax:
        return true;
    default:
        return false;
    }
}

[[noreturn]] void wrong_task(ComponentId id, TaskKind target) {
    const char* source = "MD";
    if (is_minimize_only(id))
        source = "Minimize";
    else if (id == ComponentId::AM1Evaluator)
        // Two legal homes: the whole of a SinglePoint task, and an evaluator
        // entry of an MD task (semiempirical Born-Oppenheimer dynamics).
        source = "MD/SinglePoint";
    else if (id == ComponentId::PairLJCut || id == ComponentId::NeighborList)
        source = "MD/Minimize";
    throw std::runtime_error(std::string("task_codec: component ")
                             + component_name(id) + " belongs to " + source
                             + ", not " + task_kind_name(target));
}

/// @brief Read the five AM1 fields in the order `write` laid them down.
///
/// Declared ahead of both readers because ID 9 decodes into two payloads —
/// the whole of a `SinglePointTask` and an evaluator entry of an `MDTask` —
/// and one reader is what keeps their field lists from drifting apart.
AM1EvaluatorParams read_am1(Reader& r) {
    AM1EvaluatorParams p;
    p.charge = r.i32();
    p.multiplicity = r.i32();
    p.max_iter = r.i32();
    p.density_tol = r.f64();
    p.energy_tol = r.f64();
    return p;
}

/// @brief Read the fifteen plane-wave fields in the order `write` laid them
///        down.
///
/// Statement per field rather than one braced aggregate: the widest payload
/// on the wire is also the one where a silently swapped field is hardest to
/// see, and a named assignment beside its reader makes the byte layout
/// legible at the point it is decoded.
PwEvaluatorParams read_pw(Reader& r) {
    PwEvaluatorParams p;
    p.upf_path = r.string();
    p.gth_name = r.string();
    p.ecut_ry = r.f64();
    p.k_mesh = r.i32();
    p.tol = r.f64();
    p.max_iter = r.i32();
    p.mix_beta = r.f64();
    p.pulay_depth = r.i32();
    p.xc = r.i32();
    p.spin_polarized = r.u8() != 0;
    p.multiplicity = r.i32();
    p.smearing = r.i32();
    p.degauss = r.f64();
    p.mp_order = r.i32();
    p.extra_bands = r.i32();
    return p;
}

MDComponentParams read_md_params(Reader& r, ComponentId id) {
    switch (id) {
    case ComponentId::PairLJCut:
        return PairLJCutParams{r.f64(), r.f64(), r.f64()};
    case ComponentId::NeighborList:
        return NeighborListParams{r.f64(), r.f64()};
    case ComponentId::BondHarmonic:
        return BondHarmonicParams{r.vec(), r.vec()};
    case ComponentId::AngleHarmonic:
        return AngleHarmonicParams{r.vec(), r.vec()};
    case ComponentId::DihedralMultiHarmonic: {
        DihedralMultiHarmonicParams p;
        const std::uint64_t n = r.u64();
        if (n > r.size - r.at)
            throw std::runtime_error(
                "task_codec: implausible dihedral coefficient count");
        p.Cn.reserve(static_cast<std::size_t>(n));
        for (std::uint64_t i = 0; i < n; ++i)
            p.Cn.push_back(r.vec());
        return p;
    }
    case ComponentId::VelocityVerlet:
        return VelocityVerletParams{r.f64()};
    case ComponentId::InitVelocity:
        return InitVelocityParams{r.f64(), r.u32()};
    case ComponentId::NoseHoover:
        return NoseHooverParams{r.f64(), r.f64(), r.f64()};
    case ComponentId::Thermo:
        return ThermoParams{r.i32()};
    case ComponentId::XyzRecorder:
        // Braced init: the field reads are sequenced left to right, so the
        // length-prefixed filename is consumed before the cadence.
        return XyzRecorderParams{r.string(), r.i32()};
    case ComponentId::FixLangevin:
        return FixLangevinParams{r.f64(), r.f64(), r.f64(), r.f64(), r.u32()};
    case ComponentId::FixCSVR:
        return FixCsvrParams{r.f64(), r.f64(), r.f64(), r.u32()};
    case ComponentId::NoseHooverNPT:
        return NoseHooverNptParams{
            r.f64(), r.f64(), r.f64(),
            r.f64(), r.f64(), r.f64(),
            r.i32(), r.i32(), {r.u8() != 0, r.u8() != 0, r.u8() != 0},
            r.f64()};
    case ComponentId::ParrinelloRahmanNPT:
        return ParrinelloRahmanNptParams{r.f64(), r.f64(), r.f64(), r.f64(),
                                         r.f64()};
    case ComponentId::BerendsenBarostat:
        return BerendsenBarostatParams{r.f64(), r.f64(), r.f64(), r.f64()};
    case ComponentId::PairCoulCut:
        return PairCoulCutParams{r.f64()};
    case ComponentId::CoulLong:
        return CoulLongParams{r.f64(), r.f64()};
    case ComponentId::KSpacePME:
        return KSpacePmeParams{r.f64(), r.f64(), r.i32()};
    case ComponentId::AM1Evaluator:
        return read_am1(r);
    case ComponentId::PWEvaluator:
        return read_pw(r);
    // Every Minimize-only optimizer shares one refusal arm: naming one in an
    // MD payload is a wrong-home error, not an unknown component.
    case ComponentId::LBFGS:
    case ComponentId::CG:
    case ComponentId::FIRE:
    case ComponentId::IsotropicCellRelax:
    case ComponentId::AnisotropicCellRelax:
    case ComponentId::TriclinicCellRelax:
    case ComponentId::JointIsotropicRelax:
        wrong_task(id, TaskKind::MD);
    }
    throw std::runtime_error("task_codec: invalid MD component id");
}

MinimizeComponentParams read_minimize_params(Reader& r, ComponentId id) {
    switch (id) {
    case ComponentId::PairLJCut:
        return PairLJCutParams{r.f64(), r.f64(), r.f64()};
    case ComponentId::NeighborList:
        return NeighborListParams{r.f64(), r.f64()};
    case ComponentId::LBFGS:
        return LBFGSParams{r.i32(), r.f64(), r.i32()};
    case ComponentId::CG:
        return CGParams{r.f64()};
    case ComponentId::FIRE:
        return FIREParams{r.f64(), r.f64(), r.i32(), r.f64(),
                          r.f64(), r.f64(), r.f64(), r.f64()};
    case ComponentId::IsotropicCellRelax:
        return IsotropicCellRelaxParams{r.f64(), r.f64(), r.f64(), r.f64()};
    case ComponentId::AnisotropicCellRelax:
        return AnisotropicCellRelaxParams{r.f64(),    r.f64(),     r.f64(),
                                          r.f64(),    r.u8() != 0, r.u8() != 0,
                                          r.u8() != 0};
    case ComponentId::TriclinicCellRelax:
        return TriclinicCellRelaxParams{r.f64(), r.f64(), r.f64()};
    case ComponentId::JointIsotropicRelax:
        return JointIsotropicRelaxParams{r.f64(), r.f64(), r.f64(), r.f64(),
                                         r.f64()};
    default:
        wrong_task(id, TaskKind::Minimize);
    }
}

/// @brief Read the `MinimizeTask` stop policy written by `write_criterion`.
///
/// Dispatches on the leading `u16`: `PressureConverged` and the first two
/// `ForceConverged` tolerances are both "two doubles", so an id that was
/// written but ignored would decode a pressure test back as a force test with
/// entirely plausible numbers.
///
/// @param r Byte source, positioned at the criterion id.
/// @return The decoded criterion alternative.
/// @throws std::runtime_error On an unknown criterion id.
MinimizeCriterionParams read_criterion(Reader& r) {
    const std::uint16_t raw = r.u16();
    switch (raw) {
    case 0:
        return ForceConvergedParams{r.f64(), r.f64(), r.f64(), r.f64(),
                                    r.f64()};
    case 1:
        return PressureConvergedParams{r.f64(), r.f64()};
    case 2:
        return StressConvergedParams{r.f64(), r.f64(), r.u8() != 0, r.u8() != 0,
                                     r.u8() != 0};
    default:
        throw std::runtime_error("task_codec: unknown minimize criterion_id "
                                 + std::to_string(raw));
    }
}

template <typename Phase>
Phase read_phase(Reader& r, const char* domain) {
    const std::uint32_t slot = r.u32();
    if (slot >= static_cast<std::uint32_t>(ATV::slot_count<Phase>()))
        throw std::runtime_error("task_codec: phase slot "
                                 + std::to_string(slot) + " out of range for "
                                 + domain);
    return static_cast<Phase>(slot);
}

void write_header(Writer& w, const char (&magic)[4]) {
    w.raw(magic);
    w.u32(TaskSpec::kWireVersion);
}

void read_header(Reader& r, const char (&magic)[4]) {
    char actual[4];
    for (char& byte : actual)
        byte = static_cast<char>(r.u8());
    if (std::memcmp(actual, magic, sizeof(actual)) != 0)
        throw std::runtime_error("task_codec: bad magic");
    const std::uint32_t version = r.u32();
    if (version != TaskSpec::kWireVersion)
        throw std::runtime_error("task_codec: wire version "
                                 + std::to_string(version)
                                 + " does not match exact version "
                                 + std::to_string(TaskSpec::kWireVersion));
}

} // namespace

std::vector<std::byte> pack(const TaskSpec& spec) {
    std::vector<std::byte> out;
    Writer w{out};
    write_header(w, kTaskMagic);
    w.u8(static_cast<std::uint8_t>(spec.kind()));

    if (spec.kind() == TaskKind::MD) {
        const auto& task = spec.md();
        w.u32(static_cast<std::uint32_t>(task.entries.size()));
        for (const auto& entry : task.entries) {
            w.u16(static_cast<std::uint16_t>(component_id(entry.params)));
            w.u8(entry.binding ? 1 : 0);
            if (entry.binding) {
                w.u32(static_cast<std::uint32_t>(
                    ATV::slot_of(entry.binding->phase)));
                w.i32(entry.binding->order);
            }
            write_variant(w, entry.params);
        }
        w.i64(task.n_steps);
    } else if (spec.kind() == TaskKind::SinglePoint) {
        w.u16(static_cast<std::uint16_t>(ComponentId::AM1Evaluator));
        write(w, spec.single_point().evaluator);
    } else {
        const auto& task = spec.minimize();
        w.u32(static_cast<std::uint32_t>(task.entries.size()));
        for (const auto& entry : task.entries) {
            w.u16(static_cast<std::uint16_t>(component_id(entry.params)));
            w.u8(entry.binding ? 1 : 0);
            if (entry.binding) {
                w.u32(static_cast<std::uint32_t>(
                    ATV::slot_of(entry.binding->phase)));
                w.i32(entry.binding->order);
            }
            write_variant(w, entry.params);
        }
        w.i64(task.max_iter);
        write_criterion(w, task.convergence);
    }
    return out;
}

TaskSpec unpack(const std::byte* data, std::size_t size) {
    Reader r{data, size};
    read_header(r, kTaskMagic);
    const std::uint8_t raw_kind = r.u8();
    if (raw_kind > static_cast<std::uint8_t>(TaskKind::Minimize))
        throw std::runtime_error("task_codec: unknown task_kind "
                                 + std::to_string(raw_kind));
    const TaskKind kind = static_cast<TaskKind>(raw_kind);

    if (kind == TaskKind::MD) {
        MDTask task;
        const std::uint32_t count = r.u32();
        task.entries.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const ComponentId id = decode_component_id(r.u16());
            std::optional<MDBinding> binding;
            if (r.u8() != 0)
                binding = MDBinding{read_phase<ATV::md::Phase>(r, "md::Phase"),
                                    r.i32()};
            task.entries.push_back(
                MDComponentEntry{read_md_params(r, id), binding});
        }
        task.n_steps = r.i64();
        r.require_end();
        return TaskSpec(std::move(task));
    }

    if (kind == TaskKind::SinglePoint) {
        const ComponentId id = decode_component_id(r.u16());
        if (id != ComponentId::AM1Evaluator)
            wrong_task(id, TaskKind::SinglePoint);
        SinglePointTask task{read_am1(r)};
        r.require_end();
        return TaskSpec(std::move(task));
    }

    MinimizeTask task;
    const std::uint32_t count = r.u32();
    task.entries.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const ComponentId id = decode_component_id(r.u16());
        std::optional<MinimizeBinding> binding;
        if (r.u8() != 0)
            binding = MinimizeBinding{
                read_phase<ATV::minimizer::Phase>(r, "minimizer::Phase"),
                r.i32()};
        task.entries.push_back(
            MinimizeComponentEntry{read_minimize_params(r, id), binding});
    }
    task.max_iter = r.i64();
    task.convergence = read_criterion(r);
    r.require_end();
    return TaskSpec(std::move(task));
}

std::vector<std::byte> pack_result(const TaskResult& result) {
    std::vector<std::byte> out;
    Writer w{out};
    write_header(w, kResultMagic);
    w.i64(result.iterations);
    w.u8(result.converged ? 1 : 0);
    w.string(result.stop_reason);
    return out;
}

TaskResult unpack_result(const std::byte* data, std::size_t size) {
    Reader r{data, size};
    read_header(r, kResultMagic);
    TaskResult result;
    result.iterations = r.i64();
    const std::uint8_t converged = r.u8();
    if (converged > 1)
        throw std::runtime_error(
            "task_codec: TaskResult converged byte is not boolean");
    result.converged = converged != 0;
    result.stop_reason = r.string();
    r.require_end();
    return result;
}

} // namespace ATV::cpu::task
