/// @file cpu/task/task_codec.cpp
/// @brief TaskSpec v4 typed-sum codec; no historic decoder.

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
void write(Writer& w, const LBFGSParams& p) {
    w.i32(p.memory);
    w.f64(p.alpha0);
    w.i32(p.max_ls);
}
void write(Writer& w, const ForceConvergedParams& p) {
    w.f64(p.ftol);
    w.f64(p.rms_ftol);
    w.f64(p.max_dtol);
    w.f64(p.rms_dtol);
    w.f64(p.etol);
}

template <typename Variant>
void write_variant(Writer& w, const Variant& params) {
    std::visit([&](const auto& p) { write(w, p); }, params);
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
    default:
        throw std::runtime_error("task_codec: unknown component_id "
                                 + std::to_string(raw));
    }
}

[[noreturn]] void wrong_task(ComponentId id, TaskKind target) {
    const char* source = "MD";
    if (id == ComponentId::LBFGS)
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
    case ComponentId::LBFGS:
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
    default:
        wrong_task(id, TaskKind::Minimize);
    }
}

ForceConvergedParams read_convergence(Reader& r) {
    return ForceConvergedParams{r.f64(), r.f64(), r.f64(), r.f64(), r.f64()};
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
        write(w, task.convergence);
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
    task.convergence = read_convergence(r);
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
