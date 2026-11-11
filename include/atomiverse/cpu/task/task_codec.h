/// @file cpu/task/task_codec.h
/// @brief Byte codec for TaskSpec and TaskResult.
///
/// The payload after the 4-byte `ATVT` magic is selected by TaskKind. MD and
/// Minimize encode different entry types and range-check phase slots against
/// their own typed enums. There is one byte layout; no compatibility
/// decoder or translation path exists.
#pragma once

#include <cstddef>
#include <vector>

#include "atomiverse/cpu/task/task_spec.h"

namespace ATV::cpu::task {

std::vector<std::byte> pack(const TaskSpec& spec);
TaskSpec unpack(const std::byte* data, std::size_t size);

/// Control results have their own `ATVR` magic.
std::vector<std::byte> pack_result(const TaskResult& result);
TaskResult unpack_result(const std::byte* data, std::size_t size);

} // namespace ATV::cpu::task
