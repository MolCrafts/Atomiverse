/// @file core/report.h
/// @brief Structured report emission (aligned tables + machine CSV) routed
/// through the logging module. CUDA-free, header-only.
///
/// Human-readable tables go to `report_logger()` (rank-0 console, raw pattern);
/// machine-readable CSV rows go to `metrics_logger()` (rank-0 metrics file).
/// This keeps all evaluation output "through the log module" while never
/// mixing the aligned table with CSV rows. @see core/log.h
///
/// @code
///   ATV::report::Table t("Timing breakdown (32 ranks)",
///       {{"Section", 10, ATV::report::Align::Left}, {"avg(s)"}, {"%total"}});
///   t.row({"Pair", "0.979", "15.9"});
///   t.emit();                       // -> console, rank 0
///
///   ATV::report::Csv csv("stage,ranks,Pair_s,Neigh_s");
///   csv.row("stage5,32,0.979,1.053"); // -> metrics file, rank 0
/// @endcode
#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "atomiverse/core/log.h"

namespace ATV {
namespace report {

enum class Align { Left, Right };

/// @brief One column spec for an aligned table.
struct Column {
    std::string header;
    int width = 12;
    Align align = Align::Right;
};

/// @brief An aligned, LAMMPS-`timing breakdown`-style text table. Cells are
/// pre-formatted strings (callers format numbers as they like). `emit()` sends
/// each rendered line through `report_logger()` at INFO (rank-0 console only).
class Table {
  public:
    Table(std::string title, std::vector<Column> cols)
        : title_(std::move(title)), cols_(std::move(cols)) {}

    Table& row(std::vector<std::string> cells) {
        rows_.push_back(std::move(cells));
        return *this;
    }

    /// @brief Render the whole table (title + header + separator + rows) to a
    /// single string. Used by tests and by `emit()`.
    std::string to_string() const {
        std::ostringstream o;
        if (!title_.empty())
            o << title_ << '\n';
        o << render_row(headers()) << '\n';
        o << separator() << '\n';
        for (const auto& r : rows_)
            o << render_row(r) << '\n';
        return o.str();
    }

    /// @brief Emit the table line-by-line through the report logger.
    void emit() const {
        auto& lg = report_logger();
        if (!lg.should_log(spdlog::level::info))
            return; // non-rank-0 / disabled: nothing to do
        std::istringstream in(to_string());
        std::string line;
        while (std::getline(in, line))
            lg.info("{}", line);
    }

  private:
    std::vector<std::string> headers() const {
        std::vector<std::string> h;
        h.reserve(cols_.size());
        for (const auto& c : cols_)
            h.push_back(c.header);
        return h;
    }

    std::string cell(std::size_t i, std::string_view s) const {
        const int w = cols_[i].width;
        std::ostringstream o;
        if (cols_[i].align == Align::Left)
            o << std::left << std::setw(w) << s;
        else
            o << std::right << std::setw(w) << s;
        return o.str();
    }

    std::string render_row(const std::vector<std::string>& cells) const {
        std::ostringstream o;
        for (std::size_t i = 0; i < cols_.size(); ++i) {
            if (i)
                o << "  ";
            o << cell(i, i < cells.size() ? cells[i] : std::string{});
        }
        return o.str();
    }

    std::string separator() const {
        std::ostringstream o;
        for (std::size_t i = 0; i < cols_.size(); ++i) {
            if (i)
                o << "  ";
            o << std::string(static_cast<std::size_t>(cols_[i].width), '-');
        }
        return o.str();
    }

    std::string title_;
    std::vector<Column> cols_;
    std::vector<std::vector<std::string>> rows_;
};

/// @brief Machine-readable CSV emitter. The header is written on construction;
/// rows are appended. All output goes to `metrics_logger()` (rank-0 metrics
/// file). When no metrics file is configured the logger is off, so every call
/// is a cheap no-op.
class Csv {
  public:
    explicit Csv(std::string_view header) { line(header); }
    void row(std::string_view r) { line(r); }

  private:
    static void line(std::string_view s) {
        auto& lg = metrics_logger();
        if (lg.should_log(spdlog::level::info))
            lg.info("{}", s);
    }
};

/// @brief Format a double with fixed precision (table cell helper).
inline std::string fmt(double v, int precision = 4) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(precision) << v;
    return o.str();
}

} // namespace report
} // namespace ATV
