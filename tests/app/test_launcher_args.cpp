/// @file tests/app/test_launcher_args.cpp
/// @brief Unit: `parse_cli_args` — `-in` vs `.py` vs `-h` (no I/O).
#include "cli_args.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using ATV::app::parse_cli_args;

namespace {

class Argv {
  public:
    explicit Argv(std::vector<std::string> args) : storage_(std::move(args)) {
        for (auto& s : storage_)
            ptrs_.push_back(s.data());
        ptrs_.push_back(nullptr);
    }
    int argc() const { return static_cast<int>(storage_.size()); }
    char** argv() { return ptrs_.data(); }

  private:
    std::vector<std::string> storage_;
    std::vector<char*> ptrs_;
};

TEST(CliArgs, BareInvocationIsEmpty) {
    Argv a({"atomiverse"});
    const auto c = parse_cli_args(a.argc(), a.argv());
    EXPECT_TRUE(c.in_path.empty());
    EXPECT_TRUE(c.py_script.empty());
    EXPECT_FALSE(c.help);
}

TEST(CliArgs, InFlagSetsNativePath) {
    Argv a({"atomiverse", "-in", "in.lj"});
    const auto c = parse_cli_args(a.argc(), a.argv());
    EXPECT_EQ(c.in_path, "in.lj");
    EXPECT_TRUE(c.py_script.empty());
}

TEST(CliArgs, PySuffixSelectsPythonHost) {
    Argv a({"atomiverse", "job.py", "--foo"});
    const auto c = parse_cli_args(a.argc(), a.argv());
    EXPECT_EQ(c.py_script, "job.py");
    EXPECT_TRUE(c.in_path.empty());
    EXPECT_EQ(c.py_argc, 2);
}

TEST(CliArgs, UnknownFlagThrows) {
    Argv a({"atomiverse", "-nope"});
    EXPECT_THROW(parse_cli_args(a.argc(), a.argv()), std::runtime_error);
}

} // namespace
