# Standing performance harnesses. Deliberately NOT registered with ctest —
# tests/ is unit-only and "if it is registered, it runs" (see CLAUDE.md).
# Each binary prints walls; a perf claim is the number the harness printed on a
# named node, recorded in the matching README.
if(ATV_BUILD_CUDA)
    add_subdirectory(cuda)
endif()
