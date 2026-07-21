# Prove that ATV_MOLRS_PATH is only a source-location override. An incompatible
# tree must fail during configure with the actual path/version named.
if(NOT DEFINED ATV_SOURCE_DIR OR NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "ATV_SOURCE_DIR and TEST_ROOT are required")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/molrs-cxxapi")
file(WRITE "${TEST_ROOT}/molrs-cxxapi/CXX_API_VERSION" "2\n")

include("${ATV_SOURCE_DIR}/cmake/MolrsContract.cmake")
atv_validate_molrs_source_contract("${TEST_ROOT}")
