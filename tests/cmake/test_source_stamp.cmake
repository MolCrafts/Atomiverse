# cmake/AtvSourceStamp.cmake on the three kinds of tree it is run on.
#
#   cmake -DSTAMP=<AtvSourceStamp.cmake> -DWORK=<scratch dir> -P this
#
# A git checkout stamps its HEAD and whether a tracked file differs from it;
# a copy without .git stamps what .atv-source-commit names; a bare copy
# stamps "unknown"; a malformed .atv-source-commit fails the build.

find_program(GIT git REQUIRED)
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

function(run_git dir)
    execute_process(COMMAND "${GIT}" -C "${dir}" ${ARGN}
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "git ${ARGN}: ${err}")
    endif()
    set(GIT_OUT "${out}" PARENT_SCOPE)
endfunction()

# Runs the stamp on `dir`; STAMP_RC and STAMP_TEXT are its exit code and
# header.
function(stamp dir)
    file(REMOVE "${WORK}/stamp.h")
    execute_process(COMMAND "${CMAKE_COMMAND}" -DSOURCE_DIR=${dir}
                            -DOUTPUT=${WORK}/stamp.h -P "${STAMP}"
                    RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
    set(STAMP_RC "${rc}" PARENT_SCOPE)
    set(text "")
    if(EXISTS "${WORK}/stamp.h")
        file(READ "${WORK}/stamp.h" text)
    endif()
    set(STAMP_TEXT "${text}" PARENT_SCOPE)
endfunction()

function(expect dir commit dirty)
    stamp("${dir}")
    if(NOT STAMP_RC EQUAL 0)
        message(FATAL_ERROR "stamp of ${dir} failed")
    endif()
    if(NOT STAMP_TEXT MATCHES "#define ATV_SOURCE_COMMIT \"${commit}\"\n"
       OR NOT STAMP_TEXT MATCHES "#define ATV_SOURCE_DIRTY ${dirty}\n")
        message(FATAL_ERROR
            "stamp of ${dir}: want commit ${commit}, dirty ${dirty}; got\n"
            "${STAMP_TEXT}")
    endif()
endfunction()

# 1. A git checkout: HEAD, clean; a tracked edit makes it dirty, an
#    untracked file does not.
set(repo "${WORK}/repo")
file(MAKE_DIRECTORY "${repo}")
file(WRITE "${repo}/a.txt" "one\n")
run_git("${repo}" init -q)
run_git("${repo}" add a.txt)
run_git("${repo}" -c user.name=t -c user.email=t@t -c commit.gpgsign=false
        commit -q -m first)
run_git("${repo}" rev-parse HEAD)
set(head "${GIT_OUT}")
expect("${repo}" "${head}" 0)
file(WRITE "${repo}/untracked.txt" "x\n")
expect("${repo}" "${head}" 0)
file(WRITE "${repo}/a.txt" "two\n")
expect("${repo}" "${head}" 1)

# 2. A copy without .git names its revision in .atv-source-commit.
set(copy "${WORK}/copy")
file(MAKE_DIRECTORY "${copy}")
file(WRITE "${copy}/.atv-source-commit" "${head} dirty\n")
expect("${copy}" "${head}" 1)
file(WRITE "${copy}/.atv-source-commit" "${head} clean\n")
expect("${copy}" "${head}" 0)

# 3. A bare copy is "unknown".
set(bare "${WORK}/bare")
file(MAKE_DIRECTORY "${bare}")
expect("${bare}" "unknown" -1)

# 4. A malformed file fails rather than stamping a guess.
foreach(bad "${head}" "abc123 clean" "${head} modified" "${head}0 clean")
    file(WRITE "${copy}/.atv-source-commit" "${bad}\n")
    stamp("${copy}")
    if(STAMP_RC EQUAL 0)
        message(FATAL_ERROR "accepted .atv-source-commit '${bad}'")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK}")
