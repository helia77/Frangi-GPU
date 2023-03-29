# finds the TIRA library (downloads it if it isn't present)
# set TIRA_ROOT to the directory containing the stim subdirectory (the tira repository)

include(FindPackageHandleStandardArgs)

set(TIRA_ROOT $ENV{TIRA_ROOT})

IF(NOT TIRA_ROOT)
    MESSAGE("ERROR: TIRA_ROOT environment variable must be set!")
ENDIF(NOT TIRA_ROOT)

    FIND_PATH(TIRA_INCLUDE_DIRS DOC "Path to TIRA include directory."
              NAMES tira/image.h
              PATHS ${TIRA_ROOT})

find_package_handle_standard_args(TIRA DEFAULT_MSG TIRA_INCLUDE_DIRS)
