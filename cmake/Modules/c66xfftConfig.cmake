INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_C66XFFT c66xfft)

FIND_PATH(
    C66XFFT_INCLUDE_DIRS
    NAMES c66xfft/api.h
    HINTS $ENV{C66XFFT_DIR}/include
        ${PC_C66XFFT_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    C66XFFT_LIBRARIES
    NAMES gnuradio-c66xfft
    HINTS $ENV{C66XFFT_DIR}/lib
        ${PC_C66XFFT_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(C66XFFT DEFAULT_MSG C66XFFT_LIBRARIES C66XFFT_INCLUDE_DIRS)
MARK_AS_ADVANCED(C66XFFT_LIBRARIES C66XFFT_INCLUDE_DIRS)

