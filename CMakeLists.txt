cmake_minimum_required(VERSION 3.20)
project(polypack VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(POLYPACK_BUILD_TESTS "Build polypack unit tests" ON)
option(POLYPACK_BUILD_EXAMPLES "Build polypack examples" ON)

# --- GMP (exact arbitrary-precision rationals) ------------------------------
find_path(GMP_INCLUDE_DIR NAMES gmpxx.h)
find_library(GMP_LIBRARY NAMES gmp)
find_library(GMPXX_LIBRARY NAMES gmpxx)
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY OR NOT GMPXX_LIBRARY)
  message(FATAL_ERROR
    "GMP with C++ bindings not found. Install it, e.g. `sudo apt install libgmp-dev`.")
endif()

add_library(GMP::gmp UNKNOWN IMPORTED)
set_target_properties(GMP::gmp PROPERTIES
  IMPORTED_LOCATION "${GMP_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}")
add_library(GMP::gmpxx UNKNOWN IMPORTED)
set_target_properties(GMP::gmpxx PROPERTIES
  IMPORTED_LOCATION "${GMPXX_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES GMP::gmp)

# --- Library -----------------------------------------------------------------
add_library(polypack
  src/rational.cpp
  src/geometry.cpp
  src/rotation.cpp
  src/polygon.cpp
  src/svg.cpp
  src/overlap.cpp
  src/bin.cpp
)
add_library(polypack::polypack ALIAS polypack)
target_include_directories(polypack PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(polypack PUBLIC GMP::gmpxx)
target_compile_options(polypack PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow>)

# --- Tests -------------------------------------------------------------------
if(POLYPACK_BUILD_TESTS)
  enable_testing()
  add_executable(polypack_tests tests/test_main.cpp)
  target_link_libraries(polypack_tests PRIVATE polypack)
  add_test(NAME polypack_tests COMMAND polypack_tests)
endif()

# --- Examples ----------------------------------------------------------------
if(POLYPACK_BUILD_EXAMPLES)
  add_executable(polypack_demo examples/demo.cpp)
  target_link_libraries(polypack_demo PRIVATE polypack)
endif()
