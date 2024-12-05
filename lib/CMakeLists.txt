cmake_minimum_required(VERSION 3.24)

project(mattress
    VERSION 1.0.0
    DESCRIPTION "Building the mattress shared library"
    LANGUAGES CXX)

# Importing all of the dependencies with pinned versions (even for transitive dependencies).
include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG v3.1.1
)

FetchContent_Declare(
  tatami_stats
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_stats
  GIT_TAG v1.1.2
)

FetchContent_MakeAvailable(tatami)
FetchContent_MakeAvailable(tatami_stats)

# Defining the targets.
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(mattress
    src/common.cpp
    src/init.cpp
)

set_property(TARGET mattress PROPERTY CXX_STANDARD 17)

target_link_libraries(mattress PRIVATE
    tatami
    tatami_stats
    pybind11::pybind11
)

set_target_properties(mattress PROPERTIES
    OUTPUT_NAME lib_mattress
    PREFIX ""
)
