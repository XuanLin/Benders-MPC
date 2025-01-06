cmake_minimum_required(VERSION 3.14)
project(Benders-MPC VERSION 1.0
    DESCRIPTION "Model Predictive Control using Generalized Benders Decomposition"
    LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# Set the path to the Gurobi installation
set(GUROBI_ROOT "/home/x/Desktop/gurobi1200/linux64")
set(GUROBI_INCLUDE_DIR ${GUROBI_ROOT}/include)
set(GUROBI_LIBRARY_DIR ${GUROBI_ROOT}/lib)

# Find the Gurobi libraries
find_library(GUROBI_LIB
    NAMES gurobi120
    PATHS ${GUROBI_LIBRARY_DIR}
    REQUIRED
)
find_library(GUROBICPP_LIB
    NAMES gurobi_c++
    PATHS ${GUROBI_LIBRARY_DIR}
    REQUIRED
)

# Define library targets
add_library(benders_mpc STATIC
    # Optimization - GBD
    src/optimization/gbd/base_solver.cpp
    src/optimization/gbd/base_master_solver.cpp
    src/optimization/gbd/base_sub_solver.cpp
    src/optimization/gbd/greedy_master_solver.cpp
    
    # Problems - Cart Pole
    src/problems/cart_pole/cart_pole_solver.cpp
)

pybind11_add_module(cart_pole_cpp
    src/problems/cart_pole/bindings/cart_pole_bindings.cpp
)

# Copy the compiled module to scripts/cart_pole/
add_custom_command(TARGET cart_pole_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/scripts/cart_pole/
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cart_pole_cpp> ${CMAKE_SOURCE_DIR}/scripts/cart_pole/
)

# Define include directories for the library
target_include_directories(benders_mpc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${GUROBI_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Include directories for Python bindings
target_include_directories(cart_pole_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GUROBI_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

# Link dependencies
target_link_libraries(benders_mpc
    PUBLIC
        Eigen3::Eigen
        ${GUROBICPP_LIB}
        ${GUROBI_LIB}
        m
)

# Link Python binding library with main library
target_link_libraries(cart_pole_cpp
    PRIVATE
        benders_mpc
        ${GUROBICPP_LIB}
        ${GUROBI_LIB}
)

# Set RPATH for Linux build
set_target_properties(benders_mpc PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${GUROBI_LIBRARY_DIR}"
)

# Compiler warnings
if(MSVC)
    target_compile_options(benders_mpc PRIVATE /W4)
else()
    target_compile_options(benders_mpc PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation rules
include(GNUInstallDirs)
install(TARGETS benders_mpc
    EXPORT benders_mpc-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Export targets
install(EXPORT benders_mpc-targets
    FILE benders_mpc-targets.cmake
    NAMESPACE benders_mpc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/benders_mpc
)

