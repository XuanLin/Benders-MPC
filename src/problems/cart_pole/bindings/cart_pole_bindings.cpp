#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "problems/cart_pole/cart_pole_solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cart_pole_cpp, m) {
    py::class_<optimization::CartPoleParams>(m, "CartPoleParams")
        .def(py::init<>())
        .def_readonly("mc", &optimization::CartPoleParams::mc)
        .def_readonly("mp", &optimization::CartPoleParams::mp)
        .def_readonly("ll", &optimization::CartPoleParams::ll)
        .def_readonly("g", &optimization::CartPoleParams::g)
        .def_readonly("k1", &optimization::CartPoleParams::k1)
        .def_readonly("k2", &optimization::CartPoleParams::k2)
        .def_readonly("d_left", &optimization::CartPoleParams::d_left)
        .def_readonly("d_right", &optimization::CartPoleParams::d_right)
        .def_readonly("d_max", &optimization::CartPoleParams::d_max)
        .def_readonly("u_max", &optimization::CartPoleParams::u_max)
        .def_readonly("lam_max", &optimization::CartPoleParams::lam_max)
        .def_readonly("x_lb", &optimization::CartPoleParams::x_lb)
        .def_readonly("x_ub", &optimization::CartPoleParams::x_ub)
        .def_readonly("N", &optimization::CartPoleParams::N)
        .def_readonly("nx", &optimization::CartPoleParams::nx)
        .def_readonly("nu", &optimization::CartPoleParams::nu)
        .def_readonly("nz", &optimization::CartPoleParams::nz)
        .def_readonly("nc", &optimization::CartPoleParams::nc)
        .def_readonly("dT", &optimization::CartPoleParams::dT)
        .def_readonly("h_theta", &optimization::CartPoleParams::h_theta)
        .def_readonly("max_iterations", &optimization::CartPoleParams::max_iterations)
        .def_readonly("mip_gap", &optimization::CartPoleParams::mip_gap)
        .def_readonly("verbose", &optimization::CartPoleParams::verbose);

    py::class_<optimization::CartPoleGBDSolver>(m, "CartPoleGBDSolver")
        .def(py::init<const optimization::CartPoleParams&>())
        .def("solve", &optimization::CartPoleGBDSolver::solve);
}
