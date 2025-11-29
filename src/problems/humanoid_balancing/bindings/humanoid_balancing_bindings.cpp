// humanoid_balancing_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "problems/humanoid_balancing/humanoid_balancing_solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(humanoid_balancing_cpp, m) {
    m.doc() = "Humanoid balancing GBD solver with wall contacts";

    py::class_<optimization::HumanoidBalancingParams>(m, "HumanoidBalancingParams")
        .def(py::init<>())
        // Physical parameters
        .def_readonly("m", &optimization::HumanoidBalancingParams::m)
        .def_readonly("h_com", &optimization::HumanoidBalancingParams::h_com)
        .def_readonly("h_arm", &optimization::HumanoidBalancingParams::h_arm)
        .def_readonly("l_arm", &optimization::HumanoidBalancingParams::l_arm)
        .def_readonly("Icom", &optimization::HumanoidBalancingParams::Icom)
        .def_readonly("mu", &optimization::HumanoidBalancingParams::mu)
        .def_readonly("tau_max", &optimization::HumanoidBalancingParams::tau_max)
        .def_readonly("Fmax", &optimization::HumanoidBalancingParams::Fmax)
        .def_readonly("dR", &optimization::HumanoidBalancingParams::dR)
        .def_readonly("dL", &optimization::HumanoidBalancingParams::dL)
        // Bounds
        .def_readonly("x_lb", &optimization::HumanoidBalancingParams::x_lb)
        .def_readonly("x_ub", &optimization::HumanoidBalancingParams::x_ub)
        .def_readonly("h_theta", &optimization::HumanoidBalancingParams::h_theta)
        // Horizon and dimensions
        .def_readonly("N", &optimization::HumanoidBalancingParams::N)
        .def_readonly("nx", &optimization::HumanoidBalancingParams::nx)
        .def_readonly("nu", &optimization::HumanoidBalancingParams::nu)
        .def_readonly("nz", &optimization::HumanoidBalancingParams::nz)
        .def_readonly("nc", &optimization::HumanoidBalancingParams::nc)
        .def_readonly("dT", &optimization::HumanoidBalancingParams::dT)
        // Solver settings
        .def_readonly("max_iterations", &optimization::HumanoidBalancingParams::max_iterations)
        .def_readonly("mip_gap", &optimization::HumanoidBalancingParams::mip_gap)
        .def_readonly("verbose", &optimization::HumanoidBalancingParams::verbose);

    py::class_<optimization::HumanoidBalancingGBDSolver>(m, "HumanoidBalancingGBDSolver")
        .def(py::init<const optimization::HumanoidBalancingParams&>())
        .def("solve", &optimization::HumanoidBalancingGBDSolver::solve,
             py::arg("x0"),
             py::arg("h_theta"),
             "Solve the humanoid balancing MPC problem\n\n"
             "Parameters:\n"
             "  x0: Initial state [theta, theta_dot]\n"
             "  h_theta: Constraint parameter vector\n\n"
             "Returns:\n"
             "  Dictionary with solution including:\n"
             "    - tau_ankle: Ankle torque control\n"
             "    - f_right: Right wall contact force\n"
             "    - f_left: Left wall contact force\n"
             "    - cost: Optimal cost\n"
             "    - num_iter: Number of GBD iterations\n"
             "    - right_contact_steps: Steps with right wall contact\n"
             "    - left_contact_steps: Steps with left wall contact\n"
             "    - no_contact_steps: Steps with no wall contact\n"
             "    - using_contact: Whether first step uses wall contact\n"
             "    - f_ground: Ground reaction force\n"
             "    - friction_margin: Margin before friction limit");
}

