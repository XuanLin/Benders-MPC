// cart_pole_param.hpp
#pragma once

#include "optimization/util/solver_params.hpp"
#include "common/types.hpp"
#include <math.h>

namespace optimization {

struct CartPoleParams : public util::SolverParams {

     // System physical parameters
     double mc;       // Cart mass
     double mp;       // Pendulum mass
     double ll;       // Pendulum length
     double g;        // Gravity
     double k1;       // Left wall stiffness
     double k2;       // Right wall stiffness 
     double d_left;   // Left wall position
     double d_right;  // Right wall position
     double d_max;    // Maximum displacement
     double u_max;    // Maximum actuator force
     double lam_max;  // Maximum contact force
     VectorDyn h_theta;

     // State bounds (x, theta, dx, dtheta)
     std::array<double, 4> x_lb;
     std::array<double, 4> x_ub;

     // Contact force bounds
     std::array<double, 2> lam_lb;
     std::array<double, 2> lam_ub;

     CartPoleParams() 
          : mc(1.0)
          , mp(0.4)
          , ll(0.6)
          , g(9.81)
          , k1(50.0)
          , k2(50.0)
          , d_left(0.40)
          , d_right(0.35)
          , d_max(0.6)
          , u_max(20.0)
          , lam_max(30.0)
     {
          // Initialize base class members
          dT = 0.02;
          N = 10;  // Set appropriate horizon length
          nx = 4;  // State dimension
          nu = 3;  // Control dimension (u, lambda1, lambda2)
          nz = 2;  // Binary variable dimension
          nc = 20; // Constraint dimension
          dual_len = (N + 1) * nx + N * nc;
          
          // Initialize bounds based on parameters
          x_lb = {-d_max, -M_PI/2, -2*d_max/dT, -M_PI/dT};
          x_ub = { d_max,  M_PI/2,  2*d_max/dT,  M_PI/dT};
          lam_lb = {0.0, 0.0};
          lam_ub = {lam_max, lam_max};

          // Solver settings
          max_iterations = 200;
          mip_gap = 0.2;
          verbose = false;
               
          // Initialize goal state
          x_goal = VectorDyn::Zero(nx);

          // Binary solution options
          arr_z = {{1, 0}, {0, 1}, {0, 0}};

          // Initialize matrices with correct sizes
          Q.resize(nx, nx);
          R.resize(nu, nu);
          Qn.resize(nx, nx);
          E.resize(nx, nx);
          F.resize(nx, nu);
          G.resize(nx, nz);
          H1.resize(nc, nx);
          H2.resize(nc, nu);
          H3.resize(nc, nz);

          // Initialize system matrices
          Q << 1.0,  0.0,  0.0,  0.0,
               0.0, 50.0,  0.0,  0.0,
               0.0,  0.0,  1.0,  0.0,
               0.0,  0.0,  0.0, 50.0;

          R << 0.1, 0.0, 0.0,
               0.0, 0.1, 0.0,
               0.0, 0.0, 0.1;

          Qn << 65.37,    5.589,  17.085,   -0.594,
                    5.589, 2643.984,  13.439,    60.27,
               17.085,   13.439,   21.99,    -0.64,
               -0.594,    60.27,   -0.64,    57.275;

          E = Eigen::MatrixXd::Identity(nx, nx) + 
               dT * (Eigen::Matrix4d() << 
                    0.0,               0.0, 1.0, 0.0,
                    0.0,               0.0, 0.0, 1.0,
                    0.0,           g*mp/mc, 0.0, 0.0,
                    0.0, g*(mc+mp)/(ll*mc), 0.0, 0.0
               ).finished();

          F = dT * (Eigen::MatrixXd(4, 3) << 
                    0.0,        0.0,         0.0,
                    0.0,        0.0,         0.0,
                    1/mc,        0.0,         0.0,
                    1/(ll*mc),  1/(ll*mp),  -1/(ll*mp)
               ).finished();

          G = Eigen::MatrixXd::Zero(nx, nz);

          H1 = (Eigen::MatrixXd(20, 4) << 
               0.0,  0.0,  0.0,  0.0,
               0.0,  0.0,  0.0,  0.0,
               -1.0,   ll,  0.0,  0.0,
               1.0,  -ll,  0.0,  0.0,
               1.0,  -ll,  0.0,  0.0,
               -1.0,   ll,  0.0,  0.0,
               1.0,  0.0,  0.0,  0.0,
               -1.0,  0.0,  0.0,  0.0,
               0.0,  1.0,  0.0,  0.0,
               0.0, -1.0,  0.0,  0.0,
               0.0,  0.0,  1.0,  0.0,
               0.0,  0.0, -1.0,  0.0,
               0.0,  0.0,  0.0,  1.0,
               0.0,  0.0,  0.0, -1.0,
               0.0,  0.0,  0.0,  0.0,
               0.0,  0.0,  0.0,  0.0,
               0.0,  0.0,  0.0,  0.0,
               0.0,  0.0,  0.0,  0.0,
               0.0,  0.0,  0.0,  0.0,
               0.0,  0.0,  0.0,  0.0
          ).finished();

          // Initialize H2 (u control constraints)
          H2 = (Eigen::MatrixXd(20, 3) << 
               0.0,     1.0,     0.0,
               0.0,     0.0,     1.0,
               0.0,  1.0/k1,     0.0,
               0.0, -1.0/k1,     0.0,
               0.0,     0.0,  1.0/k2,
               0.0,     0.0, -1.0/k2,
               0.0,     0.0,     0.0,
               0.0,     0.0,     0.0,
               0.0,     0.0,     0.0,
               0.0,     0.0,     0.0,
               0.0,     0.0,     0.0,
               0.0,     0.0,     0.0,
               0.0,     0.0,     0.0,
               0.0,     0.0,     0.0,
               1.0,     0.0,     0.0,
               -1.0,     0.0,     0.0,
               0.0,     1.0,     0.0,
               0.0,    -1.0,     0.0,
               0.0,     0.0,     1.0,
               0.0,     0.0,    -1.0
          ).finished();

          // Initialize H3 (z binary constraints)
          H3 = (Eigen::MatrixXd(20, 2) << 
               -lam_max,           0.0,
               0.0,                -lam_max,
               x_ub[0]-x_lb[0],    0.0,
               0.0,                0.0,
               0.0,                x_ub[0]-x_lb[0],
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0,
               0.0,                0.0
          ).finished();

          h_theta.resize(nc, 1);
          h_theta = (Eigen::VectorXd(20) << 
               0.0,                             // 0
               0.0,                             // 1
               -d_right + x_ub[0] - x_lb[0],    // 2
               d_right,                         // 3
               -d_left + x_ub[0] - x_lb[0],     // 4
               d_left,                          // 5
               d_max,                           // 6  x_max
               d_max,                           // 7  -x_min
               M_PI/2,                          // 8  theta_max
               M_PI/2,                          // 9  -theta_min
               2*d_max/dT,                      // 10 dx_max
               2*d_max/dT,                      // 11 -dx_min
               M_PI/dT,                         // 12 dtheta_max
               M_PI/dT,                         // 13 -dtheta_min
               u_max,                           // 14
               u_max,                           // 15
               lam_max,                         // 16
               0.0,                             // 17
               lam_max,                         // 18
               0.0                              // 19
          ).finished();
    }
};

} // namespace optimization
