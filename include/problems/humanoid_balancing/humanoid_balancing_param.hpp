// humanoid_balancing_param.hpp
#pragma once

#include "optimization/util/solver_params.hpp"
#include "common/types.hpp"
#include <math.h>

namespace optimization {

struct HumanoidBalancingParams : public util::SolverParams {

    // Physical parameters (from Table 1 in paper)
    double m;          // Mass [kg]
    double h_com;      // CoM height [m]
    double h_arm;      // Arm contact height [m]
    double l_arm;      // Arm stretch length [m]
    double Icom;       // Centroidal inertia [kg·m²]
    double IO;         // Pivot inertia [kg·m²]
    double mu;         // Friction coefficient [-]
    double tau_max;    // Max ankle torque [Nm]
    double Fmax;       // Max contact force [N]
    double dR;         // Right wall distance [m]
    double dL;         // Left wall distance [m]
    double g;          // Gravity [m/s²]

    // State and control bounds
    std::array<double, 2> x_lb;  // [theta_min, dtheta_min]
    std::array<double, 2> x_ub;  // [theta_max, dtheta_max]
    VectorDyn h_theta;

    HumanoidBalancingParams(
        double mass = 25.0,
        double link_com = 0.4,
        double link_arm = 0.55,
        double arm_length = 0.3,
        double inertia_com = 0.8,
        double friction_coef = 0.6,
        double max_ankle_torque = 7.0, // mass × g × (foot_width/2) = 25 × 9.8 × 0.03 = 7.3 Nm
        double max_force = 500.0,
        double right_wall_dist = 0.40,
        double left_wall_dist = -0.40
    ) 
        : m(mass)
        , h_com(link_com)
        , h_arm(link_arm)
        , l_arm(arm_length)
        , Icom(inertia_com)
        , mu(friction_coef)
        , tau_max(max_ankle_torque)
        , Fmax(max_force)
        , dR(right_wall_dist)
        , dL(left_wall_dist)
    {
        // Initialize base class members
        dT = 0.02;        // 50 Hz discretization
        N = 10;           // Prediction horizon
        nx = 2;           // State dimension: [theta, theta_dot]
        nu = 3;           // Control dimension: [tau_a, f_R, f_L]
        nz = 2;           // Binary dimension: [delta_R, delta_L]
        nc = 12;          // Number of inequality constraints (from H3 in paper)
        dual_len = (N + 1) * nx + N * nc;
        g = 9.81;
        
        // State bounds (reasonable angles for balancing)
        x_lb = {-M_PI, -5*M_PI};     
        x_ub = {M_PI, 5*M_PI};

        // Solver settings
        max_iterations = 100;
        mip_gap = 0.1;
        verbose = true;
        
        // Initialize goal state (upright)
        x_goal = VectorDyn::Zero(nx);

        // Binary solution options: right contact, left contact, no contact
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

        // Cost matrices - penalize angle deviation and velocity, light penalty on control
        Q << 1000.0,    0.0,
                0.0,  100.0; 

        R << 1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;

        Qn << 2000.0,   0.0,
                 0.0, 200.0;  // Terminal cost

        // Dynamics matrices E, F, G from Eq. (33) in paper
        // x[k+1] = E*x[k] + F*u[k] + G*delta[k]
        // Linearized around upright equilibrium (theta = 0)
        
        E << 1.0,  dT,
             (m * g * h_com / Icom) * dT,  1.0;

        F << 0.0,  0.0,  0.0,
             dT / Icom,  -(h_arm * dT) / Icom,  (h_arm * dT) / Icom;

        // Binary variables don't directly affect dynamics (only through constraints)
        G = Eigen::MatrixXd::Zero(nx, nz);

        // Constraint matrices from Eq. (34) and (35) in paper
        // H1*x + H2*u + H3*delta <= h
        
        // Compute coefficients for ground reaction force constraint
        double coef_theta = (m * m * g * h_com * h_com) / Icom;
        double coef_tau = (m * h_com) / Icom;
        double coef_fR = 1.0 - (m * h_com * h_arm) / Icom;
        double coef_fL = (m * h_com * h_arm) / Icom - 1.0;

        // Build H1 (state constraint matrix)
        H1 = (Eigen::MatrixXd(12, 2) << 
            // Ground friction constraints (rows 0-1)
            coef_theta,  0.0,
           -coef_theta,  0.0,
            // Ankle torque limits (rows 2-3) - no state dependence
            0.0,  0.0,
            0.0,  0.0,
            // Contact force bounds (rows 4-7) - no state dependence
            0.0,  0.0,
            0.0,  0.0,
            0.0,  0.0,
            0.0,  0.0,
            // Geometry activation
           -h_arm,  0.0,
            h_arm,  0.0,
            h_arm,  0.0,
           -h_arm,  0.0
        ).finished();

        // Build H2 (control constraint matrix)
        H2 = (Eigen::MatrixXd(12, 3) << 
            // Ground friction constraints (rows 0-1)
            coef_tau,  coef_fR,  coef_fL,
           -coef_tau, -coef_fR, -coef_fL,
            // Ankle torque limits (rows 2-3)
            1.0,  0.0,  0.0,
           -1.0,  0.0,  0.0,
            // Contact force non-negativity (rows 4-5)
            0.0, -1.0,  0.0,
            0.0,  0.0, -1.0,
            // Contact force upper bounds (rows 6-7)
            0.0,  1.0,  0.0,
            0.0,  0.0,  1.0,
            // Geometry activation
            0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,
            0.0,  0.0,  0.0
        ).finished();

        // Build H3 (binary constraint matrix)
        H3 = (Eigen::MatrixXd(12, 2) << 
            // Ground friction constraints (rows 0-1)
            0.0,  0.0,
            0.0,  0.0,
            // Ankle torque limits (rows 2-3)
            0.0,  0.0,
            0.0,  0.0,
            // Contact force non-negativity
            0.0,  0.0,
            0.0,  0.0,
            // Contact force upper bounds
           -Fmax,  0.0,
            0.0, -Fmax,
            // Geometry activation
            (std::abs(dL) + dR),                  0.0,
                            0.0,  (std::abs(dL) + dR),
           -(std::abs(dL) + dR),                  0.0,
                            0.0, -(std::abs(dL) + dR)
        ).finished();

        // Right-hand side vector h
        h_theta.resize(nc, 1);
        h_theta = (Eigen::VectorXd(12) << 
            // Ground friction constraints
            mu * m * g,
            mu * m * g,
            // Ankle torque limits
            tau_max,
            tau_max,
            // Contact force non-negativity
            0.0,
            0.0,
            // Contact force upper bounds
            0.0,
            0.0,
            // Geometry activation
           -(dR-l_arm) + (std::abs(dL) + dR),
            (dL+l_arm) + (std::abs(dL) + dR),
            (dR-l_arm), 
           -(dL+l_arm)
        ).finished();
    }

    const VectorDyn& getParams() const {
        return h_theta;
    }
    
};

} // namespace optimization
