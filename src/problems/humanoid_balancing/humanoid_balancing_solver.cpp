// humanoid_balancing_solver.cpp
#include "problems/humanoid_balancing/humanoid_balancing_solver.hpp"
#include <iostream>

namespace optimization {

void HumanoidBalancingGBDSolver::getSolution(std::map<std::string, double>& solution) const {
    // Basic solution components
    solution["tau_ankle"] = best_controls_[0][0];      // First control: ankle torque
    solution["f_right"] = best_controls_[0][1];        // Second control: right wall force
    solution["f_left"] = best_controls_[0][2];         // Third control: left wall force
    solution["cost"] = best_cost_;
    solution["num_iter"] = iteration_count_;
    solution["num_opt_cut"] = master_problem_->getOptimalityCutCount();
    solution["num_feas_cut"] = master_problem_->getFeasibilityCutCount();

    // Analyze contact strategy over horizon
    int right_contact_steps = 0;
    int left_contact_steps = 0;
    int no_contact_steps = 0;
    
    for (int i_n = 0; i_n < params_.N; i_n++) {
        int delta_R = best_binaries_[i_n][0];  // Right wall contact
        int delta_L = best_binaries_[i_n][1];  // Left wall contact
        
        if (delta_R == 1 && delta_L == 0) {
            right_contact_steps++;
        } else if (delta_R == 0 && delta_L == 1) {
            left_contact_steps++;
        } else if (delta_R == 0 && delta_L == 0) {
            no_contact_steps++;
        }
        // Note: delta_R == 1 && delta_L == 1 (simultaneous contact) should be 
        // prevented by the MLD constraints but we don't count it separately
    }

    solution["right_contact_steps"] = right_contact_steps;
    solution["left_contact_steps"] = left_contact_steps;
    solution["no_contact_steps"] = no_contact_steps;

    // Check if any contact is used in the first step (immediate action)
    bool using_contact = (best_binaries_[0][0] == 1) || (best_binaries_[0][1] == 1);
    solution["using_contact"] = using_contact ? 1.0 : 0.0;

    // Print contact sequence for debugging
    if (params_.verbose) {
        std::cout << "Contact sequence [Right, Left]:" << std::endl;
        for (int i_n = 0; i_n < params_.N; i_n++) {
            std::cout << "[" << best_binaries_[i_n][0] << ", " 
                      << best_binaries_[i_n][1] << "] ";
            if ((i_n + 1) % 10 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
        
        std::cout << "Contact summary: " 
                  << "Right=" << right_contact_steps 
                  << ", Left=" << left_contact_steps 
                  << ", None=" << no_contact_steps 
                  << std::endl;
    }

    // Compute ground reaction force for the first step (for analysis)
    // From paper: f_f = (m²g·l_com²/I_com)·θ + (m·l_com/I_com)·τ_a 
    //                  + (1 - m·l_com·l_arm/I_com)·f_R 
    //                  + (m·l_com·l_arm/I_com - 1)·f_L
    double theta_0 = best_states_[0][0];
    double tau_a_0 = best_controls_[0][0];
    double f_R_0 = best_controls_[0][1];
    double f_L_0 = best_controls_[0][2];
    
    double coef_theta = (params_.m * params_.m * params_.g * params_.h_com * params_.h_com) / params_.Icom;
    double coef_tau = (params_.m * params_.h_com) / params_.Icom;
    double coef_fR = 1.0 - (params_.m * params_.h_com * params_.h_arm) / params_.Icom;
    double coef_fL = (params_.m * params_.h_com * params_.h_arm) / params_.Icom - 1.0;
    
    double f_ground = coef_theta * theta_0 + coef_tau * tau_a_0 + coef_fR * f_R_0 + coef_fL * f_L_0;
    solution["f_ground"] = f_ground;
    
    // Check if friction constraint is satisfied (|f_f| <= mu*m*g)
    double friction_limit = params_.mu * params_.m * params_.g;
    solution["friction_margin"] = friction_limit - std::abs(f_ground);
}

} // namespace optimization

