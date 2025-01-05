// cart_pole_solver.cpp
#include "problems/cart_pole/cart_pole_solver.hpp"

namespace optimization {

void CartPoleGBDSolver::getSolution(std::map<std::string, double>& solution) const {
    // Basic solution components
    solution["control"] = best_controls_[0][0];
    solution["cost"] = best_cost_;
    solution["num_iter"] = iteration_count_;
    solution["num_opt_cut"] = master_problem_->getOptimalityCutCount();
    solution["num_feas_cut"] = master_problem_->getFeasibilityCutCount();

    // Check if any contact is planned
    bool planned_contact = false;
    for (int i_n = 0; i_n < params_.N; i_n++) {
        for (int i_z = 0; i_z < params_.nz; i_z++) {
            if (best_binaries_[i_n][i_z] == 1) {
                planned_contact = true;
                break;
            }
        }
        if (planned_contact) break;
    }

    // Print binary solution for debugging
    for (int i_n = 0; i_n < params_.N; i_n++) {
        for (int i_z = 0; i_z < params_.nz; i_z++) {
            std::cout << best_binaries_[i_n][i_z] << ' ';
        }
        std::cout << std::endl;
    }

    solution["planned_contact"] = planned_contact;
}

} // namespace optimization
