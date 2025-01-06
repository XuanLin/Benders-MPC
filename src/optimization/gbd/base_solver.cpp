#include "optimization/gbd/base_solver.hpp"

namespace optimization {

BaseGBDSolver::BaseGBDSolver(const util::SolverParams& params, std::unique_ptr<BaseMasterSolver> master, std::unique_ptr<BaseSubSolver> sub)
    : params_(params), master_problem_(std::move(master)), sub_problem_(std::move(sub)) {

    iteration_count_ = 0;
    best_cost_ = std::numeric_limits<double>::infinity();
    problem_solved_ = false;

    best_states_.resize(params_.N + 1, std::vector<double>(params_.nx));
    best_controls_.resize(params_.N, std::vector<double>(params_.nu));
    best_binaries_.resize(params_.N, std::vector<int>(params_.nz));
}

BaseGBDSolver::~BaseGBDSolver() = default;

void BaseGBDSolver::updateInitialConditions(const VectorDyn& x0_new, const VectorDyn& h_theta_new) {
    master_problem_->updateInitialConditions(x0_new, h_theta_new);
    sub_problem_->updateInitialConditions(x0_new, h_theta_new);
}

std::map<std::string, double> BaseGBDSolver::solve(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& h_theta) {

    if (x0.size() != params_.nx) {
        throw std::runtime_error("x0 dimension mismatch");
    }
    if (h_theta.size() != params_.nc) {
        throw std::runtime_error("h_theta dimension mismatch");
    }
    
    std::vector<double> list_f_obj_LB;
    std::vector<double> list_f_obj_UB;
    std::vector<std::string> ls_feas;

    // Current solution storage
    std::vector<std::vector<double>> x_sol(params_.N + 1, std::vector<double>(params_.nx, 0.0));
    std::vector<std::vector<double>> u_sol(params_.N, std::vector<double>(params_.nu, 0.0));
    std::vector<std::vector<int>> z_input(params_.N, std::vector<int>(params_.nz));
    double cost = 0.0;
    double const_part = 0.0;
    std::stack<VectorDyn> dual_z, dual_param;

    // Reset solver state
    iteration_count_ = 0;
    best_cost_ = std::numeric_limits<double>::max();
    problem_solved_ = false;

    // Initialize problems
    updateInitialConditions(x0, h_theta);

    for (int i_loop = 0; i_loop < params_.max_iterations; i_loop++) {
        // Solve master problem
        auto [z_input, obj_value] = solveMasterProblem();

        list_f_obj_LB.push_back(obj_value);

        // Check convergence
        if (i_loop > 0) {
            double current_gap = 0.0;
            if (list_f_obj_LB.back() > list_f_obj_UB.back()) {
                current_gap = 0.0;
            } else {
                current_gap = std::abs(list_f_obj_UB.back() - list_f_obj_LB.back()) / std::abs(list_f_obj_UB.back());
            }
            if (current_gap <= params_.mip_gap) {
                problem_solved_ = true;
                break;
            }
        }

        // Solve subproblem
        bool feas = solveSubProblem(z_input, x_sol, u_sol, cost, dual_z, dual_param, const_part);
        iteration_count_++;

        if (feas) {
            ls_feas.push_back("feas");
            master_problem_->addOptimalityCut(dual_z, dual_param, const_part);

            // Update best solution if needed
            if (cost < best_cost_) {
                best_cost_ = cost;
                best_states_ = x_sol;
                best_controls_ = u_sol;
                best_binaries_ = z_input;
                list_f_obj_UB.push_back(cost);
            } else {
                list_f_obj_UB.push_back(list_f_obj_UB.back());
            }
        } else {
            ls_feas.push_back("infeas");

            master_problem_->addFeasibilityCut(dual_z, dual_param);

            if (i_loop == 0) {
                list_f_obj_UB.push_back(std::numeric_limits<double>::max());
            } else {
                list_f_obj_UB.push_back(list_f_obj_UB.back());
            }
        }

    }

    master_problem_->storeOptimalityCut();
    master_problem_->storeFeasibilityCut();

    std::map<std::string, double> solution;
    getSolution(solution);
    return solution;

}

} // namespace optimization
