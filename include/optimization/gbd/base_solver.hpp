#pragma once

// C++ Standard Library headers
#include <map>
#include <memory>    
#include <stack>
#include <string>
#include <vector>
#include <numeric>
#include <cassert>

// Third-party headers
#include <Eigen/Core>
#include <gurobi_c++.h>

// Project headers
#include "common/types.hpp"
#include "optimization/util/solver_params.hpp"
#include "optimization/util/gurobi_env.hpp"
#include "optimization/gbd/base_master_solver.hpp"
#include "optimization/gbd/base_sub_solver.hpp"


namespace optimization {

class BaseGBDSolver : protected util::GurobiEnv {

    public:
        BaseGBDSolver(const util::SolverParams& params, std::unique_ptr<BaseMasterSolver> master, std::unique_ptr<BaseSubSolver> sub);
        virtual ~BaseGBDSolver();

        std::map<std::string, double> solve(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& h_theta);
        
    protected:
        void updateInitialConditions(const VectorDyn& x0_new, const VectorDyn& h_theta_new);
        bool optimizeSubproblem(std::vector<std::vector<int>>& z_input, 
                                std::vector<std::vector<double>>& x_sol, std::vector<std::vector<double>>& u_sol, double& f_obj, 
                                std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param, double& const_part){
                                return subproblem_->optimize(z_input, x_sol, u_sol, f_obj, dual_z, dual_param, const_part);}
        virtual std::pair<std::vector<std::vector<int>>, double> solveMasterProblem() = 0;
        virtual void getSolution(std::map<std::string, double>& solution) const = 0;    

        // Dependencies injected into the base class
        util::SolverParams params_;
        std::unique_ptr<BaseMasterSolver> master_problem_;
        std::unique_ptr<BaseSubSolver> subproblem_;

        // State tracking
        int iteration_count_;
        double best_cost_;
        bool problem_solved_;

        // Solution storage
        std::vector<std::vector<double>> best_states_;
        std::vector<std::vector<double>> best_controls_;
        std::vector<std::vector<int>> best_binaries_;

    private:
        // Prevent copying
        BaseGBDSolver(const BaseGBDSolver&) = delete;
        BaseGBDSolver& operator=(const BaseGBDSolver&) = delete;
    };

} // namespace optimization

