// cart_pole_solver.hpp
#pragma once

#include <memory>
#include "optimization/gbd/base_solver.hpp"
#include "optimization/gbd/greedy_master_solver.hpp"
#include "optimization/gbd/gurobi_sub_solver.hpp"
#include "problems/cart_pole/cart_pole_param.hpp"

namespace optimization {

class CartPoleGBDSolver : public BaseGBDSolver {
public:   
    CartPoleGBDSolver(const CartPoleParams& params, std::unique_ptr<BaseMasterSolver> master, std::unique_ptr<BaseSubSolver> sub
        ) : BaseGBDSolver(params, std::move(master), std::move(sub)), params_(params) {
        }

    // Default constructor that uses GreedyMaster and GurobiSubSolver
    explicit CartPoleGBDSolver(const CartPoleParams& params = CartPoleParams())
        : CartPoleGBDSolver(params, std::make_unique<GreedyMasterSolver>(params), std::make_unique<GurobiSubSolver>(params)) {
        }

    ~CartPoleGBDSolver() override = default;

protected:
    std::pair<std::vector<std::vector<int>>, double> solveMasterProblem() override {
        return master_problem_->solveMaster();
    }

    bool solveSubProblem(std::vector<std::vector<int>>& z_input, std::vector<std::vector<double>>& x_sol, std::vector<std::vector<double>>& u_sol, double& f_obj, 
                         std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param, double& const_part) override {
        return sub_problem_->solveSub(z_input, x_sol, u_sol, f_obj, dual_z, dual_param, const_part);
    }

    void getSolution(std::map<std::string, double>& solution) const override;

private:    
    CartPoleParams params_;
};

} // namespace optimization
