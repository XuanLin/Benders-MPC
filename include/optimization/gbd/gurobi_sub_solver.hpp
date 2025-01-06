// gurobi_sub_solver.hpp
#pragma once

#include "optimization/gbd/base_sub_solver.hpp"
#include "optimization/util/gurobi_env.hpp"
#include "gurobi_c++.h"

namespace optimization {

class GurobiSubSolver : public BaseSubSolver, protected util::GurobiEnv {
public:
    explicit GurobiSubSolver(const util::SolverParams& params);
    ~GurobiSubSolver() override = default;

    bool solveSub(const std::vector<std::vector<int>>& z_input, std::vector<std::vector<double>>& x_sol, std::vector<std::vector<double>>& u_sol, double& obj_value,
                  std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param, double& const_part) override;

protected:
    void onParamUpdate() override;

private:
    std::unique_ptr<GRBModel> model_;
    std::unique_ptr<GRBModel> model_infeas_;
    
    std::vector<GRBVar> x0_vars_, h_theta_vars_, h_d_vars_;
    std::vector<std::vector<GRBVar>> x_vars_, u_vars_, z_vars_;
    std::vector<GRBVar> x0_infeas_vars_, h_theta_infeas_vars_, h_d_infeas_vars_;
    std::vector<std::vector<GRBVar>> x_infeas_vars_, u_infeas_vars_, z_infeas_vars_;
    
    GRBQuadExpr objective_;

    void setupPrimalModel();
    void setupInfeasibilityModel();
};

} // namespace optimization
