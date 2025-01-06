// base_subproblem.hpp
#pragma once

#include <memory>
#include <stack>
#include <vector>
#include "common/types.hpp"
#include "optimization/util/solver_params.hpp"
#include "optimization/util/dual_naming.hpp"
#include "optimization/util/gurobi_env.hpp"
#include "gurobi_c++.h"

namespace optimization {

class BaseSubSolver : protected util::GurobiEnv {
public:
    explicit BaseSubSolver(const util::SolverParams& params);
    virtual ~BaseSubSolver() = default;

    virtual bool optimize(const std::vector<std::vector<int>>& z_input,
                          std::vector<std::vector<double>>& x_sol, std::vector<std::vector<double>>& u_sol, double& obj_value,
                          std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param, double& const_part);

    virtual void updateInitialConditions(const VectorDyn& x0_new, const VectorDyn& h_theta_new);

protected:
    std::unique_ptr<util::DualNameManager> dual_manager_;
    const util::SolverParams& params_;
    VectorDyn in_param_;
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
