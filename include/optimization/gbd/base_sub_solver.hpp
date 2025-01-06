// base_subproblem.hpp
#pragma once

#include <memory>
#include <stack>
#include <vector>
#include "common/types.hpp"
#include "optimization/util/solver_params.hpp"
#include "optimization/util/dual_naming.hpp"
#include "gurobi_c++.h"

namespace optimization {

class BaseSubSolver {
public:
    explicit BaseSubSolver(const util::SolverParams& params);
    virtual ~BaseSubSolver() = default;

    virtual bool solveSub(const std::vector<std::vector<int>>& z_input, std::vector<std::vector<double>>& x_sol, std::vector<std::vector<double>>& u_sol, double& obj_value,
                          std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param, double& const_part) = 0;

    virtual void updateInitialConditions(const VectorDyn& x0_new, const VectorDyn& h_theta_new);

protected:
    virtual void onParamUpdate() = 0;
    
    std::unique_ptr<util::DualNameManager> dual_manager_;
    const util::SolverParams& params_;
    VectorDyn in_param_;

};

} // namespace optimization
