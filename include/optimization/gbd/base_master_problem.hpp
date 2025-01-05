#pragma once

#include <stack>
#include <vector>
#include <utility>

#include "common/types.hpp"
#include "optimization/util/solver_params.hpp"
#include "optimization/util/gurobi_env.hpp"

namespace optimization {

class BaseMasterProblem : protected util::GurobiEnv {
public:
    explicit BaseMasterProblem(const util::SolverParams& params);
    virtual ~BaseMasterProblem() = default;

    void updateInitialConditions(const VectorDyn& x0_new, const VectorDyn& h_theta_new);
    virtual std::pair<std::vector<std::vector<int>>, double> solveMaster() = 0;
    virtual void addOptimalityCut(std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param, double const_part) = 0;
    virtual void addFeasibilityCut(std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param) = 0;
    virtual void storeOptimalityCut() = 0;
    virtual void storeFeasibilityCut() = 0;
    virtual int getOptimalityCutCount() const = 0;
    virtual int getFeasibilityCutCount() const = 0;
    
protected:
    virtual void onParamUpdate() = 0;

    const util::SolverParams& params_;
    VectorDyn in_param_;

private:
    BaseMasterProblem(const BaseMasterProblem&) = delete;
    BaseMasterProblem& operator=(const BaseMasterProblem&) = delete;
};

}
