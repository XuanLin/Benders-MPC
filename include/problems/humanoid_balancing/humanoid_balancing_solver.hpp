// humanoid_balancing_solver.hpp
#pragma once

#include <memory>
#include "optimization/gbd/base_solver.hpp"
#include "optimization/gbd/greedy_master_solver.hpp"
#include "optimization/gbd/gurobi_sub_solver.hpp"
#include "problems/humanoid_balancing/humanoid_balancing_param.hpp"

namespace optimization {

class HumanoidBalancingGBDSolver : public BaseGBDSolver {
public:   
    HumanoidBalancingGBDSolver(
        const HumanoidBalancingParams& params, 
        std::unique_ptr<BaseMasterSolver> master, 
        std::unique_ptr<BaseSubSolver> sub
    ) : BaseGBDSolver(params, std::move(master), std::move(sub)), params_(params) {
    }

    // Default constructor that uses GreedyMaster and GurobiSubSolver
    explicit HumanoidBalancingGBDSolver(const HumanoidBalancingParams& params = HumanoidBalancingParams())
        : HumanoidBalancingGBDSolver(
            params, 
            std::make_unique<GreedyMasterSolver>(params), 
            std::make_unique<GurobiSubSolver>(params)
        ) {
    }

    ~HumanoidBalancingGBDSolver() override = default;

protected:
    void getSolution(std::map<std::string, double>& solution) const override;

private:    
    HumanoidBalancingParams params_;
};

} // namespace optimization

