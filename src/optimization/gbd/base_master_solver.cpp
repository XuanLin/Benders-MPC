#include "optimization/gbd/base_master_solver.hpp"
#include <cassert>

namespace optimization {

BaseMasterSolver::BaseMasterSolver(const util::SolverParams& params)
    : params_(params) {
    in_param_ = VectorDyn::Zero(params_.dual_len);
}

void BaseMasterSolver::updateInitialConditions(const VectorDyn& x0_new, const VectorDyn& h_theta_new) {
    assert(x0_new.size() == params_.nx && "Initial state vector dimension mismatch");
    assert(h_theta_new.size() == params_.nc && "Constraint bounds vector dimension mismatch");

    // First nx elements are initial state
    in_param_.head(params_.nx) = x0_new;

    // Next N*nc elements are constraint bounds for each timestep
    for (int t = 0; t < params_.N; t++) {
        for (int i = 0; i < params_.nc; i++) {
            in_param_((params_.N + 1) * params_.nx + t * params_.nc + i) = h_theta_new[i];
        }
    }

    onParamUpdate();
}

} // namespace optimization

