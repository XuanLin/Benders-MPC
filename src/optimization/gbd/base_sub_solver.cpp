#include "optimization/gbd/base_sub_solver.hpp"
#include <cassert>

namespace optimization {

BaseSubSolver::BaseSubSolver(const util::SolverParams& params) : params_(params) {
    dual_manager_ = std::make_unique<util::DualNameManager>(params_.N, params_.nx, params_.nu, params_.nc);
    in_param_ = VectorDyn::Zero(params_.dual_len);
}

void BaseSubSolver::updateInitialConditions(const VectorDyn& x0_new, const VectorDyn& h_theta_new) {
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
