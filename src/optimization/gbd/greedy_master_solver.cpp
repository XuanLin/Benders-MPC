#include "optimization/gbd/greedy_master_solver.hpp"
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric> 

namespace optimization {

GreedyMasterSolver::GreedyMasterSolver(const util::SolverParams& params) : BaseMasterSolver(params), lookahead_(1), K_feas_(50), K_opt_(40)  {

    dual_opt_z_.resize(K_opt_);
    dual_opt_param_.resize(K_opt_);
    dual_opt_const_.resize(K_opt_, 0.0);
    for (auto& vec : dual_opt_z_) {
        vec = VectorDyn::Zero(params_.N * params_.nz);
    }
    for (auto& vec : dual_opt_param_) {
        vec = VectorDyn::Zero(params_.dual_len);
    }
    Sq_.resize(K_opt_, 0.0);

    dual_feas_z_.resize(params_.N);
    dual_feas_param_.resize(params_.N);
    Sp_.resize(params_.N);
    for (int i_n = 0; i_n < params_.N; i_n++) {
        dual_feas_z_[i_n].resize(K_feas_/params_.N);
        dual_feas_param_[i_n].resize(K_feas_/params_.N);
        Sp_[i_n].resize(K_feas_/params_.N, 0.0);
        for (auto& vec : dual_feas_z_[i_n]) {
            vec = VectorDyn::Zero(params_.N * params_.nz);
        }
        for (auto& vec : dual_feas_param_[i_n]) {
            vec = VectorDyn::Zero(params_.dual_len);
        }
    }

    new_dual_opt_z_.reserve(K_opt_);
    new_dual_opt_param_.reserve(K_opt_);
    new_dual_opt_const_.reserve(K_opt_);
    new_Sq_.reserve(K_opt_);

    new_dual_feas_z_.resize(params_.N);
    new_dual_feas_param_.resize(params_.N);
    new_Sp_.resize(params_.N);

    opt_begin_ = 0;
    opt_len_ = 0;
    opt_full_ = false;
    
    feas_begin_.resize(params_.N, 0);
    feas_len_.resize(params_.N, 0);
    feas_full_.resize(params_.N, false);

}

void GreedyMasterSolver::addOptimalityCut(std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param, double const_part){
    
    double this_Sq = const_part - dual_param.top().dot(in_param_);
    new_Sq_.push_back(this_Sq);

    new_dual_opt_z_.push_back(dual_z.top());
    new_dual_opt_param_.push_back(dual_param.top());
    new_dual_opt_const_.push_back(const_part);
    
    dual_z.pop(); 
    dual_param.pop();
}

void GreedyMasterSolver::addFeasibilityCut(std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param){

    int i_n = 0;
    while (!dual_z.empty()) {
        double this_Sp = -dual_param.top().dot(in_param_);
        
        new_dual_feas_z_[i_n].push_back(dual_z.top());
        new_dual_feas_param_[i_n].push_back(dual_param.top());
        new_Sp_[i_n].push_back(this_Sp);
        
        dual_z.pop(); 
        dual_param.pop();
        i_n++;
    }

}

void GreedyMasterSolver::storeOptimalityCut() {
    for (size_t i_p = 0; i_p < new_dual_opt_z_.size(); ++i_p) {
        dual_opt_z_[opt_begin_] = new_dual_opt_z_[i_p];
        dual_opt_param_[opt_begin_] = new_dual_opt_param_[i_p];
        dual_opt_const_[opt_begin_] = new_dual_opt_const_[i_p];
        opt_begin_++;

        if (!opt_full_) {
            opt_len_++;
        }

        if (opt_begin_ == K_opt_) {
            opt_begin_ = 0;
            opt_full_ = true;
        }
    }

    new_dual_opt_z_.clear();
    new_dual_opt_param_.clear();
    new_dual_opt_const_.clear();
    new_Sq_.clear();
}

void GreedyMasterSolver::storeFeasibilityCut() {
    for (int i_n = 0; i_n < params_.N; i_n++) {
        for (size_t i_p = 0; i_p < new_dual_feas_z_[i_n].size(); ++i_p) {
            dual_feas_z_[i_n][feas_begin_[i_n]] = new_dual_feas_z_[i_n][i_p];
            dual_feas_param_[i_n][feas_begin_[i_n]] = new_dual_feas_param_[i_n][i_p];
            feas_begin_[i_n]++;

            if (!feas_full_[i_n]) {
                feas_len_[i_n]++;
            }

            if (feas_begin_[i_n] == K_feas_/params_.N) {
                feas_begin_[i_n] = 0;
                feas_full_[i_n] = true;
            }
        }

        new_dual_feas_z_[i_n].clear();
        new_dual_feas_param_[i_n].clear();
        new_Sp_[i_n].clear();
    }
}

void GreedyMasterSolver::generateSpAndSq() {
    // Generate Sp values
    for (int i_n = 0; i_n < params_.N; i_n++) {
        for (int i_p = 0; i_p < feas_len_[i_n]; i_p++) {
            Sp_[i_n][i_p] = -dual_feas_param_[i_n][i_p].dot(in_param_);
        }
    }

    // Generate Sq values 
    for (int i_p = 0; i_p < opt_len_; i_p++) {
        Sq_[i_p] = dual_opt_const_[i_p] - dual_opt_param_[i_p].dot(in_param_);
    }

}

int GreedyMasterSolver::getOptimalityCutCount() const {
    return opt_len_ + new_Sq_.size();
}

int GreedyMasterSolver::getFeasibilityCutCount() const {
    int total = std::accumulate(feas_len_.begin(), feas_len_.end(), 0);
    for (int i_n = 0; i_n < params_.N; i_n++) {
        total += new_Sp_[i_n].size();
    }
    return total;
}

std::pair<std::vector<std::vector<int>>, double> GreedyMasterSolver::solveMaster() {
    int i_t = 0;
    bool found_z;
    std::vector<int> ls_z_opt;
    std::vector<std::list<int>> ls_all_z;
    std::list<int> ls_z_star;

    // Make local copy of Sp and Sq
    std::vector<std::vector<double>> in_Sp(params_.N);
    std::vector<std::vector<double>> new_in_Sp(params_.N);
    std::vector<double> in_Sq;
    std::vector<double> new_in_Sq;

    // Copy existing cuts
    for (int i_n = 0; i_n < params_.N; i_n++) {
        in_Sp[i_n] = Sp_[i_n];
        new_in_Sp[i_n] = new_Sp_[i_n];
    }
    in_Sq = Sq_;
    new_in_Sq = new_Sq_;

    while (i_t <= params_.N - 1) {
        auto small_mip_sol = solveTimeStep(i_t, in_Sp, in_Sq, new_in_Sp, new_in_Sq);
        ls_z_star = small_mip_sol.first;

        if (ls_z_star.empty()) {
            // Backtrack to find feasible solution
            found_z = false;
            while (!found_z) {
                i_t -= 1;
                if (i_t < 0) {
                    throw std::runtime_error("Master problem infeasible!");
                }

                // Undo the RHS changes using ls_z_opt[i_t]
                for (int ss = 0; ss < params_.nz; ss++) {
                    if (params_.arr_z[ls_z_opt[i_t]][ss] == 1) {
                        // Update feasibility cuts
                        for (int i_n = 0; i_n < params_.N; i_n++) {
                            for (size_t jj = 0; jj < feas_len_[i_n]; jj++) {
                                in_Sp[i_n][jj] += dual_feas_z_[i_n][jj](i_t * params_.nz + ss);
                            }
                            for (size_t kk = 0; kk < new_dual_feas_z_[i_n].size(); kk++) {
                                new_in_Sp[i_n][kk] += new_dual_feas_z_[i_n][kk](i_t * params_.nz + ss);
                            }
                        }
                        // Update optimality cuts
                        for (size_t jj = 0; jj < opt_len_; jj++) {
                            in_Sq[jj] += dual_opt_z_[jj](i_t * params_.nz + ss);
                        }
                        for (size_t kk = 0; kk < new_dual_opt_z_.size(); kk++) {
                            new_in_Sq[kk] += new_dual_opt_z_[kk](i_t * params_.nz + ss);
                        }
                    }
                }

                if (ls_all_z[i_t].empty()) {
                    ls_z_opt.pop_back();
                    ls_all_z.pop_back();
                } else {
                    ls_z_opt[i_t] = ls_all_z[i_t].back();
                    ls_all_z[i_t].pop_back();
                    found_z = true;
                }
            }
        } else {
            ls_all_z.push_back(ls_z_star);
            ls_z_opt.push_back(ls_all_z[i_t].back());
            ls_all_z[i_t].pop_back();

            // If at final timestep, return solution
            if (i_t == params_.N - 1) {
                std::vector<std::vector<int>> z_input(params_.N, std::vector<int>(params_.nz));
                for (int i_n = 0; i_n < params_.N; i_n++) {
                    for (int i_z = 0; i_z < params_.nz; i_z++) {
                        z_input[i_n][i_z] = params_.arr_z[ls_z_opt[i_n]][i_z];
                    }
                }
                return {z_input, small_mip_sol.second.back()};
            }
        }

        // Update RHS for next timestep
        for (int ss = 0; ss < params_.nz; ss++) {
            if (params_.arr_z[ls_z_opt[i_t]][ss] == 1) {
                for (int i_n = 0; i_n < params_.N; i_n++) {
                    for (size_t jj = 0; jj < feas_len_[i_n]; jj++) {
                        in_Sp[i_n][jj] -= dual_feas_z_[i_n][jj](i_t * params_.nz + ss);
                    }
                    for (size_t kk = 0; kk < new_dual_feas_z_[i_n].size(); kk++) {
                        new_in_Sp[i_n][kk] -= new_dual_feas_z_[i_n][kk](i_t * params_.nz + ss);
                    }
                }
                for (size_t jj = 0; jj < opt_len_; jj++) {
                    in_Sq[jj] -= dual_opt_z_[jj](i_t * params_.nz + ss);
                }
                for (size_t kk = 0; kk < new_dual_opt_z_.size(); kk++) {
                    new_in_Sq[kk] -= new_dual_opt_z_[kk](i_t * params_.nz + ss);
                }
            }
        }

        i_t += 1;
    }

    std::vector<std::vector<int>> z_input(params_.N, std::vector<int>(params_.nz));
    for (int i_n = 0; i_n < params_.N; i_n++) {
        for (int i_z = 0; i_z < params_.nz; i_z++) {
            z_input[i_n][i_z] = params_.arr_z[ls_z_opt[i_n]][i_z];
        }
    }

    return {z_input, -std::pow(2, 10)};
}

std::pair<std::list<int>, std::list<double>> GreedyMasterSolver::solveTimeStep(
    const int time_step,
    const std::vector<std::vector<double>>& Sp,
    const std::vector<double>& Sq,
    const std::vector<std::vector<double>>& new_Sp,
    const std::vector<double>& new_Sq) {

    int ahead = std::min(lookahead_ + 1, params_.N - time_step);
    std::list<int> ret_z;
    std::list<double> f_obj;
    
    // Precompute max future terms for lookahead
    std::vector<std::vector<double>> d_Sp(lookahead_);
    std::vector<std::vector<double>> d_new_Sp(lookahead_);

    // Process lookahead windows
    for (int ah = 1; ah < ahead; ah++) {
        // Process stored cuts
        for (int i_c = 0; i_c < feas_len_[time_step + ah]; i_c++) {
            double max_future = -std::numeric_limits<double>::max();
            for (size_t ii = 0; ii < params_.arr_z.size(); ii++) {
                double future = 0.0;
                for (int ss = 0; ss < params_.nz; ss++) {
                    if (params_.arr_z[ii][ss] == 1) {
                        future += dual_feas_z_[time_step + ah][i_c]((time_step + ah) * params_.nz + ss);
                    }
                }
                max_future = std::max(max_future, future);
            }
            d_Sp[ah-1].push_back(max_future);
        }

        // Process new cuts
        for (size_t s_c = 0; s_c < new_Sp[time_step + ah].size(); s_c++) {
            double max_future = -std::numeric_limits<double>::max();
            for (size_t ii = 0; ii < params_.arr_z.size(); ii++) {
                double future = 0.0;
                for (int ss = 0; ss < params_.nz; ss++) {
                    if (params_.arr_z[ii][ss] == 1) {
                        future += new_dual_feas_z_[time_step + ah][s_c]((time_step + ah) * params_.nz + ss);
                    }
                }
                max_future = std::max(max_future, future);
            }
            d_new_Sp[ah-1].push_back(max_future);
        }
    }

    // Check all possible solutions
    for (size_t ii = 0; ii < params_.arr_z.size(); ii++) {
        bool feasible = true;
        for (int ah = 0; ah < ahead && feasible; ah++) {
            // Check stored cuts
            if (feasible) {
                for (int i_c = 0; i_c < feas_len_[time_step + ah]; i_c++) {
                    double dot_sol = 0;
                    for (int ss = 0; ss < params_.nz; ss++) {
                        if (params_.arr_z[ii][ss] == 1) {
                            dot_sol += dual_feas_z_[time_step + ah][i_c](time_step * params_.nz + ss);
                        }
                    }
                    if (ah >= 1) {
                        if (dot_sol < (Sp[time_step + ah][i_c] - d_Sp[ah-1][i_c])) {
                            feasible = false;
                            break;
                        }
                    } else {
                        if (dot_sol < Sp[time_step + ah][i_c]) {
                            feasible = false;
                            break;
                        }
                    }
                }
            }

            // Check new cuts
            if (feasible) {
                for (size_t s_c = 0; s_c < new_dual_feas_z_[time_step + ah].size(); s_c++) {
                    double dot_sol = 0;
                    for (int ss = 0; ss < params_.nz; ss++) {
                        if (params_.arr_z[ii][ss] == 1) {
                            dot_sol += new_dual_feas_z_[time_step + ah][s_c](time_step * params_.nz + ss);
                        }
                    }
                    if (ah >= 1) {
                        if (dot_sol < (new_Sp[time_step + ah][s_c] - d_new_Sp[ah-1][s_c])) {
                            feasible = false;
                            break;
                        }
                    } else {
                        if (dot_sol < new_Sp[time_step + ah][s_c]) {
                            feasible = false;
                            break;
                        }
                    }
                }
            }
        }

        // Process feasible solutions
        if (feasible) {
            if (Sq.empty()) {
                ret_z.push_front(ii);
                f_obj.push_front(std::pow(10, 8));
            } else {
                double max_cost = -std::numeric_limits<double>::max();
                
                // Compute cost for stored cuts
                for (size_t jj = 0; jj < opt_len_; jj++) {
                    double this_cost = Sq[jj];
                    for (int ss = 0; ss < params_.nz; ss++) {
                        if (params_.arr_z[ii][ss] == 1) {
                            this_cost -= dual_opt_z_[jj](time_step * params_.nz + ss);
                        }
                    }
                    max_cost = std::max(max_cost, this_cost);
                }

                // Compute cost for new cuts
                for (size_t s_c = 0; s_c < new_Sq.size(); s_c++) {
                    double this_cost = new_Sq[s_c];
                    for (int ss = 0; ss < params_.nz; ss++) {
                        if (params_.arr_z[ii][ss] == 1) {
                            this_cost -= new_dual_opt_z_[s_c](time_step * params_.nz + ss);
                        }
                    }
                    max_cost = std::max(max_cost, this_cost);
                }

                // Insert solution in order
                auto pt = f_obj.begin();
                auto ptf = ret_z.begin();
                while (pt != f_obj.end() && max_cost > *pt) {
                    ++pt;
                    ++ptf;
                }
                f_obj.insert(pt, max_cost);
                ret_z.insert(ptf, ii);
            }
        }
    }

    return {ret_z, f_obj};
}

} // namespace optimization

