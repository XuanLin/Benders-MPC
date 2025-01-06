// base_subproblem.cpp
#include "optimization/gbd/gurobi_sub_solver.hpp"

namespace optimization {

GurobiSubSolver::GurobiSubSolver(const util::SolverParams& params) : BaseSubSolver(params) {

    try {
        model_ = std::make_unique<GRBModel>(get());
        model_infeas_ = std::make_unique<GRBModel>(get());
        
        model_->set(GRB_DoubleParam_BarConvTol, 1e-4);
        model_->set(GRB_IntParam_OutputFlag, 0);
        model_infeas_->set(GRB_IntParam_InfUnbdInfo, 1);
        model_infeas_->set(GRB_IntParam_OutputFlag, 0);

        setupPrimalModel();
        setupInfeasibilityModel();

    } catch (GRBException& e) {
        std::cerr << "Error code = " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
        throw;
    }
}

void GurobiSubSolver::setupPrimalModel() {

    try {
        // 1. Initialize variables ==============================================
        // Initial state variables
        x0_vars_.resize(params_.nx);
        for (int i = 0; i < params_.nx; i++) {
            x0_vars_[i] = model_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
                GRB_CONTINUOUS, "x0_" + std::to_string(i));
        }

        // Constraint bounds
        h_theta_vars_.resize(params_.nc);
        for (int i = 0; i < params_.nc; i++) {
            h_theta_vars_[i] = model_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
                GRB_CONTINUOUS, "h_theta_sub_" + std::to_string(i));
        }

        // System dynamics variables
        h_d_vars_.resize(params_.nx);
        for (int i = 0; i < params_.nx; i++) {
            h_d_vars_[i] = model_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
                GRB_CONTINUOUS, "h_d_theta_sub_" + std::to_string(i));
        }

        // State variables over horizon
        x_vars_.resize(params_.N + 1);
        for (int t = 0; t < params_.N + 1; t++) {
            x_vars_[t].resize(params_.nx);
            for (int i = 0; i < params_.nx; i++) {
                x_vars_[t][i] = model_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
                    GRB_CONTINUOUS, "x_sub_t_" + std::to_string(t) + "_item_" + std::to_string(i));
            }
        }

        // Control variables over horizon
        u_vars_.resize(params_.N);
        for (int t = 0; t < params_.N; t++) {
            u_vars_[t].resize(params_.nu);
            for (int i = 0; i < params_.nu; i++) {
                u_vars_[t][i] = model_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
                    GRB_CONTINUOUS, "u_sub_t_" + std::to_string(t) + "_item_" + std::to_string(i));
            }
        }

        // Binary variables over horizon
        z_vars_.resize(params_.N);
        for (int t = 0; t < params_.N; t++) {
            z_vars_[t].resize(params_.nz);
            for (int i = 0; i < params_.nz; i++) {
                z_vars_[t][i] = model_->addVar(0.0, 1.0, 0.0, 
                    GRB_CONTINUOUS, "z_sub_t_" + std::to_string(t) + "_item_" + std::to_string(i));
            }
        }

        // 2. Add constraints ================================================
        // Initial conditions
        for (int i = 0; i < params_.nx; i++) {
            model_->addConstr(x_vars_[0][i] == x0_vars_[i], dual_manager_->dual_names_[dual_manager_->id_x0_[i]]);
        }

        // System dynamics
        for (int t = 0; t < params_.N; t++) {
            for (int i = 0; i < params_.nx; i++) {
                GRBLinExpr expr = h_d_vars_[i];
                // State contribution
                for (int j = 0; j < params_.nx; j++) {
                    expr += params_.E(i, j) * x_vars_[t][j];
                }
                // Control contribution
                for (int j = 0; j < params_.nu; j++) {
                    expr += params_.F(i, j) * u_vars_[t][j];
                }
                // Binary variable contribution
                for (int j = 0; j < params_.nz; j++) {
                    expr += params_.G(i, j) * z_vars_[t][j];
                }
                
                model_->addConstr(x_vars_[t+1][i] == expr, dual_manager_->dual_names_[dual_manager_->id_x_dyn_[t][i]]);
            }
        }

        // Control constraints
        for (int t = 0; t < params_.N; t++) {
            for (int i = 0; i < params_.nc; i++) {
                GRBLinExpr expr = 0;
                // State contribution
                for (int j = 0; j < params_.nx; j++) {
                    expr += params_.H1(i, j) * x_vars_[t][j];
                }
                // Control contribution
                for (int j = 0; j < params_.nu; j++) {
                    expr += params_.H2(i, j) * u_vars_[t][j];
                }
                // Binary variable contribution
                for (int j = 0; j < params_.nz; j++) {
                    expr += params_.H3(i, j) * z_vars_[t][j];
                }
                
                model_->addConstr(expr <= h_theta_vars_[i], dual_manager_->dual_names_[dual_manager_->id_xuz_[t][i]]);
            }
        }

        // Set h_d_vars to zero
        for (int i = 0; i < params_.nx; i++) {
            h_d_vars_[i].set(GRB_DoubleAttr_LB, 0.0);
            h_d_vars_[i].set(GRB_DoubleAttr_UB, 0.0);
        }

        // 3. Setup objective ================================================
        objective_ = 0;
        // Control cost
        for (int t = 0; t < params_.N; t++) {
            for (int i = 0; i < params_.nu; i++) {
                objective_ += u_vars_[t][i] * params_.R(i,i) * u_vars_[t][i];
            }
        }
        
        // State cost
        for (int t = 0; t < params_.N; t++) {
            for (int i = 0; i < params_.nx; i++) {
                objective_ += (x_vars_[t][i] - params_.x_goal[i]) * 
                            params_.Q(i,i) * 
                            (x_vars_[t][i] - params_.x_goal[i]);
            }
        }
        
        // Terminal cost
        for (int i = 0; i < params_.nx; i++) {
            objective_ += (x_vars_[params_.N][i] - params_.x_goal[i]) * 
                        params_.Qn(i,i) * 
                        (x_vars_[params_.N][i] - params_.x_goal[i]);
        }

        model_->setObjective(objective_, GRB_MINIMIZE);

    } catch (GRBException& e) {
        std::cerr << "Error code = " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
        throw;
    }
}

void GurobiSubSolver::setupInfeasibilityModel() {
    // 1. Initialize variables ==============================================
    // Initial state variables
    x0_infeas_vars_.resize(params_.nx);
    for (int i = 0; i < params_.nx; i++) {
        x0_infeas_vars_[i] = model_infeas_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
            GRB_CONTINUOUS, "x0_" + std::to_string(i));
    }

    // Constraint bounds
    h_theta_infeas_vars_.resize(params_.nc);
    for (int i = 0; i < params_.nc; i++) {
        h_theta_infeas_vars_[i] = model_infeas_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
            GRB_CONTINUOUS, "h_theta_infeas_" + std::to_string(i));
    }

    // System dynamics variables
    h_d_infeas_vars_.resize(params_.nx);
    for (int i = 0; i < params_.nx; i++) {
        h_d_infeas_vars_[i] = model_infeas_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
            GRB_CONTINUOUS, "h_d_theta_infeas_" + std::to_string(i));
    }

    // State variables over horizon
    x_infeas_vars_.resize(params_.N + 1);
    for (int t = 0; t < params_.N + 1; t++) {
        x_infeas_vars_[t].resize(params_.nx);
        for (int i = 0; i < params_.nx; i++) {
            x_infeas_vars_[t][i] = model_infeas_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
                GRB_CONTINUOUS, "x_infeas_t_" + std::to_string(t) + "_item_" + std::to_string(i));
        }
    }

    // Control variables over horizon
    u_infeas_vars_.resize(params_.N);
    for (int t = 0; t < params_.N; t++) {
        u_infeas_vars_[t].resize(params_.nu);
        for (int i = 0; i < params_.nu; i++) {
            u_infeas_vars_[t][i] = model_infeas_->addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, 
                GRB_CONTINUOUS, "u_infeas_t_" + std::to_string(t) + "_item_" + std::to_string(i));
        }
    }

    // Binary variables over horizon
    z_infeas_vars_.resize(params_.N);
    for (int t = 0; t < params_.N; t++) {
        z_infeas_vars_[t].resize(params_.nz);
        for (int i = 0; i < params_.nz; i++) {
            z_infeas_vars_[t][i] = model_infeas_->addVar(0.0, 1.0, 0.0, 
                GRB_CONTINUOUS, "z_infeas_t_" + std::to_string(t) + "_item_" + std::to_string(i));
        }
    }

    // 2. Add constraints ================================================
    // Initial conditions
    for (int i = 0; i < params_.nx; i++) {
        model_infeas_->addConstr(x_infeas_vars_[0][i] == x0_infeas_vars_[i], dual_manager_->dual_names_[dual_manager_->id_x0_[i]]);
    }

    // System dynamics
    for (int t = 0; t < params_.N; t++) {
        for (int i = 0; i < params_.nx; i++) {
            GRBLinExpr expr = h_d_infeas_vars_[i];
            // State contribution
            for (int j = 0; j < params_.nx; j++) {
                expr += params_.E(i, j) * x_infeas_vars_[t][j];
            }
            // Control contribution
            for (int j = 0; j < params_.nu; j++) {
                expr += params_.F(i, j) * u_infeas_vars_[t][j];
            }
            // Binary variable contribution
            for (int j = 0; j < params_.nz; j++) {
                expr += params_.G(i, j) * z_infeas_vars_[t][j];
            }
            
            model_infeas_->addConstr(x_infeas_vars_[t+1][i] == expr, dual_manager_->dual_names_[dual_manager_->id_x_dyn_[t][i]]);
        }
    }

    // Control constraints
    for (int t = 0; t < params_.N; t++) {
        for (int i = 0; i < params_.nc; i++) {
            GRBLinExpr expr = 0;
            // State contribution
            for (int j = 0; j < params_.nx; j++) {
                expr += params_.H1(i, j) * x_infeas_vars_[t][j];
            }
            // Control contribution
            for (int j = 0; j < params_.nu; j++) {
                expr += params_.H2(i, j) * u_infeas_vars_[t][j];
            }
            // Binary variable contribution
            for (int j = 0; j < params_.nz; j++) {
                expr += params_.H3(i, j) * z_infeas_vars_[t][j];
            }
            
            model_infeas_->addConstr(expr <= h_theta_infeas_vars_[i], dual_manager_->dual_names_[dual_manager_->id_xuz_[t][i]]);
        }
    }

    // Set h_d_infeas_vars to zero
    for (int i = 0; i < params_.nx; i++) {
        h_d_infeas_vars_[i].set(GRB_DoubleAttr_LB, 0.0);
        h_d_infeas_vars_[i].set(GRB_DoubleAttr_UB, 0.0);
    }

    // For infeasibility model, we don't need an objective function
    GRBLinExpr zero_obj = 0;
    model_infeas_->setObjective(zero_obj);
}

void GurobiSubSolver::onParamUpdate() {
    try {
        // Update x0 bounds
        for (int i = 0; i < params_.nx; i++) {
            x0_vars_[i].set(GRB_DoubleAttr_LB, in_param_[i]);
            x0_vars_[i].set(GRB_DoubleAttr_UB, in_param_[i]);
            x0_infeas_vars_[i].set(GRB_DoubleAttr_LB, in_param_[i]);
            x0_infeas_vars_[i].set(GRB_DoubleAttr_UB, in_param_[i]);
        }
        
        // Update h_theta bounds
        for (int t = 0; t < params_.N; t++) {
            for (int i = 0; i < params_.nc; i++) {
                int idx = (params_.N + 1) * params_.nx + t * params_.nc + i;
                h_theta_vars_[i].set(GRB_DoubleAttr_LB, in_param_[idx]);
                h_theta_vars_[i].set(GRB_DoubleAttr_UB, in_param_[idx]);
                h_theta_infeas_vars_[i].set(GRB_DoubleAttr_LB, in_param_[idx]);
                h_theta_infeas_vars_[i].set(GRB_DoubleAttr_UB, in_param_[idx]);
            }
        }

    } catch (GRBException& e) {
        std::cerr << "Error code = " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
        throw;
    }
}

bool GurobiSubSolver::solveSub(const std::vector<std::vector<int>>& z_input, std::vector<std::vector<double>>& x_sol, std::vector<std::vector<double>>& u_sol,
    double& obj_value, std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param, double& const_part) {

    try {
        // Update binary variables in primal model
        for (int t = 0; t < params_.N; t++) {
            for (int i = 0; i < params_.nz; i++) {
                z_vars_[t][i].set(GRB_DoubleAttr_LB, z_input[t][i]);
                z_vars_[t][i].set(GRB_DoubleAttr_UB, z_input[t][i]);
            }
        }

        // Optimize primal model
        model_->optimize();
        int optimstatus = model_->get(GRB_IntAttr_Status);

        if (optimstatus == GRB_OPTIMAL) {
            // Extract solution
            for (int t = 0; t < params_.N + 1; t++) {
                for (int i = 0; i < params_.nx; i++) {
                    x_sol[t][i] = x_vars_[t][i].get(GRB_DoubleAttr_X);
                }
            }

            for (int t = 0; t < params_.N; t++) {
                for (int i = 0; i < params_.nu; i++) {
                    u_sol[t][i] = u_vars_[t][i].get(GRB_DoubleAttr_X);
                }
            }

            // Get objective
            obj_value=0.0;
        
            for (int i_n=0; i_n<params_.N; i_n++){
                for (int i_u=0; i_u<params_.nu; i_u++){
                    obj_value += u_sol[i_n][i_u]*params_.R(i_u, i_u)*u_sol[i_n][i_u];}}
            
            for (int i_n=0; i_n<params_.N; i_n++){
                for (int i_x=0; i_x<params_.nx; i_x++){
                    obj_value += (x_sol[i_n][i_x]-params_.x_goal[i_x])*params_.Q(i_x, i_x)*(x_sol[i_n][i_x]-params_.x_goal[i_x]);}}

            for (int i_x=0; i_x<params_.nx; i_x++){
                obj_value += (x_sol[params_.N][i_x]-params_.x_goal[i_x])*params_.Qn(i_x, i_x)*(x_sol[params_.N][i_x]-params_.x_goal[i_x]);}

            // obj_value = model_->get(GRB_DoubleAttr_ObjVal);  // Once a while getting negative cost, why ??

            // Get dual values
            VectorDyn dual_values(params_.dual_len);
            for (int i = 0; i < params_.dual_len; i++) {
                dual_values[i] = -model_->getConstrByName(dual_manager_->dual_names_[i]).get(GRB_DoubleAttr_Pi);
            }

            // Compute dual_z
            VectorDyn this_dual_z(params_.N * params_.nz);
            for (int i_n = 0; i_n < params_.N; i_n++) {
                for (int i_z = 0; i_z < params_.nz; i_z++) {
                    double tmp = 0;
                    for (int i_x = 0; i_x < params_.nx; i_x++) {
                        tmp += dual_values[dual_manager_->id_x_dyn_[i_n][i_x]] * params_.G(i_x, i_z);
                    }
                    for (int i_c = 0; i_c < params_.nc; i_c++) {
                        tmp -= dual_values[dual_manager_->id_xuz_[i_n][i_c]] * params_.H3(i_c, i_z);
                    }
                    this_dual_z(i_n * params_.nz + i_z) = tmp;
                }
            }
            dual_z.push(this_dual_z);

            // Compute dual_param
            VectorDyn this_dual_param(params_.dual_len);
            for (int i_x = 0; i_x < params_.nx; i_x++) {
                this_dual_param(i_x) = dual_values[dual_manager_->id_x0_[i_x]];
            }
            for (int i_n = 0; i_n < params_.N; i_n++) {
                for (int i_x = 0; i_x < params_.nx; i_x++) {
                    this_dual_param((i_n + 1) * params_.nx + i_x) = dual_values[dual_manager_->id_x_dyn_[i_n][i_x]];
                }
            }
            for (int i_n = 0; i_n < params_.N; i_n++) {
                for (int i_c = 0; i_c < params_.nc; i_c++) {
                    this_dual_param((params_.N + 1) * params_.nx + i_n * params_.nc + i_c) = dual_values[dual_manager_->id_xuz_[i_n][i_c]];
                }
            }
            dual_param.push(this_dual_param);

            // Compute const_part
            const_part = obj_value;
            for (int i_x = 0; i_x < params_.nx; i_x++) {
                const_part += x0_vars_[i_x].get(GRB_DoubleAttr_X) * dual_values[dual_manager_->id_x0_[i_x]];
            }
            for (int i_n = 0; i_n < params_.N; i_n++) {
                for (int i_c = 0; i_c < params_.nc; i_c++) {
                    const_part += h_theta_vars_[i_c].get(GRB_DoubleAttr_X) * dual_values[dual_manager_->id_xuz_[i_n][i_c]];
                }
            }
            for (int i_n = 0; i_n < params_.N; i_n++) {
                for (int i_z = 0; i_z < params_.nz; i_z++) {
                    for (int i_x = 0; i_x < params_.nx; i_x++) {
                        const_part += dual_values[dual_manager_->id_x_dyn_[i_n][i_x]] * params_.G(i_x, i_z) * z_vars_[i_n][i_z].get(GRB_DoubleAttr_X);
                    }
                    for (int i_c = 0; i_c < params_.nc; i_c++) {
                        const_part -= dual_values[dual_manager_->id_xuz_[i_n][i_c]] * params_.H3(i_c, i_z) * z_vars_[i_n][i_z].get(GRB_DoubleAttr_X);
                    }
                }
            }

            return true;

        } else {
            // Handle infeasibility
            for (int t = 0; t < params_.N; t++) {
                for (int i = 0; i < params_.nz; i++) {
                    z_infeas_vars_[t][i].set(GRB_DoubleAttr_LB, z_input[t][i]);
                    z_infeas_vars_[t][i].set(GRB_DoubleAttr_UB, z_input[t][i]);
                }
            }

            model_infeas_->optimize();

            // Get Farkas dual values
            VectorDyn dual_values(params_.dual_len);
            for (int i = 0; i < params_.dual_len; i++) {
                dual_values[i] = model_infeas_->getConstrByName(dual_manager_->dual_names_[i]).get(GRB_DoubleAttr_FarkasDual);
            }

            // Compute dual_z
            VectorDyn this_dual_z(params_.N * params_.nz);
            for (int i_n = 0; i_n < params_.N; i_n++) {
                for (int i_z = 0; i_z < params_.nz; i_z++) {
                    double tmp = 0;
                    for (int i_x = 0; i_x < params_.nx; i_x++) {
                        tmp += dual_values[dual_manager_->id_x_dyn_[i_n][i_x]] * params_.G(i_x, i_z);
                    }
                    for (int i_c = 0; i_c < params_.nc; i_c++) {
                        tmp -= dual_values[dual_manager_->id_xuz_[i_n][i_c]] * params_.H3(i_c, i_z);
                    }
                    this_dual_z(i_n * params_.nz + i_z) = tmp;
                }
            }
            dual_z.push(this_dual_z);

            // Compute dual_param
            VectorDyn this_dual_param(params_.dual_len);
            for (int i_x = 0; i_x < params_.nx; i_x++) {
                this_dual_param(i_x) = dual_values[dual_manager_->id_x0_[i_x]];
            }
            for (int i_n = 0; i_n < params_.N; i_n++) {
                for (int i_x = 0; i_x < params_.nx; i_x++) {
                    this_dual_param((i_n + 1) * params_.nx + i_x) = dual_values[dual_manager_->id_x_dyn_[i_n][i_x]];
                }
            }
            for (int i_n = 0; i_n < params_.N; i_n++) {
                for (int i_c = 0; i_c < params_.nc; i_c++) {
                    this_dual_param((params_.N + 1) * params_.nx + i_n * params_.nc + i_c) = dual_values[dual_manager_->id_xuz_[i_n][i_c]];
                }
            }
            dual_param.push(this_dual_param);

            // Process additional cuts
            bool use_additonal_cuts = true;
            if (use_additonal_cuts){
                bool all_zero;
                do {
                    all_zero = true;
                    this_dual_z.setZero();
                    for (int i_n = 0; i_n < params_.N - 1; i_n++) {
                        for (int i_z = 0; i_z < params_.nz; i_z++) {
                            if (dual_z.top()((i_n + 1) * params_.nz + i_z) != 0) {
                                this_dual_z(i_n * params_.nz + i_z) = dual_z.top()((i_n + 1) * params_.nz + i_z);
                                all_zero = false;
                            }
                        }
                    }

                    this_dual_param.setZero();
                    for (int i_x = 0; i_x < params_.nx; i_x++) {
                        if (dual_param.top()[dual_manager_->id_x_dyn_[0][i_x]] != 0) {
                            this_dual_param(i_x) = dual_param.top()[dual_manager_->id_x_dyn_[0][i_x]];
                            all_zero = false;
                        }
                    }

                    for (int i_n = 0; i_n < params_.N - 1; i_n++) {
                        for (int i_x = 0; i_x < params_.nx; i_x++) {
                            if (dual_param.top()[dual_manager_->id_x_dyn_[i_n + 1][i_x]] != 0) {
                                this_dual_param((i_n + 1) * params_.nx + i_x) = dual_param.top()[dual_manager_->id_x_dyn_[i_n + 1][i_x]];
                                all_zero = false;
                            }
                        }
                    }

                    for (int i_n = 0; i_n < params_.N - 1; i_n++) {
                        for (int i_c = 0; i_c < params_.nc; i_c++) {
                            if (dual_param.top()[dual_manager_->id_xuz_[i_n + 1][i_c]] != 0) {
                                this_dual_param((params_.N + 1) * params_.nx + i_n * params_.nc + i_c) = dual_param.top()[dual_manager_->id_xuz_[i_n + 1][i_c]];
                                all_zero = false;
                            }
                        }
                    }

                    if (!all_zero) {
                        dual_z.push(this_dual_z);
                        dual_param.push(this_dual_param);
                    }
                } while (!all_zero);
            }

            return false;
        }

    } catch (GRBException& e) {
        std::cerr << "Error code = " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
        throw;
    }
}

} // namespace optimization
