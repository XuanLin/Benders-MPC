#pragma once
#include "optimization/gbd/base_master_solver.hpp"
#include <list>

namespace optimization {

class GreedyMasterSolver : public BaseMasterSolver {
public:
    explicit GreedyMasterSolver(const util::SolverParams& params);
    ~GreedyMasterSolver() override = default;

    std::pair<std::vector<std::vector<int>>, double> solveMaster() override;
    virtual void addOptimalityCut(std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param, double const_part) override;
    virtual void addFeasibilityCut(std::stack<VectorDyn>& dual_z, std::stack<VectorDyn>& dual_param) override;
    virtual void storeOptimalityCut() override;
    virtual void storeFeasibilityCut() override;
    int getOptimalityCutCount() const override;
    int getFeasibilityCutCount() const override;

protected:
    void onParamUpdate() override {
        generateSpAndSq();
    }

private:
    virtual void generateSpAndSq();
    std::pair<std::list<int>, std::list<double>> solveTimeStep(const int time_step,
        const std::vector<std::vector<double>>& Sp, const std::vector<double>& Sq,
        const std::vector<std::vector<double>>& new_Sp, const std::vector<double>& new_Sq);

    // Problem configuration
    const int lookahead_;
    const int K_feas_;
    const int K_opt_;

    // Cut storage
    std::vector<VectorDyn> dual_opt_z_;
    std::vector<VectorDyn> dual_opt_param_;
    std::vector<double> dual_opt_const_;
    std::vector<VectorDyn> new_dual_opt_z_;
    std::vector<VectorDyn> new_dual_opt_param_;
    std::vector<double> new_dual_opt_const_;
    int opt_begin_;
    int opt_len_;
    bool opt_full_;

    // Feasibility cuts storage
    std::vector<std::vector<VectorDyn>> dual_feas_z_;        
    std::vector<std::vector<VectorDyn>> dual_feas_param_;    
    std::vector<std::vector<VectorDyn>> new_dual_feas_z_;    
    std::vector<std::vector<VectorDyn>> new_dual_feas_param_;
    std::vector<int> feas_begin_;                            
    std::vector<int> feas_len_;                              
    std::vector<bool> feas_full_;                            

    // Sp and Sq storage (already present but listed for completeness)
    std::vector<std::vector<double>> Sp_;
    std::vector<double> Sq_;
    std::vector<std::vector<double>> new_Sp_;
    std::vector<double> new_Sq_;

    // Current solution storage
    bool solution_found_ = false;
};

} // namespace optimization
