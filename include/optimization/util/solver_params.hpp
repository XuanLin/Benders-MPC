#pragma once

#include "common/types.hpp"

namespace optimization {
namespace util {

struct SolverParams {
    // System dimensions
    int N;             
    int nx;            
    int nu;            
    int nz;            
    int nc;            
    int dual_len;      
    double dT;         
    VectorDyn x_goal;  

    // System matrices
    MatrixDyn Q, Qn;   // Cost matrices
    MatrixDyn R;       // Control cost matrix
    MatrixDyn E;       // System dynamics
    MatrixDyn F;       // Control matrix
    MatrixDyn G;       // Binary variable matrix
    MatrixDyn H1;      // State constraints
    MatrixDyn H2;      // Control constraints
    MatrixDyn H3;      // Binary variable constraints

    // Possible binary solutions
    std::vector<std::vector<double>> arr_z;
    
    // Generic solver parameters
    int max_iterations;
    double mip_gap;
    bool verbose;

    SolverParams() :
        N(0), nx(0), nu(0), nz(0), nc(0), dual_len(0), dT(0),
        max_iterations(200), mip_gap(0.2), verbose(false)
    {
        // Initialize matrices as empty
        Q.resize(0,0); Qn.resize(0,0); R.resize(0,0);
        E.resize(0,0); F.resize(0,0); G.resize(0,0);
        H1.resize(0,0); H2.resize(0,0); H3.resize(0,0);
        x_goal.resize(0);
    }

    virtual ~SolverParams() = default;
};

} // namespace util
} // namespace optimization::util

