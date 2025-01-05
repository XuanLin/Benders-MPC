#include "optimization/gbd/base_solver.hpp"
#include "problems/cart_pole/cart_pole_solver.hpp"
#include <iostream>

int main() {
    std::cout << "Testing compilation..." << std::endl;
    
    // Try to create a cart pole solver
    optimization::CartPoleParams params;  // Changed this line - CartPoleParams is in the optimization namespace
    optimization::CartPoleGBDSolver solver(params);
    
    std::cout << "Compilation successful!" << std::endl;
    return 0;
}

